"""
资源干扰定位模块。

对于被识别为 resource_slowed 的 target，
找出哪个 kernel 在干扰它，通过 overlap witness 和同次运行差分。
"""

import statistics
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .records import KernelRecord, OverlapRelation, ResourceDiagnosis
from ..sketch import ExecutionSketch


# 资源兼容矩阵：(family1, family2) -> (resource, score)
RESOURCE_COMPAT = {
    ("NCCL", "MEMCPY"): ("HBM", 0.8),
    ("MEMCPY", "NCCL"): ("HBM", 0.8),
    ("NCCL", "NCCL"): ("LINK", 0.9),
    ("GEMM", "GEMM"): ("SM", 0.8),
    ("GEMM", "MEMCPY"): ("HBM_or_SM", 0.5),
    ("MEMCPY", "GEMM"): ("HBM_or_SM", 0.5),
    ("MEMCPY", "MEMCPY"): ("HBM", 0.9),
}


class ResourceLocalizer:
    """资源干扰定位。"""

    def __init__(self,
                 min_overlap_ratio: float = 0.1,
                 progress_ratio_threshold: float = 0.7):
        """
        初始化定位器。
        
        Args:
            min_overlap_ratio: 最小 overlap 比例阈值
        """
        self.min_overlap_ratio = min_overlap_ratio
        self.progress_ratio_threshold = progress_ratio_threshold

    def localize(self,
                 target: KernelRecord,
                 all_kernels: List[KernelRecord],
                 sketch: Optional[ExecutionSketch] = None) -> Optional[ResourceDiagnosis]:
        """
        定位干扰 target 的资源竞争。
        
        Args:
            target: 被减速的 kernel
            all_kernels: 所有 kernel 记录
            sketch: ExecutionSketch（用于查询资源提示）
            
        返回：ResourceDiagnosis，或 None（如果找不到）
        """
        # Step 1: 确认 target 是 slowed
        if not self._confirm_slowed(target, all_kernels):
            return None

        # Step 2: 找 overlap witness
        witnesses = self._find_overlap_witnesses(target, all_kernels)
        if not witnesses:
            return None

        # Step 3: 排序候选
        scored_witnesses = []
        for witness_id, overlap_rel in witnesses.items():
            score = self._compute_witness_score(target, witness_id, overlap_rel, 
                                               all_kernels, sketch)
            scored_witnesses.append((score, witness_id, overlap_rel))

        scored_witnesses.sort(reverse=True)

        if not scored_witnesses:
            return None

        # 选择最高分的
        score, culprit_id, overlap_rel = scored_witnesses[0]
        culprit = next((k for k in all_kernels if k.kid == culprit_id), None)

        if culprit is None:
            return None

        # Step 4: 同次运行差分证据
        same_run_diff = self._compute_same_run_differential(
            target, culprit, all_kernels
        )

        # Step 5: Dose-response
        dose_response_score = self._compute_dose_response(target, culprit, all_kernels)

        # Step 6: 资源兼容性
        resource_hint, compat_score = self._get_resource_compatibility(target, culprit)

        # Step 7: warp/block 证据。当前 collector 可能只有 launch-derived warp 字段；
        # 如果未来接入 Neutrino probe，这里会自动消费 block/warp 分位数。
        warp_block_evidence = self._extract_warp_block_evidence(target, all_kernels)

        # 综合置信度
        confidence = self._compute_confidence(
            score, same_run_diff, dose_response_score, compat_score, warp_block_evidence
        )

        return ResourceDiagnosis(
            target_id=target.kid,
            culprit_id=culprit.kid,
            target_progress=target.progress(),
            peer_progress_median=self._get_peer_median_progress(target, all_kernels),
            overlap_ratio=overlap_rel.overlap_ratio_target,
            dose_response_score=dose_response_score,
            resource_hint=resource_hint,
            warp_block_verdict=warp_block_evidence["verdict"],
            confidence=confidence,
            evidence={
                "culprit_kernel": culprit.kernel_name,
                "culprit_family": culprit.family,
                "overlap_ratio": overlap_rel.overlap_ratio_target,
                "overlap_ns": overlap_rel.overlap_ns,
                "same_run_differential": same_run_diff,
                "resource_compatibility": compat_score,
                "witness_score": score,
                "warp_block": warp_block_evidence,
            },
        )

    def _confirm_slowed(self, target: KernelRecord, all_kernels: List[KernelRecord]) -> bool:
        """确认 target 确实是 slowed（progress 异常低）。"""
        if not target.progress():
            return False

        peers = [k for k in all_kernels 
                if k.family == target.family and k.tag == target.tag and k.kid != target.kid]

        if len(peers) < 1:
            return False

        peer_progresses = [p.progress() for p in peers if p.progress()]
        if not peer_progresses:
            return False

        median = statistics.median(peer_progresses)
        return target.progress() < median * self.progress_ratio_threshold

    def _find_overlap_witnesses(self,
                               target: KernelRecord,
                               all_kernels: List[KernelRecord]) -> Dict[str, OverlapRelation]:
        """找与 target 有 overlap 的 kernel。"""
        witnesses = {}

        for candidate in all_kernels:
            if candidate.kid == target.kid:
                continue

            overlap_ns = self._compute_interval_overlap(target, candidate)
            if overlap_ns == 0:
                continue

            overlap_ratio = overlap_ns / (target.gpu_dur_ns or 1)
            if overlap_ratio < self.min_overlap_ratio:
                continue

            # 创建 overlap 关系
            target_interval = (target.gpu_start_ns, target.gpu_end_ns)
            cand_interval = (candidate.gpu_start_ns, candidate.gpu_end_ns)

            rel = OverlapRelation(
                target_id=target.kid,
                candidate_id=candidate.kid,
                target_interval=target_interval,
                candidate_interval=cand_interval,
                overlap_ns=overlap_ns,
            )

            witnesses[candidate.kid] = rel

        return witnesses

    def _compute_interval_overlap(self, k1: KernelRecord, k2: KernelRecord) -> int:
        """计算两个 kernel 的 overlap 时间。"""
        if (k1.gpu_start_ns is None or k1.gpu_end_ns is None or
            k2.gpu_start_ns is None or k2.gpu_end_ns is None):
            return 0

        overlap_start = max(k1.gpu_start_ns, k2.gpu_start_ns)
        overlap_end = min(k1.gpu_end_ns, k2.gpu_end_ns)

        return max(0, overlap_end - overlap_start)

    def _compute_witness_score(self,
                              target: KernelRecord,
                              witness_id: str,
                              overlap_rel: OverlapRelation,
                              all_kernels: List[KernelRecord],
                              sketch: Optional[ExecutionSketch]) -> float:
        """计算 witness 的因果得分。"""
        score = 0.0

        # 1. Overlap 强度（0.35）
        overlap_strength = overlap_rel.overlap_ratio_target
        score += 0.35 * overlap_strength

        # 2. 同次运行差分（0.30）
        witness = next((k for k in all_kernels if k.kid == witness_id), None)
        if witness:
            same_run_diff = self._compute_same_run_differential(target, witness, all_kernels)
            if same_run_diff:
                score += 0.30 * min(1.0, same_run_diff)

        # 3. 资源兼容性（0.20）
        if witness:
            _, compat_score = self._get_resource_compatibility(target, witness)
            score += 0.20 * compat_score

        # 4. Warp/block 证据（0.15）
        # 简化版本：暂时跳过
        score += 0.15 * 0.5

        return min(1.0, score)

    def _compute_same_run_differential(self,
                                      target: KernelRecord,
                                      witness: KernelRecord,
                                      all_kernels: List[KernelRecord]) -> Optional[float]:
        """
        计算同次运行差分。
        
        比较有 witness overlap 的 target 和没有 witness overlap 的 target。
        """
        # 找同类的其他 target
        peers = [k for k in all_kernels
                if k.family == target.family and k.tag == target.tag 
                and k.kid not in {target.kid, witness.kid}]

        if len(peers) < 1:
            return None

        # 分组：有 witness overlap 和没有
        with_witness = [target]
        without_witness = []

        for peer in peers:
            overlap = self._compute_interval_overlap(peer, witness)
            if overlap > 0:
                with_witness.append(peer)
            else:
                without_witness.append(peer)

        if not with_witness or not without_witness:
            return None

        # 计算中位数 progress
        with_prog = [k.progress() for k in with_witness if k.progress()]
        without_prog = [k.progress() for k in without_witness if k.progress()]

        if not with_prog or not without_prog:
            return None

        median_with = statistics.median(with_prog)
        median_without = statistics.median(without_prog)

        # 返回差异程度（0-1）
        if median_without > 0:
            diff = 1.0 - (median_with / median_without)
            return max(0, min(1.0, diff))

        return None

    def _compute_dose_response(self,
                              target: KernelRecord,
                              witness: KernelRecord,
                              all_kernels: List[KernelRecord]) -> float:
        """
        计算 dose-response 得分。
        
        overlap 越多，progress 越低吗？
        """
        peers = [k for k in all_kernels
                if k.family == target.family and k.tag == target.tag 
                and k.kid not in {target.kid, witness.kid}]

        if len(peers) < 2:
            return 0.0

        # 收集 (overlap_ratio, progress) 对
        data = []
        for peer in [target] + peers:
            overlap = self._compute_interval_overlap(peer, witness)
            peer_dur = peer.gpu_dur_ns or 1
            overlap_ratio = overlap / peer_dur
            progress = peer.progress()

            if progress:
                data.append((overlap_ratio, progress))

        if len(data) < 3:
            return 0.0

        # 简单规则：high overlap 组 vs low overlap 组
        data.sort(key=lambda x: x[0])
        mid = len(data) // 2

        high_overlap_prog = [p for _, p in data[mid:]]
        low_overlap_prog = [p for _, p in data[:mid]]

        if not high_overlap_prog or not low_overlap_prog:
            return 0.0

        high_median = statistics.median(high_overlap_prog)
        low_median = statistics.median(low_overlap_prog)

        if low_median > 0:
            dose_response = 1.0 - (high_median / low_median)
            return max(0, min(1.0, dose_response))

        return 0.0

    def _get_resource_compatibility(self, 
                                   target: KernelRecord,
                                   witness: KernelRecord) -> Tuple[str, float]:
        """查询资源兼容性。"""
        key = (target.family, witness.family)
        if key in RESOURCE_COMPAT:
            resource, score = RESOURCE_COMPAT[key]
            return resource, score

        # 反向查询
        key_rev = (witness.family, target.family)
        if key_rev in RESOURCE_COMPAT:
            resource, score = RESOURCE_COMPAT[key_rev]
            return resource, score

        return "unknown", 0.3

    def _get_peer_median_progress(self, 
                                 target: KernelRecord,
                                 all_kernels: List[KernelRecord]) -> Optional[float]:
        """获取同类 peer 的中位数 progress。"""
        peers = [k for k in all_kernels
                if k.family == target.family and k.tag == target.tag 
                and k.kid != target.kid]

        progresses = [p.progress() for p in peers if p.progress()]
        if progresses:
            return statistics.median(progresses)

        return None

    def _extract_warp_block_evidence(self,
                                     target: KernelRecord,
                                     all_kernels: List[KernelRecord]) -> Dict[str, object]:
        """
        提取硬件层证据。

        支持两类输入：
        1. 当前 Weaver 已有的 launch-derived 字段：total_warps/warps_per_block。
        2. 未来 Neutrino probe 注入后的 runtime 分位数字段：
           block_duration_p50_ns / block_duration_p99_ns / warp_duration_p50_ns / warp_duration_p99_ns。
        """
        payload = target.payload or {}
        p50 = self._first_number(payload, [
            "block_duration_p50_ns", "block_p50_ns", "warp_duration_p50_ns", "warp_p50_ns"
        ])
        p99 = self._first_number(payload, [
            "block_duration_p99_ns", "block_p99_ns", "warp_duration_p99_ns", "warp_p99_ns"
        ])

        evidence: Dict[str, object] = {
            "available": False,
            "verdict": "not_available",
            "total_warps": target.total_warps,
            "note": "only launch-derived warp count is available"
        }

        if p50 is None or p99 is None or p50 <= 0:
            if target.total_warps:
                evidence["available"] = True
                evidence["verdict"] = "launch_derived_only"
            return evidence

        ratio = p99 / p50
        evidence.update({
            "available": True,
            "block_or_warp_p50_ns": p50,
            "block_or_warp_p99_ns": p99,
            "p99_p50_ratio": ratio,
        })

        peer_p50 = []
        peer_p99 = []
        for peer in all_kernels:
            if peer.kid == target.kid or peer.family != target.family or peer.tag != target.tag:
                continue
            peer_payload = peer.payload or {}
            q50 = self._first_number(peer_payload, [
                "block_duration_p50_ns", "block_p50_ns", "warp_duration_p50_ns", "warp_p50_ns"
            ])
            q99 = self._first_number(peer_payload, [
                "block_duration_p99_ns", "block_p99_ns", "warp_duration_p99_ns", "warp_p99_ns"
            ])
            if q50 is not None:
                peer_p50.append(q50)
            if q99 is not None:
                peer_p99.append(q99)

        median_p50 = statistics.median(peer_p50) if peer_p50 else None
        median_p99 = statistics.median(peer_p99) if peer_p99 else None
        evidence["peer_p50_median_ns"] = median_p50
        evidence["peer_p99_median_ns"] = median_p99

        if ratio >= 3.0 and (median_p50 is None or p50 <= median_p50 * 1.3):
            evidence["verdict"] = "internal_tail"
            evidence["note"] = "p99 is high while p50 is not broadly shifted; this looks like target-internal tail"
        elif median_p50 is not None and p50 > median_p50 * 1.3:
            evidence["verdict"] = "broad_slowdown"
            evidence["note"] = "p50 and tail move together; this supports external resource interference"
        else:
            evidence["verdict"] = "inconclusive"
            evidence["note"] = "runtime warp/block samples exist but do not strongly separate tail from broad slowdown"

        return evidence

    def _first_number(self, payload: Dict[str, object], keys: List[str]) -> Optional[float]:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _compute_confidence(self,
                           witness_score: float,
                           same_run_diff: Optional[float],
                           dose_response_score: float,
                           compat_score: float,
                           warp_block_evidence: Dict[str, object]) -> float:
        """综合计算置信度。"""
        confidence = witness_score * 0.4

        if same_run_diff:
            confidence += same_run_diff * 0.3

        confidence += dose_response_score * 0.2

        confidence += compat_score * 0.1

        verdict = warp_block_evidence.get("verdict")
        if verdict == "broad_slowdown":
            confidence += 0.12
        elif verdict == "internal_tail":
            confidence -= 0.18
        elif verdict == "launch_derived_only":
            confidence += 0.02

        return max(0.05, min(0.95, confidence))

    def localize_batch(self,
                      targets: List[KernelRecord],
                      all_kernels: List[KernelRecord],
                      sketch: Optional[ExecutionSketch] = None) -> Dict[str, ResourceDiagnosis]:
        """批量定位。"""
        result = {}
        for target in targets:
            diagnosis = self.localize(target, all_kernels, sketch)
            if diagnosis:
                result[target.kid] = diagnosis

        return result
