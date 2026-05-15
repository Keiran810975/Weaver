"""
候选发现模块。

从规范化的 kernel records 中发现可疑的 target kernel：
1. 结构偏离 - unexpected event、unexpected predecessor、overlap loss
2. Normalized progress outlier - 工作进度异常低
3. Rank/bucket outlier - 与同类比较速度显著下降
"""

import statistics
from typing import Dict, List, Optional
from collections import defaultdict

from .records import KernelRecord, Candidate, SyncRecord
from .sketch_match import (
    dependency_uses_same_stream,
    expected_dependencies_for_target,
    find_matching_predecessors,
    selector_matches_kernel,
)
from ..sketch import ExecutionSketch
from ..sketch.rules import get_default_overlap_expectations


class CandidateDiscovery:
    """候选 target 发现。"""

    def __init__(self,
                 sketch: Optional[ExecutionSketch] = None,
                 min_long_gap_ns: int = 1_000_000,
                 adjacent_gap_ns: int = 100_000):
        """
        初始化候选发现。
        
        Args:
            sketch: ExecutionSketch 对象。可选；没有 sketch 时使用同次运行差分和默认 overlap 关系。
        """
        self.sketch = sketch
        self.min_long_gap_ns = min_long_gap_ns
        self.adjacent_gap_ns = adjacent_gap_ns

    def discover(self,
                 records: List[KernelRecord],
                 syncs: Optional[List[SyncRecord]] = None) -> List[Candidate]:
        """
        发现所有候选 target kernel。
        
        Args:
            records: 规范化的 kernel records
            syncs: 规范化的同步记录，可选
            
        返回：候选列表
        """
        candidates = []

        records = [r for r in records if self._is_diagnosable_target(r) or self._may_be_blocker(r)]

        # A. 结构偏离候选
        candidates.extend(self._find_structural_deviation_candidates(records, syncs or []))

        # B. 性能异常候选（基于 normalized progress）
        candidates.extend(self._find_progress_outlier_candidates(records))

        # C. Rank/bucket 异常候选
        candidates.extend(self._find_rank_outlier_candidates(records))

        return self._dedupe(candidates)

    def _find_structural_deviation_candidates(self,
                                              records: List[KernelRecord],
                                              syncs: List[SyncRecord]) -> List[Candidate]:
        """发现结构偏离候选。"""
        candidates = []

        # A1. 无法分类的 kernel 作为 blocker 有价值，但作为 target 噪声太高，
        # 尤其是 <runtime_kernel> 和初始化 kernel；不再直接提升为 target。
        for record in records:
            if record.family == "UNKNOWN" and record.operator_name and not self._has_manual_dependencies():
                candidates.append(Candidate(
                    target_id=record.kid,
                    candidate_type="structural_deviation",
                    reason="unknown_kernel",
                    evidence={
                        "kernel_name": record.kernel_name,
                        "operator_name": record.operator_name,
                    },
                    confidence=0.6,
                ))

        # A2. 意外的前驱 kernel
        candidates.extend(self._find_unexpected_predecessor_candidates(records))
        candidates.extend(self._find_manual_dependency_deviation_candidates(records))

        # A3. 预期 overlap 丧失
        candidates.extend(self._find_overlap_loss_candidates(records))

        # A4. sync/event wait 后紧接的 kernel。slide 里把 cuda synchronize / event wait
        # 作为可能破坏 overlap 的 extra event，这里即使没有完整 sketch 也先拉进候选。
        candidates.extend(self._find_sync_preceded_candidates(records, syncs))

        return candidates

    def _find_manual_dependency_deviation_candidates(self,
                                                     records: List[KernelRecord]) -> List[Candidate]:
        """用手工 expected_dependencies 对比 actual predecessor。"""
        if self.sketch is None or not getattr(self.sketch, "expected_dependencies", None):
            return []

        candidates = []
        by_stream = defaultdict(list)
        for record in records:
            by_stream[record.stream or "default"].append(record)

        for stream, kernels in by_stream.items():
            sorted_kernels = sorted(kernels, key=lambda k: k.gpu_start_ns or k.cpu_enqueue_start_ns or 0)
            for i, target in enumerate(sorted_kernels):
                if not self._should_consider_structural_target(target):
                    continue
                deps = expected_dependencies_for_target(self.sketch, target)
                if not deps:
                    continue
                actual_pred = sorted_kernels[i - 1] if i > 0 else None
                if actual_pred is None:
                    candidates.append(Candidate(
                        target_id=target.kid,
                        candidate_type="structural_deviation",
                        reason="missing_expected_predecessor",
                        evidence={
                            "manual_dependency_ids": [dep.dependency_id for dep in deps],
                            "target_family": target.family,
                            "target_tag": target.tag,
                        },
                        confidence=0.75,
                    ))
                    continue

                expected_found = []
                actual_matches = False
                for dep in deps:
                    expected_preds = find_matching_predecessors(
                        target,
                        records,
                        dep.predecessors,
                        same_stream=dependency_uses_same_stream(dep),
                    )
                    expected_found.extend(expected_preds)
                    if any(selector_matches_kernel(selector, actual_pred) for selector in dep.predecessors):
                        actual_matches = True

                if not actual_matches:
                    candidates.append(Candidate(
                        target_id=target.kid,
                        candidate_type="structural_deviation",
                        reason="unexpected_predecessor_against_manual_sketch",
                        evidence={
                            "actual_predecessor_id": actual_pred.kid,
                            "actual_predecessor_kernel": actual_pred.kernel_name,
                            "actual_predecessor_operator": actual_pred.operator_name,
                            "actual_predecessor_family": actual_pred.family,
                            "manual_dependency_ids": [dep.dependency_id for dep in deps],
                            "expected_predecessor_ids": [pred.kid for pred in expected_found],
                            "target_family": target.family,
                            "target_tag": target.tag,
                        },
                        confidence=0.85,
                    ))

        return candidates

    def _find_unexpected_predecessor_candidates(self, records: List[KernelRecord]) -> List[Candidate]:
        """找意外的前驱 kernel。"""
        candidates = []

        # 按 stream 分组
        stream_kernels = defaultdict(list)
        for record in records:
            stream = record.stream or "default"
            stream_kernels[stream].append(record)

        # 对每个 stream 按时间排序
        for stream, kernels in stream_kernels.items():
            sorted_kernels = sorted(kernels, key=lambda k: k.gpu_start_ns or k.cpu_enqueue_start_ns or 0)

            for i, target in enumerate(sorted_kernels):
                if not self._should_consider_structural_target(target):
                    continue
                if i == 0:
                    continue

                # 找前驱
                prev = sorted_kernels[i - 1]

                # 检查间隔
                if target.gpu_start_ns and prev.gpu_end_ns:
                    gap = target.gpu_start_ns - prev.gpu_end_ns
                else:
                    gap = None

                if gap is None:
                    continue

                # 如果 gap 过大，可能存在 CPU/runtime/sync 阻塞。
                if gap > self.min_long_gap_ns:
                    candidates.append(Candidate(
                        target_id=target.kid,
                        candidate_type="structural_deviation",
                        reason="large_gap_to_predecessor",
                        evidence={
                            "predecessor_id": prev.kid,
                            "gap_ns": gap,
                            "target_family": target.family,
                        },
                        confidence=0.5,
                    ))

                # 如果一个额外/轻量 kernel 紧贴 target，slide 15 中的
                # A -> extra_kernel -> target 需要优先进入依赖定位。
                if self._looks_extra(prev) and 0 <= gap <= self.adjacent_gap_ns:
                    candidates.append(Candidate(
                        target_id=target.kid,
                        candidate_type="structural_deviation",
                        reason="extra_predecessor",
                        evidence={
                            "predecessor_id": prev.kid,
                            "predecessor_family": prev.family,
                            "predecessor_kernel": prev.kernel_name,
                            "predecessor_operator": prev.operator_name,
                            "gap_ns": gap,
                        },
                        confidence=0.65,
                    ))

        return candidates

    def _find_overlap_loss_candidates(self, records: List[KernelRecord]) -> List[Candidate]:
        """找 overlap 丧失的候选。"""
        candidates = []

        # 优先使用 sketch；没有 sketch 时使用模块三的默认可并行关系做同次运行差分。
        overlap_expectations = (
            self.sketch.overlap_expectations
            if self.sketch is not None
            else get_default_overlap_expectations()
        )

        for oe in overlap_expectations:
            if oe.expected.value != "may_overlap":
                continue

            # 找 left_family 和 right_family 的 kernel
            left_kernels = [r for r in records if r.family == oe.left_family]
            right_kernels = [r for r in records if r.family == oe.right_family]

            # 检查是否有 overlap
            for left in left_kernels:
                for right in right_kernels:
                    overlap = self._compute_overlap(left, right)
                    if overlap is None:
                        continue

                    overlap_ratio = overlap / max(left.gpu_dur_ns or 1, 1)

                    # 如果 overlap 太小，标记为候选
                    if overlap_ratio < 0.1:  # 少于 10% overlap
                        candidates.append(Candidate(
                            target_id=left.kid,
                            candidate_type="structural_deviation",
                            reason="expected_overlap_loss",
                            evidence={
                                "expected_overlap_with": right.kid,
                                "actual_overlap_ratio": overlap_ratio,
                                "left_family": oe.left_family,
                                "right_family": oe.right_family,
                            },
                            confidence=0.6,
                        ))
                        break

        return candidates

    def _find_sync_preceded_candidates(self,
                                       records: List[KernelRecord],
                                       syncs: List[SyncRecord]) -> List[Candidate]:
        """找同步事件后紧接启动的 kernel。"""
        candidates = []
        if not syncs:
            return candidates

        for sync in syncs:
            sync_end = sync.ts_end_ns or sync.ts_start_ns
            if sync_end is None:
                continue
            same_scope = [
                k for k in records
                if (sync.rank is None or k.rank == sync.rank) and k.pid == sync.pid
            ]
            following = []
            for kernel in same_scope:
                start = kernel.cpu_enqueue_start_ns or kernel.gpu_start_ns
                if start is None or start < sync_end:
                    continue
                following.append((start - sync_end, kernel))
            if not following:
                continue
            following.sort(key=lambda item: item[0])
            target_item = next(
                ((gap, kernel) for gap, kernel in following if self._should_consider_structural_target(kernel)),
                None,
            )
            if target_item is None:
                continue
            gap, target = target_item
            if gap <= self.min_long_gap_ns:
                candidates.append(Candidate(
                    target_id=target.kid,
                    candidate_type="structural_deviation",
                    reason="sync_preceded_kernel",
                    evidence={
                        "sync_id": sync.sid,
                        "sync_kind": sync.kind.value,
                        "sync_duration_ns": sync.duration_ns,
                        "gap_ns": gap,
                    },
                    confidence=0.7,
                ))

        return candidates

    def _find_progress_outlier_candidates(self, records: List[KernelRecord]) -> List[Candidate]:
        """发现基于 normalized progress 的异常。"""
        candidates = []

        # 按 (family, tag) 分组
        groups = defaultdict(list)
        for record in records:
            if not self._is_diagnosable_target(record):
                continue
            if record.work_value and record.gpu_dur_ns:
                key = (record.family, record.tag)
                progress = record.progress()
                if progress is not None:
                    groups[key].append((record, progress))

        # 对每个组查找异常值
        for key, items in groups.items():
            if len(items) < 3:
                # 至少需要 3 个样本
                continue

            progresses = [p for _, p in items]
            median = statistics.median(progresses)

            # 计算 MAD（中绝对偏差）
            try:
                mad = statistics.median([abs(p - median) for p in progresses])
            except statistics.StatisticsError:
                mad = 0

            # 异常检测：progress < median - 3 * 1.4826 * mad
            threshold = median - 3 * 1.4826 * mad if mad > 0 else median * 0.6

            for record, progress in items:
                if progress < threshold:
                    candidates.append(Candidate(
                        target_id=record.kid,
                        candidate_type="progress_outlier",
                        reason="normalized_progress_low",
                        evidence={
                            "target_progress": progress,
                            "peer_median_progress": median,
                            "peer_mad": mad,
                            "unit": "work_per_ns",
                            "work_type": record.work_type,
                        },
                        confidence=0.75,
                    ))

        return candidates

    def _find_rank_outlier_candidates(self, records: List[KernelRecord]) -> List[Candidate]:
        """发现基于 rank 的异常。"""
        candidates = []

        # 按 (family, tag, 时间窗口) 分组
        groups = defaultdict(list)
        for record in records:
            if not self._is_diagnosable_target(record):
                continue
            if record.rank is not None:
                # 简化分组：按 family + tag + 100ms 时间窗口
                ts = (record.gpu_start_ns or record.cpu_enqueue_start_ns or 0) // 100_000_000
                key = (record.family, record.tag, ts)

                if record.work_value and record.gpu_dur_ns:
                    progress = record.progress()
                    if progress is not None:
                        groups[key].append((record.rank, record, progress))

        # 找 rank outlier
        for key, items in groups.items():
            if len(items) < 2:
                continue

            # 比较同一 group 中不同 rank 的 progress
            progresses = [p for _, _, p in items]
            median_progress = statistics.median(progresses)

            for rank, record, progress in items:
                # 如果该 rank 的 progress 远低于中位数
                if progress < median_progress * 0.7:
                    candidates.append(Candidate(
                        target_id=record.kid,
                        candidate_type="rank_outlier",
                        reason="rank_slower_than_peers",
                        evidence={
                            "rank": rank,
                            "target_progress": progress,
                            "peer_median_progress": median_progress,
                            "slowdown_factor": median_progress / progress if progress > 0 else float('inf'),
                        },
                        confidence=0.7,
                    ))

        return candidates

    def _looks_extra(self, record: KernelRecord) -> bool:
        """判断一个前驱是否像 slide 中的 extra kernel。"""
        if record.family in {"UNKNOWN", "MEMCPY", "ELEMENTWISE"}:
            return True
        name = record.kernel_name.lower()
        op = (record.operator_name or "").lower()
        return any(
            token in name or token in op
            for token in ("copy", "cast", "contiguous", "layout", "memset", "fill", "zero")
        )

    def _has_manual_dependencies(self) -> bool:
        return self.sketch is not None and bool(getattr(self.sketch, "expected_dependencies", None))

    def _is_manual_dependency_target(self, record: KernelRecord) -> bool:
        return bool(expected_dependencies_for_target(self.sketch, record))

    def _should_consider_structural_target(self, record: KernelRecord) -> bool:
        if not self._is_diagnosable_target(record):
            return False
        if self._has_manual_dependencies():
            return self._is_manual_dependency_target(record)
        return True

    def _is_diagnosable_target(self, record: KernelRecord) -> bool:
        if not record.kernel_name or record.kernel_name in {"<unknown>", "<runtime_kernel>"}:
            return False
        if record.family in {"UNKNOWN", "ELEMENTWISE"}:
            return False
        return True

    def _may_be_blocker(self, record: KernelRecord) -> bool:
        if record.operator_name:
            return True
        if record.family in {"GEMM", "NCCL", "MEMCPY", "REDUCTION"}:
            return True
        return False

    def _dedupe(self, candidates: List[Candidate]) -> List[Candidate]:
        """按 target/reason 合并候选，保留最高置信度和证据。"""
        merged: Dict[tuple, Candidate] = {}
        for candidate in candidates:
            key = (candidate.target_id, candidate.candidate_type, candidate.reason)
            old = merged.get(key)
            if old is None or candidate.confidence > old.confidence:
                merged[key] = candidate
        return list(merged.values())

    def _compute_overlap(self, k1: KernelRecord, k2: KernelRecord) -> Optional[int]:
        """
        计算两个 kernel 的 overlap 时间（纳秒）。
        
        返回：overlap 纳秒数，如果没有 overlap 或缺少时间信息返回 None
        """
        if k1.gpu_start_ns is None or k1.gpu_end_ns is None:
            return None
        if k2.gpu_start_ns is None or k2.gpu_end_ns is None:
            return None

        overlap_start = max(k1.gpu_start_ns, k2.gpu_start_ns)
        overlap_end = min(k1.gpu_end_ns, k2.gpu_end_ns)

        if overlap_end > overlap_start:
            return overlap_end - overlap_start

        return 0
