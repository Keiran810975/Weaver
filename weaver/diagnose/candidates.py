"""
候选发现模块。

从规范化的 kernel records 中发现可疑的 target kernel：
1. 结构偏离 - unexpected event、unexpected predecessor、overlap loss
2. Normalized progress outlier - 工作进度异常低
3. Rank/bucket outlier - 与同类比较速度显著下降
"""

import statistics
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

from .records import KernelRecord, Candidate
from ..sketch import ExecutionSketch


class CandidateDiscovery:
    """候选 target 发现。"""

    def __init__(self, sketch: ExecutionSketch):
        """
        初始化候选发现。
        
        Args:
            sketch: ExecutionSketch 对象
        """
        self.sketch = sketch
        self.unknown_families = set(t.family for t in sketch.kernel_templates if t.family == "UNKNOWN")

    def discover(self, records: List[KernelRecord]) -> List[Candidate]:
        """
        发现所有候选 target kernel。
        
        Args:
            records: 规范化的 kernel records
            
        返回：候选列表
        """
        candidates = []

        # A. 结构偏离候选
        candidates.extend(self._find_structural_deviation_candidates(records))

        # B. 性能异常候选（基于 normalized progress）
        candidates.extend(self._find_progress_outlier_candidates(records))

        # C. Rank/bucket 异常候选
        candidates.extend(self._find_rank_outlier_candidates(records))

        return candidates

    def _find_structural_deviation_candidates(self, records: List[KernelRecord]) -> List[Candidate]:
        """发现结构偏离候选。"""
        candidates = []

        # A1. 无法分类的 kernel
        for record in records:
            if record.family == "UNKNOWN":
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

        # A3. 预期 overlap 丧失
        candidates.extend(self._find_overlap_loss_candidates(records))

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
                if i == 0:
                    continue

                # 找前驱
                prev = sorted_kernels[i - 1]

                # 检查间隔
                if target.gpu_start_ns and prev.gpu_end_ns:
                    gap = target.gpu_start_ns - prev.gpu_end_ns
                else:
                    gap = None

                # 如果 gap 过大（> 1ms），可能有问题
                if gap is not None and gap > 1_000_000:  # 1ms
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

        return candidates

    def _find_overlap_loss_candidates(self, records: List[KernelRecord]) -> List[Candidate]:
        """找 overlap 丧失的候选。"""
        candidates = []

        # 从 sketch 中获取 overlap expectations
        for oe in self.sketch.overlap_expectations:
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

    def _find_progress_outlier_candidates(self, records: List[KernelRecord]) -> List[Candidate]:
        """发现基于 normalized progress 的异常。"""
        candidates = []

        # 按 (family, tag) 分组
        groups = defaultdict(list)
        for record in records:
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
