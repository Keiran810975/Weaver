"""
性能下降类型判断模块。

根据时序特性判断 target kernel 是：
1. CPU 运行时阻塞 - CPU 侧 launch 时间晚
2. 依赖阻塞 - GPU 启动时间晚
3. 资源干扰 - GPU 启动正常但执行变慢
4. 不确定 - 信息不足
"""

import statistics
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .records import KernelRecord, SlowdownDiagnosis, SlowdownType, SyncRecord


class TimingAnalyzer:
    """性能下降类型分析。"""

    def __init__(self, 
                 progress_outlier_k_sigma: float = 3.0,
                 close_threshold_ns: int = 100_000,
                 min_cpu_gap_ns: int = 1_000_000,
                 min_gpu_wait_ns: int = 100_000):  # 100 us
        """
        初始化分析器。
        
        Args:
            progress_outlier_k_sigma: MAD 的倍数，用于判断 progress outlier
            close_threshold_ns: 距离阈值，用于判断是否"紧贴"
        """
        self.progress_outlier_k_sigma = progress_outlier_k_sigma
        self.close_threshold_ns = close_threshold_ns
        self.min_cpu_gap_ns = min_cpu_gap_ns
        self.min_gpu_wait_ns = min_gpu_wait_ns

    def classify_slowdown(self, 
                         target: KernelRecord,
                         peers: Optional[List[KernelRecord]] = None,
                         all_records: Optional[List[KernelRecord]] = None,
                         syncs: Optional[List[SyncRecord]] = None) -> SlowdownDiagnosis:
        """
        分类 target 的性能下降类型。
        
        Args:
            target: 目标 kernel record
            peers: 同类的对照 kernel records（用于 progress 计算）
            
        返回：SlowdownDiagnosis
        """
        peers = peers or []
        all_records = all_records or []
        syncs = syncs or []

        # 1. CPU enqueue 晚：CPU/Python/runtime/sync 侧阻塞。
        # 这里比较的是“上一个 host launch 结束 -> 当前 enqueue 开始”的空窗，
        # 并用同类 kernel 的空窗做同次运行差分。
        cpu_gap = self._estimate_cpu_delay(target, all_records)
        if cpu_gap is not None:
            peer_cpu_gaps = self._peer_cpu_gaps(target, peers, all_records)
            is_late, baseline, mad, threshold = self._is_high_outlier(
                cpu_gap, peer_cpu_gaps, self.min_cpu_gap_ns
            )
            nearby_sync = self._find_nearby_sync(target, syncs, use_cpu_time=True)
            if is_late or nearby_sync is not None:
                evidence = {
                    "decision": "cpu_enqueue_late",
                    "reason": "CPU enqueue is delayed before the GPU launch is issued",
                    "cpu_gap_ns": cpu_gap,
                    "peer_cpu_gap_median_ns": baseline,
                    "peer_cpu_gap_mad_ns": mad,
                    "threshold_ns": threshold,
                }
                if nearby_sync is not None:
                    evidence["nearby_sync"] = {
                        "sync_id": nearby_sync.sid,
                        "kind": nearby_sync.kind.value,
                        "duration_ns": nearby_sync.duration_ns,
                    }
                return SlowdownDiagnosis(
                    target_id=target.kid,
                    slowdown_type=SlowdownType.CPU_RUNTIME_BLOCKED,
                    confidence=0.85 if nearby_sync else 0.75,
                    cpu_enqueue_delay_ns=cpu_gap,
                    evidence=evidence,
                )

        # 2. CPU enqueue 正常但 GPU start 晚：stream/event/scheduler 依赖阻塞。
        gpu_wait = self._estimate_gpu_start_delay(target)
        if gpu_wait is not None:
            peer_waits = [w for w in (self._estimate_gpu_start_delay(p) for p in peers) if w is not None]
            is_late, baseline, mad, threshold = self._is_high_outlier(
                gpu_wait, peer_waits, self.min_gpu_wait_ns
            )
            nearby_sync = self._find_nearby_sync(target, syncs, use_cpu_time=False)
            if is_late or nearby_sync is not None:
                evidence = {
                    "decision": "gpu_start_late",
                    "reason": "CPU enqueue has happened but GPU execution starts late",
                    "gpu_wait_ns": gpu_wait,
                    "peer_gpu_wait_median_ns": baseline,
                    "peer_gpu_wait_mad_ns": mad,
                    "threshold_ns": threshold,
                }
                if nearby_sync is not None:
                    evidence["nearby_sync"] = {
                        "sync_id": nearby_sync.sid,
                        "kind": nearby_sync.kind.value,
                        "duration_ns": nearby_sync.duration_ns,
                    }
                return SlowdownDiagnosis(
                    target_id=target.kid,
                    slowdown_type=SlowdownType.DEPENDENCY_BLOCKED,
                    confidence=0.8 if nearby_sync else 0.7,
                    gpu_start_delay_ns=gpu_wait,
                    evidence=evidence,
                )

        # 3. GPU start 正常但 GPU end 晚：normalized progress 或 duration 异常。
        progress_anomaly = self._check_progress_anomaly(target, peers)
        if progress_anomaly is not None:
            return SlowdownDiagnosis(
                target_id=target.kid,
                slowdown_type=SlowdownType.RESOURCE_SLOWED,
                confidence=0.78,
                normalized_progress=progress_anomaly["target_progress"],
                peer_median_progress=progress_anomaly["peer_median"],
                evidence={
                    "decision": "gpu_end_late",
                    "reason": "Normalized progress is anomalously low after GPU start",
                    "target_progress": progress_anomaly["target_progress"],
                    "peer_median": progress_anomaly["peer_median"],
                    "peer_mad": progress_anomaly["peer_mad"],
                    "threshold": progress_anomaly["threshold"],
                    "work_type": target.work_type,
                },
            )

        duration_anomaly = self._check_duration_anomaly(target, peers)
        if duration_anomaly is not None:
            return SlowdownDiagnosis(
                target_id=target.kid,
                slowdown_type=SlowdownType.RESOURCE_SLOWED,
                confidence=0.68,
                evidence={
                    "decision": "gpu_end_late",
                    "reason": "GPU duration is anomalously high among comparable kernels",
                    **duration_anomaly,
                },
            )

        # 4. 没有 peer 但有明显 overlap 的长 kernel，也保守标记为资源可疑。
        if target.gpu_dur_ns and self._has_runtime_overlap(target, all_records):
            peer_durations = [p.gpu_dur_ns for p in peers if p.gpu_dur_ns]
            if not peer_durations and target.gpu_dur_ns > self.min_cpu_gap_ns:
                return SlowdownDiagnosis(
                    target_id=target.kid,
                    slowdown_type=SlowdownType.RESOURCE_SLOWED,
                    confidence=0.55,
                    evidence={
                        "decision": "gpu_end_late",
                        "reason": "Long GPU execution overlaps with another kernel, but peer evidence is weak",
                        "target_duration_ns": target.gpu_dur_ns,
                    },
                )

        # 无法确定
        return SlowdownDiagnosis(
            target_id=target.kid,
            slowdown_type=SlowdownType.UNCERTAIN,
            confidence=0.3,
            evidence={"reason": "Insufficient evidence for classification"},
        )

    def _estimate_cpu_delay(self,
                            target: KernelRecord,
                            all_records: Optional[List[KernelRecord]] = None) -> Optional[int]:
        """
        估计 CPU 侧 enqueue 的延迟。
        
        使用同一 rank/pid 下前一个 kernel 的 CPU enqueue end 到当前 enqueue start 的间隔。
        """
        if target.cpu_enqueue_start_ns is None or not all_records:
            return None

        prev = self._find_previous_cpu_kernel(target, all_records)
        if prev is None or prev.cpu_enqueue_end_ns is None:
            return None
        gap = target.cpu_enqueue_start_ns - prev.cpu_enqueue_end_ns
        return gap if gap >= 0 else None

    def _estimate_gpu_start_delay(self, target: KernelRecord) -> Optional[int]:
        """
        估计 GPU start 的延迟。
        
        简单启发式：CPU enqueue end 到 GPU start 的间隔。
        """
        if target.cpu_enqueue_end_ns is None or target.gpu_start_ns is None:
            return None

        interval = target.gpu_start_ns - target.cpu_enqueue_end_ns
        if interval > 0:
            return interval

        return None

    def _check_progress_anomaly(self,
                                target: KernelRecord,
                                peers: List[KernelRecord]) -> Optional[Dict]:
        """
        检查 target 的 normalized progress 是否异常。
        
        Args:
            target: 目标 kernel
            peers: 同类对照 kernel
            
        返回：包含进度信息的字典，或 None（如果不异常）
        """
        if target.progress() is None:
            return None

        peer_progresses = []
        for peer in peers:
            if peer.kid == target.kid:
                continue
            p = peer.progress()
            if p is not None:
                peer_progresses.append(p)

        if len(peer_progresses) < 2:
            return None

        peer_median = statistics.median(peer_progresses)

        try:
            peer_mad = statistics.median([abs(p - peer_median) for p in peer_progresses])
        except statistics.StatisticsError:
            peer_mad = 0

        # 异常阈值
        if peer_mad > 0:
            threshold = peer_median - self.progress_outlier_k_sigma * 1.4826 * peer_mad
        else:
            threshold = peer_median * 0.6

        target_progress = target.progress()
        if target_progress < threshold:
            return {
                "target_progress": target_progress,
                "peer_median": peer_median,
                "peer_mad": peer_mad,
                "threshold": threshold,
            }

        return None

    def _check_duration_anomaly(self,
                                target: KernelRecord,
                                peers: List[KernelRecord]) -> Optional[Dict]:
        """当没有可靠 work_units 时，用 GPU duration 做同类差分。"""
        if target.gpu_dur_ns is None:
            return None
        peer_durations = [p.gpu_dur_ns for p in peers if p.kid != target.kid and p.gpu_dur_ns]
        if len(peer_durations) < 2:
            return None
        is_late, median, mad, threshold = self._is_high_outlier(
            target.gpu_dur_ns, peer_durations, self.min_gpu_wait_ns
        )
        if not is_late:
            return None
        return {
            "target_duration_ns": target.gpu_dur_ns,
            "peer_duration_median_ns": median,
            "peer_duration_mad_ns": mad,
            "threshold_ns": threshold,
        }

    def _find_previous_cpu_kernel(self,
                                  target: KernelRecord,
                                  records: List[KernelRecord]) -> Optional[KernelRecord]:
        """找同 rank/pid 下 host enqueue 顺序里的前一个 kernel。"""
        candidates = []
        for record in records:
            if record.kid == target.kid:
                continue
            if record.pid != target.pid:
                continue
            if target.rank is not None and record.rank != target.rank:
                continue
            end = record.cpu_enqueue_end_ns
            if end is None or target.cpu_enqueue_start_ns is None:
                continue
            if end <= target.cpu_enqueue_start_ns:
                candidates.append((end, record))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _peer_cpu_gaps(self,
                       target: KernelRecord,
                       peers: List[KernelRecord],
                       all_records: List[KernelRecord]) -> List[int]:
        gaps = []
        for peer in peers:
            if peer.kid == target.kid:
                continue
            gap = self._estimate_cpu_delay(peer, all_records)
            if gap is not None:
                gaps.append(gap)
        return gaps

    def _is_high_outlier(self,
                         value: int,
                         peer_values: List[int],
                         min_abs_ns: int) -> Tuple[bool, Optional[float], Optional[float], int]:
        """MAD-based high-side outlier with an absolute floor."""
        if len(peer_values) >= 2:
            median = float(statistics.median(peer_values))
            mad = float(statistics.median([abs(v - median) for v in peer_values]))
            threshold = median + self.progress_outlier_k_sigma * 1.4826 * mad if mad > 0 else median * 2.0
            threshold = max(threshold, float(min_abs_ns))
            return value > threshold, median, mad, int(threshold)
        threshold = min_abs_ns * 10 if min_abs_ns < 1_000_000 else min_abs_ns
        return value > threshold, None, None, int(threshold)

    def _find_nearby_sync(self,
                          target: KernelRecord,
                          syncs: List[SyncRecord],
                          use_cpu_time: bool) -> Optional[SyncRecord]:
        """找紧贴 target 的 sync/event wait。"""
        target_start = target.cpu_enqueue_start_ns if use_cpu_time else target.gpu_start_ns
        if target_start is None:
            return None
        nearest = None
        nearest_gap = None
        for sync in syncs:
            if target.rank is not None and sync.rank is not None and sync.rank != target.rank:
                continue
            if sync.pid != target.pid:
                continue
            sync_end = sync.ts_end_ns or sync.ts_start_ns
            gap = target_start - sync_end
            if gap < 0 or gap > self.min_cpu_gap_ns:
                continue
            if nearest_gap is None or gap < nearest_gap:
                nearest = sync
                nearest_gap = gap
        return nearest

    def _has_runtime_overlap(self,
                             target: KernelRecord,
                             records: List[KernelRecord]) -> bool:
        if target.gpu_start_ns is None or target.gpu_end_ns is None:
            return False
        for record in records:
            if record.kid == target.kid:
                continue
            if record.gpu_start_ns is None or record.gpu_end_ns is None:
                continue
            if min(target.gpu_end_ns, record.gpu_end_ns) > max(target.gpu_start_ns, record.gpu_start_ns):
                return True
        return False

    def classify_batch(self, 
                      targets: List[KernelRecord],
                      all_records: Optional[List[KernelRecord]] = None,
                      syncs: Optional[List[SyncRecord]] = None) -> Dict[str, SlowdownDiagnosis]:
        """
        批量分类。
        
        Args:
            targets: 目标 kernel 列表
            all_records: 所有 kernel 记录（用于查找 peers）
            
        返回：{kernel_id -> SlowdownDiagnosis}
        """
        # 构建 peers 映射
        peers_map = {}
        if all_records:
            by_family_tag = defaultdict(list)
            for record in all_records:
                key = (record.family, record.tag)
                by_family_tag[key].append(record)
            peers_map = by_family_tag

        result = {}
        for target in targets:
            key = (target.family, target.tag)
            peers = peers_map.get(key, [])
            diagnosis = self.classify_slowdown(target, peers, all_records or [], syncs or [])
            result[target.kid] = diagnosis

        return result
