"""
性能下降类型判断模块。

根据时序特性判断 target kernel 是：
1. CPU 运行时阻塞 - CPU 侧 launch 时间晚
2. 依赖阻塞 - GPU 启动时间晚
3. 资源干扰 - GPU 启动正常但执行变慢
4. 不确定 - 信息不足
"""

import statistics
from typing import Dict, List, Optional
from collections import defaultdict

from .records import KernelRecord, SlowdownDiagnosis, SlowdownType


class TimingAnalyzer:
    """性能下降类型分析。"""

    def __init__(self, 
                 progress_outlier_k_sigma: float = 3.0,
                 close_threshold_ns: int = 100_000):  # 100 us
        """
        初始化分析器。
        
        Args:
            progress_outlier_k_sigma: MAD 的倍数，用于判断 progress outlier
            close_threshold_ns: 距离阈值，用于判断是否"紧贴"
        """
        self.progress_outlier_k_sigma = progress_outlier_k_sigma
        self.close_threshold_ns = close_threshold_ns

    def classify_slowdown(self, 
                         target: KernelRecord,
                         peers: Optional[List[KernelRecord]] = None) -> SlowdownDiagnosis:
        """
        分类 target 的性能下降类型。
        
        Args:
            target: 目标 kernel record
            peers: 同类的对照 kernel records（用于 progress 计算）
            
        返回：SlowdownDiagnosis
        """
        # 检查 CPU enqueue 是否晚
        cpu_enqueue_delay_ns = None
        if target.cpu_enqueue_start_ns is not None:
            cpu_enqueue_delay_ns = self._estimate_cpu_delay(target)
            if cpu_enqueue_delay_ns is not None and cpu_enqueue_delay_ns > self.close_threshold_ns:
                return SlowdownDiagnosis(
                    target_id=target.kid,
                    slowdown_type=SlowdownType.CPU_RUNTIME_BLOCKED,
                    confidence=0.8,
                    cpu_enqueue_delay_ns=cpu_enqueue_delay_ns,
                    evidence={
                        "reason": "CPU enqueue time is significantly late",
                        "cpu_delay_ns": cpu_enqueue_delay_ns,
                    },
                )

        # 检查 GPU start 是否晚
        gpu_start_delay_ns = None
        if target.gpu_start_ns is not None:
            gpu_start_delay_ns = self._estimate_gpu_start_delay(target)
            if gpu_start_delay_ns is not None and gpu_start_delay_ns > self.close_threshold_ns:
                return SlowdownDiagnosis(
                    target_id=target.kid,
                    slowdown_type=SlowdownType.DEPENDENCY_BLOCKED,
                    confidence=0.7,
                    gpu_start_delay_ns=gpu_start_delay_ns,
                    evidence={
                        "reason": "GPU start time is delayed",
                        "gpu_delay_ns": gpu_start_delay_ns,
                    },
                )

        # 检查 normalized progress 是否低
        if peers and target.work_value and target.gpu_dur_ns:
            normalized_progress = self._check_progress_anomaly(target, peers)
            if normalized_progress is not None:
                return SlowdownDiagnosis(
                    target_id=target.kid,
                    slowdown_type=SlowdownType.RESOURCE_SLOWED,
                    confidence=0.75,
                    normalized_progress=normalized_progress["target_progress"],
                    peer_median_progress=normalized_progress["peer_median"],
                    evidence={
                        "reason": "Normalized progress is anomalously low",
                        "target_progress": normalized_progress["target_progress"],
                        "peer_median": normalized_progress["peer_median"],
                        "peer_mad": normalized_progress["peer_mad"],
                    },
                )

        # 无法确定
        return SlowdownDiagnosis(
            target_id=target.kid,
            slowdown_type=SlowdownType.UNCERTAIN,
            confidence=0.3,
            evidence={"reason": "Insufficient evidence for classification"},
        )

    def _estimate_cpu_delay(self, target: KernelRecord) -> Optional[int]:
        """
        估计 CPU 侧 enqueue 的延迟。
        
        简单启发式：与前一个 kernel 的 enqueue 时间比较。
        """
        if target.cpu_enqueue_start_ns is None:
            return None

        # 这里需要前驱 kernel 信息，暂时返回 None
        # 真正实现时需要传入前驱 kernel
        return None

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
        if not target.progress():
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

    def classify_batch(self, 
                      targets: List[KernelRecord],
                      all_records: Optional[List[KernelRecord]] = None) -> Dict[str, SlowdownDiagnosis]:
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
            diagnosis = self.classify_slowdown(target, peers)
            result[target.kid] = diagnosis

        return result
