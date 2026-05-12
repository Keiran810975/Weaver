"""
规范化记录定义。

将 aligned timeline 中的混合事件转换为统一的规范化记录格式，
供诊断分析使用。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class SyncKind(str, Enum):
    """同步事件类型。"""
    CUDA_SYNCHRONIZE = "cuda_synchronize"
    EVENT_WAIT = "event_wait"
    PROFILER_TIMER = "profiler_timer"
    UNKNOWN_SYNC = "unknown_sync"


class SlowdownType(str, Enum):
    """性能下降类型。"""
    CPU_RUNTIME_BLOCKED = "cpu_runtime_blocked"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    RESOURCE_SLOWED = "resource_slowed"
    UNCERTAIN = "uncertain"


@dataclass
class KernelRecord:
    """规范化的 kernel 记录。"""
    # 标识
    kid: str  # kernel id
    rank: Optional[int] = None
    pid: int = 0
    tid: Optional[int] = None
    stream: Optional[str] = None

    # 分类
    kernel_name: str = ""
    family: str = ""  # GEMM, NCCL, MEMCPY, etc.
    tag: str = ""     # GEMM_large, NCCL_64MB, etc.

    # 算子信息
    operator_name: Optional[str] = None
    operator_id: Optional[str] = None

    # 时间信息
    cpu_enqueue_start_ns: Optional[int] = None
    cpu_enqueue_end_ns: Optional[int] = None
    gpu_start_ns: Optional[int] = None
    gpu_end_ns: Optional[int] = None
    duration_ns: Optional[int] = None

    # 执行信息
    grid: Optional[Tuple[int, int, int]] = None
    block: Optional[Tuple[int, int, int]] = None
    total_warps: Optional[int] = None
    shared_memory: Optional[int] = None

    # 工作量
    work_type: str = "unknown"  # flops, bytes, elements, warps
    work_value: Optional[float] = None

    # 其他
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # hook, profiler, python

    @property
    def interval(self) -> Optional[Tuple[int, int]]:
        """返回执行区间 (start, end)。使用 GPU 时间如果可用。"""
        if self.gpu_start_ns is not None and self.gpu_end_ns is not None:
            return (self.gpu_start_ns, self.gpu_end_ns)
        if self.cpu_enqueue_start_ns is not None and self.cpu_enqueue_end_ns is not None:
            return (self.cpu_enqueue_start_ns, self.cpu_enqueue_end_ns)
        return None

    @property
    def cpu_enqueue_dur_ns(self) -> Optional[int]:
        """CPU 侧 launch 耗时。"""
        if self.cpu_enqueue_start_ns is not None and self.cpu_enqueue_end_ns is not None:
            return self.cpu_enqueue_end_ns - self.cpu_enqueue_start_ns
        return None

    @property
    def gpu_dur_ns(self) -> Optional[int]:
        """GPU 侧执行耗时。"""
        if self.gpu_start_ns is not None and self.gpu_end_ns is not None:
            return self.gpu_end_ns - self.gpu_start_ns
        return None

    def progress(self) -> Optional[float]:
        """计算工作进度（工作量/耗时）。"""
        if self.work_value is None:
            return None
        duration = self.gpu_dur_ns if self.gpu_dur_ns is not None else self.duration_ns
        if duration is None or duration <= 0:
            return None
        return self.work_value / duration


@dataclass
class OperatorRecord:
    """规范化的操作符记录。"""
    oid: str
    operator_name: str
    phase: Optional[str] = None  # forward, backward
    rank: Optional[int] = None
    pid: int = 0

    # 时间信息
    ts_start_ns: int = 0
    ts_end_ns: Optional[int] = None

    # 关联的 kernel ids
    kernel_ids: List[str] = field(default_factory=list)

    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> Optional[int]:
        if self.ts_end_ns is not None:
            return self.ts_end_ns - self.ts_start_ns
        return None


@dataclass
class SyncRecord:
    """规范化的同步事件记录。"""
    sid: str
    kind: SyncKind
    rank: Optional[int] = None
    pid: int = 0

    # 时间
    ts_start_ns: int = 0
    ts_end_ns: Optional[int] = None

    # 调用栈
    stack: List[str] = field(default_factory=list)

    # 来源
    source: str = ""  # python, hook, profiler

    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> Optional[int]:
        if self.ts_end_ns is not None:
            return self.ts_end_ns - self.ts_start_ns
        return None

    @property
    def interval(self) -> Optional[Tuple[int, int]]:
        """返回同步事件区间。"""
        if self.ts_end_ns is not None:
            return (self.ts_start_ns, self.ts_end_ns)
        return None


@dataclass
class OverlapRelation:
    """记录两个 kernel 之间的 overlap 关系。"""
    target_id: str
    candidate_id: str

    # 时间
    target_interval: Tuple[int, int]  # (start, end)
    candidate_interval: Tuple[int, int]

    # 计算的 overlap
    overlap_ns: int = 0

    @property
    def overlap_ratio_target(self) -> float:
        """相对于 target 的 overlap 比例。"""
        target_dur = self.target_interval[1] - self.target_interval[0]
        if target_dur == 0:
            return 0.0
        return min(1.0, self.overlap_ns / target_dur)

    @property
    def overlap_ratio_candidate(self) -> float:
        """相对于 candidate 的 overlap 比例。"""
        candidate_dur = self.candidate_interval[1] - self.candidate_interval[0]
        if candidate_dur == 0:
            return 0.0
        return min(1.0, self.overlap_ns / candidate_dur)


@dataclass
class Candidate:
    """可疑的 target kernel。"""
    target_id: str
    candidate_type: str  # structural_deviation, progress_outlier, rank_outlier
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "candidate_type": self.candidate_type,
            "reason": self.reason,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass
class SlowdownDiagnosis:
    """性能下降诊断结果。"""
    target_id: str
    slowdown_type: SlowdownType
    confidence: float

    # 指标
    cpu_enqueue_delay_ns: Optional[int] = None
    gpu_start_delay_ns: Optional[int] = None
    normalized_progress: Optional[float] = None
    peer_median_progress: Optional[float] = None

    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "slowdown_type": self.slowdown_type.value,
            "confidence": self.confidence,
            "cpu_enqueue_delay_ns": self.cpu_enqueue_delay_ns,
            "gpu_start_delay_ns": self.gpu_start_delay_ns,
            "normalized_progress": self.normalized_progress,
            "peer_median_progress": self.peer_median_progress,
            "evidence": self.evidence,
        }


@dataclass
class DependencyDiagnosis:
    """依赖阻塞诊断。"""
    target_id: str
    blocker_id: str
    delay_ns: int
    blocker_kind: str = "kernel"  # kernel | sync | host
    overlap_loss_ns: Optional[int] = None
    counterfactual_start_ns: Optional[int] = None
    confidence: float = 0.5

    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "blocker_id": self.blocker_id,
            "blocker_kind": self.blocker_kind,
            "delay_ns": self.delay_ns,
            "overlap_loss_ns": self.overlap_loss_ns,
            "counterfactual_start_ns": self.counterfactual_start_ns,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class ResourceDiagnosis:
    """资源干扰诊断。"""
    target_id: str
    culprit_id: str
    target_progress: Optional[float] = None
    peer_progress_median: Optional[float] = None
    overlap_ratio: float = 0.0
    dose_response_score: float = 0.0
    resource_hint: str = "unknown"
    warp_block_verdict: str = "not_available"
    confidence: float = 0.5

    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "culprit_id": self.culprit_id,
            "target_progress": self.target_progress,
            "peer_progress_median": self.peer_progress_median,
            "overlap_ratio": self.overlap_ratio,
            "dose_response_score": self.dose_response_score,
            "resource_hint": self.resource_hint,
            "warp_block_verdict": self.warp_block_verdict,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }
