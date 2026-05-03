"""
诊断模块（Module 3）。

从规范化的 timeline records 进行性能诊断：
1. 发现可疑 target kernel（结构偏离、性能异常）
2. 分类性能下降类型（CPU blocked、dependency blocked、resource slowed）
3. 定位依赖阻塞的根因
4. 定位资源干扰
5. 生成综合诊断报告
"""

from .records import (
    SyncKind,
    SlowdownType,
    KernelRecord,
    OperatorRecord,
    SyncRecord,
    OverlapRelation,
    Candidate,
    SlowdownDiagnosis,
    DependencyDiagnosis,
    ResourceDiagnosis,
)

from .normalize import TimelineNormalizer

from .candidates import CandidateDiscovery

from .timing import TimingAnalyzer

from .dependency import DependencyLocalizer

from .resource import ResourceLocalizer

from .report import DiagnosisReporter

from . import cli

__all__ = [
    # Records
    "SyncKind",
    "SlowdownType",
    "KernelRecord",
    "OperatorRecord",
    "SyncRecord",
    "OverlapRelation",
    "Candidate",
    "SlowdownDiagnosis",
    "DependencyDiagnosis",
    "ResourceDiagnosis",
    # Modules
    "TimelineNormalizer",
    "CandidateDiscovery",
    "TimingAnalyzer",
    "DependencyLocalizer",
    "ResourceLocalizer",
    "DiagnosisReporter",
    # CLI
    "cli",
]
