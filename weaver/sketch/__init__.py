"""
Weaver Sketch 模块。

用于从 aligned timeline 构建 ExecutionSketch（执行草图）。
ExecutionSketch 是轻量级的执行结构，不存历史数据，只提供语义抽象。
"""

from .schema import (
    WorkUnits,
    WorkUnitType,
    KernelTemplate,
    DependencyRule,
    DependencyType,
    OverlapRelation,
    OverlapExpectation,
    ExecutionSketch,
    KernelClass,
)
from .rules import (
    classify_kernel,
    KernelRecord,
    get_default_kernel_templates,
    get_default_dependency_rules,
    get_default_overlap_expectations,
)
from .builder import SketchBuilder
from .matcher import KernelMatcher

__all__ = [
    "WorkUnits",
    "WorkUnitType",
    "KernelTemplate",
    "DependencyRule",
    "DependencyType",
    "OverlapRelation",
    "OverlapExpectation",
    "ExecutionSketch",
    "KernelClass",
    "classify_kernel",
    "KernelRecord",
    "get_default_kernel_templates",
    "get_default_dependency_rules",
    "get_default_overlap_expectations",
    "SketchBuilder",
    "KernelMatcher",
]
