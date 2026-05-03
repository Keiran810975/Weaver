"""
ExecutionSketch 数据结构定义。

ExecutionSketch 是从当前 trace 提取的轻量级执行结构，用于支持后续因果诊断。
它不存历史时间、平均 duration 或完整执行序列，只提供：
- kernel 类型分类
- 同类分组（family/tag）
- 工作量单位
- 必要依赖
- 可能 overlap 关系
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class WorkUnitType(str, Enum):
    """工作量单位类型。"""
    FLOPS = "flops"
    BYTES = "bytes"
    ELEMENTS = "elements"
    WARPS = "warps"
    UNKNOWN = "unknown"


class DependencyType(str, Enum):
    """依赖类型。"""
    HARD = "hard"
    SOFT = "soft"
    SYNC = "sync"


class OverlapExpectation(str, Enum):
    """Overlap 预期。"""
    MAY_OVERLAP = "may_overlap"
    SHOULD_OVERLAP = "should_overlap"
    MUST_NOT_OVERLAP = "must_not_overlap"


@dataclass
class WorkUnits:
    """工作量描述。"""
    type: WorkUnitType
    value: Optional[float] = None
    formula: Optional[str] = None
    confidence: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "value": self.value,
            "formula": self.formula,
            "confidence": self.confidence,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorkUnits":
        return WorkUnits(
            type=WorkUnitType(d.get("type", "unknown")),
            value=d.get("value"),
            formula=d.get("formula"),
            confidence=d.get("confidence", "low"),
        )


@dataclass
class KernelTemplate:
    """Kernel 类型模板。"""
    template_id: str
    family: str  # GEMM | NCCL | MEMCPY | REDUCTION | UNKNOWN
    tag: str     # GEMM_large | NCCL_64MB | MEMCPY_large 等
    match: Dict[str, Any]  # kernel_name_regex, operator_regex 等
    work_units: WorkUnits
    resource_hint: List[str] = field(default_factory=list)  # ["compute", "communication", "memory"]
    expected_behavior: Dict[str, Any] = field(default_factory=dict)  # {"allow_overlap": true, "critical": true}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "family": self.family,
            "tag": self.tag,
            "match": self.match,
            "work_units": self.work_units.to_dict(),
            "resource_hint": self.resource_hint,
            "expected_behavior": self.expected_behavior,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "KernelTemplate":
        return KernelTemplate(
            template_id=d["template_id"],
            family=d["family"],
            tag=d["tag"],
            match=d.get("match", {}),
            work_units=WorkUnits.from_dict(d.get("work_units", {"type": "unknown"})),
            resource_hint=d.get("resource_hint", []),
            expected_behavior=d.get("expected_behavior", {}),
        )


@dataclass
class DependencyRule:
    """依赖关系规则。"""
    rule_id: str
    type: DependencyType  # hard | soft | sync
    source: str  # timeline | python_api | heuristic
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "type": self.type.value,
            "source": self.source,
            "description": self.description,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DependencyRule":
        return DependencyRule(
            rule_id=d["rule_id"],
            type=DependencyType(d.get("type", "hard")),
            source=d.get("source", ""),
            description=d.get("description", ""),
        )


@dataclass
class OverlapRelation:
    """Overlap 预期。"""
    relation_id: str
    left_family: str
    right_family: str
    phase: Optional[str] = None  # forward | backward | unknown
    expected: OverlapExpectation = OverlapExpectation.MAY_OVERLAP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "phase": self.phase,
            "expected": self.expected.value,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OverlapRelation":
        return OverlapRelation(
            relation_id=d["relation_id"],
            left_family=d["left_family"],
            right_family=d["right_family"],
            phase=d.get("phase"),
            expected=OverlapExpectation(d.get("expected", "may_overlap")),
        )


@dataclass
class ExecutionSketch:
    """执行草图。轻量级结构，不存历史数据，只提供语义抽象。"""
    metadata: Dict[str, Any]
    kernel_templates: List[KernelTemplate] = field(default_factory=list)
    dependency_rules: List[DependencyRule] = field(default_factory=list)
    overlap_expectations: List[OverlapRelation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "kernel_templates": [kt.to_dict() for kt in self.kernel_templates],
            "dependency_rules": [dr.to_dict() for dr in self.dependency_rules],
            "overlap_expectations": [oe.to_dict() for oe in self.overlap_expectations],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExecutionSketch":
        return ExecutionSketch(
            metadata=d.get("metadata", {}),
            kernel_templates=[KernelTemplate.from_dict(kt) for kt in d.get("kernel_templates", [])],
            dependency_rules=[DependencyRule.from_dict(dr) for dr in d.get("dependency_rules", [])],
            overlap_expectations=[OverlapRelation.from_dict(oe) for oe in d.get("overlap_expectations", [])],
        )


@dataclass
class KernelClass:
    """Kernel 分类结果。"""
    family: str  # GEMM | NCCL | MEMCPY | REDUCTION | UNKNOWN
    tag: str     # 细粒度标签
    confidence: float = 1.0
