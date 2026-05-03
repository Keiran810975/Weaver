"""
Kernel 分类规则。

根据 kernel name、operator name 和其他元数据对 kernel 进行分类，
生成 family（GEMM/NCCL/MEMCPY/REDUCTION/UNKNOWN）和 tag（粗粒度分组）。
"""

import re
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .schema import KernelClass, WorkUnits, WorkUnitType


@dataclass
class KernelRecord:
    """运行时 kernel 记录（最小化视图用于分类）。"""
    kernel_name: str
    operator_name: Optional[str] = None
    kind: Optional[str] = None
    event_type: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    grid: Optional[tuple] = None
    block: Optional[tuple] = None
    shared_memory: Optional[int] = None


def classify_kernel(record: KernelRecord) -> KernelClass:
    """
    对 kernel 进行分类。
    返回 KernelClass，包含 family 和 tag。
    """
    name = record.kernel_name.lower()
    op = (record.operator_name or "").lower()
    kind = (record.kind or "").lower()
    event_type = (record.event_type or "").lower()

    # NCCL 优先级最高
    if "nccl" in name or kind.startswith("nccl_") or event_type.startswith("nccl_"):
        return _classify_nccl(record)

    # GEMM / MatMul
    if any(p in name for p in ["gemm", "matmul", "linear"]) or any(p in op for p in ["mm", "matmul", "linear", "addmm", "bmm"]):
        return _classify_gemm(record)

    # Memory / Copy
    if any(p in name for p in ["memcpy", "copy", "dtod", "htod", "dtoh"]) or "contiguous" in op or "copy" in op:
        return _classify_memcpy(record)

    # Reduction
    if any(p in name for p in ["reduce", "sum", "mean", "max", "min"]) or any(p in op for p in ["reduce", "sum", "mean", "max", "min"]):
        return _classify_reduction(record)

    # ElementWise
    if any(p in name for p in ["add", "mul", "div", "sub", "exp", "log", "relu", "sigmoid", "tanh", "gelu"]) or \
       any(p in op for p in ["add", "mul", "div", "sub", "exp", "log", "relu", "sigmoid", "tanh", "gelu"]):
        return _classify_elementwise(record)

    # Unknown
    return KernelClass(family="UNKNOWN", tag="UNKNOWN")


def _classify_nccl(record: KernelRecord) -> KernelClass:
    """NCCL collective 分类。"""
    name = record.kernel_name.lower()
    kind = (record.kind or "").lower()

    # 尝试从 payload 推断数据大小
    payload = record.payload or {}
    count = payload.get("count", 0)
    dtype_size = payload.get("dtype_size", 4)
    data_bytes = count * dtype_size if count > 0 else 0

    # 确定操作类型
    op_type = "allreduce"
    if "gather" in name or "gather" in kind:
        op_type = "allgather"
    elif "scatter" in name or "scatter" in kind:
        op_type = "reducescatter"
    elif "broadcast" in name or "broadcast" in kind:
        op_type = "broadcast"
    elif "reduce" in name or "reduce" in kind:
        op_type = "allreduce"

    # 按数据大小分 bucket
    if data_bytes <= 16 * 1024 * 1024:
        size_tag = "<=16MB"
    elif data_bytes <= 128 * 1024 * 1024:
        size_tag = "16-128MB"
    else:
        size_tag = ">128MB"

    tag = f"NCCL_{op_type}_{size_tag}"

    return KernelClass(family="NCCL", tag=tag, confidence=0.95)


def _classify_gemm(record: KernelRecord) -> KernelClass:
    """GEMM / MatMul 分类。"""
    name = record.kernel_name.lower()
    op = (record.operator_name or "").lower()

    # 尝试从 grid/block 推断工作量
    grid = record.grid or (1, 1, 1)
    block = record.block or (128, 1, 1)
    total_warps = (grid[0] * grid[1] * grid[2]) * (block[0] * block[1] * block[2] // 32)

    # 按工作量大小分 bucket
    if total_warps < 256:
        size_tag = "small"
    elif total_warps < 4096:
        size_tag = "medium"
    else:
        size_tag = "large"

    tag = f"GEMM_{size_tag}"

    return KernelClass(family="GEMM", tag=tag, confidence=0.9)


def _classify_memcpy(record: KernelRecord) -> KernelClass:
    """Memory copy 分类。"""
    name = record.kernel_name.lower()
    payload = record.payload or {}

    # 尝试从 payload 推断大小
    bytes_count = payload.get("bytes", 0)

    if bytes_count < 16 * 1024 * 1024:
        size_tag = "small"
    else:
        size_tag = "large"

    tag = f"MEMCPY_{size_tag}"

    return KernelClass(family="MEMCPY", tag=tag, confidence=0.85)


def _classify_reduction(record: KernelRecord) -> KernelClass:
    """Reduction 分类。"""
    tag = "REDUCTION"
    return KernelClass(family="REDUCTION", tag=tag, confidence=0.8)


def _classify_elementwise(record: KernelRecord) -> KernelClass:
    """ElementWise 分类。"""
    tag = "ELEMENTWISE"
    return KernelClass(family="ELEMENTWISE", tag=tag, confidence=0.7)


def make_nccl_tag(record: KernelRecord) -> str:
    """生成 NCCL tag。"""
    return _classify_nccl(record).tag


def make_gemm_tag(record: KernelRecord) -> str:
    """生成 GEMM tag。"""
    return _classify_gemm(record).tag


def make_memcpy_tag(record: KernelRecord) -> str:
    """生成 MEMCPY tag。"""
    return _classify_memcpy(record).tag


def get_default_kernel_templates() -> list:
    """
    获取默认的 kernel 模板列表。
    这些是最常见的 kernel 类型及其特征。
    """
    from .schema import KernelTemplate, WorkUnits, WorkUnitType

    return [
        KernelTemplate(
            template_id="gemm_small",
            family="GEMM",
            tag="GEMM_small",
            match={
                "kernel_name_regex": ".*gemm.*|.*matmul.*",
                "operator_regex": ".*mm.*|.*matmul.*|.*linear.*",
            },
            work_units=WorkUnits(type=WorkUnitType.FLOPS, formula="2*M*N*K", confidence="medium"),
            resource_hint=["compute"],
            expected_behavior={"allow_overlap": True, "critical": True},
        ),
        KernelTemplate(
            template_id="gemm_medium",
            family="GEMM",
            tag="GEMM_medium",
            match={"kernel_name_regex": ".*gemm.*|.*matmul.*"},
            work_units=WorkUnits(type=WorkUnitType.FLOPS, formula="2*M*N*K", confidence="medium"),
            resource_hint=["compute"],
            expected_behavior={"allow_overlap": True, "critical": True},
        ),
        KernelTemplate(
            template_id="gemm_large",
            family="GEMM",
            tag="GEMM_large",
            match={"kernel_name_regex": ".*gemm.*|.*matmul.*"},
            work_units=WorkUnits(type=WorkUnitType.FLOPS, formula="2*M*N*K", confidence="medium"),
            resource_hint=["compute"],
            expected_behavior={"allow_overlap": True, "critical": True},
        ),
        KernelTemplate(
            template_id="nccl_allreduce",
            family="NCCL",
            tag="NCCL_allreduce_*",
            match={
                "event_kind": "nccl_all_reduce",
                "kernel_name_regex": ".*nccl.*",
            },
            work_units=WorkUnits(type=WorkUnitType.BYTES, confidence="medium"),
            resource_hint=["communication", "memory"],
            expected_behavior={"allow_overlap": True, "critical": True},
        ),
        KernelTemplate(
            template_id="nccl_allgather",
            family="NCCL",
            tag="NCCL_allgather_*",
            match={"event_kind": "nccl_all_gather"},
            work_units=WorkUnits(type=WorkUnitType.BYTES, confidence="medium"),
            resource_hint=["communication", "memory"],
            expected_behavior={"allow_overlap": True, "critical": True},
        ),
        KernelTemplate(
            template_id="nccl_reducescatter",
            family="NCCL",
            tag="NCCL_reducescatter_*",
            match={"event_kind": "nccl_reduce_scatter"},
            work_units=WorkUnits(type=WorkUnitType.BYTES, confidence="medium"),
            resource_hint=["communication", "memory"],
            expected_behavior={"allow_overlap": True, "critical": True},
        ),
        KernelTemplate(
            template_id="nccl_broadcast",
            family="NCCL",
            tag="NCCL_broadcast_*",
            match={"event_kind": "nccl_broadcast"},
            work_units=WorkUnits(type=WorkUnitType.BYTES, confidence="medium"),
            resource_hint=["communication", "memory"],
            expected_behavior={"allow_overlap": True, "critical": True},
        ),
        KernelTemplate(
            template_id="memcpy",
            family="MEMCPY",
            tag="MEMCPY_*",
            match={"kernel_name_regex": ".*memcpy.*|.*copy.*"},
            work_units=WorkUnits(type=WorkUnitType.BYTES, confidence="medium"),
            resource_hint=["memory"],
            expected_behavior={"allow_overlap": True, "critical": False},
        ),
    ]


def get_default_dependency_rules() -> list:
    """获取默认的依赖规则。"""
    from .schema import DependencyRule, DependencyType

    return [
        DependencyRule(
            rule_id="same_stream_order",
            type=DependencyType.HARD,
            source="timeline",
            description="kernels in the same CUDA stream preserve order",
        ),
        DependencyRule(
            rule_id="sync_serializes",
            type=DependencyType.HARD,
            source="python_api",
            description="cuda synchronize forces later GPU work to wait",
        ),
        DependencyRule(
            rule_id="event_wait_serializes",
            type=DependencyType.HARD,
            source="timeline",
            description="event wait on another stream enforces order",
        ),
    ]


def get_default_overlap_expectations() -> list:
    """获取默认的 overlap 预期。"""
    from .schema import OverlapRelation, OverlapExpectation

    return [
        OverlapRelation(
            relation_id="backward_compute_grad_comm",
            left_family="GEMM",
            right_family="NCCL",
            phase="backward",
            expected=OverlapExpectation.MAY_OVERLAP,
        ),
        OverlapRelation(
            relation_id="forward_compute_nccl",
            left_family="GEMM",
            right_family="NCCL",
            phase="forward",
            expected=OverlapExpectation.MAY_OVERLAP,
        ),
        OverlapRelation(
            relation_id="memcpy_compute",
            left_family="MEMCPY",
            right_family="GEMM",
            phase=None,
            expected=OverlapExpectation.MAY_OVERLAP,
        ),
    ]
