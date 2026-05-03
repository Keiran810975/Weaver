"""
Sketch rules 单元测试。

测试 kernel 分类规则是否正确。
"""

import pytest
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from weaver.sketch.rules import (
    classify_kernel, KernelRecord,
    get_default_kernel_templates,
    get_default_dependency_rules,
    get_default_overlap_expectations,
)


class TestKernelClassification:
    """测试 kernel 分类规则。"""

    def test_classify_gemm(self):
        """测试 GEMM kernel 分类。"""
        record = KernelRecord(
            kernel_name="cutlass_kernel_gemm",
            operator_name="linear",
            grid=(64, 64, 1),
            block=(256, 1, 1),
        )
        kclass = classify_kernel(record)
        assert kclass.family == "GEMM"
        assert "GEMM" in kclass.tag

    def test_classify_matmul(self):
        """测试 MatMul kernel 分类。"""
        record = KernelRecord(
            kernel_name="ampere_fp32_gemm_v1",
            operator_name="matmul",
        )
        kclass = classify_kernel(record)
        assert kclass.family == "GEMM"

    def test_classify_nccl_allreduce(self):
        """测试 NCCL AllReduce 分类。"""
        record = KernelRecord(
            kernel_name="ncclKernel_AllReduce",
            kind="nccl_all_reduce",
            event_type="nccl_all_reduce",
            payload={"count": 1024 * 1024, "dtype_size": 4},  # 4MB
        )
        kclass = classify_kernel(record)
        assert kclass.family == "NCCL"
        assert "NCCL" in kclass.tag

    def test_classify_nccl_allgather(self):
        """测试 NCCL AllGather 分类。"""
        record = KernelRecord(
            kernel_name="ncclKernel_AllGather",
            kind="nccl_all_gather",
            payload={"count": 64 * 1024 * 1024, "dtype_size": 4},  # 256MB
        )
        kclass = classify_kernel(record)
        assert kclass.family == "NCCL"

    def test_classify_memcpy(self):
        """测试 MEMCPY kernel 分类。"""
        record = KernelRecord(
            kernel_name="memcpy_cuda_device_to_device",
            payload={"bytes": 100 * 1024 * 1024},
        )
        kclass = classify_kernel(record)
        assert kclass.family == "MEMCPY"
        assert "MEMCPY" in kclass.tag

    def test_classify_copy_operator(self):
        """测试 copy 操作分类。"""
        record = KernelRecord(
            kernel_name="kernel_1",
            operator_name="copy",
        )
        kclass = classify_kernel(record)
        assert kclass.family == "MEMCPY"

    def test_classify_reduction(self):
        """测试 Reduction kernel 分类。"""
        record = KernelRecord(
            kernel_name="reduction_sum_kernel",
            operator_name="reduce_sum",
        )
        kclass = classify_kernel(record)
        assert kclass.family == "REDUCTION"

    def test_classify_elementwise_relu(self):
        """测试 ElementWise ReLU 分类。"""
        record = KernelRecord(
            kernel_name="relu_kernel",
            operator_name="relu",
        )
        kclass = classify_kernel(record)
        assert kclass.family == "ELEMENTWISE"

    def test_classify_unknown(self):
        """测试未知 kernel 分类。"""
        record = KernelRecord(
            kernel_name="unknown_exotic_kernel_xyz",
        )
        kclass = classify_kernel(record)
        assert kclass.family == "UNKNOWN"


class TestDefaultTemplates:
    """测试默认 kernel 模板。"""

    def test_get_default_kernel_templates(self):
        """测试获取默认 kernel 模板。"""
        templates = get_default_kernel_templates()
        assert len(templates) > 0

        # 检查是否包含必要的 family
        families = {t.family for t in templates}
        assert "GEMM" in families
        assert "NCCL" in families
        assert "MEMCPY" in families

    def test_gemm_template_structure(self):
        """测试 GEMM 模板结构。"""
        templates = get_default_kernel_templates()
        gemm_templates = [t for t in templates if t.family == "GEMM"]

        assert len(gemm_templates) > 0
        for t in gemm_templates:
            assert t.template_id
            assert t.family == "GEMM"
            assert t.tag
            assert t.work_units
            assert "compute" in t.resource_hint

    def test_nccl_template_has_communication_resource(self):
        """测试 NCCL 模板包含通信资源标记。"""
        templates = get_default_kernel_templates()
        nccl_templates = [t for t in templates if t.family == "NCCL"]

        assert len(nccl_templates) > 0
        for t in nccl_templates:
            assert "communication" in t.resource_hint or "memory" in t.resource_hint


class TestDefaultRules:
    """测试默认依赖规则。"""

    def test_get_default_dependency_rules(self):
        """测试获取默认依赖规则。"""
        rules = get_default_dependency_rules()
        assert len(rules) > 0

        # 检查是否包含必要的规则
        rule_ids = {r.rule_id for r in rules}
        assert "same_stream_order" in rule_ids
        assert "sync_serializes" in rule_ids


class TestDefaultOverlapExpectations:
    """测试默认 overlap 期望。"""

    def test_get_default_overlap_expectations(self):
        """测试获取默认 overlap 期望。"""
        expectations = get_default_overlap_expectations()
        assert len(expectations) > 0

        # 检查是否包含 GEMM-NCCL overlap
        gemm_nccl_overlaps = [
            e for e in expectations
            if (e.left_family == "GEMM" and e.right_family == "NCCL") or
               (e.left_family == "NCCL" and e.right_family == "GEMM")
        ]
        assert len(gemm_nccl_overlaps) > 0

    def test_overlap_expectation_values(self):
        """测试 overlap 期望值。"""
        expectations = get_default_overlap_expectations()

        for exp in expectations:
            assert exp.relation_id
            assert exp.left_family
            assert exp.right_family
            assert exp.expected.value in ["may_overlap", "should_overlap", "must_not_overlap"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
