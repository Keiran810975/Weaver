"""
从 aligned timeline 构建 ExecutionSketch。

支持两种模式：
- 模式 A：规则生成（从 trace 自动提取结构）
- 模式 B：用户提供 hints（用户指定 overlap 期望等）
"""

import json
import argparse
import os
import sys
from typing import Optional, Dict, Any, List, Set
from pathlib import Path
import re

from .schema import (
    ExecutionSketch, KernelTemplate, DependencyRule, OverlapRelation,
    WorkUnits, WorkUnitType, DependencyType, OverlapExpectation
)
from .rules import (
    classify_kernel, KernelRecord,
    get_default_kernel_templates,
    get_default_dependency_rules,
    get_default_overlap_expectations,
)


class SketchBuilder:
    """从 timeline 构建 execution sketch。"""

    def __init__(self, timeline_path: str, hints_path: Optional[str] = None):
        """
        初始化 builder。
        
        Args:
            timeline_path: aligned_timeline_rank*.ndjson 的路径
            hints_path: 可选的 user hints JSON 文件路径
        """
        self.timeline_path = timeline_path
        self.hints_path = hints_path
        self.kernel_records: List[Dict[str, Any]] = []
        self.kernel_families: Set[str] = set()
        self.kernel_tags: Set[str] = set()
        self.user_hints: Dict[str, Any] = {}

        if hints_path and os.path.exists(hints_path):
            with open(hints_path, 'r') as f:
                self.user_hints = json.load(f)

    def load_timeline(self) -> None:
        """从 NDJSON timeline 加载 kernel 事件。"""
        self.kernel_records = []

        try:
            with open(self.timeline_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        # 筛选 kernel 和 nccl 事件
                        if self._is_kernel_event(record):
                            self.kernel_records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
        except FileNotFoundError:
            raise FileNotFoundError(f"Timeline file not found: {self.timeline_path}")

    def _is_kernel_event(self, record: Dict[str, Any]) -> bool:
        """判断是否为 kernel 或 nccl 事件。"""
        kind = record.get("kind", "").lower()
        event_type = record.get("event_type", "").lower()

        return (
            "kernel" in kind or
            "hook" in kind or
            "nccl" in kind or
            "cuda" in kind or
            kind.startswith("nccl_") or
            event_type.startswith("nccl_")
        )

    def extract_kernel_families_and_tags(self) -> None:
        """从加载的记录中提取 kernel family 和 tag。"""
        self.kernel_families = set()
        self.kernel_tags = set()

        for record in self.kernel_records:
            kernel_record = self._to_kernel_record(record)
            if kernel_record:
                kclass = classify_kernel(kernel_record)
                self.kernel_families.add(kclass.family)
                self.kernel_tags.add(kclass.tag)

    def _to_kernel_record(self, timeline_record: Dict[str, Any]) -> Optional[KernelRecord]:
        """将 timeline 记录转换为 KernelRecord。"""
        kernel_name = timeline_record.get("kernel_name") or timeline_record.get("name", "")
        operator_name = timeline_record.get("operator_name")
        kind = timeline_record.get("kind")
        event_type = timeline_record.get("event_type")

        if not kernel_name:
            return None

        grid = tuple(timeline_record.get("grid", [1, 1, 1]))
        block = tuple(timeline_record.get("block", [128, 1, 1]))
        shared_memory = timeline_record.get("shared_memory")

        payload = {}
        if "count" in timeline_record:
            payload["count"] = timeline_record["count"]
        if "dtype_size" in timeline_record:
            payload["dtype_size"] = timeline_record["dtype_size"]
        if "bytes" in timeline_record:
            payload["bytes"] = timeline_record["bytes"]

        return KernelRecord(
            kernel_name=kernel_name,
            operator_name=operator_name,
            kind=kind,
            event_type=event_type,
            payload=payload if payload else None,
            grid=grid,
            block=block,
            shared_memory=shared_memory,
        )

    def build_sketch(self) -> ExecutionSketch:
        """
        构建 execution sketch。
        
        返回：ExecutionSketch 对象
        """
        # 加载 timeline
        self.load_timeline()

        # 提取 family 和 tag
        self.extract_kernel_families_and_tags()

        # 构建 metadata
        metadata = {
            "schema_version": "0.1",
            "workload": self.user_hints.get("workload", "unknown"),
            "source": "auto_from_trace",
            "timeline_path": self.timeline_path,
            "num_kernel_events": len(self.kernel_records),
            "unique_families": list(self.kernel_families),
            "unique_tags": list(self.kernel_tags),
        }

        # 获取默认模板并根据实际情况过滤
        kernel_templates = self._build_kernel_templates()

        # 获取依赖规则
        dependency_rules = self._build_dependency_rules()

        # 获取 overlap 期望
        overlap_expectations = self._build_overlap_expectations()

        return ExecutionSketch(
            metadata=metadata,
            kernel_templates=kernel_templates,
            dependency_rules=dependency_rules,
            overlap_expectations=overlap_expectations,
        )

    def _build_kernel_templates(self) -> List[KernelTemplate]:
        """构建 kernel 模板列表。"""
        default_templates = get_default_kernel_templates()
        templates = []

        if len(self.kernel_families) == 0:
            # 如果 trace 中没有检测到任何 kernel，返回所有默认模板
            # 这样即使是没有 CUDA 事件的 trace 也能有一个完整的 sketch
            templates = default_templates
        else:
            # 只保留在当前 trace 中出现的 family 的模板
            for template in default_templates:
                if template.family in self.kernel_families:
                    templates.append(template)

            # 如果 trace 中有 UNKNOWN kernel，也添加
            if "UNKNOWN" in self.kernel_families:
                templates.append(KernelTemplate(
                    template_id="unknown",
                    family="UNKNOWN",
                    tag="UNKNOWN",
                    match={"description": "unclassified kernel"},
                    work_units=WorkUnits(type=WorkUnitType.UNKNOWN),
                    resource_hint=[],
                    expected_behavior={},
                ))

        return templates

    def _build_dependency_rules(self) -> List[DependencyRule]:
        """构建依赖规则。"""
        return get_default_dependency_rules()

    def _build_overlap_expectations(self) -> List[OverlapRelation]:
        """构建 overlap 期望。"""
        default_expectations = get_default_overlap_expectations()
        expectations = []

        # 从 user hints 读取期望的 overlaps
        if "expected_overlaps" in self.user_hints:
            for overlap_hint in self.user_hints["expected_overlaps"]:
                left_family = overlap_hint.get("left_family", "")
                right_family = overlap_hint.get("right_family", "")

                # 只在 trace 中两个 family 都出现时添加
                if left_family in self.kernel_families and right_family in self.kernel_families:
                    expectations.append(OverlapRelation(
                        relation_id=f"{left_family}_{right_family}_{overlap_hint.get('phase', 'unknown')}",
                        left_family=left_family,
                        right_family=right_family,
                        phase=overlap_hint.get("phase"),
                        expected=OverlapExpectation(overlap_hint.get("expected", "may_overlap")),
                    ))

        # 添加默认期望（两个 family 都在 trace 中）
        for default_exp in default_expectations:
            if (default_exp.left_family in self.kernel_families and
                default_exp.right_family in self.kernel_families):
                # 检查是否已经从 hints 中添加
                already_added = any(
                    e.left_family == default_exp.left_family and
                    e.right_family == default_exp.right_family and
                    e.phase == default_exp.phase
                    for e in expectations
                )
                if not already_added:
                    expectations.append(default_exp)

        return expectations


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="从 aligned timeline 构建 ExecutionSketch"
    )
    parser.add_argument(
        "--timeline",
        required=True,
        help="aligned_timeline_rank*.ndjson 的路径",
    )
    parser.add_argument(
        "--hints",
        help="可选的 user hints JSON 文件",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="输出 execution_sketch.json 的路径",
    )

    args = parser.parse_args()

    # 构建 sketch
    builder = SketchBuilder(args.timeline, args.hints)
    sketch = builder.build_sketch()

    # 输出为 JSON
    output_dir = os.path.dirname(args.out)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(args.out, 'w') as f:
        json.dump(sketch.to_dict(), f, indent=2)

    print(f"Sketch written to {args.out}")
    print(f"Kernel families: {sketch.metadata.get('unique_families', [])}")
    print(f"Kernel tags: {sketch.metadata.get('unique_tags', [])}")


if __name__ == "__main__":
    main()
