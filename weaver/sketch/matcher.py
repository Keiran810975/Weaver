"""
将运行时 kernel 映射到 ExecutionSketch 中的模板节点。

matcher 提供 kernel 到 template 的映射逻辑。
"""

import re
from typing import Optional, List, Dict, Any
from .schema import ExecutionSketch, KernelTemplate
from .rules import classify_kernel, KernelRecord


class KernelMatcher:
    """将运行时 kernel 映射到 sketch template。"""

    def __init__(self, sketch: ExecutionSketch):
        """
        初始化 matcher。
        
        Args:
            sketch: ExecutionSketch 对象
        """
        self.sketch = sketch
        self.templates = sketch.kernel_templates

    def match_kernel(self, kernel_record: KernelRecord) -> Optional[KernelTemplate]:
        """
        找到最匹配的 kernel template。
        
        Args:
            kernel_record: KernelRecord 对象
            
        返回：最匹配的 KernelTemplate，如果没有找到返回 None
        """
        # 首先用分类器快速分类
        kclass = classify_kernel(kernel_record)

        # 找所有 family 匹配的模板
        candidates = [t for t in self.templates if t.family == kclass.family]

        if not candidates:
            return None

        # 如果只有一个候选，直接返回
        if len(candidates) == 1:
            return candidates[0]

        # 多个候选时，尝试更精细的匹配
        best_match = None
        best_score = -1

        for template in candidates:
            score = self._compute_match_score(kernel_record, template)
            if score > best_score:
                best_score = score
                best_match = template

        return best_match

    def _compute_match_score(self, kernel_record: KernelRecord, template: KernelTemplate) -> float:
        """
        计算 kernel 与 template 的匹配度分数。
        分数越高越匹配。
        
        Args:
            kernel_record: KernelRecord 对象
            template: KernelTemplate 对象
            
        返回：匹配度分数 [0, 1]
        """
        score = 0.0

        # 基础分数
        if template.family != "UNKNOWN":
            score += 0.2

        # 检查 kernel name 是否匹配 regex
        if "kernel_name_regex" in template.match:
            regex = template.match["kernel_name_regex"]
            if self._regex_match(kernel_record.kernel_name, regex):
                score += 0.3

        # 检查 operator name 是否匹配 regex
        if "operator_regex" in template.match and kernel_record.operator_name:
            regex = template.match["operator_regex"]
            if self._regex_match(kernel_record.operator_name, regex):
                score += 0.3

        # 检查 event_kind 是否匹配
        if "event_kind" in template.match and kernel_record.kind:
            if kernel_record.kind.lower() == template.match["event_kind"].lower():
                score += 0.2

        return min(score, 1.0)

    def _regex_match(self, text: str, pattern: str) -> bool:
        """
        使用正则表达式进行匹配。
        
        Args:
            text: 要匹配的文本
            pattern: 正则表达式模式
            
        返回：是否匹配
        """
        try:
            return bool(re.search(pattern, text, re.IGNORECASE))
        except re.error:
            return False

    def match_kernels(self, kernel_records: List[KernelRecord]) -> List[tuple[KernelRecord, Optional[KernelTemplate]]]:
        """
        批量匹配 kernel 记录。
        
        Args:
            kernel_records: KernelRecord 列表
            
        返回：(KernelRecord, KernelTemplate) 元组列表
        """
        return [(kr, self.match_kernel(kr)) for kr in kernel_records]

    def get_template_by_id(self, template_id: str) -> Optional[KernelTemplate]:
        """按 ID 获取 template。"""
        for template in self.templates:
            if template.template_id == template_id:
                return template
        return None

    def get_templates_by_family(self, family: str) -> List[KernelTemplate]:
        """获取某个 family 的所有 templates。"""
        return [t for t in self.templates if t.family == family]
