"""
诊断报告生成模块。

从各种诊断结果生成 JSON 格式的诊断报告。
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import asdict

from .records import (
    KernelRecord, Candidate, SlowdownDiagnosis, 
    DependencyDiagnosis, ResourceDiagnosis
)


class DiagnosisReporter:
    """诊断报告生成。"""

    def __init__(self, rank: int = 0):
        """
        初始化报告生成器。
        
        Args:
            rank: 本次分析的 rank 编号
        """
        self.rank = rank

    def generate_report(self,
                       candidates: List[Candidate],
                       slowdown_diagnoses: Dict[str, SlowdownDiagnosis],
                       dependency_diagnoses: Dict[str, DependencyDiagnosis],
                       resource_diagnoses: Dict[str, ResourceDiagnosis],
                       kernel_records: Optional[List[KernelRecord]] = None) -> Dict[str, Any]:
        """
        生成综合诊断报告。
        
        Args:
            candidates: 候选列表
            slowdown_diagnoses: slowdown 类型分类结果
            dependency_diagnoses: 依赖定位结果
            resource_diagnoses: 资源定位结果
            kernel_records: 原始 kernel 记录（用于参考）
            
        返回：报告 dict
        """
        # 分组报告
        reports_by_target = {}
        kernel_index = {k.kid: k for k in kernel_records or []}

        # 处理每个候选
        for candidate in candidates:
            target_id = candidate.target_id
            if target_id not in reports_by_target:
                reports_by_target[target_id] = self._create_target_report(
                    target_id, kernel_index.get(target_id)
                )

            # 关联诊断信息
            if target_id in slowdown_diagnoses:
                reports_by_target[target_id]["slowdown"] = slowdown_diagnoses[target_id].to_dict()

            if target_id in dependency_diagnoses:
                reports_by_target[target_id]["dependency"] = dependency_diagnoses[target_id].to_dict()

            if target_id in resource_diagnoses:
                reports_by_target[target_id]["resource"] = resource_diagnoses[target_id].to_dict()

            # 关联候选信息
            if "candidates" not in reports_by_target[target_id]:
                reports_by_target[target_id]["candidates"] = []
            reports_by_target[target_id]["candidates"].append(candidate.to_dict())

        # 生成摘要
        root_causes = self._build_root_cause_reports(reports_by_target, kernel_index)
        summary = self._generate_summary(reports_by_target, slowdown_diagnoses)

        # 汇总报告
        full_report = {
            "rank": self.rank,
            "summary": summary,
            "root_causes": root_causes,
            "targets": reports_by_target,
            "metadata": {
                "total_candidates": len(candidates),
                "total_slowdown_classified": len(slowdown_diagnoses),
                "total_dependency_localized": len(dependency_diagnoses),
                "total_resource_localized": len(resource_diagnoses),
            },
        }

        return full_report

    def _create_target_report(self,
                              target_id: str,
                              kernel: Optional[KernelRecord] = None) -> Dict[str, Any]:
        """创建单个 target 的报告框架。"""
        target_info = None
        if kernel is not None:
            target_info = {
                "kernel_name": kernel.kernel_name,
                "family": kernel.family,
                "tag": kernel.tag,
                "rank": kernel.rank,
                "stream": kernel.stream,
                "gpu_start_ns": kernel.gpu_start_ns,
                "gpu_end_ns": kernel.gpu_end_ns,
                "gpu_dur_ns": kernel.gpu_dur_ns,
                "cpu_enqueue_start_ns": kernel.cpu_enqueue_start_ns,
                "cpu_enqueue_end_ns": kernel.cpu_enqueue_end_ns,
                "work_type": kernel.work_type,
                "work_value": kernel.work_value,
                "progress": kernel.progress(),
            }
        return {
            "target_id": target_id,
            "target": target_info,
            "candidates": [],
            "slowdown": None,
            "dependency": None,
            "resource": None,
        }

    def _build_root_cause_reports(self,
                                  reports: Dict[str, Dict],
                                  kernel_index: Dict[str, KernelRecord]) -> List[Dict[str, Any]]:
        """生成 slide 15 对应的根因报告摘要。"""
        root_causes = []
        for target_id, report in reports.items():
            slowdown = report.get("slowdown") or {}
            dependency = report.get("dependency")
            resource = report.get("resource")
            target = report.get("target") or {}

            abnormal_type = slowdown.get("slowdown_type", "uncertain")
            evidence_chain = []
            if slowdown.get("evidence"):
                evidence_chain.append(slowdown["evidence"].get("decision", slowdown["evidence"].get("reason")))

            root = None
            slowdown_estimate_ns = None
            if dependency:
                blocker_id = dependency["blocker_id"]
                blocker_kernel = kernel_index.get(blocker_id)
                root = {
                    "type": "dependency_blocker",
                    "id": blocker_id,
                    "kind": dependency.get("blocker_kind"),
                    "name": (
                        blocker_kernel.kernel_name
                        if blocker_kernel is not None
                        else dependency.get("evidence", {}).get("blocker_name")
                    ),
                    "operator_name": (
                        blocker_kernel.operator_name
                        if blocker_kernel is not None
                        else dependency.get("evidence", {}).get("blocker_operator")
                    ),
                }
                slowdown_estimate_ns = dependency.get("delay_ns")
                evidence_chain.extend([
                    "blocker ends close to target start",
                    "counterfactual start is earlier",
                ])
                if dependency.get("overlap_loss_ns"):
                    evidence_chain.append("counterfactual overlap would recover")
            elif resource:
                culprit_id = resource["culprit_id"]
                culprit_kernel = kernel_index.get(culprit_id)
                root = {
                    "type": "resource_interference",
                    "id": culprit_id,
                    "kind": "kernel",
                    "name": culprit_kernel.kernel_name if culprit_kernel is not None else resource.get("evidence", {}).get("culprit_kernel"),
                    "operator_name": culprit_kernel.operator_name if culprit_kernel is not None else None,
                    "resource_hint": resource.get("resource_hint"),
                }
                evidence_chain.extend([
                    "target progress is lower than comparable kernels",
                    "culprit overlaps target execution",
                ])
                if resource.get("dose_response_score", 0) > 0:
                    evidence_chain.append("larger overlap correlates with lower progress")
                if resource.get("warp_block_verdict") == "broad_slowdown":
                    evidence_chain.append("warp/block distribution supports broad slowdown")
                elif resource.get("warp_block_verdict") == "internal_tail":
                    evidence_chain.append("warp/block distribution points to internal tail risk")

            if root is None and abnormal_type == "uncertain":
                continue

            root_causes.append({
                "target_id": target_id,
                "target_kernel": target.get("kernel_name"),
                "target_operator": target.get("operator_name"),
                "target_family": target.get("family"),
                "abnormal_type": abnormal_type,
                "root_cause": root,
                "slowdown_estimate_ns": slowdown_estimate_ns,
                "confidence": max(
                    slowdown.get("confidence", 0),
                    (dependency or {}).get("confidence", 0),
                    (resource or {}).get("confidence", 0),
                ),
                "evidence_chain": [item for item in evidence_chain if item],
            })

        root_causes.sort(key=lambda item: item["confidence"], reverse=True)
        return root_causes

    def _generate_summary(self,
                         reports: Dict[str, Dict],
                         slowdown_diagnoses: Dict[str, SlowdownDiagnosis]) -> Dict[str, Any]:
        """生成诊断摘要。"""
        from .records import SlowdownType

        # 统计各类型
        slowdown_counts = {
            "cpu_runtime_blocked": 0,
            "dependency_blocked": 0,
            "resource_slowed": 0,
            "uncertain": 0,
        }

        top_blockers = {}  # blocker_id -> 次数
        top_culprits = {}  # culprit_id -> 次数

        for diag in slowdown_diagnoses.values():
            slowdown_counts[diag.slowdown_type.value] += 1

        for target_id, report in reports.items():
            if report["dependency"]:
                blocker_id = report["dependency"]["blocker_id"]
                top_blockers[blocker_id] = top_blockers.get(blocker_id, 0) + 1

            if report["resource"]:
                culprit_id = report["resource"]["culprit_id"]
                top_culprits[culprit_id] = top_culprits.get(culprit_id, 0) + 1

        # 排序
        top_blockers = sorted(top_blockers.items(), key=lambda x: x[1], reverse=True)
        top_culprits = sorted(top_culprits.items(), key=lambda x: x[1], reverse=True)

        return {
            "slowdown_type_distribution": slowdown_counts,
            "top_blockers": [{"kernel_id": kid, "count": cnt} for kid, cnt in top_blockers[:5]],
            "top_culprits": [{"kernel_id": kid, "count": cnt} for kid, cnt in top_culprits[:5]],
            "target_count": len(reports),
        }

    def to_json(self, report: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        将报告转换为 JSON。
        
        Args:
            report: 报告 dict
            output_path: 如果指定，写入文件
            
        返回：JSON 字符串
        """
        json_str = json.dumps(report, indent=2, ensure_ascii=False)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)

        return json_str

    def generate_human_readable_report(self, report: Dict[str, Any]) -> str:
        """生成人可读格式的报告。"""
        lines = []

        lines.append(f"=== 诊断报告（Rank {report['rank']}) ===")
        lines.append("")

        # 摘要部分
        summary = report["summary"]
        lines.append("【摘要】")
        lines.append(f"  总候选数: {report['metadata']['total_candidates']}")
        lines.append(f"  已分类: {report['metadata']['total_slowdown_classified']}")
        lines.append(f"  依赖定位: {report['metadata']['total_dependency_localized']}")
        lines.append(f"  资源定位: {report['metadata']['total_resource_localized']}")
        lines.append("")

        lines.append("【性能下降类型分布】")
        for slowdown_type, count in summary["slowdown_type_distribution"].items():
            lines.append(f"  {slowdown_type}: {count}")
        lines.append("")

        if summary["top_blockers"]:
            lines.append("【最常见的阻塞 Kernel】")
            for item in summary["top_blockers"]:
                lines.append(f"  {item['kernel_id']}: {item['count']} 次")
            lines.append("")

        if summary["top_culprits"]:
            lines.append("【最常见的干扰源】")
            for item in summary["top_culprits"]:
                lines.append(f"  {item['kernel_id']}: {item['count']} 次")
            lines.append("")

        if report.get("root_causes"):
            lines.append("【根因摘要】")
            for item in report["root_causes"][:10]:
                root = item.get("root_cause") or {}
                lines.append(
                    f"  Target {item['target_id']} ({item.get('target_kernel')}): "
                    f"{item['abnormal_type']} -> {root.get('name', 'unknown')} "
                    f"(confidence={item['confidence']:.2%})"
                )
            lines.append("")

        # 详细目标报告
        lines.append("【详细诊断】")
        for target_id, target_report in report["targets"].items():
            target_info = target_report.get("target") or {}
            title = target_info.get("kernel_name") or target_id
            lines.append(f"\nTarget Kernel: {target_id} ({title})")
            lines.append("-" * 60)

            if target_report["slowdown"]:
                sd = target_report["slowdown"]
                lines.append(f"  性能下降类型: {sd['slowdown_type']}")
                lines.append(f"  置信度: {sd['confidence']:.2%}")
                if sd.get("evidence"):
                    lines.append(f"  证据: {sd['evidence']}")

            if target_report["dependency"]:
                dd = target_report["dependency"]
                lines.append(f"  依赖阻塞: {dd['blocker_id']}")
                lines.append(f"  延迟: {dd['delay_ns']} ns")
                lines.append(f"  置信度: {dd['confidence']:.2%}")

            if target_report["resource"]:
                rd = target_report["resource"]
                lines.append(f"  资源干扰: {rd['culprit_id']}")
                lines.append(f"  Overlap: {rd['overlap_ratio']:.2%}")
                lines.append(f"  资源类型: {rd['resource_hint']}")
                lines.append(f"  置信度: {rd['confidence']:.2%}")

        return "\n".join(lines)

    def to_html(self, report: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """生成 HTML 格式的报告。"""
        html_parts = []

        html_parts.append("""
        <html>
        <head>
            <meta charset="utf-8">
            <title>诊断报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background-color: #f0f0f0; padding: 10px; margin: 10px 0; }
                .target { border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
                .diagnosis { margin-left: 20px; padding: 5px; }
                .high-confidence { background-color: #ffcccc; }
                .medium-confidence { background-color: #ffffcc; }
                .low-confidence { background-color: #ccffcc; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            </style>
        </head>
        <body>
        """)

        html_parts.append(f"<h1>诊断报告（Rank {report['rank']}）</h1>")

        # 摘要
        summary = report["summary"]
        html_parts.append("<div class='summary'>")
        html_parts.append(f"<h2>摘要</h2>")
        html_parts.append(f"<p>总候选数: {report['metadata']['total_candidates']}</p>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>性能下降类型</th><th>数量</th></tr>")
        for slowdown_type, count in summary["slowdown_type_distribution"].items():
            html_parts.append(f"<tr><td>{slowdown_type}</td><td>{count}</td></tr>")
        html_parts.append("</table>")
        html_parts.append("</div>")

        # 详细诊断
        html_parts.append("<h2>详细诊断</h2>")
        for target_id, target_report in report["targets"].items():
            conf_class = self._get_confidence_class(target_report)
            html_parts.append(f"<div class='target {conf_class}'>")
            html_parts.append(f"<h3>Target: {target_id}</h3>")

            if target_report["slowdown"]:
                sd = target_report["slowdown"]
                html_parts.append(f"<div class='diagnosis'>")
                html_parts.append(f"<strong>性能下降类型:</strong> {sd['slowdown_type']}")
                html_parts.append(f"<br/><strong>置信度:</strong> {sd['confidence']:.1%}")
                html_parts.append(f"</div>")

            if target_report["dependency"]:
                dd = target_report["dependency"]
                html_parts.append(f"<div class='diagnosis'>")
                html_parts.append(f"<strong>依赖阻塞:</strong> {dd['blocker_id']}")
                html_parts.append(f"<br/><strong>延迟:</strong> {dd['delay_ns']} ns")
                html_parts.append(f"</div>")

            if target_report["resource"]:
                rd = target_report["resource"]
                html_parts.append(f"<div class='diagnosis'>")
                html_parts.append(f"<strong>资源干扰:</strong> {rd['culprit_id']}")
                html_parts.append(f"<br/><strong>Overlap:</strong> {rd['overlap_ratio']:.1%}")
                html_parts.append(f"</div>")

            html_parts.append("</div>")

        html_parts.append("</body></html>")

        html_str = "\n".join(html_parts)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_str)

        return html_str

    def _get_confidence_class(self, target_report: Dict) -> str:
        """根据置信度返回 CSS class。"""
        max_conf = 0.0

        if target_report["slowdown"]:
            max_conf = max(max_conf, target_report["slowdown"].get("confidence", 0))
        if target_report["dependency"]:
            max_conf = max(max_conf, target_report["dependency"].get("confidence", 0))
        if target_report["resource"]:
            max_conf = max(max_conf, target_report["resource"].get("confidence", 0))

        if max_conf > 0.75:
            return "high-confidence"
        elif max_conf > 0.5:
            return "medium-confidence"
        else:
            return "low-confidence"
