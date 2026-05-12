"""
命令行接口。

完整的诊断流程：
  1. 加载 aligned timeline
  2. 规范化为 kernel/sync records
  3. 加载 execution sketch
  4. 发现候选
  5. 分类 slowdown 类型
  6. 定位依赖阻塞
  7. 定位资源干扰
  8. 生成报告
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .normalize import TimelineNormalizer
from .candidates import CandidateDiscovery
from .timing import TimingAnalyzer
from .dependency import DependencyLocalizer
from .resource import ResourceLocalizer
from .report import DiagnosisReporter
from .records import SlowdownDiagnosis, SlowdownType


def load_execution_sketch(sketch_path: str):
    """加载 execution sketch。"""
    from ..sketch.schema import ExecutionSketch
    with open(sketch_path, 'r') as f:
        data = json.load(f)
    return ExecutionSketch.from_dict(data)


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="Weaver 诊断工具 - 从 aligned timeline 进行性能诊断"
    )

    parser.add_argument(
        "--timeline",
        type=str,
        required=True,
        help="输入 aligned_timeline_rank*.ndjson 文件路径",
    )

    parser.add_argument(
        "--sketch",
        type=str,
        required=False,
        help="execution_sketch.json 文件路径（可选）",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank 编号（默认 0）",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="输出报告路径（JSON）",
    )

    parser.add_argument(
        "--output-html",
        type=str,
        required=False,
        help="输出 HTML 报告路径",
    )

    parser.add_argument(
        "--output-text",
        type=str,
        required=False,
        help="输出文本报告路径",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出",
    )

    args = parser.parse_args()

    try:
        # Step 1: 规范化
        if args.verbose:
            print("Step 1: 加载并规范化 timeline...", file=sys.stderr)

        normalizer = TimelineNormalizer(args.timeline)
        kernels, operators, syncs = normalizer.normalize()

        if args.verbose:
            print(f"  加载了 {len(kernels)} 个 kernel 记录", file=sys.stderr)
            print(f"  加载了 {len(operators)} 个 operator 记录", file=sys.stderr)
            print(f"  加载了 {len(syncs)} 个 sync 记录", file=sys.stderr)

        # Step 2: 加载 sketch
        sketch = None
        if args.sketch:
            if args.verbose:
                print(f"Step 2: 加载 sketch: {args.sketch}", file=sys.stderr)
            sketch = load_execution_sketch(args.sketch)
        else:
            if args.verbose:
                print("Step 2: 未指定 sketch，使用空模型", file=sys.stderr)

        # Step 3: 发现候选
        if args.verbose:
            print("Step 3: 发现候选...", file=sys.stderr)

        discoverer = CandidateDiscovery(sketch)
        candidates = discoverer.discover(kernels, syncs)

        if args.verbose:
            print(f"  发现了 {len(candidates)} 个候选", file=sys.stderr)

        # Step 4: 分类 slowdown 类型
        if args.verbose:
            print("Step 4: 分类性能下降类型...", file=sys.stderr)

        candidate_ids = {c.target_id for c in candidates}
        candidate_targets = [k for k in kernels if k.kid in candidate_ids]
        if not candidate_targets and args.verbose:
            print("  没有候选 target，跳过定位阶段", file=sys.stderr)

        analyzer = TimingAnalyzer()
        slowdown_diagnoses = analyzer.classify_batch(candidate_targets, kernels, syncs)
        _promote_structural_dependency_candidates(candidates, slowdown_diagnoses)

        if args.verbose:
            print(f"  分类了 {len(slowdown_diagnoses)} 个 kernel", file=sys.stderr)

        # Step 5: 定位依赖阻塞
        if args.verbose:
            print("Step 5: 定位依赖阻塞...", file=sys.stderr)

        dep_localizer = DependencyLocalizer()
        dependency_diagnoses = {}
        for kid, diag in slowdown_diagnoses.items():
            if diag.slowdown_type.value in ("cpu_runtime_blocked", "dependency_blocked"):
                target = next((k for k in kernels if k.kid == kid), None)
                if target:
                    dep_diag = dep_localizer.localize(target, kernels, sketch, syncs)
                    if dep_diag:
                        dependency_diagnoses[kid] = dep_diag

        if args.verbose:
            print(f"  定位了 {len(dependency_diagnoses)} 个依赖阻塞", file=sys.stderr)

        # Step 6: 定位资源干扰
        if args.verbose:
            print("Step 6: 定位资源干扰...", file=sys.stderr)

        res_localizer = ResourceLocalizer()
        resource_diagnoses = {}
        for kid, diag in slowdown_diagnoses.items():
            if diag.slowdown_type.value == "resource_slowed":
                target = next((k for k in kernels if k.kid == kid), None)
                if target:
                    res_diag = res_localizer.localize(target, kernels, sketch)
                    if res_diag:
                        resource_diagnoses[kid] = res_diag

        if args.verbose:
            print(f"  定位了 {len(resource_diagnoses)} 个资源干扰", file=sys.stderr)

        # Step 7: 生成报告
        if args.verbose:
            print("Step 7: 生成报告...", file=sys.stderr)

        reporter = DiagnosisReporter(rank=args.rank)
        report = reporter.generate_report(
            candidates,
            slowdown_diagnoses,
            dependency_diagnoses,
            resource_diagnoses,
            kernels,
        )

        # 输出结果
        if args.output:
            reporter.to_json(report, args.output)
            if args.verbose:
                print(f"JSON 报告已保存到: {args.output}", file=sys.stderr)

        if args.output_html:
            reporter.to_html(report, args.output_html)
            if args.verbose:
                print(f"HTML 报告已保存到: {args.output_html}", file=sys.stderr)

        if args.output_text:
            text_report = reporter.generate_human_readable_report(report)
            Path(args.output_text).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_text, 'w', encoding='utf-8') as f:
                f.write(text_report)
            if args.verbose:
                print(f"文本报告已保存到: {args.output_text}", file=sys.stderr)

        # 打印摘要到 stdout
        summary = report["summary"]
        print("=== 诊断摘要 ===")
        print(f"总候选数: {report['metadata']['total_candidates']}")
        print(f"已分类: {report['metadata']['total_slowdown_classified']}")
        print("\n【性能下降类型分布】")
        for slowdown_type, count in summary["slowdown_type_distribution"].items():
            print(f"  {slowdown_type}: {count}")

        if args.verbose:
            print("\n诊断完成！", file=sys.stderr)

        return 0

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def _promote_structural_dependency_candidates(candidates, slowdown_diagnoses):
    """Manual sketches can prove dependency blocking before timing outlier is visible."""
    dependency_reasons = {
        "unexpected_predecessor_against_manual_sketch",
        "missing_expected_predecessor",
        "extra_predecessor",
        "sync_preceded_kernel",
    }
    for candidate in candidates:
        if candidate.reason not in dependency_reasons:
            continue
        current = slowdown_diagnoses.get(candidate.target_id)
        if current is not None and current.slowdown_type != SlowdownType.UNCERTAIN:
            continue
        slowdown_diagnoses[candidate.target_id] = SlowdownDiagnosis(
            target_id=candidate.target_id,
            slowdown_type=SlowdownType.DEPENDENCY_BLOCKED,
            confidence=max(0.65, candidate.confidence),
            evidence={
                "decision": "manual_sketch_dependency_deviation",
                "reason": candidate.reason,
                "candidate_evidence": candidate.evidence,
            },
        )


if __name__ == "__main__":
    sys.exit(main())
