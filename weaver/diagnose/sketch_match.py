"""Helpers for matching runtime records against a manual execution sketch."""

import re
from typing import Any, Dict, Iterable, List, Optional

from .records import KernelRecord


def selector_matches_kernel(selector: Dict[str, Any], kernel: KernelRecord) -> bool:
    """Return whether a kernel matches a manual sketch selector."""
    if not selector:
        return True

    if not _match_value(selector.get("family"), kernel.family):
        return False
    if not _match_value(selector.get("tag"), kernel.tag):
        return False
    if not _match_regex(selector.get("tag_regex"), kernel.tag):
        return False
    if not _match_regex(selector.get("kernel_name_regex") or selector.get("name_regex"), kernel.kernel_name):
        return False
    if not _match_regex(selector.get("operator_regex"), kernel.operator_name or ""):
        return False
    if not _match_value(selector.get("stream"), kernel.stream):
        return False
    if not _match_value(selector.get("rank"), kernel.rank):
        return False

    return True


def find_matching_predecessors(
    target: KernelRecord,
    records: List[KernelRecord],
    predecessor_selectors: Iterable[Dict[str, Any]],
    same_stream: bool = False,
) -> List[KernelRecord]:
    """Find earlier kernels matching any predecessor selector."""
    target_time = target.gpu_start_ns or target.cpu_enqueue_start_ns
    if target_time is None:
        return []

    matched = []
    for kernel in records:
        if kernel.kid == target.kid:
            continue
        if same_stream and kernel.stream != target.stream:
            continue
        end = kernel.gpu_end_ns or kernel.cpu_enqueue_end_ns
        if end is None or end > target_time:
            continue
        if any(selector_matches_kernel(selector, kernel) for selector in predecessor_selectors):
            matched.append(kernel)

    matched.sort(key=lambda item: item.gpu_end_ns or item.cpu_enqueue_end_ns or 0, reverse=True)
    return matched


def expected_dependencies_for_target(sketch: Any, target: KernelRecord) -> List[Any]:
    """Return manual expected dependency entries that match target."""
    if sketch is None:
        return []
    deps = getattr(sketch, "expected_dependencies", []) or []
    return [dep for dep in deps if selector_matches_kernel(dep.target, target)]


def dependency_uses_same_stream(dep: Any) -> bool:
    relation = getattr(dep, "relation", "")
    return relation in {"same_stream_predecessor", "immediate_predecessor"}


def _match_value(expected: Any, actual: Any) -> bool:
    if expected is None:
        return True
    if isinstance(expected, list):
        return actual in expected
    return expected == actual


def _match_regex(pattern: Optional[str], text: str) -> bool:
    if not pattern:
        return True
    return re.search(pattern, text or "", re.IGNORECASE) is not None
