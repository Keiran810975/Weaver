"""
依赖阻塞定位模块。

对于被识别为 dependency_blocked 的 target，
找出具体是哪个 kernel/sync 阻塞了它，并计算延迟贡献。
"""

from typing import Dict, List, Optional, Tuple, Union

from .records import KernelRecord, SyncRecord, DependencyDiagnosis
from .sketch_match import (
    dependency_uses_same_stream,
    expected_dependencies_for_target,
    find_matching_predecessors,
)
from ..sketch import ExecutionSketch
from ..sketch.rules import get_default_overlap_expectations


Blocker = Union[KernelRecord, SyncRecord]


class DependencyLocalizer:
    """依赖阻塞定位。"""

    def __init__(self, 
                 close_threshold_ns: int = 100_000,  # 100 us
                 min_delay_for_confirmation_ns: int = 1_000):  # 1 us
        """
        初始化定位器。
        
        Args:
            close_threshold_ns: 判断"紧贴"的阈值
            min_delay_for_confirmation_ns: 确认延迟的最小值
        """
        self.close_threshold_ns = close_threshold_ns
        self.min_delay_for_confirmation_ns = min_delay_for_confirmation_ns

    def localize(self,
                 target: KernelRecord,
                 all_kernels: List[KernelRecord],
                 sketch: Optional[ExecutionSketch] = None,
                 syncs: Optional[List[SyncRecord]] = None) -> Optional[DependencyDiagnosis]:
        """
        定位阻塞 target 的 blocker kernel。
        
        Args:
            target: 被阻塞的 kernel
            all_kernels: 所有 kernel 记录
            sketch: ExecutionSketch（用于查询预期依赖）
            
        返回：DependencyDiagnosis，或 None（如果找不到）
        """
        syncs = syncs or []

        # Step 1: 找实际前驱和紧贴 sync/event wait
        actual_pred = self._find_actual_predecessor(target, all_kernels)
        sync_blocker = self._find_blocking_sync(target, syncs)
        if actual_pred is None and sync_blocker is None:
            return None

        # Step 2: 找预期前驱
        expected_preds = self._find_expected_predecessors(target, all_kernels, sketch)

        # Step 3: 选择最像 blocker 的实际事件。
        blocker_kind, blocker = self._choose_blocker(target, actual_pred, sync_blocker, expected_preds)
        if blocker is None:
            return None

        # Step 4: 计算延迟贡献
        delay_ns = self._compute_delay(target, blocker_kind, blocker)

        # Step 5: 计算 overlap recovery
        overlap_loss_ns = self._compute_overlap_loss(target, blocker_kind, blocker, all_kernels, sketch)

        # Step 6: 计算反事实启动时间
        counterfactual_start = self._compute_counterfactual_start(
            target, blocker_kind, blocker, expected_preds, all_kernels
        )

        confidence = self._compute_confidence(target, blocker_kind, blocker, delay_ns)
        blocker_id = self._blocker_id(blocker_kind, blocker)

        return DependencyDiagnosis(
            target_id=target.kid,
            blocker_id=blocker_id,
            blocker_kind=blocker_kind,
            delay_ns=delay_ns,
            overlap_loss_ns=overlap_loss_ns,
            counterfactual_start_ns=counterfactual_start,
            confidence=confidence,
            evidence={
                "blocker_name": self._blocker_name(blocker_kind, blocker),
                "blocker_operator": self._blocker_operator(blocker_kind, blocker),
                "blocker_family": self._blocker_family(blocker_kind, blocker),
                "actual_gap_ns": self._compute_gap(target, blocker_kind, blocker),
                "expected_predecessors": [p.kid for p in expected_preds],
                "is_unexpected": self._is_unexpected(blocker_kind, blocker, expected_preds),
                "counterfactual_explanation": (
                    "target could start earlier if the blocker were removed from the local critical path"
                ),
            },
        )

    def _find_actual_predecessor(self, 
                                 target: KernelRecord,
                                 all_kernels: List[KernelRecord]) -> Optional[KernelRecord]:
        """找 target 的实际前驱（same stream）。"""
        if target.stream is None:
            return None

        # 在同 stream 中找时间最接近的前驱
        same_stream = [k for k in all_kernels if k.stream == target.stream and k.kid != target.kid]

        target_time = target.gpu_start_ns or target.cpu_enqueue_start_ns
        if target_time is None:
            return None

        # 找最晚的前驱
        candidates = []
        for k in same_stream:
            k_end = k.gpu_end_ns or k.cpu_enqueue_end_ns
            if k_end is not None and k_end <= target_time:
                candidates.append((k_end, k))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]

    def _find_expected_predecessors(self,
                                    target: KernelRecord,
                                    all_kernels: List[KernelRecord],
                                    sketch: Optional[ExecutionSketch]) -> List[KernelRecord]:
        """找预期的前驱。"""
        expected = []

        # 手工 expected_dependencies 优先级最高：这是 PPT 中 Pred_expected(target)
        # 的显式来源，不再依赖系统自动生成草图。
        manual_deps = expected_dependencies_for_target(sketch, target)
        for dep in manual_deps:
            expected.extend(find_matching_predecessors(
                target,
                all_kernels,
                dep.predecessors,
                same_stream=dependency_uses_same_stream(dep),
            ))
        if expected:
            return self._dedupe_kernels(expected)

        # 从 sketch 的通用依赖规则查询；没有可用 sketch 时，使用 same-stream 中
        # blocker 前最近的非 extra kernel 作为弱预期前驱。
        if sketch:
            for rule in sketch.dependency_rules:
                if rule.type.value == "hard":
                    # hard rule：same-stream order
                    if rule.rule_id == "same_stream_order" and target.stream:
                        pred = self._find_actual_predecessor(target, all_kernels)
                        if pred:
                            if self._looks_extra(pred):
                                non_extra = self._find_previous_non_extra_predecessor(target, all_kernels)
                                if non_extra:
                                    expected.append(non_extra)
                            else:
                                expected.append(pred)

        if not expected and target.stream:
            pred = self._find_previous_non_extra_predecessor(target, all_kernels)
            if pred:
                expected.append(pred)

        return self._dedupe_kernels(expected)

    def _find_blocking_sync(self,
                            target: KernelRecord,
                            syncs: List[SyncRecord]) -> Optional[SyncRecord]:
        """找紧贴 target 的同步事件。"""
        target_start = target.cpu_enqueue_start_ns or target.gpu_start_ns
        if target_start is None:
            return None
        candidates = []
        for sync in syncs:
            if target.rank is not None and sync.rank is not None and sync.rank != target.rank:
                continue
            if sync.pid != target.pid:
                continue
            sync_end = sync.ts_end_ns or sync.ts_start_ns
            gap = target_start - sync_end
            if 0 <= gap <= self.close_threshold_ns:
                candidates.append((gap, sync))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _choose_blocker(self,
                        target: KernelRecord,
                        actual_pred: Optional[KernelRecord],
                        sync_blocker: Optional[SyncRecord],
                        expected_preds: List[KernelRecord]) -> Tuple[str, Optional[Blocker]]:
        """在前驱 kernel 和 sync 中选择最可能的 blocker。"""
        scored: List[Tuple[float, str, Blocker]] = []

        if actual_pred is not None:
            gap = self._compute_gap(target, "kernel", actual_pred)
            score = 0.4
            if gap <= self.close_threshold_ns:
                score += 0.3
            if self._is_unexpected("kernel", actual_pred, expected_preds):
                score += 0.2
            if self._looks_extra(actual_pred):
                score += 0.1
            scored.append((score, "kernel", actual_pred))

        if sync_blocker is not None:
            gap = self._compute_gap(target, "sync", sync_blocker)
            score = 0.55
            if gap <= self.close_threshold_ns:
                score += 0.25
            if (sync_blocker.duration_ns or 0) > self.min_delay_for_confirmation_ns:
                score += 0.15
            scored.append((score, "sync", sync_blocker))

        if not scored:
            return "kernel", None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1], scored[0][2]

    def _find_closest_blocker(self,
                             target: KernelRecord,
                             candidates: List[KernelRecord]) -> Optional[KernelRecord]:
        """在候选中找"最紧贴"的阻塞者。"""
        target_start = target.gpu_start_ns or target.cpu_enqueue_start_ns
        if target_start is None:
            return None

        closest = None
        min_gap = float('inf')

        for cand in candidates:
            cand_end = cand.gpu_end_ns or cand.cpu_enqueue_end_ns
            if cand_end is None:
                continue

            gap = target_start - cand_end
            if 0 <= gap < min_gap and gap < self.close_threshold_ns:
                min_gap = gap
                closest = cand

        return closest

    def _compute_gap(self, target: KernelRecord, blocker_kind: str, blocker: Blocker) -> int:
        """计算 blocker end 到 target start 的间隔。"""
        blocker_end = self._blocker_end(blocker_kind, blocker)
        target_start = (
            target.cpu_enqueue_start_ns
            if blocker_kind == "sync"
            else (target.gpu_start_ns or target.cpu_enqueue_start_ns)
        )

        if blocker_end is None or target_start is None:
            return 0

        gap = target_start - blocker_end
        return max(0, gap)

    def _compute_delay(self, target: KernelRecord, blocker_kind: str, blocker: Blocker) -> int:
        """计算因 blocker 而产生的延迟。"""
        gap = self._compute_gap(target, blocker_kind, blocker)

        # 如果 gap 很小，认为 blocker 直接阻塞了 target
        if 0 <= gap <= self.close_threshold_ns:
            # 延迟 ≈ blocker 的执行时间 + gap
            blocker_dur = self._blocker_duration(blocker_kind, blocker)
            return blocker_dur + gap

        return 0

    def _compute_overlap_loss(self,
                             target: KernelRecord,
                             blocker_kind: str,
                             blocker: Blocker,
                             all_kernels: List[KernelRecord],
                             sketch: Optional[ExecutionSketch]) -> Optional[int]:
        """计算因 blocker 而丧失的 overlap。"""
        overlap_expectations = (
            sketch.overlap_expectations
            if sketch is not None
            else get_default_overlap_expectations()
        )

        # 找与 target 应该 overlap 的 kernel
        for oe in overlap_expectations:
            if oe.expected.value != "may_overlap":
                continue

            # 检查 target 是否匹配
            target_matches = (target.family == oe.left_family or target.family == oe.right_family)
            if not target_matches:
                continue

            # 找对侧的 kernel
            other_family = oe.right_family if target.family == oe.left_family else oe.left_family
            others = [k for k in all_kernels if k.family == other_family]

            # 计算实际 overlap 和反事实 overlap
            if not others:
                continue

            for other in others:
                actual_overlap = self._compute_interval_overlap(target, other)

                # 计算反事实：如果没有 blocker
                counterfactual_start = self._compute_counterfactual_start(
                    target, blocker_kind, blocker, [], all_kernels
                )
                if counterfactual_start:
                    counterfactual_end = counterfactual_start + (target.gpu_dur_ns or 0)
                    cf_overlap = self._compute_interval_overlap_explicit(
                        (counterfactual_start, counterfactual_end),
                        other
                    )
                else:
                    cf_overlap = actual_overlap

                overlap_loss = cf_overlap - actual_overlap if cf_overlap > actual_overlap else 0
                if overlap_loss > self.min_delay_for_confirmation_ns:
                    return overlap_loss

        return None

    def _compute_counterfactual_start(self,
                                     target: KernelRecord,
                                     blocker_kind: str,
                                     blocker: Blocker,
                                     expected_preds: List[KernelRecord],
                                     all_kernels: List[KernelRecord]) -> Optional[int]:
        """
        计算反事实启动时间。
        
        如果没有 blocker，target 应该在何时启动？
        """
        target_start = (
            target.cpu_enqueue_start_ns
            if blocker_kind == "sync"
            else (target.gpu_start_ns or target.cpu_enqueue_start_ns)
        )
        if target_start is None:
            return None

        # 反事实时间 = max(CPU enqueue end, 预期前驱 end)，再移除当前 blocker 的本地贡献。
        cf_time = target.cpu_enqueue_end_ns or target.cpu_enqueue_start_ns or 0

        for pred in expected_preds:
            pred_end = pred.gpu_end_ns or pred.cpu_enqueue_end_ns
            if pred_end:
                cf_time = max(cf_time, pred_end)

        # 移除 blocker 的贡献
        delay = self._compute_delay(target, blocker_kind, blocker)
        cf_time = max(cf_time, target_start - delay)

        return cf_time if cf_time != 0 else None

    def _compute_interval_overlap(self, k1: KernelRecord, k2: KernelRecord) -> int:
        """计算两个 kernel 的 overlap 时间。"""
        if k1.gpu_start_ns is None or k1.gpu_end_ns is None:
            return 0
        if k2.gpu_start_ns is None or k2.gpu_end_ns is None:
            return 0

        overlap_start = max(k1.gpu_start_ns, k2.gpu_start_ns)
        overlap_end = min(k1.gpu_end_ns, k2.gpu_end_ns)

        return max(0, overlap_end - overlap_start)

    def _compute_interval_overlap_explicit(self, 
                                          interval1: Tuple[int, int],
                                          k2: KernelRecord) -> int:
        """计算显式区间与 kernel 的 overlap。"""
        if k2.gpu_start_ns is None or k2.gpu_end_ns is None:
            return 0

        overlap_start = max(interval1[0], k2.gpu_start_ns)
        overlap_end = min(interval1[1], k2.gpu_end_ns)

        return max(0, overlap_end - overlap_start)

    def _compute_confidence(self, 
                           target: KernelRecord,
                           blocker_kind: str,
                           blocker: Blocker,
                           delay_ns: int) -> float:
        """计算诊断的置信度。"""
        confidence = 0.5

        # 如果 gap 很小，置信度提高
        gap = self._compute_gap(target, blocker_kind, blocker)
        if 0 <= gap <= 50_000:  # 50 us
            confidence += 0.3

        # 如果延迟很大，置信度提高
        if delay_ns > 1_000_000:  # 1 ms
            confidence += 0.1

        if blocker_kind == "sync":
            confidence += 0.05
        elif self._looks_extra(blocker):  # type: ignore[arg-type]
            confidence += 0.05

        return min(0.95, confidence)

    def _find_previous_non_extra_predecessor(self,
                                             target: KernelRecord,
                                             all_kernels: List[KernelRecord]) -> Optional[KernelRecord]:
        if target.stream is None:
            return None
        target_time = target.gpu_start_ns or target.cpu_enqueue_start_ns
        if target_time is None:
            return None
        candidates = []
        for kernel in all_kernels:
            if kernel.kid == target.kid or kernel.stream != target.stream:
                continue
            end = kernel.gpu_end_ns or kernel.cpu_enqueue_end_ns
            if end is None or end > target_time:
                continue
            if not self._looks_extra(kernel):
                candidates.append((end, kernel))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _looks_extra(self, kernel: KernelRecord) -> bool:
        if kernel.family in {"UNKNOWN", "MEMCPY", "ELEMENTWISE"}:
            return True
        name = kernel.kernel_name.lower()
        op = (kernel.operator_name or "").lower()
        return any(token in name or token in op for token in (
            "copy", "cast", "contiguous", "layout", "memset", "fill", "zero"
        ))

    def _dedupe_kernels(self, kernels: List[KernelRecord]) -> List[KernelRecord]:
        seen = set()
        result = []
        for kernel in kernels:
            if kernel.kid in seen:
                continue
            seen.add(kernel.kid)
            result.append(kernel)
        return result

    def _is_unexpected(self,
                       blocker_kind: str,
                       blocker: Blocker,
                       expected_preds: List[KernelRecord]) -> bool:
        if blocker_kind == "sync":
            return True
        return blocker.kid not in [p.kid for p in expected_preds] or self._looks_extra(blocker)

    def _blocker_id(self, blocker_kind: str, blocker: Blocker) -> str:
        return blocker.kid if blocker_kind == "kernel" else f"sync:{blocker.sid}"

    def _blocker_name(self, blocker_kind: str, blocker: Blocker) -> str:
        return blocker.kernel_name if blocker_kind == "kernel" else blocker.kind.value

    def _blocker_operator(self, blocker_kind: str, blocker: Blocker) -> Optional[str]:
        if blocker_kind == "kernel":
            return blocker.operator_name
        return blocker.payload.get("operator_name") if blocker.payload else None

    def _blocker_family(self, blocker_kind: str, blocker: Blocker) -> str:
        return blocker.family if blocker_kind == "kernel" else "SYNC"

    def _blocker_end(self, blocker_kind: str, blocker: Blocker) -> Optional[int]:
        if blocker_kind == "kernel":
            return blocker.gpu_end_ns or blocker.cpu_enqueue_end_ns
        return blocker.ts_end_ns or blocker.ts_start_ns

    def _blocker_duration(self, blocker_kind: str, blocker: Blocker) -> int:
        if blocker_kind == "kernel":
            return blocker.gpu_dur_ns or blocker.cpu_enqueue_dur_ns or 0
        return blocker.duration_ns or 0

    def localize_batch(self,
                      targets: List[KernelRecord],
                      all_kernels: List[KernelRecord],
                      sketch: Optional[ExecutionSketch] = None,
                      syncs: Optional[List[SyncRecord]] = None) -> Dict[str, DependencyDiagnosis]:
        """批量定位。"""
        result = {}
        for target in targets:
            diagnosis = self.localize(target, all_kernels, sketch, syncs or [])
            if diagnosis:
                result[target.kid] = diagnosis

        return result
