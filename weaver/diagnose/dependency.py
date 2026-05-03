"""
依赖阻塞定位模块。

对于被识别为 dependency_blocked 的 target，
找出具体是哪个 kernel/sync 阻塞了它，并计算延迟贡献。
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .records import KernelRecord, SyncRecord, DependencyDiagnosis
from ..sketch import ExecutionSketch


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
                 sketch: Optional[ExecutionSketch] = None) -> Optional[DependencyDiagnosis]:
        """
        定位阻塞 target 的 blocker kernel。
        
        Args:
            target: 被阻塞的 kernel
            all_kernels: 所有 kernel 记录
            sketch: ExecutionSketch（用于查询预期依赖）
            
        返回：DependencyDiagnosis，或 None（如果找不到）
        """
        # Step 1: 找实际前驱
        actual_pred = self._find_actual_predecessor(target, all_kernels)
        if actual_pred is None:
            return None

        # Step 2: 找预期前驱
        expected_preds = self._find_expected_predecessors(target, all_kernels, sketch)

        # Step 3: 找意外的前驱
        unexpected_preds = []
        if actual_pred.kid not in [p.kid for p in expected_preds]:
            unexpected_preds.append(actual_pred)

        # Step 4: 判断是否"紧贴"
        blocker = None
        if unexpected_preds:
            blocker = self._find_closest_blocker(target, unexpected_preds)

        if blocker is None:
            blocker = actual_pred

        # Step 5: 计算延迟贡献
        delay_ns = self._compute_delay(target, blocker)

        # Step 6: 计算 overlap recovery
        overlap_loss_ns = self._compute_overlap_loss(target, blocker, all_kernels, sketch)

        # Step 7: 计算反事实启动时间
        counterfactual_start = self._compute_counterfactual_start(
            target, blocker, expected_preds, all_kernels
        )

        confidence = self._compute_confidence(target, blocker, delay_ns)

        return DependencyDiagnosis(
            target_id=target.kid,
            blocker_id=blocker.kid,
            delay_ns=delay_ns,
            overlap_loss_ns=overlap_loss_ns,
            counterfactual_start_ns=counterfactual_start,
            confidence=confidence,
            evidence={
                "blocker_kernel": blocker.kernel_name,
                "blocker_family": blocker.family,
                "actual_gap_ns": self._compute_gap(target, blocker),
                "is_unexpected": blocker.kid not in [p.kid for p in expected_preds],
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
            if k_end is not None and k_end < target_time:
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

        # 从 sketch 的依赖规则查询
        if sketch:
            for rule in sketch.dependency_rules:
                if rule.type.value == "hard":
                    # hard rule：same-stream order
                    if rule.rule_id == "same_stream_order" and target.stream:
                        pred = self._find_actual_predecessor(target, all_kernels)
                        if pred:
                            expected.append(pred)

        # 如果没有从 sketch 得到预期，返回空
        return expected

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

    def _compute_gap(self, target: KernelRecord, blocker: KernelRecord) -> int:
        """计算 blocker end 到 target start 的间隔。"""
        blocker_end = blocker.gpu_end_ns or blocker.cpu_enqueue_end_ns
        target_start = target.gpu_start_ns or target.cpu_enqueue_start_ns

        if blocker_end is None or target_start is None:
            return 0

        gap = target_start - blocker_end
        return max(0, gap)

    def _compute_delay(self, target: KernelRecord, blocker: KernelRecord) -> int:
        """计算因 blocker 而产生的延迟。"""
        gap = self._compute_gap(target, blocker)

        # 如果 gap 很小，认为 blocker 直接阻塞了 target
        if 0 <= gap <= self.close_threshold_ns:
            # 延迟 ≈ blocker 的执行时间 + gap
            blocker_dur = blocker.gpu_dur_ns or blocker.cpu_enqueue_dur_ns or 0
            return blocker_dur + gap

        return 0

    def _compute_overlap_loss(self,
                             target: KernelRecord,
                             blocker: KernelRecord,
                             all_kernels: List[KernelRecord],
                             sketch: Optional[ExecutionSketch]) -> Optional[int]:
        """计算因 blocker 而丧失的 overlap。"""
        if not sketch:
            return None

        # 找与 target 应该 overlap 的 kernel
        for oe in sketch.overlap_expectations:
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
                    target, blocker, [], all_kernels
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
                                     blocker: KernelRecord,
                                     expected_preds: List[KernelRecord],
                                     all_kernels: List[KernelRecord]) -> Optional[int]:
        """
        计算反事实启动时间。
        
        如果没有 blocker，target 应该在何时启动？
        """
        # 反事实时间 = max(CPU enqueue, 预期前驱的 end, 其他 hard 依赖的 end)
        cf_time = target.cpu_enqueue_start_ns or 0

        for pred in expected_preds:
            pred_end = pred.gpu_end_ns or pred.cpu_enqueue_end_ns
            if pred_end:
                cf_time = max(cf_time, pred_end)

        # 移除 blocker 的贡献
        blocker_dur = blocker.gpu_dur_ns or blocker.cpu_enqueue_dur_ns or 0
        cf_time = max(cf_time - blocker_dur, target.cpu_enqueue_start_ns or 0)

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
                           blocker: KernelRecord,
                           delay_ns: int) -> float:
        """计算诊断的置信度。"""
        confidence = 0.5

        # 如果 gap 很小，置信度提高
        gap = self._compute_gap(target, blocker)
        if 0 <= gap <= 50_000:  # 50 us
            confidence += 0.3

        # 如果延迟很大，置信度提高
        if delay_ns > 1_000_000:  # 1 ms
            confidence += 0.1

        return min(0.95, confidence)

    def localize_batch(self,
                      targets: List[KernelRecord],
                      all_kernels: List[KernelRecord],
                      sketch: Optional[ExecutionSketch] = None) -> Dict[str, DependencyDiagnosis]:
        """批量定位。"""
        result = {}
        for target in targets:
            diagnosis = self.localize(target, all_kernels, sketch)
            if diagnosis:
                result[target.kid] = diagnosis

        return result
