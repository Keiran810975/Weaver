"""
规范化层：从 aligned timeline 转换为规范化记录。

主要任务：
1. 加载 aligned timeline JSON 事件
2. 将不同来源的事件（hook、profiler、python）转换为统一的 record 类型
3. 关联 profiler CUDA 事件和 hook launch 事件
4. 提取 family、tag 信息
"""

import json
import sys
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .records import KernelRecord, OperatorRecord, SyncRecord, SyncKind
from ..sketch import classify_kernel, KernelRecord as SketchKernelRecord


def _payload(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = event.get("payload", {})
    return payload if isinstance(payload, dict) else {}


def _get(event: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in event and event[key] is not None:
        return event[key]
    return _payload(event).get(key, default)


def _get_first(event: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        value = _get(event, key)
        if value is not None:
            return value
    return default


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if isinstance(value, str) and value.strip():
            return float(value)
    except ValueError:
        return None
    return None


class TimelineNormalizer:
    """从 aligned timeline 转换为规范化记录。"""

    def __init__(self, timeline_path: str):
        """
        初始化 normalizer。
        
        Args:
            timeline_path: aligned_timeline_rank*.ndjson 的路径
        """
        self.timeline_path = timeline_path
        self.raw_events: List[Dict[str, Any]] = []
        self.kernel_records: List[KernelRecord] = []
        self.operator_records: List[OperatorRecord] = []
        self.sync_records: List[SyncRecord] = []

    def load_timeline(self) -> None:
        """加载 timeline NDJSON 文件。"""
        self.raw_events = []

        try:
            with open(self.timeline_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        self.raw_events.append(event)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
        except FileNotFoundError:
            raise FileNotFoundError(f"Timeline file not found: {self.timeline_path}")

    def normalize(self) -> Tuple[List[KernelRecord], List[OperatorRecord], List[SyncRecord]]:
        """
        规范化 timeline 事件。
        
        返回：(kernel_records, operator_records, sync_records)
        """
        # 加载原始事件
        self.load_timeline()

        # 转换事件
        for event in self.raw_events:
            kind = event.get("kind", "").lower()
            layer = event.get("layer", "").lower()

            # 处理 kernel 事件
            if self._is_kernel_event(event):
                kr = self._convert_kernel_event(event)
                if kr:
                    self.kernel_records.append(kr)

            # 处理 operator 事件
            elif self._is_operator_event(event):
                opr = self._convert_operator_event(event)
                if opr:
                    self.operator_records.append(opr)

            # 处理 sync 事件
            elif self._is_sync_event(event):
                sr = self._convert_sync_event(event)
                if sr:
                    self.sync_records.append(sr)

        # 按时间戳排序
        self.kernel_records.sort(key=lambda k: k.gpu_start_ns or k.cpu_enqueue_start_ns or 0)
        self.operator_records.sort(key=lambda o: o.ts_start_ns)
        self.sync_records.sort(key=lambda s: s.ts_start_ns)

        return self.kernel_records, self.operator_records, self.sync_records

    def _is_kernel_event(self, event: Dict[str, Any]) -> bool:
        """判断是否为 kernel 事件。"""
        kind = event.get("kind", "").lower()
        layer = event.get("layer", "").lower()
        if layer == "neutrino":
            return False
        if layer == "kernel":
            return True
        return (
            kind in ("kernel_launch", "kernel_launch_ex", "launch")
            or "hook_launch" in kind
            or kind.startswith("nccl_")
            or (layer == "cuda" and "kernel" in kind)
        )

    def _is_operator_event(self, event: Dict[str, Any]) -> bool:
        """判断是否为 operator 事件。"""
        kind = event.get("kind", "").lower()
        layer = event.get("layer", "").lower()
        return layer == "operator" or "operator" in kind or kind == "profiler_operator"

    def _is_sync_event(self, event: Dict[str, Any]) -> bool:
        """判断是否为 sync 事件。"""
        kind = event.get("kind", "").lower()
        return any(s in kind for s in ["sync", "wait", "barrier", "synchronize"])

    def _convert_kernel_event(self, event: Dict[str, Any]) -> Optional[KernelRecord]:
        """将原始 kernel 事件转换为 KernelRecord。"""
        payload = _payload(event)
        kernel_name = (
            _get_first(event, ["kernel_name", "kernel", "name"])
            or payload.get("kernel_name")
            or payload.get("kernel")
            or payload.get("name")
            or ""
        )
        if not kernel_name:
            return None

        # 提取基本信息
        kid = _get(event, "id") or f"k_{id(event)}"
        rank = _get(event, "rank")
        pid = _get(event, "pid", 0)
        tid = _get(event, "tid")
        stream = _get(event, "stream")

        operator_name = _get(event, "operator_name")
        operator_id = _get(event, "operator_id")

        # 时间信息：优先使用 GPU 时间
        gpu_start_ns = _get(event, "gpu_start_ns")
        gpu_end_ns = _get(event, "gpu_end_ns")
        if gpu_start_ns and gpu_end_ns:
            duration_ns = gpu_end_ns - gpu_start_ns
        else:
            duration_ns = _get(event, "dur_ns")
            if gpu_start_ns is None and event.get("layer", "").lower() == "kernel":
                gpu_start_ns = _get(event, "ts_ns")
                if duration_ns is not None:
                    gpu_end_ns = gpu_start_ns + duration_ns

        # CPU 侧时间（来自 hook）
        cpu_enqueue_start_ns = _get_first(event, ["cpu_enqueue_start_ns", "start_ns"])
        cpu_enqueue_end_ns = _get_first(event, ["cpu_enqueue_end_ns", "end_ns"])

        # 执行信息
        grid = tuple(_get(event, "grid", [1, 1, 1]))
        block = tuple(_get(event, "block", [128, 1, 1]))
        total_warps = _get(event, "total_warps")
        shared_memory = _get_first(event, ["shared_memory", "shared_mem"])

        # 分类
        sketch_record = SketchKernelRecord(
            kernel_name=kernel_name,
            operator_name=operator_name,
            kind=event.get("kind"),
            event_type=_get(event, "event_type"),
            payload=payload,
            grid=grid,
            block=block,
        )
        kclass = classify_kernel(sketch_record)

        # 工作量
        work_type = "unknown"
        work_value = None

        explicit_flops = _number(_get_first(event, ["flops", "FLOPs", "floating_point_ops"]))
        m = _number(_get_first(event, ["m", "M"]))
        n = _number(_get_first(event, ["n", "N"]))
        k = _number(_get_first(event, ["k", "K"]))
        batch = _number(_get_first(event, ["batch", "batch_size"], 1.0)) or 1.0

        if explicit_flops is not None:
            work_type = "flops"
            work_value = explicit_flops
        elif kclass.family == "GEMM" and m and n and k:
            work_type = "flops"
            work_value = 2.0 * batch * m * n * k
        elif kclass.family == "NCCL":
            work_type = "bytes"
            count = payload.get("count", 0)
            dtype_size = payload.get("dtype_size", 4)
            work_value = count * dtype_size if count > 0 else None
        elif payload.get("bytes"):
            work_type = "bytes"
            work_value = float(payload["bytes"])
        elif total_warps:
            work_type = "warps"
            work_value = float(total_warps)

        # 创建 kernel record
        return KernelRecord(
            kid=kid,
            rank=rank,
            pid=pid,
            tid=tid,
            stream=stream,
            kernel_name=kernel_name,
            family=kclass.family,
            tag=kclass.tag,
            operator_name=operator_name,
            operator_id=operator_id,
            cpu_enqueue_start_ns=cpu_enqueue_start_ns,
            cpu_enqueue_end_ns=cpu_enqueue_end_ns,
            gpu_start_ns=gpu_start_ns,
            gpu_end_ns=gpu_end_ns,
            duration_ns=duration_ns,
            grid=grid,
            block=block,
            total_warps=total_warps,
            shared_memory=shared_memory,
            work_type=work_type,
            work_value=work_value,
            payload=payload,
            source=_get(event, "source", "unknown"),
        )

    def _convert_operator_event(self, event: Dict[str, Any]) -> Optional[OperatorRecord]:
        """将原始 operator 事件转换为 OperatorRecord。"""
        payload = _payload(event)
        operator_name = _get_first(event, ["operator_name", "name"]) or payload.get("operator_name")
        if not operator_name:
            return None

        oid = _get(event, "id") or f"op_{id(event)}"
        rank = _get(event, "rank")
        pid = _get(event, "pid", 0)
        phase = _get(event, "phase")

        ts_start_ns = _get_first(event, ["ts_ns", "start_ns"], 0)
        ts_end_ns = _get_first(event, ["ts_end_ns", "end_ns"])

        return OperatorRecord(
            oid=oid,
            operator_name=operator_name,
            phase=phase,
            rank=rank,
            pid=pid,
            ts_start_ns=ts_start_ns,
            ts_end_ns=ts_end_ns,
            payload=payload,
        )

    def _convert_sync_event(self, event: Dict[str, Any]) -> Optional[SyncRecord]:
        """将原始 sync 事件转换为 SyncRecord。"""
        kind_str = event.get("kind", "").lower()

        # 映射 kind
        if "cuda_synchronize" in kind_str or "synchronize" in kind_str:
            kind = SyncKind.CUDA_SYNCHRONIZE
        elif "event_wait" in kind_str or "wait" in kind_str:
            kind = SyncKind.EVENT_WAIT
        elif "timer" in kind_str or "barrier" in kind_str:
            kind = SyncKind.PROFILER_TIMER
        else:
            kind = SyncKind.UNKNOWN_SYNC

        sid = _get(event, "id") or f"sync_{id(event)}"
        rank = _get(event, "rank")
        pid = _get(event, "pid", 0)

        ts_start_ns = _get_first(event, ["ts_ns", "start_ns"], 0)
        ts_end_ns = _get_first(event, ["ts_end_ns", "end_ns"])

        # 提取调用栈
        stack = _get(event, "stack", [])
        if isinstance(stack, str):
            stack = [stack]

        payload = _payload(event)
        source = _get(event, "source", "unknown")

        return SyncRecord(
            sid=sid,
            kind=kind,
            rank=rank,
            pid=pid,
            ts_start_ns=ts_start_ns,
            ts_end_ns=ts_end_ns,
            stack=stack,
            source=source,
            payload=payload,
        )
