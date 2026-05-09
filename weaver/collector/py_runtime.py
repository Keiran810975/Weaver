import atexit
import gc
import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from types import FrameType
from typing import Dict, Optional, Tuple


@dataclass
class PythonCollectorConfig:
    socket_path: str = "/tmp/weaver.sock"
    sample_rate: int = 1
    include_stdlib: bool = False
    targets: Tuple[str, ...] = ()
    trace_gc: bool = True
    emit_raw_calls: bool = False


class _Sender:
    def __init__(self, sock_path: str):
        self.sock_path = sock_path
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._lock = threading.Lock()

    def send(self, event: dict):
        raw = json.dumps(event, ensure_ascii=True).encode("utf-8")
        with self._lock:
            try:
                self.sock.sendto(raw, self.sock_path)
            except OSError:
                pass


class PythonRuntimeCollector:
    """CPython profile-hook based collector.

    This uses sys.setprofile (which routes to PyEval_SetProfile internally)
    to capture function call/return events with low payload and stream online.
    """

    def __init__(self, config: Optional[PythonCollectorConfig] = None):
        self.cfg = config or PythonCollectorConfig()
        self.sender = _Sender(self.cfg.socket_path)
        self._count = 0
        self._enabled = False
        self._orig_profiler = None
        self._active: Dict[int, Tuple[int, str]] = {}
        self._gc_callback = None

    def _is_interesting(self, filename: str) -> bool:
        if self.cfg.include_stdlib:
            return True
        # Keep user/workspace code and drop most stdlib/site-packages by default
        if not filename:
            return False
        lower = filename.lower()
        if "/site-packages/" in lower or "/lib/python" in lower:
            return False
        return True

    def _frame_name(self, frame: FrameType) -> str:
        code = frame.f_code
        module = frame.f_globals.get("__name__", "")
        qualname = code.co_qualname if hasattr(code, "co_qualname") else code.co_name
        return f"{module}.{qualname}"

    def _matches_target(self, frame: FrameType) -> bool:
        if not self.cfg.targets:
            return self._is_interesting(frame.f_code.co_filename)

        full = self._frame_name(frame)
        module = frame.f_globals.get("__name__", "")
        func = frame.f_code.co_name
        for target in self.cfg.targets:
            target = target.strip()
            if not target:
                continue
            normalized = target.replace("@", ".")
            if (
                full == normalized
                or full.endswith("." + normalized)
                or module == normalized
                or module.endswith("." + normalized)
                or func == normalized
            ):
                return True
        return False

    def _build_event(self, kind: str, frame: FrameType) -> dict:
        code = frame.f_code
        name = self._frame_name(frame)
        return {
            "ts_ns": time.time_ns(),
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "layer": "python",
            "kind": kind,
            "operator_name": name,
            "payload": {
                "func": code.co_name,
                "qualname": getattr(code, "co_qualname", code.co_name),
                "file": code.co_filename,
                "line": frame.f_lineno,
                "module": frame.f_globals.get("__name__", ""),
                "operator_name": name,
            },
        }

    def _emit_operator_event(self, frame: FrameType, start_ns: int, end_ns: int):
        code = frame.f_code
        name = self._frame_name(frame)
        self.sender.send(
            {
                "ts_ns": start_ns,
                "pid": os.getpid(),
                "tid": threading.get_ident(),
                "layer": "python",
                "kind": "operator",
                "operator_name": name,
                "ts_end_ns": end_ns,
                "dur_ns": end_ns - start_ns,
                "payload": {
                    "operator_name": name,
                    "func": code.co_name,
                    "qualname": getattr(code, "co_qualname", code.co_name),
                    "file": code.co_filename,
                    "line": frame.f_lineno,
                    "module": frame.f_globals.get("__name__", ""),
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "dur_ns": end_ns - start_ns,
                },
            }
        )

    def _hook(self, frame: FrameType, event: str, _arg):
        if event not in ("call", "return"):
            return self._hook
        if not self._matches_target(frame):
            return self._hook

        self._count += 1
        if self._count % max(1, self.cfg.sample_rate) != 0:
            return self._hook

        fid = id(frame)
        now = time.time_ns()
        if event == "call":
            self._active[fid] = (now, self._frame_name(frame))
            if self.cfg.emit_raw_calls:
                self.sender.send(self._build_event(event, frame))
        elif event == "return":
            start = self._active.pop(fid, None)
            if start is not None:
                self._emit_operator_event(frame, start[0], now)
            elif self.cfg.emit_raw_calls:
                self.sender.send(self._build_event(event, frame))
        return self._hook

    def _install_gc_callback(self):
        if not self.cfg.trace_gc or self._gc_callback is not None:
            return

        state: Dict[str, int] = {}

        def _gc_callback(phase, info):
            now = time.time_ns()
            if phase == "start":
                state["start_ns"] = now
                return
            if phase != "stop":
                return
            start_ns = state.pop("start_ns", now)
            payload = {
                "operator_name": "python.gc",
                "phase": phase,
                "start_ns": start_ns,
                "end_ns": now,
                "dur_ns": now - start_ns,
                "collected": int(info.get("collected", -1)) if isinstance(info, dict) else -1,
                "uncollectable": int(info.get("uncollectable", -1)) if isinstance(info, dict) else -1,
            }
            self.sender.send(
                {
                    "ts_ns": start_ns,
                    "pid": os.getpid(),
                    "tid": threading.get_ident(),
                    "layer": "python",
                    "kind": "operator",
                    "operator_name": "python.gc",
                    "ts_end_ns": now,
                    "dur_ns": now - start_ns,
                    "payload": payload,
                }
            )

        gc.callbacks.append(_gc_callback)
        self._gc_callback = _gc_callback

    def start(self):
        if self._enabled:
            return
        self._orig_profiler = sys.getprofile()
        self._install_gc_callback()
        sys.setprofile(self._hook)
        threading.setprofile(self._hook)
        self._enabled = True

    def stop(self):
        if not self._enabled:
            return
        sys.setprofile(self._orig_profiler)
        threading.setprofile(self._orig_profiler)
        if self._gc_callback is not None:
            try:
                gc.callbacks.remove(self._gc_callback)
            except ValueError:
                pass
            self._gc_callback = None
        self._enabled = False


_global_collector: Optional[PythonRuntimeCollector] = None


def enable_python_collector(
    socket_path: str = "/tmp/weaver.sock",
    sample_rate: int = 1,
    include_stdlib: bool = False,
    targets: Tuple[str, ...] = (),
    trace_gc: bool = True,
    emit_raw_calls: bool = False,
) -> PythonRuntimeCollector:
    global _global_collector
    if _global_collector is None:
        _global_collector = PythonRuntimeCollector(
            PythonCollectorConfig(
                socket_path=socket_path,
                sample_rate=sample_rate,
                include_stdlib=include_stdlib,
                targets=targets,
                trace_gc=trace_gc,
                emit_raw_calls=emit_raw_calls,
            )
        )
        _global_collector.start()
        atexit.register(_global_collector.stop)
    return _global_collector


def enable_from_env() -> Optional[PythonRuntimeCollector]:
    """Enable the collector from sitecustomize/launcher environment settings."""
    if os.environ.get("WEAVER_AUTO_PROFILE", "0") not in ("1", "true", "TRUE", "on", "ON"):
        return None
    targets = tuple(
        item.strip()
        for item in os.environ.get("WEAVER_PYTHON_TRACE_FUNCS", "").split(",")
        if item.strip()
    )
    return enable_python_collector(
        socket_path=os.environ.get("WEAVER_SOCK", "/tmp/weaver.sock"),
        sample_rate=max(1, int(os.environ.get("WEAVER_PYTHON_SAMPLE_RATE", "1"))),
        include_stdlib=os.environ.get("WEAVER_PROFILE_INCLUDE_STDLIB", "0") in ("1", "true", "TRUE"),
        targets=targets,
        trace_gc=os.environ.get("WEAVER_TRACE_GC", "1") not in ("0", "false", "FALSE"),
        emit_raw_calls=os.environ.get("WEAVER_PROFILE_RAW_CALLS", "0") in ("1", "true", "TRUE"),
    )
