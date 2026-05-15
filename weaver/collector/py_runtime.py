import atexit
import gc
import json
import os
import signal
import socket
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType
from typing import Dict, Optional, Tuple, Union


@dataclass
class PythonCollectorConfig:
    socket_path: str = "/tmp/weaver.sock"
    sample_rate: int = 1
    include_stdlib: bool = False
    targets: Tuple[str, ...] = ()
    trace_gc: bool = True
    emit_raw_calls: bool = False
    event_budget: int = 0


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
        self._sample_rate = max(1, int(self.cfg.sample_rate))
        self._targets = tuple(t.strip().replace("@", ".") for t in self.cfg.targets if t.strip())
        self._enabled = False
        self._profile_active = False
        self._orig_profiler = None
        self._active: Dict[int, Tuple[int, str]] = {}
        self._match_cache: Dict[object, bool] = {}
        self._gc_callback = None
        self._emitted_events = 0

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
        code = frame.f_code
        cached = self._match_cache.get(code)
        if cached is not None:
            return cached

        if not self._targets:
            matched = self._is_interesting(code.co_filename)
            self._match_cache[code] = matched
            return matched

        full = self._frame_name(frame)
        module = frame.f_globals.get("__name__", "")
        func = code.co_name
        matched = False
        for target in self._targets:
            if (
                full == target
                or full.endswith("." + target)
                or module == target
                or module.endswith("." + target)
                or func == target
            ):
                matched = True
                break
        self._match_cache[code] = matched
        return matched

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
        self._after_operator_event()

    def _after_operator_event(self):
        self._emitted_events += 1
        budget = max(0, int(self.cfg.event_budget))
        if budget and self._emitted_events >= budget:
            self.pause()

    def _hook(self, frame: FrameType, event: str, _arg):
        if not self._profile_active:
            return self._hook
        if event not in ("call", "return"):
            return self._hook
        if not self._matches_target(frame):
            return self._hook

        fid = id(frame)
        now = time.time_ns()
        if event == "call":
            self._count += 1
            if self._count % self._sample_rate != 0:
                return self._hook
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
        self._profile_active = True

    def stop(self):
        if not self._enabled:
            return
        sys.setprofile(self._orig_profiler)
        threading.setprofile(self._orig_profiler)
        self._profile_active = False
        if self._gc_callback is not None:
            try:
                gc.callbacks.remove(self._gc_callback)
            except ValueError:
                pass
            self._gc_callback = None
        self._enabled = False

    def pause(self):
        if not self._enabled or not self._profile_active:
            return
        sys.setprofile(self._orig_profiler)
        threading.setprofile(self._orig_profiler)
        self._profile_active = False

    def resume(self):
        if not self._enabled or self._profile_active:
            return
        sys.setprofile(self._hook)
        threading.setprofile(self._hook)
        self._profile_active = True


class NativePythonRuntimeCollector:
    """Low-overhead CPython collector implemented in C.

    The native backend follows the Flare/DLRover shape: PyEval_SetProfile runs a
    C callback, target functions are cached by code-object address, and records
    are buffered in a native queue before a background flusher writes to Weaver.
    """

    def __init__(self, config: Optional[PythonCollectorConfig] = None):
        self.cfg = config or PythonCollectorConfig()
        self._enabled = False
        self._native = None

    def start(self):
        if self._enabled:
            return
        if not self.cfg.targets:
            raise RuntimeError("native Python tracing requires WEAVER_PYTHON_TRACE_FUNCS targets")
        from . import _native_py_trace

        queue_size = max(1024, int(os.environ.get("WEAVER_NATIVE_PY_QUEUE_SIZE", "65536")))
        event_budget = max(0, int(self.cfg.event_budget))
        _native_py_trace.start(
            sock_path=self.cfg.socket_path,
            targets=self.cfg.targets,
            sample_rate=max(1, int(self.cfg.sample_rate)),
            queue_size=queue_size,
            event_budget=event_budget,
        )
        self._native = _native_py_trace
        self._enabled = True

    def stop(self):
        if not self._enabled:
            return
        if self._native is not None:
            self._native.stop()
        self._enabled = False

    def pause(self):
        if not self._enabled or self._native is None:
            return
        self._native.pause()

    def resume(self):
        if not self._enabled or self._native is None:
            return
        self._native.resume()


class AdaptivePythonTriggerCollector:
    """Operator sampler armed by daemon-side extra-kernel triggers.

    This deliberately does not install a continuous CPython profile hook.  The
    daemon sends a process signal after the CUDA hook reports an unexpected or
    trigger-captured kernel, and the handler emits one operator attribution
    sample from the current Python stack.
    """

    def __init__(self, config: Optional[PythonCollectorConfig] = None):
        self.cfg = config or PythonCollectorConfig()
        self.sender = _Sender(self.cfg.socket_path)
        self._targets = tuple(t.strip().replace("@", ".") for t in self.cfg.targets if t.strip())
        self._match_cache: Dict[object, bool] = {}
        self._enabled = False
        self._emitted_events = 0
        self._signal_number = _parse_signal_number(os.environ.get("WEAVER_ADAPTIVE_PY_SIGNAL", "SIGUSR1"))
        self._previous_handler = None

    def _is_interesting(self, filename: str) -> bool:
        if self.cfg.include_stdlib:
            return True
        if not filename:
            return False
        lower = filename.lower()
        if "/site-packages/" in lower or "/lib/python" in lower:
            return False
        if "/weaver/collector/py_runtime.py" in lower or lower.endswith("/sitecustomize.py"):
            return False
        return True

    def _frame_name(self, frame: FrameType) -> str:
        code = frame.f_code
        module = frame.f_globals.get("__name__", "")
        qualname = code.co_qualname if hasattr(code, "co_qualname") else code.co_name
        return f"{module}.{qualname}"

    def _matches_target(self, frame: FrameType) -> bool:
        code = frame.f_code
        cached = self._match_cache.get(code)
        if cached is not None:
            return cached

        if not self._targets:
            matched = self._is_interesting(code.co_filename)
            self._match_cache[code] = matched
            return matched

        full = self._frame_name(frame)
        module = frame.f_globals.get("__name__", "")
        func = code.co_name
        matched = False
        for target in self._targets:
            if (
                full == target
                or full.endswith("." + target)
                or module == target
                or module.endswith("." + target)
                or func == target
            ):
                matched = True
                break
        self._match_cache[code] = matched
        return matched

    def _select_frame(self, frame: Optional[FrameType]) -> Optional[FrameType]:
        fallback = None
        cur = frame
        while cur is not None:
            if self._matches_target(cur):
                return cur
            if fallback is None and self._is_interesting(cur.f_code.co_filename):
                fallback = cur
            cur = cur.f_back
        return fallback

    def _handler(self, signum: int, frame: Optional[FrameType]) -> None:
        budget = max(0, int(self.cfg.event_budget))
        if budget and self._emitted_events >= budget:
            return
        selected = self._select_frame(frame)
        if selected is None:
            return
        now = time.time_ns()
        code = selected.f_code
        name = self._frame_name(selected)
        self.sender.send(
            {
                "ts_ns": now,
                "pid": os.getpid(),
                "tid": threading.get_ident(),
                "layer": "python",
                "kind": "operator",
                "source": "weaver_adaptive_py_trigger",
                "operator_name": name,
                "ts_end_ns": now,
                "dur_ns": 0,
                "payload": {
                    "operator_name": name,
                    "func": code.co_name,
                    "qualname": getattr(code, "co_qualname", code.co_name),
                    "file": code.co_filename,
                    "line": selected.f_lineno,
                    "module": selected.f_globals.get("__name__", ""),
                    "trigger": "extra_kernel_signal",
                    "signal": int(signum),
                    "collector": "adaptive_signal_stack_sample",
                },
            }
        )
        self._emitted_events += 1

    def start(self):
        if self._enabled:
            return
        self._previous_handler = signal.getsignal(self._signal_number)
        signal.signal(self._signal_number, self._handler)
        self._enabled = True

    def stop(self):
        if not self._enabled:
            return
        try:
            signal.signal(self._signal_number, self._previous_handler)
        except Exception:
            pass
        self._enabled = False

    def pause(self):
        return

    def resume(self):
        return


CollectorHandle = Union[PythonRuntimeCollector, NativePythonRuntimeCollector, AdaptivePythonTriggerCollector]


_global_collector: Optional[CollectorHandle] = None


def _parse_signal_number(value: str) -> int:
    text = str(value or "SIGUSR1").strip()
    if text.isdigit():
        return int(text)
    name = text.upper()
    if not name.startswith("SIG"):
        name = "SIG" + name
    number = getattr(signal, name, None)
    if number is None:
        raise ValueError(f"unknown signal for WEAVER_ADAPTIVE_PY_SIGNAL: {value!r}")
    return int(number)


def enable_adaptive_python_trigger_collector(
    socket_path: str = "/tmp/weaver.sock",
    include_stdlib: bool = False,
    targets: Tuple[str, ...] = (),
    event_budget: int = 0,
) -> CollectorHandle:
    global _global_collector
    if _global_collector is None:
        cfg = PythonCollectorConfig(
            socket_path=socket_path,
            include_stdlib=include_stdlib,
            targets=targets,
            trace_gc=False,
            emit_raw_calls=False,
            event_budget=max(0, int(event_budget)),
        )
        _global_collector = AdaptivePythonTriggerCollector(cfg)
        _global_collector.start()
        atexit.register(_global_collector.stop)
    return _global_collector


def enable_python_collector(
    socket_path: str = "/tmp/weaver.sock",
    sample_rate: int = 1,
    include_stdlib: bool = False,
    targets: Tuple[str, ...] = (),
    trace_gc: bool = True,
    emit_raw_calls: bool = False,
    backend: Optional[str] = None,
    event_budget: int = 0,
) -> CollectorHandle:
    global _global_collector
    if _global_collector is None:
        cfg = PythonCollectorConfig(
            socket_path=socket_path,
            sample_rate=sample_rate,
            include_stdlib=include_stdlib,
            targets=targets,
            trace_gc=trace_gc,
            emit_raw_calls=emit_raw_calls,
            event_budget=max(0, int(event_budget)),
        )
        selected = (backend or os.environ.get("WEAVER_PYTHON_COLLECTOR", "native")).lower()
        require_native = os.environ.get("WEAVER_REQUIRE_NATIVE_PY", "0") in ("1", "true", "TRUE", "on", "ON")
        if selected in ("native", "flare") and require_native and not targets:
            raise RuntimeError("native Python tracing requires non-empty WEAVER_PYTHON_TRACE_FUNCS")
        if selected in ("native", "flare", "auto") and targets:
            try:
                _global_collector = NativePythonRuntimeCollector(cfg)
                _global_collector.start()
            except Exception:
                _global_collector = None
                if require_native or selected in ("native", "flare"):
                    raise
        if _global_collector is None:
            _global_collector = PythonRuntimeCollector(cfg)
            _global_collector.start()
        atexit.register(_global_collector.stop)
    return _global_collector


def enable_from_env() -> Optional[CollectorHandle]:
    """Enable the collector from sitecustomize/launcher environment settings."""
    if os.environ.get("WEAVER_AUTO_PROFILE", "0") not in ("1", "true", "TRUE", "on", "ON"):
        return None
    targets = tuple(
        item.strip()
        for item in os.environ.get("WEAVER_PYTHON_TRACE_FUNCS", "").split(",")
        if item.strip()
    )
    if os.environ.get("WEAVER_PYTHON_ADAPTIVE", "0") in ("1", "true", "TRUE", "on", "ON"):
        return enable_adaptive_python_trigger_collector(
            socket_path=os.environ.get("WEAVER_SOCK", "/tmp/weaver.sock"),
            include_stdlib=os.environ.get("WEAVER_PROFILE_INCLUDE_STDLIB", "0") in ("1", "true", "TRUE"),
            targets=targets,
            event_budget=max(0, int(os.environ.get("WEAVER_PYTHON_EVENT_BUDGET", "0"))),
        )
    return enable_python_collector(
        socket_path=os.environ.get("WEAVER_SOCK", "/tmp/weaver.sock"),
        sample_rate=max(1, int(os.environ.get("WEAVER_PYTHON_SAMPLE_RATE", "1"))),
        include_stdlib=os.environ.get("WEAVER_PROFILE_INCLUDE_STDLIB", "0") in ("1", "true", "TRUE"),
        targets=targets,
        trace_gc=os.environ.get("WEAVER_TRACE_GC", "1") not in ("0", "false", "FALSE"),
        emit_raw_calls=os.environ.get("WEAVER_PROFILE_RAW_CALLS", "0") in ("1", "true", "TRUE"),
        backend=os.environ.get("WEAVER_PYTHON_COLLECTOR", "native"),
        event_budget=max(0, int(os.environ.get("WEAVER_PYTHON_EVENT_BUDGET", "0"))),
    )


def pause_python_collector() -> None:
    if _global_collector is not None and hasattr(_global_collector, "pause"):
        _global_collector.pause()


def resume_python_collector() -> None:
    if _global_collector is not None and hasattr(_global_collector, "resume"):
        _global_collector.resume()


@contextmanager
def python_trace_window():
    """Temporarily arm the Python collector around a suspected region."""
    resume_python_collector()
    try:
        yield
    finally:
        pause_python_collector()
