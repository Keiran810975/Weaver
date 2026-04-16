import atexit
import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from types import FrameType
from typing import Callable, Optional


@dataclass
class PythonCollectorConfig:
    socket_path: str = "/tmp/weaver.sock"
    sample_rate: int = 1
    include_stdlib: bool = False


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

    def _build_event(self, kind: str, frame: FrameType) -> dict:
        code = frame.f_code
        return {
            "ts_ns": time.time_ns(),
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "layer": "python",
            "kind": kind,
            "payload": {
                "func": code.co_name,
                "file": code.co_filename,
                "line": frame.f_lineno,
                "module": frame.f_globals.get("__name__", ""),
            },
        }

    def _hook(self, frame: FrameType, event: str, _arg):
        if event not in ("call", "return"):
            return self._hook
        code = frame.f_code
        if not self._is_interesting(code.co_filename):
            return self._hook

        self._count += 1
        if self._count % max(1, self.cfg.sample_rate) != 0:
            return self._hook

        evt = self._build_event(event, frame)
        self.sender.send(evt)
        return self._hook

    def start(self):
        if self._enabled:
            return
        self._orig_profiler = sys.getprofile()
        sys.setprofile(self._hook)
        threading.setprofile(self._hook)
        self._enabled = True

    def stop(self):
        if not self._enabled:
            return
        sys.setprofile(self._orig_profiler)
        threading.setprofile(self._orig_profiler)
        self._enabled = False


_global_collector: Optional[PythonRuntimeCollector] = None


def enable_python_collector(
    socket_path: str = "/tmp/weaver.sock",
    sample_rate: int = 1,
    include_stdlib: bool = False,
) -> PythonRuntimeCollector:
    global _global_collector
    if _global_collector is None:
        _global_collector = PythonRuntimeCollector(
            PythonCollectorConfig(
                socket_path=socket_path,
                sample_rate=sample_rate,
                include_stdlib=include_stdlib,
            )
        )
        _global_collector.start()
        atexit.register(_global_collector.stop)
    return _global_collector
