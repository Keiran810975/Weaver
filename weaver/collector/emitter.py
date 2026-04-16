import json
import os
import socket
import threading
import time
from typing import Any, Dict, Optional


class _DatagramEmitter:
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._lock = threading.Lock()

    def emit(self, event: Dict[str, Any]):
        raw = json.dumps(event, ensure_ascii=True).encode("utf-8")
        with self._lock:
            try:
                self._sock.sendto(raw, self.socket_path)
            except OSError:
                pass


_default_emitter: Optional[_DatagramEmitter] = None


def emit_event(
    kind: str,
    payload: Dict[str, Any],
    layer: str = "runtime",
    socket_path: Optional[str] = None,
):
    global _default_emitter
    sock = socket_path or os.environ.get("WEAVER_SOCK", "/tmp/weaver.sock")
    if _default_emitter is None or _default_emitter.socket_path != sock:
        _default_emitter = _DatagramEmitter(sock)

    evt = {
        "ts_ns": time.time_ns(),
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "host": socket.gethostname(),
        "layer": layer,
        "kind": kind,
        "payload": payload,
    }
    _default_emitter.emit(evt)
