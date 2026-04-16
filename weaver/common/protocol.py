import os
import socket
import time
from typing import Any, Dict


def now_ns() -> int:
    return time.time_ns()


def make_event(layer: str, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ts_ns": now_ns(),
        "pid": os.getpid(),
        "tid": getattr(threading_like(), "ident", 0),
        "host": socket.gethostname(),
        "layer": layer,
        "kind": kind,
        "payload": payload,
    }


def threading_like():
    try:
        import threading

        return threading.current_thread()
    except Exception:
        class Dummy:
            ident = 0

        return Dummy()
