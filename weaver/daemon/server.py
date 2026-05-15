import argparse
import json
import os
import queue
import signal
import socket
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Deque, Dict, Optional


class EventStore:
    def __init__(self, tail_size: int = 2000):
        self._tail: Deque[dict] = deque(maxlen=tail_size)
        self._lock = threading.Lock()
        self._layer_counter = Counter()
        self._kind_counter = Counter()
        self._warp_counter = Counter()
        self._total = 0

    def push(self, evt: dict):
        with self._lock:
            self._tail.append(evt)
            self._total += 1
            layer = evt.get("layer", "unknown")
            kind = evt.get("kind", "unknown")
            self._layer_counter[layer] += 1
            self._kind_counter[kind] += 1
            payload = evt.get("payload", {})
            if layer == "cuda" and "total_warps" in payload:
                bucket = int(payload.get("total_warps", 0))
                self._warp_counter[bucket] += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "total_events": self._total,
                "by_layer": dict(self._layer_counter),
                "by_kind": dict(self._kind_counter),
                "cuda_warp_hist": dict(self._warp_counter),
            }

    def tail(self, n: int) -> list:
        with self._lock:
            return list(self._tail)[-n:]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) in ("1", "true", "TRUE", "on", "ON")


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


class AdaptivePythonSignaler:
    """Signal the target Python process after the hook reports an extra kernel."""

    def __init__(self):
        self.enabled = _env_flag("WEAVER_DAEMON_ADAPTIVE_PY_SIGNAL", "0")
        self.signal_number = _parse_signal_number(os.environ.get("WEAVER_ADAPTIVE_PY_SIGNAL", "SIGUSR1"))
        self.min_interval_ns = max(0, int(os.environ.get("WEAVER_ADAPTIVE_PY_SIGNAL_MIN_INTERVAL_NS", "1000000")))
        self._last_sent_ns: Dict[int, int] = {}

    def maybe_signal(self, evt: dict) -> None:
        if not self.enabled or not self._is_extra_kernel_event(evt):
            return
        pid = int(evt.get("pid") or 0)
        if pid <= 0 or pid == os.getpid():
            return
        now = time.time_ns()
        last = self._last_sent_ns.get(pid, 0)
        if self.min_interval_ns and now - last < self.min_interval_ns:
            return
        try:
            os.kill(pid, self.signal_number)
            self._last_sent_ns[pid] = now
        except OSError:
            return

    @staticmethod
    def _is_extra_kernel_event(evt: dict) -> bool:
        if evt.get("layer") != "cuda" or evt.get("kind") != "kernel_launch":
            return False
        payload = evt.get("payload") or {}
        capture_mode = str(payload.get("capture_mode", "")).lower()
        trigger_reason = str(payload.get("trigger_reason", "")).lower()
        if capture_mode == "trigger_full":
            return True
        if trigger_reason in ("sequence_mismatch", "unexpected_kernel_name"):
            return True
        return False


class WeaverDatagramServer:
    def __init__(self, sock_path: str, store: EventStore, out_file: Optional[str]):
        self.sock_path = sock_path
        self.store = store
        self.out_file = out_file
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._writer_q: "queue.Queue[str]" = queue.Queue(maxsize=20000)
        self._writer_thread: Optional[threading.Thread] = None
        self._adaptive_python = AdaptivePythonSignaler()

    def start(self):
        p = Path(self.sock_path)
        if p.exists():
            p.unlink()
        self._sock.bind(self.sock_path)
        self._thread = threading.Thread(target=self._run, name="weaver-dgram", daemon=True)
        self._thread.start()
        if self.out_file:
            self._writer_thread = threading.Thread(target=self._writer_run, name="weaver-writer", daemon=True)
            self._writer_thread.start()

    def stop(self):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass

    def _writer_run(self):
        out_path = Path(self.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            while not self._stop.is_set() or not self._writer_q.empty():
                try:
                    line = self._writer_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                f.write(line)
                f.write("\n")
                f.flush()

    def _run(self):
        while not self._stop.is_set():
            try:
                data = self._sock.recv(65535)
            except OSError:
                break
            if not data:
                continue
            try:
                evt = json.loads(data.decode("utf-8", errors="ignore"))
            except json.JSONDecodeError:
                continue
            self.store.push(evt)
            if self.out_file:
                try:
                    self._writer_q.put_nowait(json.dumps(evt, ensure_ascii=True))
                except queue.Full:
                    pass
            self._adaptive_python.maybe_signal(evt)


class ApiHandler(BaseHTTPRequestHandler):
    store: EventStore = None  # type: ignore

    def _reply(self, code: int, body: dict):
        raw = json.dumps(body, ensure_ascii=True).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            return self._reply(200, {"ok": True, "ts_ns": time.time_ns()})
        if self.path == "/stats":
            return self._reply(200, self.store.snapshot())
        if self.path.startswith("/tail"):
            n = 100
            if "?" in self.path and "n=" in self.path:
                try:
                    n = int(self.path.split("n=", 1)[1])
                except ValueError:
                    n = 100
            return self._reply(200, {"events": self.store.tail(max(1, min(n, 2000)))})
        return self._reply(404, {"error": "not found"})

    def log_message(self, fmt: str, *args):
        return


def run(sock_path: str, http_host: str, http_port: int, out_file: Optional[str]):
    store = EventStore()
    dgram = WeaverDatagramServer(sock_path=sock_path, store=store, out_file=out_file)
    dgram.start()

    ApiHandler.store = store
    httpd = ThreadingHTTPServer((http_host, http_port), ApiHandler)

    stop_event = threading.Event()

    def _shutdown(*_):
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.3}, daemon=True)
    t.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        httpd.shutdown()
        dgram.stop()
        try:
            Path(sock_path).unlink(missing_ok=True)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Weaver online collector daemon")
    parser.add_argument("--sock", default="/tmp/weaver.sock", help="unix datagram socket path")
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=18731)
    parser.add_argument("--out", default="./weaver_events.ndjson", help="ndjson output file")
    args = parser.parse_args()
    run(sock_path=args.sock, http_host=args.http_host, http_port=args.http_port, out_file=args.out)


if __name__ == "__main__":
    main()
