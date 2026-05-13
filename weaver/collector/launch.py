import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


FAMILY_KERNEL_PATTERNS = {
    "GEMM": [
        "gemm",
        "sgemm",
        "dgemm",
        "hgemm",
        "matmul",
        "cutlass",
        "triton_gemm",
        "triton_mm",
        "cublas",
    ],
    "NCCL": ["nccl", "allreduce", "all_reduce", "allgather", "reducescatter"],
    "MEMCPY": ["memcpy", "copy", "transpose", "permute", "contiguous"],
    "MEMORY": ["memcpy", "copy", "transpose", "permute", "contiguous"],
    "REDUCTION": ["reduce", "softmax", "norm", "layer_norm", "rms_norm"],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_hook() -> Path:
    return _repo_root() / "hooks" / "libweaver_hook.so"


def _prepend_env_path(env: dict, key: str, value: str) -> None:
    old = env.get(key)
    env[key] = value if not old else f"{value}{os.pathsep}{old}"


def _prepend_preload(env: dict, hook: Path) -> None:
    key = "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"
    old = env.get(key)
    env[key] = str(hook) if not old else f"{hook}:{old}"


def _append_patterns(patterns: List[str], value, prefix: str = "") -> None:
    if not value:
        return
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return
    for item in values:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            patterns.append(f"{prefix}{text}" if prefix else text)


def _append_family_patterns(patterns: List[str], family) -> None:
    if not family:
        return
    if isinstance(family, str):
        families = [family]
    elif isinstance(family, (list, tuple)):
        families = family
    else:
        return
    for item in families:
        key = str(item).strip().upper()
        for pattern in FAMILY_KERNEL_PATTERNS.get(key, []):
            patterns.append(pattern)


def _expected_patterns_from_sketch(path: Optional[str]) -> List[str]:
    if not path:
        return []
    sketch_path = Path(path).expanduser().resolve()
    data = json.loads(sketch_path.read_text(encoding="utf-8"))
    patterns: List[str] = []

    for scope in (data, data.get("metadata") or {}):
        _append_patterns(patterns, scope.get("expected_kernel_names"), "exact:")
        _append_patterns(patterns, scope.get("expected_kernel_patterns"))
        _append_patterns(patterns, scope.get("expected_kernel_regexes"), "regex:")

    for template in data.get("kernel_templates", []) or []:
        match = template.get("match") or {}
        _append_patterns(patterns, match.get("kernel_name"), "exact:")
        _append_patterns(patterns, match.get("kernel_names"), "exact:")
        _append_patterns(patterns, match.get("kernel_name_substr"))
        _append_patterns(patterns, match.get("kernel_name_contains"))
        _append_patterns(patterns, match.get("kernel_name_regex"), "regex:")
        _append_patterns(patterns, match.get("name_regex"), "regex:")
        _append_family_patterns(patterns, template.get("family"))

    for dep in data.get("expected_dependencies", []) or data.get("dependency_expectations", []) or []:
        _append_family_patterns(patterns, (dep.get("target") or {}).get("family"))
        for pred in dep.get("predecessors") or []:
            _append_family_patterns(patterns, pred.get("family"))

    # Stable de-duplication keeps the environment compact.
    seen = set()
    deduped = []
    for pattern in patterns:
        if pattern not in seen:
            seen.add(pattern)
            deduped.append(pattern)
    return deduped


def _start_daemon(args) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "weaver.daemon.server",
        "--sock",
        args.sock,
        "--http-host",
        args.http_host,
        "--http-port",
        str(args.http_port),
        "--out",
        args.out,
    ]
    env = os.environ.copy()
    _prepend_env_path(env, "PYTHONPATH", str(_repo_root()))
    return subprocess.Popen(cmd, env=env)


def _wait_for_socket(sock: str, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    path = Path(sock)
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"Weaver daemon socket did not appear: {sock}")


def _target_env(args) -> dict:
    env = os.environ.copy()
    env["WEAVER_SOCK"] = args.sock
    env["WEAVER_AUTO_PROFILE"] = "1"
    env.setdefault("WEAVER_PYTHON_COLLECTOR", "native")
    env.setdefault("WEAVER_REQUIRE_NATIVE_PY", "1")
    env.setdefault(
        "WEAVER_PYTHON_TRACE_FUNCS",
        ",".join(
            [
                "torch.utils.data.dataloader@_BaseDataLoaderIter@__next__",
                "torch@cuda@synchronize",
                "torch.cuda@Event@synchronize",
                "torch.cuda@Event@wait",
                "torch.cuda@Stream@synchronize",
                "torch.cuda@Stream@wait_event",
                "torch.cuda@Stream@wait_stream",
                "torch@autograd@backward",
                "torch@autograd@grad",
            ]
        ),
    )
    env.setdefault("WEAVER_COLLECTION_MODE", args.collection_mode)
    env.setdefault("WEAVER_TRIGGER_CAPTURE_AFTER", str(max(0, args.trigger_capture_after)))
    patterns = []
    patterns.extend(_expected_patterns_from_sketch(args.sketch))
    if args.expected_kernels:
        patterns.extend([p.strip() for p in args.expected_kernels.split(",") if p.strip()])
    if patterns:
        env["WEAVER_EXPECTED_KERNELS"] = ";".join(patterns)
    env.setdefault("WEAVER_CUDA_EVENTS", "1")
    env.setdefault("WEAVER_CUDA_SYNC_ANCHOR", "1")
    env.setdefault("WEAVER_PYTHON_EVENT_BUDGET", "1")
    env.setdefault("WEAVER_EMIT_CODE_EVENTS", "0")
    env.setdefault("WEAVER_ASYNC_LAUNCH_EMIT", "1")
    env.setdefault("WEAVER_PATCH_DLSYM", "0")
    env.setdefault("WEAVER_PATCH_GETPROC", "1")
    env.setdefault("WEAVER_ENABLE_DISASM", "0")
    env.setdefault("WEAVER_TRACE_DIR", str(Path(args.out).resolve().parent))
    _prepend_env_path(env, "PYTHONPATH", str(_repo_root()))
    if args.hook:
        _prepend_preload(env, Path(args.hook).resolve())
    return env


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a target under Weaver daemon, CPython profile hook, and LD_PRELOAD CUDA hook"
    )
    parser.add_argument("--sock", default="/tmp/weaver.sock")
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=18731)
    parser.add_argument("--out", default="./weaver_events.ndjson")
    parser.add_argument("--hook", default=str(_default_hook()))
    parser.add_argument(
        "--collection-mode",
        choices=["adaptive_name", "name_only", "full"],
        default=os.environ.get("WEAVER_COLLECTION_MODE", "adaptive_name"),
        help="adaptive_name records expected kernels by name and times only triggered windows",
    )
    parser.add_argument(
        "--sketch",
        help="manual execution sketch used to derive expected kernel name patterns",
    )
    parser.add_argument(
        "--expected-kernels",
        help="comma-separated extra expected kernel patterns; use exact: or regex: prefixes when needed",
    )
    parser.add_argument(
        "--trigger-capture-after",
        type=int,
        default=int(os.environ.get("WEAVER_TRIGGER_CAPTURE_AFTER", "2")),
        help="number of launches after an unexpected kernel to capture with CUDA Event timing",
    )
    parser.add_argument("--no-daemon", action="store_true", help="use an already running daemon")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="target command after --")
    args = parser.parse_args(argv)

    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        parser.error("missing target command; use: weaver-run -- python train.py")

    hook = Path(args.hook) if args.hook else None
    if hook and not hook.exists():
        raise FileNotFoundError(f"Hook shared object not found: {hook}. Build it with `make -C hooks`.")

    daemon = None
    if not args.no_daemon:
        daemon = _start_daemon(args)
        _wait_for_socket(args.sock)

    try:
        proc = subprocess.Popen(args.cmd, env=_target_env(args))
        return proc.wait()
    finally:
        if daemon is not None:
            daemon.send_signal(signal.SIGTERM)
            try:
                daemon.wait(timeout=3)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait(timeout=3)


if __name__ == "__main__":
    raise SystemExit(main())
