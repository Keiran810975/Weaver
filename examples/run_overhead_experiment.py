import argparse
import gzip
import json
import os
import signal
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / "examples" / "overhead_workload.py"
HOOK = ROOT / "hooks" / "libweaver_hook.so"
RUNNER_VERSION = "selective_kernel_timing_slowdown_v1"

PRESETS = {
    "single_gpu_quick": {
        "output_dir": "./overhead_single_gpu_quick",
        "modes": "baseline,weaver_full,torch_profiler",
        "repeats": 1,
        "nproc_per_node": 1,
        "single_gpu": True,
        "warmup": 5,
        "iters": 20,
        "batch_size": 4,
        "seq_len": 256,
        "dim": 512,
        "hidden_dim": 2048,
        "layers": 3,
        "explicit_comm_mb": 0,
        "profiler_active": 5,
        "python_sample_rate": 1,
        "python_event_budget": 1,
        "collection_mode": "selective",
    },
    "quick": {
        "output_dir": "./overhead_v100_quick",
        "modes": "baseline,weaver_full,torch_profiler",
        "repeats": 1,
        "nproc_per_node": 2,
        "warmup": 5,
        "iters": 20,
        "batch_size": 4,
        "seq_len": 256,
        "dim": 512,
        "hidden_dim": 2048,
        "layers": 3,
        "explicit_comm_mb": 16,
        "profiler_active": 5,
        "python_sample_rate": 1,
        "python_event_budget": 1,
        "collection_mode": "selective",
    },
    "paper": {
        "output_dir": "./overhead_out",
        "modes": "baseline,weaver_full,weaver_no_disasm,torch_profiler",
        "repeats": 3,
        "nproc_per_node": 2,
        "warmup": 20,
        "iters": 100,
        "batch_size": 8,
        "seq_len": 512,
        "dim": 1024,
        "hidden_dim": 4096,
        "layers": 6,
        "explicit_comm_mb": 64,
        "profiler_active": 10,
        "python_sample_rate": 1,
        "python_event_budget": 1,
        "collection_mode": "selective",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Weaver overhead experiment modes")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="quick",
        help="quick is sized for a 2xV100 smoke/overhead run; paper restores the longer experiment",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--modes")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--nproc-per-node", type=int)
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        default=None,
        help="run workload directly on one GPU without torch.distributed/DDP/NCCL",
    )
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--explicit-comm-mb", type=int)
    parser.add_argument("--base-http-port", type=int, default=18770)
    parser.add_argument("--profiler-active", type=int)
    parser.add_argument("--profiler-record-shapes", action="store_true")
    parser.add_argument("--profiler-profile-memory", action="store_true")
    parser.add_argument("--profiler-with-stack", action="store_true")
    parser.add_argument(
        "--python-sample-rate",
        type=int,
        help="sample every N matched Python operator calls in Weaver Python modes",
    )
    parser.add_argument(
        "--python-event-budget",
        type=int,
        help="stop the CPython profile hook after N emitted Python events; use 0 for an unlimited/full Python trace",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--collection-mode",
        choices=["selective", "adaptive_name", "name_only", "full"],
        default=None,
        help="Weaver CUDA hook collection mode used by hook-enabled modes",
    )
    parser.add_argument(
        "--kernel-slowdown-target-mode",
        default="weaver_full",
        help="Weaver mode whose per-kernel CUDA Event timings are compared against the reference trace",
    )
    parser.add_argument(
        "--kernel-slowdown-topk",
        type=int,
        default=50,
        help="maximum number of exact kernel rows written to kernel_slowdown.md",
    )
    parser.add_argument(
        "--trigger-capture-after",
        type=int,
        default=int(os.environ.get("WEAVER_TRIGGER_CAPTURE_AFTER", "2")),
        help="adaptive/selective mode: number of launches after an unexpected kernel to time with CUDA Events",
    )
    parser.add_argument("--skip-hook-build", action="store_true")
    parser.add_argument("--skip-native-build", action="store_true")
    args = parser.parse_args()
    apply_preset_defaults(args)
    return args


def apply_preset_defaults(args: argparse.Namespace) -> None:
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.collection_mode is None:
        args.collection_mode = os.environ.get("WEAVER_COLLECTION_MODE", "selective")
    if args.single_gpu is None:
        args.single_gpu = False


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p95": percentile(values, 95),
        "min": float(min(values)),
        "max": float(max(values)),
        "stdev": float(statistics.pstdev(values)),
    }


def split_modes(text: str) -> List[str]:
    modes = [m.strip() for m in text.split(",") if m.strip()]
    valid = {"baseline", "weaver_full", "weaver_no_disasm", "weaver_kernel_only", "weaver_py_only", "torch_profiler"}
    unknown = sorted(set(modes) - valid)
    if unknown:
        raise ValueError(f"unknown modes: {unknown}; valid={sorted(valid)}")
    return modes


def prepend_path(env: Dict[str, str], key: str, value: str, sep: str = os.pathsep) -> None:
    old = env.get(key)
    env[key] = value if not old else f"{value}{sep}{old}"


def add_preload(env: Dict[str, str], hook: Path) -> None:
    key = "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"
    old = env.get(key)
    env[key] = str(hook) if not old else f"{hook}:{old}"


def build_hook(python: str) -> None:
    subprocess.run(["make", "-C", str(ROOT / "hooks")], check=True)


def build_native_python_trace(python: str) -> None:
    subprocess.run(["make", "-C", str(ROOT / "weaver" / "collector"), f"PYTHON={python}"], check=True)


def native_python_trace_path(python: str) -> Path:
    suffix = subprocess.check_output(
        [python, "-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"],
        text=True,
    ).strip()
    return ROOT / "weaver" / "collector" / f"_native_py_trace{suffix}"


def start_daemon(python: str, sock: Path, out_file: Path, http_port: int) -> subprocess.Popen:
    cmd = [
        python,
        "-m",
        "weaver.daemon.server",
        "--sock",
        str(sock),
        "--http-port",
        str(http_port),
        "--out",
        str(out_file),
    ]
    env = os.environ.copy()
    prepend_path(env, "PYTHONPATH", str(ROOT))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)


def wait_for_socket(sock: Path, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if sock.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"daemon socket not ready: {sock}")


def stop_daemon(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def needs_daemon(mode: str) -> bool:
    return mode.startswith("weaver_")


def needs_hook(mode: str) -> bool:
    return mode in {"weaver_full", "weaver_no_disasm", "weaver_kernel_only"}


def mode_env(
    mode: str,
    rep: int,
    run_dir: Path,
    sock: Optional[Path],
    python: str,
    python_sample_rate: int,
    python_event_budget: int,
    collection_mode: str,
    trigger_capture_after: int,
    single_gpu: bool,
) -> Dict[str, str]:
    env = os.environ.copy()
    prepend_path(env, "PYTHONPATH", str(ROOT))
    env["WEAVER_OVERHEAD_MODE"] = mode
    env["WEAVER_OVERHEAD_REP"] = str(rep)
    if single_gpu:
        env["RANK"] = "0"
        env["LOCAL_RANK"] = "0"
        env["WORLD_SIZE"] = "1"

    # Avoid NCCL init failures observed with LD_PRELOAD hooks on some systems.
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")

    if sock is not None:
        env["WEAVER_SOCK"] = str(sock)

    if mode in {"weaver_full", "weaver_no_disasm", "weaver_py_only"}:
        env["WEAVER_AUTO_PROFILE"] = "1"
        env["WEAVER_PYTHON_COLLECTOR"] = "native"
        env["WEAVER_REQUIRE_NATIVE_PY"] = "1"
        env.setdefault(
            "WEAVER_PYTHON_TRACE_FUNCS",
            "overhead_train_step,OverheadBlock.forward,weaver_overhead_forward,weaver_overhead_backward,weaver_overhead_optimizer",
        )
        env.setdefault("WEAVER_PYTHON_SAMPLE_RATE", str(max(1, python_sample_rate)))
        env.setdefault("WEAVER_PYTHON_EVENT_BUDGET", str(max(0, python_event_budget)))
        env.setdefault("WEAVER_TRACE_GC", "0")

    if needs_hook(mode):
        add_preload(env, HOOK)
        env["WEAVER_COLLECTION_MODE"] = collection_mode
        env.setdefault("WEAVER_SELECTIVE_DROP_LOW_VALUE", "1")
        env["WEAVER_TRIGGER_CAPTURE_AFTER"] = str(max(0, trigger_capture_after))
        env.setdefault("WEAVER_CUDA_EVENTS", "1")
        env.setdefault("WEAVER_CUDA_SYNC_ANCHOR", "1")
        env.setdefault("WEAVER_CUDA_EVENT_POOL", "1")
        env.setdefault("WEAVER_EMIT_CODE_EVENTS", "0")
        env.setdefault("WEAVER_ASYNC_LAUNCH_EMIT", "1")
        env["WEAVER_PATCH_DLSYM"] = "1"
        env.setdefault("WEAVER_PATCH_GETPROC", "1")
        env["WEAVER_TRACE_DIR"] = str(run_dir / "captured_kernels")
        env.setdefault("WEAVER_ENABLE_DISASM", "0")
        env["WEAVER_PYTHON"] = python

    return env


def workload_cmd(args: argparse.Namespace, mode: str, rep: int, out_dir: Path) -> List[str]:
    if args.single_gpu:
        cmd = [args.python, str(WORKLOAD), "--single-gpu"]
    else:
        cmd = [
            args.python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(args.nproc_per_node),
            str(WORKLOAD),
        ]
    cmd.extend(
        [
            "--run-mode",
            mode,
            "--repetition",
            str(rep),
            "--output-dir",
            str(out_dir),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--batch-size",
            str(args.batch_size),
            "--seq-len",
            str(args.seq_len),
            "--dim",
            str(args.dim),
            "--hidden-dim",
            str(args.hidden_dim),
            "--layers",
            str(args.layers),
            "--explicit-comm-mb",
            str(args.explicit_comm_mb),
        ]
    )
    if mode == "torch_profiler":
        cmd.append("--torch-profiler")
        cmd.extend(["--profiler-active", str(args.profiler_active)])
        if args.profiler_record_shapes:
            cmd.append("--profiler-record-shapes")
        if args.profiler_profile_memory:
            cmd.append("--profiler-profile-memory")
        if args.profiler_with_stack:
            cmd.append("--profiler-with-stack")
    return cmd


def read_log_tail(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def count_weaver_events(path: Path) -> Dict[str, object]:
    layer = Counter()
    kind = Counter()
    total = 0
    timed_kernel_launches = 0
    stream_anchor_kernel_launches = 0
    capture_modes = Counter()
    if not path.exists():
        return {
            "total": 0,
            "bytes": 0,
            "by_layer": {},
            "by_kind": {},
            "timed_kernel_launches": 0,
            "stream_anchor_kernel_launches": 0,
            "capture_modes": {},
        }
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            layer[str(event.get("layer", "unknown"))] += 1
            kind[str(event.get("kind", "unknown"))] += 1
            payload = event.get("payload") or {}
            if event.get("kind") == "kernel_launch":
                capture_modes[str(payload.get("capture_mode", "unknown"))] += 1
            if event.get("kind") == "kernel_launch" and payload.get("cuda_event_timing") is True:
                timed_kernel_launches += 1
                if payload.get("time_alignment") == "stream_anchor":
                    stream_anchor_kernel_launches += 1
    return {
        "total": total,
        "bytes": path.stat().st_size,
        "by_layer": dict(layer),
        "by_kind": dict(kind),
        "timed_kernel_launches": timed_kernel_launches,
        "stream_anchor_kernel_launches": stream_anchor_kernel_launches,
        "capture_modes": dict(capture_modes),
    }


def has_measured_step_metrics(out_dir: Path, mode: str, rep: int, expected_ranks: int) -> bool:
    paths = sorted((out_dir / mode / f"rep_{rep}").glob("rank_*/step_metrics.jsonl"))
    if len(paths) < expected_ranks:
        return False
    for path in paths:
        has_measured = False
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("measured"):
                    has_measured = True
                    break
        if not has_measured:
            return False
    return True


def validate_weaver_event_coverage(mode: str, events: Dict[str, object], log_path: Path, args: argparse.Namespace) -> None:
    by_layer = events.get("by_layer") or {}
    by_kind = events.get("by_kind") or {}
    if needs_hook(mode):
        if int(by_layer.get("hook", 0)) <= 0:
            tail = read_log_tail(log_path)
            detail = (
                f"{mode} produced no LD_PRELOAD hook init events; "
                "CUDA/NCCL hook collection is not active"
            )
            if tail:
                detail += f"\n--- torchrun.log tail ---\n{tail}"
            raise RuntimeError(detail)
        if mode in {"weaver_full", "weaver_no_disasm", "weaver_kernel_only"}:
            if int(by_kind.get("kernel_launch", 0)) <= 0:
                tail = read_log_tail(log_path)
                detail = (
                    f"{mode} produced no kernel_launch events; "
                    "ordinary CUDA kernel collection is incomplete"
                )
                if tail:
                    detail += f"\n--- torchrun.log tail ---\n{tail}"
                raise RuntimeError(detail)
            if args.collection_mode in {"full", "selective"} and int(events.get("timed_kernel_launches", 0)) <= 0:
                tail = read_log_tail(log_path)
                detail = (
                    f"{mode} produced no CUDA Event timed kernel_launch events; "
                    "GPU start/end collection is incomplete"
                )
                if tail:
                    detail += f"\n--- torchrun.log tail ---\n{tail}"
                raise RuntimeError(detail)
    if mode in {"weaver_full", "weaver_no_disasm", "weaver_py_only"}:
        if int(by_layer.get("python", 0)) <= 0:
            tail = read_log_tail(log_path)
            detail = f"{mode} produced no Python operator events"
            if tail:
                detail += f"\n--- torchrun.log tail ---\n{tail}"
            raise RuntimeError(detail)


def run_one(args: argparse.Namespace, mode: str, rep: int, out_dir: Path) -> Dict[str, object]:
    run_dir = out_dir / mode / f"rep_{rep}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sock = run_dir / "weaver.sock" if needs_daemon(mode) else None
    weaver_file = run_dir / "weaver_events.ndjson"
    daemon = None
    try:
        if needs_daemon(mode):
            daemon = start_daemon(args.python, sock, weaver_file, args.base_http_port + rep)
            wait_for_socket(sock)

        env = mode_env(
            mode,
            rep,
            run_dir,
            sock,
            args.python,
            args.python_sample_rate,
            args.python_event_budget,
            args.collection_mode,
            args.trigger_capture_after,
            args.single_gpu,
        )
        cmd = workload_cmd(args, mode, rep, out_dir)
        log_path = run_dir / "torchrun.log"
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed_s = time.perf_counter() - started
        weaver_events = count_weaver_events(weaver_file)
        result = {
            "mode": mode,
            "repetition": rep,
            "returncode": proc.returncode,
            "elapsed_s": elapsed_s,
            "log": str(log_path),
            "weaver_events": weaver_events,
        }
        with (run_dir / "run_result.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=True, indent=2)
        if proc.returncode != 0:
            tail = read_log_tail(log_path)
            detail = f"{mode} rep {rep} failed; see {log_path}"
            if tail:
                detail += f"\n--- torchrun.log tail ---\n{tail}"
            raise RuntimeError(detail)
        if not has_measured_step_metrics(out_dir, mode, rep, args.nproc_per_node):
            tail = read_log_tail(log_path)
            detail = f"{mode} rep {rep} produced no measured step metrics; see {log_path}"
            if tail:
                detail += f"\n--- torchrun.log tail ---\n{tail}"
            raise RuntimeError(detail)
        validate_weaver_event_coverage(mode, weaver_events, log_path, args)
        return result
    finally:
        stop_daemon(daemon)


def load_step_values(out_dir: Path, mode: str, field: str) -> List[float]:
    values: List[float] = []
    for path in sorted((out_dir / mode).glob("rep_*/rank_*/step_metrics.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("measured"):
                    values.append(float(row[field]))
    return values


def load_event_counts(out_dir: Path, mode: str) -> Dict[str, object]:
    totals = []
    bytes_ = []
    timed_kernel_launches = []
    stream_anchor_kernel_launches = []
    layers = Counter()
    kinds = Counter()
    capture_modes = Counter()
    for path in sorted((out_dir / mode).glob("rep_*/run_result.json")):
        data = json.loads(path.read_text())
        ev = data.get("weaver_events", {})
        totals.append(int(ev.get("total", 0)))
        bytes_.append(int(ev.get("bytes", 0)))
        timed_kernel_launches.append(int(ev.get("timed_kernel_launches", 0)))
        stream_anchor_kernel_launches.append(int(ev.get("stream_anchor_kernel_launches", 0)))
        layers.update(ev.get("by_layer", {}))
        kinds.update(ev.get("by_kind", {}))
        capture_modes.update(ev.get("capture_modes", {}))
    return {
        "total_events": sum(totals),
        "total_bytes": sum(bytes_),
        "timed_kernel_launches": sum(timed_kernel_launches),
        "stream_anchor_kernel_launches": sum(stream_anchor_kernel_launches),
        "capture_modes": dict(capture_modes),
        "by_layer": dict(layers),
        "by_kind": dict(kinds),
    }


def percent_delta(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (value - baseline) / baseline * 100.0


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def canonical_kernel_name(value: object) -> str:
    text = str(value or "<unknown>").strip()
    return text if text else "<unknown>"


def classify_kernel_family(kernel_name: str) -> str:
    name = kernel_name.lower()
    if "nccl" in name or "allreduce" in name or "all_reduce" in name:
        return "Communication"
    if (
        "gemm" in name
        or "sgemm" in name
        or "hgemm" in name
        or "cublas" in name
        or "matmul" in name
        or "cutlass" in name
    ):
        return "Compute-Matrix"
    if (
        "memcpy" in name
        or "copy" in name
        or "transpose" in name
        or "permute" in name
        or "contiguous" in name
        or "cat" in name
    ):
        return "Memory-Copy/Layout"
    if (
        "reduce" in name
        or "reduction" in name
        or "softmax" in name
        or "norm" in name
        or "sum" in name
        or "layer_norm" in name
    ):
        return "Reduction"
    return "Other/Unknown"


def load_weaver_kernel_durations_us(out_dir: Path, mode: str) -> Dict[str, List[float]]:
    durations: Dict[str, List[float]] = {}
    for path in sorted((out_dir / mode).glob("rep_*/weaver_events.ndjson")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("kind") != "kernel_launch":
                    continue
                payload = event.get("payload") or {}
                if payload.get("cuda_event_timing") is not True:
                    continue
                duration_ns = safe_float(payload.get("gpu_duration_ns"))
                if duration_ns is None:
                    duration_ns = safe_float(event.get("dur_ns"))
                if duration_ns is None or duration_ns <= 0:
                    continue
                name = canonical_kernel_name(
                    event.get("kernel_name") or payload.get("kernel_name") or payload.get("kernel")
                )
                durations.setdefault(name, []).append(duration_ns / 1000.0)
    return durations


def torch_profiler_trace_paths(out_dir: Path, mode: str) -> List[Path]:
    patterns = [
        f"{mode}/rep_*/rank_*/torch_profiler/rank_*/*.pt.trace.json",
        f"{mode}/rep_*/rank_*/torch_profiler/rank_*/*.pt.trace.json.gz",
        f"{mode}/rep_*/rank_*/torch_profiler/rank_*/*.json",
        f"{mode}/rep_*/rank_*/torch_profiler/rank_*/*.json.gz",
    ]
    seen = set()
    paths: List[Path] = []
    for pattern in patterns:
        for path in sorted(out_dir.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def load_trace_json(path: Path) -> Dict[str, object]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def is_torch_profiler_kernel_event(event: Dict[str, object]) -> bool:
    if event.get("ph") not in (None, "X"):
        return False
    category = str(event.get("cat", "")).lower()
    if "kernel" not in category:
        return False
    duration_us = safe_float(event.get("dur"))
    if duration_us is None or duration_us <= 0:
        return False
    name = canonical_kernel_name(event.get("name"))
    if name == "<unknown>":
        return False
    return True


def load_torch_profiler_kernel_durations_us(out_dir: Path, mode: str = "torch_profiler") -> Tuple[Dict[str, List[float]], List[str]]:
    durations: Dict[str, List[float]] = {}
    warnings: List[str] = []
    paths = torch_profiler_trace_paths(out_dir, mode)
    if not paths:
        return durations, [f"no torch profiler trace files found under {out_dir / mode}"]
    for path in paths:
        try:
            trace = load_trace_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"failed to read {path}: {exc}")
            continue
        events = trace.get("traceEvents")
        if not isinstance(events, list):
            warnings.append(f"{path} has no traceEvents array")
            continue
        for event in events:
            if not isinstance(event, dict) or not is_torch_profiler_kernel_event(event):
                continue
            duration_us = safe_float(event.get("dur"))
            if duration_us is None:
                continue
            name = canonical_kernel_name(event.get("name"))
            durations.setdefault(name, []).append(duration_us)
    return durations, warnings


def merge_by_family(durations_by_name: Dict[str, List[float]]) -> Dict[str, List[float]]:
    by_family: Dict[str, List[float]] = {}
    for name, values in durations_by_name.items():
        by_family.setdefault(classify_kernel_family(name), []).extend(values)
    return by_family


def slowdown_severity(slowdown_pct: Optional[float]) -> str:
    if slowdown_pct is None:
        return "unavailable"
    if slowdown_pct < -5.0:
        return "faster"
    if slowdown_pct <= 5.0:
        return "low"
    if slowdown_pct <= 15.0:
        return "moderate"
    return "high"


def duration_comparison_row(
    name: str,
    before_values: Optional[List[float]],
    after_values: Optional[List[float]],
    name_field: str,
) -> Dict[str, object]:
    before_values = before_values or []
    after_values = after_values or []
    before = summarize(before_values)
    after = summarize(after_values)
    before_median = before["median"]
    after_median = after["median"]
    available = bool(before_values) and bool(after_values) and before_median > 0
    slowdown_pct: Optional[float]
    slowdown_factor: Optional[float]
    if available:
        slowdown_pct = percent_delta(after_median, before_median)
        slowdown_factor = after_median / before_median
        delta_us: Optional[float] = after_median - before_median
    else:
        slowdown_pct = None
        slowdown_factor = None
        delta_us = None
    row = {
        name_field: name,
        "family": classify_kernel_family(name) if name_field == "kernel_name" else name,
        "comparison_available": available,
        "before_gpu_us": before,
        "after_weaver_gpu_us": after,
        "delta_gpu_us_median": delta_us,
        "before_count": len(before_values),
        "after_count": len(after_values),
        "slowdown_pct_median": slowdown_pct,
        "slowdown_factor_median": slowdown_factor,
        "severity": slowdown_severity(slowdown_pct),
    }
    if not available:
        if not before_values:
            row["unavailable_reason"] = "missing_reference_kernel_duration"
        elif not after_values:
            row["unavailable_reason"] = "missing_weaver_kernel_duration"
        else:
            row["unavailable_reason"] = "zero_reference_duration"
    return row


def build_kernel_slowdown_summary(out_dir: Path, modes: List[str], args: argparse.Namespace) -> Dict[str, object]:
    target_mode = args.kernel_slowdown_target_mode
    reference_mode = "torch_profiler" if "torch_profiler" in modes else ""
    result: Dict[str, object] = {
        "available": False,
        "reference_mode": reference_mode,
        "reference_source": "torch_profiler_cuda_kernel_trace" if reference_mode else "",
        "target_mode": target_mode,
        "target_source": "weaver_cuda_event_timed_kernel_launch",
        "unit": "microseconds",
        "note": (
            "baseline mode has no per-kernel durations; step-level overhead is still measured "
            "against baseline, while per-kernel slowdown uses the torch_profiler CUDA kernel "
            "trace as the before/reference source."
        ),
        "warnings": [],
        "by_kernel": [],
        "by_family": [],
    }
    if target_mode not in modes:
        result["reason"] = f"target mode {target_mode!r} was not run"
        return result
    if not reference_mode:
        result["reason"] = "torch_profiler mode was not run, so no per-kernel reference durations are available"
        return result

    reference, warnings = load_torch_profiler_kernel_durations_us(out_dir, reference_mode)
    target = load_weaver_kernel_durations_us(out_dir, target_mode)
    result["warnings"] = warnings
    result["reference_kernel_count"] = len(reference)
    result["reference_sample_count"] = sum(len(v) for v in reference.values())
    result["target_kernel_count"] = len(target)
    result["target_sample_count"] = sum(len(v) for v in target.values())
    if not reference:
        result["reason"] = "no CUDA kernel durations were found in the torch_profiler trace"
        return result
    if not target:
        result["reason"] = f"no CUDA Event timed kernel_launch events were found in {target_mode}"
        return result

    names = sorted(target.keys())
    kernel_rows = [
        duration_comparison_row(name, reference.get(name), target.get(name), "kernel_name")
        for name in names
    ]
    family_names = sorted(set(merge_by_family(reference)) | set(merge_by_family(target)))
    reference_family = merge_by_family(reference)
    target_family = merge_by_family(target)
    family_rows = [
        duration_comparison_row(name, reference_family.get(name), target_family.get(name), "family")
        for name in family_names
    ]
    available_rows = [r for r in kernel_rows if r["comparison_available"]]
    result.update(
        {
            "available": bool(available_rows),
            "matched_kernel_count": len(available_rows),
            "unmatched_target_kernel_count": len(kernel_rows) - len(available_rows),
            "by_kernel": sorted(
                kernel_rows,
                key=lambda row: (
                    row["comparison_available"] is not True,
                    -(row["slowdown_pct_median"] or 0.0),
                    str(row.get("kernel_name", "")),
                ),
            ),
            "by_family": sorted(
                family_rows,
                key=lambda row: (
                    row["comparison_available"] is not True,
                    -(row["slowdown_pct_median"] or 0.0),
                    str(row.get("family", "")),
                ),
            ),
        }
    )
    if not available_rows:
        result["reason"] = (
            "Weaver and torch_profiler both produced kernel durations, but exact kernel names did not match; "
            "use by_family as a coarse fallback and check kernel name normalization."
        )
    return result


def write_kernel_slowdown_files(out_dir: Path, slowdown: Dict[str, object], topk: int) -> None:
    with (out_dir / "kernel_slowdown.json").open("w", encoding="utf-8") as f:
        json.dump(slowdown, f, ensure_ascii=True, indent=2)

    kernel_rows = slowdown.get("by_kernel") or []
    family_rows = slowdown.get("by_family") or []
    lines = [
        "# Weaver Per-Kernel Slowdown",
        "",
        f"Reference: `{slowdown.get('reference_mode') or 'unavailable'}` ({slowdown.get('reference_source') or 'none'})",
        f"Target: `{slowdown.get('target_mode')}` ({slowdown.get('target_source')})",
        "",
        str(slowdown.get("note", "")),
        "",
    ]
    if not slowdown.get("available"):
        lines.extend([
            f"Status: unavailable. Reason: {slowdown.get('reason', 'unknown')}",
            "",
        ])
    lines.extend([
        "## By Family",
        "",
        "| family | before median us | after median us | delta us | slowdown | before n | after n | severity |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in family_rows:
        slowdown_pct = row.get("slowdown_pct_median")
        slowdown_text = "n/a" if slowdown_pct is None else f"{slowdown_pct:+.2f}%"
        delta = row.get("delta_gpu_us_median")
        delta_text = "n/a" if delta is None else f"{delta:+.3f}"
        lines.append(
            "| {family} | {before:.3f} | {after:.3f} | {delta} | {slowdown} | {before_n} | {after_n} | {severity} |".format(
                family=row.get("family", ""),
                before=row["before_gpu_us"]["median"],
                after=row["after_weaver_gpu_us"]["median"],
                delta=delta_text,
                slowdown=slowdown_text,
                before_n=row.get("before_count", 0),
                after_n=row.get("after_count", 0),
                severity=row.get("severity", ""),
            )
        )
    lines.extend([
        "",
        f"## By Exact Kernel (top {max(0, topk)})",
        "",
        "| kernel | family | before median us | after median us | delta us | slowdown | before n | after n | severity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in kernel_rows[: max(0, topk)]:
        slowdown_pct = row.get("slowdown_pct_median")
        slowdown_text = "n/a" if slowdown_pct is None else f"{slowdown_pct:+.2f}%"
        delta = row.get("delta_gpu_us_median")
        delta_text = "n/a" if delta is None else f"{delta:+.3f}"
        lines.append(
            "| {kernel} | {family} | {before:.3f} | {after:.3f} | {delta} | {slowdown} | {before_n} | {after_n} | {severity} |".format(
                kernel=str(row.get("kernel_name", "")).replace("|", "\\|"),
                family=row.get("family", ""),
                before=row["before_gpu_us"]["median"],
                after=row["after_weaver_gpu_us"]["median"],
                delta=delta_text,
                slowdown=slowdown_text,
                before_n=row.get("before_count", 0),
                after_n=row.get("after_count", 0),
                severity=row.get("severity", ""),
            )
        )
    if slowdown.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in slowdown["warnings"])
    (out_dir / "kernel_slowdown.md").write_text("\n".join(lines), encoding="utf-8")


def build_summary(out_dir: Path, modes: List[str], args: argparse.Namespace, run_results: List[Dict[str, object]]) -> Dict[str, object]:
    mode_summary: Dict[str, object] = {}
    for mode in modes:
        host_values = load_step_values(out_dir, mode, "host_step_ms")
        gpu_values = load_step_values(out_dir, mode, "gpu_step_ms")
        forward_values = load_step_values(out_dir, mode, "forward_ms")
        backward_values = load_step_values(out_dir, mode, "backward_ms")
        comm_values = load_step_values(out_dir, mode, "explicit_comm_ms")
        if not host_values or not gpu_values:
            raise RuntimeError(f"{mode} has no measured step metrics; refusing to write a zero-valued summary")
        host = summarize(host_values)
        gpu = summarize(gpu_values)
        forward = summarize(forward_values)
        backward = summarize(backward_values)
        comm = summarize(comm_values)
        mode_summary[mode] = {
            "host_step_ms": host,
            "gpu_step_ms": gpu,
            "forward_ms": forward,
            "backward_ms": backward,
            "explicit_comm_ms": comm,
            "weaver_events": load_event_counts(out_dir, mode),
        }

    baseline_host = mode_summary.get("baseline", {}).get("host_step_ms", {}).get("median", 0.0)
    baseline_gpu = mode_summary.get("baseline", {}).get("gpu_step_ms", {}).get("median", 0.0)
    for mode, item in mode_summary.items():
        host_median = item["host_step_ms"]["median"]
        gpu_median = item["gpu_step_ms"]["median"]
        item["overhead_vs_baseline"] = {
            "host_step_median_pct": percent_delta(host_median, baseline_host),
            "gpu_step_median_pct": percent_delta(gpu_median, baseline_gpu),
        }

    kernel_slowdown = build_kernel_slowdown_summary(out_dir, modes, args)
    summary = {
        "experiment": "weaver_three_layer_collection_overhead",
        "runner_version": RUNNER_VERSION,
        "runner_path": str(Path(__file__).resolve()),
        "method": {
            "preset": args.preset,
            "baseline": "same workload without daemon, native CPython profile hook, or LD_PRELOAD hook",
            "weaver_full": "daemon + native CPython profile hook + LD_PRELOAD CUDA/NCCL launch hook; the default selective CUDA hook records GPU start/end for GEMM/NCCL/sync, records metadata-useful kernels by name only, and drops very low-value launches",
            "steady_state": "warmup iterations are excluded from step-level overhead; kernel slowdown uses timed kernel_launch events written by the hook",
            "primary_metric": "median host_step_ms overhead vs baseline",
            "secondary_metrics": ["gpu_step_ms", "p95 host_step_ms", "event_count", "event_bytes", "per_kernel_gpu_duration_slowdown"],
            "per_kernel_slowdown": (
                "baseline does not expose per-kernel durations, so exact per-kernel before/after "
                "comparison uses torch_profiler CUDA kernel trace as the before/reference source "
                "and Weaver CUDA Event timed kernel_launch events as the after/source under collection"
            ),
        },
        "config": vars(args),
        "modes": mode_summary,
        "kernel_slowdown_vs_reference": kernel_slowdown,
        "runs": run_results,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    write_kernel_slowdown_files(out_dir, kernel_slowdown, args.kernel_slowdown_topk)
    write_markdown(out_dir / "summary.md", summary, modes)
    return summary


def write_markdown(path: Path, summary: Dict[str, object], modes: List[str]) -> None:
    rows = []
    for mode in modes:
        item = summary["modes"][mode]
        rows.append(
            "| {mode} | {host:.3f} | {gpu:.3f} | {host_oh:+.2f}% | {gpu_oh:+.2f}% | {p95:.3f} | {events} | {mb:.2f} |".format(
                mode=mode,
                host=item["host_step_ms"]["median"],
                gpu=item["gpu_step_ms"]["median"],
                host_oh=item["overhead_vs_baseline"]["host_step_median_pct"],
                gpu_oh=item["overhead_vs_baseline"]["gpu_step_median_pct"],
                p95=item["host_step_ms"]["p95"],
                events=item["weaver_events"]["total_events"],
                mb=item["weaver_events"]["total_bytes"] / (1024 * 1024),
            )
        )

    text = [
        "# Weaver Overhead Experiment",
        "",
        "Warmup iterations are excluded from the steady-state overhead calculation.",
        f"CUDA collection mode: `{summary['config'].get('collection_mode')}`.",
        "",
        "| mode | host median ms | GPU median ms | host overhead | GPU overhead | host p95 ms | Weaver events | Weaver MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "Interpretation:",
        "- The main claim should use `weaver_full` host median overhead versus `baseline`.",
        "- Default Weaver modes use selective CUDA Event timing: GEMM/NCCL/sync are timed, metadata-useful kernels are name-only, and very low-value high-frequency kernels are dropped.",
        "- Per-kernel slowdown is written to `kernel_slowdown.json` and `kernel_slowdown.md`.",
        "- `torch_profiler` is a reference diagnostic tool; the normal Weaver path should be below it.",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("summary.json", "summary.md", "kernel_slowdown.json", "kernel_slowdown.md"):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    modes = split_modes(args.modes)
    print(
        f"[weaver-overhead] runner={RUNNER_VERSION} script={Path(__file__).resolve()} output_dir={out_dir}",
        flush=True,
    )

    if any(needs_hook(mode) for mode in modes):
        if not args.skip_hook_build:
            build_hook(args.python)
        if not HOOK.exists():
            raise FileNotFoundError(f"missing hook library: {HOOK}")
    if any(mode in {"weaver_full", "weaver_no_disasm", "weaver_py_only"} for mode in modes):
        if not args.skip_native_build:
            build_native_python_trace(args.python)
        native_ext = native_python_trace_path(args.python)
        if not native_ext.exists():
            raise FileNotFoundError(f"missing native Python tracing extension: {native_ext}")

    run_results: List[Dict[str, object]] = []
    for rep in range(args.repeats):
        for mode in modes:
            print(f"[weaver-overhead] running mode={mode} rep={rep}", flush=True)
            run_results.append(run_one(args, mode, rep, out_dir))

    summary = build_summary(out_dir, modes, args, run_results)
    print(
        json.dumps(
            {
                "summary": str(out_dir / "summary.json"),
                "kernel_slowdown": str(out_dir / "kernel_slowdown.json"),
                "modes": list(summary["modes"].keys()),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
