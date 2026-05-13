import argparse
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
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / "examples" / "overhead_workload.py"
HOOK = ROOT / "hooks" / "libweaver_hook.so"

PRESETS = {
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
        "python_sample_rate": 10,
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
        "python_sample_rate": 10,
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
        help="sample every N matched Python operator calls in Weaver Python modes; use 1 for a full Python trace",
    )
    parser.add_argument("--python", default=sys.executable)
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


def mode_env(mode: str, rep: int, run_dir: Path, sock: Optional[Path], python: str, python_sample_rate: int) -> Dict[str, str]:
    env = os.environ.copy()
    prepend_path(env, "PYTHONPATH", str(ROOT))
    env["WEAVER_OVERHEAD_MODE"] = mode
    env["WEAVER_OVERHEAD_REP"] = str(rep)

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
        env.setdefault("WEAVER_TRACE_GC", "0")

    if needs_hook(mode):
        add_preload(env, HOOK)
        env.setdefault("WEAVER_CUDA_EVENTS", "0")
        env.setdefault("WEAVER_CUDA_SYNC_ANCHOR", "0")
        env.setdefault("WEAVER_CUDA_EVENT_POOL", "1")
        env.setdefault("WEAVER_PATCH_DLSYM", "1")
        env.setdefault("WEAVER_PATCH_GETPROC", "1")
        env["WEAVER_TRACE_DIR"] = str(run_dir / "captured_kernels")
        env.setdefault("WEAVER_ENABLE_DISASM", "0")
        env["WEAVER_PYTHON"] = python

    return env


def workload_cmd(args: argparse.Namespace, mode: str, rep: int, out_dir: Path) -> List[str]:
    cmd = [
        args.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(WORKLOAD),
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
    if not path.exists():
        return {"total": 0, "bytes": 0, "by_layer": {}, "by_kind": {}}
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
    return {
        "total": total,
        "bytes": path.stat().st_size,
        "by_layer": dict(layer),
        "by_kind": dict(kind),
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

        env = mode_env(mode, rep, run_dir, sock, args.python, args.python_sample_rate)
        cmd = workload_cmd(args, mode, rep, out_dir)
        log_path = run_dir / "torchrun.log"
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed_s = time.perf_counter() - started
        result = {
            "mode": mode,
            "repetition": rep,
            "returncode": proc.returncode,
            "elapsed_s": elapsed_s,
            "log": str(log_path),
            "weaver_events": count_weaver_events(weaver_file),
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
    layers = Counter()
    kinds = Counter()
    for path in sorted((out_dir / mode).glob("rep_*/run_result.json")):
        data = json.loads(path.read_text())
        ev = data.get("weaver_events", {})
        totals.append(int(ev.get("total", 0)))
        bytes_.append(int(ev.get("bytes", 0)))
        layers.update(ev.get("by_layer", {}))
        kinds.update(ev.get("by_kind", {}))
    return {
        "total_events": sum(totals),
        "total_bytes": sum(bytes_),
        "by_layer": dict(layers),
        "by_kind": dict(kinds),
    }


def percent_delta(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (value - baseline) / baseline * 100.0


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

    summary = {
        "experiment": "weaver_three_layer_collection_overhead",
        "method": {
            "preset": args.preset,
            "baseline": "same dual-GPU workload without daemon, native CPython profile hook, or LD_PRELOAD hook",
            "weaver_full": "daemon + native CPython profile hook + LD_PRELOAD CUDA/NCCL launch hook in low-overhead CPU-enqueue mode",
            "steady_state": "warmup iterations are excluded; CUDA Event timing and disassembly are disabled in the normal low-overhead path",
            "primary_metric": "median host_step_ms overhead vs baseline",
            "secondary_metrics": ["gpu_step_ms", "p95 host_step_ms", "event_count", "event_bytes"],
        },
        "config": vars(args),
        "modes": mode_summary,
        "runs": run_results,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
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
        "",
        "| mode | host median ms | GPU median ms | host overhead | GPU overhead | host p95 ms | Weaver events | Weaver MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "Interpretation:",
        "- The main claim should use `weaver_full` host median overhead versus `baseline`.",
        "- Default Weaver modes use the low-overhead CPU-enqueue CUDA hook; set `WEAVER_CUDA_EVENTS=1` only for deep timing diagnosis.",
        "- `torch_profiler` is a reference diagnostic tool; the normal Weaver path should be below it.",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("summary.json", "summary.md"):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    modes = split_modes(args.modes)

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
    print(json.dumps({"summary": str(out_dir / "summary.json"), "modes": list(summary["modes"].keys())}, ensure_ascii=True))


if __name__ == "__main__":
    main()
