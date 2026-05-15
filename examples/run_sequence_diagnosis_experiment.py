"""Run the two-GPU sequence-sketch diagnosis experiment end to end."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from weaver.collector.launch import _expected_sequence_from_sketch
from weaver.diagnose.normalize import TimelineNormalizer


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = REPO_ROOT / "hooks" / "libweaver_hook.so"
WORKLOAD = REPO_ROOT / "examples" / "sequence_diagnosis_workload.py"


def _payload(event: Dict) -> Dict:
    payload = event.get("payload", {})
    return payload if isinstance(payload, dict) else {}


def _event_rank(event: Dict) -> Optional[int]:
    rank = event.get("rank", _payload(event).get("rank"))
    return int(rank) if rank is not None else None


def _event_time(event: Dict) -> Optional[int]:
    payload = _payload(event)
    for key in ("cpu_enqueue_start_ns", "gpu_start_ns", "start_ns", "ts_ns"):
        value = event.get(key, payload.get(key))
        if value is not None:
            return int(value)
    return None


def _load_events(path: Path) -> List[Dict]:
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def _wait_for_socket(sock: Path, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if sock.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"Weaver daemon socket did not appear: {sock}")


def _tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


def _prepend_env_path(env: Dict[str, str], key: str, value: str) -> None:
    old = env.get(key)
    env[key] = value if not old else f"{value}{os.pathsep}{old}"


def _prepend_preload(env: Dict[str, str], hook: Path) -> None:
    key = "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"
    old = env.get(key)
    env[key] = str(hook) if not old else f"{hook}:{old}"


def _torchrun_cmd(nproc: int) -> List[str]:
    torchrun = shutil.which("torchrun")
    if torchrun:
        return [torchrun, "--standalone", "--nproc_per_node", str(nproc)]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(nproc)]


def _build_hook(skip_build: bool) -> None:
    if skip_build and HOOK_PATH.exists():
        return
    subprocess.run(["make", "-C", str(REPO_ROOT / "hooks")], check=True)


def _start_daemon(trace_path: Path, sock: Path, http_port: int) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "weaver.daemon.server",
        "--sock",
        str(sock),
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(http_port),
        "--out",
        str(trace_path),
    ]
    env = os.environ.copy()
    _prepend_env_path(env, "PYTHONPATH", str(REPO_ROOT))
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _stop_daemon(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _run_under_weaver(
    args: argparse.Namespace,
    run_name: str,
    workload_mode: str,
    collection_mode: str,
    sketch_path: Optional[Path],
    http_port: int,
) -> Path:
    run_dir = args.output_dir / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "weaver_events.ndjson"
    log_path = run_dir / "torchrun.log"
    sock = Path("/tmp") / f"weaver_seq_diag_{os.getpid()}_{run_name}.sock"
    if sock.exists():
        sock.unlink()

    daemon: Optional[subprocess.Popen] = None

    try:
        daemon = _start_daemon(trace_path, sock, http_port)
        _wait_for_socket(sock)

        env = os.environ.copy()
        _prepend_env_path(env, "PYTHONPATH", str(REPO_ROOT))
        _prepend_preload(env, HOOK_PATH)
        env.update(
            {
                "WEAVER_SOCK": str(sock),
                "WEAVER_COLLECTION_MODE": collection_mode,
                "WEAVER_CUDA_EVENTS": "1",
                "WEAVER_CUDA_SYNC_ANCHOR": "1",
                "WEAVER_ASYNC_LAUNCH_EMIT": "1",
                "WEAVER_AUTO_PROFILE": "0",
                "WEAVER_ENABLE_DISASM": "0",
                "WEAVER_PATCH_DLSYM": "1",
                "WEAVER_PATCH_GETPROC": "1",
                "WEAVER_SEQUENCE_REPEAT": "1",
                "WEAVER_TRIGGER_CAPTURE_AFTER": str(args.trigger_capture_after),
            }
        )
        if getattr(args, "single_gpu", False):
            env["RANK"] = "0"
            env["LOCAL_RANK"] = "0"
            env["WORLD_SIZE"] = "1"
        if sketch_path is not None:
            sequence = _expected_sequence_from_sketch(str(sketch_path))
            if sequence:
                env["WEAVER_EXPECTED_SEQUENCE"] = "\n".join(sequence)

        if getattr(args, "single_gpu", False):
            cmd = [sys.executable, str(WORKLOAD), "--single-gpu"]
        else:
            cmd = _torchrun_cmd(args.nproc_per_node) + [str(WORKLOAD)]
        if getattr(args, "compute_only", False):
            cmd.append("--compute-only")
        cmd.extend([
            "--mode",
            workload_mode,
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--dim",
            str(args.dim),
            "--delay-dim",
            str(args.delay_dim),
            "--comm-mb",
            str(args.comm_mb),
            "--copy-mb",
            str(args.copy_mb),
            "--output-dir",
            str(run_dir),
        ])

        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                ret = proc.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise RuntimeError(f"{run_name} timed out; log tail:\n{_tail(log_path)}")

        time.sleep(0.5)

        if ret != 0:
            raise RuntimeError(f"{run_name} failed with exit code {ret}; log tail:\n{_tail(log_path)}")
        return trace_path
    finally:
        _stop_daemon(daemon)
        if sock.exists():
            sock.unlink()


def _first_measured_windows(events: Iterable[Dict]) -> Dict[int, Tuple[int, int]]:
    starts: Dict[int, int] = {}
    windows: Dict[int, Tuple[int, int]] = {}
    for event in events:
        kind = str(event.get("kind", ""))
        payload = _payload(event)
        if kind not in {"iteration_start", "iteration_end"}:
            continue
        if not payload.get("measured", False):
            continue
        rank = _event_rank(event)
        ts = int(payload.get("start_ns") or event.get("ts_ns") or 0)
        if rank is None or ts <= 0:
            continue
        if kind == "iteration_start" and rank not in starts:
            starts[rank] = ts
        elif kind == "iteration_end" and rank in starts and rank not in windows:
            windows[rank] = (starts[rank], ts)
    return windows


def _node_match(kernel_name: str, family: str) -> Dict[str, str]:
    if kernel_name and kernel_name != "<unknown>":
        return {"kernel_name_regex": f"^{re.escape(kernel_name)}$"}
    if family == "GEMM":
        return {"kernel_name_regex": ".*(gemm|matmul|cublas|cutlass).*"}
    if family == "NCCL":
        return {"kernel_name_regex": ".*(nccl|allreduce|all_reduce).*"}
    if family == "MEMCPY":
        return {"kernel_name_regex": ".*(copy|memcpy|transpose|contiguous).*"}
    if family == "REDUCTION":
        return {"kernel_name_regex": ".*(reduce|sum|norm|softmax).*"}
    return {"kernel_name_regex": ".*"}


def _build_sequence_sketch(trace_path: Path, output_dir: Path, include_comm: bool = True) -> Tuple[Path, Path]:
    raw_events = _load_events(trace_path)
    windows = _first_measured_windows(raw_events)
    normalizer = TimelineNormalizer(str(trace_path))
    kernels, operators, _syncs = normalizer.normalize()

    selected = []
    for kernel in kernels:
        if kernel.kernel_name.startswith("nccl::"):
            continue
        rank = kernel.rank
        if rank not in windows:
            continue
        ts = kernel.cpu_enqueue_start_ns or kernel.gpu_start_ns
        if ts is None:
            continue
        start, end = windows[rank]
        if start <= ts <= end:
            selected.append(kernel)

    selected.sort(key=lambda k: (k.rank if k.rank is not None else -1, k.stream or "", k.cpu_enqueue_start_ns or k.gpu_start_ns or 0))
    ordinals = defaultdict(int)
    execution_nodes = []
    for idx, kernel in enumerate(selected):
        rank = kernel.rank if kernel.rank is not None else "*"
        stream = kernel.stream or "default"
        lane = (rank, stream)
        ordinal = ordinals[lane]
        ordinals[lane] += 1
        op_name = kernel.operator_name or "unknown_operator"
        safe_op = re.sub(r"[^A-Za-z0-9_]+", "_", op_name).strip("_") or "op"
        node_id = f"rank{rank}.{stream}.{ordinal:03d}.{safe_op}"
        execution_nodes.append(
            {
                "node_id": node_id,
                "rank": rank,
                "stream": stream,
                "stream_ordinal": ordinal,
                "family": kernel.family,
                "tag": kernel.tag,
                "operator_name": kernel.operator_name,
                "kernel_name": kernel.kernel_name,
                "match": _node_match(kernel.kernel_name, kernel.family),
            }
        )

    ranks = sorted({int(k.rank) for k in selected if k.rank is not None})
    expected_dependencies = []
    for rank in ranks:
        expected_dependencies.append(
            {
                "dependency_id": f"rank{rank}.gemm_A_before_gemm_B",
                "target": {"operator_regex": r"^compute\.gemm_B$", "rank": rank},
                "predecessors": [{"operator_regex": r"^compute\.gemm_A$", "rank": rank}],
                "type": "hard",
                "relation": "immediate_predecessor",
                "description": "GEMM_B should immediately follow GEMM_A on the same stream.",
            }
        )
        if include_comm:
            expected_dependencies.append(
                {
                    "dependency_id": f"rank{rank}.gemm_C_before_nccl_allreduce",
                    "target": {"operator_regex": r"^comm\.nccl_allreduce$", "rank": rank},
                    "predecessors": [{"operator_regex": r"^comm\.gemm_C$", "rank": rank}],
                    "type": "hard",
                    "relation": "immediate_predecessor",
                    "description": "NCCL all-reduce should follow GEMM_C without an inserted wait on the same stream.",
                }
            )

    sketch = {
        "metadata": {
            "schema_version": "0.4",
            "source": "clean_trace_full_collection",
            "mode": "rank_stream_global_sequence",
            "timeline_path": str(trace_path),
            "num_execution_nodes": len(execution_nodes),
            "rank_streams": {f"rank{r}.{s}": count for (r, s), count in ordinals.items()},
            "note": "execution_nodes are used by adaptive kernel-name sequence matching; expected_dependencies are used by diagnosis.",
        },
        "execution_nodes": execution_nodes,
        "expected_dependencies": expected_dependencies,
        "overlap_expectations": [] if not include_comm else [
            {
                "relation_id": "overlap_gemm_memcpy",
                "left_family": "GEMM",
                "right_family": "MEMCPY",
                "phase": "overlap",
                "expected": "may_overlap",
            }
        ],
    }

    sketch_path = output_dir / "expected_sequence_sketch.json"
    sketch_path.write_text(json.dumps(sketch, indent=2, sort_keys=True), encoding="utf-8")
    mermaid_path = output_dir / "expected_sequence.mmd"
    mermaid_path.write_text(_render_mermaid(execution_nodes), encoding="utf-8")
    return sketch_path, mermaid_path


def _render_mermaid(nodes: List[Dict]) -> str:
    by_lane = defaultdict(list)
    for node in nodes:
        by_lane[(node["rank"], node["stream"])].append(node)
    lines = ["flowchart TD"]
    for (rank, stream), lane_nodes in sorted(by_lane.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        lines.append(f"  subgraph rank{rank}_{stream}[rank {rank} / stream {stream}]")
        prev_id = None
        for node in lane_nodes:
            node_id = re.sub(r"[^A-Za-z0-9_]", "_", node["node_id"])
            label = node.get("operator_name") or node.get("family") or "kernel"
            short_kernel = (node.get("kernel_name") or "")[:48]
            lines.append(f"    {node_id}[\"{label}\\n{short_kernel}\"]")
            if prev_id is not None:
                lines.append(f"    {prev_id} --> {node_id}")
            prev_id = node_id
        lines.append("  end")
    return "\n".join(lines) + "\n"


def _count_trace(trace_path: Path) -> Dict:
    by_kind = Counter()
    capture_modes = Counter()
    matched = Counter()
    for event in _load_events(trace_path):
        by_kind[event.get("kind", "unknown")] += 1
        payload = _payload(event)
        if payload.get("capture_mode"):
            capture_modes[payload["capture_mode"]] += 1
        if "matched_expected" in payload:
            matched[str(payload["matched_expected"]).lower()] += 1
    return {
        "by_kind": dict(by_kind),
        "capture_modes": dict(capture_modes),
        "matched_expected": dict(matched),
    }


def _run_diagnosis(args: argparse.Namespace, trace_path: Path, sketch_path: Path, run_dir: Path) -> Path:
    report_path = run_dir / "diagnosis_report.json"
    text_path = run_dir / "diagnosis_report.txt"
    cmd = [
        sys.executable,
        "-m",
        "weaver.diagnose.cli",
        "--timeline",
        str(trace_path),
        "--sketch",
        str(sketch_path),
        "--rank",
        str(args.diagnose_rank),
        "--output",
        str(report_path),
        "--output-text",
        str(text_path),
        "--verbose",
    ]
    env = os.environ.copy()
    _prepend_env_path(env, "PYTHONPATH", str(REPO_ROOT))
    log_path = run_dir / "diagnose.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"diagnosis failed for {trace_path}; log tail:\n{_tail(log_path)}")
    return report_path


def _load_root_causes(report_path: Path) -> List[Dict]:
    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report.get("root_causes", [])


def _validate_expected_diagnosis(mode: str, root_causes: List[Dict]) -> Dict[str, object]:
    """Check whether the report contains the intended pedagogical root cause."""
    if mode == "extra_transpose":
        matches = [
            cause for cause in root_causes
            if (
                cause.get("target_operator") == "compute.gemm_B"
                or (
                    cause.get("target_operator") is None
                    and cause.get("target_family") == "GEMM"
                    and (cause.get("root_cause") or {}).get("operator_name") == "compute.extra_transpose"
                )
            )
            and (cause.get("root_cause") or {}).get("operator_name") == "compute.extra_transpose"
        ]
        return {
            "expected_target_operator": "compute.gemm_B",
            "expected_root_operator": "compute.extra_transpose",
            "passed": bool(matches),
            "matched_root_causes": matches,
        }
    if mode == "wait_event":
        matches = [
            cause for cause in root_causes
            if (
                cause.get("target_operator") == "comm.nccl_allreduce"
                or (
                    cause.get("target_operator") is None
                    and cause.get("target_family") == "NCCL"
                )
            )
            and (
                (cause.get("root_cause") or {}).get("operator_name") == "sync.stream_wait_event"
                or (cause.get("root_cause") or {}).get("name") in {"event_wait", "stream_wait_event"}
                or (cause.get("root_cause") or {}).get("kind") == "sync"
            )
        ]
        return {
            "expected_target_operator": "comm.nccl_allreduce",
            "expected_root_operator": "sync.stream_wait_event",
            "passed": bool(matches),
            "matched_root_causes": matches,
        }
    return {"passed": True, "matched_root_causes": []}


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean-build sketch, inject anomalies, and diagnose.")
    parser.add_argument("--output-dir", type=Path, default=Path("./sequence_diag_2gpu"))
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="run the workload directly on one GPU without torchrun/NCCL",
    )
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="remove communication/overlap phases and diagnose only the GEMM dependency anomaly",
    )
    parser.add_argument(
        "--anomalies",
        help="comma-separated anomaly modes to run; default is extra_transpose for compute-only and both anomalies otherwise",
    )
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--dim", type=int, default=1536)
    parser.add_argument("--delay-dim", type=int, default=2048)
    parser.add_argument("--comm-mb", type=int, default=16)
    parser.add_argument("--copy-mb", type=int, default=16)
    parser.add_argument(
        "--anomaly-collection-mode",
        choices=["selective", "adaptive_name", "name_only", "full"],
        default="selective",
        help="collection mode for injected anomaly diagnosis runs; selective is the low-overhead default",
    )
    parser.add_argument("--trigger-capture-after", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--diagnose-rank", type=int, default=0)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="exit non-zero if the requested injected anomalies are not diagnosed as expected",
    )
    args = parser.parse_args()
    if args.single_gpu:
        args.nproc_per_node = 1
    if args.single_gpu and not args.compute_only:
        raise RuntimeError("--single-gpu requires --compute-only because the communication phases are disabled")
    if args.anomalies:
        anomaly_modes = [item.strip() for item in args.anomalies.split(",") if item.strip()]
    else:
        anomaly_modes = ["extra_transpose"] if args.compute_only else ["extra_transpose", "wait_event"]
    valid_anomalies = {"extra_transpose", "wait_event"}
    unknown = sorted(set(anomaly_modes) - valid_anomalies)
    if unknown:
        raise RuntimeError(f"unknown anomaly modes: {unknown}")
    if args.compute_only and "wait_event" in anomaly_modes:
        raise RuntimeError("compute-only single-GPU experiment only supports extra_transpose")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _build_hook(args.skip_build)

    print(f"[weaver-seq] clean full collection -> {args.output_dir / 'clean_full'}", flush=True)
    clean_trace = _run_under_weaver(args, "clean_full", "clean", "full", None, 19131)
    sketch_path, mermaid_path = _build_sequence_sketch(clean_trace, args.output_dir, include_comm=not args.compute_only)
    print(f"[weaver-seq] sketch: {sketch_path}", flush=True)
    print(f"[weaver-seq] mermaid: {mermaid_path}", flush=True)

    summary = {
        "config": {
            "iters": args.iters,
            "warmup": args.warmup,
            "dim": args.dim,
            "delay_dim": args.delay_dim,
            "comm_mb": args.comm_mb,
            "copy_mb": args.copy_mb,
            "single_gpu": args.single_gpu,
            "compute_only": args.compute_only,
            "anomalies": anomaly_modes,
            "clean_collection_mode": "full",
            "anomaly_collection_mode": args.anomaly_collection_mode,
            "trigger_capture_after": args.trigger_capture_after,
        },
        "clean": {
            "trace": str(clean_trace),
            "sketch": str(sketch_path),
            "mermaid": str(mermaid_path),
            "counts": _count_trace(clean_trace),
        },
        "anomalies": {},
    }

    for idx, mode in enumerate(anomaly_modes):
        run_name = f"anomaly_{mode}"
        print(f"[weaver-seq] diagnosis run: {mode} ({args.anomaly_collection_mode})", flush=True)
        trace = _run_under_weaver(args, run_name, mode, args.anomaly_collection_mode, sketch_path, 19132 + idx)
        run_dir = args.output_dir / run_name
        report = _run_diagnosis(args, trace, sketch_path, run_dir)
        root_causes = _load_root_causes(report)
        validation = _validate_expected_diagnosis(mode, root_causes)
        summary["anomalies"][mode] = {
            "trace": str(trace),
            "report": str(report),
            "text_report": str(run_dir / "diagnosis_report.txt"),
            "counts": _count_trace(trace),
            "root_causes": root_causes,
            "validation": validation,
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[weaver-seq] summary: {summary_path}", flush=True)
    if args.strict_validation:
        failed = [
            mode for mode, item in summary["anomalies"].items()
            if not (item.get("validation") or {}).get("passed")
        ]
        if failed:
            print(f"[weaver-seq] validation failed: {failed}", flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
