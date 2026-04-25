import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from weaver.collector import emit_event, enable_python_collector


PHASES = ("baseline", "overlap", "serialized", "recovery")


@dataclass
class IterRecord:
    phase: str
    intensity: int
    iteration: int
    target_ms: float
    interference_ms: float
    step_ms: float
    overlap_ratio: float
    target_bandwidth_proxy_gbps: float
    interference_bandwidth_proxy_gbps: float
    rank_start_skew_ms: float
    rank_end_skew_ms: float


@dataclass
class WorkloadState:
    gemm_a: torch.Tensor
    gemm_b: torch.Tensor
    memory_a: torch.Tensor
    memory_b: torch.Tensor
    memory_c: torch.Tensor
    comm_t: torch.Tensor
    cache_t: torch.Tensor
    cache_idx: torch.Tensor


def _parse_intensity_list(text: str) -> List[int]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(max(0, int(item)))
    if not vals:
        vals = [0, 1, 2, 4]
    return vals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Weaver controlled resource contention experiment")
    p.add_argument("--target", choices=["gemm", "memory", "nccl", "cache"], required=True)
    p.add_argument("--interference", choices=["sm", "hbm", "link", "l2"], required=True)
    p.add_argument("--intensities", default="0,1,2,4")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--m-size", type=int, default=2048)
    p.add_argument("--vec-size", type=int, default=128 * 1024 * 1024)
    p.add_argument("--comm-size", type=int, default=64 * 1024 * 1024)
    p.add_argument("--cache-mb", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", default="./exp_out")
    p.add_argument("--tag", default="")
    p.add_argument("--sample-rate", type=int, default=100)
    p.add_argument("--weaver-sock", default=os.environ.get("WEAVER_SOCK", "/tmp/weaver.sock"))
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    p.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    return p


def _setup_dist(local_rank: int, world_size: int):
    if dist.is_initialized():
        return
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def _cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def _alloc_state(args) -> WorkloadState:
    torch.manual_seed(args.seed + args.rank)
    device = torch.device("cuda", args.local_rank)
    m = args.m_size
    cache_elems = (args.cache_mb * 1024 * 1024) // 4

    gemm_a = torch.randn(m, m, device=device)
    gemm_b = torch.randn(m, m, device=device)

    vec = args.vec_size
    memory_a = torch.randn(vec, device=device)
    memory_b = torch.randn(vec, device=device)
    memory_c = torch.empty(vec, device=device)

    comm_t = torch.randn(args.comm_size // 4, device=device)
    cache_t = torch.randn(max(cache_elems, 1024), device=device)
    cache_idx = torch.randint(0, cache_t.numel(), (max(cache_elems // 4, 1024),), device=device)

    return WorkloadState(
        gemm_a=gemm_a,
        gemm_b=gemm_b,
        memory_a=memory_a,
        memory_b=memory_b,
        memory_c=memory_c,
        comm_t=comm_t,
        cache_t=cache_t,
        cache_idx=cache_idx,
    )


def _all_reduce_bytes(t: torch.Tensor, world_size: int) -> float:
    per_rank = t.numel() * t.element_size()
    if world_size <= 1:
        return float(per_rank)
    return float(per_rank) * 2.0 * (world_size - 1) / world_size


def _run_target(target: str, state: WorkloadState, intensity: int):
    if target == "gemm":
        out = torch.mm(state.gemm_a, state.gemm_b)
        _ = out[0, 0].item()
        return 0.0
    if target == "memory":
        state.memory_c.copy_(state.memory_a)
        state.memory_c.add_(state.memory_b)
        _ = state.memory_c[0].item()
        return 0.0
    if target == "nccl":
        dist.all_reduce(state.comm_t, op=dist.ReduceOp.SUM)
        return _all_reduce_bytes(state.comm_t, dist.get_world_size() if dist.is_initialized() else 1)
    if target == "cache":
        idx = state.cache_idx
        probe = torch.index_select(state.cache_t, 0, idx)
        if intensity > 0:
            for _ in range(intensity):
                probe = probe + torch.index_select(state.cache_t, 0, idx)
        _ = probe[0].item()
        return 0.0
    raise ValueError(f"unknown target {target}")


def _run_interference(kind: str, state: WorkloadState, intensity: int):
    loops = max(1, intensity)
    if kind == "sm":
        base = state.gemm_a
        acc = base
        for _ in range(loops):
            acc = torch.mm(acc, state.gemm_b)
        _ = acc[0, 0].item()
        return 0.0
    if kind == "hbm":
        for _ in range(loops):
            state.memory_c.copy_(state.memory_a)
            state.memory_c.mul_(1.0001)
            state.memory_c.add_(state.memory_b)
        _ = state.memory_c[0].item()
        return 0.0
    if kind == "link":
        if dist.is_initialized():
            for _ in range(loops):
                dist.all_reduce(state.comm_t, op=dist.ReduceOp.SUM)
            return _all_reduce_bytes(state.comm_t, dist.get_world_size()) * loops
        return 0.0
    if kind == "l2":
        size = state.cache_t.numel()
        shift = min(size - 1, max(256, 4096 * loops))
        idx = (state.cache_idx * 17 + shift) % size
        probe = torch.index_select(state.cache_t, 0, idx)
        for _ in range(loops):
            probe = probe + torch.index_select(state.cache_t, 0, idx)
        _ = probe[0].item()
        return 0.0
    raise ValueError(f"unknown interference {kind}")


def _elapsed_ms(start_evt: torch.cuda.Event, end_evt: torch.cuda.Event) -> float:
    return float(start_evt.elapsed_time(end_evt))


def _gather_skew_ms(start_ns: int, end_ns: int) -> Tuple[float, float]:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return 0.0, 0.0
    starts = [None for _ in range(dist.get_world_size())]
    ends = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(starts, start_ns)
    dist.all_gather_object(ends, end_ns)
    start_skew = (max(starts) - min(starts)) / 1e6
    end_skew = (max(ends) - min(ends)) / 1e6
    return float(start_skew), float(end_skew)


def _iter_once(args, state: WorkloadState, phase: str, intensity: int, iteration: int, target_stream, intf_stream) -> IterRecord:
    target_start = torch.cuda.Event(enable_timing=True)
    target_end = torch.cuda.Event(enable_timing=True)
    intf_start = torch.cuda.Event(enable_timing=True)
    intf_end = torch.cuda.Event(enable_timing=True)

    target_bytes = 0.0
    intf_bytes = 0.0

    step_start_ns = time.time_ns()

    if phase in ("baseline", "recovery"):
        with torch.cuda.stream(target_stream):
            target_start.record(target_stream)
            target_bytes = _run_target(args.target, state, intensity)
            target_end.record(target_stream)
    elif phase == "serialized":
        with torch.cuda.stream(target_stream):
            target_start.record(target_stream)
            target_bytes = _run_target(args.target, state, intensity)
            target_end.record(target_stream)
        with torch.cuda.stream(intf_stream):
            intf_start.record(intf_stream)
            intf_bytes = _run_interference(args.interference, state, intensity)
            intf_end.record(intf_stream)
    elif phase == "overlap":
        with torch.cuda.stream(intf_stream):
            intf_start.record(intf_stream)
            intf_bytes = _run_interference(args.interference, state, intensity)
            intf_end.record(intf_stream)
        with torch.cuda.stream(target_stream):
            target_start.record(target_stream)
            target_bytes = _run_target(args.target, state, intensity)
            target_end.record(target_stream)
    else:
        raise ValueError(f"unknown phase {phase}")

    target_stream.synchronize()
    intf_stream.synchronize()
    torch.cuda.synchronize()
    step_end_ns = time.time_ns()

    target_ms = _elapsed_ms(target_start, target_end)
    interference_ms = _elapsed_ms(intf_start, intf_end) if phase in ("serialized", "overlap") else 0.0
    step_ms = (step_end_ns - step_start_ns) / 1e6

    overlap_ratio = 0.0
    if phase == "overlap" and target_ms > 0.0 and interference_ms > 0.0:
        overlap = max(0.0, target_ms + interference_ms - step_ms)
        overlap_ratio = min(1.0, overlap / min(target_ms, interference_ms))

    start_skew_ms, end_skew_ms = _gather_skew_ms(step_start_ns, step_end_ns)

    target_bw = (target_bytes / (target_ms / 1e3) / 1e9) if target_bytes > 0 and target_ms > 0 else 0.0
    intf_bw = (intf_bytes / (interference_ms / 1e3) / 1e9) if intf_bytes > 0 and interference_ms > 0 else 0.0

    emit_event(
        "iter_metric",
        {
            "phase": phase,
            "intensity": intensity,
            "iter": iteration,
            "target": args.target,
            "interference": args.interference,
            "target_ms": target_ms,
            "interference_ms": interference_ms,
            "step_ms": step_ms,
            "overlap_ratio": overlap_ratio,
            "target_bandwidth_proxy_gbps": target_bw,
            "interference_bandwidth_proxy_gbps": intf_bw,
            "rank_start_skew_ms": start_skew_ms,
            "rank_end_skew_ms": end_skew_ms,
        },
    )

    return IterRecord(
        phase=phase,
        intensity=intensity,
        iteration=iteration,
        target_ms=target_ms,
        interference_ms=interference_ms,
        step_ms=step_ms,
        overlap_ratio=overlap_ratio,
        target_bandwidth_proxy_gbps=target_bw,
        interference_bandwidth_proxy_gbps=intf_bw,
        rank_start_skew_ms=start_skew_ms,
        rank_end_skew_ms=end_skew_ms,
    )


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _mad(values: List[float], med: float) -> float:
    if not values:
        return 0.0
    dev = [abs(v - med) for v in values]
    return float(statistics.median(dev))


def _summarize(records: List[IterRecord]) -> Dict[str, Dict[str, float]]:
    by_key: Dict[str, List[IterRecord]] = {}
    for r in records:
        k = f"{r.phase}|{r.intensity}"
        by_key.setdefault(k, []).append(r)

    out: Dict[str, Dict[str, float]] = {}
    for key, items in by_key.items():
        target_vals = [x.target_ms for x in items]
        overlap_vals = [x.overlap_ratio for x in items]
        step_vals = [x.step_ms for x in items]
        target_bw = [x.target_bandwidth_proxy_gbps for x in items if x.target_bandwidth_proxy_gbps > 0]
        intf_bw = [x.interference_bandwidth_proxy_gbps for x in items if x.interference_bandwidth_proxy_gbps > 0]
        start_skew = [x.rank_start_skew_ms for x in items]
        end_skew = [x.rank_end_skew_ms for x in items]

        med = _median(target_vals)
        out[key] = {
            "count": float(len(items)),
            "target_ms_median": med,
            "target_ms_mad": _mad(target_vals, med),
            "step_ms_median": _median(step_vals),
            "overlap_ratio_median": _median(overlap_vals),
            "target_bandwidth_proxy_gbps_median": _median(target_bw),
            "interference_bandwidth_proxy_gbps_median": _median(intf_bw),
            "rank_start_skew_ms_median": _median(start_skew),
            "rank_end_skew_ms_median": _median(end_skew),
        }
    return out


def _write_outputs(args, records: List[IterRecord], summary: Dict[str, Dict[str, float]]):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.target}_{args.interference}_r{args.rank}"
    if args.tag:
        suffix = f"{args.tag}_{suffix}"

    csv_path = out_dir / f"iter_metrics_{suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "phase",
            "intensity",
            "iteration",
            "target_ms",
            "interference_ms",
            "step_ms",
            "overlap_ratio",
            "target_bandwidth_proxy_gbps",
            "interference_bandwidth_proxy_gbps",
            "rank_start_skew_ms",
            "rank_end_skew_ms",
        ])
        for r in records:
            w.writerow([
                r.phase,
                r.intensity,
                r.iteration,
                f"{r.target_ms:.6f}",
                f"{r.interference_ms:.6f}",
                f"{r.step_ms:.6f}",
                f"{r.overlap_ratio:.6f}",
                f"{r.target_bandwidth_proxy_gbps:.6f}",
                f"{r.interference_bandwidth_proxy_gbps:.6f}",
                f"{r.rank_start_skew_ms:.6f}",
                f"{r.rank_end_skew_ms:.6f}",
            ])

    summary_path = out_dir / f"summary_{suffix}.json"
    data = {
        "target": args.target,
        "interference": args.interference,
        "rank": args.rank,
        "world_size": args.world_size,
        "phases": PHASES,
        "intensities": _parse_intensity_list(args.intensities),
        "summary": summary,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)

    print(json.dumps({"iter_csv": str(csv_path), "summary_json": str(summary_path)}, ensure_ascii=True))


def run_experiment(args, mode: str):
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA GPUs.")

    enable_python_collector(socket_path=args.weaver_sock, sample_rate=max(1, args.sample_rate), include_stdlib=False)

    _setup_dist(args.local_rank, args.world_size)
    args.rank = dist.get_rank() if dist.is_initialized() else args.rank
    args.world_size = dist.get_world_size() if dist.is_initialized() else args.world_size

    torch.manual_seed(args.seed + args.rank)
    state = _alloc_state(args)

    target_stream = torch.cuda.default_stream(device=args.local_rank)
    intf_stream = torch.cuda.Stream(device=args.local_rank)

    intensities = _parse_intensity_list(args.intensities)

    emit_event(
        "exp_start",
        {
            "mode": mode,
            "target": args.target,
            "interference": args.interference,
            "rank": args.rank,
            "world_size": args.world_size,
            "intensities": intensities,
            "warmup": args.warmup,
            "iters": args.iters,
        },
    )

    records: List[IterRecord] = []
    for intensity in intensities:
        for phase in PHASES:
            for i in range(args.warmup):
                _ = _iter_once(args, state, phase, intensity, -1 - i, target_stream, intf_stream)
            for i in range(args.iters):
                records.append(_iter_once(args, state, phase, intensity, i, target_stream, intf_stream))

    summary = _summarize(records)
    _write_outputs(args, records, summary)

    emit_event(
        "exp_done",
        {
            "mode": mode,
            "target": args.target,
            "interference": args.interference,
            "rank": args.rank,
            "records": len(records),
        },
    )

    _cleanup_dist()
