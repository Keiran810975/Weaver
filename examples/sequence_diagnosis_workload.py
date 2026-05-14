"""GPU workload for sequence-sketch diagnosis experiments.

The workload is intentionally small and explicit:
GEMM_A -> GEMM_B -> reduction/layout, GEMM_C -> NCCL all-reduce,
then an overlap phase with a GEMM and a copy/layout operation.
The single-GPU compute-only mode keeps only GEMM_A -> GEMM_B -> reduction/layout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator

import torch
import torch.distributed as dist

from weaver.collector import emit_event


def _rank_env(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None and value != "" else default


@contextmanager
def operator_scope(name: str, phase: str, iteration: int, rank: int, extra: Dict | None = None) -> Iterator[None]:
    start_ns = time.time_ns()
    try:
        yield
    finally:
        end_ns = time.time_ns()
        payload = {
            "operator_name": name,
            "phase": phase,
            "iteration": iteration,
            "rank": rank,
            "start_ns": start_ns,
            "end_ns": end_ns,
        }
        if extra:
            payload.update(extra)
        emit_event("operator", payload, layer="operator")


def emit_iteration(kind: str, iteration: int, rank: int, measured: bool) -> None:
    emit_event(
        kind,
        {
            "iteration": iteration,
            "rank": rank,
            "measured": measured,
            "start_ns": time.time_ns(),
        },
        layer="runtime",
    )


def mb_to_numel(mb: int, dtype: torch.dtype = torch.float32) -> int:
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    return max(1, int(mb * 1024 * 1024 / bytes_per_elem))


def make_inputs(args: argparse.Namespace, device: torch.device) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(args.seed + _rank_env("RANK", 0))
    dim = args.dim
    delay_dim = args.delay_dim
    copy_side = max(256, int(math.sqrt(mb_to_numel(args.copy_mb))))

    return {
        "x_a": torch.randn(dim, dim, device=device, generator=g),
        "w_a": torch.randn(dim, dim, device=device, generator=g),
        "w_b": torch.randn(dim, dim, device=device, generator=g),
        "x_c": torch.randn(dim, dim, device=device, generator=g),
        "w_c": torch.randn(dim, dim, device=device, generator=g),
        "overlap_x": torch.randn(dim, dim, device=device, generator=g),
        "overlap_w": torch.randn(dim, dim, device=device, generator=g),
        "comm": torch.randn(mb_to_numel(args.comm_mb), device=device, generator=g),
        "copy_src": torch.randn(copy_side, copy_side, device=device, generator=g),
        "delay_x": torch.randn(delay_dim, delay_dim, device=device, generator=g),
        "delay_w": torch.randn(delay_dim, delay_dim, device=device, generator=g),
    }


def run_iteration(
    args: argparse.Namespace,
    tensors: Dict[str, torch.Tensor],
    copy_stream: torch.cuda.Stream,
    delay_stream: torch.cuda.Stream,
    iteration: int,
    rank: int,
) -> None:
    mode = args.mode

    with operator_scope("compute.gemm_A", "compute", iteration, rank, {"M": args.dim, "N": args.dim, "K": args.dim}):
        a = torch.mm(tensors["x_a"], tensors["w_a"])

    if mode == "extra_transpose":
        with operator_scope("compute.extra_transpose", "compute", iteration, rank, {"anomaly": "extra_transpose"}):
            extra = a.transpose(0, 1).contiguous()
            tensors["_extra_transpose"] = extra

    with operator_scope("compute.gemm_B", "compute", iteration, rank, {"M": args.dim, "N": args.dim, "K": args.dim}):
        b = torch.mm(a, tensors["w_b"])

    with operator_scope("compute.reduction_layout", "compute", iteration, rank):
        reduced = b.sum(dim=1)
        layout = reduced.contiguous()

    if args.compute_only:
        tensors["_last_scalar"] = layout[0]
        return

    with operator_scope("comm.gemm_C", "comm", iteration, rank, {"M": args.dim, "N": args.dim, "K": args.dim}):
        c = torch.mm(tensors["x_c"], tensors["w_c"])

    if mode == "wait_event":
        delay_event = torch.cuda.Event(enable_timing=False)
        with torch.cuda.stream(delay_stream):
            with operator_scope(
                "sync.delay_gemm",
                "sync",
                iteration,
                rank,
                {"anomaly": "wait_event_source", "M": args.delay_dim, "N": args.delay_dim, "K": args.delay_dim},
            ):
                delay = torch.mm(tensors["delay_x"], tensors["delay_w"])
            delay_event.record(delay_stream)

        with operator_scope("sync.stream_wait_event", "sync", iteration, rank, {"anomaly": "wait_event"}):
            torch.cuda.current_stream().wait_event(delay_event)

    with operator_scope("comm.nccl_allreduce", "comm", iteration, rank, {"bytes": args.comm_mb * 1024 * 1024}):
        dist.all_reduce(tensors["comm"], op=dist.ReduceOp.SUM)

    with torch.cuda.stream(copy_stream):
        with operator_scope("overlap.target_memcpy", "overlap", iteration, rank, {"bytes": args.copy_mb * 1024 * 1024}):
            copied = tensors["copy_src"].transpose(0, 1).contiguous()

    with operator_scope("overlap.target_gemm", "overlap", iteration, rank, {"M": args.dim, "N": args.dim, "K": args.dim}):
        target = torch.mm(tensors["overlap_x"], tensors["overlap_w"])

    torch.cuda.current_stream().wait_stream(copy_stream)
    # Keep Python references live until the iteration is enqueued.
    tensors["_last_scalar"] = layout[0] + target[0, 0] + copied[0, 0] + c[0, 0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Weaver sequence diagnosis workload")
    parser.add_argument("--mode", choices=["clean", "extra_transpose", "wait_event"], default="clean")
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="run without torch.distributed/NCCL; intended for compute-only dependency diagnosis",
    )
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="skip GEMM_C, NCCL, wait_event, and overlap phases",
    )
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--dim", type=int, default=1536)
    parser.add_argument("--delay-dim", type=int, default=2048)
    parser.add_argument("--comm-mb", type=int, default=16)
    parser.add_argument("--copy-mb", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    if args.single_gpu and not args.compute_only:
        raise RuntimeError("--single-gpu requires --compute-only because NCCL phases are disabled")
    if args.compute_only and args.mode == "wait_event":
        raise RuntimeError("--compute-only only supports clean and extra_transpose modes")

    rank = _rank_env("RANK", 0)
    local_rank = _rank_env("LOCAL_RANK", rank)
    world_size = _rank_env("WORLD_SIZE", 1)
    if args.single_gpu:
        rank = 0
        local_rank = 0
        world_size = 1
    elif world_size != 2:
        raise RuntimeError(f"this experiment expects 2 ranks, got WORLD_SIZE={world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not args.single_gpu:
        dist.init_process_group(backend="nccl")

    torch.manual_seed(args.seed + rank)
    tensors = make_inputs(args, device)
    copy_stream = torch.cuda.Stream(device=device)
    delay_stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_rank{rank}.jsonl"

    total_iters = args.warmup + args.iters
    with metrics_path.open("w", encoding="utf-8") as metrics:
        for iteration in range(total_iters):
            measured = iteration >= args.warmup
            emit_iteration("iteration_start", iteration, rank, measured)
            start = time.perf_counter_ns()
            with torch.no_grad():
                run_iteration(args, tensors, copy_stream, delay_stream, iteration, rank)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            emit_iteration("iteration_end", iteration, rank, measured)
            metrics.write(
                json.dumps(
                    {
                        "rank": rank,
                        "iteration": iteration,
                        "measured": measured,
                        "mode": args.mode,
                        "step_ns": end - start,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            metrics.flush()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
