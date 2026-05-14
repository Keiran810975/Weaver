import argparse
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StepMetric:
    run_mode: str
    repetition: int
    rank: int
    local_rank: int
    world_size: int
    iteration: int
    measured: bool
    profiled: bool
    host_step_ms: float
    gpu_step_ms: float
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    explicit_comm_ms: float
    loss: float
    tokens: int
    explicit_comm_bytes: int


class OverheadBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ln2 = nn.LayerNorm(dim)
        self.fc3 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc4 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        x = x + h
        h = self.ln2(x)
        h = self.fc4(F.silu(self.fc3(h)))
        return x + h


class OverheadModel(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([OverheadBlock(dim, hidden_dim) for _ in range(layers)])
        self.head = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU workload for Weaver overhead experiments")
    parser.add_argument("--run-mode", default=os.environ.get("WEAVER_OVERHEAD_MODE", "baseline"))
    parser.add_argument("--repetition", type=int, default=int(os.environ.get("WEAVER_OVERHEAD_REP", "0")))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--explicit-comm-mb", type=int, default=16)
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="run without torch.distributed/DDP/NCCL; used by the single-card overhead preset",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--torch-profiler", action="store_true")
    parser.add_argument("--profiler-wait", type=int)
    parser.add_argument("--profiler-warmup", type=int, default=3)
    parser.add_argument("--profiler-active", type=int, default=5)
    parser.add_argument("--profiler-record-shapes", action="store_true")
    parser.add_argument("--profiler-profile-memory", action="store_true")
    parser.add_argument("--profiler-with-stack", action="store_true")
    return parser.parse_args()


def normalize_profiler_args(args: argparse.Namespace) -> None:
    if args.profiler_wait is None:
        args.profiler_wait = max(0, args.warmup - max(0, args.profiler_warmup))


def is_profiler_active_step(args: argparse.Namespace, iteration: int) -> bool:
    if not args.torch_profiler:
        return False
    active_start = max(0, args.profiler_wait) + max(0, args.profiler_warmup)
    active_end = active_start + max(1, args.profiler_active)
    return active_start <= iteration < active_end


def setup_runtime(args: argparse.Namespace) -> None:
    if dist.is_initialized():
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This overhead workload is intentionally GPU-only")
    if args.single_gpu:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        return
    if "RANK" not in os.environ:
        raise RuntimeError("Run this script with torchrun so RANK/WORLD_SIZE are set")
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", 1))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
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


def event_elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def timed_region(fn, stream: torch.cuda.Stream) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    fn()
    end.record(stream)
    stream.synchronize()
    return event_elapsed(start, end)


def make_inputs(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return torch.randn(args.batch_size, args.seq_len, args.dim, device=device)


def make_targets(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return torch.randn(args.batch_size, args.seq_len, args.dim, device=device)


def explicit_all_reduce(buffer: Optional[torch.Tensor]) -> int:
    if buffer is None or world_size() <= 1 or not dist.is_initialized():
        return 0
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    per_rank = buffer.numel() * buffer.element_size()
    return int(per_rank * 2 * (world_size() - 1) / world_size())


def overhead_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    comm_buffer: Optional[torch.Tensor],
) -> Dict[str, float]:
    stream = torch.cuda.current_stream()
    optimizer.zero_grad(set_to_none=True)

    holder: Dict[str, torch.Tensor] = {}

    def _forward():
        with torch.profiler.record_function("weaver_overhead_forward"):
            out = model(x)
            holder["loss"] = F.mse_loss(out, y)

    forward_ms = timed_region(_forward, stream)
    loss = holder["loss"]

    def _backward():
        with torch.profiler.record_function("weaver_overhead_backward"):
            loss.backward()

    backward_ms = timed_region(_backward, stream)

    if comm_buffer is not None:
        def _explicit_comm():
            with torch.profiler.record_function("weaver_overhead_explicit_all_reduce"):
                holder["comm_bytes"] = torch.tensor(explicit_all_reduce(comm_buffer), device=x.device)

        explicit_comm_ms = timed_region(_explicit_comm, stream)
        explicit_comm_bytes = int(holder["comm_bytes"].item())
    else:
        explicit_comm_ms = 0.0
        explicit_comm_bytes = 0

    def _optimizer():
        with torch.profiler.record_function("weaver_overhead_optimizer"):
            optimizer.step()

    optimizer_ms = timed_region(_optimizer, stream)

    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": optimizer_ms,
        "explicit_comm_ms": explicit_comm_ms,
        "loss": float(loss.detach().item()),
        "explicit_comm_bytes": explicit_comm_bytes,
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def profiler_context(args: argparse.Namespace, output_dir: Path):
    if not args.torch_profiler:
        return None
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=max(0, args.profiler_wait),
            warmup=max(0, args.profiler_warmup),
            active=max(1, args.profiler_active),
            repeat=1,
        ),
        record_shapes=args.profiler_record_shapes,
        profile_memory=args.profiler_profile_memory,
        with_stack=args.profiler_with_stack,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(output_dir / "torch_profiler" / f"rank_{rank()}")
        ),
    )


def main() -> None:
    args = parse_args()
    normalize_profiler_args(args)
    setup_runtime(args)

    r = rank()
    lr = local_rank()
    ws = world_size()
    output_dir = Path(args.output_dir)
    rank_dir = output_dir / args.run_mode / f"rep_{args.repetition}" / f"rank_{r}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed + r)
    device = torch.device("cuda", lr)

    model = OverheadModel(args.dim, args.hidden_dim, args.layers).to(device)
    if not args.single_gpu:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[lr], output_device=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    x = make_inputs(args, device)
    y = make_targets(args, device)
    comm_buffer = None
    if args.explicit_comm_mb > 0 and not args.single_gpu and world_size() > 1:
        comm_elems = max(1, (args.explicit_comm_mb * 1024 * 1024) // 4)
        comm_buffer = torch.randn(comm_elems, device=device)

    metadata = {
        "run_mode": args.run_mode,
        "repetition": args.repetition,
        "rank": r,
        "local_rank": lr,
        "world_size": ws,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "workload": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "dim": args.dim,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "explicit_comm_mb": args.explicit_comm_mb,
            "single_gpu": args.single_gpu,
            "warmup": args.warmup,
            "iters": args.iters,
            "profiler_active": args.profiler_active,
            "profiler_record_shapes": args.profiler_record_shapes,
            "profiler_profile_memory": args.profiler_profile_memory,
            "profiler_with_stack": args.profiler_with_stack,
        },
        "env": {
            "WEAVER_AUTO_PROFILE": os.environ.get("WEAVER_AUTO_PROFILE", ""),
            "WEAVER_PYTHON_EVENT_BUDGET": os.environ.get("WEAVER_PYTHON_EVENT_BUDGET", ""),
            "WEAVER_CUDA_EVENTS": os.environ.get("WEAVER_CUDA_EVENTS", ""),
            "WEAVER_ENABLE_DISASM": os.environ.get("WEAVER_ENABLE_DISASM", ""),
            "WEAVER_CUDA_SYNC_ANCHOR": os.environ.get("WEAVER_CUDA_SYNC_ANCHOR", ""),
        },
    }
    with (rank_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)

    total = args.warmup + args.iters
    metrics: List[StepMetric] = []
    prof = profiler_context(args, rank_dir)

    if prof is None:
        context = None
    else:
        context = prof

    if context is not None:
        context.__enter__()
    try:
        for i in range(total):
            profiled = is_profiler_active_step(args, i)
            measured = i >= args.warmup
            if args.torch_profiler:
                measured = measured and profiled
            torch.cuda.synchronize()
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter_ns()
            step_start.record()
            parts = overhead_train_step(model, optimizer, x, y, comm_buffer)
            step_end.record()
            torch.cuda.synchronize()
            host_end = time.perf_counter_ns()

            metric = StepMetric(
                run_mode=args.run_mode,
                repetition=args.repetition,
                rank=r,
                local_rank=lr,
                world_size=ws,
                iteration=i,
                measured=measured,
                profiled=profiled,
                host_step_ms=(host_end - host_start) / 1e6,
                gpu_step_ms=event_elapsed(step_start, step_end),
                forward_ms=float(parts["forward_ms"]),
                backward_ms=float(parts["backward_ms"]),
                optimizer_ms=float(parts["optimizer_ms"]),
                explicit_comm_ms=float(parts["explicit_comm_ms"]),
                loss=float(parts["loss"]),
                tokens=args.batch_size * args.seq_len,
                explicit_comm_bytes=int(parts["explicit_comm_bytes"]),
            )
            metrics.append(metric)
            if prof is not None:
                prof.step()
    finally:
        if context is not None:
            context.__exit__(None, None, None)

    rows = [asdict(m) for m in metrics]
    write_jsonl(rank_dir / "step_metrics.jsonl", rows)

    measured_rows = [m for m in metrics if m.measured]
    summary = {
        "run_mode": args.run_mode,
        "repetition": args.repetition,
        "rank": r,
        "world_size": ws,
        "host_step_ms": summarize([m.host_step_ms for m in measured_rows]),
        "gpu_step_ms": summarize([m.gpu_step_ms for m in measured_rows]),
        "forward_ms": summarize([m.forward_ms for m in measured_rows]),
        "backward_ms": summarize([m.backward_ms for m in measured_rows]),
        "optimizer_ms": summarize([m.optimizer_ms for m in measured_rows]),
        "explicit_comm_ms": summarize([m.explicit_comm_ms for m in measured_rows]),
        "tokens_per_second_median": (
            (args.batch_size * args.seq_len * ws) / (summarize([m.host_step_ms for m in measured_rows])["median"] / 1e3)
            if measured_rows
            else 0.0
        ),
    }
    with (rank_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    if r == 0:
        print(json.dumps(summary, ensure_ascii=True))

    cleanup_dist()


if __name__ == "__main__":
    main()
