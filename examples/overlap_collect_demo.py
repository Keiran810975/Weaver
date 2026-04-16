import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from weaver.collector import emit_event, enable_python_collector


def parse_args():
    parser = argparse.ArgumentParser(description="compute-communication overlap collection demo")
    parser.add_argument("--comm_size", type=int, default=2048, help="communication tensor edge size")
    parser.add_argument("--m_size", type=int, default=2048, help="matrix size")
    parser.add_argument("--iters", type=int, default=8, help="loop iterations")
    parser.add_argument("--trace-dir", type=str, default="./out", help="trace output directory")
    parser.add_argument("--sample-rate", type=int, default=1)
    return parser.parse_args()


def resolve_backend() -> str:
    return "nccl" if torch.cuda.is_available() else "gloo"


def setup_distributed(backend: str):
    if dist.is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend)
        return
    dist.init_process_group(
        backend=backend,
        init_method="tcp://127.0.0.1:29500",
        rank=0,
        world_size=1,
    )


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def current_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def current_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def device_for_run() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def create_comm_buffer(size: int, device: torch.device):
    return torch.randn(size, size, device=device)


def collect_hardware(device: torch.device) -> dict:
    hw = {
        "backend": resolve_backend(),
        "rank": current_rank(),
        "world_size": current_world_size(),
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(device)
        hw["gpu"] = {
            "name": prop.name,
            "major": prop.major,
            "minor": prop.minor,
            "total_memory": int(prop.total_memory),
            "multi_processor_count": int(prop.multi_processor_count),
        }
    return hw


def trace_path(args, rank: int) -> Path:
    out_dir = Path(args.trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"overlap_trace_{rank}_{args.m_size}_{args.comm_size}.json"


def main():
    args = parse_args()

    enable_python_collector(
        socket_path=os.environ.get("WEAVER_SOCK", "/tmp/weaver.sock"),
        sample_rate=max(1, args.sample_rate),
        include_stdlib=False,
    )

    backend = resolve_backend()
    setup_distributed(backend)
    device = device_for_run()
    rank = current_rank()

    emit_event("job_start", {"rank": rank, "backend": backend, "device": str(device)})
    emit_event("hardware_info", collect_hardware(device))

    if device.type == "cuda":
        comm_stream = torch.cuda.Stream()
        sync_event = torch.cuda.Event()
    else:
        comm_stream = None
        sync_event = None

    A = torch.randn(args.m_size, args.m_size, device=device)
    B = torch.randn(args.m_size, args.m_size, device=device)
    COMM = create_comm_buffer(args.comm_size, device=device)

    for _ in range(2):
        _ = torch.mm(A, B)
    if device.type == "cuda":
        torch.cuda.synchronize()

    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=profile_activities,
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=max(1, args.iters), repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(args.iters):
            marker = f"weaver_sync_{i}"
            emit_event("sync_marker", {"marker": marker, "iter": i, "rank": rank})

            with torch.profiler.record_function(marker):
                if device.type == "cuda":
                    with torch.cuda.stream(comm_stream):
                        dist.all_reduce(COMM, op=dist.ReduceOp.SUM)
                        sync_event.record(comm_stream)
                else:
                    dist.all_reduce(COMM, op=dist.ReduceOp.SUM)

                C = torch.mm(A, B)
                if sync_event is not None:
                    sync_event.wait()

                # Keep tensor alive to avoid being optimized away.
                _ = C.sum().item()

            if device.type == "cuda":
                torch.cuda.synchronize()

            emit_event("iter_done", {"iter": i, "rank": rank})
            prof.step()

    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
    print(table)

    log_path = trace_path(args, rank)
    prof.export_chrome_trace(str(log_path))
    emit_event("trace_saved", {"rank": rank, "path": str(log_path)})
    emit_event("job_done", {"rank": rank})

    print(json.dumps({"rank": rank, "trace": str(log_path)}, ensure_ascii=True))
    cleanup_distributed()


if __name__ == "__main__":
    main()
