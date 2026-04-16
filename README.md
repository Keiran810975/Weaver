# Weaver Collector

Weaver is a pure information collector prototype that combines:
- Daemon-based online ingestion
- Python runtime collection via CPython profiling hook (`sys.setprofile`, backed by `PyEval_SetProfile`)
- CUDA/NCCL collection via `LD_PRELOAD` hook on driver/runtime symbols
- Warp-level visibility (`warps_per_block`, `total_warps`) emitted online for each kernel launch

## 1) Start daemon

```bash
python -m weaver.daemon.server --sock /tmp/weaver.sock --http-port 18731 --out ./weaver_events.ndjson
```

Online APIs:
- `GET /health`
- `GET /stats`
- `GET /tail?n=200`

## 2) Python layer collection

```python
from weaver.collector import enable_python_collector
enable_python_collector(socket_path="/tmp/weaver.sock", sample_rate=1)
```

Or run demo:

```bash
python examples/demo_python_collect.py
```

## 3) CUDA/NCCL hook collection

Build hook shared object:

```bash
cd hooks && make
```

Run target training with preload:

```bash
export WEAVER_SOCK=/tmp/weaver.sock
export LD_PRELOAD=$PWD/hooks/libweaver_hook.so
python your_training.py
```

The hook emits online events for:
- `cuModuleGetFunction` (kernel symbol mapping)
- `cuLaunchKernel` (grid/block/shared mem + warp-level counts)
- NCCL collectives (`ncclAllReduce`, `ncclAllGather`, `ncclReduceScatter`, `ncclBroadcast`)

## 4) Compute/Communication overlap sample and aligned timeline

Run overlap sample (single process by default, supports `torchrun` multi-process):

```bash
WEAVER_SOCK=/tmp/weaver.sock LD_PRELOAD=$PWD/hooks/libweaver_hook.so \
PYTHONPATH=. python examples/overlap_collect_demo.py --m_size 1024 --comm_size 1024 --iters 5 --trace-dir ./out
```

If you need multi-process:

```bash
WEAVER_SOCK=/tmp/weaver.sock LD_PRELOAD=$PWD/hooks/libweaver_hook.so \
PYTHONPATH=. torchrun --nproc_per_node=2 examples/overlap_collect_demo.py --m_size 1024 --comm_size 1024 --iters 5 --trace-dir ./out
```

Generate aligned timeline output (merged Weaver events + torch profiler events):

```bash
PYTHONPATH=. python -m weaver.analysis.timeline \
	--weaver ./weaver_events.ndjson \
	--trace ./out/overlap_trace_0_1024_1024.json \
	--out ./out/aligned_timeline_rank0.ndjson \
	--summary ./out/aligned_summary_rank0.json
```

Alignment details:
- Primary: explicit sync markers (`weaver_sync_i`) are matched between Weaver events and profiler events.
- Fallback: if marker match is missing, align by first timestamp boundary.
- Output includes operator/kernel/profiler records, hook records, and hardware summary.

## Event schema (high-level)

All events are JSON objects over Unix datagram socket.

Top-level keys:
- `ts_ns`
- `pid`
- `tid`
- `layer` (`python` or `cuda` or `hook`)
- `kind`
- `payload`

CUDA launch payload includes:
- `grid`, `block`, `shared_mem`
- `warps_per_block`, `total_warps`
- `warp_scope=estimated`
- `start_ns`, `end_ns`, `dur_ns`

## Notes

- Warp-level fields are launch-time derived metrics (online, low overhead).
- This prototype is collection-only and does not modify backend source code.
- For true instruction-level warp telemetry, you can extend the hook path with a probe injector similar to Neutrino's JIT probe flow.
