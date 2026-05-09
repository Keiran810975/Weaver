# Weaver Three-Layer Collection Overhead Experiment

This experiment measures the steady-state cost of Weaver's three-layer collector:

1. CPython profile-hook operator collection.
2. LD_PRELOAD CUDA/NCCL launch interception with CUDA Event timing and async polling.
3. Neutrino-style hook-after binary capture plus PTX/SASS disassembly sidecar.

## Core Idea

Run the exact same dual-GPU DDP training workload under several modes:

- `baseline`: no daemon, no CPython profile hook, no LD_PRELOAD hook.
- `weaver_full`: daemon + CPython hook + CUDA/NCCL hook + CUDA Event poller + disassembly sidecar.
- `weaver_no_disasm`: same as Weaver, but disables one-time binary disassembly to isolate steady-state online cost.
- `torch_profiler`: PyTorch profiler reference mode.

The primary metric is median per-step host wall time after warmup:

```text
overhead_pct = (median_step_ms(mode) - median_step_ms(baseline)) / median_step_ms(baseline) * 100
```

GPU Event step time, p95 step time, event count, and emitted event bytes are secondary metrics. Warmup iterations are excluded so one-time kernel capture/disassembly is amortized, which matches the intended online usage.

## Workload

The workload is intentionally more substantial than a single GEMM:

- Transformer-like MLP blocks with LayerNorm, GELU/SILU, and large Linear kernels.
- DistributedDataParallel gradient synchronization.
- An explicit 64 MB NCCL `all_reduce` every step.
- AdamW optimizer step.

This gives enough Python operator activity, compute kernels, and communication kernels to exercise the three layers.

## Run On A Dual-GPU Node

From the repository root:

```bash
PYTHONPATH=. python examples/run_overhead_experiment.py \
  --modes baseline,weaver_full,weaver_no_disasm,torch_profiler \
  --repeats 3 \
  --nproc-per-node 2 \
  --warmup 20 \
  --iters 100 \
  --output-dir ./overhead_out
```

Outputs:

- `overhead_out/summary.json`: machine-readable summary.
- `overhead_out/summary.md`: compact report table.
- `overhead_out/<mode>/rep_<n>/rank_<r>/step_metrics.jsonl`: raw per-step records.
- `overhead_out/<mode>/rep_<n>/weaver_events.ndjson`: Weaver daemon events for Weaver modes.

## Interpreting Results

Use `weaver_full` versus `baseline` as the main result. A good low-overhead claim should report:

- median host step overhead;
- p95 host step overhead;
- GPU step overhead;
- event MB per step;
- whether `weaver_full` is close to `weaver_no_disasm` after warmup.

Use `torch_profiler` only as a familiar reference point. It is expected to have much higher overhead and event volume than Weaver because it records a broader trace.
