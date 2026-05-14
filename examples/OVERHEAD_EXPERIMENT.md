# Weaver Three-Layer Collection Overhead Experiment

This experiment measures the steady-state cost of Weaver's three-layer collector:

1. Native CPython `PyEval_SetProfile` operator collection with code-object address matching.
2. LD_PRELOAD CUDA/NCCL launch interception with selective CUDA Event timing for GEMM/NCCL/sync, name-only records for metadata-useful kernels, and dropped low-value kernels.
3. Warp/block launch metadata derived online from grid/block geometry.

## Core Idea

Run the exact same dual-GPU DDP training workload under several modes:

- `baseline`: no daemon, no CPython profile hook, no LD_PRELOAD hook.
- `weaver_full`: daemon + native CPython hook + CUDA/NCCL hook. The default hook mode records GEMM/NCCL kernels and stream sync waits with CUDA Event GPU start/end, records layout/reduction-style metadata kernels by name only, and drops low-value high-frequency kernels.
- `weaver_no_disasm`: compatibility mode; normal collection already disables the disassembly sidecar.
- `torch_profiler`: PyTorch profiler reference mode.

The primary metric is median per-step host wall time after warmup:

```text
overhead_pct = (median_step_ms(mode) - median_step_ms(baseline)) / median_step_ms(baseline) * 100
```

GPU step time, p95 step time, event count, and emitted event bytes are secondary metrics. The normal path avoids CUDA Events for low-value kernels and does not emit very small uninformative launches; measured GPU start/end is kept for GEMM/NCCL/sync and trigger windows. Use `--collection-mode full` to reproduce the older all-kernel timing ablation.

By default the Weaver Python layer uses the native C collector in a short burst:
it samples matched operator calls with `--python-sample-rate 1`, emits one Python
operator anchor per rank (`--python-event-budget 1`), then unloads the CPython
profile callback. This keeps the normal path from paying profile-hook cost during
steady-state training. Use `--python-event-budget 0` when you want an unlimited
Python operator trace for deep diagnosis. The pure-Python collector is kept only
as a debugging fallback because it is not the low-overhead Flare-style path.

## Workload

The quick preset is sized for a 2xV100 node and is intended to finish in about
five minutes or less. It is still more substantial than a single GEMM:

- Transformer-like MLP blocks with LayerNorm, GELU/SILU, and large Linear kernels.
- DistributedDataParallel gradient synchronization.
- An explicit 16 MB NCCL `all_reduce` every step in the quick preset.
- AdamW optimizer step.

This gives enough Python operator activity, compute kernels, and communication kernels to exercise the three layers.

## Quick Run On A 2xV100 Node

From the repository root:

```bash
PYTHONPATH=. python examples/run_overhead_experiment.py \
  --preset quick \
  --output-dir ./overhead_v100_quick
```

Outputs:

- `overhead_v100_quick/summary.json`: machine-readable summary.
- `overhead_v100_quick/summary.md`: compact report table.
- `overhead_v100_quick/<mode>/rep_<n>/rank_<r>/step_metrics.jsonl`: raw per-step records.
- `overhead_v100_quick/<mode>/rep_<n>/weaver_events.ndjson`: Weaver daemon events for Weaver modes.

The quick preset uses:

```text
modes = baseline,weaver_no_disasm,weaver_full
repeats = 1
warmup = 5
iters = 20
batch_size = 4
seq_len = 256
dim = 512
hidden_dim = 2048
layers = 3
explicit_comm_mb = 16
python_sample_rate = 1
python_event_budget = 1
```

For an ablation that separates Python collection from the CUDA/NCCL hook:

```bash
PYTHONPATH=. python examples/run_overhead_experiment.py \
  --preset quick \
  --modes baseline,weaver_kernel_only,weaver_py_only,weaver_full \
  --output-dir ./overhead_v100_ablation
```

`torch_profiler` is intentionally not included in the default quick preset
because trace export time can be noisy. To include it as a reference while still
keeping the run short:

```bash
PYTHONPATH=. python examples/run_overhead_experiment.py \
  --preset quick \
  --modes baseline,weaver_full,torch_profiler \
  --output-dir ./overhead_v100_quick_profiler
```

For the longer paper-grade run:

```bash
PYTHONPATH=. python examples/run_overhead_experiment.py \
  --preset paper \
  --output-dir ./overhead_out
```

## Interpreting Results

Use `weaver_full` versus `baseline` as the main result. A good low-overhead claim should report:

- median host step overhead;
- p95 host step overhead;
- GPU step overhead;
- event MB per step;
- whether `weaver_full` is below the optional `torch_profiler` reference.

Use `torch_profiler` only as a familiar reference point. The normal Weaver path should be no slower than this reference; selective mode should report CUDA Event timed kernel start/end for important kernels, suspicious kernels, and their immediate neighborhood.
