# Weaver Collector

Weaver is a pure information collector prototype that combines:
- Daemon-based online ingestion
- Low-overhead Python runtime collection via a native CPython `PyEval_SetProfile` callback
- Low-overhead CUDA/NCCL collection via `LD_PRELOAD` hook on driver/runtime symbols
- Warp-level visibility (`warps_per_block`, `total_warps`) emitted online for each kernel launch

## 1) Start daemon

```bash
python -m weaver.daemon.server --sock /tmp/weaver.sock --http-port 18731 --out ./weaver_events.ndjson
```

Online APIs:
- `GET /health`
- `GET /stats`
- `GET /tail?n=200`

## 2) Daemonized one-shot collection

Build the hook and run the target through Weaver's launcher:

```bash
make -C hooks
make -C weaver/collector PYTHON=$(which python)
PYTHONPATH=. python -m weaver.collector.launch --out ./weaver_events.ndjson -- python your_training.py
```

The launcher starts the daemon, injects `hooks/libweaver_hook.so`, and enables the native CPython profile collector through `sitecustomize.py`. The target program does not need source changes.

Useful environment knobs:
- `WEAVER_PYTHON_TRACE_FUNCS`: comma-separated function filters, using the DLRover-style `module@object@function` form or plain function names.
- `WEAVER_PYTHON_COLLECTOR=native`: use the Flare/DLRover-style native C profile callback; `python` keeps the older pure-Python fallback for debugging.
- `WEAVER_REQUIRE_NATIVE_PY=1`: fail fast if the native collector was not built, instead of silently falling back.
- `WEAVER_PYTHON_SAMPLE_RATE`: sample every N matched Python operator calls.
- `WEAVER_PYTHON_EVENT_BUDGET`: stop and unload the CPython profile callback after N emitted Python events. The low-overhead launcher defaults to `1`; use `0` for an unlimited/full Python trace during deep diagnosis.
- `WEAVER_TRACE_GC=0/1`: disable or enable Python GC pause events.
- `WEAVER_ENABLE_DISASM=1`: opt into loaded GPU code capture and the Neutrino-style disassembly sidecar. Default is `0` for the low-overhead normal path.
- `WEAVER_COLLECTION_MODE=selective`: record GEMM/NCCL/memcpy/layout kernels with CUDA Event GPU start/end, while high-frequency low-value kernels stay name-only. Use `adaptive_name` for sketch-triggered timing windows, `full` for the older all-kernel timing path, or `name_only` for names only.
- `WEAVER_SELECTIVE_TIMED_REDUCTION=1`: in selective mode, also time reduction/norm/softmax-style kernels. Default is name-only for lower overhead.
- `WEAVER_SELECTIVE_UNKNOWN_FULL=1`: in selective mode, time unknown/runtime kernels. Default is name-only because these are often short and frequent.
- `WEAVER_EXPECTED_KERNELS`: semicolon/comma-separated expected kernel patterns. Prefix entries with `exact:` or `regex:` when needed; unprefixed entries are substring matches.
- `WEAVER_TRIGGER_CAPTURE_AFTER=2`: after an unexpected kernel, collect this many following launches with full CUDA Event timing to catch the next/overlap neighborhood.
- `WEAVER_CUDA_EVENTS=1`: keep CUDA Event support available for selective timing and triggered windows.
- `WEAVER_CUDA_EVENT_POOL=1`: reuse CUDA Event pairs across launches to avoid per-launch create/destroy overhead.
- `WEAVER_CUDA_SYNC_ANCHOR=1`: synchronize a per-stream anchor to align CUDA Event times onto host time. Default is `1`.

For manual deep diagnosis, `weaver.collector.py_runtime.python_trace_window()` can
temporarily resume the Python collector around a suspected region and pause it
again on exit.

## 3) Python layer collection

```python
from weaver.collector import enable_python_collector
enable_python_collector(
    socket_path="/tmp/weaver.sock",
    targets=("train_step",),
    sample_rate=1,
    backend="native",
)
```

Or run demo:

```bash
python examples/demo_python_collect.py
```

## 4) CUDA/NCCL hook collection

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
- `cuLaunchKernel`, `cuLaunchKernelEx`, and their `_ptsz` variants. Selective mode times GEMM/NCCL/memcpy/layout kernels and emits low-value launches as name-only records; triggered/full mode adds CUDA Event GPU duration, grid/block/shared mem, and warp/block counts.
- `cudaLaunchKernel`, `cudaLaunchKernelExC`, `cudaLaunchCooperativeKernel`, and their `_ptsz` variants for CUDA Runtime API launches
- `cuGetProcAddress`/`cuGetProcAddress_v2` so libraries that resolve CUDA driver symbols dynamically still route through the hook
- `cuFuncGetName` fallback naming for driver kernels when module/function name mapping is unavailable
- module/library binary captures and Neutrino-style disassembly summaries
- NCCL collectives (`ncclAllReduce`, `ncclAllGather`, `ncclReduceScatter`, `ncclBroadcast`)

## 5) Compute/Communication overlap sample and aligned timeline

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

## 6) Controlled resource contention experiments

The experiment implementation follows `experiment.md` and provides separate entries for single-node multi-GPU and multi-node multi-GPU.

### 6.1 Single-node multi-GPU

Main entry:

```bash
PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=2 \
	examples/single_node_multigpu_experiment.py \
	--target gemm --interference sm \
	--intensities 0,1,2,4 --warmup 10 --iters 30 --output-dir ./exp_out/single_node
```

One-click examples:

```bash
chmod +x examples/run_single_node_experiments.sh
examples/run_single_node_experiments.sh
```

### 6.2 Multi-node multi-GPU

Main entry (run on each node, with different `NODE_RANK`):

```bash
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 MASTER_ADDR=10.0.0.1 MASTER_PORT=29510 \
	examples/run_multi_node_experiments.sh
```

The script runs link-contention workload (`target=nccl`, `interference=link`) and emits node-local summaries.

### 6.3 Experiment outputs

Per rank files:
- `iter_metrics_*.csv`: per-iteration metrics
- `summary_*.json`: median/MAD summary by `phase|intensity`

Auto-generated report:
- `report.json`: merged quick view for slowdown/overlap/skew

### 6.4 How to read results

For each intensity level, compare:
- `baseline` vs `overlap`: contention impact
- `overlap` vs `serialized`: overlap-specific slowdown
- `recovery` vs `baseline`: whether slowdown recovers after interference removal

Key metrics and interpretation:
- `target_ms_median`: target kernel/operator slowdown
- `overlap_ratio_median`: real overlap degree (higher means stronger concurrency)
- `target_bandwidth_proxy_gbps_median`: communication/memory throughput proxy
- `rank_start_skew_ms_median`, `rank_end_skew_ms_median`: cross-rank skew and jitter signals

Use intensity sweep (`0,1,2,4...`) as dose-response evidence: if slowdown grows monotonically with intensity, contention causality is stronger.

## Collection overhead experiment

To validate low overhead, run the dual-GPU DDP benchmark:

```bash
PYTHONPATH=. python examples/run_overhead_experiment.py \
	--preset quick \
	--output-dir ./overhead_v100_quick
```

The quick preset is sized for a 2xV100 node and should finish in roughly five minutes or less. It compares the same workload without collection, with Weaver's selective low-overhead normal path, and with `torch_profiler` as a reference. Details are in [examples/OVERHEAD_EXPERIMENT.md](examples/OVERHEAD_EXPERIMENT.md).

Use `--collection-mode full` only to measure the older all-kernel CUDA Event path. The default overhead claim should use selective mode.

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
- `warp_scope=block_runtime`
- `gpu_start_ns`, `gpu_end_ns`, `gpu_duration_ns`
- `cpu_enqueue_start_ns`, `cpu_enqueue_end_ns`

## 7) ExecutionSketch 构建（模块二：草图化预期建模）

从 aligned timeline 构建轻量级执行草图，用于后续因果诊断。

### 7.1 自动构建 Sketch（规则生成模式）

```bash
PYTHONPATH=. python -m weaver.sketch.builder \
	--timeline ./out/aligned_timeline_rank0.ndjson \
	--out ./out/execution_sketch_rank0.json
```

### 7.2 带 hints 的 Sketch 构建

创建 `hints.json`:

```json
{
  "workload": "ddp_transformer",
  "expected_overlaps": [
    {"left_family": "GEMM", "right_family": "NCCL", "phase": "backward"}
  ]
}
```

运行：

```bash
PYTHONPATH=. python -m weaver.sketch.builder \
	--timeline ./out/aligned_timeline_rank0.ndjson \
	--hints hints.json \
	--out ./out/execution_sketch_rank0.json
```

### 7.3 输出 Sketch 内容

Sketch 包含：
- `kernel_templates`: Kernel 类型模板（GEMM、NCCL、MEMCPY 等）
- `dependency_rules`: 依赖规则（same-stream、sync 等）
- `overlap_expectations`: Overlap 期望（GEMM-NCCL 可能 overlap 等）

### 7.4 在代码中使用 Sketch

```python
from weaver.sketch import SketchBuilder, KernelMatcher, classify_kernel, KernelRecord

# 构建 sketch
builder = SketchBuilder("aligned_timeline_rank0.ndjson")
sketch = builder.build_sketch()

# 检查 kernel family
record = KernelRecord(kernel_name="ncclKernel_AllReduce", kind="nccl_all_reduce")
kclass = classify_kernel(record)
print(f"Family: {kclass.family}, Tag: {kclass.tag}")

# Kernel 映射
matcher = KernelMatcher(sketch)
template = matcher.match_kernel(record)
```

详细使用文档见 [weaver/sketch/README.md](weaver/sketch/README.md)

## 8) 差分式因果诊断（模块三：诊断与根因定位）

从 aligned timeline 和 execution sketch 进行深度性能诊断，识别性能下降的根本原因。

### 8.1 诊断工作流

```bash
PYTHONPATH=. python -m weaver.diagnose.cli \
	--timeline ./out/aligned_timeline_rank0.ndjson \
	--sketch ./examples/manual_expected_sketch.json \
	--rank 0 \
	--output ./diagnosis_rank0.json \
	--output-html ./diagnosis_rank0.html \
	--output-text ./diagnosis_rank0.txt \
	--verbose
```

`--sketch` is intended to be a manually written expected sketch in the current
prototype. See `examples/manual_expected_sketch.json` for the compact format:
use `expected_dependencies` to describe expected predecessor relationships and
`overlap_expectations` to describe relationships that may/should overlap.

### 8.2 诊断流程

诊断模块实现了五层分析：

1. **规范化**: `Timeline` → `KernelRecord + SyncRecord + OperatorRecord`
   - 统一多源事件格式（hook、profiler、python）
   - 关联 CPU 和 GPU 时间轴
   - 计算工作量和工作进度

2. **候选发现**: 发现可疑的 target kernel
   - 结构偏离: 无法分类、意外前驱、overlap 丧失
   - 性能异常: 工作进度异常低（MAD 检测）
   - Rank 异常: 同类比较速度显著下降

3. **Slowdown 分类**: 对每个候选进行性能下降类型分类
   - `cpu_runtime_blocked`: CPU 侧 launch 时间晚
   - `dependency_blocked`: GPU 启动时间晚（依赖阻塞）
   - `resource_slowed`: GPU 启动正常但执行变慢（资源竞争）
   - `uncertain`: 信息不足

4. **依赖定位**: 定位 dependency_blocked kernel 的阻塞源
   - 查询 same-stream 前驱
   - 检测意外前驱
   - 计算反事实启动时间和 overlap 丧失

5. **资源定位**: 定位 resource_slowed kernel 的干扰源
   - 发现 overlap witness（>10% 重叠）
   - Same-run differential（有/无干扰源的进度对比）
   - Dose-response 分析（干扰强度与性能下降关联）

### 8.3 诊断报告

JSON 格式报告示例：

```json
{
  "rank": 0,
  "summary": {
    "slowdown_type_distribution": {
      "cpu_runtime_blocked": 5,
      "dependency_blocked": 12,
      "resource_slowed": 8,
      "uncertain": 3
    },
    "top_blockers": [
      {"kernel_id": "nccl_kernel", "count": 12}
    ],
    "top_culprits": [
      {"kernel_id": "nccl_all_reduce", "count": 8}
    ]
  },
  "targets": {
    "k_slow_1": {
      "slowdown": {
        "slowdown_type": "dependency_blocked",
        "confidence": 0.85
      },
      "dependency": {
        "blocker_id": "nccl_kernel",
        "delay_ns": 8000000,
        "confidence": 0.9
      }
    }
  }
}
```

生成 HTML 和文本报告用于人工审查。

### 8.4 在代码中使用诊断模块

```python
from weaver.weaver.diagnose import (
    TimelineNormalizer,
    CandidateDiscovery,
    TimingAnalyzer,
    DependencyLocalizer,
    ResourceLocalizer,
    DiagnosisReporter
)
from weaver.weaver.sketch import load_execution_sketch

# 加载 timeline 和 sketch
normalizer = TimelineNormalizer("aligned_timeline_rank0.ndjson")
kernels, operators, syncs = normalizer.normalize()

sketch = load_execution_sketch("execution_sketch_rank0.json")

# 候选发现
discoverer = CandidateDiscovery(sketch)
candidates = discoverer.discover(kernels)

# 性能下降分类
analyzer = TimingAnalyzer()
slowdown_diags = analyzer.classify_batch(kernels)

# 依赖定位
dep_localizer = DependencyLocalizer()
dependency_diags = {
    k: dep_localizer.localize(k, kernels, sketch)
    for k in slowdown_diags if slowdown_diags[k].slowdown_type.value == "dependency_blocked"
}

# 资源定位
res_localizer = ResourceLocalizer()
resource_diags = {
    k: res_localizer.localize(k, kernels, sketch)
    for k in slowdown_diags if slowdown_diags[k].slowdown_type.value == "resource_slowed"
}

# 生成报告
reporter = DiagnosisReporter(rank=0)
report = reporter.generate_report(
    candidates, slowdown_diags, dependency_diags, resource_diags
)

# 输出
reporter.to_json(report, "diagnosis.json")
reporter.to_html(report, "diagnosis.html")
```

详细文档见 [MODULE_3_COMPLETION_SUMMARY.md](../MODULE_3_COMPLETION_SUMMARY.md)

## Notes

- Sketch 不存历史时间、平均 duration 或完整执行序列，只提供语义结构抽象。
- Sketch 是模块三（差分式因果定位）的输入。
- Warp-level fields are launch-time derived metrics (online, low overhead).
- This prototype is collection-only and does not modify backend source code.
- For true instruction-level warp telemetry, you can extend the hook path with a probe injector similar to Neutrino's JIT probe flow.
