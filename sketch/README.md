"""
Weaver Sketch 模块使用指南。

ExecutionSketch 是从当前 trace 提取的轻量级执行结构，用于支持后续因果诊断。
它不存历史时间、平均 duration 或完整执行序列，只提供语义抽象。
"""

# 快速开始

## 1. 自动构建 Sketch（规则模式）

```bash
python -m weaver.sketch.builder \
  --timeline ./out/aligned_timeline_rank0.ndjson \
  --out ./out/execution_sketch_rank0.json
```

输出：`execution_sketch_rank0.json`，包含：
- kernel_templates: Kernel 类型模板（GEMM、NCCL、MEMCPY 等）
- dependency_rules: 依赖规则（same-stream、sync 等）
- overlap_expectations: Overlap 期望（GEMM-NCCL 可能 overlap 等）


## 2. 带 hints 的 Sketch 构建

创建 `hints.json`：

```json
{
  "workload": "ddp_transformer",
  "expected_overlaps": [
    {
      "left_family": "GEMM",
      "right_family": "NCCL",
      "phase": "backward"
    }
  ]
}
```

运行：

```bash
python -m weaver.sketch.builder \
  --timeline ./out/aligned_timeline_rank0.ndjson \
  --hints hints.json \
  --out ./out/execution_sketch_rank0.json
```


## 3. 在代码中使用 Sketch

### 3.1 构建 Sketch

```python
from weaver.sketch import SketchBuilder, KernelMatcher

# 构建 sketch
builder = SketchBuilder(timeline_path, hints_path=None)
sketch = builder.build_sketch()

# 查看 sketch 内容
print(f"Families: {sketch.metadata['unique_families']}")
print(f"Templates: {len(sketch.kernel_templates)}")
print(f"Rules: {len(sketch.dependency_rules)}")
print(f"Overlaps: {len(sketch.overlap_expectations)}")
```

### 3.2 Kernel 分类

```python
from weaver.sketch import classify_kernel, KernelRecord

record = KernelRecord(
    kernel_name="cutlass_kernel_gemm",
    operator_name="linear",
    grid=(64, 64, 1),
    block=(256, 1, 1),
)

kclass = classify_kernel(record)
print(f"Family: {kclass.family}, Tag: {kclass.tag}")
```

### 3.3 Kernel 映射

```python
from weaver.sketch import KernelMatcher

matcher = KernelMatcher(sketch)

# 单个 kernel 匹配
record = KernelRecord(kernel_name="ncclKernel_AllReduce", kind="nccl_all_reduce")
template = matcher.match_kernel(record)
print(f"Matched template: {template.template_id}")

# 批量 kernel 匹配
records = [...]  # list of KernelRecord
matches = matcher.match_kernels(records)
```

### 3.4 Sketch 序列化

```python
import json

# 保存到 JSON
with open("sketch.json", "w") as f:
    json.dump(sketch.to_dict(), f, indent=2)

# 从 JSON 加载
with open("sketch.json", "r") as f:
    sketch_dict = json.load(f)
    sketch = ExecutionSketch.from_dict(sketch_dict)
```


## 4. Sketch 输出文件格式

`execution_sketch.json` 的结构：

```json
{
  "metadata": {
    "schema_version": "0.1",
    "workload": "ddp_transformer",
    "source": "auto_from_trace",
    "timeline_path": "./aligned_timeline_rank0.ndjson",
    "num_kernel_events": 512,
    "unique_families": ["GEMM", "NCCL", "MEMCPY"],
    "unique_tags": ["GEMM_large", "NCCL_allreduce_64MB", "MEMCPY_large"]
  },
  "kernel_templates": [
    {
      "template_id": "gemm_large",
      "family": "GEMM",
      "tag": "GEMM_large",
      "match": {
        "kernel_name_regex": ".*gemm.*|.*matmul.*",
        "operator_regex": ".*mm.*|.*matmul.*|.*linear.*"
      },
      "work_units": {
        "type": "flops",
        "value": null,
        "formula": "2*M*N*K",
        "confidence": "medium"
      },
      "resource_hint": ["compute"],
      "expected_behavior": {
        "allow_overlap": true,
        "critical": true
      }
    },
    ...
  ],
  "dependency_rules": [
    {
      "rule_id": "same_stream_order",
      "type": "hard",
      "source": "timeline",
      "description": "kernels in the same CUDA stream preserve order"
    },
    ...
  ],
  "overlap_expectations": [
    {
      "relation_id": "backward_compute_grad_comm",
      "left_family": "GEMM",
      "right_family": "NCCL",
      "phase": "backward",
      "expected": "may_overlap"
    },
    ...
  ]
}
```


## 5. 关键概念

### 5.1 Kernel Family

粗粒度 kernel 类型：
- `GEMM`: 矩阵乘法（包括 linear、addmm、bmm）
- `NCCL`: 集合通信（all-reduce、all-gather、reduce-scatter、broadcast）
- `MEMCPY`: 内存拷贝
- `REDUCTION`: 归约操作
- `ELEMENTWISE`: 逐元素操作
- `UNKNOWN`: 未知类型

### 5.2 Kernel Tag

细粒度分组，例如：
- `GEMM_small`: 小 GEMM（< 256 warps）
- `GEMM_medium`: 中等 GEMM （256-4096 warps）
- `GEMM_large`: 大 GEMM （> 4096 warps）
- `NCCL_allreduce_<=16MB`: 小数据 AllReduce
- `NCCL_allreduce_16-128MB`: 中等数据 AllReduce
- `NCCL_allreduce_>128MB`: 大数据 AllReduce
- `MEMCPY_small`: 小内存拷贝
- `MEMCPY_large`: 大内存拷贝

### 5.3 Work Units

工作量度量：
- `flops`: 浮点操作（用于 GEMM）
- `bytes`: 字节数（用于 NCCL、MEMCPY）
- `elements`: 元素数
- `warps`: Warp 数
- `unknown`: 未知

### 5.4 Dependency Rules

执行依赖关系：
- `same_stream_order`: 同一 CUDA stream 中 kernel 按顺序执行
- `sync_serializes`: CUDA synchronize 强制后续 GPU work 等待
- `event_wait_serializes`: Event wait 强制跨 stream 的顺序


## 6. 与模块三的集成

Sketch 的输出被模块三（差分式因果定位）使用，作为输入：

```
aligned_timeline_rank0.ndjson
    ↓
[模块二: SketchBuilder]
    ↓
execution_sketch_rank0.json
    ↓
[模块三: 差分诊断]
    ↓
diagnosis_report_rank0.json
```

Sketch 为诊断提供：
1. **结构化的 kernel 分类** - 用于识别异常
2. **依赖规则** - 用于判断是否违反了预期顺序
3. **Overlap 期望** - 用于检测 overlap loss


## 7. 扩展和定制

### 7.1 添加自定义 kernel 类型

编辑 `weaver/sketch/rules.py`，在 `classify_kernel()` 中添加新规则：

```python
if "custom_kernel_name" in name:
    return KernelClass(family="CUSTOM", tag="CUSTOM_type")
```

### 7.2 添加自定义 work_units 计算

编辑 `infer_work_units()` 函数，添加新的计算逻辑。

### 7.3 自定义 Overlap 期望

通过 `hints.json` 提供期望，或编辑 `get_default_overlap_expectations()`。
"""

__all__ = ["README"]
