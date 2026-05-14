这次结果很清楚：**8% 开销主要不是因为 GEMM 本身，而是 full 模式把大量小 kernel 也逐个做了 CUDA Event timing**。

这次 `weaver_events.ndjson` 里一共：

- `15329` 个 `kernel_launch`
- 全部 `capture_mode=full`
- 事件文件 `23MB`
- `weaver_full` 端到端开销：host `+8.79%`，GPU `+8.81%`
- torch profiler：约 `+51%`

**Kernel 构成**
按类别统计大概是：

| 类别 | 数量 | 占比 | median GPU 时间 | 作用 | 建议 |
|---|---:|---:|---:|---|---|
| Elementwise / activation / loss | 4620 | 30.1% | 13.3 us | add、GELU/SILU、MSE loss/backward | 默认不做 full timing |
| GEMM / matmul | 4095 | 26.7% | 146.4 us | Linear 前后向矩阵乘 | 保留 full timing |
| `<runtime_kernel>` | 3257 | 21.2% | 19.5 us | 未解析出名字的 runtime kernel | 不应全量采，先采样/触发采 |
| Reduction / norm | 1995 | 13.0% | 11.3 us | LayerNorm、MSE reduce | 只在 reduction 目标实验里 full timing |
| Optimizer foreach / AdamW | 840 | 5.5% | 87.0 us | AdamW 多 tensor 更新 | 开销实验可不采 |
| Fill / zero | 520 | 3.4% | 12.3 us | zero grad / fill | 不采或 name-only |
| Random init | 2 | 很少 | 32 us | 初始化随机数 | 完全不采 |

**最该过滤的**
优先级从高到低：

1. **Elementwise / activation / loss**
   - 数量最多，`4620` 个，占 `30%`
   - 单个很短，通常不是我们论文里的主要 target
   - 包括 `vectorized_elementwise_kernel`、GELU、SILU、MSE loss/backward

2. **`<runtime_kernel>`**
   - `3257` 个，占 `21%`
   - 名字没解析出来，语义价值低
   - 不建议完全丢弃，最好改成：同一个 func 指针只采前 N 次，或者只在异常窗口里 full timing

3. **Fill / zero**
   - `520` 个
   - 基本是 zero grad / fill 之类辅助 kernel
   - 对依赖诊断价值低

4. **Optimizer foreach / AdamW**
   - `840` 个
   - 如果实验关注 forward/backward 诊断，可以不 full timing
   - 如果以后专门诊断 optimizer，再打开

5. **小 LayerNorm / MSE reduction**
   - 数量也不少，`1995` 个
   - 如果论文当前模块要覆盖 Reduction 类，可以保留“目标 reduction”的 timing
   - 但开销实验里这些小 norm/reduce 可以先 name-only

**如果优化策略只保留 GEMM full timing**
这次可以从：

```text
15329 个 timed kernel
```

降到大约：

```text
4095 个 timed kernel
```

也就是减少约 `73%` 的 CUDA Event timing 和约 `77%` 的事件输出体积。8% 开销大概率会明显下降。

**推荐策略**
不要简单“完全不采”，而是分三档：

1. **full timing**
   - GEMM / matmul
   - NCCL
   - memcpy / transpose / layout
   - 被指定为 target 的 reduction / norm

2. **name-only**
   - elementwise
   - activation
   - fill / zero
   - optimizer foreach
   - 小 reduction / layernorm

3. **trigger full**
   - 如果 name-only kernel 出现在 `GEMM_A -> GEMM_B` 中间，或者不符合草图，再临时打开 full timing 窗口。
   - 这样 extra transpose 这种异常仍然能被抓住。

一句话结论：**现在 full 模式采得太全了。最值得砍的是 elementwise、runtime unknown、fill/zero、optimizer foreach。保留 GEMM/NCCL/memcpy/目标 reduction 的 full timing，再用异常触发窗口补证据，是更适合低开销诊断的方案。**