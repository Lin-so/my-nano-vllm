<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM（二次开发版）

一个从零实现的轻量级 vLLM。本项目在 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 基础上二次开发，面向**长文本、高并发**推理场景，围绕 KVCache 访存瓶颈、Prefill/Decode 调度、连续批处理与投机解码四个方向进行了系统性优化，核心代码仅约数千行 Python，适合学习与二次定制。

## 核心特性

- 🧮 **FP8 PagedAttention 算子** —— 基于 Triton 自研 FP8 E4M3 KVCache 量化与 Decode/Prefill 内核，支持 GQA、Paged KV Cache、变长序列与连续/分页双布局
- 🗓️ **Chunked Prefill 调度** —— Decode 严格优先 + Token Budget 控制，长 Prefill 自动分块，消除 Decode 饥饿
- 🔄 **Prefill/Decode 混合批处理** —— 统一 Attention 算子单次前向完成 PD 混合批，纯 Decode 走 CUDA Graph、其余走 eager
- 🎯 **N-gram 投机解码** —— 无需 draft model、零训练成本，基于序列自身历史最长后缀匹配生成草稿，目标模型一次前向并行验证，输出无损

---

## 1. FP8 PagedAttention Decode/Prefill 算子

针对长文本、高并发推理中的 KVCache 访存瓶颈，基于 Triton 实现了 FP8 E4M3 KVCache 量化以及配套的 FP8 PagedAttention Decode/Prefill 内核，减少 HBM 访存与中间结果写回。

**Decode 内核**（[fp8_attention.py](nanovllm/kernels/fp8_attention.py)）

- Grid 以 `(seq, head)` 划分，每个 program 负责一个序列的一个注意力头
- 按 `block_table` 间接寻址物理块，沿 KV 序列维度以 `BLOCK_N` 分块，向量化加载 FP8 K/V 并反量化到 FP32
- 全程在线维护 Online Softmax 状态（running max / 累加器），单次遍历完成注意力计算

**Prefill 内核**

- Grid 以 `(seq, block, head)` 划分，采用 `BLOCK_M × BLOCK_N` 分块，向量化加载 FP8 K/V
- 内建因果掩码（Causal Mask）、Online Softmax，并通过 cumulative sequence lengths（`cu_seqlens`）支持变长批处理

**通用能力**

- 支持 GQA（分组查询注意力）、Paged KV Cache、变长序列
- 同时支持**连续布局**（`block_tables=None`，新 KV 连续存放）与**分页布局**（按 block table 寻址）双布局

> **实测效果**：在 256 并发、1K ISL（输入长度）并启用 FP8 KVCache 的条件下，KVCache 容量提升约 **2 倍**，吞吐量提升约 **24%**，P99 TTFT（首 Token 延迟）下降约 **20%**。

## 2. Chunked Prefill + Decode 优先调度

原框架采用 Prefill 优先调度：长 Prefill 请求会持续占用批次，导致同批 Decode 请求被阻塞（Decode 饥饿），尾延迟恶化。

本项目实现 **Chunked Prefill + Decode 优先调度**（[scheduler.py](nanovllm/engine/scheduler.py)）：

- 引入 **Token Budget**（`max_num_batched_tokens`）控制单批 token 总量
- 设置长 Prefill 阈值（`long_prefill_token_threshold`，默认 512），超过阈值的 Prefill 自动切块、跨批次执行
- 每轮**严格优先调度全部 Decode 请求**，仅将批次内剩余预算用于插入 Prefill chunk
- 携带投机草稿的 Decode，其草稿 token 同样计入批次预算并受空闲块容量约束
- 显存不足时按队尾抢占（preempt），保障调度可行性

## 3. Prefill / Decode 混合批处理（Continuous Batching）

重写统一 Attention 算子（[attention.py](nanovllm/layers/attention.py)），将批内序列按「普通 Decode / 投机 Decode / Prefill」分组，**单次前向即可并行完成 PD 混合批**：

- 统一处理因果掩码、变长序列与 Paged KV Cache，消除 Prefill/Decode 分离计算的额外开销
- Decode 走 PagedAttention 路径，Prefill（含分块）走变长 varlen 路径，结果拼接后一次返回
- **纯 Prefill 与 PD 混合批采用 eager 模式**（批次形状动态、瓶颈在算力），**纯 Decode 采用 CUDA Graph**（形状固定、瓶颈在 launch 与访存带宽），兼顾动态调度灵活性与 kernel launch 开销

> **实测效果**：在 256 并发并配合 Chunked Prefill 调度下，P99 ITL（Token 间延迟）降低约 **82%**。

## 4. N-gram 投机解码

构建于序列自身 token 历史之上的 N-gram 投机解码（[ngram_proposer.py](nanovllm/engine/ngram_proposer.py)）：

- 取序列末尾 n-gram（默认 `n=2`），在历史中查找其最近一次出现，将其后的 token 作为草稿（最多 `k=4` 个）
- 目标模型**单次前向**并行验证「最后确认 token + 草稿」，接受与目标采样一致的最长前缀，并在首个分歧处采用目标模型的采样结果
- 拒绝时自动回退，**保持输出与目标模型完全一致（无损）**
- **无需 draft model、零训练成本**；被拒草稿占用的 KV 槽位在调度后自动回收

> **实测效果**：在高频重复 prompt 场景下，吞吐量提升约 **130%**。

---

## 其他优化

继承自 nano-vllm 的优化套件：Prefix Caching（前缀缓存）、张量并行（Tensor Parallelism）、Torch Compile、CUDA Graph 等。

## 安装

从源码安装：

```bash
git clone https://github.com/<your-user>/my-nano-vllm.git
cd my-nano-vllm
pip install -e .
```

依赖：Python 3.10+、CUDA、PyTorch、Triton、FlashAttention、Transformers。

## 模型下载

以下载 Qwen3 模型为例：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 快速开始

接口与 vLLM 基本对齐（`LLM.generate` 略有差异），完整示例见 [example.py](example.py)：

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

关键开关（见 [config.py](nanovllm/config.py)，均可按场景调整）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_chunked_prefill` | `True` | 启用 Chunked Prefill + PD 混合批 |
| `enable_fp8_kvcache` | `False` | 启用 FP8 E4M3 KVCache |
| `enable_ngram_spec_decode` | `True` | 启用 N-gram 投机解码 |
| `max_num_batched_tokens` | `8192` | 单批 token 预算（含 Prefill 与 Decode） |
| `long_prefill_token_threshold` | `512` | 长 Prefill 阈值，超过即分块 |
| `ngram_size` / `ngram_spec_num_tokens` | `2` / `4` | N-gram 的 n 与每步最多草稿数 k |
| `enforce_eager` | `False` | 是否禁用 CUDA Graph |
| `kvcache_block_size` | `256` | Paged KV Cache 块大小 |

## 性能测试

压测脚本见 [bench.py](bench.py)，用例集见 [bench_cases.py](bench_cases.py)，覆盖短问答、长文本生成、长上下文、重复生成、代码、摘要、推理与翻译等场景，统计 TTFT、ITL（含 P50/P90/P99）、吞吐量等指标，历史结果保存在 [benchmarks/](benchmarks/)。

**优化特性收益汇总（256 并发）：**

| 优化项 | 场景 | 收益 |
|--------|------|------|
| FP8 KVCache | 1K ISL | KVCache 容量 ↑ 约 2×；吞吐量 ↑ 约 24%；P99 TTFT ↓ 约 20% |
| Chunked Prefill + 混合批处理 | 256 并发 | P99 ITL ↓ 约 82% |
| N-gram 投机解码 | 高频重复 prompt | 吞吐量 ↑ 约 130% |

**基线对比（上游 nano-vllm 数据）：**

- 硬件：RTX 4070 Laptop（8GB）
- 模型：Qwen3-0.6B
- 共 256 个请求，输入/输出长度均在 100–1024 tokens 间随机采样

| 推理引擎 | 输出 Tokens | 耗时 (s) | 吞吐量 (tokens/s) |
|----------|------------|----------|-------------------|
| vLLM | 133,966 | 98.37 | 1361.84 |
| Nano-vLLM | 133,966 | 93.41 | 1434.13 |

## 项目结构

```
nanovllm/
├── kernels/fp8_attention.py   # FP8 PagedAttention Decode/Prefill Triton 内核
├── layers/attention.py        # 统一 PD 混合批 Attention 算子
├── engine/
│   ├── scheduler.py           # Chunked Prefill + Decode 优先调度
│   ├── ngram_proposer.py      # N-gram 草稿提出器
│   ├── model_runner.py        # 模型执行、变长准备、CUDA Graph 捕获
│   ├── block_manager.py       # Paged KV 块管理与前缀缓存
│   └── llm_engine.py          # 引擎主循环
├── models/qwen3.py            # Qwen3 模型实现
└── config.py                  # 全局配置与开关
```

## 致谢

本项目基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 二次开发，感谢原作者提供简洁清晰的最小 vLLM 实现。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
