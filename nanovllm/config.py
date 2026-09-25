import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    # 建议设置不超过 max_tokens*isl/(isl+osl)，其中max_tokens=KVCache总量/每token的KVCache
    max_num_batched_tokens: int = 8192 # 16384 # 单批次最大推理token，含prefill和decode，请求如果超出该值，则分块到下一批次
    max_num_seqs: int = 512 # 单批次最大请求数
    max_model_len: int = 2048 # 4096 # 单请求最大推理token，isl+osl
    gpu_memory_utilization: float = 0.9 # GPU显存最高使用比例
    tensor_parallel_size: int = 1 # GPU数
    enforce_eager: bool = False # False 表示启用 CUDA Graph 优化， True 表示禁用 CUDA Graph 优化
    enable_chunked_prefill: bool = True # 是否启用PD混合批次Chunked Prefill，启用后ITL（Token间延迟）降低，TTFP（首token耗时）增加
    enable_fp8_kvcache: bool = False # 是否启用FP8 KVCache
    long_prefill_token_threshold: int = 512 # 长prompt阈值，超过将分块
    enable_ngram_spec_decode: bool = True # 是否启用Ngram投机解码：利用序列自身的ngram历史提出草稿token，由目标模型一次前向验证，可降低decode阶段延迟
    ngram_spec_num_tokens: int = 4 # Ngram投机解码每步最多提出的草稿token数（k）
    ngram_size: int = 2 # Ngram中n的大小，即匹配历史时使用的前缀token数

    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 16 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.enable_ngram_spec_decode:
            assert self.ngram_size >= 1
            assert self.ngram_spec_num_tokens >= 1
