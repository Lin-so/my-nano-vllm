import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.kernels.fp8_attention import fp8_decode_attention, fp8_prefill_attention
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

        # KV cache 引用（由 ModelRunner.allocate_kv_cache 注入）
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q, k, v):
        context = get_context()
        split_pt = context.split_pt
        if self.k_cache.numel() > 0:
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
        
        block_tables = context.block_tables

        if block_tables is not None:
            d_block_tables = block_tables[:split_pt]
            p_block_tables = block_tables[split_pt:]
            if d_block_tables.numel() == 0: d_block_tables = None
            if p_block_tables.numel() == 0: p_block_tables = None
        else:
            d_block_tables = p_block_tables = None

        q_d = q[:split_pt]
        q_p = q[split_pt:]

        if p_block_tables is not None:
            k_p = self.k_cache
            v_p = self.v_cache
        else:
            k_p = k[split_pt:]
            v_p = v[split_pt:]

        outs = []
        if len(q_d) > 0:
            o_d = self._decode_attention(q_d, self.k_cache, self.v_cache, d_block_tables, context.context_lens)
            outs.append(o_d)
        if len(q_p) > 0:
            o_p = self._prefill_attention(q_p, k_p, v_p,
                                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                block_tables=p_block_tables)
            outs.append(o_p)

        o = torch.cat(outs,0)
        return o
    
    def _decode_attention(self, q, k, v, block_tables, context_lens):
        if k.dtype == torch.float8_e4m3fn:
            o = fp8_decode_attention(q, k, v,
                            block_tables, context_lens,
                            self.scale)
            return o
        else:
            # q: [batch_size, 1, num_heads, head_dim]
            # k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            # 在自回归任务中 decode 阶段 causal 参数设置为 True 和 False 没有任何区别
            o = flash_attn_with_kvcache(q.unsqueeze(1), k, v,
                            cache_seqlens=context_lens, block_table=block_tables, 
                            softmax_scale=self.scale, causal=True)
            
            return o.squeeze(1)

    def _prefill_attention(self, q, k, v, max_seqlen_q, max_seqlen_k, cu_seqlens_q, cu_seqlens_k, block_tables):
        if k.dtype == torch.float8_e4m3fn:
            # 只实现 causal = True 的场景
            o = fp8_prefill_attention(q, k, v,
                            max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
                            max_seqlen_k=max_seqlen_k, cu_seqlens_k=cu_seqlens_k,
                            softmax_scale=self.scale, block_tables=block_tables)
        else:
            o = flash_attn_varlen_func(q, k, v,
                            max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
                            max_seqlen_k=max_seqlen_k, cu_seqlens_k=cu_seqlens_k,
                            softmax_scale=self.scale, causal=True, block_table=block_tables)
        return o