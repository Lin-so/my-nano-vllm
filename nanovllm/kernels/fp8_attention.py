import torch
import triton
import triton.language as tl

# triton调用时，需在[]中指定grid, grid中每个program将分别启动执行以下代码, 如fp8_paged_attention_decode_kernel[(256, 32)](...)
# 参数中的torch.Tensor在传入时会转为地址ptr
# tl.constexpr是编译期常量，不可在执行过程中修改
@triton.jit
def fp8_paged_attention_decode_kernel_block_n_1(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
    block_tables_ptr, context_lens_ptr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    softmax_scale: tl.constexpr,
):
    # grid = (num_seqs, num_heads)
    # 每个 program 是一个并行执行实例，类似 CUDA 里的一个 block
    seq_idx = tl.program_id(0) # 序列索引
    head_idx = tl.program_id(1) # 注意力头索引

    # GQA 分组查询注意力, 每n个Q头对应1个KV头
    num_heads_per_kv = num_heads // num_kv_heads
    kv_head_idx = head_idx // num_heads_per_kv

    # context_lens[seq_idx]，记录的上下文长度
    context_len = tl.load(context_lens_ptr + seq_idx)

    # Q[num_seqs, num_heads, head_dim] ,获取 Q[seq_idx, head_idx] 的Q头
    q_offset = seq_idx * num_heads * head_dim + head_idx * head_dim
    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(q_ptr + q_offset + dim_offsets).to(tl.float32)

    # Online softmax 状态
    m_old = float("-inf")
    s = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    # 遍历所有 block（使用 constexpr 边界 + mask）
    for block_idx in range(max_num_blocks):
        block_start = block_idx * block_size

        # 跳过超出 context_len 的 block（triton是不支持break的，通过添加掩码跳过load）
        block_valid = block_start < context_len

        # block_tables[seq_idx, block_idx]
        physical_block = tl.load(
            block_tables_ptr + seq_idx * max_num_blocks + block_idx,
            mask=block_valid, other=0
        )

        # 遍历 block 内的 token（constexpr 循环 + mask）
        for token_offset in range(block_size):
            token_pos = block_start + token_offset
            valid = token_pos < context_len

            # KV cache 偏移
            slot = physical_block * block_size + token_offset
            kv_offset = slot * num_kv_heads * head_dim + kv_head_idx * head_dim

            # 加载 FP8 KV 并反量化
            k = tl.load(k_cache_ptr + kv_offset + dim_offsets,
                        mask=valid, other=0.0).to(tl.float32)
            v = tl.load(v_cache_ptr + kv_offset + dim_offsets,
                        mask=valid, other=0.0).to(tl.float32)

            # QK^T (scaled dot product)
            score = tl.sum(q * k) * softmax_scale

            # 无效位置设为 -inf（不影响 softmax）
            score = tl.where(valid, score, float("-inf"))

            # Online softmax update
            m_new = tl.maximum(m_old, score)
            correction = tl.exp(m_old - m_new)
            p = tl.exp(score - m_new)
            s = s * correction + p
            acc = acc * correction + p * v
            m_old = m_new

    # 归一化输出
    out = (acc / tl.maximum(s, 1e-10)).to(tl.float16)
    out_offset = seq_idx * num_heads * head_dim + head_idx * head_dim
    tl.store(out_ptr + out_offset + dim_offsets, out)

@triton.jit
def fp8_paged_attention_decode_kernel(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
    block_tables_ptr, context_lens_ptr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_N: tl.constexpr,   # 每次处理的 KV token 数，需整除 block_size，
    max_num_blocks: tl.constexpr,
    softmax_scale: tl.constexpr,
):
    # grid = (num_seqs, num_heads)
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    num_heads_per_kv = num_heads // num_kv_heads
    kv_head_idx = head_idx // num_heads_per_kv

    context_len = tl.load(context_lens_ptr + seq_idx)

    # 加载 Q: [HEAD_DIM]
    q_offset = seq_idx * num_heads * head_dim + head_idx * head_dim
    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(q_ptr + q_offset + dim_offsets).to(tl.float32)

    # Online softmax 状态（标量）
    m_old = float("-inf")
    s = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    # 用于加载 [BLOCK_N, HEAD_DIM] 的偏移
    n_offsets = tl.arange(0, BLOCK_N)

    # 遍历所有物理 block
    for block_idx in range(max_num_blocks):
        block_start = block_idx * block_size
        block_valid = block_start < context_len

        physical_block = tl.load(
            block_tables_ptr + seq_idx * max_num_blocks + block_idx,
            mask=block_valid, other=0
        )

        # 在 block 内部按 BLOCK_N 分块处理
        for chunk in range(block_size // BLOCK_N):
            chunk_start = block_start + chunk * BLOCK_N
            token_pos = chunk_start + n_offsets            # [BLOCK_N]
            valid = (token_pos < context_len) & block_valid  # [BLOCK_N]

            # 物理槽位：physical_block 内的偏移
            slot = physical_block * block_size + chunk * BLOCK_N + n_offsets  # [BLOCK_N]

            # KV 指针: [BLOCK_N, HEAD_DIM]
            kv_offset = slot[:, None] * (num_kv_heads * head_dim) \
                        + kv_head_idx * head_dim
            ptrs = kv_offset + dim_offsets[None, :]        # [BLOCK_N, HEAD_DIM]

            # 一次性加载整个 chunk 的 K、V 并反量化到 fp32
            k = tl.load(k_cache_ptr + ptrs,
                        mask=valid[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_cache_ptr + ptrs,
                        mask=valid[:, None], other=0.0).to(tl.float32)

            # QK^T: [BLOCK_N]
            scores = tl.sum(q[None, :] * k, axis=1) * softmax_scale
            # 无效位置置 -inf，不影响 softmax
            scores = tl.where(valid, scores, float("-inf"))

            # ---- 向量化 online softmax 更新 ----
            m_block = tl.max(scores, axis=0)               # 标量
            m_new = tl.maximum(m_old, m_block)
            correction = tl.exp(m_old - m_new)
            p = tl.exp(scores - m_new)                     # [BLOCK_N]

            s = s * correction + tl.sum(p, axis=0)
            # [BLOCK_N, 1] * [BLOCK_N, HEAD_DIM] -> 对 axis=0 求和得到 [HEAD_DIM]
            acc = acc * correction + tl.sum(p[:, None] * v, axis=0)
            m_old = m_new

    # 归一化输出
    out = (acc / tl.maximum(s, 1e-10)).to(tl.float16)
    out_offset = seq_idx * num_heads * head_dim + head_idx * head_dim
    tl.store(out_ptr + out_offset + dim_offsets, out)

def fp8_decode_attention(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    block_tables: torch.Tensor, context_lens: torch.Tensor, softmax_scale: float,
) -> torch.Tensor:
    """
    Args:
        q: [batch_size, num_heads, head_dim]
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim] (FP8)
        v_cache: 同上
        block_tables: [batch_size, max_num_blocks]
        context_lens: [batch_size]
        softmax_scale: attention scaling factor (1/sqrt(head_dim))

    Returns:
        [batch_size, num_heads, head_dim] attention 输出
    """
    num_heads = q.shape[1]
    num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    batch_size, max_num_blocks = block_tables.shape

    out = torch.empty_like(q)
    grid = (batch_size, num_heads)

    fp8_paged_attention_decode_kernel[grid](
        out, q, k_cache.reshape(-1), v_cache.reshape(-1),
        block_tables, context_lens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        BLOCK_N=block_size, # 默认将BLOCK_N设置为block_size，或修改为可整除block_size的数
        max_num_blocks=max_num_blocks,
        softmax_scale=softmax_scale,
    )
    return out

@triton.jit
def fp8_paged_attention_prefill_kernel(
    out_ptr, q_ptr, k_ptr, v_ptr,
    block_tables_ptr, cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr, # 每次处理的 Query token 数
    BLOCK_N: tl.constexpr, # 每次处理的 KV token 数
    kv_block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    softmax_scale: tl.constexpr,      
):
    seq_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    num_heads_per_kv = num_heads // num_kv_heads
    kv_head_idx = head_idx // num_heads_per_kv

    start_token_k = tl.load(cu_seqlens_k_ptr + seq_idx)
    end_token_k = tl.load(cu_seqlens_k_ptr + seq_idx + 1)
    num_tokens_kv = end_token_k - start_token_k

    start_token_q = tl.load(cu_seqlens_q_ptr + seq_idx)
    end_token_q = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    num_tokens_q = end_token_q - start_token_q

    computed_tokens = num_tokens_kv - num_tokens_q

    # arange中参数必须是constexpr
    offs_n = tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    q_tokens = block_idx * BLOCK_M + offs_m  # [BLOCK_M]
    q_valid = q_tokens < num_tokens_q # [BLOCK_M]

    q_offset = ((start_token_q + q_tokens) * num_heads + head_idx) * head_dim
    q = tl.load(q_ptr + q_offset[:, None] + offs_d[None, :], mask=q_valid[:, None], other=0.0).to(tl.float32) # [BLOCK_M, head_dim]

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)   # [BLOCK_M]
    l_i = tl.zeros([BLOCK_M], tl.float32)                 # [BLOCK_M]
    acc = tl.zeros([BLOCK_M, head_dim], tl.float32)       # [BLOCK_M, HEAD_DIM]

    if block_tables_ptr is None:
        # k[num_tokens, num_kv_heads, head_dim]
        num_kv_blocks = (num_tokens_kv + BLOCK_N - 1) // BLOCK_N
        for block_kv_idx in range(num_kv_blocks):
            kv_tokens = block_kv_idx * BLOCK_N + offs_n
            kv_valid = kv_tokens < num_tokens_kv
            kv_offset = ((start_token_k + kv_tokens) * num_kv_heads + kv_head_idx) * head_dim # [BLOCK_N]

            k = tl.load(k_ptr + kv_offset[:, None] + offs_d[None, :], mask=kv_valid[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_ptr + kv_offset[:, None] + offs_d[None, :], mask=kv_valid[:, None], other=0.0).to(tl.float32) # [BLOCK_N, HEAD_DIM]
            scores = tl.dot(q, tl.trans(k)) * softmax_scale

            # ---- 因果掩码 ----
            causal_mask = kv_tokens[None, :] <= (q_tokens[:, None] + computed_tokens)   # [BLOCK_M, BLOCK_N]
            scores = tl.where(causal_mask & kv_valid[None, :], scores, float("-inf"))

            # ---- Online Softmax ----
            m_ij = tl.max(scores, axis=1)                    # [BLOCK_M]
            m_new = tl.maximum(m_i, m_ij)                    # [BLOCK_M]
            alpha = tl.exp(m_i - m_new)                      # [BLOCK_M]
            p = tl.exp(scores - m_new[:, None])              # [BLOCK_M, BLOCK_N]

            l_i = l_i * alpha + tl.sum(p, axis=1)            # [BLOCK_M]
            acc = acc * alpha[:, None] + tl.dot(p, v)        # [BLOCK_M, HEAD_DIM]
            m_i = m_new
    else:
        # k[num_blocks, block_size, num_kv_heads, head_dim]
        for block_table_idx in range(max_num_blocks):
            block_start = block_table_idx * kv_block_size
            block_valid = block_start < num_tokens_kv
            physical_block = tl.load(block_tables_ptr + seq_idx * max_num_blocks + block_table_idx, mask=block_valid, other=0)
            for block_kv_idx in range(kv_block_size // BLOCK_N):
                kv_tokens = block_start + block_kv_idx * BLOCK_N + offs_n
                kv_valid = kv_tokens < num_tokens_kv

                slot = physical_block * kv_block_size + block_kv_idx * BLOCK_N + offs_n
                kv_offset = (slot * num_kv_heads + kv_head_idx) * head_dim # [BLOCK_N]

                k = tl.load(k_ptr + kv_offset[:, None] + offs_d[None, :], mask=kv_valid[:, None], other=0.0).to(tl.float32)
                v = tl.load(v_ptr + kv_offset[:, None] + offs_d[None, :], mask=kv_valid[:, None], other=0.0).to(tl.float32) # [BLOCK_N, HEAD_DIM]
                scores = tl.dot(q, tl.trans(k)) * softmax_scale

                # ---- 因果掩码 ----
                causal_mask = kv_tokens[None, :] <= (q_tokens[:, None] + computed_tokens)   # [BLOCK_M, BLOCK_N]
                scores = tl.where(causal_mask & kv_valid[None, :], scores, float("-inf"))

                # ---- 向量化 Online Softmax 更新 ----
                m_ij = tl.max(scores, axis=1)                    # [BLOCK_M]
                m_new = tl.maximum(m_i, m_ij)                    # [BLOCK_M]
                alpha = tl.exp(m_i - m_new)                      # [BLOCK_M]
                p = tl.exp(scores - m_new[:, None])              # [BLOCK_M, BLOCK_N]

                l_i = l_i * alpha + tl.sum(p, axis=1)            # [BLOCK_M]
                acc = acc * alpha[:, None] + tl.dot(p, v)        # [BLOCK_M, HEAD_DIM]
                m_i = m_new

    # out = (acc / l_i[:, None]).to(tl.float16)
    out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0).to(tl.float16)
    tl.store(out_ptr + q_offset[:, None] + offs_d[None, :], out, mask=q_valid[:, None])

def fp8_prefill_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_tables: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, 
    max_seqlen_q: int, max_seqlen_k: int, softmax_scale: float,
):
    num_tokens, num_heads, head_dim = q.shape
    num_seqs = cu_seqlens_q.shape[0] - 1

    if num_tokens <= 16:
        BLOCK_M = 16
    else:
        BLOCK_M = 32

    # 分连续存储和分页存储两种情况讨论
    if block_tables is None:
        num_kv_heads = k.shape[1]
        max_num_blocks = None
        kvcache_block_size = BLOCK_N = BLOCK_M
    else:
        kvcache_block_size, num_kv_heads = k.shape[1:3]
        max_num_blocks = block_tables.shape[1]
        BLOCK_N = min(kvcache_block_size, 32)

    out = torch.empty_like(q)

    grid = (num_seqs, (max_seqlen_q + BLOCK_M - 1) // BLOCK_M, num_heads)

    fp8_paged_attention_prefill_kernel[grid](
        out, q, k.reshape(-1), v.reshape(-1),
        block_tables, cu_seqlens_q, cu_seqlens_k, 
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        kv_block_size=kvcache_block_size,
        max_num_blocks=max_num_blocks,
        softmax_scale=softmax_scale,
    )
    return out
