import torch
import triton
import triton.language as tl

# FP8 E4M3 可表示的最大有限值
_FP8_MAX = tl.constexpr(448.0)
# compressed-tensors FP8_BLOCK 的 128x128 分块大小
BLOCK_SIZE = 128


@triton.jit
def fp8_dyn_quant_kernel(
    x_ptr, xq_ptr, sx_ptr,
    Kc,
    stride_xm,
    stride_qm,
    BLOCK_K: tl.constexpr,
):
    # grid = (M, Kc)：每个 program 量化一个 token 的一个 128 列分组
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    x = tl.load(x_ptr + pid_m * stride_xm + offs).to(tl.float32)
    amax = tl.max(tl.abs(x))
    scale = amax / _FP8_MAX
    scale = tl.where(scale > 0.0, scale, 1.0)  # 全零分组避免除零
    xq = tl.minimum(tl.maximum(x / scale, -_FP8_MAX), _FP8_MAX).to(tl.float8e4nv)

    tl.store(xq_ptr + pid_m * stride_qm + offs, xq)
    tl.store(sx_ptr + pid_m * Kc + pid_k, scale.to(sx_ptr.dtype.element_ty))


@triton.jit
def fp8_block_gemm_kernel(
    xq_ptr, sx_ptr, w_ptr, ws_ptr, out_ptr,
    M, Nc, Kc,
    stride_xm,
    stride_wn, stride_wk,
    stride_om,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # grid = (cdiv(M, BM), Nc)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)

    # xq[m,k] 直接取；w[n,k] 按 [k,n] 视角加载成 [BK, BN]，无需转置
    xp = xq_ptr + om[:, None] * stride_xm + ok[None, :]
    wp = w_ptr + ok[:, None] * stride_wk + on[None, :] * stride_wn
    mmask = (om < M)[:, None]

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for kc in range(Kc):
        xq = tl.load(xp, mask=mmask, other=0.0)
        wq = tl.load(wp)
        # 先得到该 128-K 分块的整数乘累加，再乘该块的 激活scale×权重scale，最后累加
        partial = tl.dot(xq, wq, out_dtype=tl.float32)
        s_act = tl.load(sx_ptr + om * Kc + kc, mask=om < M, other=1.0).to(tl.float32)
        s_w = tl.load(ws_ptr + pid_n * Kc + kc).to(tl.float32)
        acc += partial * (s_act[:, None] * s_w)
        xp += BK
        wp += BK * stride_wk

    tl.store(
        out_ptr + om[:, None] * stride_om + on[None, :],
        acc.to(out_ptr.dtype.element_ty), mask=mmask,
    )


def _choose_block_m(M: int, Nc: int, num_sm: int) -> int:
    """选 M 方向 tile，使 grid 中 program 数尽量达到 ~1.5*SM，且 tile 不显著大于 M。

    实测在这些瘦长 decode GEMM（K 只有 20~76 个分块）上：
    BM128 从不占优；BM64 是大批量的稳定最优；低 Nc / 小 M 需要 BM32、BM16 提占用率。
    """
    target = 3 * num_sm // 2  # ~1.5 个 wave
    bm = 64
    if triton.cdiv(M, bm) * Nc < target:
        bm = 32
    if triton.cdiv(M, bm) * Nc < target:
        bm = 16
    # 避免 M 很小时大量行被 mask 掉却照算 MMA
    cap = max(16, M)
    while bm > cap:
        bm //= 2
    return bm


def fp8_block_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """
    分块量化 FP8 线性层：y = x @ weight.T
      ① 独立访存 kernel 对输入做逐 token、逐128组的动态量化
         （若放进 GEMM 内循环，其 amax 归约会串行卡住 tensor core，把 2× 算力吃光）
      ② 纯 FP8 tensor-core GEMM，循环内只做矩阵乘 + 轻量 scale 融合
    Args:
        x: [M, K] BF16/FP16 输入激活
        weight: [N, K] FP8 E4M3 权重
        weight_scale: [Nc, Kc] BF16 权重分块 scale，分块 128x128
    Returns:
        [M, N]，dtype/device 与 x 相同
    """
    M, K = x.shape
    N = weight.shape[0]
    Nc, Kc = weight_scale.shape
    assert weight.dtype == torch.float8_e4m3fn
    assert N == Nc * BLOCK_SIZE and K == Kc * BLOCK_SIZE

    # ① 激活动态量化
    xq = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M, Kc), device=x.device, dtype=torch.bfloat16)
    fp8_dyn_quant_kernel[(M, Kc)](
        x, xq, x_scale, Kc,
        x.stride(0), xq.stride(0),
        BLOCK_K=BLOCK_SIZE, num_warps=4,
    )

    # ② 纯 FP8 GEMM
    num_sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    BM = _choose_block_m(M, Nc, num_sm)
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, BM), Nc)
    fp8_block_gemm_kernel[grid](
        xq, x_scale, weight, weight_scale, out,
        M, Nc, Kc,
        xq.stride(0),
        weight.stride(0), weight.stride(1),
        out.stride(0),
        BM=BM, BN=BLOCK_SIZE, BK=BLOCK_SIZE,
        num_warps=4, num_stages=3,
    )
    return out
