import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.kernels.fp8_linear import BLOCK_SIZE, fp8_block_linear


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # quant_config: {"weight_block_size": (128, 128), "act_group_size": 128}，None 为普通BF16线性层
        self.quant_config = quant_config
        self.quantized = quant_config is not None
        if self.quantized:
            assert quant_config["weight_block_size"] == (BLOCK_SIZE, BLOCK_SIZE)
            # TP 切分后的维度仍需与 128 分块对齐
            assert output_size % BLOCK_SIZE == 0 and input_size % BLOCK_SIZE == 0
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn))
            self.weight_scale = nn.Parameter(
                torch.empty(output_size // BLOCK_SIZE, input_size // BLOCK_SIZE,
                            dtype=torch.bfloat16))
            self.weight_scale.weight_loader = self.weight_scale_loader
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.register_parameter("weight_scale", None)
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_scale_loader(self, param: nn.Parameter, loaded_scale: torch.Tensor,
                            loaded_shard_id=None):
        param.data.copy_(loaded_scale)

    def apply_linear(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        if self.quantized:
            y = fp8_block_linear(x, self.weight, self.weight_scale)
            if bias is not None:
                y = y + bias
            return y
        return F.linear(x, self.weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: dict | None = None,
    ):
        super().__init__(input_size, output_size, bias, quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_linear(x, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: dict | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0,
                         quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_scale: torch.Tensor,
                            loaded_shard_id=None):
        # scale: [N/BLOCK, K/BLOCK]，沿 N 方向（dim 0）按 TP 切分
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        param_data.copy_(loaded_scale.narrow(0, start_idx, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_linear(x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: dict | None = None,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_scale: torch.Tensor,
                            loaded_shard_id: int):
        # 权重 offset/size 除以 BLOCK_SIZE 即 scale 的 offset/size
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size // BLOCK_SIZE
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size // BLOCK_SIZE
        param_data = param.data.narrow(0, shard_offset, shard_size)
        loaded_scale = loaded_scale.chunk(self.tp_size, 0)[self.tp_rank]
        param_data.copy_(loaded_scale)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: dict | None = None,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quant_config=quant_config)
        if quant_config is not None:
            assert head_size % BLOCK_SIZE == 0, (
                f"head_size={head_size} 与 FP8 分块大小 {BLOCK_SIZE} 不对齐")

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_scale: torch.Tensor,
                            loaded_shard_id: str):
        # 权重的行 offset/size 除以 BLOCK_SIZE 即 scale 的 offset/size
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param.data.narrow(0, shard_offset // BLOCK_SIZE, shard_size // BLOCK_SIZE)
        loaded_scale = loaded_scale.chunk(self.tp_size, 0)[self.tp_rank]
        param_data.copy_(loaded_scale)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: dict | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1,
                         quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_scale: torch.Tensor,
                            loaded_shard_id=None):
        # scale: [N/BLOCK, K/BLOCK]，沿 K 方向（dim 1）按 TP 切分
        param_data = param.data
        shard_size = param_data.size(1)
        start_idx = self.tp_rank * shard_size
        param_data.copy_(loaded_scale.narrow(1, start_idx, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.tp_rank == 0 else None
        y = self.apply_linear(x, bias)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
