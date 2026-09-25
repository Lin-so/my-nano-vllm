from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    split_pt: int = 0 # PD混合批次中，D和P的分离点
    num_spec_seqs: int = 0 # 变长组中需要输出全部位置logits的投机解码序列数（位于变长组最前面）
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

_UNSET = object()

def set_context(split_pt=_UNSET, num_spec_seqs=_UNSET, cu_seqlens_q=_UNSET, cu_seqlens_k=_UNSET,
                max_seqlen_q=_UNSET, max_seqlen_k=_UNSET, slot_mapping=_UNSET,
                context_lens=_UNSET, block_tables=_UNSET):
    ctx = _CONTEXT
    if split_pt      is not _UNSET: ctx.split_pt      = split_pt
    if num_spec_seqs is not _UNSET: ctx.num_spec_seqs = num_spec_seqs
    if cu_seqlens_q  is not _UNSET: ctx.cu_seqlens_q  = cu_seqlens_q
    if cu_seqlens_k  is not _UNSET: ctx.cu_seqlens_k  = cu_seqlens_k
    if max_seqlen_q  is not _UNSET: ctx.max_seqlen_q  = max_seqlen_q
    if max_seqlen_k  is not _UNSET: ctx.max_seqlen_k  = max_seqlen_k
    if slot_mapping  is not _UNSET: ctx.slot_mapping  = slot_mapping
    if context_lens  is not _UNSET: ctx.context_lens  = context_lens
    if block_tables  is not _UNSET: ctx.block_tables  = block_tables
    
def reset_context():
    ctx = _CONTEXT
    ctx.split_pt = 0
    ctx.num_spec_seqs = 0
    ctx.cu_seqlens_q = None
    ctx.cu_seqlens_k = None
    ctx.max_seqlen_q = 0
    ctx.max_seqlen_k = 0
    ctx.slot_mapping = None
    ctx.context_lens = None
    ctx.block_tables = None
