from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.ngram_proposer import NgramProposer


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.chunk_size = config.long_prefill_token_threshold
        self.enable_ngram_spec_decode = config.enable_ngram_spec_decode
        if self.enable_ngram_spec_decode:
            self.ngram_proposer = NgramProposer(config.ngram_size, config.ngram_spec_num_tokens)
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque() # 冒号后表示对参数类型的注解，这里表示的是存储请求的双端队列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.enable_chunked_prefill:
            return self._schedule_decode_first()
        else:
            return self._schedule_prefill_first()

    def _schedule_decode_seq(self, seq: Sequence, token_budget: int) -> bool:
        # 调度一个decode序列（可能携带ngram草稿），返回False表示该序列自身被抢占
        drafts = self.ngram_proposer.propose(seq) if self.enable_ngram_spec_decode else []
        drafts = drafts[:max(0, token_budget - 1)] # 草稿占用批次token预算，至少保留1个token给当前序列

        # 空闲块连当前1个token都放不下时，抢占队尾请求
        while self._max_drafts(seq) < 0:
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                return False

        drafts = drafts[:self._max_drafts(seq)] # 草稿数不超过空闲块可容纳的位置
        self.block_manager.allocate_spec(seq, len(drafts))
        seq.draft_tokens = drafts
        seq.num_scheduled_tokens = 1 + len(drafts)
        seq.is_prefill = False
        return True

    def _max_drafts(self, seq: Sequence) -> int:
        # 现有块+空闲块所能容纳的草稿数，可能为负（当前token需要新块而无空闲块）
        num_blocks = len(seq.block_table) + len(self.block_manager.free_block_ids)
        return num_blocks * self.block_size - seq.num_tokens

    def _schedule_decode_first(self) -> tuple[list[Sequence], list[Sequence]]:
        # 调度算法：优先处理所有 Decode 请求，将剩下的 token 预算分配给 Prefill 请求，支持PD混合批次
        decode_scheduled_seqs = []
        prefill_scheduled_seqs = []
        token_budget = self.max_num_batched_tokens

        # 优先调度 Decode 序列（Ngram投机解码的草稿计入token预算）
        while self.running and len(decode_scheduled_seqs) < self.max_num_seqs and token_budget > 0:
            seq = self.running.popleft()
            if not self._schedule_decode_seq(seq, token_budget):
                break
            token_budget -= seq.num_scheduled_tokens
            decode_scheduled_seqs.append(seq)
        self.running.extendleft(reversed(decode_scheduled_seqs))

        # 剩余的token分配给Chunked Prefill
        temp_waiting :deque[Sequence] = deque()
        while self.waiting and len(prefill_scheduled_seqs) + len(decode_scheduled_seqs) < self.max_num_seqs and token_budget > 0:
            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                self.block_manager.allocate(seq, num_cached_blocks)
                uncomputed_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                uncomputed_tokens = seq.num_tokens - seq.num_cached_tokens
            # 取未计算token、本批次token剩余预算、单批次单请求最长可计算token（分块大小）中的最小值
            seq.num_scheduled_tokens = min(uncomputed_tokens, token_budget, self.chunk_size)
            token_budget -= seq.num_scheduled_tokens
            self.waiting.popleft()
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                temp_waiting.append(seq)
            prefill_scheduled_seqs.append(seq)
        self.waiting.extendleft(reversed(temp_waiting))

        return decode_scheduled_seqs, prefill_scheduled_seqs

    def _schedule_prefill_first(self) -> tuple[list[Sequence], list[Sequence]]:
        # 调度算法：优先处理所有 Prefill 请求，直到没有新的 Prefill 请求需要处理，才开始处理 Decode 请求，不支持PD混合批次
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size # 待计算的tokens
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq，如果一批次装不下下一请求，则将下一请求直接放在下一批次
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining) # 本轮计算的tokens
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return [], scheduled_seqs

        # decode（Ngram投机解码的草稿计入批次token数）
        scheduled_seqs = []
        num_batched_tokens = 0
        while self.running and len(scheduled_seqs) < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            seq = self.running.popleft()
            token_budget = self.max_num_batched_tokens - num_batched_tokens
            if not self._schedule_decode_seq(seq, token_budget):
                break
            num_batched_tokens += seq.num_scheduled_tokens
            scheduled_seqs.append(seq)
        assert scheduled_seqs

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, []

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.draft_tokens = []
        seq.num_scheduled_tokens = 0
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], seq_tokens: list[list[int]]):
        # seq_tokens与seqs对齐：decode序列为本轮接受的1~k+1个token，prefill序列为末位采样的1个token
        for seq, new_tokens in zip(seqs, seq_tokens):
            if seq.is_prefill and seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens:
                # prefill阶段中，只有最后一个token才能得到第一个推理的token，分块中间块不产生输出token
                self.block_manager.hash_blocks(seq, seq.num_scheduled_tokens)
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                continue

            if seq.is_prefill:
                # prefill最后一块：所有scheduled token的KV均有效，仅末位logits采样出1个token
                num_new_kv = seq.num_scheduled_tokens
                output_tokens = new_tokens[:1]
            else:
                # decode（含Ngram投机解码）：草稿被拒绝的位置KV无效，只有被接受的token计入
                num_new_kv = len(new_tokens)
                output_tokens = new_tokens

            self.block_manager.hash_blocks(seq, num_new_kv)
            seq.num_cached_tokens += num_new_kv
            seq.num_scheduled_tokens = 0
            seq.draft_tokens = []

            self._append_output_tokens(seq, output_tokens)
            if not seq.is_prefill:
                # 回收被拒绝草稿占用的尾部块（需在token追加之后，按新序列长度判断）
                self.block_manager.release_tail_blocks(seq)

    def _append_output_tokens(self, seq: Sequence, token_ids: list[int]):
        for token_id in token_ids:
            if seq.num_completion_tokens == seq.max_tokens:
                break # 达到最大输出长度，截断后续token
            seq.append_token(token_id)
            seq.token_generate_time.append(perf_counter())
            if not seq.ignore_eos and token_id == self.eos:
                break

        if (not seq.ignore_eos and seq.last_token == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            if seq in self.running:
                self.running.remove(seq)
