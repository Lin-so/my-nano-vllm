from nanovllm.engine.sequence import Sequence


class NgramProposer:
    # 基于序列自身token历史的Ngram草稿提出器，思路与vLLM的NgramPromptProposer相同：
    # 查找末尾ngram在历史中的最近一次出现，将其后的token作为草稿提出

    def __init__(self, ngram_size: int, num_spec_tokens: int):
        self.ngram_size = ngram_size
        self.num_spec_tokens = num_spec_tokens

    def propose(self, seq: Sequence) -> list[int]:
        n, k = self.ngram_size, self.num_spec_tokens
        tokens = seq.token_ids
        seq_len = len(tokens)
        if seq_len < n + 1:
            return []

        ngram = tokens[seq_len - n:]
        search_end = seq_len - n # 只在该位置之前查找，排除末尾ngram自身
        first_token = ngram[0]

        match = -1
        pos = 0
        while pos < search_end:
            try:
                cand = tokens.index(first_token, pos, search_end)
            except ValueError:
                break
            if tokens[cand:cand + n] == ngram:
                match = cand # 继续向前找，最终得到最近一次出现
            pos = cand + 1

        if match == -1:
            return []

        start = match + n
        end = min(start + k, seq_len)
        return tokens[start:end]
