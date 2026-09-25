import atexit
from random import randint, seed
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn") # PyTorch多进程并行指定的一种安全进程启动方式，后续新建进程（ctx.Process）会使用该方式
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        dseqs, pseqs = self.scheduler.schedule()
        seq_tokens = self.model_runner.call("run", dseqs, pseqs)
        self.scheduler.postprocess(dseqs+pseqs, seq_tokens)
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.token_generate_time) for seq in dseqs if seq.is_finished]
        return outputs

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams] = None,
    ) -> list[str]:
        if sampling_params is None:
            sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1024)]*len(prompts)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp) # 将请求逐一加入调度器
        outputs = {}
        while not self.is_finished():
            output = self.step()
            for seq_id, token_ids, _ in output:
                outputs[seq_id] = token_ids

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
    
    def bench(
        self,
        num_seqs: int, # 请求数
        max_isl: int, # 最大输入长度
        max_osl: int, # 最大输出长度
        ):
        seed(0)
        prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_isl))] for _ in range(num_seqs)]
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_osl)) for _ in range(num_seqs)]

        for prompt, sp in zip(prompt_token_ids, sampling_params):
            self.add_request(prompt, sp)

        pbar = tqdm(total=num_seqs, desc="Generating", dynamic_ncols=True)
        e2e = [] # 每个请求的总推理时间
        ttft = [] # 每个请求的首token延迟
        itl = [] # 每个请求decode阶段的平均token生成时间
        outputs = []

        start_time = perf_counter()
        while not self.is_finished():
            output = self.step()
            for _, _, tokens_generate_time in output:
                outputs.append([token_generate_time - start_time for token_generate_time in tokens_generate_time])
                pbar.update(1)
        total_time = perf_counter() - start_time
        pbar.close()

        for i in range(num_seqs):
            e2e.append(outputs[i][-1])
            ttft.append(outputs[i][0])
            itls = [outputs[i][j+1] - outputs[i][j] for j in range(len(outputs[i])-1)]
            itl.append(sum(itls)/len(itls))

        e2e.sort()
        ttft.sort()
        itl.sort()

        p99_e2e = e2e[int(0.99*num_seqs)]
        p99_ttft = ttft[int(0.99*num_seqs)]
        p99_itl = itl[int(0.99*num_seqs)]
        
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / total_time
        print(f"P99 E2E: {p99_e2e:.2f}s, P99 TTFT: {p99_ttft:.2f}s, P99 ITL: {p99_itl:.4f}s")
        print(f"Total: {total_tokens}tok, Time: {total_time:.2f}s, Throughput: {throughput:.2f}tok/s")
              

