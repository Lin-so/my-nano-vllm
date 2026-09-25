import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/home/models/Qwen3-0.6B")
    # path = os.path.expanduser("/home/dministrator/Qwen3-4B-FP8")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    # prompts = [
    #     "introduce yourself",
    #     "list all prime numbers within 100",
    #     "请写一篇关于人工智能发展历程的短文，从图灵测试讲到 ChatGPT。",
    #     "What is the meaning of life?",
    #     "Write a short poem about the ocean",
    # ]
    prompts = [
        # "这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？",
        # "这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？",
        # "这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？",
        "这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？1",
        "这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？2",
        "这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？这篇论文的主要贡献是什么？3",
    ]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": 
              '''请根据以下文档回答问题：
              《Attention Is All You Need》论文提出了 Transformer 架构，
              完全基于自注意力机制，摒弃了循环和卷积。
              Transformer 由编码器和解码器组成，每层包含多头自注意力和前馈网络。
              该架构在机器翻译任务上取得了当时最优结果，并成为后续大语言模型的基础。
            '''},
                {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
