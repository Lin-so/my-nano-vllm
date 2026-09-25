import random
from dataclasses import dataclass


@dataclass(slots=True)
class BenchCase:
    name: str                 # 用例名称
    prompt: str | list[int]   # 文本prompt或token id列表
    temperature: float = 0.6
    max_tokens: int = 128
    ignore_eos: bool = True


# ---------------- 文本素材 ----------------

DOCUMENTS = [
    """纳米 vLLM 是一个极简的大语言模型推理框架实现。它用尽量少的代码展示了一个现代推理引擎的核心机制，
包括 PagedAttention 的分页 KV Cache 管理、连续批处理（Contiguous Batching）、Chunked Prefill、
前缀缓存、张量并行以及 CUDA Graph 加速。学习该项目可以帮助读者理解 vLLM、TensorRT-LLM 等生产级
推理框架背后的原理。分页 KV Cache 将每个序列的 KV 缓存切分成固定大小的块，通过块表进行间接寻址，
从而彻底消除了 KV Cache 的显存碎片。""",
    """《Attention Is All You Need》论文于 2017 年提出 Transformer 架构。该架构完全基于自注意力机制，
摒弃了循环神经网络和卷积操作。Transformer 由编码器和解码器堆叠而成，每个子层包含多头自注意力、
前馈网络、残差连接与层归一化。多头注意力允许模型在不同表示子空间中并行关注不同位置的信息，
位置编码则为没有顺序归纳偏置的模型注入了 token 的位置信息。该架构深刻影响了后续 BERT、GPT、
Qwen、Llama 等几乎所有大语言模型的设计。""",
    """量子计算利用量子叠加和量子纠缠进行信息处理。经典比特在任一时刻只能处于 0 或 1 状态，
而量子比特可以处于两个基态的线性叠加态。量子门对量子比特的状态进行幺正变换，测量则使叠加态
以一定概率坍缩到某个基态。Shor 算法在大数分解问题上相对经典算法具有指数级加速潜力，
Grover 算法对无序搜索问题提供平方级加速。目前量子计算仍面临退相干、纠错和噪声等工程挑战，
含噪中等规模量子（NISQ）设备是现阶段的主要研究平台。""",
]

CODE_DOC = '''def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
'''

# ---------------- 各场景用例 ----------------

def short_qa_cases() -> list[BenchCase]:
    # 短输入短输出：模拟日常快速问答，对 TTFT 敏感
    questions = [
        "中国的首都是哪里？",
        "水的化学式是什么？",
        "请用一句话解释什么是光合作用。",
        "光速大约是多少？",
        "一年有多少个星期？",
        "What is the capital of France?",
        "Explain recursion in one sentence.",
        "What language is primarily spoken in Brazil?",
        "Who wrote the play Hamlet?",
        "How many continents are there on Earth?",
    ]
    return [BenchCase(f"short_qa_{i}", q, max_tokens=64) for i, q in enumerate(questions)]


def long_generation_cases() -> list[BenchCase]:
    # 短输入长输出：自由生成，长 decode，考察持续解码吞吐与 ITL 稳定性
    tasks = [
        "请以《未来的城市》为题写一篇科幻短文。",
        "写一个关于机器人学会做梦的故事。",
        "请撰写一篇介绍人工智能发展历史的文章，从早期符号主义讲到大语言模型。",
        "Write a detailed product review for a fictional smartphone that can read minds.",
        "请写一封写给十年后的自己的信。",
        "Write a fantasy story opening featuring a floating library.",
    ]
    return [BenchCase(f"long_gen_{i}", t, max_tokens=512) for i, t in enumerate(tasks)]


def long_context_cases() -> list[BenchCase]:
    # 长输入：长上下文理解，考察 prefill 性能与长文能力
    questions = [
        "请回答：纳米 vLLM 项目展示了哪些核心机制？",
        "分页 KV Cache 是如何消除显存碎片的？",
        "Transformer 架构由哪些部分组成？",
        "多头注意力的作用是什么？",
        "Shor 算法和 Grover 算法分别有什么加速优势？",
        "目前量子计算面临哪些工程挑战？",
    ]
    docs = [DOCUMENTS[i % len(DOCUMENTS)] for i in range(len(questions))]
    cases = []
    for i, (doc, q) in enumerate(zip(docs, questions)):
        # 重复文档构造长上下文
        prompt = f"以下是参考资料，请根据资料回答问题。\n\n资料：{doc * 4}\n\n问题：{q}\n答案："
        cases.append(BenchCase(f"long_ctx_{i}", prompt, max_tokens=128))
    return cases


def repetitive_cases() -> list[BenchCase]:
    # 强重复文本：Ngram 投机解码命中率高的场景
    sentences = [
        "我们需要认真讨论这个问题。",
        "The quick brown fox jumps over the lazy dog.",
        "实践是检验真理的唯一标准。",
        "To be, or not to be, that is the question.",
        "重要的事情说三遍。",
        "All work and no play makes Jack a dull boy.",
    ]
    cases = []
    for i, s in enumerate(sentences):
        prompt = f"请继续重复下面这句话，不要停下来：\n{s * 25}\n继续："
        cases.append(BenchCase(f"repetitive_{i}", prompt, max_tokens=256))
    return cases


def code_cases() -> list[BenchCase]:
    # 代码生成 / 解释 / 调试，输出结构化程度高
    tasks = [
        "用 Python 实现一个线程安全的单例模式，给出完整代码和注释。",
        "Write a Python function to find the longest common subsequence of two strings.",
        f"请解释下面这段代码的时间复杂度，并给出一个优化版本：\n{CODE_DOC}",
        "用 SQL 编写一个查询：找出每个部门薪资第二高的员工。",
        "Write a Rust function that reverses a linked list iteratively.",
        "请用 Python 实现一个简单的 LRU Cache，要求 get/put 均为 O(1)。",
        "Debug the following snippet and explain the bug:\n```js\nfor (var i = 0; i < 3; i++)\n  setTimeout(() => console.log(i), 100)\n```",
        "用 C++ 写一个生产者-消费者示例，使用互斥锁和条件变量。",
    ]
    return [BenchCase(f"code_{i}", t, max_tokens=256) for i, t in enumerate(tasks)]


def summarization_cases() -> list[BenchCase]:
    # 摘要：中等输入、短输出
    intros = ["请用三句话总结下面的文章：\n\n", "Summarize the following text in bullet points:\n\n"]
    cases = []
    for i in range(6):
        doc = DOCUMENTS[i % len(DOCUMENTS)] * 2
        prompt = f"{intros[i % len(intros)]}{doc}"
        cases.append(BenchCase(f"summarize_{i}", prompt, max_tokens=128))
    return cases


def reasoning_cases() -> list[BenchCase]:
    # 推理 / 数学：思维链较长
    tasks = [
        "一个水池有甲、乙两个进水管和丙一个出水管。甲单独开 6 小时注满，乙单独开 8 小时注满，丙单独开 12 小时放空。三管齐开多久注满？",
        "小明比小红大 3 岁，5 年后两人年龄之和是 27 岁。问他们现在各多少岁？",
        "A train travels 60 km in 45 minutes. What is its average speed in km/h?",
        "甲乙两人从相距 30 公里的两地相向而行，甲速 4 km/h，乙速 6 km/h，几小时后相遇？",
        "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Explain.",
        "一件商品先涨价 20%，再降价 20%，最终价格相比原价是高了还是低了？变化多少？",
        "How many integers between 1 and 200 are divisible by both 3 and 5?",
        "鸡兔同笼，共有头 35 个、脚 94 只，问鸡和兔各有多少只？",
    ]
    return [BenchCase(f"reasoning_{i}", t, max_tokens=256) for i, t in enumerate(tasks)]


def translation_cases() -> list[BenchCase]:
    # 翻译：中等输出
    pairs = [
        ("请将下面的中文翻译成英文：人工智能正在深刻改变医疗、教育和交通行业。", "cn2en"),
        ("Translate into Chinese: The best way to predict the future is to invent it.", "en2cn"),
        ("请将下面的中文翻译成英文：书山有路勤为径，学海无涯苦作舟。", "cn2en"),
        ("Translate into Chinese: Stay hungry, stay foolish.", "en2cn"),
        ("翻译成英文：这家创业公司专注于使用大模型优化企业客服流程。", "cn2en"),
        ("Translate into Chinese: Not all those who wander are lost.", "en2cn"),
        ("翻译成英文：春眠不觉晓，处处闻啼鸟。", "cn2en"),
        ("Translate into Chinese: A journey of a thousand miles begins with a single step.", "en2cn"),
    ]
    return [BenchCase(f"translate_{tag}_{i}", t, max_tokens=128) for i, (t, tag) in enumerate(pairs)]


def prefix_shared_cases() -> list[BenchCase]:
    # 共享长前缀 + 不同问题：考察前缀缓存（Prefix Caching）
    shared = f"系统使用手册\n{DOCUMENTS[0] * 2}\n{DOCUMENTS[1] * 2}\n"
    questions = [
        "这个框架的 KV Cache 是怎么管理的？",
        "Chunked Prefill 有什么作用？",
        "支持张量并行吗？",
        "Transformer 的编码器包含什么？",
        "多头注意力是什么？",
        "位置编码解决了什么问题？",
        "CUDA Graph 是必须启用的吗？",
        "前缀缓存是如何工作的？",
    ]
    return [BenchCase(f"prefix_shared_{i}", shared + f"\n用户问题：{q}\n回答：", max_tokens=128)
            for i, q in enumerate(questions)]


def random_token_cases(vocab_size: int) -> list[BenchCase]:
    # 合成随机 token 输入：无任何语义先验，用于压测
    rng = random.Random(0)
    lengths = [128, 256, 384, 512, 640, 768, 896, 1024]
    return [BenchCase(f"random_tokens_{l}",
                      [rng.randrange(vocab_size) for _ in range(l)],
                      max_tokens=128) for l in lengths]


def build_groups(vocab_size: int) -> dict[str, list[BenchCase]]:
    groups = {
        "short_qa": short_qa_cases(),
        "long_generation": long_generation_cases(),
        "long_context": long_context_cases(),
        "repetitive": repetitive_cases(),
        "code": code_cases(),
        "summarization": summarization_cases(),
        "reasoning": reasoning_cases(),
        "translation": translation_cases(),
        "prefix_shared": prefix_shared_cases(),
        "random_tokens": random_token_cases(vocab_size),
    }

    # 混合场景：每个组抽一个，模拟真实线上流量
    mixed = []
    for name, cases in groups.items():
        mixed.append(BenchCase(f"mixed_{name}", cases[0].prompt,
                               temperature=cases[0].temperature,
                               max_tokens=min(cases[0].max_tokens, 192),
                               ignore_eos=cases[0].ignore_eos))
    groups["mixed"] = mixed
    return groups
