import os
import json
import math
import argparse
from time import perf_counter
from collections import Counter

import torch

from nanovllm import LLM, SamplingParams
from bench_cases import BenchCase, build_groups


# ---------------- 统计辅助 ----------------

def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def summarize(values: list[float]) -> dict:
    vals = sorted(values)
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals) if vals else float("nan"),
        "min": vals[0] if vals else float("nan"),
        "p50": percentile(vals, 0.50),
        "p90": percentile(vals, 0.90),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "max": vals[-1] if vals else float("nan"),
    }


def fmt_ms(x: float) -> str:
    return f"{x * 1000:8.2f}"


def print_dist_table(title: str, dists: dict[str, dict]):
    # dists: 行名 -> summarize结果；时间以ms打印
    print(f"  {title:<14}{'mean':>9}{'min':>9}{'p50':>9}{'p90':>9}{'p95':>9}{'p99':>9}{'max':>9}")
    for name, s in dists.items():
        print(f"  {name:<14}{s['mean']*1000:9.2f}{s['min']*1000:9.2f}{s['p50']*1000:9.2f}"
              f"{s['p90']*1000:9.2f}{s['p95']*1000:9.2f}{s['p99']*1000:9.2f}{s['max']*1000:9.2f}")


# ---------------- 投机解码统计 ----------------

def install_spec_hook(llm: LLM, orig_post, stats: dict):
    # 包装scheduler.postprocess，在调度器处理前统计每个decode序列的草稿接受情况
    def hook(seqs, seq_tokens):
        stats["steps"] += 1
        for seq, accepted in zip(seqs, seq_tokens):
            if seq.is_prefill:
                continue
            stats["decode_entries"] += 1
            num_drafts = len(seq.draft_tokens)
            if num_drafts == 0:
                continue
            stats["draft_steps"] += 1
            stats["proposed"] += num_drafts
            stats["accepted_drafts"] += min(len(accepted) - 1, num_drafts)
            stats["accept_hist"][len(accepted)] += 1
            if len(accepted) == num_drafts + 1:  # 草稿全接受并产出bonus token
                stats["bonus_steps"] += 1
        return orig_post(seqs, seq_tokens)
    llm.scheduler.postprocess = hook


def new_spec_stats() -> dict:
    return {"steps": 0, "decode_entries": 0, "draft_steps": 0,
            "proposed": 0, "accepted_drafts": 0, "bonus_steps": 0,
            "accept_hist": Counter()}


def print_spec_stats(stats: dict):
    if stats["draft_steps"] == 0:
        print("  [投机解码] 本轮没有序列提出草稿")
        return
    rate = stats["accepted_drafts"] / stats["proposed"]
    avg_accept = (stats["accepted_drafts"] + stats["draft_steps"]) / stats["draft_steps"]
    bonus_rate = stats["bonus_steps"] / stats["draft_steps"]
    print(f"  [投机解码] 调度步数={stats['steps']}, decode序列条目={stats['decode_entries']}, "
          f"提出草稿步数={stats['draft_steps']}")
    print(f"  [投机解码] 提出草稿={stats['proposed']}, 接受草稿={stats['accepted_drafts']}, "
          f"草稿接受率={rate:.2%}, 每草稿步平均产出={avg_accept:.2f} token, bonus率={bonus_rate:.2%}")
    hist = dict(sorted(stats["accept_hist"].items()))
    print(f"  [投机解码] 接受长度(含correction/bonus)直方图: {hist}")


# ---------------- 单场景执行与指标汇总 ----------------

def run_group(llm: LLM, cases: list[BenchCase]) -> tuple[dict, float]:
    for case in cases:
        sp = SamplingParams(temperature=case.temperature,
                            max_tokens=case.max_tokens,
                            ignore_eos=case.ignore_eos)
        llm.add_request(case.prompt, sp)

    t0 = perf_counter()
    records = {}
    while not llm.is_finished():
        dseqs, pseqs = llm.scheduler.schedule()
        seq_tokens = llm.model_runner.call("run", dseqs, pseqs)
        llm.scheduler.postprocess(dseqs + pseqs, seq_tokens)
        for seq in dseqs + pseqs:
            if seq.is_finished and seq.seq_id not in records:
                records[seq.seq_id] = {
                    "prompt_tokens": seq.num_prompt_tokens,
                    "completion_tokens": seq.num_completion_tokens,
                    "times": [t - t0 for t in seq.token_generate_time],
                }
    return records, perf_counter() - t0


def build_metrics(records: dict, wall: float, spec_stats: dict) -> dict:
    recs = list(records.values())
    ttft = [r["times"][0] for r in recs if r["times"]]
    e2e = [r["times"][-1] for r in recs if r["times"]]
    per_req_itl = []
    all_itl = []
    for r in recs:
        gaps = [b - a for a, b in zip(r["times"], r["times"][1:])]
        all_itl.extend(gaps)
        if gaps:
            per_req_itl.append(sum(gaps) / len(gaps))

    prompt_toks = sum(r["prompt_tokens"] for r in recs)
    completion_toks = sum(r["completion_tokens"] for r in recs)

    return {
        "num_requests": len(recs),
        "wall_time": wall,
        "prompt_tokens": prompt_toks,
        "completion_tokens": completion_toks,
        "avg_isl": prompt_toks / len(recs) if recs else 0,
        "avg_osl": completion_toks / len(recs) if recs else 0,
        "request_throughput": len(recs) / wall,
        "output_token_throughput": completion_toks / wall,
        "total_token_throughput": (prompt_toks + completion_toks) / wall,
        "ttft": summarize(ttft),
        "itl": summarize(all_itl),
        "per_request_itl": summarize(per_req_itl),
        "e2e": summarize(e2e),
        "spec": {
            "steps": spec_stats["steps"],
            "decode_entries": spec_stats["decode_entries"],
            "draft_steps": spec_stats["draft_steps"],
            "proposed": spec_stats["proposed"],
            "accepted_drafts": spec_stats["accepted_drafts"],
            "draft_accept_rate": (spec_stats["accepted_drafts"] / spec_stats["proposed"]
                                  if spec_stats["proposed"] else None),
            "bonus_steps": spec_stats["bonus_steps"],
            "accept_hist": dict(sorted(spec_stats["accept_hist"].items())),
        },
    }


def print_group_report(name: str, requested: int, m: dict, spec_enabled: bool):
    print("=" * 78)
    print(f"场景: {name}  (提交 {requested} / 完成 {m['num_requests']})")
    print("-" * 78)
    print(f"  总耗时: {m['wall_time']:.2f}s")
    print(f"  Prompt tokens: {m['prompt_tokens']} (avg {m['avg_isl']:.1f}), "
          f"Completion tokens: {m['completion_tokens']} (avg {m['avg_osl']:.1f})")
    print(f"  请求吞吐: {m['request_throughput']:.2f} req/s | "
          f"输出token吞吐: {m['output_token_throughput']:.2f} tok/s | "
          f"总token吞吐: {m['total_token_throughput']:.2f} tok/s")
    print_dist_table("延迟(ms)", {
        "TTFT": m["ttft"],
        "ITL": m["itl"],
        "ITL/req": m["per_request_itl"],
        "E2E": m["e2e"],
    })
    if spec_enabled:
        print_spec_stats_local(m["spec"])


def print_spec_stats_local(spec: dict):
    if spec["draft_steps"] == 0:
        print("  [投机解码] 本场景没有序列提出草稿")
        return
    rate = spec["draft_accept_rate"]
    hist = spec["accept_hist"]
    bonus_rate = spec["bonus_steps"] / spec["draft_steps"]
    print(f"  [投机解码] 草稿步={spec['draft_steps']}, 提出={spec['proposed']}, "
          f"接受={spec['accepted_drafts']}, 接受率={rate:.2%}, bonus率={bonus_rate:.2%}")
    print(f"  [投机解码] 接受长度直方图: {hist}")


# ---------------- 参数解析 ----------------

def parse_args():
    p = argparse.ArgumentParser(description="nanovllm 多场景 Benchmark")
    p.add_argument("--model", default="/home/models/Qwen3-4B")
    p.add_argument("--eager", action="store_true", help="禁用 CUDA Graph")
    p.add_argument("--spec", action="store_true", help="开启 Ngram 投机解码")
    p.add_argument("--ngram-size", type=int, default=2)
    p.add_argument("--spec-tokens", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-batched-tokens", type=int, default=8192)
    p.add_argument("--replicas", type=int, default=1, help="每条用例复制份数，用于提高并发")
    p.add_argument("--groups", default="", help="逗号分隔的场景名，默认全部")
    p.add_argument("--warmup", type=int, default=1, help="预热轮数")
    p.add_argument("--json", default="", help="将完整报告写入JSON文件")
    return p.parse_args()


def print_config_banner(llm: LLM, args):
    cfg = llm.model_runner.config
    hf = cfg.hf_config
    _, total_mem = torch.cuda.mem_get_info()
    print("=" * 78)
    print("Benchmark 配置")
    print("=" * 78)
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({total_mem / 2**30:.1f} GB), "
          f"CUDA {torch.version.cuda}, torch {torch.__version__}")
    print(f"  模型: {os.path.abspath(args.model)}")
    print(f"  dtype={hf.dtype}, layers={hf.num_hidden_layers}, "
          f"hidden={hf.hidden_size}, vocab={hf.vocab_size}")
    print(f"  max_model_len={cfg.max_model_len}, max_num_batched_tokens={cfg.max_num_batched_tokens}, "
          f"max_num_seqs={cfg.max_num_seqs}")
    print(f"  KVCache: {cfg.num_kvcache_blocks} blocks x {cfg.kvcache_block_size} tokens")
    print(f"  enforce_eager={cfg.enforce_eager}, chunked_prefill={cfg.enable_chunked_prefill}, "
          f"fp8_kvcache={cfg.enable_fp8_kvcache}")
    print(f"  ngram_spec_decode={cfg.enable_ngram_spec_decode}"
          + (f" (k={cfg.ngram_spec_num_tokens}, n={cfg.ngram_size})" if cfg.enable_ngram_spec_decode else ""))
    print(f"  replicas={args.replicas}")


def main():
    args = parse_args()
    llm_kwargs = dict(
        enforce_eager=args.eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        enable_ngram_spec_decode=args.spec,
        ngram_size=args.ngram_size,
        ngram_spec_num_tokens=args.spec_tokens,
    )
    llm = LLM(args.model, **llm_kwargs)

    print_config_banner(llm, args)

    # 预热
    for _ in range(args.warmup):
        llm.generate(["Hello, welcome to the benchmark warmup."],
                     SamplingParams(temperature=0.6, max_tokens=8, ignore_eos=True))

    vocab_size = llm.model_runner.config.hf_config.vocab_size
    all_groups = build_groups(vocab_size)
    selected = list(all_groups) if not args.groups else args.groups.split(",")

    orig_post = llm.scheduler.postprocess
    spec_stats = new_spec_stats()
    if args.spec:
        install_spec_hook(llm, orig_post, spec_stats)

    reports = {}
    aggregate_records = []
    total_wall = 0.0
    for name in selected:
        cases = all_groups[name]
        # 长度校验：超出max_model_len的用例跳过
        valid, skipped = [], 0
        for case in cases:
            if isinstance(case.prompt, str):
                isl = len(llm.tokenizer.encode(case.prompt))
            else:
                isl = len(case.prompt)
            if isl + case.max_tokens > args.max_model_len:
                skipped += 1
                print(f"[跳过] {case.name}: isl={isl}, 超过 max_model_len={args.max_model_len}")
            else:
                valid.append(case)
        if skipped:
            print(f"场景 {name} 跳过 {skipped} 条超长用例")
        if not valid:
            continue

        run_cases = valid * args.replicas
        spec_stats.clear()
        spec_stats.update(new_spec_stats())
        records, wall = run_group(llm, run_cases)
        m = build_metrics(records, wall, spec_stats)
        reports[name] = m
        print_group_report(name, len(run_cases), m, args.spec)
        aggregate_records.extend(records.values())
        total_wall += wall

    # 全局汇总
    if aggregate_records:
        agg_spec = new_spec_stats()
        # 合并各场景spec统计
        for m in reports.values():
            agg_spec["steps"] += m["spec"]["steps"]
            agg_spec["decode_entries"] += m["spec"]["decode_entries"]
            agg_spec["proposed"] += m["spec"]["proposed"]
            agg_spec["accepted_drafts"] += m["spec"]["accepted_drafts"]
            agg_spec["draft_steps"] += m["spec"]["draft_steps"]
            agg_spec["bonus_steps"] += m["spec"]["bonus_steps"]
            for k, v in m["spec"]["accept_hist"].items():
                agg_spec["accept_hist"][int(k)] += v
        aggregate = build_metrics(
            {i: r for i, r in enumerate(aggregate_records)}, total_wall, agg_spec)

        print("\n" + "#" * 78)
        print("# 全局汇总")
        print("#" * 78)
        print(f"  场景数: {len(reports)}, 请求总数: {aggregate['num_requests']}, "
              f"总耗时: {aggregate['wall_time']:.2f}s")
        print(f"  Prompt tokens: {aggregate['prompt_tokens']} (avg {aggregate['avg_isl']:.1f}), "
              f"Completion tokens: {aggregate['completion_tokens']} (avg {aggregate['avg_osl']:.1f})")
        print(f"  请求吞吐: {aggregate['request_throughput']:.2f} req/s | "
              f"输出token吞吐: {aggregate['output_token_throughput']:.2f} tok/s | "
              f"总token吞吐: {aggregate['total_token_throughput']:.2f} tok/s")
        print_dist_table("延迟(ms)", {
            "TTFT": aggregate["ttft"],
            "ITL": aggregate["itl"],
            "ITL/req": aggregate["per_request_itl"],
            "E2E": aggregate["e2e"],
        })
        if args.spec:
            print_spec_stats(agg_spec)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "groups": reports}, f, ensure_ascii=False, indent=2)
        print(f"\n完整报告已写入: {args.json}")


if __name__ == "__main__":
    main()
