#!/usr/bin/env python3
"""Does a prompt's output depend on what else is in the batch?

vLLM schedules requests together. If the answer changes depending on the
neighbours a request happened to share a step with, then results are a function
of server load, and any evaluation run against a busy server is not reproducible.

The test: generate one prompt alone, then generate the same prompt inside
batches of increasing size, greedy, and compare token by token. Any difference
is batch dependence, because nothing about the request changed.

    python scripts/probe.py --model Qwen/Qwen3-1.7B --sizes 1 2 4 8 16 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROBES = [
    "Explain why the sky is blue in one sentence.",
    "List three prime numbers.",
    "def binary_search(arr, target):",
    "The capital of France is",
    "Summarise the water cycle in two sentences.",
]

FILLER = [
    "Write a limerick about a cat.", "What is the boiling point of water?",
    "Name three programming languages.", "Describe photosynthesis briefly.",
    "Who painted the Mona Lisa?", "What causes tides?",
    "Explain recursion to a beginner.", "List the planets in order.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--quant", default=None, help="pass through to vLLM, e.g. fp8")
    ap.add_argument("--out", type=Path, default=Path("results/probe.json"))
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    kw = dict(model=args.model, max_model_len=1024, gpu_memory_utilization=0.85,
              enforce_eager=True, seed=0)
    if args.quant:
        kw["quantization"] = args.quant
    llm = LLM(**kw)
    # greedy: no sampling randomness, so any difference is the engine
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    print(f"model {args.model}  quant {args.quant or 'none'}  greedy\n")
    baseline: dict[str, list[int]] = {}
    rows = []

    for size in args.sizes:
        for i, probe in enumerate(PROBES):
            # the probe plus enough filler to reach the batch size
            batch = [probe] + [FILLER[(i + j) % len(FILLER)] for j in range(size - 1)]
            out = llm.generate(batch, params)
            ids = list(out[0].outputs[0].token_ids)
            if size == args.sizes[0]:
                baseline[probe] = ids
                continue
            ref = baseline[probe]
            n = min(len(ref), len(ids))
            first = next((j for j in range(n) if ref[j] != ids[j]), None)
            rows.append({"batch_size": size, "probe": probe,
                         "identical": first is None and len(ref) == len(ids),
                         "first_divergence": first})

    print(f"{'batch size':>11}{'probes':>9}{'identical':>11}{'first divergence':>18}")
    print("-" * 49)
    for size in args.sizes[1:]:
        sel = [r for r in rows if r["batch_size"] == size]
        same = sum(r["identical"] for r in sel)
        firsts = [r["first_divergence"] for r in sel if r["first_divergence"] is not None]
        med = f"token {sorted(firsts)[len(firsts)//2]}" if firsts else "n/a"
        print(f"{size:>11}{len(sel):>9}{f'{same}/{len(sel)}':>11}{med:>18}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": args.model, "quant": args.quant, "rows": rows}, indent=2))
    total = len(rows)
    diff = sum(not r["identical"] for r in rows)
    print(f"\n{diff} of {total} comparisons differed from the batch-of-"
          f"{args.sizes[0]} baseline")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
