#!/usr/bin/env python3
"""How often does batch composition change a served answer?

The composition test showed one prompt in three producing several different
answers depending on its neighbours. Three prompts cannot support a rate, and a
rate is the first thing anyone will ask for. This runs the same test over a
larger prompt set and reports the fraction affected.

    python scripts/rate.py --n-prompts 60 --compositions 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Spread across registers, because the effect needs a close-run decision
# somewhere and open-ended prose plausibly has more of them than a lookup.
PROMPTS = [
    "Summarise the water cycle in two sentences.",
    "Explain why the sky is blue in one sentence.",
    "The capital of France is",
    "List three prime numbers.",
    "def binary_search(arr, target):",
    "Describe the causes of the First World War.",
    "Write a haiku about autumn.",
    "What is the difference between TCP and UDP?",
    "Explain recursion to a ten year old.",
    "Name four countries in South America.",
    "How does a refrigerator work?",
    "Write a limerick about a stubborn goat.",
    "What is the time complexity of quicksort?",
    "Summarise the plot of Macbeth.",
    "Explain what a hash table is.",
    "Give three uses for a paperclip.",
    "What causes rainbows?",
    "def merge_sort(items):",
    "Describe the Doppler effect briefly.",
    "Why do leaves change colour in autumn?",
    "Explain the difference between weather and climate.",
    "Write a two line poem about the sea.",
    "What is photosynthesis?",
    "How do vaccines work?",
    "List the noble gases.",
    "Explain gradient descent in simple terms.",
    "What is the boiling point of water at sea level?",
    "Describe how a bill becomes law.",
    "Write a short product description for a water bottle.",
    "What is the Pythagorean theorem?",
]

FILLER_POOL = [
    "Hi.", "What is 2+2?", "Name a colour.",
    "Write a limerick about a cat that is afraid of water and lives in a boat.",
    "Explain the causes of the French Revolution in as much detail as you can.",
    "Who painted the Mona Lisa?",
    "List the first twenty Fibonacci numbers with commentary on each.",
    "Describe photosynthesis.", "def quicksort(arr):",
    "Translate 'good morning' into five languages.",
]


def digest(ids) -> str:
    return hashlib.sha1(bytes(str(list(ids)), "utf8")).hexdigest()[:10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--compositions", type=int, default=6)
    ap.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("results/rate.json"))
    args = ap.parse_args()

    import os
    import time

    from vllm import LLM, SamplingParams
    import vllm

    # vLLM reads this at import/init time, so it has to be set in the
    # environment before launching, not here.
    invariant = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"

    llm = LLM(model=args.model, max_model_len=1024, gpu_memory_utilization=0.85,
              enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    prompts = PROMPTS[: args.n_prompts]
    print(f"vLLM {vllm.__version__}  {args.model}  batch {args.batch_size}  "
          f"{args.compositions} compositions  {len(prompts)} prompts")
    print(f"VLLM_BATCH_INVARIANT={'1 (deterministic kernels)' if invariant else '0 (default)'}\n")

    # Throughput, measured on the same work the correctness sweep does, so the
    # cost figure and the invariance figure come from one run rather than two
    # differently shaped ones.
    t0 = time.perf_counter()
    warm = llm.generate([prompts[0]] * args.batch_size, params)
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in warm)
    throughput = gen_tokens / (time.perf_counter() - t0)

    rows = []
    for i, probe in enumerate(prompts):
        alone = list(llm.generate([probe], params)[0].outputs[0].token_ids)
        outs = []
        for c in range(args.compositions):
            fill = [FILLER_POOL[(c * 3 + j) % len(FILLER_POOL)]
                    for j in range(args.batch_size - 1)]
            outs.append(list(llm.generate([probe] + fill, params)[0].outputs[0].token_ids))
        variants = {digest(o) for o in outs} | {digest(alone)}
        rows.append({"prompt": probe, "distinct": len(variants),
                     "matched_alone": sum(o == alone for o in outs)})
        if len(variants) > 1:
            print(f"  affected: {len(variants)} variants, "
                  f"{rows[-1]['matched_alone']}/{args.compositions} matched alone  "
                  f"{probe[:52]}")

    affected = [r for r in rows if r["distinct"] > 1]
    print(f"\n{len(affected)} of {len(rows)} prompts affected "
          f"({len(affected)/len(rows):.0%})")
    if affected:
        worst = max(affected, key=lambda r: r["distinct"])
        print(f"worst prompt produced {worst['distinct']} distinct answers: "
              f"{worst['prompt'][:60]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"vllm": vllm.__version__, "model": args.model,
         "batch_size": args.batch_size, "batch_invariant": invariant,
         "throughput_tok_s": throughput, "rows": rows}, indent=2))
    print(f"throughput on a batch of {args.batch_size}: {throughput:.1f} tok/s")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
