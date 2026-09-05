#!/usr/bin/env python3
"""Is it the size of the batch, or who is in it?

The probe sweep showed divergence at batch 2, 4 and 8 but not 16 and 32, which
rules out a simple monotone size effect. This holds the batch size fixed and
varies only the neighbours. If one prompt yields several different outputs at a
single batch size, then a request's result depends on which other requests it
happened to be scheduled with, which is the strongest form of the problem: it
means a served result is a function of unrelated traffic.

    python scripts/composition.py --model Qwen/Qwen3-1.7B --batch-size 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROBES = [
    "Summarise the water cycle in two sentences.",   # diverged in the sweep
    "The capital of France is",                      # diverged in the sweep
    "Explain why the sky is blue in one sentence.",  # did not
]

# Deliberately varied in length and register, since padding to the longest
# sequence in a batch is one plausible route for a neighbour to matter.
FILLER_POOL = [
    "Hi.",
    "What is 2+2?",
    "Name a colour.",
    "Write a limerick about a cat that is afraid of water and lives in a boat.",
    "Explain the causes of the French Revolution in as much detail as you can.",
    "Who painted the Mona Lisa?",
    "List the first twenty Fibonacci numbers with commentary on each.",
    "Describe photosynthesis.",
    "def quicksort(arr):",
    "Translate 'good morning' into five languages.",
]


def digest(ids) -> str:
    return hashlib.sha1(bytes(str(list(ids)), "utf8")).hexdigest()[:10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--compositions", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("results/composition.json"))
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=1024, gpu_memory_utilization=0.85,
              enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    print(f"{args.model}  batch size fixed at {args.batch_size}  "
          f"{args.compositions} different neighbour sets\n")

    results = {}
    for probe in PROBES:
        outs = []
        for c in range(args.compositions):
            # rotate through the pool so each composition is a different set
            fill = [FILLER_POOL[(c * 3 + j) % len(FILLER_POOL)]
                    for j in range(args.batch_size - 1)]
            gen = llm.generate([probe] + fill, params)
            outs.append(list(gen[0].outputs[0].token_ids))
        # also alone, as the reference
        alone = list(llm.generate([probe], params)[0].outputs[0].token_ids)

        variants = {digest(o) for o in outs}
        matches_alone = sum(o == alone for o in outs)
        results[probe] = {"distinct_outputs": len(variants),
                          "matched_alone": matches_alone,
                          "compositions": args.compositions}
        print(f"  {probe[:46]:<48} {len(variants)} distinct output(s) across "
              f"{args.compositions} compositions, {matches_alone} matched alone")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": args.model, "batch_size": args.batch_size, "results": results}, indent=2))
    worst = max(v["distinct_outputs"] for v in results.values())
    print(f"\nmost distinct outputs any single prompt produced at a fixed "
          f"batch size: {worst}")
    if worst > 1:
        print("A request's output depends on which unrelated requests share its batch.")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
