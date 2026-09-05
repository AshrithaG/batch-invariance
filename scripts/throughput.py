#!/usr/bin/env python3
"""What does batch invariance actually cost in throughput?

vLLM's documentation says enabling batch invariance costs performance and that
the trade-off is intentional, without saying how much. This measures it across
batch sizes, with repeats, discarding a warmup, and reporting the median rather
than a single timing.

Run once per mode:
    VLLM_BATCH_INVARIANT=0 python scripts/throughput.py --out results/tput_default.json
    VLLM_BATCH_INVARIANT=1 python scripts/throughput.py --out results/tput_invariant.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time
from pathlib import Path

PROMPT = "Explain the water cycle, the causes of rainbows, and how a refrigerator works."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--no-eager", action="store_true",
                    help="allow CUDA graph capture, which is vLLM's default. The "
                         "eager numbers compare the two modes fairly but neither "
                         "gets graphs; this asks what a real deployment sees.")
    ap.add_argument("--out", type=Path, default=Path("results/throughput.json"))
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    import vllm

    invariant = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
    llm = LLM(model=args.model, max_model_len=1024, gpu_memory_utilization=0.85,
              enforce_eager=not args.no_eager, seed=0)
    # ignore_eos so every request generates exactly max_tokens; otherwise the
    # measurement is partly a measurement of how long the answers happened to be
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                            ignore_eos=True)

    print(f"vLLM {vllm.__version__}  {args.model}  "
          f"VLLM_BATCH_INVARIANT={'1' if invariant else '0'}  "
          f"cuda_graphs={'on' if args.no_eager else 'off'}")
    print(f"{args.repeats} repeats per batch size, first discarded as warmup\n")
    print(f"{'batch':>7}{'median tok/s':>15}{'min':>10}{'max':>10}{'spread':>9}")
    print("-" * 51)

    results = {}
    for bs in args.batch_sizes:
        prompts = [PROMPT] * bs
        rates = []
        for r in range(args.repeats + 1):     # +1 warmup
            t0 = time.perf_counter()
            out = llm.generate(prompts, params)
            dt = time.perf_counter() - t0
            toks = sum(len(o.outputs[0].token_ids) for o in out)
            if r:                              # discard warmup
                rates.append(toks / dt)
        med = st.median(rates)
        results[bs] = {"median": med, "min": min(rates), "max": max(rates),
                       "all": rates}
        print(f"{bs:>7}{med:>15.1f}{min(rates):>10.1f}{max(rates):>10.1f}"
              f"{(max(rates)-min(rates))/med:>8.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"vllm": vllm.__version__, "model": args.model,
         "batch_invariant": invariant, "cuda_graphs": args.no_eager,
         "max_tokens": args.max_tokens,
         "results": {str(k): v for k, v in results.items()}}, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
