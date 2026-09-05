#!/usr/bin/env python3
"""Does VLLM_BATCH_INVARIANT=1 deliver, and what does it cost?

vLLM ships a batch invariance mode that constrains kernels to one reduction
strategy so results stop depending on batch composition. The documentation says
it costs performance and that the trade-off is deliberate, but not how much.
This compares a run with the flag off against a run with it on: whether the
divergence actually goes away, and what the throughput price is.

    VLLM_BATCH_INVARIANT=0 python scripts/rate.py --out results/default.json
    VLLM_BATCH_INVARIANT=1 python scripts/rate.py --out results/invariant.json
    python scripts/compare_modes.py results/default.json results/invariant.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def affected(run: dict) -> list[dict]:
    """Prompts whose answer changed with the batch they were in."""
    return [r for r in run["rows"] if r["distinct"] > 1]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_modes.py default.json invariant.json")
    default = json.loads(Path(sys.argv[1]).read_text())
    invariant = json.loads(Path(sys.argv[2]).read_text())

    print(f"vLLM {default['vllm']}   {default['model']}   "
          f"batch {default['batch_size']}\n")
    print(f"{'mode':<28}{'affected':>12}{'worst':>8}{'tok/s':>10}")
    print("-" * 58)

    for label, run in (("default", default), ("VLLM_BATCH_INVARIANT=1", invariant)):
        n_aff = len(affected(run))
        n_all = len(run["rows"])
        worst = max((r["distinct"] for r in run["rows"]), default=1)
        tput = run.get("throughput_tok_s")
        tput_s = f"{tput:.1f}" if tput else "n/a"
        print(f"{label:<28}{f'{n_aff}/{n_all}':>12}{worst:>8}{tput_s:>10}")

    a, b = default.get("throughput_tok_s"), invariant.get("throughput_tok_s")
    if a and b:
        print(f"\ncost of determinism: {1 - b / a:.1%} of throughput "
              f"({a:.1f} down to {b:.1f} tok/s)")
        print("note: this figure comes from a single generation; use "
              "scripts/throughput.py for a measured one")

    still = affected(invariant)
    if still:
        print(f"\nbatch invariance did NOT hold on {len(still)} prompt(s):")
        for r in still:
            print(f"  {r['distinct']} distinct answers   {r['prompt'][:56]}")
        print("\nThat is a bug in a documented feature, not a duplicate of the "
              "known nondeterminism reports.")
    else:
        print("\nbatch invariance held on every prompt tested")


if __name__ == "__main__":
    main()
