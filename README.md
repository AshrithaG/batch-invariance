# What determinism costs in vLLM

On vLLM 0.28.0, half of ordinary prompts return a different answer depending on
which unrelated requests share their batch. vLLM ships a fix for this,
`VLLM_BATCH_INVARIANT=1`. The documentation says it costs performance and does
not say how much.

**It works completely, and it costs 27 to 30% of throughput at every batch size.**

| batch | default tok/s | invariant tok/s | cost | run-to-run noise |
|---|---|---|---|---|
| 1 | 71.9 | 52.3 | 27.2% | 3.3% |
| 4 | 276.9 | 192.9 | 30.3% | 0.3% |
| 16 | 1098.5 | 777.9 | 29.2% | 1.5% |
| 64 | 4168.0 | 3049.9 | 26.8% | 1.6% |

Qwen3-1.7B, one RTX 4090, 128 tokens per request with `ignore_eos`, five timed
repeats per cell after a discarded warmup, median reported. Noise is an order of
magnitude below the effect.

## The cost is flat, which says where it comes from

At batch 1 there is no batch to be invariant across, and the cost is still
27.2%. So this is not the price of giving up batch-size-dependent
optimizations. It is the price of the constrained reduction strategy itself,
paid per kernel whether or not anything is batched.

That is a more optimistic reading than a scaling cost would be. An overhead that
grows with batch size would be inherent to the trade-off; a flat one is an
implementation property, and implementations improve. The vLLM docs describe
batch invariance as under active development with performance work planned,
which is consistent with this shape.

## The problem it fixes

Same prompt, same seed, `temperature=0`, greedy, same model, same GPU, same
process. Only the other requests in the batch differ.

| distinct answers for one prompt | prompts |
|---|---|
| 1 (unaffected) | 15 |
| 2 | 9 |
| 3 | 5 |
| 4 | 1 |

**15 of 30 prompts affected.** For six of them, *no* batched composition ever
reproduced what the prompt returns on its own: validate against an idle server,
deploy, and you will never see that answer again.

With `VLLM_BATCH_INVARIANT=1`: **0 of 30 affected.** The fix does what it says.

## Two things the earlier reports did not establish

This behaviour is known upstream and was reported in
[#5898](https://github.com/vllm-project/vllm/issues/5898) and
[#11658](https://github.com/vllm-project/vllm/issues/11658), the latter closed
as not planned. Those reports establish that it happens. Two further things are
measured here.

**It is composition, not size.** Sweeping batch size with a fixed filler set
produced divergence at 2, 4 and 8 but *not* at 16 and 32, which rules out a
monotone size effect. Holding batch size fixed at 4 and varying only the
neighbours reproduces it on its own: one prompt produced three different answers
across ten compositions. Padding is a plausible route, since a batch pads to its
longest member and the filler pool deliberately mixes three-word prompts with
long ones.

**It is deterministic given the batch.** Running the full sweep twice gave 25 of
25 identical comparisons, the same prompts diverging at the same batch sizes
both times. This is not flakiness. Any single case reproduces exactly.

## Why only half the prompts

The effect needs a close decision to land on. Reduced-precision arithmetic moves
logits slightly; that only changes the emitted token when the top two candidates
are within that margin of each other.

A [companion repo](https://github.com/AshrithaG/specdec) measured this mechanism
directly in a different setting, comparing a batched forward against a
sequential one over the same tokens on the same weights. In bfloat16 the two
paths disagreed by a median of 0.203 in logit space against 0.00002 in float32,
and argmax flips occurred at a 50% rate where the top-two margin was under 0.01
and never where it exceeded 0.25.

So "The capital of France is" was stable under every composition tested, and
"Give three uses for a paperclip" produced three different answers. Open-ended
generation passes through more near-ties.

## Reproducing

```bash
python scripts/probe.py         # does batch size change the output
python scripts/composition.py   # or is it who is in the batch
python scripts/rate.py          # how often, over 30 prompts

VLLM_BATCH_INVARIANT=0 python scripts/throughput.py --out results/tput_default.json
VLLM_BATCH_INVARIANT=1 python scripts/throughput.py --out results/tput_invariant.json
```

## Scope

One model, one GPU, one vLLM version, greedy decoding, 30 prompts, batch sizes
up to 64. The 27 to 30% figure is for this configuration and should not be
quoted as a general constant; the point is that it is measurable and currently
unpublished, not that it is universal.

Throughput uses `ignore_eos` so every request emits exactly 128 tokens.
Without it the measurement partly reflects how long each answer happened to be.
