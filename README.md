# What determinism costs in vLLM

On vLLM 0.28.0, half of ordinary prompts return a different answer depending on
which unrelated requests share their batch. vLLM ships a fix for this,
`VLLM_BATCH_INVARIANT=1`. The documentation says it costs performance and does
not say how much.

**It works completely, and in vLLM's default configuration it costs 54 to 67%
of throughput.**

| batch | default tok/s | invariant tok/s | cost |
|---|---|---|---|
| 1 | 226.1 | 75.5 | 66.6% |
| 4 | 802.9 | 298.1 | 62.9% |
| 16 | 3112.6 | 1220.9 | 60.8% |
| 64 | 11152.0 | 5176.1 | 53.6% |

Qwen3-1.7B, one RTX 4090, vLLM 0.28.0, CUDA graphs enabled, 128 tokens per
request with `ignore_eos`, five timed repeats per cell after a discarded warmup,
median reported. Run-to-run spread was 0.3 to 3.3%, an order of magnitude below
the effect.

## Measure it with CUDA graphs on, or understate it by half

Running both modes with `enforce_eager=True` gives a much smaller number:

| batch | eager default | eager invariant | eager cost | cost with CUDA graphs |
|---|---|---|---|---|
| 1 | 71.9 | 52.3 | 27.2% | **66.6%** |
| 4 | 276.9 | 192.9 | 30.3% | **62.9%** |
| 16 | 1098.5 | 777.9 | 29.2% | **60.8%** |
| 64 | 4168.0 | 3049.9 | 26.8% | **53.6%** |

That comparison is fair between the two modes and misleading for anyone deciding
whether to enable the flag, because it compares two configurations that are both
handicapped. CUDA graphs are on by default in vLLM, and the two modes do not
benefit from them equally:

| | speedup from CUDA graph capture |
|---|---|
| default kernels | 2.7x to 3.1x |
| batch-invariant kernels | 1.4x to 1.7x |

The batch-invariant path captures far less well, so most of the real cost is the
optimization you give up rather than the kernels themselves being slower.

The eager cost is flat across batch size, including at batch 1 where there is no
batch to be invariant across, which says the kernel overhead alone is roughly
28%. The graph-enabled cost falls as batches grow, from 66.6% at batch 1 to
53.6% at batch 64, so larger batches recover part of the capture penalty.

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
up to 64. The 54 to 67% figure is for this configuration and should not be
quoted as a general constant; the point is that it is measurable and currently
unpublished, not that it is universal.

Throughput uses `ignore_eos` so every request emits exactly 128 tokens.
Without it the measurement partly reflects how long each answer happened to be.

The headline figure is measured with CUDA graphs enabled because that is vLLM's
default. Both sets of numbers are reported above, since the eager comparison is
the one that isolates the kernels while the graph-enabled one is the one that
answers "what does this cost me".
