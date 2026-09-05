# Your answer depends on who else is in the batch

On vLLM 0.28.0, greedy decoding, temperature 0, fixed seed, **half of ordinary
prompts return a different answer depending on which unrelated requests happen
to share their batch.**

Nothing about the request changes. Same prompt, same sampling parameters, same
model, same GPU, same process. Only the neighbours differ.

```bash
python scripts/probe.py        # does batch size change the output?
python scripts/composition.py  # or is it who is in the batch?
python scripts/rate.py         # how often, over 30 prompts
```

## What was measured

Qwen3-1.7B, vLLM 0.28.0, one RTX 4090, `enforce_eager=True`, `seed=0`,
`temperature=0`, 64 new tokens. Each prompt is generated alone, then inside a
batch of 4 with six different sets of filler prompts, and the outputs compared
token by token.

| distinct answers for one prompt | prompts |
|---|---|
| 1 (unaffected) | 15 |
| 2 | 9 |
| 3 | 5 |
| 4 | 1 |

**15 of 30 prompts affected.** The worst, "What causes rainbows?", produced four
different answers across seven runs of the identical request.

Among the affected prompts, the number of batched runs that matched the solo run
at all:

| compositions matching solo | prompts |
|---|---|
| 0 of 6 | **6** |
| 1 to 2 of 6 | 5 |
| 4 to 5 of 6 | 4 |

For six prompts, **no** batched composition reproduced what the prompt returns
on its own. If you validate a prompt against an idle server and then deploy it,
those are answers you will never see again.

## It is deterministic, which is what makes it reportable

Running the entire sweep twice gave **25 of 25 identical comparisons**, the same
prompts diverging at the same batch sizes both times. This is not run-to-run
flakiness. Output is a deterministic function of the batch, so a maintainer can
reproduce any single case exactly.

## Size is not the variable, composition is

The first sweep varied batch size with a fixed filler set and found divergence
at sizes 2, 4 and 8 but **not** at 16 and 32. That non-monotonicity rules out a
simple "bigger batch, more drift" story.

Holding the batch size fixed at 4 and varying only the neighbours reproduces the
effect on its own, which locates it in composition rather than size. Padding is
a plausible route, since the filler pool deliberately mixes three-word prompts
with long ones and a batch is padded to its longest member.

## Why only half the prompts

The effect needs a close decision to land on. Reduced-precision arithmetic moves
logits by a small amount; that only changes the emitted token when the top two
candidates are within that amount of each other.

A companion experiment in a
[separate repo](https://github.com/AshrithaG/specdec) measured the same
mechanism directly in a different setting. Comparing a batched forward against a
sequential one over the same tokens on the same model: bfloat16 disagreed by a
median of 0.203 in logit space against 0.00002 in float32, and argmax flips
occurred at a 50% rate where the top-two margin was under 0.01 and **never**
where it exceeded 0.25. The largest margin that flipped was 0.125, against a
99th-percentile numerical disagreement of 0.484.

So prompts whose generations pass through many near-ties are affected and
prompts whose next token is usually obvious are not. "The capital of France is"
was stable across every composition tested. "Give three uses for a paperclip"
produced three different answers.

## What this is not

One model, one GPU, one vLLM version, batch size 4, thirty prompts, greedy
decoding. The rate is a rate for this configuration, not a general constant.

It is also not news that reduced-precision batched inference is not
bit-reproducible; that is understood in serving circles. What is measured here
is the part usually left qualitative: how often it changes a real answer, that
it is deterministic given the batch, that composition rather than size is the
variable, and that for some prompts the solo output is unreachable once batching
is on.

## Why it matters beyond reproducibility

Any evaluation run against a shared or autoscaling server is measuring a
function of traffic. Two identical requests can receive different answers, and a
benchmark result can move depending on what else was in flight. Caching keyed on
the prompt returns an answer that the same request would not produce now. None
of that is visible unless you look for it, because every individual response
looks fine.
