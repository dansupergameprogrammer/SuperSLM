# The upstream fidelity pin — §13 item 2a's committed oracle

`upstream_pin.json` is the committed form of the plan-of-record's §13 item 2a oracle: the
upstream HF `transformers` model's own greedy token sequences over the five-member S3a
reference prompt pack, pinned at a content-addressed revision.

## Why the oracle is committed rather than generated

Item 2a is the only oracle in S3a that is not a consistency oracle. Every other one
compares our integer path to our own float path, and a faithful-looking but wrong
reimplementation — a wrong RoPE `theta`, a transposed projection, a swapped attention
scale — is wrong in **both** and ships green. Item 2a is what catches that class, which is
why the plan-of-record §13 dim 10 calls it *"not optional and not substitutable."*

It is also the only oracle in S3a whose generator reaches outside this repository, and §13
item 6 (hermetic oracle provenance, S-HARDEN-5) forbids that as a provenance basis: *"a
generator that imports anything outside the repository cannot establish that a committed
oracle came from the reference implementation it names."*

The plan-of-record §11 S3.8 resolves the two against each other, and this directory is that
resolution implemented:

> the **committed** form of item 2a's oracle is the upstream model's token sequences pinned
> at a content-addressed revision — model revision hash, `transformers` version, tokenizer
> hash, generation config — with a regeneration leg that reproduces them from the pin and
> byte-compares, and a cell asserting the committed pin matches the artifact the S3a suite
> converts from. Where the regeneration leg cannot run in CI, it runs as a recorded pre-tag
> step and the fixture carries that run's record.

## The four files

| File | Runs where | What it is |
|---|---|---|
| `upstream_pin.json` | — | The oracle. Generated once, committed, never hand-edited. |
| `gen_upstream_pin.py` | pre-tag, by hand | The generator. Reaches outside this repository on purpose; not in any CI job. |
| `check_upstream_pin.py` | CI | Self-consistency, structure, and the artifact tie — no `torch`, no `transformers`, no checkpoint. |
| `test_upstream_pin.py` | CI + pre-tag | The red suite. Every guard proven able to fail on the fault it exists to catch. |

## What is pinned, and why each field is load-bearing

- **`model_revision`** — the HF snapshot commit. This is the content address. Without it,
  `Qwen/Qwen2.5-1.5B-Instruct` names a moving target and the oracle has no provenance at
  all.
- **`config_sha256`, `tokenizer_sha256`, `tokenizer_config_sha256`,
  `generation_config_sha256`** — the files that decide what tokens a prompt becomes and what
  the decode contract is. A revision hash alone would not survive a cache that resolved to
  different bytes; these are what `test_pinned_checkpoint_hashes_still_hold` checks the live
  cache against.
- **`chat_template_sha256` is `null` at this revision, and that is a recorded fact rather
  than a hole.** Verified against the cached snapshot: this checkpoint ships no
  `chat_template.jinja`, because at `transformers` 5.13.1 the Qwen2.5 chat template is
  carried inside `tokenizer_config.json` — which *is* pinned, and which therefore already
  covers the template's bytes. The field is emitted anyway so that a future revision which
  does ship a separate template file produces a non-null value rather than silently
  matching. It is deliberately **not** in `check_upstream_pin.REQUIRED_PIN_FIELDS`: a
  required-field check over a legitimately absent file would fail on a correct pin.
- **`transformers_version`** — item 2a compares against the upstream *implementation*, so
  the library version is part of the oracle's identity, not incidental. A different major
  is a different oracle.
- **`artifact_source_fingerprint`** — the `source_fingerprint` of the calibrated artifact
  the S3a suite converts from (`tools/convert_model.py --artifact`). This is the tie the
  plan-of-record asks for: without it the pin would be a set of tokens with no stated
  relationship to the model under test.
- **the rendered `input_ids` and `messages` per member** — so nothing on the consuming path
  re-renders a chat template from a corpus that lives in another repository. The corpus's
  own SHA-256 is recorded as the source of the three reused members, and is not a runtime
  dependency of the check.

## The decode is argmax, not `transformers.generate`

`generate` routes through a `GenerationConfig` whose defaults — sampling, repetition
penalty, `eos` handling — are library-version state. The oracle must be the model's
arithmetic, not the library's decode policy, so `gen_upstream_pin.py` takes a raw argmax
over the model's own logits. The checkpoint's generation config is still pinned, as a
recorded fact about the checkpoint rather than as an input this decode obeys.

**This is not a hypothetical concern on this checkpoint.** Its own pinned
`generation_config.json` reads:

```json
{"do_sample": true, "temperature": 0.7, "top_k": 20, "top_p": 0.8,
 "repetition_penalty": 1.1, "transformers_version": "4.37.0"}
```

`do_sample: true`. An oracle generated through `model.generate()` would have **sampled**,
producing a non-reproducible sequence that a byte-compare would then enshrine as if it were
the model's arithmetic. Note also that the config declares `transformers_version 4.37.0`
while the pin was generated under 5.13.1 — the config is a checkpoint-era artifact, which is
exactly why it is pinned as a recorded fact and never as an input.

`torch.argmax` returns the first maximal index, which is the same lowest-index tie-break
§6.5 pins for our own `select_greedy`. The generator asserts that per step rather than
assuming it.

## Why 32 tokens

Not a round number. Measured at `transformers` 5.13.1 against this checkpoint
(`Tools/superslm_spike/tests/test_pipeline.py`, the §9 G-2 block): a wrong RoPE `theta`
moved the logits by 13% on a single 8-token forward and **flipped no argmax at all**; over
32-token generations it agreed 33.3%, with one prompt's first divergence at **token 23**.
A pin shorter than that agrees with a wrong model, which is the one thing this oracle
exists not to do.

The five-member pack's own below-band member is 20 prompt tokens, and this repository's own
pilot measurement reproduces the same blind spot on it — the seeded wrong-`theta` arm is
indistinguishable from the correct arm at that length. The pack carries four long members
for exactly this reason.

## Re-pinning

A re-pin replaces a committed oracle and is a deliberate act, not a fix for a red test.
`gen_upstream_pin.py` refuses to overwrite without `--force`. When you do re-pin, record
here: the date, which pinned field moved (revision, library version, pack), and why.

| Date | Change | Reason |
|---|---|---|
| 2026-07-29 | Initial pin | S3a §14.3's item 2a threshold derivation; first committed form of the oracle. |
