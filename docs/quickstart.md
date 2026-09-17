# Quickstart: convert your own model

SuperSLM does not ship pre-converted model artifacts — converting a Hugging
Face checkpoint is fast enough, and the point of an open converter, that
hosting artifacts would not add anything the converter itself doesn't
already give you. This walkthrough takes a Hugging Face checkpoint all the
way to decoded output through the compiled engine, plus a worked example of
converting a LoRA adapter on top of it.

Every command below is the real CLI, run as documented by its own `--help`.

## Prerequisites

- Python 3.10+ with `pip install -r requirements.txt` run from the repo root
- A C++20 compiler and CMake 3.16+ (see [the README's build section](../README.md#building))
- A local Hugging Face checkpoint directory — `config.json`, the
  `safetensors` weight files, and the tokenizer files. This walkthrough uses
  Qwen2.5-1.5B-Instruct (Apache-2.0) as the running example; any checkpoint
  in the Qwen2.5 tokenizer lineage works the same way.

Build the conversion and decode binaries before starting:

```
cmake -B build && cmake --build build --target sslm_verify --target sslm_generate --target t2700_fused_k_capture
```

## 1. Convert a QK-norm checkpoint

QK-norm checkpoints, including Qwen3-Embedding-0.6B, use one conversion command. It runs the
factored float calibration, makes the provisional table, captures the full 600-record corpus three
times (A/B/C, one shared prefix per pass), rebuilds the table from `max(float, A)` then
`max(float, A, B)`, verifies the final artifact, and writes every peak table plus pass-C clipping
details to `<work>/flow-report.json` and `<work>/peak-tables.npz`. Qwen3-Embedding-0.6B requires
`--channel-scale-headroom 1.25`; the default remains 1.0 for other models.

The resulting QK artifact carries the `QKC1` channel table and header flag bit 2 (`0x4`). Its
source scales are serialized as exact binary64 bits; the landing reciprocal, exponent, ratio, and
carried softmax scale are loader-checked derivations. A QK artifact containing retired fused-K
metadata or keys is rejected. Do not reuse a QK artifact made by a pre-1.5.0 conversion tree:
rerun this flow from the checkpoint when a replacement is required.

```
python tools/t2700_fused_k_calibration.py flow \
  --checkpoint <path to HF checkpoint> \
  --out <final model output path>.sslm \
  --work <new empty conversion work directory> \
  --capture build/t2700_fused_k_capture \
  --channel-scale-headroom 1.25 \
  --verifier build/sslm_verify
```

`--out` and `--work` must not exist. The command refuses pass C only when clipped direct-K
landings exceed one per million observations; at or below that rate its report records the count
and every overshooting `(layer, head, channel)`. On the Ryzen 9 3950X, Qwen3-Embedding-0.6B took
about 5,226 s end to end at `--channel-scale-headroom 1.25`; A/B/C each took about 1,495 s and
remained within the 30-minute bar. Its pass C clipped 7 of 282,103,808 callbacks
(0.025 per million). At the default 1.0, pass C clipped 7,713 of 282,103,808
(27.341 per million), exceeded the one-per-million limit, and raised `ChannelScaleDidNotConverge`.

## 1a. Calibrate a non-QK-norm checkpoint

```
python tools/calibrate_checkpoint.py --checkpoint <path to HF checkpoint> --out <artifact dir>
```

This is calibration only: it reads the raw checkpoint and writes a
calibrated artifact directory that the next step consumes. At a reduced,
fixture-sized corpus it takes minutes. The command prints a
`written fingerprint: ...` line and the calibrated artifact directory's
path on success.

## 2. Convert the tokenizer

```
python tools/convert_tokenizer.py --ckpt <path to HF checkpoint> --emit <tokenizer output path>.sslm
```

Reads the checkpoint's tokenizer files and emits a `.sslm` artifact holding
the tokenizer, chat template, and the pinned Unicode tables the runtime's
integer-only tokenizer needs — no platform Unicode or regex library on the
inference path. Prints a `wrote ... fingerprint ...` line on success.

## 3. Convert the model (non-QK-norm path only)

```
python tools/convert_model.py --artifact <artifact dir from step 1> --out <model output path>.sslm
```

Reads the calibrated artifact, quantizes it, and writes the model `.sslm`.
This step also invokes the compiled `sslm_verify` binary as an independent
check: the artifact is loaded back through the real C++ loader and its
contents are compared against what was written, so a successful run means
an independently-verified artifact, not just "the writer didn't crash."
Look for `verified: independent loader accepted the artifact` in the
output.

To make the same artifact eligible for 1.2's opt-in damped-greedy decoder,
add `--enable-damped-greedy` during conversion:

```
python tools/convert_model.py --artifact <artifact dir from step 1> --out <model output path>.sslm --enable-damped-greedy
```

That switch adds the DGC1 constants and feature bit. Omitting it preserves the
ordinary greedy-only artifact shape.

You now have two `.sslm` files — the tokenizer artifact from step 2 and the
model artifact from step 1 or 3 — and both are what the engine needs to
decode.

## 4. Decode

```
build/sslm_generate <model>.sslm <tokenizer>.sslm "<prompt>" --max-new <N>
```

Runs the prompt through the compiled engine and prints the decoded tokens
and text. `sslm_generate` also accepts `--adapter <adapter>.sslm` to decode
through a runtime-attached LoRA adapter (see step 5), `--stop a,b,c` for
stop-token IDs, and `--dump-logits <path>` for raw logit inspection.

For a model converted with `--enable-damped-greedy`, select the ruled defaults
with one additional flag:

```
build/sslm_generate <model>.sslm <tokenizer>.sslm "<prompt>" --max-new <N> --decode-mode damped-greedy
```

The defaults are `alpha=2` (`--alpha-q15 65536`), anti-LM order `2`, and
`top_k=min(6, vocab_size)`. Advanced callers can override them with
`--alpha-q15`, `--anti-lm-order`, and `--top-k`; `--alpha-q15` takes a Q15-scaled
integer, not a decimal alpha value. Greedy remains the default when
`--decode-mode` is absent.

The same walkthrough works through the [`sslm_*` C ABI](api.md) instead of
this CLI tool for embedding the engine directly in another process — including
[schema-constrained generation](api.md#schema-constrained-generation), which
this CLI tool does not currently expose a flag for.

## 5. Worked example: convert a LoRA adapter

Adapter conversion follows the same pattern, using a PEFT LoRA adapter
directory (`adapter_config.json` + adapter weights) and the model artifact
from step 3 as the base it binds to:

```
python tools/sslm_convert_adapter.py --adapter <PEFT adapter dir> --base <model>.sslm --out <adapter output path>.sslm
```

By default the adapter converts to a runtime-additive artifact — attach it
to a running model with `sslm_generate --adapter` (or the equivalent ABI
call) without touching the base weights, and switch it out again in a
fraction of a second (see the README's runtime-switchable-adapters
section). Pass `--fallback=merge` to instead produce a merged-and-quantized
artifact with the adapter baked permanently into the weights, for the rare
case a runtime-additive artifact is rejected. Like model conversion, this
step invokes `sslm_verify` and prints `verified: independent loader
accepted the artifact` on success.

The project develops against a LoRA specialization it calls the
"shopkeeper" adapter — a worked example of exactly this conversion, used
throughout the project's own runtime-adapter testing and cited above for
the adapter-switch numbers.

## Verifying the whole pipeline from a clean checkout

`tools/verify_clean_checkout.ps1` is the project's own release-verification
procedure: it extracts a clean copy of the repository with no ancestor
outside it, builds the CMake targets this walkthrough uses, and runs all
five steps above — plus a full round trip through `sslm_generate` — against
that clean copy, failing loudly and naming the exact step if anything
breaks. It is not a CI job (steps 1 and 5 need a real checkpoint and
adapter present locally, and the full-scale run costs real wall-clock
time), but it is the same walkthrough this document describes, executed
end to end and checked automatically:

```
pwsh -File tools/verify_clean_checkout.ps1 -Checkpoint <path to HF checkpoint> -Adapter <path to PEFT adapter dir>
```

Pass `-CalibrationRecordLimit` and `-AdapterLayerLimit` with small numbers
to run the same nine steps at a reduced, fixture scale in minutes instead
of at full release-gate cost — useful for confirming your own environment
can run every step's real code path before committing to the full-scale
run.
