#!/usr/bin/env python3
"""Generate `upstream_pin.json` — §13 item 2a's oracle, in its COMMITTED form.

## Why this file exists at all

§13 item 2a is the only oracle in S3a that is not a consistency oracle: it asserts our
greedy token sequence agrees with the **upstream model's own framework** (HF
`transformers`), which is what catches a faithful-looking but wrong reimplementation — a
wrong RoPE theta, a transposed projection, a swapped attention scale — that every
self-parity oracle ships green.

It is also the only oracle in S3a whose generator reaches outside this repository, and
§13 item 6 (hermetic oracle provenance, S-HARDEN-5) forbids that as a provenance basis:
*"a generator that imports anything outside the repository cannot establish that a
committed oracle came from the reference implementation it names."*

The plan-of-record §11 S3.8 resolves the two against each other exactly as this directory
implements it: the committed form of the oracle is **the upstream model's token sequences
pinned at a content-addressed revision** — model revision hash, `transformers` version,
tokenizer hash, generation config — with a regeneration leg that reproduces them from the
pin and byte-compares, and a cell asserting the committed pin matches the artifact the S3a
suite converts from. Where the regeneration leg cannot run in CI it runs as a recorded
pre-tag step and the fixture carries that run's record.

This script IS that pre-tag step. It is not part of any CI job. `check_upstream_pin.py` is
the CI-reachable half and needs neither `torch`, `transformers`, nor the checkpoint.

## What it pins, and why each field

- `model_revision` — the HF snapshot commit. The content address. Without it "Qwen2.5-1.5B-
  Instruct" names a moving target.
- `config_sha256`, `tokenizer_sha256`, `tokenizer_config_sha256`, `chat_template_sha256`,
  `generation_config_sha256` — the four files that decide what tokens a prompt becomes and
  what the decode contract is. A revision hash alone would not survive a cache that
  resolved differently.
- `transformers_version` — the upstream *implementation*, which is the thing item 2a
  compares against. A different major is a different oracle.
- `artifact_source_fingerprint` — the calibrated artifact the S3a suite converts from,
  tying the pin to the model whose tokens it claims to be about.
- The rendered `input_ids` per pack member, so the comparison never depends on re-rendering
  a chat template from a corpus that lives in another repository.

Usage (from this repository's root, with the checkpoint cached):

    python tests/reference/upstream_pin/gen_upstream_pin.py [--tokens 32] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PIN_PATH = HERE / "upstream_pin.json"

MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct"

# The calibrated artifact the S3a suite converts from (`tools/convert_model.py --artifact`).
# Its `manifest.json` carries the `source_fingerprint` recorded into the pin.
ARTIFACT_DIR = pathlib.Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct-w8a8")

# The reference prompt pack lives in the sibling records repository, with the spike that
# renders it. Reaching for it HERE is the point of this script being the out-of-CI step.
# `SUPERSLM_SPIKE_TOOLS` overrides the permanent checkout — needed only while the pack
# module is still on an unmerged branch there; the default is the permanent checkout on
# purpose (D-SLM268: a record must not depend on data that lives only in a worktree).
SPIKE_TOOLS = pathlib.Path(os.environ.get("SUPERSLM_SPIKE_TOOLS", r"D:\Wizard\Tools"))

PINNED_FILES = {
    "config_sha256": "config.json",
    "tokenizer_sha256": "tokenizer.json",
    "tokenizer_config_sha256": "tokenizer_config.json",
    "generation_config_sha256": "generation_config.json",
}


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(payload) -> bytes:
    """One serialization, used for both the committed file and every hash over it."""
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8")


def content_sha256(payload) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def checkpoint_dir():
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(MODEL_REPO, "config.json")
    if not isinstance(found, str):
        raise SystemExit(
            f"{MODEL_REPO} is not in the local HF cache; item 2a's oracle cannot be "
            f"generated without the upstream model it is an oracle about"
        )
    return pathlib.Path(found).parent


def upstream_greedy(checkpoint, input_ids, count):
    """HF's own fp32 greedy continuation — the upstream framework item 2a names.

    Argmax, not `model.generate`: `generate` routes through a `GenerationConfig` whose
    defaults (sampling, repetition penalties, `eos` handling) are library-version state,
    and the oracle must be the model's arithmetic rather than the library's decode policy.
    The generation config is still pinned above — as a recorded fact about the checkpoint,
    not as an input this decode obeys.

    `torch.argmax` returns the FIRST maximal index, which is the same lowest-index
    tie-break §6.5 pins for our own `select_greedy`. Asserted, not assumed, below.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), dtype=torch.float32).eval()
    ids = torch.tensor([list(input_ids)])
    produced = []
    with torch.no_grad():
        for _ in range(count):
            row = model(ids).logits[0, -1]
            nxt = int(torch.argmax(row))
            if int(row.argmax()) != int((row == row.max()).nonzero()[0]):
                raise AssertionError(
                    "torch.argmax did not return the lowest maximal index on this build; "
                    "the upstream tie-break no longer matches §6.5's and the pinned "
                    "sequences would encode a different selection rule"
                )
            produced.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    del model
    return produced


def build_pin(count: int) -> dict:
    sys.path.insert(0, str(SPIKE_TOOLS))
    from superslm_spike import baseline, pipeline                       # noqa: E402
    from superslm_spike.s3a_parity import prompt_pack                   # noqa: E402
    import torch                                                       # noqa: E402
    import transformers                                                # noqa: E402

    checkpoint = checkpoint_dir()
    tokenize_prompt = pipeline._checkpoint_tokenize_prompt(checkpoint)
    members = prompt_pack.reference_prompt_pack()
    rendered = prompt_pack.render(members, tokenize_prompt)

    artifact_manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))

    pin = {
        "model_repo": MODEL_REPO,
        "model_revision": checkpoint.name,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "artifact_source_fingerprint": artifact_manifest["source_fingerprint"],
        "corpus_sha256": baseline.corpus_sha256(),
    }
    for field, filename in PINNED_FILES.items():
        path = checkpoint / filename
        pin[field] = sha256_file(path) if path.exists() else None
    template = checkpoint / "chat_template.jinja"
    pin["chat_template_sha256"] = sha256_file(template) if template.exists() else None
    generation = checkpoint / "generation_config.json"
    pin["generation_config"] = (json.loads(generation.read_text(encoding="utf-8"))
                                if generation.exists() else None)

    entries = []
    for member, input_ids in zip(members, rendered):
        print(f"[member {member.index}] {member.axis}, {len(input_ids)} tokens -> "
              f"{count} upstream tokens", flush=True)
        tokens = upstream_greedy(checkpoint, input_ids, count)
        entries.append({
            "index": member.index,
            "axis": member.axis,
            "source": member.source,
            "messages": [dict(message) for message in member.messages],
            "input_ids": list(input_ids),
            "token_count": len(input_ids),
            "expected_verdict": member.expected_verdict,
            "upstream_tokens": tokens,
        })

    payload = {
        "schema": "superslm.s3a.item2a.upstream_pin",
        "schema_version": 1,
        "pin": pin,
        "decode": {
            "strategy": "greedy",
            "max_new_tokens": count,
            "tie_break": "lowest index",
            "dtype": "float32",
            "note": "raw argmax over the model's own logits, not transformers.generate",
        },
        "pack": {
            "record": ("Claude/Vitruvius/"
                       "SuperSLM_S3a_ReferencePromptPack_Design-2026-07-28.md"),
            "decision": "D-SLM350",
            "fingerprint": prompt_pack.pack_fingerprint(rendered),
            "members": entries,
        },
    }
    payload["content_sha256"] = content_sha256(
        {k: v for k, v in payload.items() if k != "content_sha256"})
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32,
                        help="upstream tokens per member. 32 is not a round number: measured "
                             "at transformers 5.13.1, a wrong RoPE theta flips no argmax at "
                             "all on a single 8-token forward and one prompt's first "
                             "divergence is at token 23, so a shorter pin ships the defect")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing pin (a re-pin is a deliberate commit)")
    args = parser.parse_args(argv)

    if PIN_PATH.exists() and not args.force:
        raise SystemExit(
            f"{PIN_PATH.name} already exists. Re-pinning replaces a committed oracle and is "
            f"a deliberate act: pass --force, and record why in PIN.md"
        )

    payload = build_pin(args.tokens)
    PIN_PATH.write_bytes(canonical_bytes(payload) + b"\n")
    print(f"wrote {PIN_PATH} ({PIN_PATH.stat().st_size} bytes)")
    print(f"content_sha256 {payload['content_sha256']}")
    print(f"model_revision {payload['pin']['model_revision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
