#!/usr/bin/env python3
"""The CI-reachable half of §13 item 2a's pin — no `torch`, no `transformers`, no checkpoint.

`gen_upstream_pin.py` is the recorded pre-tag step: it reaches outside this repository, so
it cannot run on a bare CI box and §13 item 6 would not accept it as a provenance basis if
it could. What CI *can* do is everything that does not need the upstream model, and this
module is exactly that set:

1. **Self-consistency.** `content_sha256` recomputed from the file's own body. Catches a
   hand-edit after pinning — the failure mode a committed oracle is most exposed to, and
   the one no amount of upstream availability would catch.
2. **Structural completeness.** Every pin field present and non-null, all five pack members
   present in band-position order, every member's `upstream_tokens` at the declared length,
   and every token id inside the checkpoint's vocabulary.
3. **The artifact tie.** The pin's `artifact_source_fingerprint` against the
   `source_fingerprint` in the manifest of the calibrated artifact the S3a suite converts
   from — the plan-of-record's *"a cell asserting the committed pin matches the artifact the
   S3a suite converts from"*. This leg needs the artifact directory, which is not in this
   repository; it reports UNAVAILABLE rather than passing vacuously when it is absent, on
   the same visible-skip discipline as T-1320.

Exits non-zero and names the failing check. `--artifact` overrides the artifact directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PIN_PATH = HERE / "upstream_pin.json"

DEFAULT_ARTIFACT = pathlib.Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct-w8a8")

REQUIRED_PIN_FIELDS = (
    "model_repo", "model_revision", "transformers_version", "torch_version",
    "artifact_source_fingerprint", "corpus_sha256", "config_sha256", "tokenizer_sha256",
    "tokenizer_config_sha256", "generation_config_sha256", "generation_config",
)

# Qwen2.5-1.5B-Instruct's vocabulary, from its own config.json. Carried as a literal because
# this check must run without the checkpoint; the pin's `config_sha256` is what ties the
# literal to the file it came from.
VOCAB_SIZE = 151936


class PinFailure(Exception):
    """A pinned oracle that does not check out. Never downgraded to a warning."""


def load_pin(path=PIN_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(payload) -> bytes:
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8")


def check_self_consistent(pin: dict) -> None:
    recorded = pin.get("content_sha256")
    if not recorded:
        raise PinFailure("the pin carries no content_sha256; it cannot attest to itself")
    body = {k: v for k, v in pin.items() if k != "content_sha256"}
    actual = hashlib.sha256(canonical_bytes(body)).hexdigest()
    if actual != recorded:
        raise PinFailure(
            f"content_sha256 mismatch: the file hashes to {actual} and records {recorded}. "
            f"The pinned oracle has been edited since it was generated. Regenerate it with "
            f"gen_upstream_pin.py --force rather than correcting the hash"
        )


def check_structure(pin: dict) -> None:
    if pin.get("schema") != "superslm.s3a.item2a.upstream_pin":
        raise PinFailure(f"unexpected schema {pin.get('schema')!r}")

    fields = pin.get("pin", {})
    missing = [name for name in REQUIRED_PIN_FIELDS if fields.get(name) in (None, "")]
    if missing:
        raise PinFailure(
            f"pin fields absent or null: {missing}. A content-addressed pin with a hole in "
            f"it addresses nothing"
        )

    members = pin.get("pack", {}).get("members", [])
    if [m["index"] for m in members] != [1, 2, 3, 4, 5]:
        raise PinFailure(
            f"pack members are {[m.get('index') for m in members]}; the closed pack "
            f"(D-SLM350) is exactly five, in band-position order"
        )

    declared = pin["decode"]["max_new_tokens"]
    for member in members:
        tokens = member["upstream_tokens"]
        if len(tokens) != declared:
            raise PinFailure(
                f"member {member['index']} carries {len(tokens)} upstream tokens against a "
                f"declared {declared}"
            )
        if len(member["input_ids"]) != member["token_count"]:
            raise PinFailure(
                f"member {member['index']}'s input_ids length disagrees with its token_count")
        out_of_range = [t for t in tokens + member["input_ids"]
                        if not 0 <= t < VOCAB_SIZE]
        if out_of_range:
            raise PinFailure(
                f"member {member['index']} carries token ids outside [0, {VOCAB_SIZE}): "
                f"{out_of_range[:5]}"
            )


def check_artifact_tie(pin: dict, artifact_dir: pathlib.Path) -> str:
    """The pin against the calibrated artifact the S3a suite converts from.

    Returns "OK" or "UNAVAILABLE: <reason>". Never returns OK for an absent artifact — a
    provenance check that passes when it cannot see the thing it checks is worse than no
    check, because it reads green on a report.
    """
    manifest = artifact_dir / "manifest.json"
    if not manifest.exists():
        return (f"UNAVAILABLE: {artifact_dir} carries no manifest.json; the artifact the "
                f"S3a suite converts from is not present on this machine")
    recorded = json.loads(manifest.read_text(encoding="utf-8")).get("source_fingerprint")
    expected = pin["pin"]["artifact_source_fingerprint"]
    if recorded != expected:
        raise PinFailure(
            f"the pinned oracle was generated against artifact fingerprint {expected} and "
            f"{artifact_dir} carries {recorded}. The upstream token sequences are about a "
            f"different calibrated model than the one S3a converts"
        )
    return "OK"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--pin", default=str(PIN_PATH))
    args = parser.parse_args(argv)

    pin = load_pin(pathlib.Path(args.pin))
    try:
        check_self_consistent(pin)
        print("self-consistency      OK")
        check_structure(pin)
        print("structure             OK")
        tie = check_artifact_tie(pin, pathlib.Path(args.artifact))
        print(f"artifact tie          {tie}")
    except PinFailure as failure:
        print(f"UPSTREAM PIN FAILED: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
