#!/usr/bin/env python3
"""One-time precompute for tests/reference/superslm_spike/criterion2_prompt_pack_pinned.json
(T-1522, SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.1c item 3 -- the reference pack §12
item 3 pins, D-SLM350).

The producer (`run_criterion2_trace.py`) must run `dynamic_engine.forward_dynamic_vec`
"once per reference prompt from §12 item 3's pack" -- the closed five-member pack specified
in `Claude/Vitruvius/SuperSLM_S3a_ReferencePromptPack_Design-2026-07-28.md` and already
implemented, in `D:\\Wizard`, as `Tools/superslm_spike/s3a_parity/prompt_pack.py`. Rendering
a pack member to token ids requires the checkpoint's own tokenizer and chat template
(`transformers.AutoTokenizer.apply_chat_template`) -- a live, real-weight-adjacent dependency
this repository's own vendoring boundary (Sec11 S3.1c item 1) deliberately keeps out of the
vendored closure ("`baseline.py` alone would pull `torch` and `transformers` into a build
that must never acquire a real-weight dependency").

This mirrors `precompute_pinned.py`'s own established convention for exactly this shape of
problem: run the live, environment-dependent computation EXACTLY ONCE, by hand, against
`D:\\Wizard`'s own tooling, and commit the concrete output -- so the CI-facing producer needs
only the vendored closure itself, numpy, and this pinned JSON, and never touches a
tokenizer, `transformers`, or `D:\\Wizard` at build/CI time. `criterion2_prompt_pack_pinned.json`
is a PRECOMPUTED DERIVATIVE, not a vendored copy of an upstream file -- like
`rope_tables_pinned.json`, it is hashed by `check_provenance.py` for SELF-consistency only (a
hand-edit after the fact), not as a byte-identical copy of any one source-repo path; the design
record it transcribes (D-SLM350) is the specification, and the render's own
`token_count`/`pack_fingerprint` cross-check below is what proves this transcription is
faithful to it.

This script itself is on `check_no_criterion2_closure_imports.py`'s allowlist (T-1523):
it is not one of item 3's own three named pieces (producer/consumer/comparator), but it is
the same shape of thing under that guard's own stated principle -- Python-side tooling for
the criterion-2 harness's own operation, never the C++ engine's implementation, and never a
runtime dependency any shipped binary acquires.

NOT part of any CI job and not itself regenerated or diffed by one -- run by hand, once,
whenever the reference pack's own design record changes (which is a closed artifact, per
D-SLM350, and is not expected to). Requires a `D:\\Wizard` checkout on
`SUPERSLM_REFERENCE_SOURCE` (the same environment variable `check_source_drift.py` uses) with
`Tools/superslm_spike` importable and the `Qwen/Qwen2.5-1.5B-Instruct` tokenizer available to
`transformers` (locally cached is sufficient; `HF_HUB_OFFLINE=1` is honoured, never required).
"""

from __future__ import annotations

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPIKE_DIR = os.path.join(_THIS_DIR, "superslm_spike")
OUT_PATH = os.path.join(_SPIKE_DIR, "criterion2_prompt_pack_pinned.json")

CHECKPOINT = "Qwen/Qwen2.5-1.5B-Instruct"
DESIGN_RECORD = "Claude/Vitruvius/SuperSLM_S3a_ReferencePromptPack_Design-2026-07-28.md (D-SLM350)"

ENV_VAR = "SUPERSLM_REFERENCE_SOURCE"


def main() -> int:
    source = os.environ.get(ENV_VAR)
    if not source or not os.path.isdir(source):
        print(
            f"precompute_criterion2_prompt_pack.py: {ENV_VAR} must point at a D:\\Wizard "
            f"checkout with Tools/superslm_spike importable -- got {source!r}",
            file=sys.stderr,
        )
        return 1

    tools_dir = os.path.join(source, "Tools")
    if not os.path.isdir(os.path.join(tools_dir, "superslm_spike")):
        print(
            f"precompute_criterion2_prompt_pack.py: {tools_dir!r} carries no superslm_spike "
            f"package -- {ENV_VAR} does not point at a real D:\\Wizard checkout",
            file=sys.stderr,
        )
        return 1

    # This repository's OWN vendored tests/reference/superslm_spike/ carries an
    # __init__.py (a regular package); D:\Wizard's Tools/superslm_spike/ does
    # not (an implicit PEP 420 namespace package). CPython's import system
    # prefers a regular package over a namespace one REGARDLESS of sys.path
    # order once both are visible in the same process -- so simply inserting
    # tools_dir at sys.path[0] is not enough while this script's own
    # directory (or '', the empty-string CWD entry) is also on sys.path and
    # resolves the vendored copy first. Strip every sys.path entry that could
    # resolve to THIS repository's vendored package before importing from
    # D:\Wizard, and restore the original sys.path afterward -- this script
    # is a one-time, hand-run tool, not something the shipped producer or any
    # CI job imports, so scoping the path swap to this one import is safe.
    original_sys_path = list(sys.path)
    sys.path[:] = [
        p for p in sys.path
        if os.path.normcase(os.path.normpath(p or os.getcwd())) != os.path.normcase(os.path.normpath(_THIS_DIR))
    ]
    sys.path.insert(0, tools_dir)
    try:
        from superslm_spike.s3a_parity import prompt_pack
        from superslm_spike.pipeline import _checkpoint_tokenize_prompt
    finally:
        sys.path[:] = original_sys_path

    members = prompt_pack.reference_prompt_pack()
    tokenize_prompt = _checkpoint_tokenize_prompt(CHECKPOINT)
    rendered = prompt_pack.render(members, tokenize_prompt)  # raises on any expected-count drift
    fingerprint = prompt_pack.pack_fingerprint(rendered)

    payload = {
        "_comment": (
            "Precomputed once (tests/reference/precompute_criterion2_prompt_pack.py) from "
            f"the design record ({DESIGN_RECORD}), rendered through the real checkpoint's own "
            "chat template and tokenizer. run_criterion2_trace.py reads this file instead of "
            "calling transformers.AutoTokenizer at build/CI time."
        ),
        "checkpoint": CHECKPOINT,
        "design_record": DESIGN_RECORD,
        "fingerprint": fingerprint,
        "members": [
            {
                "index": member.index,
                "axis": member.axis,
                "source": member.source,
                "expected_verdict": member.expected_verdict,
                "input_ids": ids,
            }
            for member, ids in zip(members, rendered)
        ],
    }

    with open(OUT_PATH, "w", encoding="ascii", newline="\n") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")

    print(f"wrote {OUT_PATH}")
    print(f"{len(payload['members'])} members, fingerprint {fingerprint}")
    for m in payload["members"]:
        print(f"  #{m['index']} {m['axis']!r}: {len(m['input_ids'])} tokens, {m['expected_verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
