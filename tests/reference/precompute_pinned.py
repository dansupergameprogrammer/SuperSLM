#!/usr/bin/env python3
"""One-time precompute for tests/reference/superslm_spike/rope_tables_pinned.json
(S-HARDEN-5, F3, design S3.2).

`rope.rope_tables()` derives its cos_q30/sin_q30 entries via `math.cos`/`math.sin`,
and `intmath.py`'s module-level `_LN2 = math.log(2)` feeds the i-exp quantum every
fixture that calls `i_exp_ln2_quantum` -- both are libm transcendentals, which are
not IEEE-correctly-rounded and are demonstrably ULP-different across
glibc/MSVCRT/Apple libm. A committed fixture golden that embeds their live output
is a function of the runner's C library, resolved fresh at generation time, not of
the repository's own pinned content.

This script runs those two transcendental-dependent computations EXACTLY ONCE,
against the vendored reference (tests/reference/superslm_spike/), and commits the
concrete integer/float results to rope_tables_pinned.json. `tests/gen_intmath_fixtures.py`
reads that file instead of calling `rope.rope_tables(...)` or recomputing the
LN2-derived quantum at generation time -- so the fixture generator's own regeneration
path never executes math.sin/cos/log/exp/pow again after this file is committed
(the concrete precomputed inputs are pinned, not merely the Python source that could
still recompute them).

This script is NOT part of the `generators` CI job and is not itself regenerated or
diffed by it -- it is run by hand, once, whenever the vendored reference is
deliberately re-vendored (tests/reference/PROVENANCE.md). Re-running it must
reproduce the committed rope_tables_pinned.json byte-for-byte on the same machine;
its output is not certified cross-platform by this script itself -- that guarantee
is exactly what removing the live computation from the CI-facing generation path
establishes (S-HARDEN-5 design S3.2, S8 step 1c's note).
"""

from __future__ import annotations

import json
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPIKE_DIR = os.path.join(_THIS_DIR, "superslm_spike")
sys.path.insert(0, _SPIKE_DIR)
sys.path.insert(0, _THIS_DIR)

import intmath as im  # noqa: E402
import superslm_spike.rope as rope  # noqa: E402

OUT_PATH = os.path.join(_SPIKE_DIR, "rope_tables_pinned.json")

# The exact (head_dim, context_cap, theta) gen_intmath_fixtures.py's general-angle
# rope cases use -- a Qwen2.5-1.5B-shaped RoPE configuration.
HEAD_DIM = 128
CONTEXT_CAP = 128
THETA = 1000000.0


def main() -> None:
    cos_table, sin_table = rope.rope_tables(HEAD_DIM, CONTEXT_CAP, THETA)
    ln2 = im._LN2  # noqa: SLF001 -- the exact module constant gen_intmath_fixtures.py pins

    payload = {
        "_comment": (
            "Precomputed once from the vendored superslm_spike reference "
            "(tests/reference/precompute_pinned.py). gen_intmath_fixtures.py reads "
            "this file instead of calling rope.rope_tables(...) or math.log(2) at "
            "generation time."
        ),
        "rope": {
            "head_dim": HEAD_DIM,
            "context_cap": CONTEXT_CAP,
            "theta": THETA,
            "cos_q30": cos_table,
            "sin_q30": sin_table,
        },
        "ln2": {
            "hex": ln2.hex(),
            "repr": repr(ln2),
        },
    }

    with open(OUT_PATH, "w", encoding="ascii", newline="\n") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")

    print(f"wrote {OUT_PATH}")
    print(f"rope: {CONTEXT_CAP} positions x {HEAD_DIM // 2} pairs")
    print(f"ln2: {ln2!r} ({ln2.hex()})")


if __name__ == "__main__":
    main()
