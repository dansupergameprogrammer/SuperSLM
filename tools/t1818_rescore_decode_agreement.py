#!/usr/bin/env python3
"""T-1818 -- re-score the SLM-side decode oracle's stored token sequences under
every defensible comparison semantic.

WHY THIS EXISTS. `tools/t1800_decode_agreement.py` (and the two scripts that
copied its comparison, `tools/t1800_bf16_fp32_floor.py` and
`tools/t1810_slm_engine_vs_fp32.py`) compute their headline as

    n = min(len(a_ids), len(b_ids))
    n_agree = sum(1 for p in range(n) if a_ids[p] == b_ids[p])

pooled over the prompt population. Both arms are FREE-RUNNING greedy decodes
from an identical prompt: they share context only up to the first position at
which they choose different tokens. From that position on, each arm conditions
on its own history, so position p > d (d = first divergence) in arm A and
position p in arm B were produced from different contexts. A match there is two
different trajectories coinciding, not two arms of one computation agreeing.
The `min()` additionally discards, without recording it, every position past
the shorter arm's end when the two arms stop at different lengths.

This script does not decode. It reads the token sequences the runs already
stored in their JSON output and re-scores them, so every candidate semantic is
computed over exactly the same population with no new capture.

THE SEMANTICS COMPUTED
    fixed_index      the filed semantic: fixed index, min-length truncation,
                     pooled. Reproduces the number in the source JSON exactly;
                     included so every other column is read against it.
    valid_prefix     positions restricted to the causally comparable prefix.
                     Position p is comparable only if both arms produced
                     identical tokens at every position < p, i.e. both arms
                     were conditioning on the same context when they chose
                     position p. For a prompt whose arms first differ at index
                     d, that is positions 0..d inclusive: d agreements out of
                     d+1 comparable positions. For a prompt with no difference
                     inside the compared range, all min-length positions are
                     comparable.
    max_len_denom    fixed index in the numerator, max(len_a, len_b) in the
                     denominator: the length-mismatch-charged variant of the
                     filed semantic. Isolates what min() truncation costs.
    shift_aligned    best-shift overlap over the window T-1756's
                     `tools/agreement_alignment_guard.py` established, using
                     that module rather than a re-implementation. Reported for
                     the alignment question only; it is not a candidate for
                     the oracle's headline (see the T-1818 record).

Also reported per file: how many prompts trip the alignment guard's own refusal
threshold, and how many prompts have arms of unequal length.

Usage:
    python tools/t1818_rescore_decode_agreement.py \\
        --input <label>=<path-to-run.json> [--input ...] \\
        [--guard-path <dir containing agreement_alignment_guard.py>] \\
        [--out <path.json>]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

# T-1756's guard is not merged to main; its module is the single source for the
# shift-search convention and is loaded from its own worktree by default rather
# than re-implemented here (a second copy of a metric is a second metric).
DEFAULT_GUARD_PATH = Path(
    r"D:\SuperSLM\.worktrees\t1756-e7-alignment-guard\tools"
)


def load_guard(guard_dir: Path):
    mod_path = guard_dir / "agreement_alignment_guard.py"
    if not mod_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("agreement_alignment_guard", mod_path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because the module defines a dataclass, and
    # `dataclasses` resolves a class's own module out of `sys.modules`.
    sys.modules["agreement_alignment_guard"] = module
    spec.loader.exec_module(module)
    return module


def wilson_halfwidth(k: int, n: int, z: float = 1.96) -> float:
    """95% Wilson-score CI half-width on a proportion k/n -- the same
    construction `tools/t1800_decode_agreement.py` uses, so a resolving power
    computed here is comparable to the one it filed."""
    if n == 0:
        return float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (hi - lo) / 2


def id_fields(row: dict) -> tuple[str, str]:
    """The three source scripts name their two id lists differently
    (`int8_output_ids`/`float_output_ids`, `bf16_output_ids`/`fp32_output_ids`,
    `int8_output_ids`/`fp32_output_ids`). Take them in stored order: the first
    is the arm under test, the second is the reference."""
    keys = [k for k in row if k.endswith("_output_ids")]
    if len(keys) != 2:
        raise SystemExit(f"expected exactly two *_output_ids fields, found {keys}")
    return keys[0], keys[1]


def first_divergence(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    for p in range(n):
        if a[p] != b[p]:
            return p
    return None


def score(path: Path, guard) -> dict:
    data = json.loads(path.read_text())
    rows = data["per_prompt"]
    a_key, b_key = id_fields(rows[0])

    fi_agree = fi_total = 0
    vp_agree = vp_total = 0
    ml_total = 0
    sa_agree = sa_total = 0
    seq_match = 0
    unequal = 0
    guard_flagged = []
    per_prompt = []

    for row in rows:
        a, b = row[a_key], row[b_key]
        n = min(len(a), len(b))
        if len(a) != len(b):
            unequal += 1

        # fixed_index -- the filed semantic, recomputed rather than read back
        fi = sum(1 for p in range(n) if a[p] == b[p])
        fi_agree += fi
        fi_total += n

        # valid_prefix -- causally comparable positions only
        d = first_divergence(a, b)
        if d is not None:
            vp_a, vp_n = d, d + 1
        elif len(a) != len(b):
            # No token differs inside the shared range, but one arm stopped and
            # the other did not. That stop decision was itself made from an
            # identical context, so it is a comparable position and it
            # disagrees: n agreements out of n+1 comparable positions. Without
            # this branch the semantic would score such a prompt as perfect
            # agreement while the two arms produced different sequences.
            vp_a, vp_n = n, n + 1
        else:
            vp_a, vp_n = n, n
        vp_agree += vp_a
        vp_total += vp_n

        # max_len_denom -- same numerator, length mismatch charged
        ml_total += max(len(a), len(b))

        # shift_aligned -- best overlap in the guard's own search window
        best_shift, best = 0, fi
        if guard is not None:
            for s in range(-4, 5):
                if s == 0:
                    continue
                ov = guard._shift_overlap(a, b, s)
                if ov > best:
                    best, best_shift = ov, s
            try:
                guard.scored_agreement(a, b)
            except guard.PossibleAlignmentArtifact as exc:
                guard_flagged.append(
                    {
                        "index": row["index"],
                        "question": row.get("question"),
                        "s0_matches": exc.s0_matches,
                        "best_matches": exc.best_matches,
                        "best_shift": exc.best_shift,
                        "length": exc.length,
                        "first_divergence": d,
                    }
                )
        sa_agree += best
        sa_total += n

        seq_match += int(a == b)
        per_prompt.append(
            {
                "index": row["index"],
                "question": row.get("question"),
                "len_a": len(a),
                "len_b": len(b),
                "first_divergence": d,
                "fixed_index_agree": fi,
                "fixed_index_n": n,
                "valid_prefix_agree": vp_a,
                "valid_prefix_n": vp_n,
                "max_len_denom_n": max(len(a), len(b)),
                "shift_aligned_agree": best,
                "shift_aligned_best_shift": best_shift,
                "full_sequence_match": a == b,
            }
        )

    def rate(k, n):
        return {
            "agree": k,
            "n": n,
            "rate": (k / n) if n else float("nan"),
            "wilson_95_halfwidth": wilson_halfwidth(k, n),
        }

    return {
        "source": str(path),
        "label": data.get("label"),
        "n_prompts": len(rows),
        "arm_field": a_key,
        "reference_field": b_key,
        "unequal_length_prompts": unequal,
        "guard_flagged_prompts": guard_flagged,
        "semantics": {
            "fixed_index": rate(fi_agree, fi_total),
            "valid_prefix": rate(vp_agree, vp_total),
            "max_len_denom": rate(fi_agree, ml_total),
            "shift_aligned": rate(sa_agree, sa_total),
        },
        "full_sequence_exact_match": rate(seq_match, len(rows)),
        "per_prompt": per_prompt,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="a stored decode-agreement run to re-score; repeatable",
    )
    parser.add_argument("--guard-path", default=str(DEFAULT_GUARD_PATH))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    guard = load_guard(Path(args.guard_path))
    if guard is None:
        print(
            f"WARNING: agreement_alignment_guard.py not found under {args.guard_path}; "
            "the shift_aligned column and the guard-flag counts are omitted.",
            file=sys.stderr,
        )

    results = {}
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"--input must be LABEL=PATH, got {spec!r}")
        label, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"input not found: {p}")
        results[label] = score(p, guard)

    header = f"{'figure':<24} {'semantic':<16} {'agree/N':>12} {'rate':>9} {'+/-95%':>9}"
    print("=" * len(header))
    print("T-1818 decode-oracle re-score -- no decoding, stored sequences only")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label, res in results.items():
        for sem, v in res["semantics"].items():
            print(
                f"{label:<24} {sem:<16} {v['agree']:>5}/{v['n']:<6} "
                f"{v['rate']:>9.4f} {v['wilson_95_halfwidth']:>9.4f}"
            )
        f = res["full_sequence_exact_match"]
        print(
            f"{label:<24} {'full_seq_exact':<16} {f['agree']:>5}/{f['n']:<6} "
            f"{f['rate']:>9.4f} {f['wilson_95_halfwidth']:>9.4f}"
        )
        print(
            f"{'':<24} {'(context)':<16} unequal-length prompts="
            f"{res['unequal_length_prompts']}, guard-flagged prompts="
            f"{len(res['guard_flagged_prompts'])}"
        )
        print("-" * len(header))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"full result set written to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
