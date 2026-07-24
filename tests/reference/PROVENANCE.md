# Provenance — vendored reference implementation

This directory vendors, verbatim, the two files of `D:\Wizard\Tools\superslm_spike`
that `tests/gen_intmath_fixtures.py` and `tests/gen_matmul_fixtures.py` import as
their pinned reference implementation (S-HARDEN-5, F3). The copy is byte-identical
to a specific, hashed, dated source state, machine-checked by
`tests/reference/check_provenance.py` rather than by this file alone.

## Source

- Source repository: `D:\Wizard` (a separate repository from this one).
- Source path: `Tools/superslm_spike/{intmath.py,rope.py}`.
- Source commit (last commit to touch either file, as of vendoring):
  `38bc8929e0933f901611ad4e979420d1321f01a7`.

## Vendored files — SHA-256

| File | SHA-256 |
|---|---|
| `superslm_spike/intmath.py` | `d780f5f17f1cd9d0db83359adcf541506d413bf375e2468074a25e22741a52da` |
| `superslm_spike/rope.py` | `4e54dda3cf9004d63700732d63419d0135ee459db8b5ecf428c441c4c08bcff7` |
| `superslm_spike/rope_tables_pinned.json` | `4e79a90101c3447302296f93ef0bdd3240a7972fbcdb76a3ac6acbf1fe1c66b7` |

`rope_tables_pinned.json` is not itself a vendored copy of an upstream file --
it is the one-time precomputed output of `tests/reference/precompute_pinned.py`,
run once against the two vendored files above, and committed here for exactly
the same reason they are: so `tests/gen_intmath_fixtures.py` never needs to call
a libm transcendental (`math.cos`/`math.sin`/`math.log`) at generation time
(S-HARDEN-5 design S3.2).

`check_provenance.py` recomputes each SHA-256 from disk and compares it against the
values recorded above, exiting non-zero and naming the mismatched file if any
disagrees. A re-vendor is a deliberate commit: copy the updated source files,
re-run `precompute_pinned.py`, record the new source commit above, and recompute
and update these hashes together.
