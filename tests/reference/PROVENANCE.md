# Provenance — vendored reference implementation

This directory vendors, verbatim, files of `D:\Wizard\Tools\superslm_spike`
that this repository's own generators/gates import as pinned reference
material (S-HARDEN-5, F3). Each copy is byte-identical to a specific, hashed,
dated source state, machine-checked by `tests/reference/check_provenance.py`
rather than by this file alone.

## Source

- Source repository: `D:\Wizard` (a separate repository from this one).
- Source path: `Tools/superslm_spike/{intmath.py,rope.py}` — full-file vendors,
  imported by `tests/gen_intmath_fixtures.py`/`tests/gen_matmul_fixtures.py`.
- Source commit (last commit to touch either file, as of vendoring):
  `38bc8929e0933f901611ad4e979420d1321f01a7`.
- Source path: `Tools/superslm_spike/pipeline.py`, lines 191 and 2253 only —
  a NARROW EXCERPT (`superslm_spike/pipeline_prob_width_ceiling.py`, D-SLM367,
  S3.3), never the whole file (F-S3-3, `SuperSLM_S3a_WalkingSkeleton_Plan.md`
  §4.3: reading the rest of this file's own logic to build a "join" oracle
  produces a correlated oracle, not an independent one — this excerpt exists
  solely to tie the shipped C++ numerator ceiling to the name Dan's ruling
  ties it to). Source commit `ca67e90ead90373fc55680a67e2b41e0d7c9abca`;
  whole-file SHA-256 `80daca8cd134d8798b3a49b5d315fb06b7b33e73d1c96b564f55d6e9da7a984e`
  (recorded here as the re-vendor trigger — ANY edit to `pipeline.py`
  invalidates this excerpt's pin, even one outside lines 191/2253, matching
  the same conservative whole-file-invalidates convention `intmath.py`/
  `rope.py` already use).

## Vendored files — SHA-256

| File | SHA-256 |
|---|---|
| `superslm_spike/intmath.py` | `d780f5f17f1cd9d0db83359adcf541506d413bf375e2468074a25e22741a52da` |
| `superslm_spike/rope.py` | `4e54dda3cf9004d63700732d63419d0135ee459db8b5ecf428c441c4c08bcff7` |
| `superslm_spike/rope_tables_pinned.json` | `4e79a90101c3447302296f93ef0bdd3240a7972fbcdb76a3ac6acbf1fe1c66b7` |
| `superslm_spike/pipeline_prob_width_ceiling.py` | `6f79f2c75e274a8b5218e75b4a58bee20fc35bf9c652afef2d0353e19c46398b` |

`rope_tables_pinned.json` is not itself a vendored copy of an upstream file --
it is the one-time precomputed output of `tests/reference/precompute_pinned.py`,
run once against the two vendored files above, and committed here for exactly
the same reason they are: so `tests/gen_intmath_fixtures.py` never needs to call
a libm transcendental (`math.cos`/`math.sin`/`math.log`) at generation time
(S-HARDEN-5 design S3.2). `pipeline_prob_width_ceiling.py`'s own SHA-256 is of
the excerpt file as committed here (self-consistency — catches a hand-edit
after vendoring); the whole-file source SHA-256 recorded above it is the
re-vendor trigger against `D:\Wizard`'s own `pipeline.py`.

`check_provenance.py` recomputes each SHA-256 from disk and compares it against the
values recorded above, exiting non-zero and naming the mismatched file if any
disagrees. A re-vendor is a deliberate commit: copy the updated source files,
re-run `precompute_pinned.py` (for the full vendors), record the new source
commit(s) above, and recompute and update these hashes together.

## Cross-repo reach — `tools/reference_pipeline/` vs this directory's frozen copy

T-2123/T-2137 (converter vendoring, design SS3.5 ruling item 1) vendors a second,
**live**, product copy of `intmath.py`/`rope.py` at `tools/reference_pipeline/`
(the converter's own runtime dependency), separate from this directory's frozen
golden-fixture copy above. The two copies are supposed to be able to diverge — a
live copy gets a bug fix; the frozen copy stays frozen until someone deliberately
re-vendors it — but a divergence nobody decided and nobody can see is not
acceptable. `check_provenance.py`'s cross-repo reach check passes a file on either
byte-identity between the two locations, or an explicit, dated row below recording
the two as intentionally decoupled.

| File | Date | Reason |
|---|---|---|
| `rope.py` | 2026-08-16 | T-2123/T-2137 B0 rewrote `tools/reference_pipeline/rope.py`'s internal import from `from superslm_spike.intmath import rounding_divide_by_pot` to `from reference_pipeline.intmath import rounding_divide_by_pot` as part of vendoring the module out of the cross-tree `D:\Wizard` dependency (§3.2). This directory's frozen copy under `superslm_spike/rope.py` is untouched by that rename (S-HARDEN-5's own frozen-copy discipline: it stays pinned to its recorded source commit above). The two files therefore differ by exactly this one import line; every other line is byte-identical. |

`intmath.py` carries no row here because it has no internal `superslm_spike`-prefixed
import to rewrite (checked at source, T-2123/T-2137 design §2.1) — its two copies
remain byte-identical and the cross-repo check enforces that directly.
