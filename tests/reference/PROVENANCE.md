# Provenance — vendored reference implementation

This directory vendors, verbatim, files of `D:\Wizard\Tools\superslm_spike`
that this repository's own generators/gates import as pinned reference
material (S-HARDEN-5, F3; extended for §12 criterion 2, T-1519/T-1520). Each
copy is byte-identical to a specific, hashed, dated source state, machine-checked
by `tests/reference/check_provenance.py` rather than by this file alone.

## Source

**For the six rows below that are full-file vendored copies of a `D:\Wizard`
file, the recorded SHA-256 is computed over the source file's git-blob content
(LF-normalized), never a Windows working-tree checkout (CRLF) of it.** `D:\Wizard`
carries `* text=auto` and this repository carries `*.py eol=lf`, so `git show
<commit>:<path>` and this directory's own committed copies agree; a raw disk
read of a Windows checkout of `D:\Wizard`'s `.py` files does not, and hashes
to a different value for the same content. The other three rows --
`pipeline_prob_width_ceiling.py` (a locally-authored excerpt) and
`rope_tables_pinned.json`/`criterion2_prompt_pack_pinned.json` (precomputed
derivatives) -- have no upstream source file to hash against; their recorded
SHA-256 is a self-hash of this repository's own committed content, for
self-consistency only, detailed in the notes below the table.

- Source repository: `D:\Wizard` (a separate repository from this one).
- Source path: `Tools/superslm_spike/{intmath.py,rope.py}` — full-file vendors,
  imported by `tests/gen_intmath_fixtures.py`/`tests/gen_matmul_fixtures.py`.
  Source commit (last commit to touch either file, as of vendoring):
  `38bc8929e0933f901611ad4e979420d1321f01a7`.
- Source path: `Tools/superslm_spike/pipeline.py`, lines 191 and 2253 only —
  a NARROW EXCERPT (`superslm_spike/pipeline_prob_width_ceiling.py`, D-SLM367,
  S3.3), never the whole file (F-S3-3, `SuperSLM_S3a_WalkingSkeleton_Plan.md`
  §4.3: reading the rest of this file's own logic to build a "join" oracle
  produces a correlated oracle, not an independent one — this excerpt exists
  solely to tie the shipped C++ numerator ceiling to the name Dan's ruling
  ties it to). Source commit `ca67e90ead90373fc55680a67e2b41e0d7c9abca`;
  whole-file SHA-256 `27affab8d53085532d392d947e01e125859ef8236f55abdebb49eb6b1c0dcf49`
  (the git-blob hash, matching the `criterion2-closure` row below for the same
  file at the same commit; recorded here as the re-vendor trigger — ANY edit to `pipeline.py`
  invalidates this excerpt's pin, even one outside lines 191/2253, matching
  the same conservative whole-file-invalidates convention `intmath.py`/
  `rope.py` already use).
- Source path: `Tools/superslm_spike/{dynamic_engine.py,pipeline.py,silu_lut.py,
  constrain.py}` — full-file vendors, the transitive import closure §12
  criterion 2's comparator (`tests/reference_parity_driver`, T-1522) runs
  against (`SuperSLM_S3a_WalkingSkeleton_Plan.md` §11 S3.1c item 1). Source
  commit (the latest commit in `D:\Wizard` to touch any of the four files, as
  of vendoring 2026-08-05): `ca67e90ead90373fc55680a67e2b41e0d7c9abca` — the
  same commit already pinning the `pipeline.py` excerpt above, confirmed by
  `git log` at vendoring time rather than assumed from that coincidence.

## Vendored files — SHA-256

| File | SHA-256 | Source commit | Group |
|---|---|---|---|
| `superslm_spike/intmath.py` | `d780f5f17f1cd9d0db83359adcf541506d413bf375e2468074a25e22741a52da` | `38bc8929e0933f901611ad4e979420d1321f01a7` | `intmath-rope` |
| `superslm_spike/rope.py` | `4e54dda3cf9004d63700732d63419d0135ee459db8b5ecf428c441c4c08bcff7` | `38bc8929e0933f901611ad4e979420d1321f01a7` | `intmath-rope` |
| `superslm_spike/rope_tables_pinned.json` | `4e79a90101c3447302296f93ef0bdd3240a7972fbcdb76a3ac6acbf1fe1c66b7` | `38bc8929e0933f901611ad4e979420d1321f01a7` | `intmath-rope` |
| `superslm_spike/pipeline_prob_width_ceiling.py` | `77ad28077b2f39fc64541ad3dd9be889b273149e30e1eaab5d8455537068f680` | `ca67e90ead90373fc55680a67e2b41e0d7c9abca` | `pipeline-ceiling` |
| `superslm_spike/dynamic_engine.py` | `593e9b41dff762d783a620dff28a612e15a086fc85735405c1f41ecb1774a059` | `ca67e90ead90373fc55680a67e2b41e0d7c9abca` | `criterion2-closure` |
| `superslm_spike/pipeline.py` | `27affab8d53085532d392d947e01e125859ef8236f55abdebb49eb6b1c0dcf49` | `ca67e90ead90373fc55680a67e2b41e0d7c9abca` | `criterion2-closure` |
| `superslm_spike/silu_lut.py` | `be17bf7edfebb7d9931942aa6d01fbf86a769b1b6b94c96db1e273666fbd7995` | `ca67e90ead90373fc55680a67e2b41e0d7c9abca` | `criterion2-closure` |
| `superslm_spike/constrain.py` | `50b3396edbf4f32e3cea4fbc9e42a0b1159e8cd5e122b2a672450a98e804bdd2` | `ca67e90ead90373fc55680a67e2b41e0d7c9abca` | `criterion2-closure` |
| `superslm_spike/criterion2_prompt_pack_pinned.json` | `b8b7582d44d16b1158be242495b04ba1d1648f33829d8c68b02ca52e6009d927` | `0dca923bcf3bd05b79859c8023590bb211fc214c` | `criterion2-prompt-pack` |

`rope_tables_pinned.json` is not itself a vendored copy of an upstream file --
it is the one-time precomputed output of `tests/reference/precompute_pinned.py`,
run once against the two vendored files above, and committed here for exactly
the same reason they are: so `tests/gen_intmath_fixtures.py` never needs to call
a libm transcendental (`math.cos`/`math.sin`/`math.log`) at generation time
(S-HARDEN-5 design S3.2). `pipeline_prob_width_ceiling.py`'s own SHA-256 is of
the excerpt file as committed here (self-consistency — catches a hand-edit
after vendoring); the whole-file source SHA-256 recorded above it is the
re-vendor trigger against `D:\Wizard`'s own `pipeline.py`.

`superslm_spike/criterion2_prompt_pack_pinned.json` is also not a vendored
copy of an upstream file, for the same reason and shape as
`rope_tables_pinned.json`: it is the one-time precomputed output of
`tests/reference/precompute_criterion2_prompt_pack.py`, run once against
`D:\Wizard`'s `Tools/superslm_spike/s3a_parity/prompt_pack.py` (the closed
five-member reference pack, D-SLM350) and the checkpoint's own live tokenizer,
and committed here so `tests/reference/run_criterion2_trace.py` (T-1522) never
needs `transformers`, a tokenizer, or a `D:\Wizard` checkout at build/CI time.
Its recorded `Source commit` is the last commit in `D:\Wizard` to touch either
`prompt_pack.py` or the design record it transcribes — a staleness signal
(re-precompute if either changes), not a byte-identity claim: unlike the
full-file vendors above, this file's content is the tokenizer's OUTPUT over
that source, not a copy of it, so `check_source_drift.py`'s byte-comparison
mechanism does not apply to it (scope note, T-1522 build).

**The `Group` column is a machine-checked property, not a documentary one
(T-1529).** Every row sharing a `Group` value must record an identical
`Source commit` — `check_provenance.py` groups the parsed rows and asserts
this, failing loudly and naming the group and the disagreeing files/commits if
not. Four groups exist: `intmath-rope` (the two original full-file vendors
plus `rope_tables_pinned.json`, already sharing one commit informally before
this mechanism existed — folded under it rather than left an unchecked
convention); `pipeline-ceiling` (the narrow excerpt, a group of one, trivially
self-consistent, deliberately pinned independently of the wide closure below,
D-SLM367); `criterion2-closure` (the four files this section adds — the group
this property exists for, since `dynamic_engine.py`'s bit-equality claim
against `pipeline.py` is proven jointly by both files at one shared state, and
pinning them at different commits could combine two states that claim was
never proven to hold between); `criterion2-prompt-pack` (the precomputed
reference-pack fixture above, a group of one, self-consistent by construction).

`check_provenance.py` recomputes each SHA-256 from disk and compares it against the
values recorded above, exiting non-zero and naming the mismatched file if any
disagrees; it separately groups the parsed rows by `Group` and asserts every
row within a group records an identical `Source commit`, exiting non-zero and
naming the group and the disagreeing files/commits if not. A re-vendor is a
deliberate commit: copy the updated source files, re-run `precompute_pinned.py`
(for the full vendors), record the new source commit(s) above, and recompute
and update these hashes together.
