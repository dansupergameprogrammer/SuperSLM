"""Vendored excerpt of `Tools/superslm_spike/pipeline.py` — the two names the
C32 softmax-row numerator ceiling is tied to (D-SLM365/366/367; SuperSLM_S3a
_WalkingSkeleton_Plan.md Sec11 S3.3 Sec3, Sec6.2).

WHY A NARROW EXCERPT RATHER THAN THE WHOLE FILE. `pipeline.py` is thousands of
lines and most of it (including `quantize_multiplier`, the ctx_fold emitter,
etc.) is deliberately NEVER read by this campaign — F-S3-3's own finding
(SuperSLM_S3a_WalkingSkeleton_Plan.md Sec4.3) is that reading the emitter's
source to build a "join" oracle produces a correlated oracle, not an
independent one. `PROB_FRAC_BITS`/`PROB_WIDTH_CEILING` are the two exceptions:
Dan's own ruling (D-SLM367) ties the shipped C++ ceiling to this file's
existing constant BY NAME, so a test that checks the tie is checking two
citations agree, not re-deriving anything from the emitter's own logic.

Verbatim from `Tools/superslm_spike/pipeline.py` lines 191 and 2253 (source
commit `ca67e90ead90373fc55680a67e2b41e0d7c9abca`, whole-file SHA-256
`80daca8cd134d8798b3a49b5d315fb06b7b33e73d1c96b564f55d6e9da7a984e` —
PROVENANCE.md). A re-vendor is a deliberate commit: re-copy these two lines,
recompute the whole-file hash, and update PROVENANCE.md together, the same
protocol `intmath.py`/`rope.py` already follow.

Test-design record:
Claude/Curie/superslm-s3.3-attention-interior-test-design-2026-07-28.md
"""

# pipeline.py:191
PROB_FRAC_BITS = 15

# pipeline.py:2253 (D-SLM367: "the conservative 2^62-based value this guard has
# always used ... ratifies it as the shipped threshold across all three paths
# rather than widening it")
PROB_WIDTH_CEILING = (2 ** 62) >> PROB_FRAC_BITS
