"""T-1689's own red-first fixture: a minimal, fully synthetic, hand-computed
`.sslm` model artifact (plus a trivial tokenizer artifact) driving
`out/sslm_layer_trace.exe` end to end -- the SAME binary the real nine-prompt
census (kv_saturation_report.py) drives -- with weights chosen so that a
known, hand-computed subset of K/V-landing values exceed the pinned
[-127,127] code range (design Claude/Vitruvius/superslm-t1683-source-
attribution-design-2026-08-02.md S5's own red-first-proof requirement).

Every intermediate value below (root, wide[], d_prime, the normalized
scale, the requantized codes, the K/V-landing accumulate, and the
LandingRescale output) is derived by hand from the engine's own arithmetic,
transcribed from source (RmsNormSite/forward_sites.cpp, NormalizeScale/
DynamicScaleReciprocal/RequantTokenCodeWide/CombineCarriedScale/intmath.cpp
and checked_chain_funnel.cpp, LandingRescale/ClampRopeCode/forward_sites.cpp)
-- not estimated from output behavior. See the T-1689 build log
(Claude/Brunel/t1689-kv-saturation-census-build-2026-08-02.md, in the
D:\\Wizard worktree) for the full derivation.

Geometry: hidden_size=2, num_hidden_layers=1, num_attention_heads=1,
num_key_value_heads=1, head_dim=2, intermediate_size=2, context_cap=1,
vocab_size=4, tie_word_embeddings=True. A single-token prompt (the letter
"a", encoded by a one-entry tokenizer to token id 0) -- no prefill, position
0 only, matching this campaign's own scope.

The hand derivation, for embed_weights row 0 = {5,-5} (any nonzero
antipodal-ish int8 pair works -- RMSNorm's own ratio-based construction
makes the downstream chain invariant to it, proven by this same derivation
covering EmbedEntry's own identical RequantChainChecked composition):

  RmsNormSite(h={5,-5}, gain={1,1}, site_constant=CANONICAL):
    sumsq = 50; root = ISqrt(50<<32 / 2) = ISqrt(2^30*100) = 327680 (exact:
    100*2^30 = (2^15*10)^2). wide[i] = floor(h[i]<<32 / root) * g[i] =
    {65536, -65536} (exact: 5*2^32 / (2^15*10) = 2^17*0.5 = 65536).
    d_prime = MaxAbsReduceWide(wide) = 65536. NormalizeScale(65536):
    p = 63-clz64(65536) = 16, s = 30-16 = 14, dn = 65536<<14 = 2^30 exactly.
    r = DynamicScaleReciprocal(2^30) = 2^32 (model.cpp's own pinned
    reference point: "DynamicScaleReciprocal(2^30) == 4294967296").
    d_prime_factor = {2^30, -14}. running = CombineCarriedScale(CANONICAL=
    {2^30,-30}, {2^30,-14}): e = (-30 + -14 + 31) = -13; m =
    SaturatingRoundingDoublingHighMul(2^30,2^30) = round(2^60/2^31) = 2^29
    exactly; since m < 2^30, m<<=1 -> 2^30, e-=1 -> -14. So
    normed_scale = {m=2^30, e=-14} (NOT canonical -- CombineCarriedScale's
    own fold, not a re-assertion of the input).
    out_codes[i] = RequantTokenCodeWide(wide[i], r=2^32, s=14): exponent =
    62-14 = 48; magnitude = round(|x|*127*r / 2^48). For |x|=65536=2^16:
    127*2^16*2^32/2^48 = 127*2^(16+32-48) = 127*2^0 = 127 exactly.
    normed = {127, -127}.

  K/V landing (identity {2 K weight, e_t chosen so the composed rescale is
  a true identity on this normed_scale}): kv_landing e_t = -14, r_t =
  2^(62-(e_a-e_t))/m_a = 2^(62-0)/2^30 = 2^32 = 4294967296 (checked: within
  the loader's own [2^31+1, 2^32] domain, docs/sslm_format.md /
  model.cpp's own KvLandingReciprocalOutOfDomain range, inclusive at the
  ceiling).
    LandingRescale(branch_code, m_a=2^30, r_t=2^32, e_a=-14, e_t=-14):
    exponent = 62-(e_a-e_t) = 62-0 = 62. result = branch_code * 2^30 * 2^32
    / 2^62 = branch_code EXACTLY (2^30*2^32 == 2^62).

  IDENTITY k/v weight [[1,0],[0,1]]: kacc = {normed[0]*1, normed[1]*1} =
  {127, -127}. LandingRescale -> {127, -127} (in [-127,127] INCLUSIVE --
  ClampRopeCode's own domain, forward_sites.cpp -- no clamp fires).
  Expected per-layer saturation delta: 0 (K: 0 + V: 0).

  SATURATING k/v weight [[127,0],[0,0]]: kacc = {normed[0]*127, 0} =
  {16129, 0}. LandingRescale(16129,...) = 16129 EXACTLY (same identity
  rescale) -- outside [-127,127], ClampRopeCode saturates to 127, and
  LandingRescale's own predicated increment fires (magnitude_exceeds_clamp,
  forward_sites.cpp). kacc[1]=0 stays in domain, no increment.
  Expected per-layer saturation delta: 2 (K dim0 + V dim0; dim1 both
  in-domain on both K and V, matching this campaign's own
  CriticalOneFixture precedent shape, tests/test_main.cpp).

RoPE (cos=2^30 ["1.0"], sin=0 -- an identity rotation, forward_sites.cpp's
RopeApplySite) leaves the just-landed K unchanged after this instrument's
own K/V-landing count has already been taken, so it does not perturb either
prediction above.

q_proj/o_proj/gate_proj/up_proj are the ALL-ZERO weight matrix, not identity
-- this is a build-time-executed finding, not a hand derivation (below).
K/V-landing saturation is the only mechanism under test; the attention
score, MLP, and residual-reconcile paths are otherwise unexercised
(un-derived) territory this fixture does not need to compose meaningfully,
only successfully. A zero q_proj/o_proj/gate_proj/up_proj is the smallest
change that keeps the layer's OTHER four RequantChainChecked-family sites
(attn_ctx/o_proj funnels, and both residual reconciles) trivially in
domain: a zero-weight projection produces an all-zero output row
regardless of its input's carried scale (GemmInt8AccumulateRow's own
dot-product is exactly 0 against an all-zero weight row, for any input),
so the attention branch and MLP branch both contribute nothing to their
own residual, and ResidualReconcileSite's own rescale of a genuinely-zero
branch code cannot overflow at ANY exponent mismatch between the residual
stream's carried scale and the branch's own (0 * any finite ratio is
exactly 0) -- closing the ChainInputOutOfDomain failure a non-zero
q/o/gate/up weight produced at the bench (attn_residual, then
mlp_residual, each in turn, at every embed/softmax exponent combination
tried) BY CONSTRUCTION, not by further exponent tuning. See the build log
for the executed trace (`DEBUGTRACE site=layer0.mlp_residual
d_prime=8792299636140172`, `d_prime=13746465617618` at attn_residual,
before this fix) that located the failure to these two sites specifically.

softmax_khead0 = {m=2^30, e=-60}: IExpScaleConstants's own derivation
(intmath.cpp) needs `combine(q_scale, softmax_khead)`'s exponent inside a
narrow usable band so its own M=q_b^2+q_c lands in (0, kSoftmaxRowMaxSafe
Exponent] with q_ln2>0 -- computed by a small Python transcription of
IExpScaleConstants (build log), not asserted; with q_proj all-zero (above),
q_scale is a fixed point of the fold chain (running == normed_scale ==
{2^30,-14}, since combining {2^30,-14} with the canonical q_site_constant
and then with the all-zero row's own d_prime_factor -- itself {2^30,-30},
the all-zero-row guard's NormalizeScale(1) result -- reproduces {2^30,-14}
exactly), giving a usable softmax_khead exponent band of [-68,-47]; -60 is
picked from inside it, not at either edge.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

import sslm_format as F
import sslm_model_writer as W
from unicode_tables import Unicode

# --- geometry -----------------------------------------------------------
HIDDEN_SIZE = 2
NUM_HIDDEN_LAYERS = 1
NUM_ATTENTION_HEADS = 1
NUM_KEY_VALUE_HEADS = 1
HEAD_DIM = 2
INTERMEDIATE_SIZE = 2
CONTEXT_CAP = 1
VOCAB_SIZE = 4

# --- the hand-derived constants (see module docstring) -------------------
CANONICAL = (1073741824, -30)  # {m=2^30, e=-30} -- an ordinary fixed site scale.
# T-1689 build log (Claude/Brunel/t1689-kv-saturation-census-build-2026-08-02.md):
# the loader's own CompositionScaleOutOfDomain check (model.cpp) rejects a
# CompositionConstants exponent outside a pinned "no-UB floor" -- measured at
# the bench to be [-80, 7]. CriticalOneFixture's raw C++ construction
# (tests/test_main.cpp) bypasses this loader check entirely (its LayerWeights
# fields are wired by hand, never marshaled from a loaded artifact), so its
# own mlp_act/softmax_khead constants (e=-96, e=-86) are not loader-legal and
# are not reused here -- CANONICAL is loader-legal and, since this fixture's
# own hand derivation never exercises the MLP or softmax arithmetic paths
# for its actual assertion (only the K/V landing count), any in-domain value
# is equally sound for these two sites.
MLP_ACT_CONST = CANONICAL
SOFTMAX_KHEAD_CONST = (1073741824, -60)  # see module docstring: the derived-usable range is [-68,-47].
EMBED_CONST = CANONICAL  # never folded into embed_codes itself (dynamic per-row scale) --
# but IS the residual stream's own carried scale downstream (ResidualReconcileSite).
KV_LANDING_E_T = -14
KV_LANDING_R_T = 4294967296  # 2^32, derived in the module docstring.

_IDENTITY2X2 = np.array([[1, 0], [0, 1]], dtype=np.int8)
_ZERO2X2 = np.array([[0, 0], [0, 0]], dtype=np.int8)  # module docstring: q/o/gate/up_proj.
_SATURATING_KV = np.array([[127, 0], [0, 0]], dtype=np.int8)


def _identity_fold(channels: int) -> np.ndarray:
    """A (identity=1, mult=0, shift=0) row per output channel -- WSC1's own
    per-output-channel fold format (docs/sslm_format.md), the true pass-
    through ApplyWeightScaleFold's own identity!=0 branch reads
    (forward_sites.cpp)."""
    return np.array([(1, 0, 0) for _ in range(channels)], dtype=np.int32)


def build_kv_saturation_model(saturating: bool) -> bytes:
    """Builds the one-layer model artifact. `saturating=True` uses the
    weight row that drives the K/V landing outside [-127,127] (module
    docstring); `saturating=False` uses the identity weight, the same-call
    negative control."""
    kv_weight = _SATURATING_KV if saturating else _IDENTITY2X2

    cfg = W.write_cfg1(
        hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS, num_key_value_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM, intermediate_size=INTERMEDIATE_SIZE, vocab_size=VOCAB_SIZE,
        context_cap=CONTEXT_CAP, tie_word_embeddings=True, kv_precision=0, kv_block_size=1,
        unicode_major=0, unicode_minor=0, unicode_patch=0, rope_theta=10000.0, rms_norm_eps=1e-6)

    embed = np.zeros((VOCAB_SIZE, HIDDEN_SIZE), dtype=np.int8)
    embed[0] = [5, -5]
    gain = np.ones((HIDDEN_SIZE,), dtype=np.int8)

    weights = W.write_tensor_manifest(W.WGT1, np.int8, {
        "embed": embed,
        "layer0.q_proj": _ZERO2X2,
        "layer0.k_proj": kv_weight,
        "layer0.v_proj": kv_weight,
        "layer0.o_proj": _ZERO2X2,
        "layer0.gate_proj": _ZERO2X2,
        "layer0.up_proj": _ZERO2X2,
        "layer0.down_proj": _IDENTITY2X2,
        "layer0.attn_norm.gain": gain,
        "layer0.mlp_norm.gain": gain,
        "final_norm.gain": gain,
    })

    weight_scales = W.write_tensor_manifest(W.WSC1, np.int32, {
        "layer0.q_proj": _identity_fold(HIDDEN_SIZE),
        "layer0.k_proj": _identity_fold(NUM_KEY_VALUE_HEADS * HEAD_DIM),
        "layer0.v_proj": _identity_fold(NUM_KEY_VALUE_HEADS * HEAD_DIM),
        "layer0.o_proj": _identity_fold(HIDDEN_SIZE),
        "layer0.gate_proj": _identity_fold(INTERMEDIATE_SIZE),
        "layer0.up_proj": _identity_fold(INTERMEDIATE_SIZE),
        "layer0.down_proj": _identity_fold(HIDDEN_SIZE),
        "layer0.ctx_fold": _identity_fold(NUM_ATTENTION_HEADS),
    })

    rope = W.write_tensor_manifest(W.ROP1, np.int64, {
        "cos": np.array([1073741824], dtype=np.int64),
        "sin": np.array([0], dtype=np.int64),
    })

    composition_constants = W.write_kvc1(2, {
        "embed": EMBED_CONST,
        "final_norm": CANONICAL,
        "layer0.attn_norm": CANONICAL,
        "layer0.q_proj": CANONICAL,
        "layer0.o_proj": CANONICAL,
        "layer0.attn_ctx": CANONICAL,
        "layer0.attn_residual": CANONICAL,
        "layer0.mlp_norm": CANONICAL,
        "layer0.gate_proj": CANONICAL,
        "layer0.up_proj": CANONICAL,
        "layer0.mlp_act": MLP_ACT_CONST,
        "layer0.down_proj": CANONICAL,
        "layer0.mlp_residual": CANONICAL,
        "layer0.softmax_khead0": SOFTMAX_KHEAD_CONST,
    })

    kv_landing_reciprocals = W.write_kvc1(3, {
        "layer0.k_head0": (1, KV_LANDING_E_T, KV_LANDING_R_T),
        "layer0.v_head0": (1, KV_LANDING_E_T, KV_LANDING_R_T),
    })

    sections = [
        F.Section(F.SectionType.CONFIG, cfg),
        F.Section(F.SectionType.SIGMOID_LUT, W.write_sil1()),
        F.Section(F.SectionType.WEIGHTS, weights),
        F.Section(F.SectionType.WEIGHT_SCALES, weight_scales),
        F.Section(F.SectionType.ROPE_TABLES, rope),
        F.Section(F.SectionType.COMPOSITION_CONSTANTS, composition_constants),
        F.Section(F.SectionType.KV_LANDING_RECIPROCALS, kv_landing_reciprocals),
    ]
    data, _fingerprint = F.build_artifact(sections)
    return data


def build_tiny_tokenizer() -> bytes:
    """A degenerate but structurally valid tokenizer artifact: one vocab
    entry (id 0, decoding to the byte "a"), every byte mapped to id 0 (this
    fixture only ever encodes the single-character prompt "a"), no merges,
    no specials. Real pinned Unicode tables (Unicode.build(), the same
    production table this repo's real tokenizer conversion uses) so "a"
    classifies as \\p{L} and the pretokenizer emits it as one piece."""
    byte_to_id = [0] * 256
    id_to_bytes = [b"a"]

    tok = bytearray()
    tok += b"TOK1"
    tok += struct.pack("<IIIII", 1, len(id_to_bytes), 0, 0, 0)
    for b in byte_to_id:
        tok += struct.pack("<I", b)
    offs, blob = [0], bytearray()
    for by in id_to_bytes:
        blob += by
        offs.append(len(blob))
    for o in offs:
        tok += struct.pack("<I", o)
    tok += struct.pack("<I", len(blob))
    tok += blob
    # no merges, no specials -- both counts already written as 0 above.
    tok += struct.pack("<I", 0)  # special_offsets[0] (special_count==0: one entry, value 0)
    tok += struct.pack("<I", 0)  # special_blob_len
    # (no special_blob bytes)

    u = Unicode.build()
    sections = [
        F.Section(F.SectionType.CONFIG, b'{"tokenizer":"t1689-fixture"}'),
        F.Section(F.SectionType.TOKENIZER, bytes(tok)),
        F.Section(F.SectionType.UNICODE_TABLES, u.serialize()),
        F.Section(F.SectionType.SIGMOID_LUT, W.write_sil1()),
    ]
    data, _fingerprint = F.build_artifact(sections)
    return data


def write_fixture(out_dir: Path) -> tuple[Path, Path, Path]:
    """Writes the tokenizer once and both model variants; returns
    (tokenizer_path, saturating_model_path, nonsaturating_model_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "t1689_fixture.tok.sslm"
    sat_path = out_dir / "t1689_fixture_saturating.sslm"
    nosat_path = out_dir / "t1689_fixture_identity.sslm"
    tok_path.write_bytes(build_tiny_tokenizer())
    sat_path.write_bytes(build_kv_saturation_model(saturating=True))
    nosat_path.write_bytes(build_kv_saturation_model(saturating=False))
    return tok_path, sat_path, nosat_path


PROMPT = "a"
EXPECTED_DELTAS_SATURATING = [2]
EXPECTED_DELTAS_IDENTITY = [0]
