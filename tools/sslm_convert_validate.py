"""sslm_convert_validate.py — the converter's validate phase (S-HARDEN-3, F13).

`convert_model.py`'s writer previously coerced every array it emitted
(`np.asarray(..., dtype=np.int8)` and friends) with no prior range check: an
`int16 [128,-129]` calibrated array silently became `int8 [-128,127]`, and a
`float [1.9,-1.9]` array silently became `int8 [1,-1]` -- both NumPy 2.5.1
confirmed defects (SuperSLM_IndependentReview_Evaluation-2026-07-21.md F13).
Coercion is not proof: this is SS11's reject-over-degrade law running
backwards, since the converter is the ONE place that still has the
calibration pipeline's real values before they are narrowed into the
runtime's fixed integer widths.

`validate_model(model)` is a two-phase gate run BEFORE `build_sections`
touches a single array: every check below either passes cleanly or raises
`ConverterValidationError` with a stable `.code` (so a red test can assert
WHICH check fired, not merely that conversion failed -- the S-HARDEN-2
review's own lesson: "a rejection test must assert WHICH rejection fired ...
or a shadowing check makes it a vacuous witness"). Nothing here repairs,
clamps, or rounds a value into range; every violation is a hard rejection.

`model` is any object exposing the same attribute contract `build_sections`
(convert_model.py) already reads: `.config` (hidden_size, num_hidden_layers,
num_attention_heads, num_key_value_heads, head_dim, intermediate_size,
vocab_size, context_cap, tie_word_embeddings), `.weights` (name -> array),
`.dynamic_biases` (site -> (q_b, codes)), `.rope_tables` (cos, sin),
`.weight_scales` (name -> per-channel float scales), `.scales.scale(key)`
(per-head float scale lookup), `.composition_constants` (name -> (m, e)),
`.kv_landing_scales` / `.kv_landing_reciprocals` (name -> tuple). A hand-built
`types.SimpleNamespace` satisfying this contract is exactly what the test
suite constructs -- validate_model never imports the cross-tree spike, so it
is testable on a bare checkout with no calibrated checkpoint (the same
constraint that keeps tools/gen_iexp_domain_fixtures.py in the CI
`generators` job while gen_intmath_fixtures.py/gen_matmul_fixtures.py, which
DO import the spike, are not: .github/workflows/tests.yml's own comment).
"""

import math
import unicodedata

import numpy as np

# Bounds mirrored from include/superslm/model.h / src/model.cpp -- never
# re-derived here, so a drift in either side is a single-source diff, not two
# independently-typed literals silently disagreeing.
INT8_MIN, INT8_MAX = -128, 127
INT32_MIN, INT32_MAX = -2147483648, 2147483647
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
WEIGHT_SCALE_SHIFT_MIN, WEIGHT_SCALE_SHIFT_MAX = 0, 31  # RoundingDivideByPOT's documented int32 exponent domain

# convert_model.py's own CFG1 Unicode pin -- the running interpreter's
# unicodedata must agree with it, or the pinned value is a description of a
# different Python than the one doing the conversion (S-HARDEN-3 gate text:
# "Unicode/version coherence").
PINNED_UNICODE_VERSION = (15, 1, 0)


class ConverterValidationError(ValueError):
    """Raised by every validate_model check. `.code` is a stable, testable
    diagnostic name -- distinct from `str(err)`, which also carries the
    offending values for a human reading the converter's console output."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(f"{code}: {message}")


def _reject(code, message):
    raise ConverterValidationError(code, message)


def _check_exact_range(name, values, lo, hi, code):
    """Reject-over-degrade for one array: every element must already be an
    exact integer within [lo, hi] BEFORE any narrowing cast runs. A
    non-integral element (e.g. a float calibration artifact that was never
    meant to be exactly representable) and an out-of-range element are
    distinct failure reasons, so they carry distinct codes -- collapsing them
    into one code is exactly the "shadowing check" the S-HARDEN-2 review
    lesson warns against.
    """
    a = np.asarray(values)
    if a.size == 0:
        return
    if not np.issubdtype(a.dtype, np.integer):
        frac = np.asarray(a, dtype=np.float64)
        if not np.all(frac == np.round(frac)):
            _reject(code + "NotIntegral", f"{name} contains a non-integral value (dtype {a.dtype})")
        if not np.all(np.isfinite(frac)):
            _reject(code + "NotFinite", f"{name} contains a non-finite value (dtype {a.dtype})")
    amin = int(a.min())
    amax = int(a.max())
    if amin < lo or amax > hi:
        _reject(code, f"{name} range [{amin},{amax}] outside the required [{lo},{hi}]")


def _check_shape(name, values, code_prefix):
    """Rank in [1,4] (MAX_TENSOR_RANK) and every declared dimension positive
    -- mirrors write_tensor_manifest's own `rank outside [1, MAX_TENSOR_RANK]`
    raise, but as a validate-phase rejection with a stable code instead of a
    bare ValueError from inside the serializer, and it additionally catches a
    zero-sized dimension (an empty tensor a rank check alone does not see).
    """
    a = np.asarray(values)
    if a.ndim < 1 or a.ndim > 4:
        _reject(code_prefix + "BadRank", f"{name} rank {a.ndim} outside [1,4]")
    if any(d <= 0 for d in a.shape):
        _reject(code_prefix + "ZeroDim", f"{name} shape {a.shape} has a non-positive dimension")


def check_config_geometry(cfg):
    """SuperSLM_Plan.md S17.3 cell 4, as amended by T-2432 Track A step 6 (GS-06,
    D-SLM5244) and T-2441 Minor 1 (D-SLM5446)/Minor 7 (D-SLM5440, fix D-SLM5452): heads % kv_heads == 0, kv_heads <= heads,
    head_dim != 0, hidden_size != 0. The relation between hidden_size and the
    heads/head_dim product that this docstring used to state as a still-enforced
    contract is REMOVED, not loosened, once q_width decouples Q/O's own width from
    hidden_size -- see this function's own GS-06 comment below for the removal itself.
    The zero-boundary cases are checked FIRST and independently -- the coverage audit's
    own specification (2026-07-21 S3.3): `heads % kv_heads` below would raise
    ZeroDivisionError if reached with kv_heads == 0, so both zero cases must be rejected
    with a named diagnostic before the modulus is ever evaluated, mirroring
    superslm::CheckConfigGeometry (include/superslm/proof_manifest.h) exactly.
    """
    heads = int(cfg.num_attention_heads)
    kv_heads = int(cfg.num_key_value_heads)
    head_dim = int(cfg.head_dim)
    hidden_size = int(cfg.hidden_size)

    if heads == 0:
        _reject("ZeroAttentionHeads", "num_attention_heads == 0")
    if kv_heads == 0:
        _reject("ZeroKeyValueHeads", "num_key_value_heads == 0")
    # T-2441 (Poirot 327ee29-t2438-ask5-tracka-review.md, Minor 7, D-SLM5440; fix D-SLM5452): R1's removal
    # (below) left these two zero cases unguarded on this direct-call surface -- executed and
    # confirmed before this fix: head_dim/hidden_size appeared nowhere in this module outside
    # the deleted check, so a zero-dimension checkpoint converted cleanly and was caught only
    # later, at C++ load. Mirrors superslm::CheckConfigGeometry's own identical fix
    # (src/proof_manifest.cpp), same zero-boundary-first ordering.
    if head_dim == 0:
        _reject("ZeroHeadDim", "head_dim == 0")
    if hidden_size == 0:
        _reject("ZeroHiddenSize", "hidden_size == 0")
    if kv_heads > heads:
        _reject("KvHeadsExceedsHeads", f"num_key_value_heads ({kv_heads}) > num_attention_heads ({heads})")
    if heads % kv_heads != 0:
        _reject("HeadsNotDivisibleByKv", f"num_attention_heads ({heads}) % num_key_value_heads ({kv_heads}) != 0")
    # SSLM-GEOMETRY-SITE: GS-06
    # T-2432 (Track A step 6, design §2.5 GS-06/§6 Track A step 6, D-SLM5244): the R1 identity
    # `hidden_size == num_attention_heads * head_dim` is no longer enforced -- q_width
    # (`num_attention_heads * head_dim`) is threaded independently of hidden_size once the
    # forward path decouples Q/O's own width. Removed rather than loosened, mirroring
    # CheckConfigGeometry's own C++ widening (src/proof_manifest.cpp). Confirmed correct by
    # execution against the real Qwen3-Embedding-0.6B candidate, T-2423's spike (D-SLM5290).
    # T-2441 (Minor 7, D-SLM5440; fix D-SLM5452): both names are read by the ZeroHeadDim/ZeroHiddenSize
    # checks above this point in the function, but neither is read again after it -- `del`
    # here means "not needed past this point," not "unused," now that the zero-boundary gap
    # is closed.
    del hidden_size, head_dim


def check_unicode_version_coherence(major, minor, patch, running_version=None):
    """The CFG1 unicode_major/minor/patch fields the converter writes must
    agree with the running interpreter's `unicodedata.unidata_version` --
    the same version the tokenizer converter's `UnicodeData` classmethod
    reads (tools/unicode_tables.py). `running_version` is injectable for
    testing without depending on the test runner's own Python build.
    """
    raw = running_version if running_version is not None else unicodedata.unidata_version
    parts = [int(p) for p in raw.split(".")]
    while len(parts) < 3:
        parts.append(0)
    actual = tuple(parts[:3])
    pin = (int(major), int(minor), int(patch))
    if actual != pin:
        _reject("UnicodeVersionMismatch",
                f"CFG1 pins Unicode {pin[0]}.{pin[1]}.{pin[2]} but the running interpreter's "
                f"unicodedata.unidata_version is {actual[0]}.{actual[1]}.{actual[2]}")


def check_scale_positivity(name, scale):
    """Every calibration scale folded into a WSC1 (mult, shift) pair must be
    finite and strictly positive before it is folded -- a zero or negative
    scale folds into a nonsensical multiplier/shift with no diagnostic
    anywhere downstream, and a non-finite one propagates NaN/inf through
    quantize_multiplier silently.
    """
    v = float(scale)
    if not math.isfinite(v):
        _reject("NonFiniteScale", f"{name} scale {v} is not finite")
    if v <= 0:
        _reject("NonPositiveScale", f"{name} scale {v} must be > 0")


def check_fold_bounds(name, identity, mult, shift):
    """The (identity, mult, shift) triple `_fold_ops_tensor`/`_ctx_fold_tensor`
    compute, checked against exactly the domain SslmModel::Load's own
    ValidateWeightScalesDomain (src/model.cpp) enforces at load time --
    catching a bad calibration BEFORE it is serialized rather than only after
    a round trip through the loader. `mult` is deliberately unchecked (any
    int32 is safe -- SaturatingRoundingDoublingHighMul saturates the sole
    overflow pair, per D-SLM142, mirrored from the runtime gate's own comment).
    """
    if int(identity) not in (0, 1):
        _reject("FoldIdentityNotBool", f"{name} identity {identity} not in {{0,1}}")
    if not (WEIGHT_SCALE_SHIFT_MIN <= int(shift) <= WEIGHT_SCALE_SHIFT_MAX):
        _reject("FoldShiftOutOfBounds",
                f"{name} shift {shift} outside [{WEIGHT_SCALE_SHIFT_MIN},{WEIGHT_SCALE_SHIFT_MAX}]")
    if not (INT32_MIN <= int(mult) <= INT32_MAX):
        _reject("FoldMultOutOfBounds", f"{name} mult {mult} outside int32 range")


def check_required_groups(model):
    """Required tensor and key sets: every top-level output group
    `build_sections` consumes must be present and non-empty -- an empty group
    is a calibration pipeline that silently produced nothing for a required
    section, which the old writer would happily serialize as a zero-tensor
    WGT1/BIA1/etc (itself then rejected downstream by SslmTensorManifest,
    but only after a wasted round trip and with a less specific diagnostic
    than this check gives). Also enforces the one cross-set relation the
    fold pipeline requires structurally: every WeightScales key must name an
    actual Weights tensor -- an orphaned fold entry is a converter bug, not a
    model fact.

    `dynamic_biases` is deliberately NOT in the always-required tuple below
    (CKN-03, T-2539). **T-2543 correction to the grounding.** The frozen
    design's own text (`t2408` Sec6 Track C step 4) argues `dynamic_biases`'s
    emptiness is "verified total by the same L5/L6 totality check that
    already runs upstream in `load_model`" -- read as a claim that the
    totality check VERIFIES bias completeness, that argument is circular
    under `_upstream_names`'s own per-projection gating: a gated bias entry
    is removed from BOTH sides of the totality comparison (the map's own
    demanded set and the checkpoint's present set) before that check ever
    runs, so the check has no way to observe a truncated bias population as
    anomalous -- it was never capable of catching that shape, symmetric
    bias sets or not. **The real, non-circular reason `dynamic_biases` needs
    no group-emptiness check:** its population is a deterministic function of
    `float_biases`'s own population, computed by a single, TOTAL, filter-free
    dict comprehension (`pipeline.py`'s `dynamic_biases = {name: ... for
    name, values in float_biases.items()}`) -- empty input produces empty
    output and non-empty input produces non-empty output, by construction,
    regardless of what any totality check does or does not verify elsewhere.
    A checkpoint biasing some layers or projections and not others (legitimate
    per CKN-02's own per-projection-independent gating, matching the runtime's
    own independent per-projection nullptr support) was never caught by this
    group-emptiness check even before CKN-03, and is not caught by any other
    check in this pipeline today -- that is an accepted architectural
    generality, not a hole CKN-03 opened.
    """
    for name in ("weights", "weight_scales", "composition_constants",
                 "kv_landing_scales", "kv_landing_reciprocals"):
        group = getattr(model, name)
        if not group:
            _reject("EmptyRequiredGroup", f"model.{name} is empty")

    cos, sin = model.rope_tables
    if len(cos) == 0 or len(sin) == 0:
        _reject("EmptyRequiredGroup", "model.rope_tables cos/sin is empty")

    orphaned = sorted(set(model.weight_scales) - set(model.weights))
    if orphaned:
        _reject("OrphanedWeightScaleKey", f"weight_scales names tensors absent from weights: {orphaned}")


def validate_model(model, *, fold_ops_tensor, ctx_fold_tensor, unicode_major=15, unicode_minor=1,
                    unicode_patch=0, running_unicode_version=None):
    """The full validate phase, run before any array is cast or serialized.
    `fold_ops_tensor`/`ctx_fold_tensor` are convert_model.py's own
    `_fold_ops_tensor`/`_ctx_fold_tensor` (passed in rather than imported, so
    this module carries no dependency on convert_model.py and stays testable
    standalone) -- validate_model calls them to check every fold's shift
    bound BEFORE build_sections calls them again to serialize, which is
    acceptable duplication: both are pure functions of the same calibration
    scales, not writer state, so calling them twice cannot itself diverge.

    Order matters for diagnostic precision, not correctness: required groups
    first (a check downstream of an empty group would raise a confusing
    secondary error), then geometry, then per-array dtype/range/shape, then
    scale positivity and fold bounds, then Unicode coherence last (cheapest
    and least likely to be the actual defect in a hostile calibration input).
    """
    check_required_groups(model)
    check_config_geometry(model.config)

    cfg = model.config

    # Weights: exact int8 range and integralness -- the F13 finding's own
    # reproduction (int16 [128,-129] silently wrapping to int8 [-128,127]).
    for name in sorted(model.weights):
        arr = model.weights[name]
        _check_exact_range(f"weights[{name!r}]", arr, INT8_MIN, INT8_MAX, "WeightsOutOfInt8Range")
        _check_shape(f"weights[{name!r}]", arr, "Weights")

    # Biases: the C28 dynamic-bias codes, int64.
    for site in sorted(model.dynamic_biases):
        _q_b, codes = model.dynamic_biases[site]
        _check_exact_range(f"dynamic_biases[{site!r}]", codes, INT64_MIN, INT64_MAX, "BiasesOutOfInt64Range")
        _check_shape(f"dynamic_biases[{site!r}]", codes, "Biases")

    # RopeTables: int64, and each element additionally must clear
    # RopeApplyPair's own load-time domain ([-2^30, 2^30], src/model.cpp) --
    # checked here too so a bad RoPE table is caught at conversion, not only
    # when SslmModel::Load rejects the artifact after it is already written.
    cos, sin = model.rope_tables
    rope_abs_max = 1 << 30
    for label, table in (("rope_tables.cos", cos), ("rope_tables.sin", sin)):
        _check_exact_range(label, table, INT64_MIN, INT64_MAX, "RopeTablesOutOfInt64Range")
        _check_exact_range(label, table, -rope_abs_max, rope_abs_max, "RopeTablesOutOfRuntimeDomain")
        _check_shape(label, table, "RopeTables")

    # WeightScales: the per-channel fold + the per-layer ctx-fold, computed
    # here exactly as build_sections computes them, so the shift/mult/
    # identity bound is checked against the same values that will be
    # serialized (never a re-derivation that could silently diverge).
    for name in sorted(model.weight_scales):
        channel_scales = model.weight_scales[name]
        for s in channel_scales:
            check_scale_positivity(f"weight_scales[{name!r}]", s)
        rows = fold_ops_tensor(channel_scales)
        for r, (identity, mult, shift) in enumerate(rows):
            check_fold_bounds(f"weight_scales[{name!r}] row {r}", identity, mult, shift)

    group = cfg.num_attention_heads // cfg.num_key_value_heads
    for layer in range(cfg.num_hidden_layers):
        for h in range(cfg.num_key_value_heads):
            check_scale_positivity(f"layer{layer}.v_head{h}.scale",
                                   model.scales.scale(f"layer{layer}.v_head{h}.scale"))
        rows = ctx_fold_tensor(model, layer)
        for h, (identity, mult, shift) in enumerate(rows):
            check_fold_bounds(f"layer{layer}.ctx_fold head {h}", identity, mult, shift)
    del group  # computed by ctx_fold_tensor itself; kept here only for the loop bound above

    # CompositionConstants (m, e): the exact no-UB floor SslmModel::Load's
    # ValidateCompositionConstantsDomain enforces (src/model.cpp), checked
    # here at conversion time for the same reason as the RoPE domain above.
    m_abs_max = (1 << 31) - 1
    e_min, e_max = -80, 7
    for name, (m, e) in model.composition_constants.items():
        if not (-m_abs_max <= int(m) <= m_abs_max):
            _reject("CompositionScaleMOutOfDomain", f"composition_constants[{name!r}] m={m} outside "
                    f"[-{m_abs_max},{m_abs_max}]")
        if not (e_min <= int(e) <= e_max):
            _reject("CompositionScaleEOutOfDomain", f"composition_constants[{name!r}] e={e} outside "
                    f"[{e_min},{e_max}]")

    # D-SLM7036: the bit-2 capability is content-derived from paired QK gain
    # scales. Its source-scale roles need the stronger canonical-positive
    # domain before build_sections can emit QKC1 or marshal can derive 127*K.
    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        q_key, k_key = f"{prefix}.q_norm.gain", f"{prefix}.k_norm.gain"
        q_present, k_present = q_key in model.weight_scales, k_key in model.weight_scales
        if q_present != k_present:
            _reject("QkChannelTableGeometryMismatch",
                    f"{prefix} has asymmetric q_norm.gain/k_norm.gain weight-scale presence")
        if not q_present:
            continue
        for role in ("q_norm", "k_norm"):
            key = f"{prefix}.{role}"
            m, _e = model.composition_constants.get(key, (0, 0))
            if not ((1 << 30) <= int(m) <= (1 << 31) - 1):
                _reject("CompositionScaleOutOfDomain",
                        f'CompositionConstants entry "{key}" m={m} outside the QK '
                        "canonical-positive domain [1073741824,2147483647]")

    check_unicode_version_coherence(unicode_major, unicode_minor, unicode_patch, running_unicode_version)
