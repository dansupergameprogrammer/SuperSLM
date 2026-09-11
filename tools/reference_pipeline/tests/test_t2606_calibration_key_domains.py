"""T-2606 -- every maxima key the calibrator feeds `_derive_scales` is observed on ONE
tensor domain: the one its consumer lands onto.

T-2604's localization (Claude/Linnaeus/t2604-qwen3-block0-localization-2026-09-11.md
§1/§4, D-SLM6652, conductor-verified) found the instance: `_float_layer` observes the
raw projected K under `f"{prefix}.k"`, then applies QK-norm and RoPE, then observes the
TRANSFORMED K under that same raw key again (pipeline.py:3196 at faef137), so
`_projection_scale` sizes the raw-K landing grid (`layer{L}.k_proj.requant` /
`k_head{h}.scale`, which `LandTokenKVRow` lands the RAW k_proj accumulator onto BEFORE
`ApplyQkNormSite`'s K branch) from the post-norm peak. On the real Qwen3-Embedding-0.6B
artifact that grid is ~89x too coarse and 99.4% of K codes land at zero. The adjacent
comment (pipeline.py:3181-3184) says the raw key is "untouched"; the executable
statement at :3196 contradicts it. The line predates QK-norm support (2026-08-16), when
the second observation differed from raw K only by RoPE (bounded by sqrt(2)) and was
harmless.

This file pins the instance RED (it fails at faef137 for exactly this reason, naming
both peaks) and the CLASS it belongs to: one domain per consumed key, enumerated from
the code at run time (`_derive_scales`'s own maxima reads), with `f"{prefix}.q` -- the
same shape at :3195 -- dispositioned as a second instance, and `k_normed`'s deliberate
pre/post-RoPE union (D-SLM6263/6264) kept GREEN as one domain by design.

The consumer domains are derived from `_derive_scales` (pipeline.py:2112-2332) and the
C++ landings its site constants feed: K's raw accumulator lands through LandTokenKVRow
(src/forward/forward_sites.cpp:1843-1851) before ApplyQkNormSite's K branch
(:1486-1509); Q's raw accumulator funnels through ProjectAndFunnel
(forward_sites.cpp:1816-1822) before ApplyQkNormSite's Q branch normalizes from those
raw codes (q_norm is per-head RMS -- insensitive to the landing scale in the mean, but
a coarser grid degrades the resolution of exactly the codes it normalizes, the same
mechanism as the K defect); the QK-normed K relands through ApplyQkNormSite's second
LandingRescale onto `k_normed_head{h}.scale` and `RopeApplySite` rotates the landed
codes afterward -- which is why that key's domain spans BOTH sides of RoPE.

Red at faef137, by cell (the builder greens them in T-2607; the mutation cells below
are green now and prove the red cells discriminate):
- the raw-K instance cell, the raw-Q second-instance cell, the class cell, and the
  capture cell are RED at faef137;
- the enumeration cell is green at faef137 (it is the guard: it fails when a consumed
  key carries no domain assertion, or when the committed domain list rots).
"""

import importlib.util
import sys

import numpy as np

import conftest
from conftest import require

MODULE = "reference_pipeline.pipeline"

# Measured (this session's probe, the checked-in §11 fixture): at gain scale 1.0 the
# fixture is TAME for this defect -- raw K peaks (layer0 2.9996, layer1 6.1693) dominate
# the post-QK-norm pre-RoPE peaks (1.8092, 2.3317), so the polluted running max already
# equals the raw peak and the instance cannot show. Scaling every layer's q_norm/k_norm
# gains by 20 makes the post-norm peaks ~10x the raw peaks (executed: layer1 raw K
# 4.8856 vs maxima 46.6968; layer1 raw Q 3.2591 vs maxima 32.8548), which is the
# Qwen3-Embedding-0.6B's own shape in miniature (raw K ~5.4 vs post-k_norm ~432,
# D-SLM6652). The checked-in fixture and corpus are untouched; the scale lives in this
# file's own fixture copy only.
_QKNORM_GAIN_SCALE = 20.0


def _fixture_config(pipeline):
    """The §11 fixture model's shape (test_ask5_trackb_oracle_qk_norm_parity.py's own
    `_fixture_config`, inlined per this suite's no-cross-module-imports convention) --
    carries non-uniform q_norm/k_norm gains unconditionally (_weight_shapes, T-2539)."""
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def _scaled_qk_norm_fixture(pipeline):
    """The §11 fixture with every layer's q_norm.gain/k_norm.gain scaled by
    _QKNORM_GAIN_SCALE -- a QK-norm fixture where the post-norm Q/K are much larger than
    the raw projections, the condition T-2606's instance cells need to show the defect
    (the checked-in gains are too tame; see _QKNORM_GAIN_SCALE). The norm gains are
    pinned int8 codes at the unit scale, so scaling the floats and the weight scales by
    the same factor keeps the fixture self-consistent."""
    cfg = _fixture_config(pipeline)
    _, weight_scales, floats = pipeline._pinned_weights(cfg)
    floats = dict(floats)
    weight_scales = dict(weight_scales)
    for layer in range(cfg.num_hidden_layers):
        for norm in ("q_norm", "k_norm"):
            name = f"layer{layer}.{norm}.gain"
            floats[name] = np.asarray(floats[name], dtype=np.float64) * _QKNORM_GAIN_SCALE
            weight_scales[name] = tuple(
                scale * _QKNORM_GAIN_SCALE for scale in weight_scales[name])
    return cfg, weight_scales, pipeline._dict_float_source(floats)


# ==============================================================================
# The committed domain list: one row per maxima key `_derive_scales` consumes.
# ==============================================================================
#
# Each row: (allowed observation phases, the consumer that lands the key, the tensor
# domain that consumer expects). The phases are the calibration walks' own crossing
# structure, tracked by this file's instrumentation: "pre_qk_norm" (before
# `_apply_qk_norm` -- the raw projection outputs), "post_qk_norm" (after QK-norm,
# before RoPE), "post_rope" (after `_float_rope`), "post_layer" (after `_float_layer`
# returns -- the final-norm region of `_float_forward_many`). A key observed at a
# phase its domain does not span is an observation on a foreign tensor folded into the
# key -- the class T-2604 localized, of which pipeline.py:3196 is the K instance.
#
# k_normed is ONE domain by design (D-SLM6263/6264): the engine lands the QK-normed K
# once and `RopeApplySite` rotates the landed codes in place afterward, so that key's
# landing scale must enclose the QK-normed K's pre- AND post-RoPE component peaks. Both
# observations are on that one tensor; the cells keep them green and this entry says so.
#
# `q` and `k` are the keys the defect lives on: their consumers land the RAW projection
# outputs, so every observation of them must be pre-QK-norm. `_kv_calibration_capture`'s
# own maxima observes `q` (and `attn_norm.out`) too -- the sibling walk Arm D/E's
# `_layer_q_scale` derives Q's production landing scale from -- so its observations are
# graded by the same rows.

_DOMAIN_TABLE = {
    "attn_norm.out": (
        {"pre_qk_norm"},
        "layer{L}.attn_norm.requant, landed by RmsNormSite (forward_sites.cpp) at the "
        "head of each layer",
        "the attn-normed residual stream",
    ),
    "q": (
        {"pre_qk_norm"},
        "layer{L}.q_proj.requant, funneled by ProjectAndFunnel (forward_sites.cpp) onto "
        "the raw q_proj accumulator BEFORE ApplyQkNormSite's Q branch normalizes from "
        "those raw codes",
        "the raw q_proj output",
    ),
    "k": (
        {"pre_qk_norm"},
        "layer{L}.k_proj.requant / layer{L}.k_head{h}.scale, landed by LandTokenKVRow "
        "(forward_sites.cpp) onto the raw k_proj accumulator BEFORE ApplyQkNormSite's K "
        "branch relands the post-norm codes",
        "the raw k_proj output",
    ),
    "v": (
        {"pre_qk_norm"},
        "layer{L}.v_proj.requant / layer{L}.v_head{h}.scale, landed by LandTokenKVRow",
        "the raw v_proj output",
    ),
    "k_normed": (
        {"post_qk_norm", "post_rope"},
        "layer{L}.k_normed_head{h}.scale, landed by ApplyQkNormSite's second "
        "LandingRescale and clamped by RopeApplySite after it rotates the landed codes",
        "the QK-normed K, pre- and post-RoPE union -- ONE domain by design "
        "(D-SLM6263/6264)",
    ),
    "attn_ctx": (
        {"post_rope"},
        "layer{L}.attn_ctx.requant",
        "the attention context",
    ),
    "attn_out": (
        {"post_rope"},
        "layer{L}.o_proj.requant (the site `_derive_scales` derives from this key)",
        "the raw o_proj output",
    ),
    "attn_residual": (
        {"post_rope"},
        "layer{L}.attn_residual requant",
        "the post-attention-add residual stream",
    ),
    "mlp_norm.out": (
        {"post_rope"},
        "layer{L}.mlp_norm.requant",
        "the mlp-normed residual stream",
    ),
    "gate": (
        {"post_rope"},
        "layer{L}.gate_proj.requant",
        "the raw gate_proj output",
    ),
    "up": (
        {"post_rope"},
        "layer{L}.up_proj.requant",
        "the raw up_proj output",
    ),
    "mlp_act": (
        {"post_rope"},
        "layer{L}.mlp_act.requant",
        "the SiLU(gate)*up activation",
    ),
    "mlp_out": (
        {"post_rope"},
        "layer{L}.down_proj.requant (the site `_derive_scales` derives from this key)",
        "the raw down_proj output",
    ),
    "mlp_residual": (
        {"post_rope"},
        "layer{L}.mlp_residual requant",
        "the post-MLP-add residual stream",
    ),
    "final_norm.out": (
        {"post_layer"},
        "final_norm.requant",
        "the final-normed hidden",
    ),
}


def _suffix_of(key):
    """`layer0.k` -> `k`; `final_norm.out` has no layer prefix and stays whole."""
    if key.startswith("layer") and "." in key:
        return key.split(".", 1)[1]
    return key


# ==============================================================================
# The instrument: one shared recorder for every cell (red and mutant alike).
# ==============================================================================


class _Walk:
    """One instrumented calibration walk's records. Both the red cells (run against the
    real module at the tree under test) and the mutant cells (run against a scratch
    deleted-observation module) grade the SAME records through the SAME predicates, so
    a mutant cell's green and its red twin's failure differ only in the module under
    test -- never in the check (`test_ask5_trackb_oracle_qk_norm_parity.py`'s own
    `_assert_apply_qk_norm_called` convention).

    `observed`: {(caller, maxima key): {pipeline phase: peak}} -- every `_observe` call
    that carried a non-None maxima dict, with the phase at which it was made. `phase`:
    the walk's current crossing state. `project_in`/`project_out`: per `_float_project`
    weight name, the max |input|/|output| over the walk -- the INDEPENDENT ground truth
    for the raw projection domains (recorded at the projection itself, before any
    reshape or observe, so it cannot inherit the observation stream's keying).
    `norm_in`/`norm_out`: per (layer prefix, "q"/"k"), the max |incoming|/|outgoing| of
    `_apply_qk_norm` -- the raw projection peak (cross-checked against
    `project_out`) and the post-QK-norm pre-RoPE peak.
    """

    def __init__(self):
        self.observed = {}
        self.project_in = {}
        self.project_out = {}
        self.norm_in = {}
        self.norm_out = {}
        self.phase = {"now": "pre_qk_norm"}


def _instrumented(pipeline, walk, run):
    """Run `run()` with T-2606's instrumentation installed on `pipeline`, restoring every
    patched attribute in `finally` (the shared module is left exactly as found).

    The phase protocol: `_float_layer`'s entry opens a layer (pre_qk_norm); its
    `_apply_qk_norm` call crosses QK-norm (post_qk_norm); its `_float_rope` calls cross
    RoPE (post_rope); its return closes the layer (post_layer). `_kv_calibration_capture`
    opens each of its (layer, sequence) iterations at the `_float_rmsnorm` call its own
    body makes (caller `_kv_calibration_capture` -> pre_qk_norm), so its observations are
    phased by the same crossings -- the two walks' observations are graded by one table.
    """
    orig = {name: getattr(pipeline, name) for name in (
        "_observe", "_apply_qk_norm", "_float_rope", "_float_project", "_float_rmsnorm",
        "_float_layer")}

    def observe(maxima, name, values):
        if maxima is None:
            return orig["_observe"](maxima, name, values)
        peak = float(np.abs(np.asarray(values, dtype=np.float64)).max(initial=0.0))
        caller = sys._getframe(1).f_code.co_name
        phase = walk.phase["now"]
        key = (caller, name)
        walk.observed.setdefault(key, {})
        if peak > walk.observed[key].get(phase, 0.0):
            walk.observed[key][phase] = peak
        return orig["_observe"](maxima, name, values)

    def project(float_weight, name, values):
        out = orig["_float_project"](float_weight, name, values)
        for book, arr in ((walk.project_in, values), (walk.project_out, out)):
            peak = float(np.abs(np.asarray(arr, dtype=np.float64)).max(initial=0.0))
            if peak > book.get(name, 0.0):
                book[name] = peak
        return out

    def apply_qk_norm(q, k, tensors, prefix, cfg):
        for tag, arr in (("q", q), ("k", k)):
            peak = float(np.abs(np.asarray(arr, dtype=np.float64)).max(initial=0.0))
            if peak > walk.norm_in.get((prefix, tag), 0.0):
                walk.norm_in[(prefix, tag)] = peak
        q2, k2 = orig["_apply_qk_norm"](q, k, tensors, prefix, cfg)
        for tag, arr in (("q", q2), ("k", k2)):
            peak = float(np.abs(np.asarray(arr, dtype=np.float64)).max(initial=0.0))
            if peak > walk.norm_out.get((prefix, tag), 0.0):
                walk.norm_out[(prefix, tag)] = peak
        walk.phase["now"] = "post_qk_norm"
        return q2, k2

    def rope(vectors, theta):
        out = orig["_float_rope"](vectors, theta)
        walk.phase["now"] = "post_rope"
        return out

    def rmsnorm(x, eps):
        if sys._getframe(1).f_code.co_name == "_kv_calibration_capture":
            walk.phase["now"] = "pre_qk_norm"
        return orig["_float_rmsnorm"](x, eps)

    def float_layer(cfg, tensors, hidden, maxima, prefix):
        walk.phase["now"] = "pre_qk_norm"
        out = orig["_float_layer"](cfg, tensors, hidden, maxima, prefix)
        walk.phase["now"] = "post_layer"
        return out

    pipeline._observe = observe
    pipeline._apply_qk_norm = apply_qk_norm
    pipeline._float_rope = rope
    pipeline._float_project = project
    pipeline._float_rmsnorm = rmsnorm
    pipeline._float_layer = float_layer
    try:
        return run()
    finally:
        for name, fn in orig.items():
            setattr(pipeline, name, fn)


def _run_calibrate(pipeline, cfg, float_weight):
    """One instrumented `_calibrate` run over the checked-in calibration corpus."""
    records = pipeline.calibration_records()
    tokenize = pipeline._fixture_tokenize_prompt(cfg)
    record_tokenize = pipeline._bridge_record_tokenizer(tokenize)
    walk = _Walk()

    def run():
        return pipeline._calibrate(cfg, float_weight, records, record_tokenize)

    return _instrumented(pipeline, walk, run), walk


def _run_capture(pipeline, cfg, float_weight):
    """One instrumented `_kv_calibration_capture` run -- the sibling walk whose own
    maxima Arm D/E's `_layer_q_scale` derives Q's production landing scale from."""
    records = pipeline.calibration_records()
    tokenize = pipeline._fixture_tokenize_prompt(cfg)
    record_tokenize = pipeline._bridge_record_tokenizer(tokenize)
    walk = _Walk()

    def run():
        return pipeline._kv_calibration_capture(cfg, float_weight, records, record_tokenize)[1]

    return _instrumented(pipeline, walk, run), walk


# ==============================================================================
# The predicates -- one implementation each, shared by the red cells and the mutant
# cells, so "caught" cannot differ between them.
# ==============================================================================


def _grade_walk(walk, maxima, walk_name, demand_full_table):
    """Grade one instrumented walk against `_DOMAIN_TABLE`. Three checks per key:

    1. every observation's pipeline phase is one the key's domain spans (magnitude-
       independent: a foreign-domain observation violates this even when it does not
       move the running max);
    2. the key's running max equals the peak of its on-domain observations alone --
       the value pin, which catches a foreign observation that DID move the max (the
       damage direction: the landing grid is sized from the wrong peak);
    3. a key recorded in maxima with NO on-domain observation is reported -- a
       calibration blind on its own consumed key.

    With `demand_full_table`, every tabled key must be observed somewhere on the walk
    (the calibrated walk feeds `_derive_scales` every key it consumes; the capture walk
    observes only `q` and `attn_norm.out` by design, so it does not).
    """
    violations = []
    for (caller, key), phase_peaks in sorted(walk.observed.items()):
        entry = _DOMAIN_TABLE.get(_suffix_of(key))
        if entry is None:
            violations.append(
                f"{walk_name}: {key} (observed from {caller}) has no entry in the "
                f"committed domain table -- a calibration key was observed that no "
                f"domain assertion covers")
            continue
        allowed, consumer, domain = entry
        for phase, peak in sorted(phase_peaks.items()):
            if phase not in allowed:
                violations.append(
                    f"{walk_name}: maxima[{key!r}] was observed at pipeline phase "
                    f"{phase!r} (peak {peak!r}, from {caller}), but its consumer -- "
                    f"{consumer} -- lands {domain}, which this walk observes at "
                    f"phase(s) {sorted(allowed)}; an observation at {phase!r} folds a "
                    f"foreign tensor domain into the key (the class T-2604 localized "
                    f"at pipeline.py:3196, D-SLM6652)")
    for key, value in sorted(maxima.items()):
        entry = _DOMAIN_TABLE.get(_suffix_of(key))
        if entry is None:
            continue
        allowed = entry[0]
        on_domain = [
            peak for (_, observed_key), phase_peaks in walk.observed.items()
            if observed_key == key
            for phase, peak in phase_peaks.items() if phase in allowed]
        if not on_domain:
            violations.append(
                f"{walk_name}: maxima[{key!r}] holds {value!r} but the key was never "
                f"observed on its consumer's domain phase(s) {sorted(allowed)} -- the "
                f"calibration is blind on this consumed key")
            continue
        domain_peak = max(on_domain)
        if value != domain_peak:
            violations.append(
                f"{walk_name}: maxima[{key!r}] holds {value!r}, not the peak "
                f"{domain_peak!r} of its own domain -- a foreign-domain observation "
                f"moved the key's running max, so the landing scale is sized from the "
                f"wrong tensor (the damage T-2604 measured: the K grid ~89x too "
                f"coarse, 99.4% of K codes at zero)")
    if demand_full_table:
        observed_suffixes = {_suffix_of(key) for (_, key) in walk.observed}
        for suffix in sorted(set(_DOMAIN_TABLE) - observed_suffixes):
            violations.append(
                f"{walk_name}: the domain-table key {suffix!r} was never observed on "
                f"this walk -- the calibration is blind on a consumed key")
    return violations


def _assert_raw_projection_key_pure(cfg, maxima, walk, which):
    """The instance predicate: `maxima[layer{L}.{which}]` equals the peak of the raw
    {which}_proj output EXACTLY, at every layer. Ground truth is `project_out` -- the
    `_float_project` call's own output, recorded at the projection before any reshape
    or observe, so it cannot inherit the observation stream's keying; `norm_in`
    cross-checks it (both are the raw projection output, measured at two different
    seams). The message names the raw peak, the observed (polluted) value, and the
    post-QK-norm pre-RoPE peak."""
    consumer = _DOMAIN_TABLE[which][1]
    domain = _DOMAIN_TABLE[which][2]
    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        key = f"{prefix}.{which}"
        raw_peak = walk.project_out[f"{prefix}.{which}_proj"]
        cross_check = walk.norm_in[(prefix, which)]
        observed = maxima[key]
        post_norm = walk.norm_out[(prefix, which)]
        # Materiality (the parity file's own discipline): the fixture must actually
        # separate the two domains, or this cell could pass vacuously on a tame fixture.
        assert post_norm > raw_peak, (
            f"{key}: the fixture no longer separates the domains (post-QK-norm "
            f"pre-RoPE peak {post_norm!r} <= raw {which}_proj peak {raw_peak!r}) -- "
            f"scale the q_norm/k_norm gains up (see _QKNORM_GAIN_SCALE) or this cell "
            f"cannot show the defect it exists to pin"
        )
        assert observed == raw_peak, (
            f"maxima[{key!r}] = {observed!r}, not the raw {which}_proj output peak "
            f"{raw_peak!r} (cross-checked at the QK-norm boundary: {cross_check!r}; the "
            f"post-QK-norm pre-RoPE peak is {post_norm!r}): the raw-{which} calibration "
            f"key observed a TRANSFORMED {which.upper()} under the raw key "
            f"(pipeline.py `_float_layer`'s post-QK-norm/post-RoPE "
            f"_observe(maxima, f\"{{prefix}}.{which}\", ...) call), so the consumer "
            f"-- {consumer} -- lands {domain} onto a grid sized from the wrong tensor "
            f"domain. T-2604 localized this on the real Qwen3-Embedding-0.6B artifact "
            f"as an ~89x too-coarse K grid with 99.4% of K codes at zero (D-SLM6652); "
            f"the fix (T-2607) removes the foreign-domain observation"
        )


# ==============================================================================
# The cells.
# ==============================================================================


def test_raw_k_calibration_key_equals_the_raw_k_proj_peak_exactly():
    """T-2606's instance cell, RED at faef137: on a QK-norm fixture where post-norm K is
    much larger than raw K, `maxima[layer{L}.k]` must equal the peak of the raw k_proj
    output exactly -- the key `layer{L}.k_proj.requant` / `k_head{h}.scale` consumes,
    which `LandTokenKVRow` lands the RAW k_proj accumulator onto before
    `ApplyQkNormSite`'s K branch.

    Measured pre-fix on this fixture (gain scale 20, this session's probe):
    layer0 maxima=32.91414608964846 vs raw peak 2.9995579663144536 (post-norm pre-RoPE
    36.183115709728476); layer1 maxima=46.696818416866044 vs raw peak 4.88558142908726
    (post-norm pre-RoPE 46.63390271322812)."""
    pipeline = require(MODULE)
    cfg, _, float_weight = _scaled_qk_norm_fixture(pipeline)
    maxima, walk = _run_calibrate(pipeline, cfg, float_weight)
    _assert_raw_projection_key_pure(cfg, maxima, walk, "k")
    for layer in range(cfg.num_hidden_layers):
        print(f"maxima[layer{layer}.k]={maxima[f'layer{layer}.k']!r} "
              f"raw={walk.project_out[f'layer{layer}.k_proj']!r}")


def test_raw_q_calibration_key_equals_the_raw_q_proj_peak_exactly():
    """T-2606's second-instance cell, RED at faef137: `f"{prefix}.q"` at pipeline.py:3195
    has the same shape as :3196 -- `_float_layer` observes the post-QK-norm, post-RoPE Q
    under the raw-q key, and `_kv_calibration_capture` mirrors it at :2676. Its consumer
    is `layer{L}.q_proj.requant`, which `ProjectAndFunnel` (forward_sites.cpp) funnels
    the RAW q_proj accumulator onto before `ApplyQkNormSite`'s Q branch normalizes from
    those raw codes; `softmax.input` reads the weight-derived post-norm gain scale, NOT
    this key, and nothing requantizes post-norm Q back onto the q_proj funnel scale --
    so the key's domain is the raw q_proj output and this is a second instance of the
    same defect, in both walks. Measured pre-fix on this fixture: layer0
    maxima=41.49181527237871 vs raw peak 4.046075037689526; layer1 maxima=32.854844166511825
    vs raw peak 3.259073149112257 (milder than K's ~10x on this fixture, and milder on the
    real artifact -- q_proj held cosine 0.9994 where K fell to 0.48 -- but the same
    wrong-domain mechanism, and the same fix locus)."""
    pipeline = require(MODULE)
    cfg, _, float_weight = _scaled_qk_norm_fixture(pipeline)
    maxima, walk = _run_calibrate(pipeline, cfg, float_weight)
    _assert_raw_projection_key_pure(cfg, maxima, walk, "q")
    for layer in range(cfg.num_hidden_layers):
        print(f"maxima[layer{layer}.q]={maxima[f'layer{layer}.q']!r} "
              f"raw={walk.project_out[f'layer{layer}.q_proj']!r}")


def test_every_consumed_maxima_key_is_observed_on_its_consumers_landing_domain():
    """T-2606's class cell, RED at faef137: on the calibrated walk, every key the domain
    table covers is observed only at the pipeline phase(s) its consumer's tensor domain
    spans, its running max equals that domain's peak alone, and every tabled key is
    observed somewhere. At faef137 the violations are `layer{L}.q` and `layer{L}.k`
    observed at post_rope (the post-QK-norm, rotated tensors) in addition to pre_qk_norm
    -- the instance and the second instance. `k_normed`'s two observations (post_qk_norm
    AND post_rope) are both on the QK-normed K, one domain by design (D-SLM6263/6264),
    and stay green under this grading."""
    pipeline = require(MODULE)
    cfg, _, float_weight = _scaled_qk_norm_fixture(pipeline)
    maxima, walk = _run_calibrate(pipeline, cfg, float_weight)
    violations = _grade_walk(walk, maxima, "the calibrated walk", demand_full_table=True)
    assert not violations, "\n".join(violations)


def test_the_kv_calibration_capture_observes_q_on_its_consumers_landing_domain():
    """T-2606's capture cell, RED at faef137: `_kv_calibration_capture`'s own maxima --
    the dict `_layer_q_scale` derives Q's production landing scale from
    (pipeline.py:2707-2728, whose docstring at :2719-2721 states the union as if it were
    the contract) -- observes `q` at :2670 (raw, pre-norm, correct domain) AND at :2676
    (post-QK-norm, post-RoPE, foreign). The same domain row grades it: the capture's
    maxima[layer{L}.q] must equal its raw q_proj peak alone."""
    pipeline = require(MODULE)
    cfg, _, float_weight = _scaled_qk_norm_fixture(pipeline)
    maxima, walk = _run_capture(pipeline, cfg, float_weight)
    violations = _grade_walk(walk, maxima, "the _kv_calibration_capture walk",
                             demand_full_table=False)
    assert not violations, "\n".join(violations)
    for layer in range(cfg.num_hidden_layers):
        print(f"capture maxima[layer{layer}.q]={maxima[f'layer{layer}.q']!r} "
              f"raw={walk.project_out[f'layer{layer}.q_proj']!r}")


class _TrackingMaxima(dict):
    """A maxima dict that records every key read out of it -- the run-time enumeration
    of `_derive_scales`'s consumed-key set, so the committed domain list is checked
    against the code's own consumption rather than trusted (`_output_scale`'s
    `maxima.get(name, 0.0)` is the one read `_derive_scales` makes per key; `__getitem__`
    is recorded too so a future read pattern cannot fall outside the enumeration)."""

    def __init__(self, base):
        super().__init__(base)
        self.consumed = set()

    def get(self, name, default=None):
        self.consumed.add(name)
        return super().get(name, default)

    def __getitem__(self, name):
        self.consumed.add(name)
        return super().__getitem__(name)


def test_every_derive_scales_consumed_key_carries_a_domain_assertion():
    """T-2606's enumeration cell (green at faef137 -- it is the guard): the keys
    `_derive_scales` consumes are enumerated FROM THE CODE AT RUN TIME, and every one of
    them must carry an entry in the committed domain table -- and the table must carry
    nothing the code no longer consumes, so the list cannot rot in either direction. A
    consumed key with no domain assertion fails here by name; so does a tabled key that
    stopped being consumed. `k_normed` drops out of both sets together on a checkpoint
    without QK-norm (its consumption is presence-gated at pipeline.py:2253-2256), which
    is why the fixture used here carries QK-norm."""
    pipeline = require(MODULE)
    cfg, weight_scales, float_weight = _scaled_qk_norm_fixture(pipeline)
    maxima, _ = _run_calibrate(pipeline, cfg, float_weight)
    tracking = _TrackingMaxima(maxima)
    pipeline._derive_scales(cfg, tracking, weight_scales, {})
    consumed = {_suffix_of(key) for key in tracking.consumed}
    tabled = set(_DOMAIN_TABLE)
    missing = consumed - tabled
    assert not missing, (
        f"_derive_scales consumed maxima keys with no domain assertion: {sorted(missing)} "
        f"-- every consumed key must carry a row in _DOMAIN_TABLE (T-2606's class pin)"
    )
    stale = tabled - consumed
    assert not stale, (
        f"the committed domain table carries keys _derive_scales no longer consumes: "
        f"{sorted(stale)} -- the table has rotted; reconcile it with the code"
    )
    print(f"consumed maxima keys ({len(tracking.consumed)}): {sorted(tracking.consumed)}")


# ==============================================================================
# Mutation proofs (green at faef137): each loads a scratch deleted-observation mutant
# -- a copy of pipeline.py transformed in tmp_path, never committed, never touching the
# imported module -- and shows the SAME shared predicate that is red on the real tree
# is green on the corrected behavior. Modeled on
# test_k_normed_union_maxima_is_lost_under_the_post_rope_observation_deletion_mutant
# (test_ask5_trackb_oracle_qk_norm_parity.py:562).
# ==============================================================================


def _load_mutant(transform, tmp_path):
    """Load a MUTATED copy of pipeline.py's own real source as an isolated module in
    tmp_path (inlined from test_ask5_trackb_oracle_qk_norm_parity.py's
    `_load_mutant_pipeline_module` per this suite's no-cross-module-imports convention)."""
    pipeline = require(MODULE)
    real_path = conftest.TOOLS_DIR / "reference_pipeline" / "pipeline.py"
    real_source = real_path.read_text(encoding="utf-8")
    mutated = transform(real_source)
    assert mutated != real_source, "sanity: the transform must actually change the source"
    mutant_path = tmp_path / "pipeline_mutant.py"
    mutant_path.write_text(mutated, encoding="utf-8")
    module_name = f"_t2606_pipeline_mutant_{id(mutant_path)}"
    spec = importlib.util.spec_from_file_location(module_name, mutant_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    module.CALIBRATION_CORPUS_PATH = pipeline.CALIBRATION_CORPUS_PATH
    return module


# The first line of the comment block that immediately follows pipeline.py's two
# post-transform raw-key observations (:3195-:3196 at faef137) -- unique text in the
# file, so anchoring on it targets exactly that block and cannot match the raw
# projection observations at :3156-:3158 (which are followed by the v observation).
_POST_TRANSFORM_BLOCK_COMMENT = (
    "    # (D-SLM6263): the k_normed key's SECOND observation, post-RoPE -- the running max above\n"
)


def _delete_raw_k_post_transform_observation(text):
    """Delete `_float_layer`'s post-transform raw-K observation (pipeline.py:3196 at
    faef137) -- the exact line whose presence the instance cell pins red."""
    anchor = (
        '    _observe(maxima, f"{prefix}.k", k)\n'
        + _POST_TRANSFORM_BLOCK_COMMENT
    )
    assert text.count(anchor) == 1, (
        "sanity: the post-transform raw-K observation block must match verbatim, once"
    )
    mutated = text.replace(
        anchor,
        "    # T-2606 mutant: the post-transform raw-K observation deleted\n"
        + _POST_TRANSFORM_BLOCK_COMMENT,
        1,
    )
    # The leading "\n" pins the 4-space indentation of `_float_layer`'s own raw
    # observations: the capture's 12-space-indented raw-q observation (pipeline.py:2670)
    # contains the un-anchored form as a substring, so an anchor without it cannot
    # count _float_layer's observations alone.
    assert mutated.count('\n    _observe(maxima, f"{prefix}.k", k)\n') == 1, (
        "sanity: exactly the raw projection's own k observation must survive"
    )
    return mutated


def _delete_both_raw_key_post_transform_observations(text):
    """Delete both post-transform raw-key observations (pipeline.py:3195 and :3196 at
    faef137) -- the full corrected behavior for the maxima keys on the calibrated walk."""
    anchor = (
        '    _observe(maxima, f"{prefix}.q", q)\n'
        '    _observe(maxima, f"{prefix}.k", k)\n'
        + _POST_TRANSFORM_BLOCK_COMMENT
    )
    assert text.count(anchor) == 1, (
        "sanity: the post-transform raw-key observation block must match verbatim, once"
    )
    mutated = text.replace(
        anchor,
        "    # T-2606 mutant: the post-transform raw-Q and raw-K observations deleted\n"
        + _POST_TRANSFORM_BLOCK_COMMENT,
        1,
    )
    assert mutated.count('\n    _observe(maxima, f"{prefix}.k", k)\n') == 1, (
        "sanity: exactly the raw projection's own k observation must survive"
    )
    assert mutated.count('\n    _observe(maxima, f"{prefix}.q", q)\n') == 1, (
        "sanity: exactly the raw projection's own q observation must survive in "
        "_float_layer (the capture's own 12-space-indented raw-q observation is "
        "untouched by this mutant; the leading newline pins the indentation so it "
        "cannot be counted here)"
    )
    return mutated


def _delete_capture_post_rope_q_observation(text):
    """Delete `_kv_calibration_capture`'s post-RoPE q observation (pipeline.py:2676 at
    faef137) -- the capture walk's own foreign-domain observation."""
    anchor = '            _observe(maxima, f"{prefix}.q", q_rope)\n'
    assert text.count(anchor) == 1, (
        "sanity: the capture's post-RoPE q observation must match verbatim, once"
    )
    mutated = text.replace(
        anchor,
        "            # T-2606 mutant: the capture's post-RoPE q observation deleted\n",
        1,
    )
    assert '_observe(maxima, f"{prefix}.q", q_rope)' not in mutated, (
        "sanity: the capture's post-RoPE q observation must be gone"
    )
    return mutated


def test_the_instance_predicate_is_green_under_the_raw_k_observation_deletion_mutant(tmp_path):
    """Mutation proof 1 (T-2606): delete `_float_layer`'s own post-transform raw-K
    observation (pipeline.py:3196) -- the exact line whose presence the instance cell
    pins -- and the instance predicate (the SAME shared implementation the red cell
    runs) is GREEN on the mutant: maxima[layer{L}.k] equals the raw k_proj peak exactly
    at every layer. Re-inserting the line -- the real tree at faef137, which IS the
    re-inserted state -- turns the same predicate red (quoted in this session's run of
    the red cell above)."""
    mutant = _load_mutant(_delete_raw_k_post_transform_observation, tmp_path)
    cfg, _, float_weight = _scaled_qk_norm_fixture(mutant)
    maxima, walk = _run_calibrate(mutant, cfg, float_weight)
    _assert_raw_projection_key_pure(cfg, maxima, walk, "k")
    for layer in range(cfg.num_hidden_layers):
        print(f"mutant maxima[layer{layer}.k]={maxima[f'layer{layer}.k']!r} "
              f"raw={walk.project_out[f'layer{layer}.k_proj']!r}")


def test_the_class_and_q_predicates_are_green_under_both_raw_key_observations_deleted(tmp_path):
    """Mutation proof 2 (T-2606): delete BOTH post-transform raw-key observations
    (pipeline.py:3195 and :3196) -- the full corrected behavior on the calibrated walk --
    and the class predicate (zero violations across the whole domain table) and the
    raw-Q instance predicate are GREEN on the mutant. The k_normed union observations
    are untouched by this mutant and stay green by design (D-SLM6263/6264)."""
    mutant = _load_mutant(_delete_both_raw_key_post_transform_observations, tmp_path)
    cfg, _, float_weight = _scaled_qk_norm_fixture(mutant)
    maxima, walk = _run_calibrate(mutant, cfg, float_weight)
    violations = _grade_walk(walk, maxima, "the mutant calibrated walk",
                             demand_full_table=True)
    assert not violations, "\n".join(violations)
    _assert_raw_projection_key_pure(cfg, maxima, walk, "q")
    _assert_raw_projection_key_pure(cfg, maxima, walk, "k")


def test_the_capture_predicate_is_green_under_the_capture_post_rope_q_observation_deleted(tmp_path):
    """Mutation proof 3 (T-2606): delete `_kv_calibration_capture`'s own post-RoPE q
    observation (pipeline.py:2676) and the capture predicate is GREEN on the mutant --
    the capture's maxima[layer{L}.q] equals its raw q_proj peak alone, the sibling walk's
    own share of the corrected behavior."""
    mutant = _load_mutant(_delete_capture_post_rope_q_observation, tmp_path)
    cfg, _, float_weight = _scaled_qk_norm_fixture(mutant)
    maxima, walk = _run_capture(mutant, cfg, float_weight)
    violations = _grade_walk(walk, maxima, "the mutant capture walk",
                             demand_full_table=False)
    assert not violations, "\n".join(violations)