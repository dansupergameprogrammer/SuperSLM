"""convert_model.py — emit a `.sslm` model artifact from a calibrated QuantizedModel.

Loads the spike's calibrated model, runs the C24/C25/C27 folds OFFLINE (float never
reaches the runtime — §6.8; `quantize_multiplier`/`_reference_fold` are offline-only),
and emits the binary model sections via `sslm_model_writer`. Proves the whole model
pipeline the S1 way: Python emits -> the C++ loader accepts byte-identically and the
ModelView reads every section back. Build-time tooling; Python ships nothing (§11).

S-HARDEN-3 (F13): conversion is now the two-phase checked transaction
SuperSLM_Plan.md §13 item 7 specifies — validate (sslm_convert_validate.validate_model,
reject-over-degrade for every dtype/range/geometry/scale/fold-bound claim), THEN
serialize (build_sections, now writing explicit little-endian dtypes), THEN invoke
the independent C++ verifier (sslm_convert_manifest.verify_and_merge, which runs the
compiled `sslm_verify` binary and raises on rejection or a failed geometry
cross-check) and emit the combined proof manifest. Coercion is not proof: the old
two-line pipeline (build_sections then write_artifact, with no validate phase and no
independent load-back) is exactly what let an int16 [128,-129] array become
int8 [-128,127] silently — this module's own defect finding.
"""

import argparse
import os
import sys

import numpy as np

import sslm_convert_manifest as M  # noqa: E402
import sslm_convert_validate as V  # noqa: E402
import sslm_format as F  # noqa: E402
import sslm_model_writer as W  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The spike (calibrated-model loader + the offline fold pipeline) lives in the Wizard
# records tree; this is build-time tooling, so a cross-tree import is fine (nothing
# ships) -- but it is a LAZY import (S-HARDEN-3), triggered only when the real
# `_fold_ops_tensor`/`_ctx_fold_tensor` defaults below are actually called, not at
# module load. This is what keeps convert_model.py's own `build_sections`/`main`
# importable on a bare checkout (no D:\Wizard sibling) for tests and CI that supply
# their own fold functions -- the same constraint sslm_convert_validate.py's module
# docstring documents for every other test module in this slot.
_SPIKE_ROOT = r"D:\Wizard\Tools"


def _load_spike():
    if _SPIKE_ROOT not in sys.path:
        sys.path.insert(0, _SPIKE_ROOT)
    from superslm_spike import artifact_cache, pipeline  # noqa: E402
    return artifact_cache, pipeline


def _fold_ops_tensor(channel_scales):
    """C24/C25 per-channel fold as a [num_channels, 3] int32 tensor: each row is
    (identity, mult, shift). `pipeline._reference_fold` returns None for a true
    pass-through channel (C24's identity) and an offline (mult, shift) otherwise."""
    _artifact_cache, pipeline = _load_spike()
    folds, _s_ref = pipeline._reference_fold(list(channel_scales))
    rows = [(1, 0, 0) if f is None else (0, int(f[0]), int(f[1])) for f in folds]
    return np.asarray(rows, dtype=np.int32)


def _ctx_channel_scales(model, layer):
    """The per-head V scale one layer's ctx-fold approximates, in `_ctx_fold_tensor`'s own
    row order -- factored out so `build_sections`'s `fold_approximation_error`
    aggregation (T-408 §3.1/§8 step 1) and `_ctx_fold_tensor` itself share one source of
    the per-head scale gather rather than deriving it twice and risking a silent
    divergence."""
    cfg = model.config
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    s_v = [model.scales.scale(f"layer{layer}.v_head{h}.scale") for h in range(cfg.num_key_value_heads)]
    return [s_v[head // group] for head in range(cfg.num_attention_heads)]


def _ctx_fold_tensor(model, layer):
    """C27/D-SLM57 attention-context per-head fold as a [num_attention_heads, 3] int32
    tensor. Mirrors dynamic_engine's offline computation: f_v / s_v_max -> (mult, shift),
    identity where a head already sits at the max V scale."""
    _artifact_cache, pipeline = _load_spike()
    s_v = _ctx_channel_scales(model, layer)
    s_v_max = max(s_v)
    rows = []
    for f_v in s_v:
        if f_v == s_v_max:
            rows.append((1, 0, 0))
        else:
            m, sh = pipeline.quantize_multiplier(f_v / s_v_max)
            rows.append((0, int(m), int(sh)))
    return np.asarray(rows, dtype=np.int32)


def _fold_relative_error(mult, shift, exact):
    """T-408 §3.1's per-channel relative error between a gemmlowp (mult, shift) pair's
    reconstructed real value and the exact ratio it approximates.

    The design's own pseudocode states `approx = mult * 2**-shift`; the real value a
    (mult, shift) pair reconstructs to also carries the pair's implicit `2**-31`
    normalization -- `intmath.saturating_rounding_doubling_high_mul`'s
    `(a*b + 2**30) >> 31` applies this same scaling to `mult` before
    `rounding_divide_by_pot`'s `2**-shift` step runs, and `quantize_multiplier`'s own
    construction (`multiplier = round(fraction * (1 << 31))`) is what produces a `mult`
    in `[2**30, 2**31)` that needs it. Read directly from the codebase (StandardsDocument
    §5.4: exactness verified at source, never by construction) rather than applying the
    design's literal formula, which would report a >>1 relative error for every
    non-identity fold and fail this metric's own CI range gate (§6 item 3:
    `0.0 <= value < 1.0`) on every real conversion."""
    approx = mult * (2.0 ** -31) * (2.0 ** -shift)
    return abs(approx - exact) / exact


def _fold_max_relative_error(rows, channel_scales):
    """T-408 §3.1's `fold_approximation_error`, aggregated over one WSC1 tensor's rows:
    the maximum relative error over every non-identity fold in `rows`, fed from the same
    `channel_scales` the caller already passed to `fold_ops_tensor`/`ctx_fold_tensor` --
    no new data crosses the cache boundary and no fold is recomputed a second time.
    Identity rows (rows[i][0] == 1) contribute 0.0 by construction: no approximation was
    performed on that channel, which is the true, non-fabricated relative error for it,
    whether the row came from a real fold or an injected fixture's own pass-through."""
    s_ref = max(channel_scales)
    errors = [0.0]
    for (identity, mult, shift), s_i in zip(rows, channel_scales):
        if identity:
            continue
        errors.append(_fold_relative_error(int(mult), int(shift), s_i / s_ref))
    return max(errors)


def build_sections(model, *, fold_ops_tensor=None, ctx_fold_tensor=None):
    """`fold_ops_tensor`/`ctx_fold_tensor` default to this module's own
    spike-backed implementations above; a caller that cannot import the spike
    (every test in this slot, and the pinned CI fixture) injects its own pure
    functions instead -- the same injection design as `sslm_convert_validate.
    validate_model`, so the SAME real `build_sections` runs in both the
    production path and the CI gate, never a reimplementation that could
    silently diverge from what actually ships.

    Returns `(sections, fold_approximation_error)` (T-408 §5/§8 step 1):
    `fold_approximation_error` is the maximum relative error (§3.1) over every
    per-channel weight fold and per-head ctx-fold this call emits, aggregated
    from the same WSC1 rows and channel scales already computed to build the
    WeightScales section -- no second pass over the tensors, no new data
    crossing the cache boundary.
    """
    fold_ops_tensor = fold_ops_tensor or _fold_ops_tensor
    ctx_fold_tensor = ctx_fold_tensor or _ctx_fold_tensor
    cfg = model.config
    sections = []
    fold_errors = [0.0]

    # Config (CFG1). q_b is uniform 30 across dynamic_biases (verified); kv/block are v1
    # forward-looking. Unicode version is the tokenizer's pin, imported from
    # sslm_convert_validate.PINNED_UNICODE_VERSION -- the single source of
    # truth, not a second independently-typed copy of the same three integers
    # (S-HARDEN-5, F8).
    sections.append(F.Section(F.SectionType.CONFIG, W.write_cfg1(
        hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, intermediate_size=cfg.intermediate_size, vocab_size=cfg.vocab_size,
        context_cap=cfg.context_cap, tie_word_embeddings=cfg.tie_word_embeddings,
        kv_precision=W.KV_PRECISION_INT8, kv_block_size=16,
        unicode_major=V.PINNED_UNICODE_VERSION[0], unicode_minor=V.PINNED_UNICODE_VERSION[1],
        unicode_patch=V.PINNED_UNICODE_VERSION[2],
        rope_theta=cfg.rope_theta, rms_norm_eps=cfg.rms_norm_eps)))

    # Weights (WGT1, int8).
    weights = {k: np.asarray(model.weights[k], dtype=np.int8) for k in sorted(model.weights)}
    sections.append(F.Section(F.SectionType.WEIGHTS, W.write_tensor_manifest(W.WGT1, np.int8, weights)))

    # Biases (BIA1, int64) — the C28 dynamic-bias codes, per site.
    dbias = {}
    for site in sorted(model.dynamic_biases):
        _q_b, codes = model.dynamic_biases[site]
        dbias[site] = np.asarray(codes, dtype=np.int64)
    sections.append(F.Section(F.SectionType.BIASES, W.write_tensor_manifest(W.BIA1, np.int64, dbias)))

    # RopeTables (ROP1, int64) — cos, sin.
    cos, sin = model.rope_tables
    rope = {"cos": np.asarray(cos, dtype=np.int64), "sin": np.asarray(sin, dtype=np.int64)}
    sections.append(F.Section(F.SectionType.ROPE_TABLES, W.write_tensor_manifest(W.ROP1, np.int64, rope)))

    # WeightScales (WSC1, int32) — per-channel fold ops + the per-layer ctx-fold.
    wsc = {}
    for k in sorted(model.weight_scales):
        channel_scales = model.weight_scales[k]
        rows = fold_ops_tensor(channel_scales)
        wsc[k] = rows
        fold_errors.append(_fold_max_relative_error(rows, channel_scales))
    for L in range(cfg.num_hidden_layers):
        rows = ctx_fold_tensor(model, L)
        wsc[f"layer{L}.ctx_fold"] = rows
        fold_errors.append(_fold_max_relative_error(rows, _ctx_channel_scales(model, L)))
    sections.append(F.Section(F.SectionType.WEIGHT_SCALES, W.write_tensor_manifest(W.WSC1, np.int32, wsc)))

    # Composition constants (KVC1, 2 words) — plus the uniform bias q_b, stored so the
    # runtime C28 bias reconcile has it (value 0 unused).
    cc = {k: (int(model.composition_constants[k][0]), int(model.composition_constants[k][1]))
          for k in sorted(model.composition_constants)}
    cc["bias.q_b"] = (30, 0)
    sections.append(F.Section(F.SectionType.COMPOSITION_CONSTANTS, W.write_kvc1(2, cc)))

    # KV landing scales (KVC1, 2 words) and reciprocals (KVC1, 3 words).
    kls = {k: tuple(int(x) for x in model.kv_landing_scales[k]) for k in sorted(model.kv_landing_scales)}
    sections.append(F.Section(F.SectionType.KV_LANDING_SCALES, W.write_kvc1(2, kls)))
    klr = {k: tuple(int(x) for x in model.kv_landing_reciprocals[k]) for k in sorted(model.kv_landing_reciprocals)}
    sections.append(F.Section(F.SectionType.KV_LANDING_RECIPROCALS, W.write_kvc1(3, klr)))

    # SigmoidLut (SIL1) — required from v2 (C10, D-SLM68). S-HARDEN-1 (F1): this was
    # never wired in, so the standard converter emitted a formally invalid v2
    # artifact that the loader's presence-only Config check let through unnoticed
    # (the correlated-oracle failure §17.3 exists to catch). The table is the fixed
    # universal construction (build_sigmoid_lut) the runtime's pinned canonical
    # content check (ParseSigmoidLut, src/model.cpp) validates against byte-for-byte.
    sections.append(F.Section(F.SectionType.SIGMOID_LUT, W.write_sil1()))

    return sections, max(fold_errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="calibrated artifact directory")
    ap.add_argument("--out", required=True, help="output .sslm path")
    ap.add_argument("--verifier", default=None,
                    help="path to the compiled sslm_verify binary (default: searched under build/)")
    ap.add_argument("--manifest-out", default=None,
                    help="path for the combined proof manifest (default: <out>.manifest.json)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip invoking the independent C++ verifier (debugging only -- "
                         "the artifact's 'must load Ok' contract is NOT discharged without it)")
    args = ap.parse_args()

    artifact_cache, _pipeline = _load_spike()
    model = artifact_cache.load_artifact(args.artifact)

    # Phase 1: validate. Every dtype/range/integralness/finiteness/geometry/
    # required-key/shape/scale-positivity/fold-bound/Unicode-coherence claim
    # is checked BEFORE a single array is cast — reject-over-degrade, not the
    # old writer's silent np.asarray(..., dtype=X) coercion.
    V.validate_model(model, fold_ops_tensor=_fold_ops_tensor, ctx_fold_tensor=_ctx_fold_tensor,
                     unicode_major=V.PINNED_UNICODE_VERSION[0],
                     unicode_minor=V.PINNED_UNICODE_VERSION[1],
                     unicode_patch=V.PINNED_UNICODE_VERSION[2])

    # Phase 2: serialize (explicit little-endian dtypes throughout sslm_model_writer.py).
    sections, fold_approximation_error = build_sections(model)
    fp = F.write_artifact(args.out, sections)
    print(f"wrote {args.out}")
    print(f"fingerprint {fp}")
    print(f"sections {len(sections)}: " + ", ".join(str(s.type) for s in sections))

    # Phase 3: invoke the independent C++ verifier and emit the proof manifest
    # (§13 item 7). This is what discharges the writer's "must load Ok"
    # contract by INVOKING the loader, not by asserting it.
    if args.skip_verify:
        print("--skip-verify set: the independent verifier was NOT run; the artifact's "
              "'must load Ok' contract is unproven for this conversion")
        return

    verifier_cmd = [args.verifier] if args.verifier else None
    manifest = M.verify_and_merge(_REPO_ROOT, args.out, args.artifact, verifier_cmd=verifier_cmd,
                                  manifest_out_path=args.manifest_out, model=model,
                                  fold_approximation_error=fold_approximation_error)
    manifest_path = args.manifest_out or (args.out + ".manifest.json")
    print(f"verified: independent loader accepted the artifact")
    print(f"proof manifest: {manifest_path}")


if __name__ == "__main__":
    main()
