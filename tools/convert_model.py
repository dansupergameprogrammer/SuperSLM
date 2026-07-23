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

# The spike (calibrated-model loader + the offline fold pipeline) lives in the Wizard
# records tree; this is build-time tooling, so a cross-tree import is fine (nothing ships).
_SPIKE_ROOT = r"D:\Wizard\Tools"
if _SPIKE_ROOT not in sys.path:
    sys.path.insert(0, _SPIKE_ROOT)
from superslm_spike import artifact_cache, pipeline  # noqa: E402

import sslm_convert_manifest as M  # noqa: E402
import sslm_convert_validate as V  # noqa: E402
import sslm_format as F  # noqa: E402
import sslm_model_writer as W  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fold_ops_tensor(channel_scales):
    """C24/C25 per-channel fold as a [num_channels, 3] int32 tensor: each row is
    (identity, mult, shift). `pipeline._reference_fold` returns None for a true
    pass-through channel (C24's identity) and an offline (mult, shift) otherwise."""
    folds, _s_ref = pipeline._reference_fold(list(channel_scales))
    rows = [(1, 0, 0) if f is None else (0, int(f[0]), int(f[1])) for f in folds]
    return np.asarray(rows, dtype=np.int32)


def _ctx_fold_tensor(model, layer):
    """C27/D-SLM57 attention-context per-head fold as a [num_attention_heads, 3] int32
    tensor. Mirrors dynamic_engine's offline computation: f_v / s_v_max -> (mult, shift),
    identity where a head already sits at the max V scale."""
    cfg = model.config
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    s_v = [model.scales.scale(f"layer{layer}.v_head{h}.scale") for h in range(cfg.num_key_value_heads)]
    s_v_max = max(s_v)
    rows = []
    for head in range(cfg.num_attention_heads):
        f_v = s_v[head // group]
        if f_v == s_v_max:
            rows.append((1, 0, 0))
        else:
            m, sh = pipeline.quantize_multiplier(f_v / s_v_max)
            rows.append((0, int(m), int(sh)))
    return np.asarray(rows, dtype=np.int32)


def build_sections(model):
    cfg = model.config
    sections = []

    # Config (CFG1). q_b is uniform 30 across dynamic_biases (verified); kv/block are v1
    # forward-looking. Unicode version is the tokenizer's pin (15.1.0).
    sections.append(F.Section(F.SectionType.CONFIG, W.write_cfg1(
        hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, intermediate_size=cfg.intermediate_size, vocab_size=cfg.vocab_size,
        context_cap=cfg.context_cap, tie_word_embeddings=cfg.tie_word_embeddings,
        kv_precision=W.KV_PRECISION_INT8, kv_block_size=16,
        unicode_major=15, unicode_minor=1, unicode_patch=0,
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
    wsc = {k: _fold_ops_tensor(model.weight_scales[k]) for k in sorted(model.weight_scales)}
    for L in range(cfg.num_hidden_layers):
        wsc[f"layer{L}.ctx_fold"] = _ctx_fold_tensor(model, L)
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

    return sections


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

    model = artifact_cache.load_artifact(args.artifact)

    # Phase 1: validate. Every dtype/range/integralness/finiteness/geometry/
    # required-key/shape/scale-positivity/fold-bound/Unicode-coherence claim
    # is checked BEFORE a single array is cast — reject-over-degrade, not the
    # old writer's silent np.asarray(..., dtype=X) coercion.
    V.validate_model(model, fold_ops_tensor=_fold_ops_tensor, ctx_fold_tensor=_ctx_fold_tensor,
                     unicode_major=15, unicode_minor=1, unicode_patch=0)

    # Phase 2: serialize (explicit little-endian dtypes throughout sslm_model_writer.py).
    sections = build_sections(model)
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
                                  manifest_out_path=args.manifest_out)
    manifest_path = args.manifest_out or (args.out + ".manifest.json")
    print(f"verified: independent loader accepted the artifact")
    print(f"proof manifest: {manifest_path}")


if __name__ == "__main__":
    main()
