"""convert_model.py — emit a `.sslm` model artifact from a calibrated QuantizedModel.

Loads the spike's calibrated model, runs the C24/C25/C27 folds OFFLINE (float never
reaches the runtime — §6.8; `quantize_multiplier`/`_reference_fold` are offline-only),
and emits the binary model sections via `sslm_model_writer`. Proves the whole model
pipeline the S1 way: Python emits -> the C++ loader accepts byte-identically and the
ModelView reads every section back. Build-time tooling; Python ships nothing (§11).
"""

import argparse
import sys

import numpy as np

# The spike (calibrated-model loader + the offline fold pipeline) lives in the Wizard
# records tree; this is build-time tooling, so a cross-tree import is fine (nothing ships).
_SPIKE_ROOT = r"D:\Wizard\Tools"
if _SPIKE_ROOT not in sys.path:
    sys.path.insert(0, _SPIKE_ROOT)
from superslm_spike import artifact_cache, pipeline  # noqa: E402

import sslm_format as F  # noqa: E402
import sslm_model_writer as W  # noqa: E402


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

    return sections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="calibrated artifact directory")
    ap.add_argument("--out", required=True, help="output .sslm path")
    args = ap.parse_args()

    model = artifact_cache.load_artifact(args.artifact)
    sections = build_sections(model)
    fp = F.write_artifact(args.out, sections)
    print(f"wrote {args.out}")
    print(f"fingerprint {fp}")
    print(f"sections {len(sections)}: " + ", ".join(str(s.type) for s in sections))


if __name__ == "__main__":
    main()
