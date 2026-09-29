#!/usr/bin/env python3
"""Builds a synthetic .sslm at real projection geometry (plan rev 2, S1-0), through the same
reference-pipeline path the in-tree fixture uses (tools/_t2199_s8_synthetic_full_model_fixture.py),
with only the config changed. Plain conversion (no DGC1, no schema), QK-norm gains stripped as the
in-tree fixture does, small vocabulary (the vocabulary does not enter the prefill GEMMs).
Usage: python3 synth_artifact.py <SuperSLM>/tools {0.5B|0.6B|1.5B|wide} out.sslm [layers]"""
import resource, sys, time
tools = sys.argv[1]
sys.path.insert(0, tools); sys.path.insert(0, tools + "/reference_pipeline")
from dataclasses import replace
import convert_model as C, pipeline as P, sslm_format as F

GEOM = {
    # Qwen3-0.6B projections: hidden 1024, 16 q heads x 128, 8 kv x 128, mlp 3072
    "0.6B": dict(hidden_size=1024, num_attention_heads=16, num_key_value_heads=8, head_dim=128, intermediate_size=3072),
    # Qwen2.5-1.5B projections: hidden 1536, 12 q heads x 128, 2 kv x 128, mlp 8960
    "1.5B": dict(hidden_size=1536, num_attention_heads=12, num_key_value_heads=2, head_dim=128, intermediate_size=8960),
    # Qwen2.5-0.5B projections: hidden 896, 14 q heads x 64, 2 kv x 64, mlp 4864
    "0.5B": dict(hidden_size=896, num_attention_heads=14, num_key_value_heads=2, head_dim=64, intermediate_size=4864),
    # the threading slice's test fixture: wide enough for >= 32 panel-aligned tasks (N >= 1024)
    "wide": dict(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64, intermediate_size=1024),
}
name, out = sys.argv[2], sys.argv[3]
layers = int(sys.argv[4]) if len(sys.argv) > 4 else 28
cfg = P.ModelConfig(num_hidden_layers=layers, vocab_size=256, rope_theta=10000.0, rms_norm_eps=1e-6,
                    tie_word_embeddings=True, context_cap=2048, **GEOM[name])
t0 = time.time()
model = P.fixture_model(cfg)
model = replace(model,
    weights={k: v for k, v in model.weights.items() if not (k.endswith(".q_norm.gain") or k.endswith(".k_norm.gain"))},
    weight_scales={k: v for k, v in model.weight_scales.items() if not (k.endswith(".q_norm.gain") or k.endswith(".k_norm.gain"))})
t1 = time.time()
sections, _ = C.build_sections(model)
data = F.build_artifact(sections)
if isinstance(data, tuple): data = data[0]
open(out, "wb").write(data)
t2 = time.time()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"{name} layers={layers}: model {t1 - t0:.1f} s, convert+write {t2 - t1:.1f} s, {len(data):,} bytes, peak RSS {rss:.0f} MiB")
