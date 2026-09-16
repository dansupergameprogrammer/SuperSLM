"""T-2704 durable-residual-state topology manifest.

The allowed state is the residual site's local selected scale and local wide
row before the existing funnel.  The manifest rejects durable state carriers
and residual output commits before that funnel boundary.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "src/forward/forward_sites.cpp",
    ROOT / "src/gpu/shaders/attn_residual_site.hlsl",
    ROOT / "src/gpu/shaders/mlp_residual_site.hlsl",
)
FORBIDDEN = ("residual_buffer", "residual_sidecar", "residual_group", "residual_peel",
             "persistent_residual", "durable_residual")


def test_t2704_residual_topology_has_no_durable_state_carrier():
    text = "\n".join(path.read_text(encoding="utf-8") for path in SOURCES).lower()
    for forbidden in FORBIDDEN:
        assert forbidden not in text, f"durable residual state carrier introduced: {forbidden}"
