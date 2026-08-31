"""T-2432 Track A acceptance fixture: a non-square-geometry checkpoint directory, built the
same way tools/_calibrate_checkpoint_fixture.py builds its square one (same tokenizer/config/
safetensors shape, same Qwen2.5-style "model."-prefixed, biased tensor convention -- Track C
is untouched by this build, so the checkpoint must still satisfy Track C's pre-existing,
unmodified _upstream_names/check_required_groups contract). The only change is geometry:
NUM_ATTENTION_HEADS=4, HEAD_DIM=4, HIDDEN_SIZE=8 -> q_width=16 != hidden_size=8 (non-square).
No q_norm/k_norm tensors are written -- isolating this fixture from Track B, per the design's
own red-state fixture spec (design Sec6 Track A, "Red first").

Run as a script: writes the checkpoint directory to the given path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _calibrate_checkpoint_fixture as base


def build(checkpoint_dir: Path, *, hidden_size, num_attention_heads, num_key_value_heads,
          head_dim, intermediate_size, num_hidden_layers=1, seed=0) -> Path:
    base.HIDDEN_SIZE = hidden_size
    base.NUM_ATTENTION_HEADS = num_attention_heads
    base.NUM_KEY_VALUE_HEADS = num_key_value_heads
    base.HEAD_DIM = head_dim
    base.INTERMEDIATE_SIZE = intermediate_size
    base.NUM_HIDDEN_LAYERS = num_hidden_layers
    return base.build_fixture_checkpoint(checkpoint_dir, seed=seed)


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("t2432_nonsquare_checkpoint")
    kind = sys.argv[2] if len(sys.argv) > 2 else "nonsquare"
    if kind == "nonsquare":
        # q_width = 4*4 = 16 != hidden_size = 8.
        build(out_dir, hidden_size=8, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
              intermediate_size=16)
    elif kind == "square":
        # q_width = 2*4 = 8 == hidden_size = 8 -- the existing-incumbent regression fixture.
        build(out_dir, hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=4,
              intermediate_size=16)
    else:
        raise SystemExit(f"unknown kind {kind!r}")
    print(f"wrote checkpoint to {out_dir}")
