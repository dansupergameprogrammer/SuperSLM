"""Print the carried-scale reference's layer-0 post-attention int8 digest for token IDs."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).with_name("reference_pipeline")))
import artifact_cache
import pipeline


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(f"usage: {sys.argv[0]} <artifact-cache-dir> <comma-separated-token-ids> [--stepped]", file=sys.stderr)
        return 2
    model = artifact_cache.load_artifact(Path(sys.argv[1]))
    tokens = [int(v) for v in sys.argv[2].split(",")]
    if len(sys.argv) == 4 and sys.argv[3] != "--stepped":
        print(f"unknown option: {sys.argv[3]}", file=sys.stderr)
        return 2
    cache = pipeline.new_kv_cache(model) if len(sys.argv) == 4 else None
    row = None
    for token in tokens if cache is not None else [tokens]:
        row = pipeline.forward_dynamic_attention_layer(
            model, [token] if cache is not None else token, 0, cache=cache)[-1]
    codes = bytes((int(v) & 0xFF) for v in row)
    print(f"attention_residual_codes_sha256={hashlib.sha256(codes).hexdigest()} count={len(codes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
