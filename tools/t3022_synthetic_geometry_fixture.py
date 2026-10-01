"""Generate compact synthetic models with real 0.5B/1.5B KV page geometry.

Weights use the repository's pinned synthetic pipeline; layer count, KV heads,
head width and cap match the real artifacts. The resulting .sslm files are
inputs to E0 prefill reference capture, not substitutes for real-model runs.
"""

import argparse
import dataclasses
import hashlib

import gen_decode_threading_fixture as G


SHAPES = {
    "0p5": (24, 64, 4096),
    "1p5": (28, 128, 32768),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shape", choices=SHAPES)
    ap.add_argument("out")
    args = ap.parse_args()
    layers, head_dim, cap = SHAPES[args.shape]
    cfg = dataclasses.replace(G.config(head_dim), hidden_size=4 * head_dim,
                              num_hidden_layers=layers, context_cap=cap)
    model = G.strip_qk_norm(G.fixture_model(cfg, {"k": G.K_PEAK_FACTOR_DEF}))
    data, fingerprint = G.artifact(model, with_schema=True)
    with open(args.out, "wb") as stream:
        stream.write(data)
    print(f"{args.out} bytes={len(data)} sha256={hashlib.sha256(data).hexdigest()} "
          f"fingerprint={fingerprint} layers={layers} kv_heads=2 head_dim={head_dim} cap={cap}")


if __name__ == "__main__":
    main()
