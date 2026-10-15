"""Paged-KV plan rev 16.21 §23.8 step T0: the extra cloud fixtures the R9 red suite reads.

The reference pipeline's construction (tools/gen_paged_kv_fixture.py, tools/gen_decode_threading_fixture.py),
at the geometries the R9 cells need and the generated fixtures do not carry. Two layers, two KV heads,
hidden 192, four query heads and vocabulary 256 unless a row says otherwise; QK-norm stripped and the
damped-greedy constants on, as pkv_def. The calibration is computed once per (head_dim, layers, heads,
KV heads) at cap 160 and reused at every cap of that shape: it reads prompts far shorter than any cap
it is reused at, and the artifact's cap enters only CFG1 and the RoPE table. Generated fresh by the
leg that runs the suite and never committed, as the other paged-KV fixtures are.

  r9_d2_c8        D = 2, cap 8: a cap below one rule page (cell 4.10; also the D = 2 grid base)
  r9_d64_c63      D = 64, caps 63, 64, 65 and 127: the page condition's boundary row (cell 4.10)
  r9_d64_c64
  r9_d64_c65
  r9_d64_c127
  r9_d64_c4096    D = 64, cap 4,096: the D = 64 grid base and the 0.5B-shaped cloud geometry
  r9_d512_c64     D = 512 and 1,024: the grid's floor rows (bases; the cells re-cap them with the
  r9_d1024_c64    RoPE table stripped, which no geometry-only verb reads)
  r9_l3_c4096     three layers, D = 48, cap 4,096: runs = 12, which does not divide 64 (cell 6.6b)
  r9_d48_c100     D = 48, cap 100: the rule's B is the cap, so a page pool and a contiguous pool share one
                  geometry (cell 4.15's admitted cross-kind pair, rev 16.23)
  r9_q8_c64       36 layers, 8 query heads, 8 KV heads, D = 128: the Qwen3-8B K/V geometry of
                  cell 4.14's overflow pair (the cell sets Int16 K/V and cap 40,960 in CFG1).
                  8 query heads, not 32: the block product reads layers, KV heads, D and the
                  element width only, and 32 heads held 8 GB in the calibration pass

CFG1's kv_block_size is the converter's 16 in every file; the cells patch it where a row needs
another value. Every file's SHA-256 is printed; the cells do not pin these hashes (they compare
within one run, never against a recorded reference).

Usage: python -I tests/paged-kv/r9_gen_fixtures.py OUT_DIR [NAME ...]
"""

import dataclasses
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir, "tools"))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "reference_pipeline"))

import gen_decode_threading_fixture as G  # noqa: E402

CAL_CAP = 160

ROWS = (
    # name, head_dim, cap, layers, query heads, KV heads
    ("r9_d2_c8", 2, 8, 2, 4, 2),
    ("r9_d64_c63", 64, 63, 2, 4, 2),
    ("r9_d64_c64", 64, 64, 2, 4, 2),
    ("r9_d64_c65", 64, 65, 2, 4, 2),
    ("r9_d64_c127", 64, 127, 2, 4, 2),
    ("r9_d64_c4096", 64, 4096, 2, 4, 2),
    ("r9_d512_c64", 512, 64, 2, 4, 2),
    ("r9_d1024_c64", 1024, 64, 2, 4, 2),
    ("r9_l3_c4096", 48, 4096, 3, 4, 2),
    ("r9_d48_c100", 48, 100, 2, 4, 2),
    ("r9_q8_c64", 128, 64, 36, 8, 8),
)


def shape(head_dim, cap, layers, heads, kv_heads):
    return dataclasses.replace(G.config(head_dim), context_cap=cap, num_hidden_layers=layers,
                               num_attention_heads=heads, num_key_value_heads=kv_heads)


def _replicate_layers(by_key, layers):
    """{"layer<l>.<site>": v} for l in 0..layers-1, from a two-layer map: layer l takes layer l % 2's."""
    out = {}
    for key, value in by_key.items():
        head, dot, site = key.partition(".")
        if head.startswith("layer") and head[len("layer"):].isdigit():
            src = int(head[len("layer"):])
            for layer in range(src, layers, 2):
                out[f"layer{layer}{dot}{site}"] = value
        else:
            out[key] = value
    return out


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    out_dir = argv[1]
    only = set(argv[2:])
    os.makedirs(out_dir, exist_ok=True)
    for name, head_dim, cap, layers, heads, kv_heads in ROWS:
        if only and name not in only:
            continue
        start = time.time()
        cal = shape(head_dim, CAL_CAP, layers, heads, kv_heads)
        cfg = shape(head_dim, cap, layers, heads, kv_heads)
        if layers > 8:
            # A deep row borrows a two-layer calibration of the same head shape: layer l takes layer
            # (l mod 2)'s peaks. The calibration pass over 36 layers held 7 to 8 GB; the cells that
            # read a deep row are sizing cells, and no verdict reads its decode.
            cal = shape(head_dim, CAL_CAP, 2, heads, kv_heads)
            if cal not in G._CALIBRATION:
                G.fixture_model(cal, {"k": G.K_PEAK_FACTOR_DEF})
            G._CALIBRATION[cfg] = tuple(_replicate_layers(d, layers) for d in G._CALIBRATION[cal])
        elif cfg != cal:
            if cal not in G._CALIBRATION:
                G.fixture_model(cal, {"k": G.K_PEAK_FACTOR_DEF})
            G._CALIBRATION[cfg] = G._CALIBRATION[cal]
        model = G.strip_qk_norm(G.fixture_model(cfg, {"k": G.K_PEAK_FACTOR_DEF}))
        data, fingerprint = G.artifact(model, True)
        path = os.path.join(out_dir, name + ".sslm")
        with open(path, "wb") as f:
            f.write(data)
        print(f"wrote {path}: {len(data)} bytes, sha256={hashlib.sha256(data).hexdigest()}, "
              f"fingerprint={fingerprint}, {time.time() - start:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
