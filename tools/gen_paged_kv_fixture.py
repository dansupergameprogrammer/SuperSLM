"""Paged-KV plan (rev 16.1) §8 and step R0: the cloud fixtures for the paged-KV red suite.

The same construction as tools/gen_decode_threading_fixture.py (the reference pipeline's pinned
weights and calibration, converted by `convert_model.build_sections`), at the context caps the
paged-KV cells need. The plan's cells are written at cap 4096 with `kv_block_size` 16, so `B = 16`
and a whole-cap reservation is 256 pages; this geometry keeps that page arithmetic while a token's
K/V is 384 bytes, so a 1,500-token run costs milliseconds. Deterministic and hermetic on a given
host, generated fresh by every leg that runs the suite and never committed, as the decode-threading
fixtures are. pkv_qk.sslm carries a QKC1 table and inherits F-QK's host dependence (its calibration
peaks follow numpy's float64 exp); no cell pins its hash. The others carry no QKC1 table.

Variants (two layers, vocabulary 256):

  pkv_def.sslm   F-DEF's geometry and calibration knob at cap 4096: hidden 192, 4 query heads and
                 2 KV heads of 48, intermediate 512, QK-norm stripped, the damped-greedy constants
                 and the S8 schema. The ABI cells' default model.
  pkv_qk.sslm    F-QK's geometry at cap 4096 (QK-norm kept, heads of 128): the direct_qk score
                 branch (cell 4.6).
  pkv_odd.sslm   pkv_def at cap 4100, which 16 does not divide, so `B = cap` (cell 4.4).
  pkv_32k.sslm   pkv_def at cap 32768 (cells 2.9 and 9.5's second cap).

Usage: python tools/gen_paged_kv_fixture.py OUT_DIR
"""

import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import gen_decode_threading_fixture as G  # noqa: E402


def config(head_dim, context_cap):
    return dataclasses.replace(G.config(head_dim), context_cap=context_cap)


VARIANTS = (
    ("pkv_def.sslm", 48, 4096, True, {"k": G.K_PEAK_FACTOR_DEF}, True),
    ("pkv_qk.sslm", 128, 4096, False, {}, False),
    ("pkv_odd.sslm", 48, 4100, True, {"k": G.K_PEAK_FACTOR_DEF}, True),
    ("pkv_32k.sslm", 48, 32768, True, {"k": G.K_PEAK_FACTOR_DEF}, True),
)


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    out_dir = argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for name, head_dim, context_cap, strip, factors, with_schema in VARIANTS:
        start = time.time()
        model = G.fixture_model(config(head_dim, context_cap), factors)
        if strip:
            model = G.strip_qk_norm(model)
        data, fingerprint = G.artifact(model, with_schema)
        path = os.path.join(out_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"wrote {path}: {len(data)} bytes, fingerprint={fingerprint}, "
              f"{time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
