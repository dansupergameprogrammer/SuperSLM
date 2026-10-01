"""Paged-KV plan (rev 16.1) §8 and step R0: the cloud fixtures for the paged-KV red suite.

The same construction as tools/gen_decode_threading_fixture.py (the reference pipeline's pinned
weights and calibration, converted by `convert_model.build_sections`), at the context caps the
paged-KV cells need. The plan's cells are written at cap 4096 with `kv_block_size` 16, so `B = 16`
and a whole-cap reservation is 256 pages; this geometry keeps that page arithmetic while a token's
K/V is 384 bytes, so a 1,500-token run costs milliseconds. Generated fresh by every leg that runs
the suite and never committed, as the decode-threading fixtures are.

Every fixture is byte-identical on every host, and the cells hold them to it: each reference
(tests/paged-kv/reference/<tag>_<fixture>.ref) records the sha256 of the fixture it was made from,
and pkv_common.h fails a run against any other. This generator checks the same hashes after
writing and exits non-zero on a mismatch, so a host that builds different bytes is caught here,
not in the cells.

pkv_qk is the one that needs help. F-QK's bytes follow the host's float library
(docs/decode-threading/fixture-premise.txt, "Host dependence of F-QK"): its QKC1 table is exact in
the calibration's float64 peaks, whose low bits follow numpy's CPU-dispatched np.exp, and its
RoPE table rounds math.pow/cos/sin results to Q2.30, some of which sit within a few ulps of a
rounding midpoint. So pkv_qk is built from a committed pin,
tests/paged-kv/reference/pins/pkv_qk_host_pin.json, holding what this generator computed on the
reference's host:

  calibration   the per-site maxima and the per-(KV head, channel) post-RoPE K peaks that
                `pipeline._calibrate` returns, as exact float hex. The calibration pass (the only
                np.exp, and the BLAS matmuls) does not run for pkv_qk.
  rope          the 64 inverse frequencies (math.pow) as float hex, and the Q2.30 value of every
                cos/sin entry whose float lies within MARGIN_ULPS ulps of a rounding midpoint.
                Every other entry rounds the same under any cos/sin within MARGIN_ULPS - 1 ulps of
                the reference host's.

Everything else in the file is integer arithmetic, exact rationals, or IEEE-754 basic operations
(+, -, *, /, sqrt, frexp), which every conforming host rounds the same. The other three variants
carry no QKC1 table, their calibration-derived constants are coarse enough that the float
differences do not move them, and they have matched on every host measured; they are generated
as before and checked by the same hashes.

`--record-qk-pin` recomputes the pin on this host instead of reading it, and skips the hash check.
Use it only when the fixtures change on purpose, through
`PKV_RECORD_QK_PIN=1 tools/build_paged_kv_reference.sh`, which then records new references
against the new fixtures; commit the pin with them.

Variants (two layers, vocabulary 256):

  pkv_def.sslm   F-DEF's geometry and calibration knob at cap 4096: hidden 192, 4 query heads and
                 2 KV heads of 48, intermediate 512, QK-norm stripped, the damped-greedy constants
                 and the S8 schema. The ABI cells' default model.
  pkv_qk.sslm    F-QK's geometry at cap 4096 (QK-norm kept, heads of 128): the direct_qk score
                 branch (cell 4.6). Built from the pin above.
  pkv_odd.sslm   pkv_def at cap 4100, which 16 does not divide, so `B = cap` (cell 4.4).
  pkv_32k.sslm   pkv_def at cap 32768 (cells 2.9 and 9.5's second cap).

Usage: python tools/gen_paged_kv_fixture.py [--record-qk-pin] OUT_DIR
"""

import dataclasses
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import gen_decode_threading_fixture as G  # noqa: E402
import pipeline as P  # noqa: E402  (on sys.path through G)
from reference_pipeline import rope as R  # noqa: E402

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                             "tests", "paged-kv", "reference")
QK_PIN = os.path.join(REFERENCE_DIR, "pins", "pkv_qk_host_pin.json")
QK_FIXTURE = "pkv_qk.sslm"
# A cos/sin entry this close to a Q2.30 rounding midpoint (in ulps of its own float) is pinned.
MARGIN_ULPS = 64


def config(head_dim, context_cap):
    return dataclasses.replace(G.config(head_dim), context_cap=context_cap)


VARIANTS = (
    ("pkv_def.sslm", 48, 4096, True, {"k": G.K_PEAK_FACTOR_DEF}, True),
    (QK_FIXTURE, 128, 4096, False, {}, False),
    ("pkv_odd.sslm", 48, 4100, True, {"k": G.K_PEAK_FACTOR_DEF}, True),
    ("pkv_32k.sslm", 48, 32768, True, {"k": G.K_PEAK_FACTOR_DEF}, True),
)


def _config_key(cfg):
    return {k: v for k, v in sorted(dataclasses.asdict(cfg).items())}


def _inv_freq(cfg):
    return [cfg.rope_theta ** (-2.0 * i / cfg.head_dim) for i in range(cfg.head_dim // 2)]


def _rope_tables(cfg, inv_freq, overrides):
    """rope.rope_tables' construction over the given frequencies, with pinned entries."""
    cos_table, sin_table = [], []
    for position in range(cfg.context_cap):
        cos_table.append([round(math.cos(position * f) * R.ROPE_ONE) for f in inv_freq])
        sin_table.append([round(math.sin(position * f) * R.ROPE_ONE) for f in inv_freq])
    tables = {"cos": cos_table, "sin": sin_table}
    for name, position, column, value in overrides:
        tables[name][position][column] = value
    return cos_table, sin_table


def _fragile_entries(cfg, inv_freq):
    """Every cos/sin entry whose float lies within MARGIN_ULPS ulps of a Q2.30 midpoint."""
    out = []
    for position in range(cfg.context_cap):
        for column, f in enumerate(inv_freq):
            for name, fn in (("cos", math.cos), ("sin", math.sin)):
                value = fn(position * f)
                if value == 0.0:
                    continue
                scaled = value * R.ROPE_ONE
                distance = abs(scaled - math.floor(scaled) - 0.5)
                if distance < MARGIN_ULPS * math.ulp(value) * R.ROPE_ONE:
                    out.append([name, position, column, round(scaled)])
    return out


def record_qk_pin(cfg):
    G.fixture_model(cfg, {})  # runs and caches the calibration
    maxima, channel_peaks = G._CALIBRATION[cfg]
    inv_freq = _inv_freq(cfg)
    pin = {
        "config": _config_key(cfg),
        "calibration": {
            "maxima": {k: float(maxima[k]).hex() for k in sorted(maxima)},
            "qk_channel_peaks": {k: [[float(v).hex() for v in row] for row in channel_peaks[k]]
                                 for k in sorted(channel_peaks)},
        },
        "rope": {
            "margin_ulps": MARGIN_ULPS,
            "inv_freq": [f.hex() for f in inv_freq],
            "overrides": _fragile_entries(cfg, inv_freq),
        },
    }
    with open(QK_PIN, "w", newline="\n") as f:
        json.dump(pin, f, indent=1)
        f.write("\n")
    print(f"recorded {QK_PIN}: {len(maxima)} maxima, {len(channel_peaks)} peak layers, "
          f"{len(pin['rope']['overrides'])} RoPE overrides")


def load_qk_pin(cfg):
    """Prime the calibration cache from the pin; return the pinned RoPE tables."""
    with open(QK_PIN) as f:
        pin = json.load(f)
    if pin["config"] != _config_key(cfg):
        raise SystemExit(f"{QK_PIN} was recorded for {pin['config']}, not {_config_key(cfg)}; "
                         "rerun with --record-qk-pin")
    cal = pin["calibration"]
    maxima = {k: float.fromhex(v) for k, v in cal["maxima"].items()}
    peaks = {k: np.array([[float.fromhex(v) for v in row] for row in rows], dtype=np.float64)
             for k, rows in cal["qk_channel_peaks"].items()}
    G._CALIBRATION[cfg] = (maxima, peaks)
    inv_freq = [float.fromhex(v) for v in pin["rope"]["inv_freq"]]
    return _rope_tables(cfg, inv_freq, [tuple(e) for e in pin["rope"]["overrides"]])


def expected_hashes():
    """{fixture file: sha256} from the references' headers; every tag must agree."""
    out = {}
    for path in sorted(glob.glob(os.path.join(REFERENCE_DIR, "*.ref"))):
        with open(path) as f:
            fields = dict(p.split("=", 1) for p in f.readline().split()[1:] if "=" in p)
        name = fields["fixture"] + ".sslm"
        if out.setdefault(name, fields["sha256"]) != fields["sha256"]:
            raise SystemExit(f"references disagree on {name}'s hash ({path})")
    return out


def main(argv):
    args = argv[1:]
    record = "--record-qk-pin" in args
    args = [a for a in args if a != "--record-qk-pin"]
    if len(args) != 1:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    out_dir = args[0]
    os.makedirs(out_dir, exist_ok=True)
    expected = {} if record else expected_hashes()
    failed = []
    for name, head_dim, context_cap, strip, factors, with_schema in VARIANTS:
        start = time.time()
        cfg = config(head_dim, context_cap)
        rope_tables = None
        if name == QK_FIXTURE:
            if record:
                record_qk_pin(cfg)
            rope_tables = load_qk_pin(cfg)
        model = G.fixture_model(cfg, factors)
        if rope_tables is not None:
            model = P.with_rope_tables(model, rope_tables)
        if strip:
            model = G.strip_qk_norm(model)
        data, fingerprint = G.artifact(model, with_schema)
        path = os.path.join(out_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        sha = hashlib.sha256(data).hexdigest()
        print(f"wrote {path}: {len(data)} bytes, sha256={sha}, fingerprint={fingerprint}, "
              f"{time.time() - start:.1f} s")
        if name in expected and sha != expected[name]:
            failed.append(f"{name}: generated {sha}, the references were recorded against "
                          f"{expected[name]}")
    for line in failed:
        print(f"FIXTURE HASH MISMATCH {line}", file=sys.stderr)
    if failed:
        print("This host's float library builds different bytes; the paged-KV cells would fail on "
              "them. Report the mismatch; do not re-record the references on this host.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
