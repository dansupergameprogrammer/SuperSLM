#!/usr/bin/env python3
r"""T-1835 -- the gates, run before any figure is read.

Four checks, each executed, none inherited:

  G1  ALL-TOGGLES-ON REPRODUCES THE INSTRUMENT IT EXTENDS, BIT-FOR-BIT.
      `--single-arm base` at batch 1 must produce byte-identical dumps to T-1809's
      committed arm C, and `--single-arm null` to its arm A. This is the null gate the
      commission names: if the eighteen toggles all set to "on" were not exactly T-1809's
      activation arm, every per-site figure below would be about a different construction.

  G2  A BATCHED GEMM's OUTPUT ROWS ARE INDEPENDENT.
      Cell A carries three extra copies of the `base` configuration at spread batch
      indices. All four must be bit-identical. Every within-run contrast in this ticket
      rests on this: that batch element b's arithmetic does not depend on what the other
      elements are computing. Asserted by execution rather than by reading the kernel.

  G3  THE ONLINE POOLED VECTOR IS T-1777's OWN POOLING.
      The capture computes each arm's pooled embedding in-process (full dumps for 56 arms
      would be ~78 GB). For the two arms that DO get full dumps, the in-process vector must
      equal `t1777_retrieval_report.pooled_final_layer_float` read back off the dump.

  G4  THE TWO CELLS' SEPARATION IS MEASURED, NOT ASSUMED.
      Cell A (B=59) and cell B (B=56) select different GEMM kernels, so they are different
      numeric realizations of the same 56 configurations. Their per-arm pooled separation
      is the run-to-run floor under every per-site figure, and it is reported here rather
      than discovered later.

Usage
    python tools\t1835_gate_null.py --out-dir out --t1777-tools <dir> \
        --t1809-out D:\SuperSLM\.worktrees\t1809-activation-interaction\out
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pooled(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    return list(z["labels"]), z["vectors"], z["fingerprints"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="the run root holding t1835_* outputs")
    ap.add_argument("--t1777-tools", required=True)
    ap.add_argument("--t1809-out", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    import t1777_retrieval_report as rr

    out = Path(args.out_dir)
    t1809 = Path(args.t1809_out)
    report = {}

    print("=== G1  all-toggles-on reproduces T-1809, bit-for-bit ===")
    g1 = {}
    for arm, gate_dir, ref_dir in (
            ("base", out / "t1835_gate_base" / "dumps" / "base", t1809 / "t1809_arm_c"),
            ("null", out / "t1835_gate_null" / "dumps" / "null", t1809 / "t1809_arm_a")):
        files = sorted(p.name for p in gate_dir.glob("*.float.bin"))
        same = [f for f in files if _sha(gate_dir / f) == _sha(ref_dir / f)]
        g1[arm] = {"n_files": len(files), "n_bit_identical": len(same),
                   "reference": ref_dir.name}
        verdict = "PASS" if len(same) == len(files) and files else "FAIL"
        print(f"  {arm:5s} vs {ref_dir.name}: {len(same)}/{len(files)} byte-identical "
              f"-> {verdict}")
        if len(same) != len(files):
            for f in files:
                if f not in same:
                    print(f"      MISMATCH {f}")
    report["G1"] = g1

    print("\n=== G2  batched-GEMM row independence ===")
    g2 = {}
    for cell in ("t1835_cellA", "t1835_cellB"):
        pooled_dir = out / cell / "pooled"
        if not (pooled_dir / "base.npz").exists():
            continue
        labels, base_v, _ = _pooled(pooled_dir / "base.npz")
        dups = sorted(p.stem for p in pooled_dir.glob("basedup*.npz"))
        rows = {}
        for d in dups:
            _, dv, _ = _pooled(pooled_dir / f"{d}.npz")
            rows[d] = bool(np.array_equal(base_v, dv))
            print(f"  {cell}: base vs {d}: bit-identical = {rows[d]}")
        if not dups:
            print(f"  {cell}: no duplicate-base arms in this run (expected for cell B)")
        g2[cell] = rows
    report["G2"] = g2

    print("\n=== G3  the online pooled vector is T-1777's own pooling ===")
    g3 = {}
    for cell in ("t1835_cellA", "t1835_cellB"):
        for arm in ("base", "null"):
            dump_dir = out / cell / "dumps" / arm
            npz = out / cell / "pooled" / f"{arm}.npz"
            if not dump_dir.exists() or not npz.exists():
                continue
            labels, vecs, _ = _pooled(npz)
            worst = 0.0
            n_exact = 0
            for i, lab in enumerate(labels):
                ref = rr.pooled_final_layer_float(dump_dir / f"{lab}.float.bin")
                d = float(np.abs(ref - vecs[i]).max())
                worst = max(worst, d)
                n_exact += int(d == 0.0)
            g3[f"{cell}/{arm}"] = {"n": len(labels), "n_exact": n_exact,
                                   "worst_abs": worst}
            print(f"  {cell}/{arm}: {n_exact}/{len(labels)} exact, worst |delta| = {worst:g}")
    report["G3"] = g3

    print("\n=== G4  cell A against cell B, per arm (the run-to-run floor) ===")
    a_dir = out / "t1835_cellA" / "pooled"
    b_dir = out / "t1835_cellB" / "pooled"
    g4 = {}
    if a_dir.exists() and b_dir.exists():
        names = sorted(p.stem for p in b_dir.glob("*.npz"))
        for name in names:
            if not (a_dir / f"{name}.npz").exists():
                continue
            la, va, _ = _pooled(a_dir / f"{name}.npz")
            lb, vb, _ = _pooled(b_dir / f"{name}.npz")
            assert la == lb, f"{name}: label order differs between cells"
            num = np.linalg.norm(va - vb, axis=1)
            den = np.maximum(np.linalg.norm(vb, axis=1), 1e-300)
            g4[name] = {"mean_rel": float((num / den).mean()),
                        "max_rel": float((num / den).max())}
        worst = sorted(g4.items(), key=lambda kv: -kv[1]["mean_rel"])
        print(f"  {len(g4)} arms compared; pooled relative separation between the cells")
        for name, v in worst[:5]:
            print(f"    {name:28s} mean {v['mean_rel']:.4f}  max {v['max_rel']:.4f}")
        print("    ...")
        for name, v in worst[-3:]:
            print(f"    {name:28s} mean {v['mean_rel']:.4f}  max {v['max_rel']:.4f}")
    report["G4"] = g4

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
