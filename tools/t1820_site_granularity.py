#!/usr/bin/env python3
"""T-1820 PHASE L -- per-site quantization granularity, from T-1795's own stage dumps.

Reads T-1795's `p*_stages.txt` read-only. Each stage vector in that dump is a COMMITTED int8
value: every element is an integer multiple of one CarriedScale, recovered here from the dump
itself. For each stage this reports

  g = q * sqrt(n/12) / ||v||

the relative L2 error a round-to-nearest quantization at that stage's own step imparts to a
vector of that stage's own magnitude -- the site's own granularity, in the units the
residual-stream measurements are already reported in.

DISCRIMINATION: g ranks the dumped sites by how coarsely each is quantized relative to its own
signal, which is the noise each injects per use. It does NOT establish how much of the layer
output's error each site contributes: T-1819 measured the staged-to-propagated ratio spanning
0.4x-5.0x across layers for one of these sites, so a rank in g is not a rank in contribution.
The denominator is the COMMITTED (already quantized) vector rather than an exact one, which
biases g by the same order as the quantization it measures -- stated, not corrected.

Read-only on every input.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1820_storage_floor import recover_step  # noqa: E402

STAGES = ["h_in", "normed_attn", "attn_branch", "attn_stream", "normed_mlp", "mlp_branch", "h_out"]


def read_stages(path: Path):
    lines = path.read_text(encoding="ascii").splitlines()
    n = int(lines[0])
    i = 1
    out = {}
    for _ in range(n):
        layer = int(lines[i].split()[1])
        i += 1
        vecs = {}
        for name in STAGES:
            parts = lines[i].split()
            vecs[name] = [float(x) for x in parts[1 : 1 + int(parts[0])]]
            i += 1
        out[layer] = vecs
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--prompts", default="p1,p2,p3")
    args = ap.parse_args(argv)
    d = Path(args.dump_dir)
    per = [read_stages(d / f"{p}_stages.txt") for p in args.prompts.split(",")]
    layers = sorted(per[0])

    print(__doc__.split("Read-only")[0].strip())
    print(f"\nprompts: {args.prompts}   layers: {layers}   (mean over prompts)")
    print("\nlayer | site          |     q       | max|code| | railed |  ||v||   |    g")
    agg = {}
    for L in layers:
        for name in STAGES:
            qs, cm, rl, nv, gs = [], [], [], [], []
            for P in per:
                v = P[L][name]
                q, integ, cmax = recover_step(v)
                n = len(v)
                nrm = math.sqrt(sum(x * x for x in v))
                railed = sum(1 for x in v if abs(round(x / q)) >= 127)
                g = q * math.sqrt(n / 12.0) / nrm
                qs.append(q); cm.append(cmax); rl.append(railed); nv.append(nrm); gs.append(g)
            m = lambda z: sum(z) / len(z)
            agg.setdefault(name, []).append(m(gs))
            print(f"{L:5d} | {name:13s} | {m(qs):11.6g} | {m(cm):9.1f} | {m(rl):6.1f} |"
                  f" {m(nv):8.2f} | {m(gs):.4f}")
        print()
    print("site granularity g, ranked by mean over the six checkpoint layers:")
    for name, gs in sorted(agg.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {name:13s} mean {sum(gs)/len(gs):.4f}   range {min(gs):.4f}-{max(gs):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
