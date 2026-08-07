#!/usr/bin/env python3
"""T-1796: analyze the boundary-proximity dumps tools/t1796_boundary_probe.cpp writes.

Reads out/t1796/p{1,2,3}_boundary.txt (3 held-out prompts, self-checked bit-for-bit against
production at capture time -- see the probe's own stdout log, out/t1796/p{1,2,3}_run.log).
No float reference is read or required: every figure here is a property of the engine's own
integer arithmetic, already exact at capture time (the probe's SelfCheckRequantBoundary /
K-V-landing / probability-quantization self-checks all pass 0 failures on all 3 prompts,
over ~44.9M + 3072 + ~2.9k elements respectively per prompt).

Reports, per site (SITE rows) and per layer where the site name carries a layer number:
  - the boundary-distance distribution: n, mean, min, %within{1,5,10}%, and the 20-bin
    histogram (0..0.5 in steps of 0.025)
  - the perturbation-to-code-flip conversion curve: cumulative share of values within
    distance eps, for eps in {0.5%, 1%, 2%, 5%, 10%, 20%}, read directly off the histogram
  - concentration: per-channel and per-position coefficient of variation of the
    within-5%-of-boundary fraction, and the single most concentrated channel/position
"""
import sys
import re
import math
from collections import defaultdict

PROMPTS = ["p1", "p2", "p3"]
DUMP_DIR = "out/t1796"


def parse_dump(path):
    sites = {}  # site_name -> dict(n, sum_dist, min, w1, w5, w10, hist[20])
    channels = defaultdict(dict)  # site_name -> {channel_idx: (n, sum_dist, w5)}
    positions = defaultdict(dict)  # site_name -> {token_index: (n, sum_dist, w5)}
    with open(path) as f:
        for line in f:
            if line.startswith("SITE "):
                parts = line.split()
                name = parts[1]
                d = {}
                for p in parts[2:]:
                    if p.startswith("hist="):
                        d["hist"] = [int(x) for x in p[5:].split(",")]
                    else:
                        k, v = p.split("=", 1)
                        d[k] = float(v) if k == "mean" or k == "min" else int(v)
                sites[name] = d
            elif line.startswith("CHANNEL "):
                parts = line.split()
                name, ch = parts[1], int(parts[2])
                n = int(parts[3].split("=")[1])
                mean = float(parts[4].split("=")[1])
                w5 = int(parts[5].split("=")[1])
                channels[name][ch] = (n, mean, w5)
            elif line.startswith("POSITION "):
                parts = line.split()
                name, pos = parts[1], int(parts[2])
                n = int(parts[3].split("=")[1])
                mean = float(parts[4].split("=")[1])
                w5 = int(parts[5].split("=")[1])
                positions[name][pos] = (n, mean, w5)
    return sites, channels, positions


def normalize_site_name(name):
    # The probe's own RunLayerLoop call passed site_prefix="layer", and LayerSite()
    # (forward_sites.cpp) prepends "<prefix>.layer<N>." -- collapse the resulting
    # "layer.layerN.suffix" down to "layerN.suffix" for readability; leave
    # non-per-layer names (kv_landing_k, kv_landing_v, prob_quant) untouched.
    m = re.match(r"^layer\.(layer\d+\..+)$", name)
    return m.group(1) if m else name


def layer_of(name):
    m = re.match(r"^layer(\d+)\.", name)
    return int(m.group(1)) if m else None


def suffix_of(name):
    m = re.match(r"^layer\d+\.(.+)$", name)
    return m.group(1) if m else name


def cdf_at(hist, n, eps):
    # hist: 20 bins, width 0.025, covering [0,0.5). eps in [0,0.5].
    # RESOLVING-POWER FLOOR: the histogram bins at width 0.025 (2.5%), so any eps < 0.025
    # is indistinguishable from eps=0.025 by this histogram alone -- it floors to whichever
    # bin eps falls in. Callers that need eps < 2.5% MUST use the exact within{1,5,10}pct
    # counters (computed directly in the probe from an unbinned distance comparison, not
    # from this histogram) rather than this function. This function is only precise at eps
    # that land on a bin boundary (multiples of 0.025).
    if n == 0:
        return 0.0
    nbins = min(20, int(math.ceil(round(eps / 0.025, 6))))
    return sum(hist[:nbins]) / n


def coefficient_of_variation(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var) / mean


def main():
    all_sites = {}  # prompt -> {site: data}
    all_channels = {}
    all_positions = {}
    for p in PROMPTS:
        path = f"{DUMP_DIR}/{p}_boundary.txt"
        sites, channels, positions = parse_dump(path)
        all_sites[p] = {normalize_site_name(k): v for k, v in sites.items()}
        all_channels[p] = {normalize_site_name(k): v for k, v in channels.items()}
        all_positions[p] = {normalize_site_name(k): v for k, v in positions.items()}

    site_names = sorted(set().union(*[set(s.keys()) for s in all_sites.values()]))

    print("=" * 100)
    print("T-1796: boundary-distance distribution, per site x layer x prompt")
    print("distance in [0, 0.5]: 0 = exactly on the rounding/truncation boundary, 0.5 = bin center")
    print("=" * 100)

    # --- Pooled-by-suffix (across layers), per prompt, to see which SITE KIND runs hottest ---
    suffix_pool = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0, 0, 0, [0] * 20]))
    for p in PROMPTS:
        for name, d in all_sites[p].items():
            suf = suffix_of(name) if layer_of(name) is not None else name
            agg = suffix_pool[suf][p]
            agg[0] += d["n"]
            agg[1] += d["mean"] * d["n"]
            agg[2] += d["within1pct"]
            agg[3] += d["within5pct"]
            agg[4] += d["within10pct"]
            for i in range(20):
                agg[5][i] += d["hist"][i]

    print("\n--- Per site-kind, pooled across all measured layers, per prompt ---")
    print(f"{'site':24s} {'prompt':4s} {'n':>10s} {'mean':>8s} {'%<1%':>7s} {'%<5%':>7s} {'%<10%':>8s}")
    for suf in sorted(suffix_pool.keys()):
        for p in PROMPTS:
            if p not in suffix_pool[suf]:
                continue
            n, sm, w1, w5, w10, hist = suffix_pool[suf][p]
            if n == 0:
                continue
            print(f"{suf:24s} {p:4s} {n:10d} {sm/n:8.4f} {100*w1/n:6.2f}% {100*w5/n:6.2f}% {100*w10/n:7.2f}%")

    # --- Per layer, per site (full detail) -- resolving power via inter-prompt spread ---
    print("\n" + "=" * 100)
    print("Per layer x site, resolving power (mean %within5% across 3 held-out prompts, mean/std)")
    print("=" * 100)
    per_layer_site = defaultdict(dict)  # (layer,suffix) -> prompt -> pct_within5
    for p in PROMPTS:
        for name, d in all_sites[p].items():
            layer = layer_of(name)
            if layer is None:
                continue
            suf = suffix_of(name)
            pct5 = 100.0 * d["within5pct"] / d["n"] if d["n"] else 0.0
            per_layer_site[(layer, suf)][p] = pct5

    print(f"{'layer':>5s} {'site':24s} {'p1_%<5%':>9s} {'p2_%<5%':>9s} {'p3_%<5%':>9s} {'mean':>7s} {'std':>7s} {'mean/std':>9s}")
    checkpoint_layers = {0, 1, 2, 9, 18, 27}
    rows = sorted(per_layer_site.keys())
    for (layer, suf) in rows:
        vals = [per_layer_site[(layer, suf)].get(p, None) for p in PROMPTS]
        if any(v is None for v in vals):
            continue
        mean = sum(vals) / 3
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / 3)
        ratio = mean / std if std > 1e-9 else float("inf")
        flag = " *checkpoint" if layer in checkpoint_layers else ""
        print(f"{layer:5d} {suf:24s} {vals[0]:8.3f}% {vals[1]:8.3f}% {vals[2]:8.3f}% {mean:6.3f}% {std:6.4f} {ratio:8.2f}{flag}")

    # --- Conversion curve: perturbation size eps -> fraction of codes that would flip ---
    # eps in {0.01, 0.05, 0.10} use the EXACT within{1,5,10}pct counters (unbinned, computed
    # directly in the probe); eps=0.20 lands exactly on a histogram bin boundary (0.20/0.025=8)
    # so the histogram-derived figure there is also exact. No eps below 0.025 other than the
    # three exact points is reported, because the histogram alone cannot resolve it (see
    # cdf_at's own comment) -- reporting one would silently overclaim resolution this
    # measurement does not have, exactly the failure this campaign's own standards flag.
    print("\n" + "=" * 100)
    print("Conversion curve: perturbation size (as a fraction of bin width) -> share of codes flipped")
    print("Pooled across ALL chain-funnel sites and ALL measured layers, per prompt. eps=0.01/0.05/0.10")
    print("are EXACT (unbinned) counters; eps=0.20 is exact via an aligned histogram bin boundary.")
    print("No eps below 0.025 other than these three exact points is reported (resolving-power floor")
    print("of the 0.025-wide histogram would otherwise be silently overclaimed).")
    print("=" * 100)
    eps_exact = [0.01, 0.05, 0.10]
    eps_hist = [0.20]
    header = f"{'prompt':6s}" + "".join(f"{'eps='+str(e):>10s}" for e in eps_exact + eps_hist)
    print(header)
    for p in PROMPTS:
        total_n = 0
        total_w = {0.01: 0, 0.05: 0, 0.10: 0}
        total_hist = [0] * 20
        for name, d in all_sites[p].items():
            if layer_of(name) is None:
                continue
            total_n += d["n"]
            total_w[0.01] += d["within1pct"]
            total_w[0.05] += d["within5pct"]
            total_w[0.10] += d["within10pct"]
            for i in range(20):
                total_hist[i] += d["hist"][i]
        row = f"{p:6s}"
        for e in eps_exact:
            row += f"{100*total_w[e]/total_n:9.3f}%"
        for e in eps_hist:
            row += f"{100*cdf_at(total_hist, total_n, e):9.3f}%"
        print(row)

    # Same curve, K/V landing and probability quantization separately (different geometries).
    for site_key, label in [("kv_landing_k", "K landing"), ("kv_landing_v", "V landing"),
                             ("prob_quant", "probability quantization (floor geometry)")]:
        print(f"\n{label} conversion curve:")
        print(header)
        for p in PROMPTS:
            d = all_sites[p].get(site_key)
            if d is None:
                continue
            row = f"{p:6s}"
            for e in eps_exact:
                key = {0.01: "within1pct", 0.05: "within5pct", 0.10: "within10pct"}[e]
                row += f"{100*d[key]/d['n']:9.3f}%"
            for e in eps_hist:
                row += f"{100*cdf_at(d['hist'], d['n'], e):9.3f}%"
            print(row)

    # --- Concentration: per-channel and per-position CV of the within-5% fraction ---
    print("\n" + "=" * 100)
    print("Concentration: coefficient of variation (CV) of the within-5%-of-boundary fraction,")
    print("across channels and across token positions, per site (pooled p1-p3 where matched)")
    print("CV near 0 = uniform across channels/positions (no exploitable structure).")
    print("CV large, with one outlier far above the rest = concentrated (an exploitable structure).")
    print("=" * 100)
    for p in PROMPTS:
        print(f"\n--- {p} ---")
        print(f"{'site':24s} {'n_channels':>10s} {'chan_CV':>8s} {'max_chan_frac':>13s} {'max_chan_id':>11s} "
              f"{'n_positions':>11s} {'pos_CV':>8s} {'max_pos_frac':>12s} {'max_pos_id':>10s}")
        for name in sorted(all_channels[p].keys()):
            ch = all_channels[p][name]
            fracs = [(w5 / n if n else 0.0, cidx) for cidx, (n, mean, w5) in ch.items() if n >= 5]
            if not fracs:
                continue
            cv = coefficient_of_variation([f for f, _ in fracs])
            max_frac, max_id = max(fracs)
            pos = all_positions[p].get(name, {})
            pfracs = [(w5 / n if n else 0.0, pidx) for pidx, (n, mean, w5) in pos.items() if n >= 5]
            if pfracs:
                pcv = coefficient_of_variation([f for f, _ in pfracs])
                pmax_frac, pmax_id = max(pfracs)
            else:
                pcv, pmax_frac, pmax_id = 0.0, 0.0, -1
            print(f"{name:24s} {len(fracs):10d} {cv:8.3f} {max_frac:12.3f}% {max_id:11d} "
                  f"{len(pfracs):11d} {pcv:8.3f} {pmax_frac:11.3f}% {pmax_id:10d}")

    print("\n" + "=" * 100)
    print("Clamp/saturation frame -- these are DIFFERENT boundary phenomena (a clamp is a boundary")
    print("with no bin on the far side), placed here for scale comparison, not re-measured:")
    print("  - T-1758/D-SLM1107-1112: one channel saturating at the residual site, holding")
    print("    36.03%/27.06% of position-0 reference energy at layers 25/26.")
    print("  - T-1781/D-SLM1226-1229: RoPE's clamp engaged 239/41,244,672 opportunities")
    print("    (5.80e-6); RMSNorm's engaged 0/23,073.")
    for p in PROMPTS:
        neg_k = None
        for line in open(f"{DUMP_DIR}/{p}_boundary.txt"):
            if line.startswith("META kv_landing_negative_k_total"):
                neg_k = int(line.strip().split("=")[1])
        d = all_sites[p].get("kv_landing_k")
        n_k = d["n"] if d else 0
        print(f"  {p}: K/V landing exact-left-shift (k<0, no rounding boundary at all) = "
              f"{neg_k} of {neg_k + 2*n_k} landing calls (K+V)")


if __name__ == "__main__":
    main()
