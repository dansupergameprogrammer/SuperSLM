# T-1825 confirmation strike: the fold's site-1 discriminant, measured on site 1's
# own reachable domain -- the artifact's real int8 embedding rows.
#
# The design (S11 gate A6 value-level arm; S12 shape/platform row; S16 fold table)
# carries 401/1536 differing layer-0 attn_norm codes as the EXECUTED [M] discriminant
# for a bypassed site-1 refusal.  That number was measured on the T-1825 strike's
# synthetic row (seed 18250807, magnitudes ~4e3-1e7, one 20x outlier).
# S8.5 of the SAME design establishes site 1's reachable domain: EmbedEntry widens an
# already-int8 embedding row (forward_sites.cpp:561-565; header "widens each element to
# int64 (bound 127)"), so |x| <= 127 and D' <= 127.
#
# This probe re-runs the fold's own producer/consumer pair on the real population.
# Reproduce: python probe_t1825_confirm_domain.py   (python 3 + numpy)
import io
import zipfile

import numpy as np

NPZ = r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct-w8a8\arrays.npz"
N = 1536
K_CAP = 3  # design S6.1 row 1 / S8.5's own cell: site 1's norm consumer, S6.3b


def load_embed():
    z = zipfile.ZipFile(NPZ)
    with z.open("weight_embed.npy") as f:
        return np.load(io.BytesIO(f.read()))


def c22_codes(x, d):
    """C22 value semantics, half-away.  x: (R,N) int64, d: (R,1) int64."""
    a = np.abs(x) * 127
    q = (2 * a + d) // (2 * d)
    return np.where(x >= 0, q, -q)


def produce(rows, G, k_cap):
    """Design S4.1 steps 1-3, vectorised.  rows: (R,N) int64."""
    R = rows.shape[0]
    grp = np.abs(rows).reshape(R, N // G, G)
    Dg = np.maximum(grp.max(axis=2), 1)                    # (R, N/G), C20's >= 1 guard
    D = np.maximum(Dg.max(axis=1, keepdims=True), 1)       # (R,1) -- identical to the
                                                           # ungrouped reduction (S4.1 step 1)
    kg = np.zeros_like(Dg)
    for k in range(1, k_cap + 1):                          # largest k <= k_cap with Dg<<k <= D
        kg = np.where((Dg << k) <= D, k, kg)
    kg_full = np.repeat(kg, G, axis=1)                     # (R,N)
    codes = c22_codes(rows << kg_full, D)
    assert np.abs(codes).max() <= 127                      # in range by construction
    return codes, kg, kg_full, D


def rmsnorm_codes(v):
    """Layer-0 attn_norm at value level -- the fold-3 probe's own model, vectorised."""
    rms = np.sqrt((v * v).sum(axis=1, keepdims=True) / v.shape[1])
    y = v / rms
    m = np.abs(y).max(axis=1, keepdims=True)
    t = np.rint(y / m * (1 << 20))                         # int(round(.)) as the probe does
    a = np.abs(t) * 127
    d = 1 << 20
    q = (2 * a + d) // (2 * d)
    return np.where(t >= 0, q, -q)


def measure(rows, G, label, chunk=4096):
    misread_l, diff_l, refined_l = [], [], []
    for lo in range(0, rows.shape[0], chunk):
        blk = rows[lo:lo + chunk].astype(np.int64)
        codes, kg, kg_full, D = produce(blk, G, K_CAP)
        s_row = D / 127.0
        # Path A: the S4.2 consumer with k[] in hand.  Path B: the consumer reading the
        # committed SequenceLayerState, whose contract carries codes + ONE CarriedScale.
        vA = codes * s_row / (2.0 ** kg_full)
        vB = codes * s_row
        misread_l.append(((kg_full > 0) & (codes != 0)).sum(axis=1))
        diff_l.append((rmsnorm_codes(vA) != rmsnorm_codes(vB)).sum(axis=1))
        refined_l.append((kg > 0).sum(axis=1))
    misread = np.concatenate(misread_l)
    diff = np.concatenate(diff_l)
    refined = np.concatenate(refined_l)
    ngrp = N // G
    print(f"  {label}")
    print(f"    refined groups / {ngrp:<5}  mean {refined.mean():8.2f}   max {refined.max():5d}"
          f"   rows with 0 refined: {(refined == 0).sum():6d}/{rows.shape[0]}")
    print(f"    misread channels /1536  mean {misread.mean():8.2f}   max {misread.max():5d}"
          f"   p99 {int(np.percentile(misread, 99)):5d}")
    print(f"    attn_norm codes differing /1536  mean {diff.mean():8.2f}   max {diff.max():5d}"
          f"   p99 {int(np.percentile(diff, 99)):5d}")
    print(f"    rows reaching the design's 401/1536 discriminant: "
          f"{(diff >= 401).sum()}/{rows.shape[0]}")
    print(f"    rows at 0/1536 (indistinguishable from the adopted ruling): "
          f"{(diff == 0).sum()}/{rows.shape[0]}")
    return diff


def main():
    emb = load_embed()
    print(f"artifact: {NPZ}")
    print(f"embedding table: {emb.shape} {emb.dtype}  "
          f"(vocab x hidden_size; site 1's entire reachable input population)")
    rowmax = np.abs(emb.astype(np.int64)).max(axis=1)
    print(f"D' = row max-abs: min {rowmax.min()}  median {int(np.median(rowmax))}  "
          f"max {rowmax.max()}   rows at D'=127: {(rowmax == 127).sum()}")
    print()

    rows = emb
    print("== site 1's reachable domain: every one of the artifact's own embedding rows ==")
    for G in (32, 128, 256):
        measure(rows, G, f"G = {G}, k_cap = {K_CAP}  (design S6.1 row 1 / S8.5's own cell)")
    # The stage-B ceiling arm runs G at its floor.  Site 1's floor is 1 (S6.3d: 2 only at
    # site 3).  G=1 is the arm the kill line is drawn from.
    measure(rows, 1, "G = 1, k_cap = 3   (the stage-B ceiling arm's own setting, S11)")
    print()

    print("== the design's own cell, for comparison: the T-1825 synthetic row ==")
    import random
    rng = random.Random(18250807)
    x = []
    for _ in range(N):
        mag = int(4096 * (2.718281828 ** rng.gauss(0.0, 1.6)))
        sign = -1 if rng.random() < 0.5 else 1
        x.append(sign * max(mag, 1))
    x[940] = 20 * max(abs(v) for v in x)
    syn = np.array([x], dtype=np.int64)
    print(f"  |x| range on that row: {np.abs(syn).min()} .. {np.abs(syn).max()}   "
          f"(site 1's own domain bound is 127)")
    measure(syn, 32, "G = 32, k_cap = 3  -- the row the 401/1536 [M] figure was taken on")


if __name__ == "__main__":
    main()
