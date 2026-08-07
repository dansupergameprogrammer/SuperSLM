#!/usr/bin/env python3
"""T-1797 E0 analysis: is the engine's residual-stream drift against the float32 reference
systematic across inputs (correctable offline), and how fast does it change over positions
(a live-correction rate bound)?

Inputs: out/t1797/p{i}_eng.bin and p{i}_float.bin, raw float32 [pos][state 0..28][1536]
(t1797_multipos_probe.cpp / t1797_float_multipos.py). All 15 prompts share one chat-template
system prefix, so their leading positions carry IDENTICAL states; those positions are
detected empirically and excluded from every cross-prompt statistic (they would report
perfect consistency trivially).

Held-out discipline: every fitted correction (bias, diagonal gain, ridge linear map) is fit
on a subset of PROMPTS and evaluated on the held-out prompts; positions within a prompt are
correlated and are never the split unit. Resolving power = spread over repeated random
splits.

Models, per state s (0=embedding output, k=output of layer k-1), drift d = eng - flt:
  bias    : d ~= mu                      (foldable into a bias term)
  diag    : eng ~= g (.) flt             (foldable into next layer's weights, diagonal)
  ridge-W : d ~= W flt, ridge, full rank (foldable into next layer's weights, general
            linear; rank-r truncations show the sample-efficiency curve)
Reported as held-out fraction of drift ENERGY removed: R = 1 - E||d - dhat||^2 / E||d||^2.

Rate over positions, per state: cos(d_t, d_{t+delta}) for delta in {1,2,4,8,16}, within
prompt, body positions only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("out/t1797")
N_PROMPTS = 15
N_STATES = 29
HID = 1536
RNG = np.random.default_rng(0x1797)
N_SPLITS = 6
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0]  # scaled by tr(F)/dim


def load(pid: str, side: str) -> np.ndarray:
    meta = {}
    for line in (OUT / f"{pid}_{side}.meta").read_text().splitlines():
        k, v = line.split()
        meta[k] = v
    n_pos, n_states, hid = int(meta["positions"]), int(meta["states"]), int(meta["hidden"])
    arr = np.fromfile(OUT / f"{pid}_{side}.bin", dtype=np.float32)
    return arr.reshape(n_pos, n_states, hid).astype(np.float64)


def main() -> int:
    eng = [load(f"p{i}", "eng") for i in range(1, N_PROMPTS + 1)]
    flt = [load(f"p{i}", "float") for i in range(1, N_PROMPTS + 1)]
    for i in range(N_PROMPTS):
        assert eng[i].shape == flt[i].shape, i

    # --- empirical shared-prefix length (identical embedding-state rows across prompts) ---
    min_pos = min(e.shape[0] for e in eng)
    l_pre = 0
    for t in range(min_pos):
        same = all(
            np.array_equal(eng[i][t], eng[0][t]) and np.array_equal(flt[i][t], flt[0][t])
            for i in range(1, N_PROMPTS)
        )
        if not same:
            l_pre = t
            break
    print(f"shared prefix positions (excluded from cross-prompt stats): {l_pre}")

    # --- assemble per-state sample matrices over (prompt, body position) ---
    # samples[s] = (F, E, prompt_idx): float vec, engine vec
    F_all, E_all, P_all, is_last = [], [], [], []
    for i in range(N_PROMPTS):
        n_pos = eng[i].shape[0]
        for t in range(l_pre, n_pos):
            F_all.append(flt[i][t])   # (29, HID)
            E_all.append(eng[i][t])
            P_all.append(i)
            is_last.append(t == n_pos - 1)
    F_all = np.stack(F_all)  # (n, 29, HID)
    E_all = np.stack(E_all)
    P_all = np.asarray(P_all)
    is_last = np.asarray(is_last)
    n = F_all.shape[0]
    print(f"samples (prompt, position): {n} over {N_PROMPTS} prompts")

    D_all = E_all - F_all

    results = {}
    for s in range(N_STATES):
        F = F_all[:, s, :]
        E = E_all[:, s, :]
        D = D_all[:, s, :]
        d_energy = np.sum(D * D, axis=1)
        f_energy = np.sum(F * F, axis=1)
        rel_l2 = np.sqrt(d_energy / np.maximum(f_energy, 1e-30))
        cos = np.sum(E * F, axis=1) / np.maximum(
            np.linalg.norm(E, axis=1) * np.linalg.norm(F, axis=1), 1e-30)

        r_bias, r_diag, r_ridge, r_rank = [], [], [], {1: [], 8: [], 64: []}
        for _ in range(N_SPLITS):
            perm = RNG.permutation(N_PROMPTS)
            train_p, test_p = perm[: N_PROMPTS // 2], perm[N_PROMPTS // 2:]
            tr = np.isin(P_all, train_p)
            te = np.isin(P_all, test_p)
            base = np.sum(D[te] * D[te])

            mu = D[tr].mean(axis=0)
            r_bias.append(1.0 - np.sum((D[te] - mu) ** 2) / base)

            g = np.sum(E[tr] * F[tr], axis=0) / np.maximum(np.sum(F[tr] * F[tr], axis=0), 1e-30)
            r_diag.append(1.0 - np.sum((E[te] - g * F[te]) ** 2) / base)

            # ridge linear map d ~= W f ; lambda selected by inner split of train prompts
            Ftr, Dtr = F[tr], D[tr]
            Fc = Ftr.T @ Ftr
            C = Dtr.T @ Ftr
            scale = np.trace(Fc) / HID
            # inner selection
            inner_train = np.isin(P_all, train_p[: len(train_p) // 2])
            inner_val = np.isin(P_all, train_p[len(train_p) // 2:])
            Fi = F[inner_train].T @ F[inner_train]
            Ci = D[inner_train].T @ F[inner_train]
            best_lam, best_score = None, -np.inf
            for lam in LAMBDAS:
                Wi = np.linalg.solve(Fi + lam * scale * np.eye(HID), Ci.T).T
                pred = F[inner_val] @ Wi.T
                score = 1.0 - np.sum((D[inner_val] - pred) ** 2) / np.sum(D[inner_val] ** 2)
                if score > best_score:
                    best_score, best_lam = score, lam
            W = np.linalg.solve(Fc + best_lam * scale * np.eye(HID), C.T).T
            pred = F[te] @ W.T
            r_ridge.append(1.0 - np.sum((D[te] - pred) ** 2) / base)
            # rank truncations
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            for r in r_rank:
                Wr = (U[:, :r] * S[:r]) @ Vt[:r]
                predr = F[te] @ Wr.T
                r_rank[r].append(1.0 - np.sum((D[te] - predr) ** 2) / base)

        results[s] = {
            "rel_l2_mean": float(rel_l2.mean()),
            "cos_mean": float(cos.mean()),
            "rel_l2_last_only": float(rel_l2[is_last].mean()),
            "cos_last_only": float(cos[is_last].mean()),
            "R_bias": [float(np.mean(r_bias)), float(np.std(r_bias))],
            "R_diag": [float(np.mean(r_diag)), float(np.std(r_diag))],
            "R_ridge": [float(np.mean(r_ridge)), float(np.std(r_ridge))],
            "R_rank1": [float(np.mean(r_rank[1])), float(np.std(r_rank[1]))],
            "R_rank8": [float(np.mean(r_rank[8])), float(np.std(r_rank[8]))],
            "R_rank64": [float(np.mean(r_rank[64])), float(np.std(r_rank[64]))],
        }

    # --- drift direction rate over positions (within prompt) ---
    deltas = [1, 2, 4, 8, 16]
    rate = {s: {dl: [] for dl in deltas} for s in range(N_STATES)}
    for i in range(N_PROMPTS):
        n_pos = eng[i].shape[0]
        D = eng[i] - flt[i]  # (pos, 29, HID)
        for s in range(N_STATES):
            for dl in deltas:
                for t in range(l_pre, n_pos - dl):
                    a, b = D[t, s], D[t + dl, s]
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    if na > 1e-30 and nb > 1e-30:
                        rate[s][dl].append(float(a @ b / (na * nb)))
    for s in range(N_STATES):
        results[s]["drift_autocos"] = {
            str(dl): [float(np.mean(v)), float(np.std(v)), len(v)]
            for dl, v in rate[s].items()
        }

    # --- channel structure of the drift ---
    for s in range(N_STATES):
        D = D_all[:, s, :]
        ch_energy = np.sum(D * D, axis=0)
        total = ch_energy.sum()
        top = np.argsort(ch_energy)[::-1]
        # stability: split prompts, correlate per-channel energy profiles
        stab = []
        for _ in range(N_SPLITS):
            perm = RNG.permutation(N_PROMPTS)
            a = np.isin(P_all, perm[: N_PROMPTS // 2])
            b = np.isin(P_all, perm[N_PROMPTS // 2:])
            ea = np.sum(D[a] * D[a], axis=0)
            eb = np.sum(D[b] * D[b], axis=0)
            stab.append(float(np.corrcoef(ea, eb)[0, 1]))
        results[s]["chan_top1_share"] = float(ch_energy[top[0]] / total)
        results[s]["chan_top16_share"] = float(ch_energy[top[:16]].sum() / total)
        results[s]["chan_top4_idx"] = [int(c) for c in top[:4]]
        results[s]["chan_profile_stability"] = [float(np.mean(stab)), float(np.std(stab))]

    with open(OUT / "e0_results.json", "w") as f:
        json.dump(results, f, indent=1)

    # --- report ---
    hdr = (f"{'st':>3} {'relL2':>6} {'cos':>6} | {'R_bias':>12} {'R_diag':>12} "
           f"{'R_ridge':>12} {'R_rk8':>12} | {'acos1':>6} {'acos8':>6} | "
           f"{'top1%':>5} {'top16%':>6} {'stab':>5}")
    print(hdr)
    for s in range(N_STATES):
        r = results[s]
        print(f"{s:>3} {r['rel_l2_mean']:6.3f} {r['cos_mean']:6.3f} | "
              f"{r['R_bias'][0]:5.3f}+-{r['R_bias'][1]:5.3f} "
              f"{r['R_diag'][0]:5.3f}+-{r['R_diag'][1]:5.3f} "
              f"{r['R_ridge'][0]:5.3f}+-{r['R_ridge'][1]:5.3f} "
              f"{r['R_rank8'][0]:5.3f}+-{r['R_rank8'][1]:5.3f} | "
              f"{r['drift_autocos']['1'][0]:6.3f} {r['drift_autocos']['8'][0]:6.3f} | "
              f"{100*r['chan_top1_share']:5.1f} {100*r['chan_top16_share']:6.1f} "
              f"{r['chan_profile_stability'][0]:5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
