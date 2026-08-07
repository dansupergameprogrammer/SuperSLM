#!/usr/bin/env python3
"""T-1819 attack on claim 3: "four positionwise sub-blocks jointly account for only
8-24% of the layer's residual change, and their contributions are non-additive".

Two defects are tested by execution.

(1) STAGE MISMATCH.  T-1795 divides every sub-block figure by layer_delta_norm =
    ||h_out - h_in||, a residual-space quantity, but two of the numerators are NOT
    residual-space quantities:
        norm_attn = ||normed_attn_engine - normed_attn_exact||   (NORMED space)
        norm_mlp  = ||normed_mlp_engine  - normed_mlp_exact||    (NORMED space)
    This script recomputes norm_mlp PROPAGATED into h_out space, where it is
    commensurable with the joint arm.  norm_attn is not propagatable positionwise --
    normed_attn feeds attention, which reads every prior position's K/V -- so it has no
    h_out-space value at all under this method, and is reported as uncomputable.

(2) NON-ADDITIVITY.  T-1795 compares sum(||e_i||) against ||e_joint||.  Those are not
    the same quantity: for k mutually orthogonal error vectors of comparable size the
    sum of norms exceeds the norm of the sum by ~sqrt(k) by the triangle inequality
    alone, with no interaction of any kind.  This script builds the EXACT additive
    vector decomposition of the joint error,

        h_out_engine - h_out_joint = v_add_attn + v_add_attn_via_mlp + v_norm_mlp
                                     + v_mlp_only + v_add_mlp

    (verified by execution: the residual of that identity is reported), and reports
    ||sum v_i|| against sum ||v_i|| and the pairwise cosines between the v_i.  If the
    cosines are near zero, the "non-additivity" is the triangle inequality, not an
    interaction, and the budget DOES close -- as a vector sum, which is the only way an
    error budget ever closes.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

DEFAULT_MODEL = Path(
    r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
    r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
CHECKPOINT_LAYERS = (0, 1, 2, 9, 18, 27)
COMPONENTS = ("add_attn", "add_attn_via_mlp", "norm_mlp", "mlp_only", "add_mlp")


def read_vec(f):
    p = f.readline().split()
    n = int(p[0])
    v = [float(x) for x in p[1:1 + n]]
    assert len(v) == n
    return np.array(v, dtype=np.float64)


def read_stages(path: Path):
    layers = {}
    with open(path) as f:
        count = int(f.readline().strip())
        for _ in range(count):
            h = f.readline().split()
            assert h[0] == "layer"
            layers[int(h[1])] = {k: read_vec(f) for k in
                                 ("h_in", "normed_attn", "attn_branch", "attn_stream",
                                  "normed_mlp", "mlp_branch", "h_out")}
    return layers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    model = AutoModelForCausalLM.from_pretrained(
        a.model, local_files_only=True, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    model.to(dev)
    print("model.dtype:", model.dtype, "device:", dev,
          "allow_tf32:", torch.backends.cuda.matmul.allow_tf32 if dev == "cuda" else "n/a")

    def T(v):
        return torch.tensor(np.asarray(v, dtype=np.float64).tolist(),
                            dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)

    def npv(t):
        return t[0, 0].double().cpu().numpy()

    rows = []
    for pi, sp in enumerate(a.stages, start=1):
        layers = read_stages(Path(sp))
        for L in CHECKPOINT_LAYERS:
            e = layers[L]
            lm = model.model.layers[L]
            with torch.no_grad():
                # T-1795's own isolated-arm convention: exact norm evaluated on the
                # ENGINE's own real input at that stage.
                normed_attn_exact = npv(lm.input_layernorm(T(e["h_in"])))
                normed_mlp_exact_E = npv(lm.post_attention_layernorm(T(e["attn_stream"])))
                mlp_of_exact_norm_E = npv(lm.mlp(T(normed_mlp_exact_E)))
                mlp_of_engine_norm = npv(lm.mlp(T(e["normed_mlp"])))
                # joint arm: exact residual add, then exact norm + exact MLP on it
                attn_stream_X = e["h_in"] + e["attn_branch"]
                normed_mlp_X = npv(lm.post_attention_layernorm(T(attn_stream_X)))
                mlp_of_X = npv(lm.mlp(T(normed_mlp_X)))
                h_out_joint = attn_stream_X + mlp_of_X

            # ---- exact additive vector decomposition of the joint error ----
            v = {
                "add_attn":         e["attn_stream"] - attn_stream_X,
                "add_attn_via_mlp": mlp_of_exact_norm_E - mlp_of_X,
                "norm_mlp":         mlp_of_engine_norm - mlp_of_exact_norm_E,
                "mlp_only":         e["mlp_branch"] - mlp_of_engine_norm,
                "add_mlp":          e["h_out"] - e["attn_stream"] - e["mlp_branch"],
            }
            joint_vec = e["h_out"] - h_out_joint
            recon = sum(v.values())
            residual = float(np.linalg.norm(recon - joint_vec))

            delta = float(np.linalg.norm(e["h_out"] - e["h_in"]))
            norms = {k: float(np.linalg.norm(x)) for k, x in v.items()}
            cos = {f"{p}|{q}": float(v[p] @ v[q] /
                                     (np.linalg.norm(v[p]) * np.linalg.norm(v[q])))
                   for p, q in combinations(COMPONENTS, 2)}
            rows.append(dict(
                prompt=pi, layer=L, delta=delta,
                joint=float(np.linalg.norm(joint_vec)),
                identity_residual=residual,
                sum_of_norms=float(sum(norms.values())),
                rss=float(np.sqrt(sum(x ** 2 for x in norms.values()))),
                norms=norms, cos=cos,
                # T-1795's own stage-mixed figures, reproduced
                t1795_norm_attn=float(np.linalg.norm(e["normed_attn"] - normed_attn_exact)),
                t1795_norm_mlp=float(np.linalg.norm(e["normed_mlp"] - normed_mlp_exact_E)),
                t1795_mlp=float(np.linalg.norm(e["mlp_branch"] - mlp_of_exact_norm_E)),
                t1795_add_attn=float(np.linalg.norm(e["attn_stream"] - attn_stream_X)),
                t1795_add_mlp=float(np.linalg.norm(
                    e["h_out"] - e["attn_stream"] - e["mlp_branch"])),
                norm_mlp_propagated=norms["norm_mlp"],
            ))

    def agg(L, k):
        return float(np.mean([r[k] for r in rows if r["layer"] == L]))

    print()
    print("=== reproduction of T-1795 Phase B (mean of 3 prompts) ===")
    print(f"{'L':>3} {'norm_attn':>10} {'add_attn':>9} {'norm_mlp':>9} {'mlp':>9} "
          f"{'add_mlp':>9} {'joint':>9} {'delta':>9} {'|identity resid|':>17}")
    for L in CHECKPOINT_LAYERS:
        print(f"{L:>3} {agg(L,'t1795_norm_attn'):>10.3f} {agg(L,'t1795_add_attn'):>9.3f} "
              f"{agg(L,'t1795_norm_mlp'):>9.3f} {agg(L,'t1795_mlp'):>9.3f} "
              f"{agg(L,'t1795_add_mlp'):>9.3f} {agg(L,'joint'):>9.3f} {agg(L,'delta'):>9.2f} "
              f"{agg(L,'identity_residual'):>17.2e}")

    print()
    print("=== defect 1: norm_mlp staged (NORMED space) vs propagated (h_out space) ===")
    print(f"{'L':>3} {'staged':>9} {'%delta':>8} {'propagated':>11} {'%delta':>8} {'overstated':>11}")
    for L in CHECKPOINT_LAYERS:
        s, p, d = agg(L, 't1795_norm_mlp'), agg(L, 'norm_mlp_propagated'), agg(L, 'delta')
        print(f"{L:>3} {s:>9.3f} {100*s/d:>7.1f}% {p:>11.3f} {100*p/d:>7.1f}% {s/p:>10.1f}x")

    print()
    print("=== defect 2: the exact vector decomposition of the joint error ===")
    print(f"{'L':>3} " + " ".join(f"{c:>17}" for c in COMPONENTS) +
          f" {'sum||v||':>9} {'||sum v||':>10} {'ratio':>7} {'sqrt(k)':>8}")
    for L in CHECKPOINT_LAYERS:
        ns = {c: float(np.mean([r["norms"][c] for r in rows if r["layer"] == L]))
              for c in COMPONENTS}
        s = agg(L, 'sum_of_norms')
        j = agg(L, 'joint')
        print(f"{L:>3} " + " ".join(f"{ns[c]:>17.3f}" for c in COMPONENTS) +
              f" {s:>9.3f} {j:>10.3f} {s/j:>7.2f} {np.sqrt(len(COMPONENTS)):>8.2f}")

    print()
    print("=== pairwise cosines between the component error vectors (mean of 3 prompts) ===")
    keys = list(rows[0]["cos"].keys())
    print(f"{'L':>3} " + " ".join(f"{k:>26}" for k in keys))
    for L in CHECKPOINT_LAYERS:
        print(f"{L:>3} " + " ".join(
            f"{float(np.mean([r['cos'][k] for r in rows if r['layer']==L])):>26.3f}"
            for k in keys))

    print()
    print("=== root-sum-square (the correct addition for orthogonal errors) vs joint ===")
    print(f"{'L':>3} {'RSS':>10} {'joint':>10} {'RSS/joint':>10}")
    for L in CHECKPOINT_LAYERS:
        print(f"{L:>3} {agg(L,'rss'):>10.3f} {agg(L,'joint'):>10.3f} "
              f"{agg(L,'rss')/agg(L,'joint'):>10.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))
    print("written:", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
