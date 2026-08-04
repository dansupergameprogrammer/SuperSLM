#!/usr/bin/env python3
"""Supplementary analysis for T-1695: reuses the site dumps
t1695_downproj_recoverable_ceiling.py already wrote (out/t1695_downproj_ceiling/*.sitedump)
-- no new engine runs. Computes RMSE with channel 609 EXCLUDED from the error
metric itself (as well as from the scale derivation for Arm D), to separate
"does excluding 609 from the shared scale help the OTHER 1535 channels" from
"what happens to channel 609's own reconstruction" -- the two are conflated
in the whole-row RMSE the main drive reports, and a real per-channel outlier
migration handles the outlier channel's own quantization separately (e.g. a
weight-side rescale), never simply "let it clip and eat the aggregate error."
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR
import shadow_layer_recompute as S
from sslm_artifact_reader import read_artifact
import t1695_downproj_recoverable_ceiling as T

OUT_DIR = T.OUT_DIR
artifact = read_artifact(str(KSR.MODEL_PATH))
OUTLIER = T.OUTLIER_CHANNEL

rows_out = []
for label, question, role in KSR.POPULATION:
    site_dump_path = os.path.join(OUT_DIR, f"{label}.sitedump")
    site_records = {r.site: r for r in S.load_site_dump(site_dump_path)}
    for row, layer_index in zip(T.TARGET_ROWS, T.TARGET_LAYER_INDICES):
        layer = artifact.layers[layer_index]
        def rec(name):
            return site_records[f"layer{layer_index}.{name}"]
        down_rec = rec("down_proj.requant")
        act_rec = rec("mlp_act")
        down_site = S.SiteFoldTriple(kind="down_proj", w_int8=layer.down_weight,
            identity=layer.down_fold[:,0], mult=layer.down_fold[:,1], shift=layer.down_fold[:,2])
        b, a_recon, c, d_recon, d_prime_d = T.compute_arms(
            down_rec.x_int, down_rec.d_prime, down_rec.codes, act_rec.codes, down_site)
        mask = np.ones(len(b), dtype=bool)
        mask[OUTLIER] = False
        err_a_excl = T.rmse(a_recon[mask], b[mask])
        err_d_excl = T.rmse(d_recon[mask], b[mask])
        rec_d_excl = (err_a_excl - err_d_excl) / err_a_excl if err_a_excl > 0 else float("nan")
        # channel 609's own reconstruction error, each arm, isolated
        err_a_609 = abs(a_recon[OUTLIER] - b[OUTLIER])
        err_d_609 = abs(d_recon[OUTLIER] - b[OUTLIER])
        rows_out.append(dict(label=label, role=role, row=row,
            err_a_excl609_rmse=err_a_excl, err_d_excl609_rmse=err_d_excl,
            recoverable_d_excl609=rec_d_excl,
            err_a_ch609=err_a_609, err_d_ch609=err_d_609))
    print(f"{label}: done", file=sys.stderr)

cols = list(rows_out[0].keys())
tsv_path = os.path.join(OUT_DIR, "results_excl609.tsv")
with open(tsv_path, "w", encoding="utf-8") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows_out:
        f.write("\t".join(str(r[c]) for c in cols) + "\n")
print(f"wrote {tsv_path}", file=sys.stderr)
