#!/usr/bin/env python3
"""T-1705 -- tests whether the down_proj.requant output-boundary channel
invariance T-1704 found (0/28 wander at the TRUE global (position,channel)
cell, all 9 campaign prompts) is a property of the MODEL, or an artifact of
the fact that all 9 campaign prompts share one chat-template preamble
(system prompt + `<|im_start|>user\\n` opening).

Varies the preamble across 4 conditions -- baseline (campaign's own system
prompt), a different short system prompt, an absent system prompt, and a
substantially longer system prompt -- and re-runs T-1704's own all-position
capture + true-global-cell decomposition under each, then asks three
questions:
  1. Does channel/position wander (per T-1704 Sec.1's own method) reproduce
     under the baseline, and how does it change under the other 3?
  2. Is the WINNING CHANNEL the same channel across all 4 conditions, per
     layer and globally? (the decisive question)
  3. Does the winning POSITION track "last token of the preamble" as
     preamble length changes (the sharpest position-anchored-vs-model-
     anchored discriminator)?

Reuses `kv_saturation_report.POPULATION` (the same 9 questions, same driver,
same all-position capture mechanism T-1704 built and proved invariant) --
only the SYSTEM_PROMPT half of `build_prompt` varies. Reuses
`t1704_fast_allpos_reader.iter_chain_records_fast` (validated against the
canonical reader by `test_t1704_fast_allpos_reader.py`) for all decoding.

Usage: python tools\\t1705_preamble_variation.py [--keep-dumps]
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402
import t1704_fast_allpos_reader as FAST  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DRIVER_EXE = os.path.join(REPO_ROOT, "out", "sslm_layer_trace.exe")
OUT_ROOT = os.path.join(REPO_ROOT, "out", "t1705_preamble_variation")

NUM_LAYERS = 28
SITES = ["mlp_act", "down_proj.requant"]
NUM_CHANNELS = {"mlp_act": 8960, "down_proj.requant": 1536}
LOCUS_LAYERS = set(range(21, 27))

# --- The 4 conditions. Content and token length both vary; baseline is the
# campaign's own system prompt (must reproduce T-1704's 0/28 / 16/28). ---
CONDITIONS = [
    ("baseline", KSR.SYSTEM_PROMPT),
    ("different_short", "Reply briefly."),
    ("absent", None),
    ("longer", (
        "You are a rigorous technical assistant embedded in a software engineering "
        "workflow. When answering, be precise, cite the exact reasoning that leads "
        "to your conclusion, avoid speculation presented as fact, and prefer short "
        "correct answers over long uncertain ones. If a question has a single "
        "well-defined answer, state it directly before any elaboration."
    )),
]


def build_prompt(question: str, system_prompt) -> str:
    if system_prompt is None:
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def magnitude(codes: np.ndarray, d_prime: int) -> np.ndarray:
    return np.abs(codes.astype(np.float64)) * d_prime / 127.0


def run_driver(prompt: str, dump_path: str, site_path: str | None, all_positions: bool,
                timeout=600):
    cmd = [str(DRIVER_EXE), str(KSR.MODEL_PATH), str(KSR.TOKENIZER_PATH), prompt,
           "--dump", dump_path]
    if site_path is not None:
        cmd += ["--site-dump", site_path]
        if all_positions:
            cmd += ["--site-dump-all-positions"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"driver failed (exit {proc.returncode})\n"
                            f"cmd={cmd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    if "self_check: production and manual-replay paths agree" not in proc.stdout:
        raise RuntimeError(f"self-check line missing from stdout:\n{proc.stdout}")
    return proc


def analyze_prompt_full(path: str):
    """One streaming pass over one all-position site-dump. Returns, per site:
      layer_best: {(site,layer): (position,channel,value)} -- true global
        argmax cell over every captured (position,channel) pair.
      global_best: {site: (layer,position,channel,value)} -- single largest
        cell across the whole 28-layer walk.
      channel_max: {site: ndarray[NUM_CHANNELS]} -- elementwise max|X_c| over
        every captured (layer,position) -- the all-position analog of
        T-1698/T-1700's own per-channel overall-max statistic (which was
        restricted to the last position only).
      position_token_id: this prompt's own token sequence (position -> id).
      max_position: last captured position.
    """
    layer_best = {(s, l): (-1, -1, -1.0) for s in SITES for l in range(NUM_LAYERS)}
    global_best = {}
    channel_max = {s: np.zeros(NUM_CHANNELS[s], dtype=np.float64) for s in SITES}
    position_token_id = {}
    max_position = -1

    for rec in FAST.iter_chain_records_fast(path):
        if "." not in rec.site or not rec.site.startswith("layer"):
            continue
        layer_str, _, name = rec.site.partition(".")
        if name not in SITES:
            continue
        try:
            layer = int(layer_str[len("layer"):])
        except ValueError:
            continue

        position_token_id[rec.position] = rec.token_id
        if rec.position > max_position:
            max_position = rec.position

        mag = magnitude(rec.codes, rec.d_prime)
        np.maximum(channel_max[name], mag, out=channel_max[name])
        c = int(np.argmax(mag))
        v = float(mag[c])

        key = (name, layer)
        _, _, best_v = layer_best[key]
        if v > best_v:
            layer_best[key] = (rec.position, c, v)

        gb = global_best.get(name)
        if gb is None or v > gb[3]:
            global_best[name] = (layer, rec.position, c, v)

    return {
        "layer_best": layer_best,
        "global_best": global_best,
        "channel_max": channel_max,
        "position_token_id": position_token_id,
        "max_position": max_position,
    }


def shared_prefix_len(seqs: dict) -> int:
    min_len = min(len(s) for s in seqs.values())
    n = 0
    for i in range(min_len):
        vals = {seqs[label][i] for label in seqs}
        if len(vals) == 1:
            n = i + 1
        else:
            break
    return n


def run_condition(name: str, system_prompt, keep_dumps: bool) -> dict:
    cond_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(cond_dir, exist_ok=True)

    print(f"\n{'=' * 78}\nCONDITION: {name}  system_prompt={system_prompt!r}\n{'=' * 78}")

    results = {}
    site_paths = {}
    for label, question, role in KSR.POPULATION:
        prompt = build_prompt(question, system_prompt)
        dump_path = os.path.join(cond_dir, f"{label}.dump")
        site_path = os.path.join(cond_dir, f"{label}.sitedump")
        run_driver(prompt, dump_path, site_path, all_positions=True)
        size_mb = os.path.getsize(site_path) / 1e6
        print(f"  captured {label!r} role={role} ({size_mb:.1f} MB)")
        site_paths[label] = site_path
        results[label] = analyze_prompt_full(site_path)

    # --- Guard 2: capture-off vs all-position decode byte-identity, first 2 prompts. ---
    invariance_ok = True
    for label, question, role in KSR.POPULATION[:2]:
        prompt = build_prompt(question, system_prompt)
        off_dump = os.path.join(cond_dir, f"{label}.off.dump")
        all_dump = os.path.join(cond_dir, f"{label}.allpos_check.dump")
        run_driver(prompt, off_dump, None, all_positions=False)
        run_driver(prompt, all_dump, os.path.join(cond_dir, f"{label}.allpos_check.sitedump"),
                   all_positions=True)
        ident = filecmp.cmp(off_dump, all_dump, shallow=False)
        invariance_ok = invariance_ok and ident
        print(f"  invariance check {label!r}: capture-off .dump == all-position .dump: {ident}")
        for p in (off_dump, all_dump, os.path.join(cond_dir, f"{label}.allpos_check.sitedump")):
            os.remove(p)
    if not invariance_ok:
        raise RuntimeError(f"condition {name!r}: capture invariance check FAILED -- stop")

    # --- Shared token-id prefix (this condition's own preamble boundary). ---
    seqs = {}
    for label, _q, _r in KSR.POPULATION:
        r = results[label]
        seqs[label] = [r["position_token_id"].get(p) for p in range(r["max_position"] + 1)]
    preamble_len = shared_prefix_len(seqs)
    print(f"  shared token-id prefix (this condition's preamble): {preamble_len} positions "
          f"(0..{preamble_len - 1})")

    # --- Section 1 method (T-1704): per (site,layer), true global decomposition. ---
    per_site = {}
    for site_name in SITES:
        position_wander = 0
        channel_wander = 0
        layer_channel = {}  # layer -> set of winning channels across 9 prompts
        layer_position_rel = {}  # layer -> set of (winning_position - (preamble_len-1))
        for layer in range(NUM_LAYERS):
            positions = set()
            channels = set()
            rel_positions = set()
            for label, _q, _r in KSR.POPULATION:
                pos, ch, val = results[label]["layer_best"][(site_name, layer)]
                positions.add(pos)
                channels.add(ch)
                rel_positions.add(pos - (preamble_len - 1))
            if len(positions) > 1:
                position_wander += 1
            if len(channels) > 1:
                channel_wander += 1
            layer_channel[layer] = channels
            layer_position_rel[layer] = rel_positions
        per_site[site_name] = {
            "position_wander_layers": position_wander,
            "channel_wander_layers": channel_wander,
            "total_layers": NUM_LAYERS,
            "layer_channel": {l: sorted(s) for l, s in layer_channel.items()},
            "layer_position_rel_to_preamble_last": {l: sorted(s) for l, s in layer_position_rel.items()},
        }
        print(f"  {site_name}: channel wanders {channel_wander}/{NUM_LAYERS} layers, "
              f"position wanders {position_wander}/{NUM_LAYERS} layers")

    # --- Global top cell per site, and whether it traces to the preamble boundary. ---
    global_top = {}
    for site_name in SITES:
        best = None
        for label, _q, _r in KSR.POPULATION:
            gb = results[label]["global_best"].get(site_name)
            if gb is None:
                continue
            layer, pos, ch, val = gb
            if best is None or val > best[4]:
                best = (label, layer, pos, ch, val)
        label, layer, pos, ch, val = best
        rel = pos - (preamble_len - 1)
        global_top[site_name] = {
            "label": label, "layer": layer, "position": pos, "channel": ch, "value": val,
            "in_preamble": pos < preamble_len,
            "is_preamble_last_token": pos == preamble_len - 1,
            "relative_to_preamble_last": rel,
        }
        print(f"  {site_name} GLOBAL top cell: prompt={label} layer={layer} position={pos} "
              f"(preamble_len={preamble_len}, rel_to_preamble_last={rel:+d}) channel={ch} value={val:.6g}")

    # --- rank0/median, all-position analog of the published last-position statistic. ---
    rank_stats = {}
    for site_name in SITES:
        overall = np.zeros(NUM_CHANNELS[site_name], dtype=np.float64)
        for label, _q, _r in KSR.POPULATION:
            np.maximum(overall, results[label]["channel_max"][site_name], out=overall)
        order = np.argsort(-overall)
        rank0 = float(overall[order[0]])
        median = float(np.median(overall))
        rank_stats[site_name] = {
            "rank0_channel": int(order[0]), "rank0_value": rank0,
            "median": median, "rank0_over_median": rank0 / median,
        }
        print(f"  {site_name}: rank0={rank0:.6g} (ch {int(order[0])}) median={median:.6g} "
              f"rank0/median={rank0 / median:.4f}")

    out = {
        "condition": name,
        "system_prompt": system_prompt,
        "preamble_len": preamble_len,
        "invariance_ok": invariance_ok,
        "per_site": per_site,
        "global_top": global_top,
        "rank_stats": rank_stats,
    }

    if not keep_dumps:
        shutil.rmtree(cond_dir)
        print(f"  deleted {cond_dir} (stats extracted)")

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-dumps", action="store_true")
    ap.add_argument("--out-json", default=os.path.join(OUT_ROOT, "results.json"))
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    if not os.path.isfile(DRIVER_EXE):
        print(f"FAILED: {DRIVER_EXE} not built", file=sys.stderr)
        return 1

    all_results = {}
    for name, sysprompt in CONDITIONS:
        all_results[name] = run_condition(name, sysprompt, args.keep_dumps)

    # --- Guard 1: baseline must reproduce T-1704's 0/28 (down_proj.requant) / 16/28 (mlp_act). ---
    base = all_results["baseline"]["per_site"]
    baseline_ok = (base["down_proj.requant"]["channel_wander_layers"] == 0
                   and base["mlp_act"]["channel_wander_layers"] == 16)
    print(f"\nGUARD 1 (baseline reproduces T-1704): down_proj.requant channel_wander="
          f"{base['down_proj.requant']['channel_wander_layers']} (expect 0), mlp_act channel_wander="
          f"{base['mlp_act']['channel_wander_layers']} (expect 16) -> {'PASS' if baseline_ok else 'FAIL'}")

    with open(args.out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nwrote {args.out_json}")

    if not baseline_ok:
        print("STOP: baseline did not reproduce T-1704 -- harness is not the one that produced "
              "the finding under test.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
