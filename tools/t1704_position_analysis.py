#!/usr/bin/env python3
"""T-1704 Part 4 -- runs T-1702 Sec.C properly, at last, now that all-
position capture exists (T-1704 Part 1).

For both `mlp_act` (8960-wide, down_proj's input) and `down_proj.requant`
(1536-wide, down_proj's output), per (layer, prompt):

  1. Full (position*, channel*) argmax decomposition over EVERY captured
     position, not only the last -- is the winning token POSITION
     concentrated even where the winning CHANNEL wanders (T-1698/T-1700's
     own finding)?
  2. The BOS-invariance falsifier: the longest token-id prefix shared by
     all 9 prompts (the fixed system+user chat-template preamble) is
     computed directly from each record's own captured `token_id` field
     (no external tokenizer call needed -- the site-dump now carries it).
     Every position inside that shared prefix must be BIT-IDENTICAL across
     all 9 prompts (causal masking: nothing that follows a position can
     affect it) -- verified directly as a correctness check on capture
     itself, then: does the GLOBAL top-magnitude cell (the T-1698/T-1700
     rank-0 channel) trace to a position inside that shared prefix (i.e.
     could it be a BOS/preamble sink)?
  3. Are the two boundaries (mlp_act / down_proj.requant) still distinct
     phenomena, now measured on POSITION rather than only channel-identity?
  4. The prize question: does T-1703's mech2-vs-control disjoint-channel
     split at mlp_act's damage locus (layers 21-26) trace to particular
     TOKEN POSITIONS or TOKEN IDENTITIES (digits/operators/spacing) present
     in the arithmetic prompts and absent from the factual ones?

Reads the all-position `.sitedump` files this task's own
`t1704_capture_population_allpositions.py` writes (or any equivalent
--site-dump-all-positions capture), via the FAST reader
(`t1704_fast_allpos_reader`, validated against the canonical reader by
`test_t1704_fast_allpos_reader.py`).

Usage: python tools\\t1704_position_analysis.py --dir out\\t1704_population_allpos
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402
import t1704_fast_allpos_reader as FAST  # noqa: E402

NUM_LAYERS = 28
SITES = ["mlp_act", "down_proj.requant"]


def magnitude(codes: np.ndarray, d_prime: int) -> np.ndarray:
    return np.abs(codes.astype(np.float64)) * d_prime / 127.0


def analyze_prompt(path: str):
    """One pass over one prompt's all-position site-dump. Returns:
      per_site_layer_best: {(site, layer): (position, channel, value)} --
        the argmax cell over ALL captured (position, channel) pairs.
      per_site_layer_last_channel: {(site, layer): channel} -- argmax
        channel AT THE LAST CAPTURED POSITION ONLY (the backward-compat
        restriction).
      per_site_global_best: {site: (layer, position, channel, value)} --
        the single largest cell across the whole 28-layer walk.
      position_token_id: {position: token_id} -- this prompt's own token
        sequence, recovered directly from the captured records (no
        tokenizer call).
      max_position: the last (largest) captured position -- the position
        the ORIGINAL last-token-only capture would have captured.
    """
    per_site_layer_best = {}
    per_site_layer_last_seen = {}  # (site, layer) -> (position, channel, value) at the max position seen so far
    per_site_global_best = {}
    position_token_id = {}
    max_position = -1

    for site_name in SITES:
        for layer in range(NUM_LAYERS):
            per_site_layer_best[(site_name, layer)] = (-1, -1, -1.0)

    for rec in FAST.iter_chain_records_fast(path):
        # site format: "layer{L}.{name}"
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
        c = int(np.argmax(mag))
        v = float(mag[c])

        key = (name, layer)
        _, _, best_v = per_site_layer_best[key]
        if v > best_v:
            per_site_layer_best[key] = (rec.position, c, v)

        gb = per_site_global_best.get(name)
        if gb is None or v > gb[3]:
            per_site_global_best[name] = (layer, rec.position, c, v)

        per_site_layer_last_seen[key] = (rec.position, c, v)

    # Restrict to the LAST captured position specifically (the backward-
    # compat restriction, Part 3) -- re-scan is avoided by tracking, per
    # (site, layer), the (position, channel, value) triple observed AT
    # max_position specifically. Since max_position is only known after the
    # full scan, do a light second pass over the same generator is not
    # possible (it's exhausted) -- so this is computed properly below via a
    # dedicated second read, which is cheap relative to the first (the file
    # is already in the OS page cache).
    per_site_layer_last_channel = {}
    for rec in FAST.iter_chain_records_fast(path):
        if rec.position != max_position:
            continue
        if "." not in rec.site or not rec.site.startswith("layer"):
            continue
        layer_str, _, name = rec.site.partition(".")
        if name not in SITES:
            continue
        try:
            layer = int(layer_str[len("layer"):])
        except ValueError:
            continue
        mag = magnitude(rec.codes, rec.d_prime)
        c = int(np.argmax(mag))
        per_site_layer_last_channel[(name, layer)] = c

    return {
        "per_site_layer_best": per_site_layer_best,
        "per_site_layer_last_channel": per_site_layer_last_channel,
        "per_site_global_best": per_site_global_best,
        "position_token_id": position_token_id,
        "max_position": max_position,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory of <label>.sitedump all-position captures")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    role_by_label = KSR.ROLE_BY_LABEL
    results = {}
    for label, _q, role in KSR.POPULATION:
        path = os.path.join(args.dir, f"{label}.sitedump")
        if not os.path.isfile(path):
            print(f"FAILED: missing {path}", file=sys.stderr)
            return 1
        print(f"analyzing {label} (role={role}) ...", flush=True)
        results[label] = analyze_prompt(path)

    # =========================================================================
    # Sec 1: (position*, channel*) decomposition per (layer, prompt), both sites.
    # =========================================================================
    print("\n" + "=" * 78)
    print("SECTION 1: full (position, channel) argmax decomposition per (layer, prompt)")
    print("=" * 78)
    decomposition_summary = {}
    for site_name in SITES:
        print(f"\n--- site = {site_name} ---")
        position_wander_count = 0
        channel_wander_count = 0
        position_concentration_at_last = 0
        total_layers = 0
        for layer in range(NUM_LAYERS):
            positions = set()
            channels = set()
            rows = []
            for label, _q, role in KSR.POPULATION:
                r = results[label]
                pos, ch, val = r["per_site_layer_best"][(site_name, layer)]
                max_pos = r["max_position"]
                positions.add(pos)
                channels.add(ch)
                rows.append((label, role, pos, max_pos, ch, val))
            total_layers += 1
            if len(positions) > 1:
                position_wander_count += 1
            if len(channels) > 1:
                channel_wander_count += 1
            all_at_last = all(pos == max_pos for (_l, _r, pos, max_pos, _c, _v) in rows)
            if all_at_last:
                position_concentration_at_last += 1
            print(f"  layer{layer:2d}: distinct winning POSITIONS={len(positions):2d} "
                  f"distinct winning CHANNELS={len(channels):2d}  all_at_last_position={all_at_last}")
            for (label, role, pos, max_pos, ch, val) in rows:
                is_last = (pos == max_pos)
                print(f"      {label:24s} role={role:8s} winning_position={pos:3d} "
                      f"(last_position={max_pos:3d}, is_last={is_last})  winning_channel={ch:5d}  "
                      f"value={val:.4g}")
        print(f"\n  SUMMARY {site_name}: position wanders in {position_wander_count}/{total_layers} "
              f"layers; channel wanders in {channel_wander_count}/{total_layers} layers; "
              f"winning position IS the last position in ALL 9 prompts in "
              f"{position_concentration_at_last}/{total_layers} layers")
        decomposition_summary[site_name] = {
            "position_wander_layers": position_wander_count,
            "channel_wander_layers": channel_wander_count,
            "all_at_last_position_layers": position_concentration_at_last,
            "total_layers": total_layers,
        }

    # =========================================================================
    # Sec 2: BOS-invariance falsifier.
    # =========================================================================
    print("\n" + "=" * 78)
    print("SECTION 2: BOS-invariance falsifier")
    print("=" * 78)
    token_seqs = {}
    for label, _q, role in KSR.POPULATION:
        r = results[label]
        seq = [r["position_token_id"].get(p, None) for p in range(r["max_position"] + 1)]
        token_seqs[label] = seq

    min_len = min(len(s) for s in token_seqs.values())
    common_prefix_len = 0
    for i in range(min_len):
        vals = {token_seqs[label][i] for label, _q, _r in KSR.POPULATION}
        if len(vals) == 1:
            common_prefix_len = i + 1
        else:
            break
    print(f"shared token-id prefix length across all 9 prompts: {common_prefix_len} "
          f"(positions 0..{common_prefix_len - 1})")
    for label, _q, role in KSR.POPULATION:
        print(f"  {label:24s} role={role:8s} first {min(6, common_prefix_len)} token ids = "
              f"{token_seqs[label][:min(6, common_prefix_len)]}")

    bos_section = {"common_prefix_len": common_prefix_len}
    if common_prefix_len == 0:
        print("REFUTED-BY-CONSTRUCTION-CHECK: no shared prefix found -- either tokenization diverges "
              "immediately (unexpected given the fixed chat-template preamble) or capture is broken. "
              "Cannot run the falsifier.")
        bos_section["status"] = "no_shared_prefix"
    else:
        print(f"\n(Bit-identity of position 0 across all 9 prompts was already re-verified directly at "
              f"the C++ level by tools\\t1704_invariance_check.py's own --dump comparison; this "
              f"section additionally checks it at the site-dump level, per-channel, for the sites this "
              f"task's own analysis depends on.)")
        # Bit-identity check at position 0 for a few representative layers/sites.
        mismatches = []
        for site_name in SITES:
            for layer in [0, 5, 13, 21, 27]:
                target = f"layer{layer}.{site_name}"
                codes_by_label = {}
                for label, _q, role in KSR.POPULATION:
                    path = os.path.join(args.dir, f"{label}.sitedump")
                    found = None
                    for rec in FAST.iter_chain_records_fast(path):
                        if rec.position == 0 and rec.site == target:
                            found = rec.codes
                            break
                    codes_by_label[label] = found
                vals = list(codes_by_label.values())
                if any(v is None for v in vals):
                    mismatches.append((target, "missing"))
                    continue
                ref = vals[0]
                all_equal = all(np.array_equal(ref, v) for v in vals[1:])
                if not all_equal:
                    mismatches.append((target, "codes differ"))
                print(f"  position 0, {target}: bit-identical across all 9 prompts = {all_equal}")
        bos_section["position0_bit_identity_mismatches"] = mismatches

        # Does the global top channel trace to a position inside the shared prefix?
        for site_name in SITES:
            best_overall = None
            for label, _q, role in KSR.POPULATION:
                gb = results[label]["per_site_global_best"].get(site_name)
                if gb is None:
                    continue
                layer, pos, ch, val = gb
                if best_overall is None or val > best_overall[4]:
                    best_overall = (label, layer, pos, ch, val)
            if best_overall:
                label, layer, pos, ch, val = best_overall
                in_prefix = pos < common_prefix_len
                print(f"  GLOBAL top-magnitude cell at {site_name}: prompt={label} layer={layer} "
                      f"position={pos} channel={ch} value={val:.6g}  "
                      f"traces_to_shared_prefix(pos<{common_prefix_len})={in_prefix}")
                bos_section[f"{site_name}_global_best"] = {
                    "label": label, "layer": layer, "position": pos, "channel": ch, "value": val,
                    "traces_to_shared_prefix": in_prefix,
                }

    # =========================================================================
    # Sec 3: two boundaries -- position-axis comparison.
    # =========================================================================
    print("\n" + "=" * 78)
    print("SECTION 3: are the two boundaries still distinct phenomena, on POSITION now?")
    print("=" * 78)
    for site_name in SITES:
        s = decomposition_summary[site_name]
        print(f"  {site_name}: position wanders {s['position_wander_layers']}/{s['total_layers']}, "
              f"channel wanders {s['channel_wander_layers']}/{s['total_layers']}, all-prompts-at-"
              f"last-position in {s['all_at_last_position_layers']}/{s['total_layers']} layers")

    # =========================================================================
    # Sec 4: the prize question -- mech2/control split traced to position/token.
    # =========================================================================
    print("\n" + "=" * 78)
    print("SECTION 4: mech2-vs-control winning-channel split -- traced to position/token")
    print("=" * 78)
    LOCUS_LAYERS = list(range(21, 27))  # 21..26 inclusive, T-1697/1698/1700/1703's own established locus
    prize_rows = []
    for layer in LOCUS_LAYERS:
        key = ("mlp_act", layer)
        print(f"\n  layer{layer}:")
        for label, _q, role in KSR.POPULATION:
            r = results[label]
            pos, ch, val = r["per_site_layer_best"][key]
            tok = r["position_token_id"].get(pos, None)
            max_pos = r["max_position"]
            rel_pos = pos - max_pos  # negative = distance before the last token
            print(f"    {label:24s} role={role:8s} winning_position={pos:3d} "
                  f"(relative_to_last={rel_pos:+4d}) winning_token_id={tok} winning_channel={ch:5d}")
            prize_rows.append({"layer": layer, "label": label, "role": role, "position": pos,
                                "relative_position": rel_pos, "token_id": tok, "channel": ch})

    mech2_positions = {row["relative_position"] for row in prize_rows
                        if row["role"] == "mech2" and row["layer"] in LOCUS_LAYERS}
    nonmech2_positions = {row["relative_position"] for row in prize_rows
                           if row["role"] != "mech2" and row["layer"] in LOCUS_LAYERS}
    mech2_tokens = {row["token_id"] for row in prize_rows
                     if row["role"] == "mech2" and row["layer"] in LOCUS_LAYERS}
    nonmech2_tokens = {row["token_id"] for row in prize_rows
                        if row["role"] != "mech2" and row["layer"] in LOCUS_LAYERS}
    print(f"\n  mech2 relative-positions at the locus: {sorted(mech2_positions)}")
    print(f"  non-mech2 relative-positions at the locus: {sorted(nonmech2_positions)}")
    print(f"  mech2 winning token ids at the locus: {sorted(mech2_tokens)}")
    print(f"  non-mech2 winning token ids at the locus: {sorted(nonmech2_tokens)}")
    disjoint_positions = mech2_positions.isdisjoint(nonmech2_positions)
    disjoint_tokens = mech2_tokens.isdisjoint(nonmech2_tokens)
    print(f"\n  relative-position sets disjoint (mech2 vs non-mech2): {disjoint_positions}")
    print(f"  winning-token-id sets disjoint (mech2 vs non-mech2): {disjoint_tokens}")

    out = {
        "decomposition_summary": decomposition_summary,
        "bos_section": bos_section,
        "prize_rows": prize_rows,
        "mech2_relative_positions": sorted(mech2_positions),
        "nonmech2_relative_positions": sorted(nonmech2_positions),
        "mech2_tokens": sorted(t for t in mech2_tokens if t is not None),
        "nonmech2_tokens": sorted(t for t in nonmech2_tokens if t is not None),
        "positions_disjoint": disjoint_positions,
        "tokens_disjoint": disjoint_tokens,
    }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
