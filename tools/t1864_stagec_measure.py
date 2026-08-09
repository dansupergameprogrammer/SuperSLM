#!/usr/bin/env python3
r"""T-1864 -- stage C of the T-1822 activation-scale remedy design: the
attribution and configuration-sweep arms, WITHOUT sites 4 and 7 (design's own
D-SLM2087 exclusion, unchanged since stage B) and WITHOUT the rotation
funding comparator (design §11 stage C names six arms; this ticket's own
casebook, Claude/Brunel/t1864-stagec-funding-line-2026-08-09.md, records why
the rotation arm -- gate A5's three sub-cells against a distinct T-1797-
pattern construction -- was not built this session and is flagged rather
than approximated).

THE QUESTION (design §11 stage C, "which sites carry the remedy's recovery")
------------------------------------------------------------------------------
Stage B measured two POOLED configurations (M1 all seven sites + M2 all
three sites, at two G settings) and established (T-1861, debunked at T-1862)
that `ceiling` is un-refuted and `middle` is not an independent survivor --
resolves only pooled across the id/ood split. Stage C isolates the two
mechanisms and the two families to attribute which of them the recovery
comes from:

  * m1_g32       M1 ONLY (no M2 peel) at all seven M1 sites, G=32 uniformly.
  * m1_g128      M1 ONLY (no M2 peel) at all seven M1 sites, G=128 uniformly.
  * m2_residuals M2 ONLY (peel, no M1 grouping) at the two non-16 M2 sites
                 (11, 18) -- "M2-only at residuals", §11.
  * m2_site16    M2 ONLY (peel, no M1 grouping) at site 16 alone.
  * m1m2_g128    The M1+M2 candidate config -- M1 at G=128 across the seven
                 M1 sites, composed peel-first-then-group with M2 at its
                 three sites (11, 16, 18) -- value-identical construction to
                 stage B's `middle` arm, included here as a same-batch
                 reproduction so it is checked against T-1861's committed
                 `middle` vector rather than assumed to reproduce across a
                 different batch width (§6.3d's own G4 caution).

THE CONSTRUCTION, AT VALUE LEVEL
---------------------------------
Identical `rt_m1m2` to T-1861 (`tools/t1861_stageb_measure.py`), reproduced
here unchanged rather than re-derived independently -- the composition rule
(peel-first-then-group at site 16, §6.3d), the per-site k_cap table (§6.3d),
and the G floor are all T-1861's own, imported directly.

WHAT IS NOT MODELLED, INHERITED FROM T-1809/T-1835/T-1859/T-1861 UNCHANGED
----------------------------------------------------------------------------
Exactly T-1861's own list (its header, reproduced): the i-exp polynomial's
own value error, the SiLU LUT's interpolation error, the integer RMSNorm,
exact int32 matmul accumulation, the sub-quantum roundings, `R`'s own
representation error, `final_norm`, and the single-scale weight path. This
construction also does not model gate A3's per-arm i-exp domain certification
(§6.3e) -- not built or run this session, exactly the open precondition
T-1861 named for its own two arms, now open for these five as well.

Reproduce
    python tools\t1864_stagec_measure.py --docs <t1777>\out\t1777_corpus\docs.jsonl \
        --out-dir out\t1864_stagec --t1777-tools D:\SuperSLM\.worktrees\t1777-retrieval-agreement\tools
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1835_site_toggle_dump as t1835  # noqa: E402
import t1861_stageb_measure as t1861  # noqa: E402

SYSTEM_PROMPT = t1835.SYSTEM_PROMPT
ALL_SITES = t1835.ALL_SITES

M1_SITES = t1861.M1_SITES          # (2, 3, 9, 10, 12, 16, 17)
M2_SITES = t1861.M2_SITES          # (11, 16, 18)
M2_RESIDUAL_SITES = (11, 18)       # M2 sites outside M1's own set membership question --
                                    # "M2-only at residuals", §11: the two M2 sites that are
                                    # NOT the shared mlp_act site (16).
K_CAP = t1861.K_CAP
ROPE_SAFE_SITES = t1861.ROPE_SAFE_SITES
rt_m1m2 = t1861.rt_m1m2

TOUCHED_SITES = frozenset(M1_SITES) | frozenset(M2_SITES)


def _cfg_m1_only(g):
    """M1 at G=`g` across all seven M1 sites, no peel anywhere."""
    return {s: (g, K_CAP[s], s in ROPE_SAFE_SITES, False) for s in M1_SITES}


def _cfg_m2_only(sites):
    """M2 peel-only at `sites`, grouping disabled (G=1, k_cap=0 is a no-op,
    matching T-1861's own M2_ONLY_SITES convention)."""
    return {s: (1, 0, False, True) for s in sites}


def _cfg_m1m2_g128():
    """M1 at G=128 (all seven M1 sites) composed with M2 (three M2 sites) --
    value-identical to T-1861's `middle` arm."""
    cfg = {s: (128, K_CAP[s], s in ROPE_SAFE_SITES, s in M2_SITES) for s in M1_SITES}
    for s in frozenset(M2_SITES) - frozenset(M1_SITES):
        cfg[s] = (1, 0, False, True)
    return cfg


ARM_CONFIGS = {
    "m1_g32": _cfg_m1_only(32),
    "m1_g128": _cfg_m1_only(128),
    "m2_residuals": _cfg_m2_only(M2_RESIDUAL_SITES),
    "m2_site16": _cfg_m2_only((16,)),
    "m1m2_g128": _cfg_m1m2_g128(),
}
# Every arm must have an entry (possibly a no-op) for every touched site, so
# StageBGate's per-site config lookup never sees a missing key.
for _name, _cfg in ARM_CONFIGS.items():
    for _s in TOUCHED_SITES:
        _cfg.setdefault(_s, (1, 0, False, False))


def build_arms(dup_base=2):
    arms = [("null", frozenset()), ("base", ALL_SITES)]
    for name in ("m1_g32", "m1_g128", "m2_residuals", "m2_site16", "m1m2_g128"):
        arms.append((name, ALL_SITES))
    span = len(arms)
    for j in range(dup_base - 1, -1, -1):
        arms.insert(round(j * span / max(dup_base, 1)), (f"basedup{j+1}", ALL_SITES))
    return arms


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--system", default=SYSTEM_PROMPT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--t1777-tools", required=True)
    parser.add_argument("--spike-root", default=r"D:\Wizard\Tools")
    parser.add_argument("--artifact-metadata",
                        default=r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct-w8a8"
                                r"\metadata.json")
    parser.add_argument("--dup-base", type=int, default=2)
    parser.add_argument("--single-arm", default=None)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    sys.path.insert(0, str(Path(args.spike_root).resolve()))
    import t1740_pooled_float_dump as fd  # noqa: E402

    ARM_CONFIGS_ALL = dict(t1861.ARM_CONFIGS)
    ARM_CONFIGS_ALL.update(ARM_CONFIGS)
    t1861.ARM_CONFIGS.update(ARM_CONFIGS)  # StageBGate reads t1861-module-level ARM_CONFIGS

    if args.single_arm:
        all_arms = build_arms(dup_base=0)
        sel = [a for a in all_arms if a[0] == args.single_arm]
        if not sel:
            raise SystemExit(f"unknown arm {args.single_arm!r}")
        arms = sel
    else:
        arms = build_arms(dup_base=args.dup_base)

    docs = []
    with open(args.docs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    if args.limit is not None:
        docs = docs[: args.limit]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_arg = args.model if args.model is not None else str(fd.DEFAULT_MODEL)
    model_path = fd._resolve_default_model(Path(model_arg))
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype="auto",
        attn_implementation="eager")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"model+tokenizer load: {time.perf_counter() - t0:.2f}s device={device} "
          f"documents={len(docs)} arms={len(arms)} dtype={model.dtype}", flush=True)

    m1m2_arms = {"m1_g32", "m1_g128", "m2_residuals", "m2_site16", "m1m2_g128"}
    gate = t1861.StageBGate(arms, device, m1m2_arms)
    k_scales, v_scales = t1835.read_kv_landing_scales(
        args.artifact_metadata, model.config.num_hidden_layers,
        model.config.num_key_value_heads)
    print(f"kv landing scales read from artifact: {len(k_scales)} layers", flush=True)

    handles = t1861.install_forward_stageb(model, k_scales, v_scales, gate)
    print(f"forward installed: B={gate.B} arms={gate.names} hooks={len(handles)}", flush=True)

    out_dir = Path(args.out_dir)
    (out_dir / "pooled").mkdir(parents=True, exist_ok=True)

    pooled = {name: {} for name in gate.names}
    fingerprints = {}
    n_ok = n_failed = 0
    t_start = time.perf_counter()
    for i, doc in enumerate(docs):
        label = doc["label"]
        t_doc = time.perf_counter()
        messages = [{"role": "system", "content": args.system},
                    {"role": "user", "content": doc["text"]}]
        templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                  return_tensors="pt")
        input_ids = templated["input_ids"].to(device)
        n_positions = input_ids.shape[1]
        input_ids_b = input_ids.repeat(gate.B, 1).contiguous()
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)

        try:
            captured = t1835.capture_all_positions_batched(model, input_ids_b, device)
            t1835.endpoint_self_check_batched(model, captured, input_ids_b)
        except AssertionError as e:
            print(f"FAILED label={label}: {e}", flush=True)
            n_failed += 1
            continue

        n_layers = model.config.num_hidden_layers
        n_rows = n_layers + 1
        fp = fd.fnv1a64(prompt_text)
        fingerprints[label] = fp

        last_row = np.stack(captured[n_rows - 1], axis=0)
        pooled_arm = last_row[1:].astype(np.float64).mean(axis=0)
        for b, name in enumerate(gate.names):
            pooled[name][label] = pooled_arm[b]

        del captured
        if device == "cuda":
            torch.cuda.empty_cache()
        n_ok += 1
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(docs)}] label={label} n_pos={n_positions} "
                  f"this_doc={time.perf_counter() - t_doc:.2f}s elapsed={elapsed:.1f}s "
                  f"avg={elapsed/(i+1):.2f}s/doc", flush=True)

    total = time.perf_counter() - t_start
    labels = sorted(fingerprints)
    for name in gate.names:
        np.savez(out_dir / "pooled" / f"{name}.npz",
                 labels=np.array(labels),
                 fingerprints=np.array([fingerprints[l] for l in labels], dtype=np.uint64),
                 vectors=np.stack([pooled[name][l] for l in labels]))
    with open(out_dir / "arms.json", "w", encoding="utf-8") as f:
        json.dump({name: sorted(cfg) for name, cfg in arms}, f, indent=2)

    print(f"batch_done: {n_ok} ok, {n_failed} failed (of {len(docs)}), arms={gate.B}, "
          f"capture_total={total:.1f}s avg={total/max(1,len(docs)):.2f}s/doc", flush=True)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
