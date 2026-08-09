#!/usr/bin/env python3
r"""T-1861 -- stage B of the T-1822 activation-scale remedy design: the remedy
class's ceiling and middle arms, WITHOUT sites 4 and 7 (design's own D-SLM2087
exclusion, pending Dan's §29.4 decision), graded on the encoder oracle.

THE QUESTION (design §11 stage B, §9.1's kill line)
----------------------------------------------------
Does the remedy class -- M1's seven Rule-1-funnel sites (2, 3, 9, 10, 12, 16,
17) plus M2's three peeled sites (11, 16, 18), §8.1's included set minus sites
4/7 -- clear the float-tier kill line? Two arms:

  * ceiling: M1 at each site's own structural G floor (2 at site 3, 1
    elsewhere, §6.3d) with k_cap at each site's own §6.3-derived maximum,
    plus M2 (P=4, r_cap=7) at its three sites.
  * middle:  M1 at G=128 uniformly, same per-site k_cap, plus M2 (P=4,
    r_cap=7) at its three sites.

THE CONSTRUCTION, AT VALUE LEVEL (mirrors §11 stage A's own instrument
description: "a parameterized replacement of [T-1809's rt_dynamic] ... per-
group PoT snap and fixed-P peel, per site family, mirroring §4-§5's
committed-value semantics exactly at the value level")
----------------------------------------------------------------------------
`rt_m1m2(x, G, k_cap, rope_safe, peel, P, r_cap)` implements design §4.1
steps 1-4 (the grouped funnel) and, where `peel` is set, §5 steps 1-4 (the
fixed-P outlier peel) ahead of it, composed peel-first-then-group exactly as
§6.3d's M1xM2 composition states for site 16: the peel selects on the full
wide row, the grid derives from the unpeeled remainder (`D'_grid`), and M1's
groups then form over the unpeeled members against that same grid.  Every
group's `D'_g` and `k_g` is computed fresh from the row currently being
landed -- the "legal (positionwise)" class §7's own table names for M1's
`D'_g`/`k_g`, the identical disposition T-1859's Option E arm was audited
against (`Claude/Brunel/t1859-optione-recovery-measurement-2026-08-08.md`
§2).  No calibrated table, no offline statistic, and no channel-index list is
read anywhere in this construction.

WHAT IS NOT MODELLED, INHERITED FROM T-1809/T-1835/T-1859 UNCHANGED
---------------------------------------------------------------------------
Exactly T-1835's own list (this file's header, reproduced): the i-exp
polynomial's own value error, the SiLU LUT's interpolation error, the integer
RMSNorm, exact int32 matmul accumulation, the sub-quantum roundings, `R`'s
own representation error, `final_norm`, and the single-scale weight path.
This construction ALSO does not model gate A3's own site-3 i-exp domain
check (§6.3e) -- the float tier never computes the integer i-exp triple, so
it cannot abort the way the real engine would at an illegal configuration.
Per the design's own naming of this exact hazard ("the float tiers ... would
silently grade a configuration the engine cannot run"), gate A3's per-arm
certification for THIS ticket's actual two arms (all seven M1 sites plus M2
engaged simultaneously) was not built or run this session -- named as an
open precondition in this ticket's own casebook, not silently assumed
satisfied. What is measured here is the remedy class's RECALL CEILING at
these configurations, exactly the same qualification T-1859 named for its
own K/V-sidecar interface gap.

SITE PARAMETERS (design §6.3d, §5's defaults; site numbering identical to
`tools/t1835_site_toggle_dump.py`)
---------------------------------------------------------------------------
M1 sites and their k_cap (from consumer family, §6.3d):
  site 2  (attn_norm)        k_cap 6   (1536-wide GEMM consumer)
  site 3  (q_proj_requant)   k_cap 10  (score path, configured cap; RoPE-safe
                                         predicate applies, §6.2/§4.1 step 2)
  site 9  (attn_ctx)         k_cap 6   (1536-wide GEMM consumer, o_proj)
  site 10 (o_proj_requant)   k_cap 6   (provisional, residual-add consumer)
  site 12 (mlp_norm)         k_cap 6   (1536-wide GEMM consumer, gate/up)
  site 16 (mlp_act)          k_cap 3   (executed with the peel fix-up term)
  site 17 (down_proj_requant) k_cap 6  (provisional, residual-add consumer)
M2 sites (P=4, r_cap=7, C_max=2^14 -- §5, §6.3c defaults):
  site 11 (attn_residual), site 16 (mlp_act, composed with M1 above),
  site 18 (mlp_residual)
G floor per site (§6.3d): 2 at site 3 (RoPE pairing), 1 elsewhere.

Reproduce
    python tools\t1861_stageb_measure.py --docs <t1777>\out\t1777_corpus\docs.jsonl \
        --out-dir out\t1861_stageb --t1777-tools D:\SuperSLM\.worktrees\t1777-retrieval-agreement\tools
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

SYSTEM_PROMPT = t1835.SYSTEM_PROMPT
SITE_NAMES = t1835.SITE_NAMES
ALL_SITES = t1835.ALL_SITES
INT8_MAX = t1835.INT8_MAX

# M1's seven Rule-1-funnel sites (design §8.1).
M1_SITES = (2, 3, 9, 10, 12, 16, 17)
# M2's three peeled sites (design §8.1, §6.1).
M2_SITES = (11, 16, 18)
# Sites this ticket's construction touches at all (union), §8.1 minus 4/7.
TOUCHED_SITES = frozenset(M1_SITES) | frozenset(M2_SITES)
# Every other included-in-`base` site keeps its existing (T-1809/T-1835) construction.
UNTOUCHED_SITES = ALL_SITES - TOUCHED_SITES

C_MAX = 1 << 14          # §5 step 3's checked bound
DEFAULT_P = 4            # §5, §6.3c default
DEFAULT_R_CAP = 7        # §5, §6.3c default

# Per-site k_cap, design §6.3d.
K_CAP = {2: 6, 3: 10, 9: 6, 10: 6, 12: 6, 16: 3, 17: 6}
# Per-site G floor (ceiling arm), design §6.3d.
G_FLOOR = {2: 1, 3: 2, 9: 1, 10: 1, 12: 1, 16: 1, 17: 1}
# RoPE-transiting sites use §6.2's RoPE-safe admissibility predicate (§4.1 step 2).
ROPE_SAFE_SITES = frozenset({3})

ARM_CONFIGS = {
    # name -> {site: (G, k_cap, rope_safe, peel)}
    "ceiling": {s: (G_FLOOR[s], K_CAP[s], s in ROPE_SAFE_SITES, s in M2_SITES)
                for s in M1_SITES},
    "middle": {s: (128, K_CAP[s], s in ROPE_SAFE_SITES, s in M2_SITES)
               for s in M1_SITES},
}
# Sites in M2 only (not M1): grouping refused there (residual/committed-state
# sites, §6.3c's refusal row / §8.5) -- peel only, no M1 group.
M2_ONLY_SITES = frozenset(M2_SITES) - frozenset(M1_SITES)
for _name in ARM_CONFIGS:
    for s in M2_ONLY_SITES:
        ARM_CONFIGS[_name][s] = (1, 0, False, True)   # G=1/k_cap=0 => grouping is a no-op


# ==============================================================================
# The M1+M2 construction, value-level, batched over the arm axis via ArmGate
# ==============================================================================

def _admissible_k(dg, d, k_cap, rope_safe):
    """Largest k in [0, k_cap] admissible. dg, d: (..., n_groups) int64-valued float tensors."""
    import torch
    k = torch.zeros_like(dg)
    for kk in range(1, k_cap + 1):
        if rope_safe:
            cond = (127.0 * (dg * (2.0 ** kk))) <= (90.0 * d)
        else:
            cond = (dg * (2.0 ** kk)) <= d
        k = torch.where(cond, torch.full_like(k, float(kk)), k)
    return k


def rt_m1m2(x, G, k_cap, rope_safe, peel, P=DEFAULT_P, r_cap=DEFAULT_R_CAP):
    """Design §4.1 (grouped funnel) with, when `peel`, §5 (fixed-P peel) composed
    peel-first-then-group (§6.3d's M1xM2 composition rule). x: (..., N) float.
    Returns the reconstructed value at the refined grid -- the value-level
    abstraction T-1809/T-1835/T-1859 use throughout this campaign.
    """
    import torch
    dtype = x.dtype
    xf = x.to(torch.float64)
    N = xf.shape[-1]

    if peel and P > 0:
        mag = xf.abs()
        # Top-P channels by magnitude, lowest-index tie-break (house convention).
        _, top_idx = torch.topk(mag, P, dim=-1, largest=True, sorted=False)
        peel_mask = torch.zeros_like(mag, dtype=torch.bool)
        peel_mask.scatter_(-1, top_idx, True)
        unpeeled_mask = ~peel_mask
        d_full = mag.amax(dim=-1, keepdim=True)
        d_full = torch.where(d_full > 0, d_full, torch.ones_like(d_full))
        unpeeled_mag = torch.where(unpeeled_mask, mag, torch.zeros_like(mag))
        unpeeled_max = unpeeled_mag.amax(dim=-1, keepdim=True)
        unpeeled_max = torch.where(unpeeled_max > 0, unpeeled_max, torch.ones_like(unpeeled_max))
        ceil_div = torch.ceil(d_full / float(1 << r_cap))
        d_grid = torch.maximum(unpeeled_max, ceil_div)
        step = d_grid / float(INT8_MAX)
        # Bulk (unpeeled) channels at the row grid, before M1 grouping.
        row_for_group = torch.where(unpeeled_mask, xf, torch.zeros_like(xf))
        d_row = d_grid  # the row max M1 groups against at this site
    else:
        peel_mask = torch.zeros_like(xf, dtype=torch.bool)
        unpeeled_mask = torch.ones_like(xf, dtype=torch.bool)
        d_row = xf.abs().amax(dim=-1, keepdim=True)
        d_row = torch.where(d_row > 0, d_row, torch.ones_like(d_row))
        step = d_row / float(INT8_MAX)
        row_for_group = xf

    if G >= 1 and k_cap > 0:
        shape = xf.shape
        n_groups = N // G
        grp = row_for_group.reshape(*shape[:-1], n_groups, G)
        dg = torch.maximum(grp.abs().amax(dim=-1), torch.ones_like(grp[..., 0]))  # §4.1 step1 C20 guard
        # A fully-peeled group (site 16, small G): its unpeeled max is the guard
        # value 1 already, matching §6.3d's stated "a fully peeled group has
        # D'_g = 1 and takes the uniform formula's own value".
        kg = _admissible_k(dg, d_row.squeeze(-1) if d_row.dim() == dg.dim() + 1 else d_row, k_cap, rope_safe)
        kg_full = kg.repeat_interleave(G, dim=-1)
    else:
        kg_full = torch.zeros_like(xf)

    scale2k = 2.0 ** kg_full
    # §4.1 step 3: codes on the pre-shifted operand, C22 composite, in [-127,127].
    codes = torch.round(row_for_group * scale2k / step)
    codes = codes.clamp(-INT8_MAX, INT8_MAX)
    bulk_hat = codes * step / scale2k
    bulk_hat = torch.where(unpeeled_mask, bulk_hat, torch.zeros_like(bulk_hat))

    if peel and P > 0:
        peel_codes = torch.round(xf * peel_mask.to(xf.dtype) / step)
        peel_codes = peel_codes.clamp(-float(C_MAX), float(C_MAX))
        peel_hat = torch.where(peel_mask, peel_codes * step, torch.zeros_like(xf))
        out = bulk_hat + peel_hat
    else:
        out = bulk_hat

    return out.to(dtype), kg_full


# ==============================================================================
# The patched forward -- T-1835's install_forward, with sites in TOUCHED_SITES
# additionally gated through rt_m1m2 per this arm's own (G, k_cap, rope_safe,
# peel) instead of T-1809's rt_dynamic when this arm selects the M1M2 path.
# ==============================================================================

class StageBGate(t1835.ArmGate):
    """Extends T-1835's ArmGate: for arms in `m1m2_arms`, sites in TOUCHED_SITES
    are landed through `rt_m1m2` at that arm's own per-site config instead of
    T-1809's rt_dynamic; every other site (and every other arm) is unchanged.
    """

    def __init__(self, arms, device, m1m2_arms):
        super().__init__(arms, device)
        self.m1m2_arms = m1m2_arms
        # For an M1M2 arm and a touched site: which config index (order-stable).
        self._cfg = {}
        for s in TOUCHED_SITES:
            cfgs = []
            for name in self.names:
                if name in m1m2_arms and s in ARM_CONFIGS[name]:
                    cfgs.append(ARM_CONFIGS[name][s])
                else:
                    cfgs.append(None)
            self._cfg[s] = cfgs

    def apply_m1m2(self, site: int, x, original):
        """Batch-wise: arms selecting this site's M1M2 path get rt_m1m2(x, cfg);
        every other arm in the batch gets `original` (this call's identity/base
        path, applied per element via torch.where as ArmGate.apply already does).
        """
        import torch
        cfgs = self._cfg[site]
        # Group batch elements by identical config (few distinct configs: at
        # most 2, ceiling/middle) to avoid one rt_m1m2 call per batch element.
        distinct = sorted({c for c in cfgs if c is not None})
        out = original
        for cfg in distinct:
            G, k_cap, rope_safe, peel = cfg
            hat, _kg = rt_m1m2(x, G, k_cap, rope_safe, peel)
            sel = [i for i, c in enumerate(cfgs) if c == cfg]
            m = torch.zeros(self.B, dtype=torch.bool, device=x.device)
            m[sel] = True
            m = m.view(-1, *([1] * (x.dim() - 1)))
            out = torch.where(m, hat, out)
        return out


def install_forward_stageb(model, k_scales, v_scales, gate: StageBGate):
    """T-1835's install_forward, with TOUCHED_SITES routed through apply_m1m2
    ahead of (in place of) the ordinary gate.apply(site, rt_dynamic(...), ...)
    call. UNTOUCHED_SITES are wired exactly as T-1835's own install_forward.
    """
    import torch
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    handles = []

    def make_landing_hook(site):
        def hook(_module, _args, output):
            landed, _delta = t1835.rt_dynamic(output)
            ordinary = gate.apply(site, landed, output)
            if site in TOUCHED_SITES:
                return gate.apply_m1m2(site, output, ordinary)
            return ordinary
        return hook

    if gate.on_anywhere(1):
        handles.append(model.model.embed_tokens.register_forward_hook(make_landing_hook(1)))
    for layer in model.model.layers:
        if gate.on_anywhere(2):
            handles.append(layer.input_layernorm.register_forward_hook(make_landing_hook(2)))
        if gate.on_anywhere(12):
            handles.append(
                layer.post_attention_layernorm.register_forward_hook(make_landing_hook(12)))

    def make_attention_forward(attn, layer_idx):
        k_scale = k_scales[layer_idx] if k_scales is not None else None
        v_scale = v_scales[layer_idx] if v_scales is not None else None

        def forward(hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn.head_dim)

            q_raw = attn.q_proj(hidden_states)
            k_flat = attn.k_proj(hidden_states)
            v_flat = attn.v_proj(hidden_states)

            # site 3: q_proj.requant. Ordinary (T-1809) landing first -- covers
            # `null`/`base`/`basedup*` -- then M1's grouped construction
            # overrides it for `ceiling`/`middle` batch elements only. `q_delta`
            # (site 6's re-land grid) is T-1809's own delta on the RAW row,
            # computed before either landing -- identical convention to T-1835's
            # own site 6 (site 6 re-lands on the grid site 3 WOULD choose,
            # independent of whether site 3's landing engaged).
            _q_landed_base, q_delta = t1835.rt_dynamic(q_raw)
            q_ordinary = gate.apply(3, _q_landed_base, q_raw)
            q_flat = gate.apply_m1m2(3, q_raw, q_ordinary)
            # sites 4/5: unchanged static K/V landing (excluded, D-SLM2087).
            if k_scale is not None:
                k_flat = gate.apply(4, t1835.rt_static(k_flat, k_scale), k_flat)
                v_flat = gate.apply(5, t1835.rt_static(v_flat, v_scale), v_flat)

            query_states = q_flat.view(hidden_shape).transpose(1, 2)
            key_states = k_flat.view(hidden_shape).transpose(1, 2)
            value_states = v_flat.view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # sites 6/7: unchanged post-RoPE clamps (site 7 excluded, D-SLM2087;
            # site 6 out of scope, §8.2). Neither is in TOUCHED_SITES, so both
            # run T-1835's own on/off gate unmodified -- "on" (T-1809's static
            # construction) for `base`/`basedup*`/`ceiling`/`middle` (all four
            # configured with `ALL_SITES`, build_arms), "off" (identity) for
            # `null` only.
            query_states = gate.apply(
                6, t1835.rt_on_grid(query_states, q_delta.unsqueeze(1)), query_states)
            if k_scale is not None:
                key_states = gate.apply(7, t1835.rt_static(key_states, k_scale), key_states)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, layer_idx)

            key_rep = repeat_kv(key_states, attn.num_key_value_groups)
            value_rep = repeat_kv(value_states, attn.num_key_value_groups)

            q32 = query_states.to(torch.float32)
            k32 = key_rep.to(torch.float32)
            v32 = value_rep.to(torch.float32)
            attn_weights = torch.matmul(q32 * attn.scaling, k32.transpose(2, 3))
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, : k32.shape[-2]].to(
                    attn_weights.dtype)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                                       dtype=torch.float32)
            attn_weights = gate.apply(8, t1835.rt_prob_q15(attn_weights), attn_weights)

            attn_output = torch.matmul(attn_weights, v32)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous().to(
                hidden_states.dtype)
            # site 9: attn_ctx. Ordinary landing first (covers null/base), M1M2
            # overrides for ceiling/middle, same pattern as site 3.
            attn_ctx_raw = attn_output
            attn_ctx_ordinary = gate.apply(9, t1835.rt_dynamic(attn_ctx_raw)[0], attn_ctx_raw)
            attn_output = gate.apply_m1m2(9, attn_ctx_raw, attn_ctx_ordinary)

            attn_output = attn.o_proj(attn_output)
            # site 10: o_proj.requant.
            o_raw = attn_output
            o_ordinary = gate.apply(10, t1835.rt_dynamic(o_raw)[0], o_raw)
            attn_output = gate.apply_m1m2(10, o_raw, o_ordinary)
            return attn_output, None

        return forward

    def make_mlp_forward(mlp):
        def forward(x):
            gate_t = mlp.gate_proj(x)
            up = mlp.up_proj(x)
            # sites 13/14: unchanged (out of M1M2 scope, D-SLM1607).
            gate_t = gate.apply(13, t1835.rt_dynamic(gate_t)[0], gate_t)
            up = gate.apply(14, t1835.rt_dynamic(up)[0], up)
            sig = torch.sigmoid(gate_t.to(torch.float32)).to(gate_t.dtype)
            sig = gate.apply(15, t1835.rt_sigmoid_q15(sig), sig)
            act = gate_t * sig * up
            # site 16: M1+M2 composed (peel-first-then-group, §6.3d). Ordinary
            # landing first (null/base), M1M2 composed construction overrides
            # for ceiling/middle.
            act_raw = act
            act_ordinary = gate.apply(16, t1835.rt_dynamic(act_raw)[0], act_raw)
            act = gate.apply_m1m2(16, act_raw, act_ordinary)
            out = mlp.down_proj(act)
            # site 17: down_proj.requant.
            out_raw = out
            out_ordinary = gate.apply(17, t1835.rt_dynamic(out_raw)[0], out_raw)
            out = gate.apply_m1m2(17, out_raw, out_ordinary)
            return out
        return forward

    def make_layer_forward(layer):
        def forward(hidden_states, attention_mask=None, position_ids=None,
                    past_key_values=None, use_cache=False, position_embeddings=None,
                    **kwargs):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states, _ = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            hidden_states = residual + hidden_states
            # site 11: attn_residual, M2-only (peel, no grouping). Ordinary
            # landing first (null/base), M2 peel overrides for ceiling/middle.
            r11_raw = hidden_states
            r11_ordinary = gate.apply(11, t1835.rt_dynamic(r11_raw)[0], r11_raw)
            hidden_states = gate.apply_m1m2(11, r11_raw, r11_ordinary)

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            # site 18: mlp_residual, M2-only (peel, no grouping).
            r18_raw = hidden_states
            r18_ordinary = gate.apply(18, t1835.rt_dynamic(r18_raw)[0], r18_raw)
            hidden_states = gate.apply_m1m2(18, r18_raw, r18_ordinary)
            return hidden_states
        return forward

    for idx, layer in enumerate(model.model.layers):
        layer.self_attn.forward = make_attention_forward(layer.self_attn, idx)
        layer.mlp.forward = make_mlp_forward(layer.mlp)
        layer.forward = make_layer_forward(layer)

    return handles


def build_arms(dup_base=0):
    arms = [("null", frozenset()), ("base", ALL_SITES),
            ("ceiling", ALL_SITES), ("middle", ALL_SITES)]
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
    parser.add_argument("--dup-base", type=int, default=52,
                        help="extra `base` copies to hold batch width at T-1835's own cell B "
                             "(56 total: null+base+ceiling+middle=4, plus 52 basedup)")
    parser.add_argument("--single-arm", default=None)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    sys.path.insert(0, str(Path(args.spike_root).resolve()))
    import t1740_pooled_float_dump as fd  # noqa: E402

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

    m1m2_arms = {"ceiling", "middle"}
    gate = StageBGate(arms, device, m1m2_arms)
    k_scales, v_scales = t1835.read_kv_landing_scales(
        args.artifact_metadata, model.config.num_hidden_layers,
        model.config.num_key_value_heads)
    print(f"kv landing scales read from artifact: {len(k_scales)} layers", flush=True)

    handles = install_forward_stageb(model, k_scales, v_scales, gate)
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
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                  return_tensors="pt")
        input_ids = templated["input_ids"].to(device)
        n_positions = input_ids.shape[1]
        input_ids_b = input_ids.repeat(gate.B, 1).contiguous()

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
