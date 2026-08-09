#!/usr/bin/env python3
r"""T-1859 -- measure option E (S29.4) at sites 4 and 7: a live, per-position refinement
exponent computed fresh from each row at landing time, the same class of derivation M1 uses
at its other seven sites (S4.1/S4.2), applied to the K landing (site 4) and the K post-RoPE
clamp (site 7) instead of T-1809/T-1835's static calibrated scale.

THIS EXTENDS T-1835's INSTRUMENT, UNCHANGED IN ARITHMETIC FOR EVERY SITE E DOES NOT TOUCH
-----------------------------------------------------------------------------------------
`rt_dynamic`, `rt_static`, `rt_on_grid`, `rt_prob_q15`, `rt_sigmoid_q15`, `read_kv_landing_scales`,
`ALL_SITES`, `SITE_NAMES`, `capture_all_positions_batched`, `endpoint_self_check_batched`,
`_write_dump`, `SYSTEM_PROMPT`, `INT8_MAX` are imported from `tools/t1835_site_toggle_dump.py`
(committed at ef4575f, this worktree's base) and used verbatim -- not re-derived. Sites 1, 2, 3,
6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 run through the identical code path T-1835 built;
only sites 4 and 7 gain a second, independently-toggleable construction.

TWO NEW VIRTUAL SITES, NOT A REPLACEMENT OF 4/7
------------------------------------------------
Site 19 ("k_landing_E") and site 20 ("k_rope_clamp_E") are Option E's constructions at the K
landing and the post-RoPE K clamp. They are mutually exclusive with sites 4/7 in every arm this
script builds (never both a site-4/19 pair or a site-7/20 pair active for the same arm), so the
`base` and `null` arms reproduce T-1835's own bit-for-bit (no site 19/20 ever active there).

OPTION E'S ARITHMETIC, READ AT SOURCE FROM S4.1/S4.2 OF
`Claude/Vitruvius/t1822-activation-scale-remedy-design-2026-08-07.md` (this project's own
design document, D:\Wizard checkout), NOT A NEIGHBOUR OF IT
-----------------------------------------------------------------------------------------------
S4.1 (M1's producer side, the grouped funnel): for a row of width W split into groups of a
fixed size G, `D' = max|row|` (the row max, identical to `rt_dynamic`'s own `d`), and per group
`D'_g = max|group|`. `k_g` is the largest `k` in `[0, k_cap]` with the group's ADMISSIBILITY
PREDICATE satisfied:
  - ordinary sites: `(D'_g << k) <= D'`
  - a RoPE-TRANSITING site (S6.2, S4.1 step 2): `k = 0`, or `127*(D'_g << k) <= 90*D'` -- the
    RoPE-safe bound, because a group refined past that ratio can be lifted above the rotation's
    own clamp ceiling of 127.
Codes: `q_i = round(row[i] * 2^k_g / D' * 127)`, clamped to [-127, 127]; the group's effective
step is `(D'/127) * 2^-k_g` -- finer than the row's own step by `2^k_g`.

S4.2 (M1's consumer side, exact shift-alignment): a downstream consumer that must read a value
already landed on a per-group grid re-lands it on the SAME grid rather than deriving a new one
-- `A_i = q_i << (K - k_g)` at the engine's int level, which in this float-domain simulation is
"round-trip through the SAME per-group delta the producer chose", i.e. `rt_on_grid` per group
instead of per row.

SITE 4 IS ROPE-TRANSITING FOR K, EXACTLY AS SITE 3 IS FOR Q (S6.2's own analogy, "Site 7 takes
the static K scale and is independent of site 4 in the same way" -- restated here for E: site 4's
landing happens BEFORE `apply_rotary_pos_emb` is called on the key states in T-1835's own forward,
so a group refined at site 4 that is not RoPE-safe can be lifted past the post-rotation clamp at
site 7. Site 4's Option-E landing therefore uses the RoPE-safe predicate, identically to how M1
already treats site 3 (S6.2, D-SLM1816) -- this is what "the same class of derivation M1 already
uses at its other seven sites" cashes out to for site 4 specifically.

SITE 7 RE-LANDS ON SITE 4'S OWN GRID WHEN SITE 4'S E CONSTRUCTION RAN, AND COMPUTES ITS OWN
GROUPED REFINEMENT (ORDINARY, NOT ROPE-SAFE -- NOTHING FURTHER ROTATES A POST-ROPE ROW) WHEN
SITE 4 DID NOT
------------------------------------------------------------------------------------------------
This is the exact site-6/site-3 pattern T-1835's own header already documents ("Site 6 therefore
remains exactly the engine's operation when site 3 is on, and is a well-defined operation ... when
site 3 is off"), applied to the K side rather than restated only for Q.

GROUP SIZE AND k_cap ARE THIS SCRIPT'S OWN MODELLING PARAMETERS, NOT PART OF S29.4's SPEC
-------------------------------------------------------------------------------------------
S29.4 specifies the DERIVATION CLASS, not a group size -- the K/V-store sidecar this option would
require has never been built. `GROUP_SIZE = 32` groups the 128-wide per-(layer, kv_head) K row
into four groups, one of the sizes S4.1 itself names as an example (`G in {32, 128, 256}`, stated
for the 1536- and 8960-wide rows; 32 is the only one of those three that divides 128 into more
than one group -- 128 would collapse to a single group, i.e. no refinement at all). `K_CAP = 8` is
generous headroom for a float-domain simulation with no int64 overflow to guard; whether it binds
in practice is measured and reported, not assumed.

WHAT THIS SCRIPT DOES NOT MODEL, inherited from T-1835/T-1809 unchanged, PLUS ONE NEW ITEM
--------------------------------------------------------------------------------------------
Everything T-1835's own header names (the i-exp polynomial's own value error, the SiLU LUT's
interpolation error, integer RMSNorm, exact int32 matmul accumulation, sub-quantum roundings,
`R`'s own representation error, `final_norm`, the single-scale weight path). New here: the actual
K/V-store FORMAT this option would require (a `k[]` sidecar field per cached row, S29.4's own
"Interface" cost column) is not built -- this script measures the RECALL CEILING of the
DERIVATION, not the storage or save/restore-contract engineering S29.4 prices separately.
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

GROUP_SIZE = 32
K_CAP = 8


# ==============================================================================
# Option E's arithmetic -- new, specified above, read at source from S4.1/S4.2
# ==============================================================================

def _admissible_k(t1835, dg, drow, k_cap: int, rope_safe: bool):
    """`k_g` in [0, k_cap] for every group: the largest k whose admissibility predicate holds.

    `dg`: (..., ng, 1) per-group max-abs. `drow`: (..., 1, 1) row max-abs, broadcastable to
    `dg`'s shape. `k=0` is always admissible under both predicates (ordinary: `D'_g <= D'` holds
    by construction, D' being the max over all groups including this one; RoPE-safe: `k=0` is an
    explicit disjunct) -- so `k_g` is always well-defined and this never falls through undefined.
    """
    import torch
    best = torch.zeros_like(dg, dtype=torch.int64)
    found = torch.zeros_like(dg, dtype=torch.bool)
    for k in range(k_cap, -1, -1):
        if rope_safe:
            if k == 0:
                ok = torch.ones_like(dg, dtype=torch.bool)
            else:
                ok = (127.0 * (dg * (2.0 ** k)) <= 90.0 * drow)
        else:
            ok = (dg * (2.0 ** k) <= drow)
        take = ok & (~found)
        best = torch.where(take, torch.full_like(best, k), best)
        found = found | take
    assert bool(found.all()), "k_g undefined for some group -- k=0 admissibility broke"
    return best


def rt_grouped(t1835, x, group_size: int, k_cap: int, rope_safe: bool):
    """M1's grouped per-position refinement (S4.1), over the LAST axis of `x`.

    Returns `(x_hat, k_g, delta_row)`: `k_g` has shape (..., ng, 1) (int64, the per-group
    refinement exponent), `delta_row` has shape (..., 1) (the row's own shared quantum D'/127,
    S4.1 step 1's unchanged single reduction -- identical to what `rt_dynamic` computes as `d`).
    """
    import torch
    dtype = x.dtype
    xf = x.to(torch.float32)
    *lead, W = xf.shape
    assert W % group_size == 0, f"row width {W} not divisible by group size {group_size}"
    ng = W // group_size
    d = xf.abs().amax(dim=-1, keepdim=True)                       # D', (...,1)
    d = torch.where(d > 0, d, torch.ones_like(d))                  # C20's all-zero guard
    xg = xf.view(*lead, ng, group_size)
    dg = xg.abs().amax(dim=-1, keepdim=True)                       # D'_g, (...,ng,1); a
    # degenerate (all-zero) group's dg is left at 0 rather than guarded to 1 -- 0 is a valid
    # admissibility-predicate operand (0 <= D' always holds, so an all-zero group gets maximal
    # refinement k=k_cap, harmlessly, since delta_g is derived from delta_row, never from dg
    # itself; forcing dg to 1 here previously broke k=0's own admissibility whenever D' < 1).
    drow = d.unsqueeze(-2)                                          # (...,1,1) vs (...,ng,1)
    k_g = _admissible_k(t1835, dg, drow, k_cap, rope_safe)          # (...,ng,1)
    delta_row = d / t1835.INT8_MAX                                  # (...,1) row quantum
    delta_g = delta_row.unsqueeze(-2) * (2.0 ** (-k_g.to(torch.float32)))  # (...,ng,1)
    q = t1835._round_half_away_from_zero(xg / delta_g).clamp(-t1835.INT8_MAX, t1835.INT8_MAX)
    x_hat = (q * delta_g).view(*lead, W).to(dtype)
    return x_hat, k_g, delta_row


def rt_grouped_reland(t1835, x, k_g, delta_row, group_size: int):
    """M1's consumer-side re-unification (S4.2), per group: re-land `x` on the ALREADY-CHOSEN
    grid `(k_g, delta_row)` a prior `rt_grouped` call picked, rather than deriving a new one.
    `k_g`/`delta_row` must already be broadcastable to `x`'s leading dims (the caller transposes
    them to match, e.g. S<->H when the row moves from (B,S,H,head_dim) to (B,H,S,head_dim)).
    """
    import torch
    dtype = x.dtype
    xf = x.to(torch.float32)
    *lead, W = xf.shape
    ng = W // group_size
    xg = xf.view(*lead, ng, group_size)
    delta_g = delta_row.unsqueeze(-2) * (2.0 ** (-k_g.to(torch.float32)))
    q = t1835._round_half_away_from_zero(xg / delta_g).clamp(-t1835.INT8_MAX, t1835.INT8_MAX)
    x_hat = (q * delta_g).view(*lead, W).to(dtype)
    return x_hat


# ==============================================================================
# The two virtual sites and the extended arm gate
# ==============================================================================

EXTRA_SITE_NAMES = {19: "k_landing_E", 20: "k_rope_clamp_E"}


class ExtArmGate:
    """T-1835's `ArmGate`, parameterised over an explicit site-id set rather than that module's
    global `SITE_NAMES` -- copied rather than subclassed so this file's legality is checkable by
    reading it alone, per the ticket's own instruction to audit this arm's own arithmetic."""

    def __init__(self, arms, site_ids, device):
        import torch
        self.names = [a[0] for a in arms]
        self.configs = [a[1] for a in arms]
        self.B = len(arms)
        self._m = {}
        self._any = {}
        for s in site_ids:
            col = [s in cfg for cfg in self.configs]
            self._m[s] = torch.tensor(col, dtype=torch.bool, device=device)
            self._any[s] = any(col)

    def on_anywhere(self, site: int) -> bool:
        return self._any[site]

    def apply(self, site: int, landed, original):
        import torch
        m = self._m[site].view(-1, *([1] * (landed.dim() - 1)))
        return torch.where(m, landed, original)


def build_arms(t1835):
    """Six informative arms plus dup-base/pad copies, filled to EXACTLY batch width 56 -- T-1835's
    own cell B, so `base`'s recall@1 in this run is guaranteed identical to T-1835's own 0.665272
    by that ticket's own G2 (batched-GEMM row independence, executed, all four `base` copies
    bit-identical regardless of what else shares the batch) -- reusing T-1835's own measured
    batch1-to-batch56 offset (+0.041841, S4) rather than introducing a new, unmeasured batch width.
    """
    ALL18 = t1835.ALL_SITES
    arms = [
        ("null", frozenset()),
        ("base", ALL18),
        ("E_only19_site4_alone", frozenset({19})),
        ("E_only20_site7_alone", frozenset({20})),
        ("E_only19_20_pair", frozenset({19, 20})),
        ("E_recovery_4and7", (ALL18 - {4, 7}) | {19, 20}),
        ("E_recovery_site4_only", (ALL18 - {4}) | {19}),
        ("E_recovery_site7_only", (ALL18 - {7}) | {20}),
    ]
    for i in range(1):
        arms.append((f"basedup{i + 1}", ALL18))
    while len(arms) < 56:
        arms.append((f"basepad{len(arms):02d}", ALL18))
    assert len(arms) == 56, len(arms)
    names = [a[0] for a in arms]
    assert len(names) == len(set(names)), "duplicate arm name"
    return arms


def install_forward_e(t1835, model, k_scales, v_scales, gate, kg_stats=None):
    """T-1835's `install_forward`, unchanged for every site except the attention forward, which
    gains sites 19/20 alongside the untouched 4/7. Sites 1,2,3,6,8,9,10,11,12,13,14,15,16,17,18
    are copied verbatim from `t1835_site_toggle_dump.install_forward`.

    `kg_stats`, if given a dict, accumulates `max_k_observed` and `n_k_at_cap` (a count of
    groups whose chosen k_g landed exactly at K_CAP -- the signal that the cap bound and a
    higher cap might have refined further) across every call, for the diagnostics `main` prints.
    """
    import torch
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    handles = []

    def _record(k_g_tensor):
        if kg_stats is None:
            return
        kmax = int(k_g_tensor.max().item())
        n_at_cap = int((k_g_tensor == K_CAP).sum().item())
        kg_stats["max_k_observed"] = max(kg_stats.get("max_k_observed", 0), kmax)
        kg_stats["n_k_at_cap"] = kg_stats.get("n_k_at_cap", 0) + n_at_cap
        kg_stats["n_groups"] = kg_stats.get("n_groups", 0) + k_g_tensor.numel()

    def make_landing_hook(site):
        def hook(_module, _args, output):
            landed, _delta = t1835.rt_dynamic(output)
            return gate.apply(site, landed, output)
        return hook

    if gate.on_anywhere(1):
        handles.append(model.model.embed_tokens.register_forward_hook(make_landing_hook(1)))
    for layer in model.model.layers:
        if gate.on_anywhere(2):
            handles.append(layer.input_layernorm.register_forward_hook(make_landing_hook(2)))
        if gate.on_anywhere(12):
            handles.append(
                layer.post_attention_layernorm.register_forward_hook(make_landing_hook(12)))

    n_kv_heads = model.config.num_key_value_heads

    def make_attention_forward(attn, layer_idx):
        k_scale = k_scales[layer_idx] if k_scales is not None else None
        v_scale = v_scales[layer_idx] if v_scales is not None else None
        head_dim = attn.head_dim

        def forward(hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn.head_dim)

            q_flat = attn.q_proj(hidden_states)
            k_flat = attn.k_proj(hidden_states)
            v_flat = attn.v_proj(hidden_states)

            # site 3: q_proj.requant -- dynamic, over the whole 1536-wide row. Unchanged.
            q_landed, q_delta = t1835.rt_dynamic(q_flat)
            q_flat = gate.apply(3, q_landed, q_flat)

            # sites 4/5: the STATIC per-head K/V landing. Unchanged -- for elements where site 19
            # is active, site 4's mask is False by construction (build_arms never sets both), so
            # `k_flat` after this line is still the untouched float K row for those elements.
            if k_scale is not None:
                k_flat = gate.apply(4, t1835.rt_static(k_flat, k_scale), k_flat)
                v_flat = gate.apply(5, t1835.rt_static(v_flat, v_scale), v_flat)

            # site 19: Option E's grouped, RoPE-safe refinement at the K landing (S4.1's
            # admissibility predicate for a RoPE-transiting site, S6.2/D-SLM1816 -- site 4 is
            # RoPE-transiting for K exactly as site 3 is for Q, because `apply_rotary_pos_emb`
            # below runs on `key_states` after this landing).
            k_g19 = delta19 = None
            if gate.on_anywhere(19) or gate.on_anywhere(20):
                B, S, _ = k_flat.shape
                kh = k_flat.view(B, S, n_kv_heads, head_dim)
                k_hat19, k_g19, delta19 = rt_grouped(
                    t1835, kh, GROUP_SIZE, K_CAP, rope_safe=True)
                _record(k_g19)
                k_hat19 = k_hat19.reshape(B, S, n_kv_heads * head_dim).to(k_flat.dtype)
                k_flat = gate.apply(19, k_hat19, k_flat)

            query_states = q_flat.view(hidden_shape).transpose(1, 2)
            key_states = k_flat.view(hidden_shape).transpose(1, 2)
            value_states = v_flat.view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # sites 6/7: the post-RoPE clamps. Unchanged.
            query_states = gate.apply(
                6, t1835.rt_on_grid(query_states, q_delta.unsqueeze(1)), query_states)
            if k_scale is not None:
                key_states = gate.apply(7, t1835.rt_static(key_states, k_scale), key_states)

            # site 20: Option E's post-RoPE re-landing. Where site 19 also ran for this batch
            # element, re-land on site 19's OWN chosen grid (S4.2's exact shift-alignment,
            # transposed S<->H to match `key_states`' own axis order). Where site 19 did not run
            # (site 20 alone), compute an independent grouped refinement on the post-RoPE row,
            # ORDINARY (non-RoPE-safe) predicate -- nothing further rotates a post-RoPE row, the
            # same "site 7 independent of site 4" reading T-1835's own header states for the
            # static construction, carried to E's.
            if gate.on_anywhere(20):
                if k_g19 is not None:
                    k_g19_t = k_g19.transpose(1, 2)
                    delta19_t = delta19.transpose(1, 2)
                    key_paired = rt_grouped_reland(t1835, key_states, k_g19_t, delta19_t,
                                                   GROUP_SIZE)
                else:
                    key_paired = key_states
                key_alone, k_g20_alone, _ = rt_grouped(
                    t1835, key_states, GROUP_SIZE, K_CAP, rope_safe=False)
                _record(k_g20_alone)
                mask19 = gate._m[19].view(-1, 1, 1, 1)
                key_E20 = torch.where(mask19, key_paired, key_alone)
                key_states = gate.apply(20, key_E20, key_states)

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
            attn_output = gate.apply(9, t1835.rt_dynamic(attn_output)[0], attn_output)

            attn_output = attn.o_proj(attn_output)
            attn_output = gate.apply(10, t1835.rt_dynamic(attn_output)[0], attn_output)
            return attn_output, None

        return forward

    def make_mlp_forward(mlp):
        def forward(x):
            gate_t = mlp.gate_proj(x)
            up = mlp.up_proj(x)
            gate_t = gate.apply(13, t1835.rt_dynamic(gate_t)[0], gate_t)
            up = gate.apply(14, t1835.rt_dynamic(up)[0], up)
            sig = torch.sigmoid(gate_t.to(torch.float32)).to(gate_t.dtype)
            sig = gate.apply(15, t1835.rt_sigmoid_q15(sig), sig)
            act = gate_t * sig * up
            act = gate.apply(16, t1835.rt_dynamic(act)[0], act)
            out = mlp.down_proj(act)
            out = gate.apply(17, t1835.rt_dynamic(out)[0], out)
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
            hidden_states = gate.apply(11, t1835.rt_dynamic(hidden_states)[0], hidden_states)

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = gate.apply(18, t1835.rt_dynamic(hidden_states)[0], hidden_states)
            return hidden_states
        return forward

    for idx, layer in enumerate(model.model.layers):
        layer.self_attn.forward = make_attention_forward(layer.self_attn, idx)
        layer.mlp.forward = make_mlp_forward(layer.mlp)
        layer.forward = make_layer_forward(layer)

    return handles


# ==============================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--system", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--t1777-tools", required=True)
    parser.add_argument("--t1835-tools", required=True,
                        help="directory holding t1835_site_toggle_dump.py")
    parser.add_argument("--spike-root", default=r"D:\Wizard\Tools")
    parser.add_argument("--artifact-metadata",
                        default=r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct-w8a8"
                                r"\metadata.json")
    parser.add_argument("--single-arm", default=None)
    parser.add_argument("--k-g-diagnostics", action="store_true",
                        help="log the observed k_g distribution (checks K_CAP does not bind)")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1835_tools).resolve()))
    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    sys.path.insert(0, str(Path(args.spike_root).resolve()))
    import t1835_site_toggle_dump as t1835  # noqa: E402
    import t1740_pooled_float_dump as fd  # noqa: E402

    system_prompt = args.system if args.system is not None else t1835.SYSTEM_PROMPT

    all_arms = build_arms(t1835)
    if args.single_arm:
        sel = [a for a in all_arms if a[0] == args.single_arm]
        if not sel:
            raise SystemExit(f"unknown arm {args.single_arm!r}")
        arms = sel
    else:
        arms = list(all_arms)

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
          f"documents={len(docs)} arms={len(arms)} dtype={model.dtype} "
          f"GROUP_SIZE={GROUP_SIZE} K_CAP={K_CAP}", flush=True)

    site_ids = set(t1835.ALL_SITES) | set(EXTRA_SITE_NAMES)
    gate = ExtArmGate(arms, site_ids, device)
    k_scales, v_scales = t1835.read_kv_landing_scales(
        args.artifact_metadata, model.config.num_hidden_layers,
        model.config.num_key_value_heads)
    print(f"kv landing scales read from artifact: {len(k_scales)} layers, "
          f"layer0 k={k_scales[0]:.6g} v={v_scales[0]:.6g}", flush=True)

    kg_stats = {} if args.k_g_diagnostics else None
    handles = install_forward_e(t1835, model, k_scales, v_scales, gate, kg_stats=kg_stats)
    print(f"forward installed: B={gate.B} hooks={len(handles)}", flush=True)

    out_dir = Path(args.out_dir)
    (out_dir / "pooled").mkdir(parents=True, exist_ok=True)

    pooled = {name: {} for name in gate.names}
    fingerprints = {}

    n_ok = n_failed = 0
    t_start = time.perf_counter()
    for i, doc in enumerate(docs):
        label = doc["label"]
        t_doc = time.perf_counter()
        messages = [{"role": "system", "content": system_prompt},
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

        last_row = np.stack(captured[n_rows - 1], axis=0)          # (n_pos, B, H)
        pooled_arm = last_row[1:].astype(np.float64).mean(axis=0)  # (B, H)
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
    if kg_stats is not None:
        with open(out_dir / "kg_diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(kg_stats, f, indent=2)
        print(f"k_g diagnostics: max_k_observed={kg_stats.get('max_k_observed')} "
              f"(K_CAP={K_CAP}) n_k_at_cap={kg_stats.get('n_k_at_cap', 0)} "
              f"of n_groups={kg_stats.get('n_groups', 0)}", flush=True)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
