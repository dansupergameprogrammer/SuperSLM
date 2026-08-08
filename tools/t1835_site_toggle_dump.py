#!/usr/bin/env python3
r"""T-1835 -- per-site activation-quantization isolation over all 18 engine sites.

T-1809 measured the activation CLASS as one arm (`--arm C`: every landing on) and
established that it carries essentially the whole encoder-side deficit. It did not say
which of the eighteen sites carries what. This script answers that, by making each of
T-1809's eighteen landings independently switchable and running many configurations
through ONE forward.

THE CONSTRUCTION IS T-1809's, UNCHANGED IN ARITHMETIC
-----------------------------------------------------
`rt_dynamic`, `rt_static`, `rt_on_grid`, `rt_prob_q15`, `rt_sigmoid_q15`,
`read_kv_landing_scales`, `degrade_weights` and the patched attention/MLP/layer forwards
are T-1809's own (`tools/t1809_activation_arm_dump.py`, committed at c385665/171904c in
`.worktrees/t1809-activation-interaction`), copied here verbatim in arithmetic. The ONE
change is that each landing is now gated per BATCH ELEMENT:

    x = torch.where(site_mask_for_this_batch_element, landed, x)

so batch element `b` runs configuration `ARMS[b]`. With the mask all-True the expression
is the landing; with it all-False it is the identity -- which is exactly T-1809's `on`
boolean, one level down. That makes T-1809's arm C the `base` element of this run and its
arm A the `null` element, and both are gated against T-1809's own committed dumps.

WHY BATCH THE ARMS RATHER THAN RUN THEM SERIALLY
-------------------------------------------------
The capture is a token-at-a-time incremental forward (T-1777's own composition), so every
step is a single-token forward whose wall time is dominated by reading the 1.5B-parameter
weight set out of VRAM and by Python-level per-site work -- neither of which grows with the
batch. Fifty-six arms in one batch therefore cost approximately what one arm costs, where
fifty-six serial arms would cost fifty-six times as much. The measured consequence of
batching is not assumed away: `--single-arm` runs any one configuration at batch 1 through
the identical code, and `tools/t1835_gate_null.py` reports the exact deviation between the
batch-1 and batched captures of the same configuration, and between the batch-1 capture and
T-1809's own.

SITE NUMBERING is `Claude/Vitruvius/t1814-activation-site-candidate-map-2026-08-07.md` §2's,
which reproduces T-1809 §2's table. Eleven of the eighteen (3-9 and the K/V landings) are
attention-internal and appear in no dump this campaign has taken.

WHAT IS NOT MODELLED is T-1809 §3's list, inherited unchanged: the i-exp polynomial's own
value error, the SiLU LUT's interpolation error, the integer RMSNorm, exact int32 matmul
accumulation, the sub-quantum C24/C25/C26/C28 roundings, `R`'s own representation error,
`final_norm`, and the single-scale weight path at `embed`/`lm_head`/the norm gains. A
per-site attribution produced here is an attribution over the MODELLED sites.

ONE COUPLING, STATED BECAUSE IT IS A MODELLING DECISION, NOT A DETAIL
---------------------------------------------------------------------
Site 6 (the post-RoPE q clamp) re-lands the rotated query on the grid site 3 chose. The
grid is `D'/127` with `D' = max|q row|`, and `rt_dynamic` derives it from its INPUT, so the
value is the same whether or not site 3's landing was applied. Site 6 therefore remains
exactly the engine's operation when site 3 is on, and is a well-defined operation (land on
the grid site 3 WOULD have chosen) when site 3 is off. Site 7 takes the static K scale and
is independent of site 4 in the same way.

Outputs
-------
  * `<out>/pooled/<arm>.npz`   -- the pooled final-layer embedding per document per arm,
    computed here as `t1777_retrieval_report.pooled_final_layer_float` computes it
    (positions 1..n-1 of row r-1, float64 mean); the identity of the two is gated, not
    asserted, by writing full dumps for the gate arms and re-reading them with T-1777's
    own function.
  * `<out>/dumps/<arm>/<label>.float.bin` -- full T-1740-format dumps, for the arms named
    by `--full-dump` only. Full dumps for all arms would be ~78 GB; the gate arms are
    written in full so the pooled path can be validated against the unmodified reader.
  * `<out>/drift/<arm>.npy` -- per (document, position, row) relative L2 and cosine of that
    arm's residual state against the `null` element's, both computed inside this run from
    the same batch, so no cross-run comparison enters the drift figures.

Usage
-----
    python tools\t1835_site_toggle_dump.py \
        --docs <t1777>\out\t1777_corpus\docs.jsonl \
        --out-dir out\t1835 \
        --t1777-tools D:\SuperSLM\.worktrees\t1777-retrieval-agreement\tools \
        --full-dump base --full-dump null
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

_ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJ = ("gate_proj", "up_proj", "down_proj")

INT8_MAX = 127
PROB_FRAC_BITS = 15      # pipeline.PROB_FRAC_BITS
SIGMOID_FRAC_BITS = 15   # pipeline.SIGMOID_FRAC_BITS


# ==============================================================================
# The eighteen sites (T-1814 §2 numbering == T-1809 §2's table)
# ==============================================================================

SITE_NAMES = {
    1: "embed",
    2: "attn_norm",
    3: "q_proj_requant",
    4: "k_proj_landing",
    5: "v_proj_landing",
    6: "q_rope_clamp",
    7: "k_rope_clamp",
    8: "softmax_q15",
    9: "attn_ctx",
    10: "o_proj_requant",
    11: "attn_residual",
    12: "mlp_norm",
    13: "gate_proj_requant",
    14: "up_proj_requant",
    15: "silu_sigmoid_q15",
    16: "mlp_act",
    17: "down_proj_requant",
    18: "mlp_residual",
}
ALL_SITES = frozenset(SITE_NAMES)

# Groups worth a joint arm. Each is a set the campaign already reasons about as a unit.
GROUPS = {
    "attn_internal": frozenset({3, 4, 5, 6, 7, 8, 9}),      # T-1814's eleven, minus norms/o/res
    "attn_block": frozenset({2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
    "mlp_block": frozenset({12, 13, 14, 15, 16, 17, 18}),
    "kv_landing": frozenset({4, 5}),                         # the two static-scale sites
    "rope_clamps": frozenset({6, 7}),
    "residual_pair": frozenset({11, 18}),                    # T-1795/T-1820's instrumented pair
    "positionwise_five": frozenset({10, 13, 14, 16, 17}),    # T-1795's four sub-blocks + mlp_act
    "norms": frozenset({2, 12}),
    "fixed_point": frozenset({8, 15}),                       # the two non-int8 representations
}


def build_arms() -> list[tuple[str, frozenset]]:
    arms: list[tuple[str, frozenset]] = [
        ("base", ALL_SITES),
        ("null", frozenset()),
    ]
    for s in sorted(SITE_NAMES):
        arms.append((f"off{s:02d}_{SITE_NAMES[s]}", ALL_SITES - {s}))
    for s in sorted(SITE_NAMES):
        arms.append((f"only{s:02d}_{SITE_NAMES[s]}", frozenset({s})))
    for g, members in GROUPS.items():
        arms.append((f"offG_{g}", ALL_SITES - members))
    for g, members in GROUPS.items():
        arms.append((f"onlyG_{g}", members))
    names = [a[0] for a in arms]
    assert len(names) == len(set(names)), "duplicate arm name"
    return arms


# ==============================================================================
# The activation grids -- T-1809's, verbatim in arithmetic
# ==============================================================================

def _round_half_away_from_zero(t):
    """`pipeline._round_half_away_from_zero` in torch: sign(x) * floor(|x| + 0.5)."""
    import torch
    return torch.sign(t) * torch.floor(torch.abs(t) + 0.5)


def rt_dynamic(x):
    """The engine's per-token dynamic int8 round-trip over the LAST axis.

    Returns (x_hat, delta) with `delta = D'/127` the row's quantum. `delta` is derived from
    the INPUT row, so a caller that must re-land a later value on the same grid gets the
    grid site 3 chose regardless of whether site 3's landing was applied.
    """
    import torch
    dtype = x.dtype
    xf = x.to(torch.float32)
    d = xf.abs().amax(dim=-1, keepdim=True)
    d = torch.where(d > 0, d, torch.ones_like(d))          # C20's all-zero guard
    q = _round_half_away_from_zero(xf * INT8_MAX / d).clamp(-INT8_MAX, INT8_MAX)
    delta = d / INT8_MAX
    return (q * delta).to(dtype), delta


def rt_static(x, scale):
    """The engine's K/V landing: a fixed float-domain scale, symmetric int8, saturating."""
    import torch
    dtype = x.dtype
    xf = x.to(torch.float32)
    q = _round_half_away_from_zero(xf / scale).clamp(-INT8_MAX, INT8_MAX)
    return (q * scale).to(dtype)


def rt_on_grid(x, delta):
    """Re-land a value on an ALREADY-CHOSEN grid, saturating at +/-127 codes."""
    import torch
    dtype = x.dtype
    xf = x.to(torch.float32)
    q = _round_half_away_from_zero(xf / delta).clamp(-INT8_MAX, INT8_MAX)
    return (q * delta).to(dtype)


def rt_prob_q15(p):
    """The engine's probability representation: `(e << 15) // total`, i.e. floor to Q15."""
    import torch
    dtype = p.dtype
    pf = p.to(torch.float32)
    unit = float(1 << PROB_FRAC_BITS)
    return (torch.floor(pf * unit) / unit).to(dtype)


def rt_sigmoid_q15(s):
    """The SiLU sigmoid's Q15 output width (`silu_lut.silu_sigmoid_q15`)."""
    import torch
    dtype = s.dtype
    sf = s.to(torch.float32)
    unit = float(1 << SIGMOID_FRAC_BITS)
    return (_round_half_away_from_zero(sf * unit) / unit).to(dtype)


# ==============================================================================
# Per-arm gating
# ==============================================================================

class ArmGate:
    """Holds one boolean row per site, of width B, and applies it by `torch.where`.

    `torch.where` is exact in both directions: where the mask is False the original tensor's
    bits are returned unchanged, so an arm with a site off is bit-identical to a forward that
    never called the landing at all.
    """

    def __init__(self, arms, device):
        import torch
        self.names = [a[0] for a in arms]
        self.configs = [a[1] for a in arms]
        self.B = len(arms)
        self._m = {}
        self._any = {}
        for s in SITE_NAMES:
            col = [s in cfg for cfg in self.configs]
            self._m[s] = torch.tensor(col, dtype=torch.bool, device=device)
            self._any[s] = any(col)

    def on_anywhere(self, site: int) -> bool:
        return self._any[site]

    def apply(self, site: int, landed, original):
        """`landed` where this arm enables `site`, `original` elsewhere. Batch is axis 0."""
        import torch
        m = self._m[site].view(-1, *([1] * (landed.dim() - 1)))
        return torch.where(m, landed, original)


# ==============================================================================
# The patched forward -- T-1809's, with per-arm gating at each of the 18 sites
# ==============================================================================

def read_kv_landing_scales(metadata_path, n_layers, n_kv_heads):
    """The engine's calibrated K/V landing scales, read from its own artifact.

    Asserts the per-head degeneracy rather than assuming it (T-1809's own assertion).
    """
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)
    nonlinear = {e["name"]: float(e["scale"]) for e in meta["scales"]["nonlinear"]}
    k_scales, v_scales = [], []
    for layer in range(n_layers):
        ks = [nonlinear[f"layer{layer}.k_head{h}.scale"] for h in range(n_kv_heads)]
        vs = [nonlinear[f"layer{layer}.v_head{h}.scale"] for h in range(n_kv_heads)]
        assert len(set(ks)) == 1 and len(set(vs)) == 1, (
            f"layer{layer}: this artifact carries per-HEAD K/V scales "
            f"(k={ks}, v={vs}); this script's per-tensor read is no longer valid")
        k_scales.append(ks[0])
        v_scales.append(vs[0])
    return k_scales, v_scales


def install_forward(model, k_scales, v_scales, gate: ArmGate):
    """Install the one patched forward every arm runs. Returns the hook handles."""
    import torch
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    handles = []

    def make_landing_hook(site):
        def hook(_module, _args, output):
            landed, _delta = rt_dynamic(output)
            return gate.apply(site, landed, output)
        return hook

    # site 1 (embed) and sites 2/12 (the two RMSNorm outputs). `model.model.norm`
    # (final_norm) is deliberately NOT hooked -- the graded dump's last row is the last
    # decoder layer's PRE-final-norm output, so that site is outside the measured path.
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

            q_flat = attn.q_proj(hidden_states)
            k_flat = attn.k_proj(hidden_states)
            v_flat = attn.v_proj(hidden_states)

            # site 3: q_proj.requant -- dynamic, over the WHOLE 1536-wide row.
            q_landed, q_delta = rt_dynamic(q_flat)
            q_flat = gate.apply(3, q_landed, q_flat)
            # sites 4/5: the STATIC per-head K/V landing.
            if k_scale is not None:
                k_flat = gate.apply(4, rt_static(k_flat, k_scale), k_flat)
                v_flat = gate.apply(5, rt_static(v_flat, v_scale), v_flat)

            query_states = q_flat.view(hidden_shape).transpose(1, 2)
            key_states = k_flat.view(hidden_shape).transpose(1, 2)
            value_states = v_flat.view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # sites 6/7: the post-RoPE clamps. Each re-lands on the grid it already had --
            # q on site 3's own dynamic quantum, k on the static K scale.
            query_states = gate.apply(
                6, rt_on_grid(query_states, q_delta.unsqueeze(1)), query_states)
            if k_scale is not None:
                key_states = gate.apply(7, rt_static(key_states, k_scale), key_states)

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
            # site 8: the probability tensor's Q15 fixed-point width (floor).
            attn_weights = gate.apply(8, rt_prob_q15(attn_weights), attn_weights)

            attn_output = torch.matmul(attn_weights, v32)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous().to(
                hidden_states.dtype)
            # site 9: attn_ctx -- dynamic, over the whole concatenated context row.
            attn_output = gate.apply(9, rt_dynamic(attn_output)[0], attn_output)

            attn_output = attn.o_proj(attn_output)
            # site 10: o_proj.requant -- dynamic.
            attn_output = gate.apply(10, rt_dynamic(attn_output)[0], attn_output)
            return attn_output, None

        return forward

    def make_mlp_forward(mlp):
        def forward(x):
            gate_t = mlp.gate_proj(x)
            up = mlp.up_proj(x)
            # sites 13/14: gate_proj.requant and up_proj.requant -- dynamic.
            gate_t = gate.apply(13, rt_dynamic(gate_t)[0], gate_t)
            up = gate.apply(14, rt_dynamic(up)[0], up)
            sig = torch.sigmoid(gate_t.to(torch.float32)).to(gate_t.dtype)
            # site 15: the sigmoid's Q15 output width.
            sig = gate.apply(15, rt_sigmoid_q15(sig), sig)
            act = gate_t * sig * up
            # site 16: mlp_act -- dynamic, over gate * sigmoid * up.
            act = gate.apply(16, rt_dynamic(act)[0], act)
            out = mlp.down_proj(act)
            # site 17: down_proj.requant -- dynamic.
            out = gate.apply(17, rt_dynamic(out)[0], out)
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
            # site 11: attn_residual (C26) -- dynamic, over the reconciled sum.
            hidden_states = gate.apply(11, rt_dynamic(hidden_states)[0], hidden_states)

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            # site 18: mlp_residual (C26) -- dynamic.
            hidden_states = gate.apply(18, rt_dynamic(hidden_states)[0], hidden_states)
            return hidden_states
        return forward

    for idx, layer in enumerate(model.model.layers):
        layer.self_attn.forward = make_attention_forward(layer.self_attn, idx)
        layer.mlp.forward = make_mlp_forward(layer.mlp)
        layer.forward = make_layer_forward(layer)

    return handles


# ==============================================================================
# The capture -- T-1777's own composition, widened to the batch axis
# ==============================================================================

def capture_all_positions_batched(model, input_ids_b, device):
    """Every position's per-layer output for every batch element.

    `t1740_pooled_float_dump.capture_all_positions`, with the hook keeping `[:, -1, :]`
    (every arm) rather than `[0, -1, :]` (the first). Everything else -- the
    token-at-a-time incremental forward, the explicit DynamicCache, hooks on
    `embed_tokens` and each decoder layer rather than `output_hidden_states=True` -- is
    that function's own, and the equality of the two is gated by `t1835_gate_null.py`
    rather than asserted here.

    Returns {row_index: list over positions of (B, H) float32 numpy on the host}.
    """
    from transformers import DynamicCache
    import torch

    n_layers = model.config.num_hidden_layers
    captured = {i: [] for i in range(n_layers + 1)}

    def make_hook(idx):
        def hook(module, args, output):
            t = output if not isinstance(output, tuple) else output[0]
            captured[idx].append(t.detach()[:, -1, :].float().cpu().numpy())
        return hook

    handles = [model.model.embed_tokens.register_forward_hook(make_hook(0))]
    handles += [model.model.layers[i].register_forward_hook(make_hook(i + 1))
                for i in range(n_layers)]
    try:
        cache = DynamicCache()
        with torch.no_grad():
            for t in range(input_ids_b.shape[1]):
                model(input_ids=input_ids_b[:, t:t + 1], past_key_values=cache, use_cache=True)
    finally:
        for h in handles:
            h.remove()
    return captured


def endpoint_self_check_batched(model, captured, input_ids_b):
    """T-1777's `endpoint_self_check_last_position`, over the batch.

    The captured final-layer row at the last position must NOT be bit-identical to the
    model's own `last_hidden_state` obtained from a separate incremental re-run -- proving
    the capture bypassed `tie_last_hidden_states` rather than reproducing it. Checked for
    EVERY batch element, so a single arm silently falling through would fail it.
    """
    from transformers import DynamicCache
    import torch

    n_layers = model.config.num_hidden_layers
    last_layer_row = captured[n_layers][-1]                  # (B, H) numpy

    cache = DynamicCache()
    with torch.no_grad():
        for t in range(input_ids_b.shape[1]):
            out = model.model(input_ids=input_ids_b[:, t:t + 1], past_key_values=cache,
                              use_cache=True)
    model_own = out.last_hidden_state[:, -1, :].float().cpu().numpy()

    same = np.all(last_layer_row == model_own, axis=1)
    if bool(same.any()):
        idx = int(np.flatnonzero(same)[0])
        raise AssertionError(
            f"endpoint_self_check FAILED at batch element {idx}: the captured layer-"
            f"{n_layers} row at the last position is bit-identical to the model's own "
            f"last_hidden_state -- the capture reproduced the tie_last_hidden_states "
            f"overwrite instead of bypassing it")


# ==============================================================================

def _write_dump(path: Path, arr_bpH: list, arm_index: int, n_positions: int,
                n_rows: int, hidden: int, fingerprint: int):
    """T-1740's dump format for one arm: header then position-major, row-minor float32."""
    with open(path, "wb") as f:
        f.write(struct.pack("<QQQQ", n_positions, n_rows, hidden, fingerprint))
        for pos in range(n_positions):
            for row in range(n_rows):
                f.write(arr_bpH[row][pos][arm_index].astype("float32").tobytes())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", required=True, help="T-1777's own docs.jsonl (read-only)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--system", default=SYSTEM_PROMPT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--t1777-tools", required=True)
    parser.add_argument("--spike-root", default=r"D:\Wizard\Tools")
    parser.add_argument("--artifact-metadata",
                        default=r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct-w8a8"
                                r"\metadata.json")
    parser.add_argument("--single-arm", default=None,
                        help="run ONLY this arm, at batch 1, through the identical code")
    parser.add_argument("--dup-base", type=int, default=0,
                        help="insert N extra copies of the `base` configuration at spread "
                             "batch indices. Two purposes: the copies must come out "
                             "bit-identical to each other and to `base`, which is the "
                             "executed proof that a batched GEMM's output rows are "
                             "independent (the assumption every within-run contrast rests "
                             "on); and they change B, which changes the kernel the run "
                             "selects, so a second run at a different --dup-base is a "
                             "genuinely different numeric cell rather than a repeat.")
    parser.add_argument("--full-dump", action="append", default=[],
                        help="arm name to write full T-1740-format dumps for; repeatable")
    parser.add_argument("--no-drift", action="store_true",
                        help="skip the per-layer drift accumulation (single-arm gate runs)")
    parser.add_argument("--no-self-check", action="store_true",
                        help="skip the endpoint self-check (never used for a filed arm)")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    sys.path.insert(0, str(Path(args.spike_root).resolve()))
    import t1740_pooled_float_dump as fd  # noqa: E402

    all_arms = build_arms()
    if args.single_arm:
        sel = [a for a in all_arms if a[0] == args.single_arm]
        if not sel:
            raise SystemExit(f"unknown arm {args.single_arm!r}")
        arms = sel
    else:
        arms = list(all_arms)
        if args.dup_base:
            base_cfg = dict(all_arms)["base"]
            span = len(arms)
            for j in range(args.dup_base - 1, -1, -1):
                arms.insert(round(j * span / args.dup_base), (f"basedup{j+1}", base_cfg))

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

    gate = ArmGate(arms, device)
    k_scales, v_scales = read_kv_landing_scales(
        args.artifact_metadata, model.config.num_hidden_layers,
        model.config.num_key_value_heads)
    print(f"kv landing scales read from artifact: {len(k_scales)} layers, "
          f"layer0 k={k_scales[0]:.6g} v={v_scales[0]:.6g}, "
          f"layer27 k={k_scales[-1]:.6g} v={v_scales[-1]:.6g}", flush=True)

    handles = install_forward(model, k_scales, v_scales, gate)
    print(f"forward installed: B={gate.B} hooks={len(handles)}", flush=True)

    out_dir = Path(args.out_dir)
    (out_dir / "pooled").mkdir(parents=True, exist_ok=True)
    full_dump = [a for a in args.full_dump]
    for name in full_dump:
        if name not in gate.names:
            raise SystemExit(f"--full-dump {name}: not among the arms being run")
        (out_dir / "dumps" / name).mkdir(parents=True, exist_ok=True)
    want_drift = (not args.no_drift) and ("null" in gate.names) and gate.B > 1
    if want_drift:
        (out_dir / "drift").mkdir(parents=True, exist_ok=True)
        null_idx = gate.names.index("null")

    pooled = {name: {} for name in gate.names}
    fingerprints = {}
    drift_rel = []      # per sample: (B, rows)
    drift_cos = []
    drift_index = []    # (label, position)

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
            captured = capture_all_positions_batched(model, input_ids_b, device)
            if not args.no_self_check:
                endpoint_self_check_batched(model, captured, input_ids_b)
        except AssertionError as e:
            print(f"FAILED label={label}: {e}", flush=True)
            n_failed += 1
            continue

        n_layers = model.config.num_hidden_layers
        hidden = model.config.hidden_size
        n_rows = n_layers + 1
        fp = fd.fnv1a64(prompt_text)
        fingerprints[label] = fp

        # `t1777_retrieval_report.pooled_final_layer_float`: positions 1..n-1 of row r-1,
        # cast to float64, mean over the position axis.
        last_row = np.stack(captured[n_rows - 1], axis=0)          # (n_pos, B, H)
        pooled_arm = last_row[1:].astype(np.float64).mean(axis=0)  # (B, H)
        for b, name in enumerate(gate.names):
            pooled[name][label] = pooled_arm[b]

        for name in full_dump:
            b = gate.names.index(name)
            _write_dump(out_dir / "dumps" / name / f"{label}.float.bin",
                        captured, b, n_positions, n_rows, hidden, fp)

        if want_drift:
            # Per (position, arm, row): relative L2 and cosine of that arm's residual state
            # against the `null` element of the SAME batch. Accumulated one row at a time --
            # the whole (rows, positions, arms, hidden) float64 tensor is ~1 GB at these
            # sizes and this loop keeps the working set at one row of it.
            rel_doc = np.empty((n_positions, gate.B, n_rows), dtype=np.float32)
            cos_doc = np.empty((n_positions, gate.B, n_rows), dtype=np.float32)
            for r in range(n_rows):
                sr = np.stack(captured[r], axis=0).astype(np.float64)     # (n_pos, B, H)
                ref = sr[:, null_idx, :]                                  # (n_pos, H)
                ref_n = np.linalg.norm(ref, axis=-1)                      # (n_pos,)
                arm_n = np.linalg.norm(sr, axis=-1)                       # (n_pos, B)
                rel_doc[:, :, r] = (np.linalg.norm(sr - ref[:, None, :], axis=-1)
                                    / np.maximum(ref_n[:, None], 1e-300))
                dot = np.einsum("pbh,ph->pb", sr, ref)
                cos_doc[:, :, r] = dot / np.maximum(arm_n * ref_n[:, None], 1e-300)
                del sr, dot
            for p in range(n_positions):
                drift_rel.append(rel_doc[p])                              # (B, rows)
                drift_cos.append(cos_doc[p])
                drift_index.append((label, p))
            del rel_doc, cos_doc

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
    if want_drift and drift_rel:
        np.savez_compressed(
            out_dir / "drift" / "drift.npz",
            arms=np.array(gate.names),
            rel=np.stack(drift_rel),          # (samples, B, rows)
            cos=np.stack(drift_cos),
            labels=np.array([d[0] for d in drift_index]),
            positions=np.array([d[1] for d in drift_index], dtype=np.int32))
    with open(out_dir / "arms.json", "w", encoding="utf-8") as f:
        json.dump({name: sorted(cfg) for name, cfg in arms}, f, indent=2)

    print(f"batch_done: {n_ok} ok, {n_failed} failed (of {len(docs)}), arms={gate.B}, "
          f"capture_total={total:.1f}s avg={total/max(1,len(docs)):.2f}s/doc", flush=True)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
