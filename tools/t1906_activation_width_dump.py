#!/usr/bin/env python3
r"""T-1906 -- the activation-width curve: T-1809 arm C at widths between int8 and float.

T-1809 measured exactly two points on the activation-width axis: int8 (arm C, recall@1
0.6234) and float (arm A, the null, 0.9833). Nobody has measured a point in between. This
script is `t1809_activation_arm_dump.py` (`D:\SuperSLM\.worktrees\t1809-activation-interaction
\tools\t1809_activation_arm_dump.py`, commit `c385665`) with exactly ONE generalization: the
activation landings' symmetric integer range, fixed there at INT8_MAX=127, is now a CLI
parameter `--bits` (default 8, reproducing T-1809 arm C bit-exact). Every other line of the
construction -- the site list, the rounding rules, the K/V static-scale source, the float32
attention accumulation, the capture and self-check functions -- is unchanged from T-1809's own
file. This is deliberate: T-1906's whole validity rests on changing only the activation bit
width against T-1809 arm C's exact construction (`Claude/Brunel/t1906-activation-width-curve
-2026-08-11.md` SS3).

Only arm C's construction is exercised by this ticket (weights float, activations
round-tripped at `--bits`). The A/B/D arms and the weight-degradation path are retained
unmodified from the T-1809 file rather than deleted, because deleting code not on this
ticket's path is a bigger diff against the reused harness than leaving it inert; they are
not invoked by this ticket's capture runs.

**What the `--bits` generalization touches, and what it does NOT touch.** The width axis is
the symmetric integer range at the "dynamic" and "static" sites (site classes 1-7, 9-14,
16-18 in T-1809 SS2) -- everywhere the engine's own int8 round-trip appears. It does NOT
touch the two Q15 fixed-point sites, softmax probabilities (class 8) and the SiLU sigmoid
(class 15): those are a different representation with their own frac-bit width, not part of
the int8-family activation class this axis varies, and D-SLM2353's parked tier (residual
int16 carry, encoder Q16/K16/V8) is itself scoped to the same int-family sites, never to the
Q15 nonlinear outputs. Holding them fixed keeps arm C's own site set identical across every
width point, which SS3 of this ticket's brief requires.

Original T-1809 docstring, preserved because the construction it describes is unchanged below:
--------------------------------------------------------------------------------------------
Completes the 2x2 matrix T-1805 opened. T-1805 measured arm B (weights round-tripped,
activations float) through HuggingFace's own forward. This script measures all four cells
through ONE forward, so the interaction term D - B - C is an arithmetic over figures that
share a construction rather than across two of them:

    arm        weights          activations
    A          float            float          <- the null; the construction's own ceiling
    B          round-tripped    float          <- T-1805's arm, re-measured in this cell
    C          float            round-tripped
    D          round-tripped    round-tripped

Every arm runs the SAME patched control flow. `--arm` sets two booleans and nothing else:
whether the 196 projection weights are round-tripped, and whether the activation landings
fire. A null that ran through a different code path would leave the difference between the
paths inside every number this ticket reports.

The attention body accumulates in float32 while the rest of the forward stays in the
checkpoint's bfloat16. That is a fidelity choice, not a convenience: the engine accumulates
attention exactly in int32 (`pipeline.int_matmul`, envelope-checked), so float32 is the
closer analogue of what it does; and measured on this corpus, a bfloat16 eager accumulation
diverges from T-1777's own SDPA-produced reference by up to 45 absolute at the graded layer
(reference max-abs 250) where the float32 accumulation diverges by 3. The residual gap
between this forward and T-1777's is what arm A measures and bounds.

WHAT "ROUND-TRIPPING AN ACTIVATION" MEANS HERE, derived from the engine's own source
--------------------------------------------------------------------------------------
The shipped engine is the W8A8 **dynamic** chain (`src/forward/checked_chain_funnel.cpp`,
whose bit-equality oracle is `superslm_spike.dynamic_engine.forward_dynamic_vec`), not the
static-calibration path (`pipeline._vec_forward`). Two distinct activation-scale rules
exist in that chain and this script reproduces both:

  * **Dynamic, per token, per site** -- the rule at every site but one. The engine takes
    `D' = max|wide row|` over that site's row for that token (`_chain_record_vec`), normalizes
    it (`intmath.normalize_scale`), and emits
        q = clamp(round_half_away_from_zero(x * 127 * R / 2**(62-s)), -127, 127)
    with `R = round_half_up(2**62 / Dn)` (`intmath.dynamic_scale_reciprocal`,
    `intmath.requant_token_code`). Composing the three constants, that is exactly
        q = clamp(round_half_away_from_zero(x * 127 / D'), -127, 127)
    up to R's own ~2**-31 relative representation error, which this script does not model
    (named in the record's omission list). Dequantized: `x_hat = q * D' / 127`.
    The rule is scale-free -- it normalizes by the row's own max-abs -- so applying it to a
    float row is the exact analogue of applying it to the engine's integer wide row.

  * **Static, per K/V head** -- the ONE site with a calibrated scale. The engine lands k and
    v on `model.kv_landing_scales[f"layer{L}.{k|v}_head{h}"]`, an offline constant derived by
    `pipeline._derive_scales` from the calibration corpus's max-abs. Those scales are read
    here **from the engine's own calibrated artifact** (`metadata.json`'s
    `scales.nonlinear[layer{L}.{k|v}_head{h}.scale]`), never re-derived. On this artifact
    every head of a layer carries the same value (stated as a known degeneracy in
    `pipeline._derive_composition_constants`'s own docstring); the script asserts that
    rather than assuming it.

Two fixed-point (non-int8) activation representations are also modelled:
  * softmax probabilities in Q(PROB_FRAC_BITS=15), floor -- the engine computes
    `(e << 15) // total`, a floor, so the row's probabilities sum to slightly under 1.
  * the SiLU sigmoid in Q(SIGMOID_FRAC_BITS=15) -- `silu_lut.silu_sigmoid_q15`'s output width.

WHAT IS NOT MODELLED -- read the record's omission list; the short form:
  * the i-exp polynomial's own value error and the SiLU LUT's own interpolation error (only
    their OUTPUT WIDTHS are modelled; softmax and sigmoid values themselves are float here)
  * the integer RMSNorm (i-sqrt / Q16) -- the norm is float here
  * the C24/C25 per-channel weight-scale fold, the C28 bias reconcile, and the C26 residual
    reconcile's int roundings (all sub-quantum against the site requants that follow them)
  * `embed`, `lm_head`, and the RMSNorm gains' own weight quantization (single-scale path);
    arm D degrades exactly the 196 tensors arm B degraded, so B and D stay comparable
  * exact int32 matmul accumulation -- the projection matmuls here are the checkpoint's
    bfloat16; only the attention body accumulates in float32
  * `final_norm` -- the graded dump's last row is the last decoder layer's PRE-final-norm
    output, so this site is outside the measured path entirely

REUSED UNMODIFIED, BY IMPORT (this script owns none of them):
  * `superslm_spike.pipeline.quantize_weight_per_channel` / `dequantize_weight_per_channel`
    -- the engine's own weight quantizer, for arm D's weight side, identical to T-1805's.
  * `t1740_pooled_float_dump.capture_all_positions` / `endpoint_self_check_last_position`
    / `fnv1a64` / `_resolve_default_model` -- T-1777's own capture, read-only import from
    its worktree, so the dumps are byte-layout-identical to T-1777's own float dumps.

Usage
-----
    python tools\t1906_activation_width_dump.py --arm C --bits 16 \
        --docs <t1777>\out\t1777_corpus\docs.jsonl \
        --out-dir out\t1906_arm_c_b16 \
        --t1777-tools D:\SuperSLM\.worktrees\t1777-retrieval-agreement\tools \
        --artifact-metadata D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct-w8a8\metadata.json

`--bits 8` reproduces T-1809 arm C bit-exact (used for this ticket's fidelity check).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJ = ("gate_proj", "up_proj", "down_proj")

DEFAULT_BITS = 8
AMAX = (1 << (DEFAULT_BITS - 1)) - 1   # 127 at the default -- T-1809's INT8_MAX, generalized.
                                        # Set once in main() from --bits before install_forward
                                        # runs; every arm in this ticket's own runs is a single
                                        # process (T-1809's own constraint, unchanged here), so a
                                        # module global read by the closures below is safe and is
                                        # the smallest diff against the reused file.
MAX_EXACT_FLOAT32_BITS = 25   # T-1907 finding S3. AMAX = 2**(bits-1)-1 is exactly representable
                                # in float32 (24-bit mantissa) through bits=25 (AMAX=16777215 <
                                # 2**24) and stops at bits=26 (AMAX=33554431, float32 renders
                                # 33554432) -- verified this session with numpy.float32, matching
                                # the casebook's own independently measured cell.
SCALE_HEADROOM = 1.0 + 2.0 ** -20   # T-1907 finding C2. Mirrors pipeline._SCALE_HEADROOM
                                     # exactly -- the margin _output_scale applies to its
                                     # width-independent floor term so the requant multiplier
                                     # stays strictly representable (< 1) rather than touching
                                     # its own domain boundary.
INT8_REFERENCE_AMAX = 127   # NEVER reassigned. The artifact's own calibration grid width --
                             # pipeline._output_scale derives every K/V landing scale as
                             # calibration_maxabs / 127, so a scale read from the artifact is
                             # always an int8-grid value regardless of --bits. rt_static (T-1907
                             # finding C1) rescales that value to the CURRENT AMAX by the ratio
                             # INT8_REFERENCE_AMAX / AMAX; at bits=8 the ratio is exactly 1.0.
PROB_FRAC_BITS = 15      # pipeline.PROB_FRAC_BITS -- NOT part of the width axis, see docstring
SIGMOID_FRAC_BITS = 15   # pipeline.SIGMOID_FRAC_BITS -- NOT part of the width axis, see docstring


# ==============================================================================
# The two activation grids
# ==============================================================================

def _round_half_away_from_zero(t):
    """`pipeline._round_half_away_from_zero` in torch: sign(x) * floor(|x| + 0.5).

    Spelled out rather than `torch.round`, which is round-half-EVEN -- a different tie
    rule from the one C3 pins for the runtime and `intmath` implements.
    """
    import torch
    return torch.sign(t) * torch.floor(torch.abs(t) + 0.5)


def rt_dynamic(x):
    """The engine's per-token dynamic int8 round-trip over the LAST axis, generalized to AMAX.

    Returns (x_hat, delta) where `delta = D'/AMAX` is the row's quantum, needed by callers
    that must re-land a later value on the SAME grid (the post-RoPE clamp). At AMAX=127 (the
    default, `--bits 8`) this is T-1809's own rule, bit-exact.

    Computed in float32 regardless of the tensor's dtype: at bfloat16's 8 mantissa bits the
    quotient `x*AMAX/D'` cannot be rounded to an integer reliably, which would make the grid
    itself an artifact of the accumulator rather than of the engine's rule. The dequantized
    result is cast back to the input dtype, so the forward stays in the checkpoint's own
    compute precision.
    """
    import torch
    dtype = x.dtype
    xf = x.to(torch.float32)
    d = xf.abs().amax(dim=-1, keepdim=True)
    d = torch.where(d > 0, d, torch.ones_like(d))          # C20's all-zero guard
    q = _round_half_away_from_zero(xf * AMAX / d).clamp(-AMAX, AMAX)
    delta = d / AMAX
    return (q * delta).to(dtype), delta


def rt_static(x, scale, floor):
    """The engine's K/V landing, RE-DERIVED to an AMAX-wide grid -- T-1907 findings C1 (fixed
    round 1) and C2 (fixed this round).

    `scale` is always the ARTIFACT's calibrated int8-grid value `S_8`
    (`calibration_maxabs / 127`), read once and passed by every caller unchanged. `floor` is
    `pipeline._output_scale`'s SECOND term -- `input_product * (1 + 2**-20)`, a per-site
    constant that does NOT move with `AMAX` -- read from the SAME artifact by
    `read_kv_landing_scales` and passed alongside `scale`.

    `_output_scale`'s real rule is `max(calibration_maxabs / AMAX, floor)`, not the first term
    alone (`D-SLM2493`, T-1907 round 2 finding C2). Modelling only the first term, as this
    function did through round 1, matches the engine's own derivation at `bits<=12` (the
    calibrated term dominates at every one of the artifact's 56 K/V sites through there) and
    diverges above it -- measured on this artifact, 1 of 56 sites at `bits=14`, 38 of 56 at
    `bits=16`, worst ratio 5.38x, always in the direction that makes the round-1 formula's grid
    FINER than the engine's own rule would allow. `scale_b = max(scale * INT8_REFERENCE_AMAX /
    AMAX, floor)` reproduces the engine's actual derivation at both regimes, using inputs this
    tool already reads (no new artifact fields).

    At `bits=8`, `scale * INT8_REFERENCE_AMAX / AMAX == scale` exactly (`127/127 == 1.0`) and
    the calibrated term wins at all 56 sites by a minimum 47.9x margin (T-1907 round 2's own
    execution), so `floor` never binds and `scale_b == scale` bit-for-bit -- the fidelity check
    re-run after this fix must still pass 40/40 (`D-SLM2481`/`D-SLM2489`, reconfirmed again this
    round). Below the crossover (`--bits` roughly 8-12 on this artifact) resolution improves with
    width and the reconstruction is calibrated-term-bound, exactly as round 1's docstring
    described. ABOVE the crossover, `scale_b` is PINNED AT `floor` -- resolution stops
    improving with further width at that site, because `floor` does not move with `AMAX` at all.

    Saturation (T-1907 finding S2, corrected): the clip point is `AMAX * scale_b`. Below the
    crossover this is `AMAX * scale * 127 / AMAX == scale * 127` -- a CONSTANT, independent of
    `AMAX` -- so round 1's docstring claim that wider `bits` "saturates less often" was false at
    every width this harness could reach without the floor (measured: the layer-0 K clip point
    is `321.189087517720` at `bits` 8, 12, 16 and 25, identical to the last printed digit, under
    the round-1 formula). Once the floor binds, `AMAX * scale_b == AMAX * floor`, which DOES
    grow with `AMAX` -- so saturation only eases in the regime where resolution has already
    stopped improving, never in the regime where it is still improving. The two properties trade
    off; they do not both move together at any width.
    """
    import torch
    dtype = x.dtype
    xf = x.to(torch.float32)
    scale_b = max(scale * (INT8_REFERENCE_AMAX / AMAX), floor)
    q = _round_half_away_from_zero(xf / scale_b).clamp(-AMAX, AMAX)
    return (q * scale_b).to(dtype)


def rt_on_grid(x, delta):
    """Re-land a value on an ALREADY-CHOSEN grid, saturating at +/-AMAX codes.

    This is the post-RoPE step: the engine rotates int8 codes with Q30 tables (a rounding
    back onto the same grid) and clamps the result to [-AMAX, AMAX] without choosing a new
    scale. `delta` broadcasts against `x`'s trailing axes.
    """
    import torch
    dtype = x.dtype
    xf = x.to(torch.float32)
    q = _round_half_away_from_zero(xf / delta).clamp(-AMAX, AMAX)
    return (q * delta).to(dtype)


def rt_prob_q15(p):
    """The engine's probability representation: `(e << 15) // total`, i.e. floor to Q15.

    Floor, not round: the engine's own integer division truncates, which is why its rows sum
    to slightly under one. Reproducing the rounding direction matters because the deficit is
    systematic rather than zero-mean.
    """
    import torch
    dtype = p.dtype
    pf = p.to(torch.float32)
    unit = float(1 << PROB_FRAC_BITS)
    return (torch.floor(pf * unit) / unit).to(dtype)


def rt_sigmoid_q15(s):
    """The SiLU sigmoid's Q15 output width (`silu_lut.silu_sigmoid_q15`).

    The LUT's own interpolation error is NOT modelled -- only the width of the value it
    returns. Round-half-away rather than floor: the LUT interpolates toward the nearest
    node rather than truncating.
    """
    import torch
    dtype = s.dtype
    sf = s.to(torch.float32)
    unit = float(1 << SIGMOID_FRAC_BITS)
    return (_round_half_away_from_zero(sf * unit) / unit).to(dtype)


# ==============================================================================
# Arm D's weight side -- identical in scope and quantizer to T-1805's arm B
# ==============================================================================

def _iter_projection_modules(model):
    n_layers = model.config.num_hidden_layers
    for i in range(n_layers):
        layer = model.model.layers[i]
        for proj in _ATTN_PROJ:
            yield f"layer{i}.{proj}", getattr(layer.self_attn, proj)
        for proj in _MLP_PROJ:
            yield f"layer{i}.{proj}", getattr(layer.mlp, proj)


def degrade_weights(model, quantize_weight_per_channel, dequantize_weight_per_channel):
    """T-1805's arm-B weight substitution, verbatim in scope and quantizer.

    196 tensors (28 layers x 7 projections), per-output-channel symmetric int8 max-abs,
    output_axis=0 (an `nn.Linear` weight is [out_features, in_features] and the engine's
    `_PROJECTION_OUTPUT_AXIS = 0` scales per row of that layout).
    """
    import numpy as np
    import torch

    n_touched = 0
    for _name, module in _iter_projection_modules(model):
        w = module.weight.detach()
        w_np = w.to(torch.float32).cpu().numpy().astype(np.float64)
        codes, scales = quantize_weight_per_channel(w_np, output_axis=0)
        deq = dequantize_weight_per_channel(codes, scales, output_axis=0)
        deq_t = torch.from_numpy(deq.astype(np.float32)).to(dtype=w.dtype, device=w.device)
        with torch.no_grad():
            module.weight.copy_(deq_t)
        n_touched += 1
    assert n_touched == 196, f"expected 196 projection tensors, touched {n_touched}"
    return n_touched


# ==============================================================================
# The patched forward -- HuggingFace's own modules, quantized at the engine's sites
# ==============================================================================

def read_kv_landing_scales(metadata_path, n_layers, n_kv_heads):
    """The engine's calibrated K/V landing scales AND their width-independent floors, read
    from the artifact -- T-1907 finding C2 (fixed this round; the floor is new, the scale read
    is unchanged from round 1).

    Asserts the per-head degeneracy rather than assuming it: `_derive_scales` derives one
    scale per K (respectively V) TENSOR and writes it to every head, so a real per-head
    calibration would break this assert loudly instead of being silently averaged away.

    The floor is `pipeline._output_scale`'s second term for this site,
    `input_product * SCALE_HEADROOM` where `input_product = input_scale * max(weight_scales)`
    (`pipeline.projection_scale`'s own composition) -- read from `scales.requant[layer{L}.
    {k|v}_proj.requant]`, the SAME artifact this function already opens, no new file. The
    requant entry's own `output_scale` is asserted equal to the nonlinear per-head scale
    already read for that layer, confirming this reads the site that actually produced
    `k_scales`/`v_scales` rather than a lookalike (T-1907 round 2's own verification method,
    reproduced here as a structural check rather than a one-off probe).
    """
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)
    nonlinear = {e["name"]: float(e["scale"]) for e in meta["scales"]["nonlinear"]}
    requant = {e["name"]: e for e in meta["scales"]["requant"]}
    k_scales, v_scales, k_floors, v_floors = [], [], [], []
    for layer in range(n_layers):
        ks = [nonlinear[f"layer{layer}.k_head{h}.scale"] for h in range(n_kv_heads)]
        vs = [nonlinear[f"layer{layer}.v_head{h}.scale"] for h in range(n_kv_heads)]
        assert len(set(ks)) == 1 and len(set(vs)) == 1, (
            f"layer{layer}: this artifact carries per-HEAD K/V scales "
            f"(k={ks}, v={vs}); this script's per-tensor read is no longer valid"
        )
        k_scales.append(ks[0])
        v_scales.append(vs[0])
        for proj, scale_val, floors in (("k", ks[0], k_floors), ("v", vs[0], v_floors)):
            entry = requant[f"layer{layer}.{proj}_proj.requant"]
            assert entry["output_scale"] == scale_val, (
                f"layer{layer}.{proj}_proj.requant output_scale "
                f"({entry['output_scale']}) != the nonlinear per-head scale ({scale_val}) -- "
                f"this is not the site that produced k_scales/v_scales"
            )
            input_product = float(entry["input_scale"]) * max(entry["weight_scales"])
            floors.append(input_product * SCALE_HEADROOM)
    return k_scales, v_scales, k_floors, v_floors


def install_forward(model, k_scales, v_scales, k_floors, v_floors, quantize_activations: bool):
    """Install the one patched forward every arm runs.

    `quantize_activations` is the ONLY difference between the null and the activation arms:
    when it is False every landing below degenerates to the identity and the surrounding
    arithmetic is untouched. There is no second implementation for the null, so the null
    cannot drift from the arms it is the ceiling for.

    Patched by bound method on the INSTANCES rather than on the class, so nothing leaks into
    another model built in the same process and no import-time global state exists.

    Returns the list of hook handles (the caller keeps them alive; removing them would
    silently un-quantize the embedding and the norms).
    """
    import torch
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    on = quantize_activations
    handles = []

    # --- sites: embed (C23), attn_norm.out and mlp_norm.out (C23 scale-killing) ---
    # Installed as forward hooks so the module's own arithmetic is HuggingFace's untouched.
    # `model.model.norm` (final_norm) is deliberately NOT hooked: the graded dump's last row
    # is the last decoder layer's PRE-final-norm output, so that site is outside the path
    # this ticket measures.
    def dynamic_landing_hook(_module, _args, output):
        out, _delta = rt_dynamic(output)
        return out

    if on:
        handles.append(model.model.embed_tokens.register_forward_hook(dynamic_landing_hook))
        for layer in model.model.layers:
            handles.append(layer.input_layernorm.register_forward_hook(dynamic_landing_hook))
            handles.append(
                layer.post_attention_layernorm.register_forward_hook(dynamic_landing_hook))

    def make_attention_forward(attn, layer_idx):
        """`Qwen2Attention.forward` with the engine's activation landings inserted.

        Every line that is not a landing is the stock implementation's own line, in its own
        order, calling the stock modules. The one deliberate departure from stock is the
        float32 attention accumulation, argued in the module docstring; it is present in
        every arm including the null, so it cancels out of every comparison this ticket makes.
        """
        k_scale = k_scales[layer_idx] if k_scales is not None else None
        v_scale = v_scales[layer_idx] if v_scales is not None else None
        k_floor = k_floors[layer_idx] if k_floors is not None else None
        v_floor = v_floors[layer_idx] if v_floors is not None else None

        def forward(hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn.head_dim)

            q_flat = attn.q_proj(hidden_states)
            k_flat = attn.k_proj(hidden_states)
            v_flat = attn.v_proj(hidden_states)
            q_delta = None
            if on:
                # site: q_proj.requant -- dynamic, over the WHOLE row (all heads), one D'
                # per token, matching `_chain_record_vec(..., "q_proj.requant", t, folded)`
                # where `folded` is the full 1536-wide accumulator row.
                q_flat, q_delta = rt_dynamic(q_flat)
                # sites: k_proj / v_proj -- STATIC per-head landing, the calibrated scale.
                k_flat = rt_static(k_flat, k_scale, k_floor)
                v_flat = rt_static(v_flat, v_scale, v_floor)

            query_states = q_flat.view(hidden_shape).transpose(1, 2)
            key_states = k_flat.view(hidden_shape).transpose(1, 2)
            value_states = v_flat.view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if on:
                # sites: the post-RoPE clamps. The engine rotates int8 codes and clamps to
                # [-127, 127] WITHOUT choosing a new scale, so each side re-lands on the grid
                # it already had: q on its own dynamic quantum, k on the static K scale.
                # q_delta is (B, S, 1) over the pre-view layout; unsqueeze(1) makes it
                # (B, 1, S, 1), which broadcasts against (B, heads, S, head_dim).
                query_states = rt_on_grid(query_states, q_delta.unsqueeze(1))
                key_states = rt_static(key_states, k_scale, k_floor)

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
            if on:
                # site: the probability tensor's Q15 fixed-point width (floor).
                attn_weights = rt_prob_q15(attn_weights)

            attn_output = torch.matmul(attn_weights, v32)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous().to(
                hidden_states.dtype)
            if on:
                # site: attn_ctx -- dynamic, over the whole concatenated context row.
                attn_output, _ = rt_dynamic(attn_output)

            attn_output = attn.o_proj(attn_output)
            if on:
                # site: o_proj.requant -- dynamic.
                attn_output, _ = rt_dynamic(attn_output)
            return attn_output, None

        return forward

    def make_mlp_forward(mlp):
        def forward(x):
            gate = mlp.gate_proj(x)
            up = mlp.up_proj(x)
            if on:
                # sites: gate_proj.requant and up_proj.requant -- dynamic.
                gate, _ = rt_dynamic(gate)
                up, _ = rt_dynamic(up)
            # SiLU is x*sigmoid(x). The engine multiplies the gate CODE by a Q15 sigmoid, so
            # the sigmoid is the quantized factor and the product is not quantized here --
            # `mlp_act` below is where the product lands.
            sig = torch.sigmoid(gate.to(torch.float32)).to(gate.dtype)
            if on:
                # site: the sigmoid's Q15 output width.
                sig = rt_sigmoid_q15(sig)
            act = gate * sig * up
            if on:
                # site: mlp_act -- dynamic, over gate * sigmoid * up.
                act, _ = rt_dynamic(act)
            out = mlp.down_proj(act)
            if on:
                # site: down_proj.requant -- dynamic.
                out, _ = rt_dynamic(out)
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
            if on:
                # site: attn_residual (C26) -- dynamic, over the reconciled sum.
                hidden_states, _ = rt_dynamic(hidden_states)

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            if on:
                # site: mlp_residual (C26) -- dynamic.
                hidden_states, _ = rt_dynamic(hidden_states)
            return hidden_states
        return forward

    for idx, layer in enumerate(model.model.layers):
        layer.self_attn.forward = make_attention_forward(layer.self_attn, idx)
        layer.mlp.forward = make_mlp_forward(layer.mlp)
        layer.forward = make_layer_forward(layer)

    return handles


# ==============================================================================
# T-1907 finding S1 (round 1) -- width provenance. A `.float.bin` file carries no arm or
# bit-width field (its 4-word header is `n_positions, n_layers+1, hidden_size,
# prompt_fingerprint`, identical to T-1777's own format, which `t1906_grade_arms.py` must keep
# reading unmodified). The provenance this ticket's width axis needs lives in a directory-level
# sidecar instead: `manifest.json`, written into `--out-dir`.
#
# T-1907 finding C1 (round 2) -- the manifest as filed in round 1 certified the RUN that last
# wrote into a directory, not the FILES in it: `main()` never cleared `--out-dir`, so a
# `--limit`/`--bits` re-run into a populated directory left a manifest describing the new run
# sitting beside files from an old one, and the grader trusted the manifest's `bits` without
# ever checking that its `n_ok`/label set matched what was actually on disk. Two changes close
# it: `main()` now refuses (or clears, with `--overwrite`) a non-empty `--out-dir` before
# capture starts, so a manifest can only ever describe the directory it is sitting in; and the
# manifest now carries the exact label SET captured (`labels`), which the grader requires to
# equal the set it actually graded -- strictly stronger than comparing counts, at the cost of
# one field.
# ==============================================================================

MANIFEST_NAME = "manifest.json"


def write_manifest(out_dir: Path, *, arm: str, bits: int, amax: int, docs_source: str,
                    complete: bool, n_ok: int = None, n_failed: int = None,
                    labels: list = None) -> None:
    """Write/overwrite this directory's own provenance record. Called twice per run: once
    before capture starts (`complete=False`, so an interrupted run still leaves a directory
    that HONESTLY reports it never finished, rather than no manifest at all -- a missing
    manifest and an incomplete one are different failures and the grader treats them
    differently), and once after (`complete=True`, with final counts and the exact label set
    captured, T-1907 finding C1 round 2).
    """
    import json as _json
    manifest = {
        "tool": "t1906_activation_width_dump.py",
        "arm": arm,
        "bits": bits,
        "amax": amax,
        "docs_source": docs_source,
        "complete": complete,
    }
    if n_ok is not None:
        manifest["n_ok"] = n_ok
    if n_failed is not None:
        manifest["n_failed"] = n_failed
    if labels is not None:
        manifest["labels"] = labels
    with open(out_dir / MANIFEST_NAME, "w", encoding="utf-8") as f:
        _json.dump(manifest, f, indent=2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=("A", "B", "C", "D"),
                        help="A = null, B = weights only, C = activations only, "
                             "D = both. This ticket's captures use C only.")
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS,
                        help="symmetric integer width at the dynamic/static activation "
                             "sites (T-1809's site classes 1-7/9-14/16-18); AMAX = "
                             "2**(bits-1)-1. Default 8 reproduces T-1809 arm C bit-exact. "
                             "Does NOT affect the Q15 softmax/sigmoid sites (classes 8/15). "
                             "Bounded to [2, 25] -- below 2, AMAX=0; above 25, AMAX is no "
                             "longer exactly representable in float32 (T-1907 finding S3).")
    parser.add_argument("--docs", required=True, help="T-1777's own docs.jsonl (read-only)")
    parser.add_argument("--out-dir", required=True,
                         help="must be empty or not-yet-created unless --overwrite is passed "
                              "(T-1907 finding C1, round 2) -- a capture into a populated "
                              "directory used to MERGE with whatever was already there, leaving "
                              "a manifest describing the new run beside files from an old one.")
    parser.add_argument("--overwrite", action="store_true",
                         help="clear --out-dir before capture if it already holds files, "
                              "instead of refusing (T-1907 finding C1, round 2)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--system", default=SYSTEM_PROMPT)
    parser.add_argument("--limit", type=int, default=None,
                        help="capture only the first N documents (smoke runs, fidelity subset)")
    parser.add_argument("--t1777-tools", required=True)
    parser.add_argument("--spike-root", default=r"D:\Wizard\Tools")
    parser.add_argument("--artifact-metadata",
                        default=r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct-w8a8\metadata.json",
                        help="the engine's own calibrated artifact metadata -- the source of "
                             "the static K/V landing scales; read-only, never re-derived")
    args = parser.parse_args(argv)

    if args.bits < 2:
        raise SystemExit(f"--bits must be >= 2 (got {args.bits}); a 1-bit symmetric signed "
                          f"range has AMAX=0 and every landing collapses to zero")
    if args.bits > MAX_EXACT_FLOAT32_BITS:
        raise SystemExit(
            f"--bits must be <= {MAX_EXACT_FLOAT32_BITS} (got {args.bits}); AMAX = "
            f"2**(bits-1)-1 stops being exactly representable in float32 at bits="
            f"{MAX_EXACT_FLOAT32_BITS + 1} (float32 has a 24-bit mantissa; AMAX first exceeds "
            f"2**24 there), and every round-trip in this harness computes in float32 "
            f"(T-1907 finding S3) -- confirmed by execution: bits=26 gives AMAX=33554431, "
            f"which float32 renders as 33554432")
    global AMAX
    AMAX = (1 << (args.bits - 1)) - 1
    print(f"activation width: --bits {args.bits} -> AMAX={AMAX} "
          f"({'bit-exact reproduction of T-1809 arm C' if args.bits == 8 else 'width sweep point'})",
          flush=True)

    # T-1907 finding C1 (round 2). Checked before the model load (minutes-long) rather than
    # after, so a bad invocation fails fast: a capture used to MERGE into whatever `--out-dir`
    # already held -- writing over labels it reproduces and leaving every other file untouched
    # -- so a directory could carry files from more than one run while its manifest described
    # only the last one. Refuse by default; `--overwrite` clears the directory first.
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        existing = [p for p in out_dir.iterdir() if p.is_file()]
        if existing:
            if not args.overwrite:
                raise SystemExit(
                    f"REFUSED: --out-dir {out_dir} already holds {len(existing)} file(s) "
                    f"(e.g. {existing[0].name}) -- capturing into it would MERGE this run's "
                    f"output with whatever is already there, and the manifest this run writes "
                    f"would then describe files it did not produce (T-1907 finding C1, round "
                    f"2). Pass --overwrite to clear the directory first, or choose a fresh "
                    f"--out-dir.")
            for p in existing:
                p.unlink()
            print(f"--overwrite: cleared {len(existing)} pre-existing file(s) from {out_dir} "
                  f"before capture", flush=True)

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    sys.path.insert(0, str(Path(args.spike_root).resolve()))
    import t1740_pooled_float_dump as fd  # noqa: E402

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
    print(f"arm={args.arm} bits={args.bits} model+tokenizer load: "
          f"{time.perf_counter() - t0:.2f}s device={device} "
          f"documents={len(docs)} resolved_model_dtype={model.dtype}", flush=True)

    quantize_weights = args.arm in ("B", "D")
    quantize_activations = args.arm in ("C", "D")

    if quantize_weights:
        from superslm_spike.pipeline import (  # noqa: E402
            quantize_weight_per_channel, dequantize_weight_per_channel)
        t_deg = time.perf_counter()
        n = degrade_weights(model, quantize_weight_per_channel, dequantize_weight_per_channel)
        print(f"weight degradation: {n}/196 projection tensors round-tripped through the "
              f"engine's own int8 per-output-channel quantizer in "
              f"{time.perf_counter() - t_deg:.2f}s", flush=True)

    k_scales = v_scales = k_floors = v_floors = None
    if quantize_activations:
        k_scales, v_scales, k_floors, v_floors = read_kv_landing_scales(
            args.artifact_metadata, model.config.num_hidden_layers,
            model.config.num_key_value_heads)
        print(f"kv landing scales read from artifact: {len(k_scales)} layers, "
              f"layer0 k={k_scales[0]:.6g} v={v_scales[0]:.6g}, "
              f"layer27 k={k_scales[-1]:.6g} v={v_scales[-1]:.6g}, "
              f"layer0 k_floor={k_floors[0]:.6g} v_floor={v_floors[0]:.6g} "
              f"(T-1907 C2)", flush=True)

    handles = install_forward(model, k_scales, v_scales, k_floors, v_floors,
                               quantize_activations)
    print(f"forward installed: arm={args.arm} quantize_weights={quantize_weights} "
          f"quantize_activations={quantize_activations} hooks={len(handles)}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)   # already refused/cleared above if populated
    write_manifest(out_dir, arm=args.arm, bits=args.bits, amax=AMAX,
                    docs_source=str(Path(args.docs).resolve()), complete=False)

    n_ok = n_failed = 0
    labels_ok = []
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

        try:
            captured = fd.capture_all_positions(model, input_ids, device)
            fd.endpoint_self_check_last_position(model, captured, input_ids, device)
        except AssertionError as e:
            print(f"FAILED label={label}: {e}", flush=True)
            n_failed += 1
            continue

        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        fingerprint = fd.fnv1a64(prompt_text)
        with open(out_dir / f"{label}.float.bin", "wb") as fbin:
            fbin.write(struct.pack("<QQQQ", n_positions, n_layers + 1, hidden_size, fingerprint))
            for pos in range(n_positions):
                for idx in range(n_layers + 1):
                    fbin.write(captured[idx][pos].cpu().numpy().astype("float32").tobytes())
        del captured
        if device == "cuda":
            torch.cuda.empty_cache()
        n_ok += 1
        labels_ok.append(label)
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(docs)}] label={label} n_pos={n_positions} "
                  f"this_doc={time.perf_counter() - t_doc:.2f}s elapsed={elapsed:.1f}s "
                  f"avg={elapsed/(i+1):.2f}s/doc", flush=True)

    total = time.perf_counter() - t_start
    print(f"batch_done: arm={args.arm} bits={args.bits} {n_ok} ok, {n_failed} failed "
          f"(of {len(docs)}), capture_total={total:.1f}s avg={total/max(1,len(docs)):.2f}s/doc",
          flush=True)
    write_manifest(out_dir, arm=args.arm, bits=args.bits, amax=AMAX,
                    docs_source=str(Path(args.docs).resolve()), complete=(n_failed == 0),
                    n_ok=n_ok, n_failed=n_failed, labels=labels_ok)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
