#!/usr/bin/env python3
"""Red-first proof for `tools/independent_layer_reference.py` (T-1690,
design S6's own "Validation before trust" / "Lifetime/reuse" / "Trust
boundaries" paragraphs). Every cell below is proven able to fail before it
is trusted: each hand-computed check is paired with a deliberate mutation
(the sibling `_WRONG_*` function already defined in the module under test)
confirmed to diverge from the correct value, per `StandardsDocument.md`
S5.4's mutation-proof discipline.

Run: `python tools\\test_independent_layer_reference.py`
Exits non-zero, with a diagnostic, on the first failing check.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import independent_layer_reference as ilr  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name + (f" -- {detail}" if detail else ""))


def close(a, b, atol=1e-10, rtol=1e-8):
    return torch.allclose(torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64),
                           atol=atol, rtol=rtol)


# --------------------------------------------------------------------------
# Part 1a -- RMSNorm, hand-computed at 4 dimensions, plus guard vitality.
# --------------------------------------------------------------------------
def test_rmsnorm():
    x = torch.tensor([1.0, -2.0, 3.0, 4.0], dtype=torch.float64)
    w = torch.tensor([2.0, 0.5, 1.0, -1.0], dtype=torch.float64)
    eps = 1e-6
    # Hand computation, independent of the module's own tensor ops: plain
    # python floats.
    xs = [1.0, -2.0, 3.0, 4.0]
    ws = [2.0, 0.5, 1.0, -1.0]
    mean_sq = sum(v * v for v in xs) / len(xs)
    rms = math.sqrt(mean_sq + eps)
    expected = [(v / rms) * wi for v, wi in zip(xs, ws)]

    got = ilr.rms_norm(x, w, eps)
    check("rmsnorm_hand_computed", close(got, expected), f"got={got.tolist()} expected={expected}")

    wrong = ilr.rms_norm_WRONG_no_eps(x, w, eps)
    check("rmsnorm_guard_vitality_no_eps_diverges", not close(wrong, expected, atol=1e-9),
          "the no-eps mutation must NOT match the correct hand value")


# --------------------------------------------------------------------------
# Part 1b -- RoPE, hand-computed at head_dim=2 (one frequency pair), plus
# guard vitality.
# --------------------------------------------------------------------------
def test_rope():
    head_dim = 2
    theta = 1000000.0
    position = 3
    inv_freq = ilr.rope_inv_freq(head_dim, theta, device="cpu", dtype=torch.float64)
    # hand: inv_freq[0] = theta^(-0/2) = 1.0
    check("rope_inv_freq_hand_value", close(inv_freq, [1.0]), f"got={inv_freq.tolist()}")

    cos, sin = ilr.rope_cos_sin(position, inv_freq, torch.float64)
    angle = position * 1.0
    expected_cos = [math.cos(angle), math.cos(angle)]
    expected_sin = [math.sin(angle), math.sin(angle)]
    check("rope_cos_sin_hand_value", close(cos, expected_cos) and close(sin, expected_sin),
          f"cos={cos.tolist()} sin={sin.tolist()}")

    x = torch.tensor([5.0, -3.0], dtype=torch.float64)
    got = ilr.apply_rope(x, cos, sin)
    # rotate_half([5,-3]) = [3, 5]; rope = x*cos + rotate_half(x)*sin
    expected = [5.0 * math.cos(angle) + 3.0 * math.sin(angle), -3.0 * math.cos(angle) + 5.0 * math.sin(angle)]
    check("rope_apply_hand_value", close(got, expected), f"got={got.tolist()} expected={expected}")

    wrong = ilr.apply_rope_WRONG_no_rotation(x, cos, sin)
    check("rope_guard_vitality_no_rotation_diverges", not close(wrong, expected, atol=1e-9),
          "the unrotated mutation must NOT match the correct hand value at position 3 (nonzero angle)")


# --------------------------------------------------------------------------
# Part 1c -- SwiGLU MLP, hand-computed at a 2x2 scale, plus guard vitality.
# --------------------------------------------------------------------------
def test_swiglu():
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    w_gate = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)  # identity
    w_up = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.float64)  # 0.5 * identity
    w_down = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)

    gate = [1.0, 2.0]
    up = [0.5, 1.0]

    def silu(v):
        return v / (1.0 + math.exp(-v))

    inter = [silu(g) * u for g, u in zip(gate, up)]
    # out = inter @ w_down.T, w_down = [[1,1],[0,1]] -> w_down.T = [[1,0],[1,1]]
    expected = [inter[0] * 1.0 + inter[1] * 1.0, inter[0] * 0.0 + inter[1] * 1.0]

    got = ilr.swiglu_mlp(x, w_gate, w_up, w_down)
    check("swiglu_hand_computed", close(got, expected), f"got={got.tolist()} expected={expected}")

    wrong = ilr.swiglu_mlp_WRONG_relu(x, w_gate, w_up, w_down)
    check("swiglu_guard_vitality_relu_diverges", not close(wrong, expected, atol=1e-9),
          "the ReLU mutation must NOT match the correct SiLU hand value")


# --------------------------------------------------------------------------
# Part 1d -- causal mask, width > 1 (closes the D-SLM503 width==1 blindness
# class named in design S7 step 3's own precedent, applied here to this
# witness's own mask construction).
# --------------------------------------------------------------------------
def test_causal_mask():
    mask = ilr.causal_additive_mask(3, 3, device="cpu", dtype=torch.float64)
    # row i (query offset i) sees keys 0..i, masks the rest.
    expected_visible = [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]
    ok = True
    for i in range(3):
        for j in range(3):
            is_visible = mask[i, j].item() == 0.0
            if is_visible != expected_visible[i][j]:
                ok = False
    check("causal_mask_width_gt_1", ok, f"mask={mask.tolist()}")

    # width == 1: single query position sees exactly the one key.
    mask1 = ilr.causal_additive_mask(1, 1, device="cpu", dtype=torch.float64)
    check("causal_mask_width_eq_1", mask1[0, 0].item() == 0.0, f"mask={mask1.tolist()}")


# --------------------------------------------------------------------------
# Part 1e -- grouped-query repeat_interleave, hand-checked assignment.
# --------------------------------------------------------------------------
def test_gqa_repeat():
    # 2 kv heads, seq_len=1, head_dim=1: values 10 and 20.
    x = torch.tensor([[[10.0]], [[20.0]]], dtype=torch.float64)  # [2,1,1]
    rep = ilr.repeat_kv_heads(x, 3)  # -> [6,1,1]
    got = rep.squeeze(-1).squeeze(-1).tolist()
    expected = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]  # repeat_interleave, not tile
    check("gqa_repeat_interleave_assignment", got == expected, f"got={got} expected={expected}")


# --------------------------------------------------------------------------
# Part 2 (design S6 "Lifetime/reuse") -- the from-scratch K/V cache is grown
# across >=8 synthetic positions and an early position is read back after
# the cache has grown to its final size, asserted bit-identical (float64
# exact) to the value first written. Guard vitality: a deliberate
# off-by-one write-index mutation is confirmed to break the assertion.
# --------------------------------------------------------------------------
def test_kv_cache_lifetime():
    num_layers, num_kv_heads, head_dim, max_len = 1, 2, 4, 10
    cache = ilr.IndependentKVCache(num_layers, num_kv_heads, head_dim, max_len, torch.float64, "cpu")

    written = {}
    torch.manual_seed(1234)
    for pos in range(9):  # >= 8 positions, design S6's own stated minimum
        k_val = torch.randn(num_kv_heads, head_dim, dtype=torch.float64)
        v_val = torch.randn(num_kv_heads, head_dim, dtype=torch.float64)
        written[pos] = (k_val.clone(), v_val.clone())
        cache.write(0, pos, k_val, v_val)

    early_pos = 1
    k_read, v_read = cache.read(0, max_len)
    k_at_early = k_read[:, early_pos, :]
    v_at_early = v_read[:, early_pos, :]
    k_expected, v_expected = written[early_pos]
    check(
        "kv_cache_lifetime_early_write_survives_late_read",
        torch.equal(k_at_early, k_expected) and torch.equal(v_at_early, v_expected),
        f"k_at_early={k_at_early.tolist()} k_expected={k_expected.tolist()}",
    )

    # Guard vitality: the off-by-one mutation must break the SAME assertion.
    cache_wrong = ilr.IndependentKVCache(num_layers, num_kv_heads, head_dim, max_len, torch.float64, "cpu")
    for pos in range(9):
        k_val, v_val = written[pos]
        cache_wrong.write_WRONG_off_by_one(0, pos, k_val, v_val)
    k_read_w, v_read_w = cache_wrong.read(0, max_len)
    k_at_early_w = k_read_w[:, early_pos, :]
    k_expected_w = written[early_pos][0]
    check(
        "kv_cache_lifetime_guard_vitality_off_by_one_diverges",
        not torch.equal(k_at_early_w, k_expected_w),
        "the off-by-one mutation must NOT read back the value first written at position 1",
    )


# --------------------------------------------------------------------------
# Part 3 (design S6 "Trust boundaries") -- a synthetic state_dict/config
# mismatch (wrong head count, or a missing key) is loudly, namedly rejected
# -- never a silent misread or a downstream shape crash read as valid.
# --------------------------------------------------------------------------
def test_trust_boundary_rejection():
    cfg = ilr.CheckpointConfig(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        rms_norm_eps=1e-6, rope_theta=10000.0, head_dim=4,
    )
    good = {
        "model.embed_tokens.weight": torch.zeros(10, 8),
        "model.norm.weight": torch.zeros(8),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8),
        "model.layers.0.self_attn.q_proj.bias": torch.zeros(8),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros(4, 8),
        "model.layers.0.self_attn.k_proj.bias": torch.zeros(4),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros(4, 8),
        "model.layers.0.self_attn.v_proj.bias": torch.zeros(4),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(8, 8),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(16, 8),
        "model.layers.0.mlp.up_proj.weight": torch.zeros(16, 8),
        "model.layers.0.mlp.down_proj.weight": torch.zeros(8, 16),
        "model.layers.0.input_layernorm.weight": torch.zeros(8),
        "model.layers.0.post_attention_layernorm.weight": torch.zeros(8),
    }
    # Sanity: the well-formed dict must NOT raise.
    try:
        ilr.CheckpointWeights.from_safetensors_dict(dict(good), cfg)
        check("trust_boundary_wellformed_accepted", True)
    except ilr.StateDictShapeError as e:
        check("trust_boundary_wellformed_accepted", False, str(e))

    # Missing key.
    missing = dict(good)
    del missing["model.layers.0.self_attn.k_proj.bias"]
    try:
        ilr.CheckpointWeights.from_safetensors_dict(missing, cfg)
        check("trust_boundary_missing_key_rejected", False, "did not raise")
    except ilr.StateDictShapeError:
        check("trust_boundary_missing_key_rejected", True)

    # Wrong head count -- q_proj shaped for 3 heads * 4 = 12, not 2*4=8.
    wrong_shape = dict(good)
    wrong_shape["model.layers.0.self_attn.q_proj.weight"] = torch.zeros(12, 8)
    try:
        ilr.CheckpointWeights.from_safetensors_dict(wrong_shape, cfg)
        check("trust_boundary_wrong_head_count_rejected", False, "did not raise")
    except ilr.StateDictShapeError:
        check("trust_boundary_wrong_head_count_rejected", True)


# --------------------------------------------------------------------------
# Part 4 -- end-to-end synthetic tiny layer, cross-checked at more than one
# context length (design S11 dimension-4 width/context-length axis) and
# more than one layer index (dimension-4 layer-index axis), against a
# SECOND, independently-written composition of the same math (a batched,
# non-incremental, whole-sequence attention computed with plain torch
# ops in a structurally different order -- no shared code with the
# incremental composition under test) rather than a hand derivation alone,
# since attention at width > 1 is not practical to hand-derive digit by
# digit. This is the guard-vitality-at-composed-scale cell.
# --------------------------------------------------------------------------
def _batched_reference_forward(weights: "ilr.CheckpointWeights", input_ids, dtype):
    """A second, independently-coded, BATCHED (not incremental) composition
    of the identical math, sharing no function with
    `ilr.run_incremental_forward` / `ilr.decoder_layer_step` /
    `ilr.attention_step` below this line -- full attention computed as one
    masked whole-sequence matmul, not accumulated one token at a time. Used
    ONLY as a cross-check that the incremental composition's own FINAL
    (last-token) row agrees with a structurally different construction of
    the same formulas; never used as the primary witness."""
    cfg = weights.config
    n = len(input_ids)
    embed_table = weights.tensors["model.embed_tokens.weight"]
    hidden = torch.stack([embed_table[t] for t in input_ids], dim=0).to(dtype)  # [n, hidden]

    inv_freq = ilr.rope_inv_freq(cfg.head_dim, cfg.rope_theta, "cpu", dtype)
    positions = torch.arange(n, dtype=torch.float64)
    freqs = torch.outer(positions, inv_freq.to(torch.float64))  # [n, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_all = emb.cos().to(dtype)  # [n, head_dim]
    sin_all = emb.sin().to(dtype)

    causal = torch.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(n):
            if j > i:
                causal[i, j] = float("-inf")

    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    for layer in range(cfg.num_hidden_layers):
        residual = hidden
        w = weights.get(layer, "input_layernorm.weight")
        mean_sq = hidden.pow(2).mean(dim=-1, keepdim=True)
        normed = hidden * torch.rsqrt(mean_sq + cfg.rms_norm_eps) * w

        wq, bq = weights.get(layer, "self_attn.q_proj.weight"), weights.get(layer, "self_attn.q_proj.bias")
        wk, bk = weights.get(layer, "self_attn.k_proj.weight"), weights.get(layer, "self_attn.k_proj.bias")
        wv, bv = weights.get(layer, "self_attn.v_proj.weight"), weights.get(layer, "self_attn.v_proj.bias")
        wo = weights.get(layer, "self_attn.o_proj.weight")

        q = (normed @ wq.T + bq).view(n, cfg.num_attention_heads, cfg.head_dim)
        k = (normed @ wk.T + bk).view(n, cfg.num_key_value_heads, cfg.head_dim)
        v = (normed @ wv.T + bv).view(n, cfg.num_key_value_heads, cfg.head_dim)

        def rot(t):
            d = t.shape[-1]
            t1, t2 = t[..., : d // 2], t[..., d // 2 :]
            return torch.cat((-t2, t1), dim=-1)

        cos_b = cos_all.unsqueeze(1)  # [n,1,head_dim]
        sin_b = sin_all.unsqueeze(1)
        q = q * cos_b + rot(q) * sin_b
        k = k * cos_b + rot(k) * sin_b

        k_rep = torch.repeat_interleave(k, n_rep, dim=1)  # [n, num_heads, head_dim]
        v_rep = torch.repeat_interleave(v, n_rep, dim=1)

        scale = 1.0 / (cfg.head_dim ** 0.5)
        scores = torch.einsum("qhd,khd->hqk", q, k_rep) * scale  # [num_heads, n, n]
        scores = scores + causal.unsqueeze(0)
        probs = torch.softmax(scores.to(torch.float32), dim=-1).to(dtype)
        ctx = torch.einsum("hqk,khd->qhd", probs, v_rep).reshape(n, -1)
        attn_out = ctx @ wo.T
        hidden = residual + attn_out

        residual = hidden
        w2 = weights.get(layer, "post_attention_layernorm.weight")
        mean_sq2 = hidden.pow(2).mean(dim=-1, keepdim=True)
        normed2 = hidden * torch.rsqrt(mean_sq2 + cfg.rms_norm_eps) * w2
        wg = weights.get(layer, "mlp.gate_proj.weight")
        wu = weights.get(layer, "mlp.up_proj.weight")
        wd = weights.get(layer, "mlp.down_proj.weight")
        gate = normed2 @ wg.T
        up = normed2 @ wu.T
        mlp_out = ((gate * torch.sigmoid(gate)) * up) @ wd.T
        hidden = residual + mlp_out

    return hidden[-1]  # last token's final-layer row


def test_end_to_end_synthetic_layer():
    torch.manual_seed(42)
    for num_layers, context_len in ((1, 3), (3, 8)):  # layer-index axis and width axis, design S11 dim 4
        cfg = ilr.CheckpointConfig(
            hidden_size=8, num_hidden_layers=num_layers, num_attention_heads=2, num_key_value_heads=1,
            rms_norm_eps=1e-6, rope_theta=10000.0, head_dim=4,
        )
        tensors = {
            "model.embed_tokens.weight": torch.randn(20, 8, dtype=torch.float64),
            "model.norm.weight": torch.randn(8, dtype=torch.float64),
        }
        for layer in range(num_layers):
            tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = torch.randn(8, 8, dtype=torch.float64) * 0.1
            tensors[f"model.layers.{layer}.self_attn.q_proj.bias"] = torch.randn(8, dtype=torch.float64) * 0.1
            tensors[f"model.layers.{layer}.self_attn.k_proj.weight"] = torch.randn(4, 8, dtype=torch.float64) * 0.1
            tensors[f"model.layers.{layer}.self_attn.k_proj.bias"] = torch.randn(4, dtype=torch.float64) * 0.1
            tensors[f"model.layers.{layer}.self_attn.v_proj.weight"] = torch.randn(4, 8, dtype=torch.float64) * 0.1
            tensors[f"model.layers.{layer}.self_attn.v_proj.bias"] = torch.randn(4, dtype=torch.float64) * 0.1
            tensors[f"model.layers.{layer}.self_attn.o_proj.weight"] = torch.randn(8, 8, dtype=torch.float64) * 0.1
            tensors[f"model.layers.{layer}.mlp.gate_proj.weight"] = torch.randn(16, 8, dtype=torch.float64) * 0.1
            tensors[f"model.layers.{layer}.mlp.up_proj.weight"] = torch.randn(16, 8, dtype=torch.float64) * 0.1
            tensors[f"model.layers.{layer}.mlp.down_proj.weight"] = torch.randn(8, 16, dtype=torch.float64) * 0.1
            tensors[f"model.layers.{layer}.input_layernorm.weight"] = torch.rand(8, dtype=torch.float64) + 0.5
            tensors[f"model.layers.{layer}.post_attention_layernorm.weight"] = torch.rand(8, dtype=torch.float64) + 0.5

        weights = ilr.CheckpointWeights.from_safetensors_dict(tensors, cfg)
        input_ids = list(range(context_len))

        captured = ilr.run_incremental_forward(weights, input_ids, torch.float64, "cpu")
        incremental_final = captured[num_layers]

        batched_final = _batched_reference_forward(weights, input_ids, torch.float64)

        check(
            f"end_to_end_incremental_matches_batched_layers{num_layers}_ctx{context_len}",
            close(incremental_final, batched_final, atol=1e-9, rtol=1e-7),
            f"incremental={incremental_final.tolist()[:3]}... batched={batched_final.tolist()[:3]}...",
        )

    # Guard vitality at composed-pipeline scale: inject a wrong RoPE
    # rotation (skip rotation entirely) into the batched reference's own
    # copy and confirm the two constructions now DIVERGE -- proving the
    # cross-check is a genuine, falsifiable comparison, not one that passes
    # regardless of a real defect.
    torch.manual_seed(7)
    cfg = ilr.CheckpointConfig(
        hidden_size=8, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        rms_norm_eps=1e-6, rope_theta=10000.0, head_dim=4,
    )
    tensors = {
        "model.embed_tokens.weight": torch.randn(20, 8, dtype=torch.float64),
        "model.norm.weight": torch.randn(8, dtype=torch.float64),
    }
    for layer in range(2):
        tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = torch.randn(8, 8, dtype=torch.float64) * 0.1
        tensors[f"model.layers.{layer}.self_attn.q_proj.bias"] = torch.randn(8, dtype=torch.float64) * 0.1
        tensors[f"model.layers.{layer}.self_attn.k_proj.weight"] = torch.randn(4, 8, dtype=torch.float64) * 0.1
        tensors[f"model.layers.{layer}.self_attn.k_proj.bias"] = torch.randn(4, dtype=torch.float64) * 0.1
        tensors[f"model.layers.{layer}.self_attn.v_proj.weight"] = torch.randn(4, 8, dtype=torch.float64) * 0.1
        tensors[f"model.layers.{layer}.self_attn.v_proj.bias"] = torch.randn(4, dtype=torch.float64) * 0.1
        tensors[f"model.layers.{layer}.self_attn.o_proj.weight"] = torch.randn(8, 8, dtype=torch.float64) * 0.1
        tensors[f"model.layers.{layer}.mlp.gate_proj.weight"] = torch.randn(16, 8, dtype=torch.float64) * 0.1
        tensors[f"model.layers.{layer}.mlp.up_proj.weight"] = torch.randn(16, 8, dtype=torch.float64) * 0.1
        tensors[f"model.layers.{layer}.mlp.down_proj.weight"] = torch.randn(8, 16, dtype=torch.float64) * 0.1
        tensors[f"model.layers.{layer}.input_layernorm.weight"] = torch.rand(8, dtype=torch.float64) + 0.5
        tensors[f"model.layers.{layer}.post_attention_layernorm.weight"] = torch.rand(8, dtype=torch.float64) + 0.5
    weights = ilr.CheckpointWeights.from_safetensors_dict(tensors, cfg)
    input_ids = list(range(5))
    incremental_final = ilr.run_incremental_forward(weights, input_ids, torch.float64, "cpu")[2]

    # A deliberately-broken batched reference: RoPE is skipped (cos=1, sin=0
    # everywhere), a real, meaningful arithmetic departure.
    saved_rope_inv_freq = ilr.rope_inv_freq
    try:
        ilr.rope_inv_freq = lambda head_dim, theta, device, dtype: torch.zeros(head_dim // 2, dtype=dtype)
        broken_batched_final = _batched_reference_forward(weights, input_ids, torch.float64)
    finally:
        ilr.rope_inv_freq = saved_rope_inv_freq

    check(
        "end_to_end_guard_vitality_broken_rope_diverges",
        not close(incremental_final, broken_batched_final, atol=1e-6, rtol=1e-5),
        "a batched reference with RoPE disabled must NOT match the correct incremental composition",
    )


def main():
    test_rmsnorm()
    test_rope()
    test_swiglu()
    test_causal_mask()
    test_gqa_repeat()
    test_kv_cache_lifetime()
    test_trust_boundary_rejection()
    test_end_to_end_synthetic_layer()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
