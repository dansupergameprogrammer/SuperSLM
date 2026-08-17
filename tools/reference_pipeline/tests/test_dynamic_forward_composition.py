"""End-to-end composition cells for the §15 W8A8-dynamic forward (C23–C30, §17 G-8 items 11–16).

Per-site composition properties: norm scale-kill (C23, item 11), the identity channel's true
pass-through with the executed 25-vs-26 witness (C24, item 12), token-wide fold conformance
(C25, item 12), K/V static per-head landing via the per-key reconciliation (C27, item 14), and
the C30 integration sentinel (out_scale never computed; the derivation routed through the
pinned intmath function — item 16's integration half).

Brief: Claude/Curie/packets/eval/packet-08.md. Trace shape per the A-4-amended pin
(Claude/Curie/SuperSLM_Section15_Eval_TestDesign-2026-07-16.md §13): chain sites carry the
C19-C22 fields plus (m_out, e_out); k/v projection sites are per (token, head) records with a
"head" key, the head's STATIC landing scale as (m_out, e_out), "R_key", and "codes" — and NO
Dprime/Dn/s/R (the C27 landing is a reconciliation to a static target, not the dynamic chain).

Re-authored by Curie 2026-07-16 (verification pass, amendment A-4): the delivered cell 2
construction failed a conformant build (executed: a real non-identity fold on channel 1 moves
that channel's code to 13, so equality-to-pass-through could never hold), cell 4 predated the
per-head trace amendment, and cell 5's smoke call used zero coefficients the pinned
positive-only rule rejects.
"""

import numpy as np
import pytest

from conftest import api, require
import rung1_ref
from fractions import Fraction

from composition_ref import (
    scale_mul_oracle,
    mbqm_oracle,
    fold_row_oracle,
    quantize_row_oracle,
    canonical_scale_oracle,
    residual_reconcile_oracle,
)

MODULE = "reference_pipeline.pipeline"
INTMATH = "reference_pipeline.intmath"

# E7's coefficient fixture constants (positive per N2-5; values ride C7/C8's S2 pin).
LN2_Q30, B_Q30, CA_Q30 = 744261117, 1452772687, 1030312935


def fixture_config(pipeline):
    """The §11 fixture model's shape, mirrored from test_dynamic_forward.py."""
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def _model():
    fixture_model = api(MODULE, "fixture_model")
    return fixture_model(fixture_config(require(MODULE)))


def site_offline_constant(model, site):
    """The model's emitted canonical constant for a site (§12.1 artifact pin:
    `model.composition_constants[site] -> (m, e)`), red-marked while absent."""
    table = getattr(model, "composition_constants", None)
    if table is None or site not in table:
        raise AssertionError(
            f"red-unimplemented: composition_constants[{site!r}] absent — "
            f"design record §12.1 artifact-surface pin")
    return table[site]


def test_a_norm_sites_carried_scale_is_gain_derived_not_forwarded():
    """C23 (item 11): a norm's carried scale is GAIN-DERIVED; the cell fails an
    implementation that forwards the incoming per-token scale through RMSNorm.

    Conformance loop: all three norm sites' (m_out, e_out) equal the gain-derived oracle
    recomputation — scale_mul(offline constant incl. the folded 1/127, (Dn, -s)) per C26's
    offline-fold rule (Curie A-5 §14.3) — which no forwarding implementation can reproduce
    (its m_out carries the incoming per-token factor).
    Decisiveness guard: restricted to layer0.attn_norm, the one site whose upstream is the
    embed record (Curie A-4 — the delivered helper mapped EVERY norm's upstream to embed,
    which is wrong for mlp_norm/layer1; the guard needs one true upstream, and layer0's
    attn_norm has it). The guard proves the forwarded-scale candidate differs, so the
    conformance assertion is non-vacuous.
    """
    forward_dynamic = api(MODULE, "forward_dynamic")
    model = _model()

    trace = []
    forward_dynamic(model, [0, 1, 3, 5], trace=trace)

    for site_prefix in ("layer0.attn_norm", "layer0.mlp_norm", "layer1.attn_norm"):
        records = [r for r in trace if r["site"].startswith(site_prefix)]
        assert records, f"no trace records at {site_prefix}"
        m_g, e_g = site_offline_constant(model, records[0]["site"])
        for rec in records:
            # C26 (Curie A-5 §14.3 correction): the site's offline constant folds the
            # 1/127 (and the gain-derived static factor) at conversion; the runtime
            # per-token factor is D' in its EXACT canonical form (Dn, -s) — C21's own
            # identity, no runtime rounding — entering through the one pinned high-mul.
            expected = scale_mul_oracle(m_g, e_g, rec["Dn"], -rec["s"])
            assert (rec["m_out"], rec["e_out"]) == expected, (
                f"{rec['site']} token {rec['token_index']}: carried scale is not "
                f"gain-derived (C23 scale-kill)")

    # Decisiveness guard, layer0.attn_norm only (its upstream IS the embed record).
    # Quantifier: AT LEAST ONE discriminating record (Curie A-7): an exactly-1.0
    # incoming scale composed with an even mantissa is the high-mul IDENTITY —
    # (m+1)//2 then the renormalizing left shift returns m for even m — a legitimate,
    # reachable edge (e.g. an S_embed = 1.0 artifact at D' = 127), so demanding every
    # record discriminate would fail correct implementations on legitimate data. A
    # decisiveness guard's job is one witness. (Bench 2026-07-16: under the pinned
    # derivation THIS fixture discriminates on all four tokens — S_embed is 1/127,
    # so no upstream is 1.0 — but the guard must not assume that of every artifact.)
    attn_records = [r for r in trace if r["site"].startswith("layer0.attn_norm")]
    embed_by_token = {r["token_index"]: r for r in trace if r["site"] == "embed"}
    discriminating = 0
    for rec in attn_records:
        upstream = embed_by_token[rec["token_index"]]
        gain_derived = (rec["m_out"], rec["e_out"])
        forwarded = scale_mul_oracle(*gain_derived,
                                     upstream["m_out"], upstream["e_out"])
        if forwarded != gain_derived:
            discriminating += 1
    assert discriminating >= 1, (
        "fixture not decisive: every incoming scale composes as the identity, so "
        "forwarding would be invisible at every record")


def test_the_fold_passes_the_identity_channel_through_untouched():
    """C24 (item 12): the identity (max) channel is a TRUE PASS-THROUGH — no multiply,
    no shift — discriminated from the near-identity (2^31-1, 0) by the executed witness.

    Witness recomputed through composition_ref (never copied): the near-identity folds
    2^30+1 -> 2^30, and x = 215593831 quantizes to code 25 under true pass-through vs 26
    under the near-identity fold — the ±1 on the max channel moves D' and the code.

    Construction (Curie A-4, replacing the delivered version, which compared a row with a
    REAL non-identity fold on channel 1 against the unfolded row — executed: that moves
    channel 1's code to 13 and the equality fails on a CONFORMANT build): both arms below
    run through the implementation's own fold with channel 1 held identity, so the ONLY
    difference is how channel 0 (the max channel) is treated.
    """
    fold_projection_accumulator = api(MODULE, "fold_projection_accumulator")

    row = [2 ** 30 + 1, 215593831]

    # Oracle decisiveness (in-cell, executed at the spec pass):
    assert mbqm_oracle(2 ** 30 + 1, (1 << 31) - 1, 0) == 2 ** 30
    assert quantize_row_oracle(row)[0][1] == 25                       # true pass-through
    assert quantize_row_oracle([2 ** 30, 215593831])[0][1] == 26      # near-identity fold

    # (a) an all-identity fold returns the row unchanged — no multiply, no shift.
    assert list(fold_projection_accumulator(row, [None, None])) == row

    # (b) the implementation's identity channel reproduces the pass-through code...
    identity_folded = fold_projection_accumulator(row, [None, None])
    assert quantize_row_oracle(identity_folded)[0][1] == 25, (
        "the identity channel is not a true pass-through: the max channel moved and "
        "the token's codes shifted (C24)")

    # ...and its EXPLICIT near-identity fold on the max channel lands on the divergent
    # side, proving the two representations are distinguishable through the
    # implementation's own fold path.
    near_folded = fold_projection_accumulator(row, [((1 << 31) - 1, 0), None])
    assert quantize_row_oracle(near_folded)[0][1] == 26


def test_a_folded_row_quantizes_to_the_oracle_codes_token_wide():
    """C25 (item 12): fold-rounding conformance at the token level — element-wise fold
    exactness AND whole-row code conformance (the divergence class is token-wide through
    the shared D', so the assertion is over the full code row, never per-element-only).
    """
    fold_projection_accumulator = api(MODULE, "fold_projection_accumulator")
    quantize_multiplier = api(MODULE, "quantize_multiplier")

    rng = np.random.default_rng(8003)
    row = [int(v) for v in rng.integers(-(2 ** 31), 2 ** 31, size=8)]

    folds = []
    for j in range(8):
        if j == 3:                       # the reference (identity) channel
            folds.append(None)
        else:
            ratio = 0.1 * (j + 1) / 8    # deterministic ratios in (0, 1)
            folds.append(quantize_multiplier(ratio))

    expected_folded = fold_row_oracle(row, folds)
    actual_folded = list(fold_projection_accumulator(row, folds))
    assert actual_folded == expected_folded

    assert quantize_row_oracle(actual_folded)[0] == quantize_row_oracle(expected_folded)[0]


def test_k_and_v_land_at_static_per_head_scales_via_the_per_key_reconciliation():
    """C27 as corrected by D-SLM58 (item 14): K/V land at STATIC per-head scales via the
    one-rounding C26 composite with the OFFLINE reciprocal; q stays per-token; no runtime
    key-axis reduction, and NO runtime reciprocal at this site.

    Trace shape per A-4 as amended by A-7 (the D-SLM58 re-pin — R_key is RETIRED, there
    is no per-token reciprocal to record): k/v site records are per (token, head), carry
    "head", "m_in"/"e_in" (the incoming carried scale — the per-key-token quantity,
    shared by k and v), the head's static landing scale as (m_out, e_out), "x_int" (the
    head's folded accumulator slice), and "codes"; NO Dprime/Dn/s/R (the landing is a
    reconciliation to a static target, not the dynamic chain).
    """
    forward_dynamic = api(MODULE, "forward_dynamic")
    model = _model()

    trace = []
    forward_dynamic(model, [0, 1, 3, 5], trace=trace)

    k_records = [r for r in trace if r["site"] == "layer0.k_proj.requant"]
    v_records = [r for r in trace if r["site"] == "layer0.v_proj.requant"]
    q_records = [r for r in trace if r["site"] == "layer0.q_proj.requant"]
    norm_by_token = {r["token_index"]: r for r in trace
                     if r["site"].startswith("layer0.attn_norm")}
    assert k_records and v_records and q_records

    # STATIC per-head: for each head, the landing scale is identical across tokens.
    heads = sorted({r["head"] for r in k_records})
    assert heads == [0, 1], f"fixture has 2 kv heads; trace names {heads}"
    k_head_scales = {}
    for head in heads:
        scales = {(r["m_out"], r["e_out"]) for r in k_records if r["head"] == head}
        assert len(scales) == 1, (
            f"k head {head} landing scale varies across tokens — not static (C27)")
        k_head_scales[head] = scales.pop()
    for head in sorted({r["head"] for r in v_records}):
        scales = {(r["m_out"], r["e_out"]) for r in v_records if r["head"] == head}
        assert len(scales) == 1

    # q stays per-token (the executed pairwise-distinct-D' fixture makes this decisive).
    q_scales = {(r["m_out"], r["e_out"]) for r in q_records}
    assert len(q_scales) >= 2, "q_proj's carried scale did not vary per token"

    # The incoming carried scale is the per-key-token quantity, shared by k and v:
    # (m_in, e_in) equals the attn_norm record's carried scale for that token.
    for token_index, norm_rec in norm_by_token.items():
        expected_in = (norm_rec["m_out"], norm_rec["e_out"])
        token_ins = {(r["m_in"], r["e_in"]) for r in k_records + v_records
                     if r["token_index"] == token_index}
        assert token_ins == {expected_in}, (
            f"token {token_index}: m_in/e_in does not carry the incoming scale "
            f"(or k/v/heads disagree)")

    # THE D-SLM58 LANDING, recomputed end to end: codes == the one-rounding composite
    # with the OFFLINE reciprocal over canonical(S_kh/S_ref) — per element, clamped.
    s_ref_k = max(model.weight_scales["layer0.k_proj"])
    for rec in k_records:
        s_kh = model.scales.scale(f"layer0.k_head{rec['head']}.scale")
        m_t, e_t = canonical_scale_oracle(Fraction(s_kh) / Fraction(s_ref_k))
        r_t = rung1_ref.reciprocal_oracle(m_t)  # OFFLINE — over the static mantissa
        expected_codes = tuple(
            max(-127, min(127, residual_reconcile_oracle(
                int(x), rec["m_in"], r_t, rec["e_in"], e_t)))
            for x in rec["x_int"])
        assert tuple(rec["codes"]) == expected_codes, (
            f"k head {rec['head']} token {rec['token_index']}: the landing is not the "
            f"pinned D-SLM58 composite")

    # No runtime key-axis reduction: a longer prefix cannot move the static constants
    # (a runtime key-axis-common scale would — the KV-cache bit-identity argument).
    trace2 = []
    forward_dynamic(model, [0, 1, 3, 5, 7], trace=trace2)
    for head in heads:
        scales2 = {(r["m_out"], r["e_out"]) for r in trace2
                   if r["site"] == "layer0.k_proj.requant" and r["head"] == head}
        assert scales2 == {k_head_scales[head]}


def test_iexp_constants_are_derived_through_the_pinned_intmath_function(monkeypatch):
    """C30 integration sentinel (item 16's integration half): the forward derives its
    per-token i-exp constants through the module-global `intmath.iexp_scale_constants`
    (the rung-1 clz64 pin class — instrumentable, so a divergent inline derivation cannot
    hide), and no trace record carries a runtime out_scale (both nonlinear consumers are
    same-scale ratios; out_scale cancels by construction — a runtime out_scale is the
    forbidden float reaching the reproducible path; E4's float sentinel covers the float
    half, this covers the shape half).

    Apparatus smoke (A-3 discipline): the counting wrapper is executed in-cell against a
    direct call with VALID positive coefficients BEFORE the forward runs — so the
    apparatus is proven live even while the forward is red. (Curie A-4: the delivered
    smoke call passed zero coefficients, which the pinned positive-only rule (N2-5)
    rejects — it would have raised on a conformant build.)
    """
    intmath = require(INTMATH)
    forward_dynamic = api(MODULE, "forward_dynamic")
    real = api(INTMATH, "iexp_scale_constants")

    calls = []

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(intmath, "iexp_scale_constants", counting)

    # Smoke: a direct call through the patched module-global name, valid coefficients.
    smoke = intmath.iexp_scale_constants(2 ** 30, -36, LN2_Q30, 30, B_Q30, 30, CA_Q30, 30)
    assert len(calls) == 1 and isinstance(smoke, tuple) and len(smoke) == 3, (
        "counting apparatus failed its green-path smoke")
    calls.clear()

    model = _model()
    trace = []
    forward_dynamic(model, [0, 1, 3, 5], trace=trace)
    assert calls, "the forward derived no i-exp constants through the pinned function"

    # Shape half: pinned keys per site type; out_scale nowhere.
    chain_base = {"site", "token_index", "x_int", "Dprime", "Dn", "s", "R", "codes",
                  "m_out", "e_out"}
    kv_base = {"site", "token_index", "head", "x_int", "m_in", "e_in", "codes",
               "m_out", "e_out"}   # R_key RETIRED by D-SLM58 (no runtime reciprocal)
    for rec in trace:
        keys = set(rec)
        if rec["site"].endswith(("k_proj.requant", "v_proj.requant")):
            assert keys >= kv_base, f"{rec['site']}: missing {kv_base - keys}"
        else:
            assert keys >= chain_base, f"{rec['site']}: missing {chain_base - keys}"
        assert "out_scale" not in keys, (
            f"{rec['site']} carries a runtime out_scale (forbidden by C30)")
