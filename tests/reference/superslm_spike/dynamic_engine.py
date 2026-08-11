"""The vectorized dynamic engine — the harness's execution vehicle (D-SLM59 stage 2).

The scalar `pipeline.forward_dynamic` is and stays THE normative reference (E9-anchored
to the independent oracle); this module exists because the scalar measures ~4.7 min/token
at the real checkpoint and the §15 eval needs throughput. Conformance is **bit-equality
to the scalar** — logits AND trace, record-for-record — never a tolerance
(test_dynamic_engine.py; A-8 §17.2).

What is vectorized, and why each step keeps the scalar's bits:

* **Matmuls** (the dominant cost) ride the static arm's float64-exact-carrier trick:
  int8 codes against int8 weights widened once (`pipeline._gemm_weight`), summed in
  float64 where every partial sum is an exactly-representable integer (worst dot here is
  8960 · 127 · 127 ≈ 1.4e8, and the attention context peaks ≈ context · 2^15 · 127 —
  both far under 2^53). The bound is CHECKED, not assumed: every product goes through
  `pipeline.int_matmul`, whose `assert_exact_accumulation` raises where the envelope
  would break.
* **RMSNorm, RoPE, and the C25 fold** reuse the pipeline's vec kernels
  (`vec_i_sqrt`, `vec_rope_apply_pair` via `_vec_rope`-equivalent math,
  `vec_multiply_by_quantized_multiplier`), each already held bit-equal to its
  `intmath`/`rope` scalar on the exhaustive edge corpora by the standing suite.
* **The C22 requant composite** (`|x·127·R| ≤ 127·2^63` — the 2^70 lane class, C20/C22's
  law: int64 lanes alone wrap) gets an explicit two-limb decomposition
  (`requant_token_codes_vec`): magnitudes split at 2^32, carries propagated in int64,
  the half-away-from-zero rounding folded into the limbs as a magnitude half-up plus
  sign restore. Exactness argued line-by-line at the kernel.
* **The 2^94-class sites** — the C26 residual reconciliation, the D-SLM58 k/v landing,
  C30's `R_m²` derivation — use the exact-object fallback: per-element delegation to
  `intmath`'s scalar constructions over Python ints (arbitrary precision, bit-equal by
  construction). These sites are O(hidden) per token, three orders below the matmuls,
  so the fallback costs nothing measurable; the lane law is satisfied by not putting
  them in lanes at all (A-8 §17.2 leaves the decomposition choice per site).
* **i-exp / softmax / sigmoid element steps** stay Python-int: their operands are
  unbounded in principle (the constants grow as the scale shrinks) and their counts are
  small (~2 · intermediate + heads · context per token).

The forward mirrors the scalar's loop skeleton statement-for-statement — same site
order, same record construction — so trace equality is structural, not coincidental.

T-1894 (T-1822 design Sec31.2, D-SLM2357): ported from `brunel/t1891-optionG-spike`
onto production `main` (this module did not exist there before this build — verified,
D-SLM2401). Two changes from the spike's own copy, both in Sec31.2.3's own text:

* `option_g_fused_k_landing`'s default is no longer a caller-supplied `False` on the
  whole-token entry points alone. It is `None` on every entry point this module
  exposes (`forward_dynamic_vec`, `_forward_dynamic_vec_layers`, `MicroStepState`,
  `begin_token`) and, when `None`, is resolved from `model.option_g_fused_k_landing`
  (`pipeline.QuantizedModel`'s own new field, this build) — the Python-side sibling of
  the artifact header `flags` bit the compiled engine reads
  (`SslmArtifact.OptionGFusedKLandingEnabled()`). An explicit `True`/`False` still
  overrides, but only as a test-only knob, never production's own selection.
* `MicroStepState`/`begin_token` now accept and forward this parameter at all. The
  spike's own copy did not: `MicroStepState.__init__` called
  `_forward_dynamic_vec_layers` with no `option_g_fused_k_landing` argument, so a
  resumed (micro-step) decode always ran the legacy order regardless of what the
  model or the whole-token path selected — named on the spike's own untested list
  ("threading the flag through the resumable path is out of this spike's scope").
  This build closes that gap by deriving from `model` at both entry points rather
  than threading a caller-required parameter through the resumable frame.

Test-design record: Claude/Curie/SuperSLM_Section15_Eval_TestDesign-2026-07-16.md §17.2.
"""

from __future__ import annotations

import numpy as np

from superslm_spike import intmath
from superslm_spike import pipeline
from superslm_spike import rope
from superslm_spike import silu_lut
from superslm_spike.pipeline import (
    NORM_FRAC_BITS,
    PROB_FRAC_BITS,
    _IEXP_LN2_Q,
    _IEXP_B_Q,
    _IEXP_CA_Q,
    _IEXP_QFMT,
)

__all__ = [
    "forward_dynamic_vec",
    "ALL_LAYERS",
    "MicroStepState",
    "begin_token",
    "decode_step",
    "requant_token_codes_vec",
    "residual_reconcile_vec",
    "landing_rescale_vec",
    "iexp_constants_vec",
]


# ==============================================================================
# The four vec kernels (A-8 §17.2's unit surfaces)
# ==============================================================================


def _requant_row_int64(row_arr: np.ndarray, r: int, s: int) -> list:
    """C22 over one row in int64 lanes via a two-limb decomposition.

    q_i = clamp(round_half_away_from_zero((x_i · 127 · R) / 2^(62−s)), −127, 127).

    Preconditions (both guaranteed by the chain's own order — C21 has already accepted
    D' = max|x| ≤ 2^31 before any requant runs): |x_i| ≤ 2^31, and s ∈ [−1, 30] so the
    divide exponent k = 62 − s ∈ [32, 63].

    Exactness argument, limb by limb (K = 127·R < 2^39; a = |x| ≤ 2^31):
      * K splits as K_hi·2^32 + K_lo with K_hi < 2^7, K_lo < 2^32.
      * hi = a·K_hi ≤ 2^38; lo = a·K_lo ≤ 2^31·(2^32−1) = 2^63 − 2^31 — both int64-safe.
      * The rounding constant 2^(k−1) (k ≥ 32) is c_hi·2^32 + c_lo with c_lo nonzero
        only at k = 32 (c_lo = 2^31). Carries are propagated BEFORE adding c_lo so no
        intermediate exceeds 2^63 − 1.
      * With |N| + 2^(k−1) = H·2^32 + L (0 ≤ L < 2^32), floor((H·2^32 + L)/2^k) =
        H >> (k−32) exactly for k ≥ 32: the discarded L/2^32 < 1 cannot move the floor
        (H mod 2^(k−32) ≤ 2^(k−32)−1, so adding < 1 never crosses a multiple).
      * Half-away-from-zero = magnitude half-up + sign restore (the clamp bound is
        symmetric, so clamping the magnitude first is identical).
    """
    k = 62 - s
    if not 32 <= k <= 63:
        # Outside the chain's reachable band: exact-object fallback, same semantics.
        return [intmath.requant_token_code(int(x), r, s) for x in row_arr.tolist()]
    big_k = 127 * r
    k_hi, k_lo = big_k >> 32, big_k & 0xFFFFFFFF
    c = 1 << (k - 1)
    c_hi, c_lo = c >> 32, c & 0xFFFFFFFF

    sign = np.sign(row_arr)
    a = np.abs(row_arr)
    hi = a * np.int64(k_hi)
    lo = a * np.int64(k_lo)
    # Propagate lo's carry into hi BEFORE adding the rounding constant's low limb,
    # so lo stays below 2^32 when c_lo lands (no int64 overflow path exists).
    hi = hi + np.int64(c_hi) + (lo >> np.int64(32))
    lo = (lo & np.int64(0xFFFFFFFF)) + np.int64(c_lo)
    hi = hi + (lo >> np.int64(32))
    q_mag = hi >> np.int64(k - 32)
    q_mag = np.minimum(q_mag, np.int64(127))
    return [int(v) for v in (sign * q_mag)]


def requant_token_codes_vec(x_rows, r: int, s: int) -> list:
    """C22 over a batch of rows sharing one (R, s): list of code lists.

    The 2^70 lane site (the joint-attainment point x = INT32_MIN, R = 2^32, s = −1
    reaches |x·127·R| = 127·2^63 exactly); the decomposition above is what makes int64
    lanes sufficient. Bit-equal to `intmath.requant_token_code` element-wise.
    """
    return [_requant_row_int64(np.asarray(row, dtype=np.int64), r, s) for row in x_rows]


def residual_reconcile_vec(codes, m_b: int, r_h: int, e_b: int, e_h: int) -> list:
    """C26's residual reconciliation over a sequence of branch codes.

    The 2^70-class site (|code·m_b·R_h| ≤ 127·2^31·2^32). Exact-object fallback:
    per-element delegation to `intmath.residual_reconcile` (Python ints, arbitrary
    precision) — bit-equal by construction; the site is O(hidden) per token, so lanes
    buy nothing the matmuls haven't already paid for.
    """
    return [intmath.residual_reconcile(int(c), m_b, r_h, e_b, e_h) for c in codes]


def landing_rescale_vec(acc_values, m_a: int, r_t: int, e_a: int, e_t: int) -> list:
    """The D-SLM58 k/v landing composite over a sequence of folded accumulator values.

    The 2^94-class site (|acc·m_a·R_t| reaches (2^31−1)·(2^31−1)·2^32). Same C26
    composite with the D-SLM58 roles — the incoming mantissa m_a is the MULTIPLIER,
    R_t the OFFLINE reciprocal over the static target — realized by the same
    exact-object delegation.
    """
    return [intmath.residual_reconcile(int(a), m_a, r_t, e_a, e_t) for a in acc_values]


def iexp_constants_vec(m_values, e: int, ln2_q: int, ln2_fmt: int,
                       b_q: int, b_fmt: int, ca_q: int, ca_fmt: int) -> list:
    """C30's R-composed derivation over a sequence of mantissas at one exponent.

    `q_c`'s `R_m²` term is the ~2^94 wide intermediate; exact-object delegation to
    `intmath.iexp_scale_constants` per mantissa. Raises the scalar's own ValueError at
    the fine-scale rejection (`e = −78` at the Q30 formats), computes at the `e = −77`
    boundary — the same class at the same edge, because it IS the same construction.
    """
    return [intmath.iexp_scale_constants(int(m), e, ln2_q, ln2_fmt, b_q, b_fmt,
                                         ca_q, ca_fmt)
            for m in m_values]


# ==============================================================================
# The vectorized forward — bit-equal to pipeline.forward_dynamic
# ==============================================================================


def _chain_record_vec(model, site, token_index, wide_arr, incoming, trace):
    """The engine's C19-C22 chain + carried-scale step over one wide int64 row —
    the exact twin of the scalar `_chain_record` (same constructions, same trace
    record), with the requant vectorized through the two-limb kernel.

    D' comes off a numpy reduction over int64 (exact — the wide rows are bounded far
    below 2^63 by their constructions, and C21 rejects D' > 2^31 identically to the
    scalar because it IS the scalar's `normalize_scale`).
    """
    d = int(np.abs(wide_arr).max(initial=0))
    d_prime = max(d, 1)                       # C20's all-zero guard, verbatim
    dn, s = intmath.normalize_scale(d_prime)
    r = intmath.dynamic_scale_reciprocal(dn)
    codes = _requant_row_int64(wide_arr, r, s)

    constants = model.composition_constants
    site_constant = None
    for candidate in (site, site.split(".requant")[0]):
        if candidate in constants:
            site_constant = constants[candidate]
            break
    if site_constant is None:
        raise KeyError(f"forward_dynamic_vec: no composition_constants entry for {site!r}")
    m_out, e_out = intmath.carried_scale_product([*incoming, site_constant, (dn, -s)])
    if trace is not None:
        trace.append({
            "site": site, "token_index": token_index,
            "x_int": tuple(int(v) for v in wide_arr),
            "Dprime": d_prime, "Dn": dn, "s": s, "R": r, "codes": tuple(codes),
            "m_out": m_out, "e_out": e_out,
        })
    return codes, (m_out, e_out)


def _fold_arrays(weight_scales):
    """The C24/C25 fold table as vec-kernel operands: (identity_mask, mults, shifts).

    Identity channels (C24's true pass-through) are selected OUT by the mask; their
    lanes still run the vec composite on a dummy (1<<30, 0) pair whose result is
    discarded by the select — computed-then-dropped, never observable.
    """
    folds, _ = pipeline._reference_fold(weight_scales)
    identity = np.array([f is None for f in folds])
    mults = np.array([1 << 30 if f is None else f[0] for f in folds], dtype=np.int64)
    shifts = np.array([0 if f is None else f[1] for f in folds], dtype=np.int64)
    return identity, mults, shifts


def _fold_rows(acc_rows: np.ndarray, fold_ops) -> np.ndarray:
    """C25 across whole rows: one vec C1 composite per non-identity channel, the
    identity channels passed through untouched (C24). Bit-equal to the scalar fold:
    `vec_multiply_by_quantized_multiplier` is held bit-equal to `intmath`'s composite
    by the standing suite, and the accumulator × multiplier product (< 2^62) is
    int64-safe per the C25 row's own width note."""
    identity, mults, shifts = fold_ops
    folded = pipeline.vec_multiply_by_quantized_multiplier(acc_rows, mults, shifts)
    return np.where(identity, acc_rows, folded)


def _rmsnorm_wide(hidden_arr: np.ndarray, gain_arr: np.ndarray, hidden_size: int):
    """§6.3's RMSNorm numerator × gain, over all tokens at once (int64 exact:
    the Q(2·16) values peak ≈ 2^39·127 ≪ 2^63). `vec_i_sqrt` is the standing
    bit-equal digit recurrence; the floor divisions are numpy's floor-consistent
    int64 divisions, matching Python `//` on every sign."""
    sums = (hidden_arr * hidden_arr).sum(axis=-1)
    roots = pipeline.vec_i_sqrt((sums << np.int64(2 * NORM_FRAC_BITS)) // np.int64(hidden_size))
    roots = np.maximum(roots, np.int64(1))
    normed = (hidden_arr << np.int64(2 * NORM_FRAC_BITS)) // roots.reshape(-1, 1)
    return normed * gain_arr


def _rotate_rows(rows_arr: np.ndarray, cos_table, sin_table, positions) -> np.ndarray:
    """§6.4's pairwise rotation over (tokens, head_dim) rows at absolute positions,
    through the standing bit-equal `vec_rope_apply_pair`, clamped to int8 like the
    scalar's post-rotation clamp."""
    even = rows_arr[:, 0::2]
    odd = rows_arr[:, 1::2]
    cos = cos_table[positions, :]
    sin = sin_table[positions, :]
    r_even, r_odd = pipeline.vec_rope_apply_pair(even, odd, cos, sin)
    out = np.empty_like(rows_arr)
    out[:, 0::2] = np.clip(r_even, -127, 127)
    out[:, 1::2] = np.clip(r_odd, -127, 127)
    return out


def _rotate_wide_pair_row(seg, cos_row, sin_row):
    """T-1894 (T-1822 §31.2's production Option-G build; carried from T-1891 §29.4/§30,
    D-SLM2305) -- the fused pre-landing K rotation, the reference-side mirror of the
    engine's `RopeApplyPairWide` (`forward_sites.cpp`). Applies `rope.rope_apply_pair`
    -- the SAME Q2.30 rotation and SAME single C3 (ties-away-from-zero) rounding the
    narrow post-landing rotation already uses -- pairwise over `seg`, the WIDE
    (unlanded, pre-`landing_rescale_vec`) per-head K accumulator segment, at this
    token's absolute position's table row.

    `seg`'s elements and `rope.rope_apply_pair`'s own arithmetic are plain Python ints
    (arbitrary precision) -- exact by construction, with no width limit of its own to
    mirror the engine's SU128 facility against. This is why no "wide" sibling of
    `rope.rope_apply_pair` is needed on the reference side: the scalar primitive was
    already exact at any magnitude, only the engine's fixed-width C++ needed a new
    primitive (`RopeApplyPairWide`) to reach the same exactness.

    Returns a list the same length as `seg`, still WIDE (unclamped, un-landed) -- the
    caller lands it through `landing_rescale_vec` exactly as the un-rotated accumulator
    was landed before, per the design's own construction ("one `LandingRescale` per
    head on the existing static per-head K landing constants").
    """
    pairs = len(seg) // 2
    out = list(seg)
    for i in range(pairs):
        x = int(seg[2 * i])
        y = int(seg[2 * i + 1])
        rx, ry = rope.rope_apply_pair(x, y, int(cos_row[i]), int(sin_row[i]))
        out[2 * i] = rx
        out[2 * i + 1] = ry
    return out


def _resolve_option_g_fused_k_landing(model, option_g_fused_k_landing):
    """T-1894 (design Sec31.2.3, D-SLM2357): the ONE resolution point both the
    whole-token and the resumable (micro-step) entry points call. `None` (every
    entry point's own default) reads `model.option_g_fused_k_landing` -- the
    Python-side sibling of the artifact header `flags` bit the compiled engine's
    `SslmArtifact.OptionGFusedKLandingEnabled()` reads. An explicit `True`/`False`
    is a test-only override (mirroring how the engine's own gates force specific
    states), never production's own selection mechanism.
    """
    if option_g_fused_k_landing is None:
        return bool(model.option_g_fused_k_landing)
    return bool(option_g_fused_k_landing)


def _forward_dynamic_vec_layers(model, tokens, cache=None, trace=None,
                                 option_g_fused_k_landing=None):
    """`forward_dynamic_vec`'s body, as a resumable generator — §8.1's layer-range micro-step.

    Yields the index of each layer as it completes; returns (via StopIteration.value) the
    int32 logits once the final layer has executed and the head has run. This is the ONE
    law: `forward_dynamic_vec` drains it whole, and `DecodeStep`'s `layerBudget` advances it
    a bounded number of layers per call. Both spend the identical statements in the identical
    order, so bit-identity across every budget is STRUCTURAL rather than a second
    implementation that happens to agree — a duplicated loop would be free to drift, and the
    §17 dim-1 cells would be checking a copy against a copy.

    The suspended state is exactly what §7 calls the mid-token residual — `hidden_rows` plus
    its per-token `hidden_scale` — and the generator's own resume point is the layer-position
    marker. Both live in this frame, which belongs to the one sequence being advanced, never
    in a batch-shared workspace: the property §8.2's batch-invariance claim rests on, and the
    thing Loki strike 2 forced into the sequence's own state.

    Each layer's `cache.extend` therefore runs exactly once, because a resumed generator
    continues after the yield rather than re-entering the loop body.

    T-1894 (design Sec31.2.3): `option_g_fused_k_landing` resolves through
    `_resolve_option_g_fused_k_landing` -- the SAME resolution `forward_dynamic_vec`
    and `MicroStepState` both call, before this generator is ever constructed, so the
    resumable and whole-token paths share one already-resolved boolean rather than
    each re-reading `model` independently.
    """
    cfg = model.config
    group = pipeline.attention_group_size(cfg)
    tokens = list(tokens)
    steps = len(tokens)
    start = 0 if cache is None else cache.length
    pipeline._check_positions(cfg, steps, start)

    cos_rows, sin_rows = model.rope_tables
    cos_table = np.asarray(cos_rows, dtype=np.int64)
    sin_table = np.asarray(sin_rows, dtype=np.int64)
    positions = np.arange(start, start + steps)
    head_dim = cfg.head_dim
    num_kv_heads = cfg.num_key_value_heads

    def project(name, codes_arr):
        """int8 codes × widened weight in the checked float64-exact carrier."""
        return pipeline.int_matmul(codes_arr, pipeline._gemm_weight(model, name))

    # C28's runtime reciprocal, memoized per incoming mantissa (shared by the
    # projections one norm output feeds) — the scalar's own discipline.
    recip_cache: dict = {}

    def recip_a(m_a):
        r_a = recip_cache.get(m_a)
        if r_a is None:
            r_a = intmath.dynamic_scale_reciprocal(m_a)
            recip_cache[m_a] = r_a
        return r_a

    def biased_fold_row(site, folded_arr, in_scale):
        """C28's runtime entry over one folded row (exact-object per element — the
        |B·R_a| intermediate is unbounded by int64)."""
        entry = model.dynamic_biases.get(site)
        if entry is None:
            return folded_arr
        q_b, codes = entry
        m_a, e_a = in_scale
        r_a = recip_a(m_a)
        adjusted = [int(f) + intmath.bias_reconcile(int(b), q_b, r_a, e_a)
                    for f, b in zip(folded_arr.tolist(), codes)]
        return np.asarray(adjusted, dtype=np.int64)

    fused_k_landing = _resolve_option_g_fused_k_landing(model, option_g_fused_k_landing)

    # --- the entry (C23 embed) ---
    hidden_wide = np.asarray(model.weights["embed"], dtype=np.int64)[tokens, :]
    hidden_rows = np.empty((steps, cfg.hidden_size), dtype=np.int64)
    hidden_scale = [None] * steps
    for t in range(steps):
        codes, scale = _chain_record_vec(model, "embed", t, hidden_wide[t], [], trace)
        hidden_rows[t] = codes
        hidden_scale[t] = scale

    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"

        # --- attn_norm (C23 scale-killing) ---
        gain = np.asarray(model.weights[f"{prefix}.attn_norm.gain"], dtype=np.int64)
        wide_all = _rmsnorm_wide(hidden_rows, gain, cfg.hidden_size)
        normed = np.empty((steps, cfg.hidden_size), dtype=np.int64)
        norm_scale = [None] * steps
        for t in range(steps):
            codes, scale = _chain_record_vec(
                model, f"{prefix}.attn_norm", t, wide_all[t], [], trace)
            normed[t] = codes
            norm_scale[t] = scale

        # --- q/k/v projections ---
        q_raw = project(f"{prefix}.q_proj", normed)
        k_raw = project(f"{prefix}.k_proj", normed)
        v_raw = project(f"{prefix}.v_proj", normed)

        q_fold_ops = _fold_arrays(model.weight_scales[f"{prefix}.q_proj"])
        q_folded_all = _fold_rows(q_raw, q_fold_ops)
        q_rows = np.empty_like(q_folded_all)
        q_scale = [None] * steps
        for t in range(steps):
            folded = biased_fold_row(f"{prefix}.q_proj", q_folded_all[t], norm_scale[t])
            codes, scale = _chain_record_vec(
                model, f"{prefix}.q_proj.requant", t, folded, [norm_scale[t]], trace)
            q_rows[t] = codes
            q_scale[t] = scale

        q_heads = [_rotate_rows(q_rows[:, h * head_dim:(h + 1) * head_dim],
                                cos_table, sin_table, positions)
                   for h in range(cfg.num_attention_heads)]

        k_fold_ops = _fold_arrays(model.weight_scales[f"{prefix}.k_proj"])
        v_fold_ops = _fold_arrays(model.weight_scales[f"{prefix}.v_proj"])
        k_folded_all = _fold_rows(k_raw, k_fold_ops)
        v_folded_all = _fold_rows(v_raw, v_fold_ops)
        k_landed = np.empty((steps, num_kv_heads, head_dim), dtype=np.int64)
        v_landed = np.empty((steps, num_kv_heads, head_dim), dtype=np.int64)
        for t in range(steps):
            m_a, e_a = norm_scale[t]
            pos = int(positions[t])
            k_folded = biased_fold_row(f"{prefix}.k_proj", k_folded_all[t], norm_scale[t])
            v_folded = biased_fold_row(f"{prefix}.v_proj", v_folded_all[t], norm_scale[t])
            for head in range(num_kv_heads):
                m_target, e_target = model.kv_landing_scales[f"{prefix}.k_head{head}"]
                _m_t, e_t, r_t = model.kv_landing_reciprocals[f"{prefix}.k_head{head}"]
                seg = k_folded[head * head_dim:(head + 1) * head_dim]
                if fused_k_landing:
                    # T-1894/D-SLM2305: rotate the WIDE pre-landing segment pairwise at
                    # this token's absolute position, THEN land once -- K carries one
                    # int8 boundary instead of two (§31.2's construction). V and Q are
                    # untouched by this flag.
                    seg = _rotate_wide_pair_row(seg, cos_table[pos], sin_table[pos])
                landed = [max(-127, min(127, v)) for v in
                          landing_rescale_vec(seg, m_a, r_t, e_a, e_t)]
                k_landed[t, head] = landed
                if trace is not None:
                    trace.append({
                        "site": f"{prefix}.k_proj.requant", "token_index": t, "head": head,
                        "x_int": tuple(int(v) for v in seg), "m_in": m_a, "e_in": e_a,
                        "codes": tuple(landed), "m_out": m_target, "e_out": e_target,
                    })
                m_target_v, e_target_v = model.kv_landing_scales[f"{prefix}.v_head{head}"]
                _m_t_v, e_t_v, r_t_v = model.kv_landing_reciprocals[f"{prefix}.v_head{head}"]
                seg_v = v_folded[head * head_dim:(head + 1) * head_dim]
                landed_v = [max(-127, min(127, v)) for v in
                            landing_rescale_vec(seg_v, m_a, r_t_v, e_a, e_t_v)]
                v_landed[t, head] = landed_v
                if trace is not None:
                    trace.append({
                        "site": f"{prefix}.v_proj.requant", "token_index": t, "head": head,
                        "x_int": tuple(int(v) for v in seg_v), "m_in": m_a, "e_in": e_a,
                        "codes": tuple(landed_v), "m_out": m_target_v, "e_out": e_target_v,
                    })

        if fused_k_landing:
            # T-1894/D-SLM2305: K is already rotated at landing time (above) -- the
            # post-landing `_rotate_rows` this narrow path used is not run for K, the
            # same "one int8 boundary instead of two" property the engine-side flag
            # realizes by deleting `RopeApplySite`'s K read/rotate and write-back loop.
            k_heads = [k_landed[:, h, :] for h in range(num_kv_heads)]
        else:
            k_heads = [_rotate_rows(k_landed[:, h, :], cos_table, sin_table, positions)
                       for h in range(num_kv_heads)]
        v_heads = [v_landed[:, h, :] for h in range(num_kv_heads)]

        if cache is not None:
            k_arr = np.stack(k_heads, axis=1)
            v_arr = np.stack(v_heads, axis=1)
            k_full, v_full = cache.extend(layer, k_arr, v_arr)
            k_heads = [k_full[:, h, :] for h in range(num_kv_heads)]
            v_heads = [v_full[:, h, :] for h in range(num_kv_heads)]

        # --- attention (per-query C30 softmax constants; C27's regime) ---
        total_keys = int(k_heads[0].shape[0])
        context = np.empty((steps, cfg.num_attention_heads * head_dim), dtype=np.int64)
        for head in range(cfg.num_attention_heads):
            kv_head = head // group
            keys = np.asarray(k_heads[kv_head], dtype=np.int64)
            values = np.asarray(v_heads[kv_head], dtype=np.int64)
            softmax_static = model.composition_constants[f"{prefix}.softmax_khead{kv_head}"]
            scores = pipeline.int_matmul(q_heads[head], keys.T.astype(np.int64))
            probs = np.zeros((steps, total_keys), dtype=np.int64)
            for t in range(steps):
                sm_m, sm_e = intmath.carried_scale_product([q_scale[t], softmax_static])
                q_ln2, q_b, q_c = intmath.iexp_scale_constants(
                    sm_m, sm_e, _IEXP_LN2_Q, _IEXP_QFMT, _IEXP_B_Q, _IEXP_QFMT,
                    _IEXP_CA_Q, _IEXP_QFMT)
                width = t + start + 1
                present = [int(v) for v in scores[t, :width]]
                shifted = intmath.shift_by_max(present)
                exps = [intmath.i_exp_from_constants(q, q_ln2, q_b, q_c) for q in shifted]
                # The same probability-width refusal `_vec_softmax` has always carried
                # (pipeline._guard_probability_width), against the same named ceiling
                # (D-SLM367). This path is exact-object Python, so without it the row
                # silently produces a value the int64 C++ port cannot represent — and
                # this forward IS the port's bit-equality oracle, so an unguarded
                # reference makes the two disagree by construction on exactly the rows
                # the port must refuse. The peak is taken over Python ints rather than
                # through numpy, because the values this guard exists to catch are the
                # ones that do not fit int64.
                peak = max((abs(int(v)) for v in exps), default=0)
                if peak > pipeline.PROB_WIDTH_CEILING:
                    raise pipeline.ExactnessEnvelopeExceeded(
                        f"an i-exp output of {peak} cannot be normalized at "
                        f"{PROB_FRAC_BITS} fractional bits without overflowing int64; "
                        f"the softmax input scale is too fine"
                    )
                total = sum(exps)
                probs[t, :width] = [(e << PROB_FRAC_BITS) // max(total, 1) for e in exps]
            # probabilities ≤ 2^15, values ≤ 127, dot ≤ context length: exact carrier.
            context[:, head * head_dim:(head + 1) * head_dim] = pipeline.int_matmul(
                probs, values)

        # --- attn_ctx (C23 + D-SLM57's per-head pre-fold) ---
        s_v = [model.scales.scale(f"{prefix}.v_head{h}.scale") for h in range(num_kv_heads)]
        s_v_max = max(s_v)
        ctx_folds = []
        for h in range(cfg.num_attention_heads):
            f_v = s_v[h // group]
            entry = None if f_v == s_v_max else pipeline.quantize_multiplier(f_v / s_v_max)
            ctx_folds.extend([entry] * head_dim)
        ctx_identity = np.array([f is None for f in ctx_folds])
        ctx_mults = np.array([1 << 30 if f is None else f[0] for f in ctx_folds], dtype=np.int64)
        ctx_shifts = np.array([0 if f is None else f[1] for f in ctx_folds], dtype=np.int64)
        ctx_folded_all = np.where(
            ctx_identity, context,
            pipeline.vec_multiply_by_quantized_multiplier(context, ctx_mults, ctx_shifts))

        attn_out = np.empty((steps, cfg.num_attention_heads * head_dim), dtype=np.int64)
        attn_scale = [None] * steps
        for t in range(steps):
            codes, scale = _chain_record_vec(
                model, f"{prefix}.attn_ctx", t, ctx_folded_all[t], [], trace)
            attn_out[t] = codes
            attn_scale[t] = scale

        o_raw = project(f"{prefix}.o_proj", attn_out)
        o_fold_ops = _fold_arrays(model.weight_scales[f"{prefix}.o_proj"])
        o_folded_all = _fold_rows(o_raw, o_fold_ops)
        o_rows = np.empty_like(o_folded_all)
        o_scale = [None] * steps
        for t in range(steps):
            folded = biased_fold_row(f"{prefix}.o_proj", o_folded_all[t], attn_scale[t])
            codes, scale = _chain_record_vec(
                model, f"{prefix}.o_proj.requant", t, folded, [attn_scale[t]], trace)
            o_rows[t] = codes
            o_scale[t] = scale

        # --- attn residual (C26) ---
        new_hidden = np.empty_like(hidden_rows)
        new_scale = [None] * steps
        for t in range(steps):
            m_h, e_h = hidden_scale[t]
            m_b, e_b = o_scale[t]
            r_h = recip_a(m_h)
            reconciled = residual_reconcile_vec(o_rows[t], m_b, r_h, e_b, e_h)
            wide = hidden_rows[t] + np.asarray(reconciled, dtype=np.int64)
            codes, scale = _chain_record_vec(
                model, f"{prefix}.attn_residual", t, wide, [(m_h, e_h)], trace)
            new_hidden[t] = codes
            new_scale[t] = scale
        hidden_rows, hidden_scale = new_hidden, new_scale

        # --- mlp_norm (C23 scale-killing) ---
        gain = np.asarray(model.weights[f"{prefix}.mlp_norm.gain"], dtype=np.int64)
        wide_all = _rmsnorm_wide(hidden_rows, gain, cfg.hidden_size)
        mlp_normed = np.empty((steps, cfg.hidden_size), dtype=np.int64)
        mlp_norm_scale = [None] * steps
        for t in range(steps):
            codes, scale = _chain_record_vec(
                model, f"{prefix}.mlp_norm", t, wide_all[t], [], trace)
            mlp_normed[t] = codes
            mlp_norm_scale[t] = scale

        gate_raw = project(f"{prefix}.gate_proj", mlp_normed)
        up_raw = project(f"{prefix}.up_proj", mlp_normed)
        gate_folded_all = _fold_rows(gate_raw, _fold_arrays(
            model.weight_scales[f"{prefix}.gate_proj"]))
        up_folded_all = _fold_rows(up_raw, _fold_arrays(
            model.weight_scales[f"{prefix}.up_proj"]))

        gate_rows = np.empty_like(gate_folded_all)
        up_rows = np.empty_like(up_folded_all)
        gate_scale = [None] * steps
        up_scale = [None] * steps
        for t in range(steps):
            gf = biased_fold_row(f"{prefix}.gate_proj", gate_folded_all[t], mlp_norm_scale[t])
            codes, scale = _chain_record_vec(
                model, f"{prefix}.gate_proj.requant", t, gf, [mlp_norm_scale[t]], trace)
            gate_rows[t] = codes
            gate_scale[t] = scale
            uf = biased_fold_row(f"{prefix}.up_proj", up_folded_all[t], mlp_norm_scale[t])
            codes, scale = _chain_record_vec(
                model, f"{prefix}.up_proj.requant", t, uf, [mlp_norm_scale[t]], trace)
            up_rows[t] = codes
            up_scale[t] = scale

        # --- mlp_act (SwiGLU; per-token C30 silu constants) ---
        activation = np.empty_like(gate_rows)
        act_scale = [None] * steps
        for t in range(steps):
            m_g, e_g = gate_scale[t]
            gate_list = gate_rows[t].tolist()
            sigmoids = np.empty(cfg.intermediate_size, dtype=np.int64)
            for i, value in enumerate(gate_list):
                sigmoids[i] = silu_lut.silu_sigmoid_q15(value, m_g, e_g)
            # gate (±127) × sigmoid (≤ 2^15) × up (±127): peaks ≈ 2^29, int64-safe.
            wide = gate_rows[t] * sigmoids * up_rows[t]
            codes, scale = _chain_record_vec(
                model, f"{prefix}.mlp_act", t, wide,
                [gate_scale[t], up_scale[t]], trace)
            activation[t] = codes
            act_scale[t] = scale

        down_raw = project(f"{prefix}.down_proj", activation)
        down_folded_all = _fold_rows(down_raw, _fold_arrays(
            model.weight_scales[f"{prefix}.down_proj"]))
        down_rows = np.empty_like(down_folded_all)
        down_scale = [None] * steps
        for t in range(steps):
            folded = biased_fold_row(f"{prefix}.down_proj", down_folded_all[t], act_scale[t])
            codes, scale = _chain_record_vec(
                model, f"{prefix}.down_proj.requant", t, folded, [act_scale[t]], trace)
            down_rows[t] = codes
            down_scale[t] = scale

        # --- mlp residual (C26) ---
        new_hidden = np.empty_like(hidden_rows)
        new_scale = [None] * steps
        for t in range(steps):
            m_h, e_h = hidden_scale[t]
            m_b, e_b = down_scale[t]
            r_h = recip_a(m_h)
            reconciled = residual_reconcile_vec(down_rows[t], m_b, r_h, e_b, e_h)
            wide = hidden_rows[t] + np.asarray(reconciled, dtype=np.int64)
            codes, scale = _chain_record_vec(
                model, f"{prefix}.mlp_residual", t, wide, [(m_h, e_h)], trace)
            new_hidden[t] = codes
            new_scale[t] = scale
        hidden_rows, hidden_scale = new_hidden, new_scale

        # §8.1: the layer is complete and the token is not. The micro-step boundary is
        # here and nowhere else — the residual above is a consistent, resumable state.
        yield layer

    # --- final norm + logits ---
    gain = np.asarray(model.weights["final_norm.gain"], dtype=np.int64)
    wide_all = _rmsnorm_wide(hidden_rows, gain, cfg.hidden_size)
    final_rows = np.empty((steps, cfg.hidden_size), dtype=np.int64)
    for t in range(steps):
        codes, _ = _chain_record_vec(model, "final_norm", t, wide_all[t], [], trace)
        final_rows[t] = codes

    head_name = "embed" if cfg.tie_word_embeddings else "lm_head"
    logits = pipeline.int_matmul(final_rows, pipeline._gemm_weight(model, head_name))
    return pipeline._to_int32(logits)


def forward_dynamic_vec(model, tokens, cache=None, trace=None,
                         option_g_fused_k_landing=None):
    """The vectorized W8A8-dynamic full-stack forward — bit-equal to the scalar
    `pipeline.forward_dynamic` on logits and trace (the module docstring carries the
    per-step exactness arguments). Same contract: int32 logits one row per token,
    the pinned trace record shapes, a REAL cache with the same bit-identity claim.

    §8.1's "`layerBudget = all` degenerates to whole-token steps", spelled as code: this
    drains `_forward_dynamic_vec_layers` without pausing. The signature and the bits are
    unchanged from before the micro-step existed.

    T-1894 (T-1822 design Sec31.2.3, D-SLM2357): `option_g_fused_k_landing` defaults to
    `None`, resolved from `model.option_g_fused_k_landing` -- the Python-side sibling of
    the artifact `flags` bit the compiled engine reads. An explicit `True`/`False` is a
    test-only override, never production's own selection. `MicroStepState`/
    `begin_token`/`decode_step` (below) resolve through the identical single source of
    truth, so the resumable and whole-token paths cannot disagree about which order
    runs for the same loaded model.
    """
    stepper = _forward_dynamic_vec_layers(model, tokens, cache=cache, trace=trace,
                                           option_g_fused_k_landing=option_g_fused_k_landing)
    while True:
        try:
            next(stepper)
        except StopIteration as complete:
            return complete.value


# --- §8.1's layer-range micro-step: the frame knob ------------------------------

ALL_LAYERS = -1
"""§8.1's `layerBudget = all` — spend whatever the token needs in one call."""


class MicroStepState:
    """One sequence resting between micro-steps — §7's mid-token residual and its
    layer-position marker, held in the sequence's OWN state.

    The residual and the marker live in the suspended generator frame this object owns.
    Nothing here is shared with another sequence, so interleaving another sequence's
    `decode_step` between two of this one's cannot perturb it (§8.2's batch-invariance
    claim; Loki strike 2's Manifestation A is the failure this shape forecloses).

    T-1894 (design Sec31.2.3): `option_g_fused_k_landing` (default `None`) is now
    accepted and forwarded to `_forward_dynamic_vec_layers`, which resolves it from
    `model` exactly as `forward_dynamic_vec` does -- the gap this build closes (the
    prior copy of this class never forwarded the parameter at all, so a resumed
    decode always ran the legacy order regardless of the model or the whole-token
    path's own selection).
    """

    __slots__ = ("_stepper", "layers_done", "logits", "complete")

    def __init__(self, model, tokens, cache=None, trace=None, option_g_fused_k_landing=None):
        self._stepper = _forward_dynamic_vec_layers(
            model, tokens, cache=cache, trace=trace,
            option_g_fused_k_landing=option_g_fused_k_landing)
        self.layers_done = 0
        self.logits = None
        self.complete = False


def begin_token(model, tokens, cache=None, trace=None,
                 option_g_fused_k_landing=None) -> MicroStepState:
    """Open a token's forward pass without executing any layer of it.

    T-1894 (design Sec31.2.3): forwards `option_g_fused_k_landing` (default `None`,
    resolved from `model`) to `MicroStepState` -- see that class's own comment.
    """
    return MicroStepState(model, tokens, cache=cache, trace=trace,
                           option_g_fused_k_landing=option_g_fused_k_landing)


def decode_step(state: MicroStepState, layer_budget: int = ALL_LAYERS) -> bool:
    """Advance `state` through up to `layer_budget` layers of the current token (§8.1).

    Returns True iff the token completed on this call, at which point `state.logits`
    carries the int32 logits — "a token completes when its final layer executes;
    `out_tokens` reports completion per sequence". A call that exhausts its budget
    mid-token returns False and leaves the sequence in a consistent, resumable state.

    `layer_budget` is a count of layers, `ALL_LAYERS` for §8.1's "all". Zero and negative
    budgets are rejected rather than degraded (§11's reject-over-degrade): a zero-budget
    call that silently did nothing would let a host spin forever believing it was making
    progress, and a negative one has no reading at all.
    """
    if layer_budget != ALL_LAYERS and layer_budget < 1:
        raise ValueError(
            f"layerBudget is a layer count >= 1, or ALL_LAYERS for §8.1's 'all'; "
            f"got {layer_budget}. A zero or negative budget is rejected, never "
            f"degraded into a no-op that a host would mistake for progress."
        )
    if state.complete:
        return True

    spent = 0
    while layer_budget == ALL_LAYERS or spent < layer_budget:
        try:
            next(state._stepper)
        except StopIteration as done:
            state.logits = done.value
            state.complete = True
            return True
        state.layers_done += 1
        spent += 1
    return False
