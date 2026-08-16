"""Arbitrary-precision reference oracles for the eval-forward site-level composition
(constructions C23-C30, SuperSLM_Plan.md S6.8, D-SLM56).

Sibling to rung1_ref.py (the C19-C22 oracle module, frozen): this module carries the
COMPOSITION oracles the S17 G-8 item 11-16 cells assert against. Same discipline:

  * exact Python `int` / `fractions.Fraction` arithmetic -- NEVER a float on an oracle path;
  * never a recode of intmath.py's / pipeline.py's implementation STEPS -- each function
    recomputes the pinned FORMULA from the plan row's own definition (G-8(1): "never
    bit-equality against a twin");
  * this module must never import intmath or pipeline (two independent derivations of the
    same formula); it MAY import rung1_ref and silu_lut_ref (oracle building on oracle --
    both are independent test-tree oracle modules, never the production implementation).

Construction citations, all SuperSLM_Plan.md S6.8 (D-SLM56, 2026-07-16):
  C1-C3  -- srdhm/rdbpot/mbqm: the gemmlowp reference definitions (C2 ties toward +inf --
            the whole primitive collapses to floor((a*b + 2^30)/2^31), D-SLM36's derivation;
            C3 ties away from zero).
  C24/C25 -- projection fold to S_ref = max_j S_w[j]; identity channel TRUE PASS-THROUGH;
            non-identity channels one mbqm rounding each (offline constants).
  C26    -- canonical carried scale (m in [2^30, 2^31), integer e); offline constants
            round-half-even (C14's precedent); runtime scale products through one C2-class
            high-mul; residual reconciliation
            rescaled_i = round_half_away_from_zero((branch_code_i * m_b * R_h) / 2^(62-(e_b-e_h))).
  C10    -- SwiGLU sigmoid: C10's fixed-point piecewise-linear LUT (index derivation, node
            selection, linear interpolation, C3 rounding), NOT i-exp-sigmoid -- reconciled
            2026-07-27 (SuperSLM_S3a_WalkingSkeleton_Plan.md SS11 S3.0, F-S3-1) so this
            module's mlp_act composition matches the C10 construction the reference forward
            now computes. Delegated to `silu_lut_ref.py`'s independent oracle rather than
            reimplemented here (oracle building on oracle, this module's own precedent with
            rung1_ref); i-exp-sigmoid is retained in this module only for C30/softmax, which
            is unaffected by F-S3-1.
  C28    -- bias reconciliation (CORRECTED formula, second-pass S2-1):
            b'[j] = round_half_away_from_zero((B[j] * R_a) / 2^(q_B + 62 + e_a)).
            The sign-inverted old form 2^(62 - e_a) is exported ONLY as the negative-control
            reference (G-8 item 15: a cell must FAIL an implementation computing it).
  C30    -- per-token i-exp constant derivation, R-COMPOSED (the pinned definition):
            q_ln2 = (LN2_Q * R_m) >> (fmt + 62 + e), q_b likewise,
            q_c = (CA_Q * R_m^2) >> (fmt_ca + 124 + 2e); truncating shifts (operands
            positive: shift IS floor); a negative shift is the FINE-scale rejection.
            The exact-division twin is exported ONLY as the negative-control fixture
            (second-pass W2-4: twin-equality would fail a conformant implementation at
            legitimate boundary divergences).

Test-design record: Claude/Curie/SuperSLM_Section15_Eval_TestDesign-2026-07-16.md (amendment A-3).
Derivation record: Claude/Vitruvius/SuperSLM_EvalForward_Composition_Proposal-2026-07-16.md.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from rung1_ref import (
    INT32_MAX,
    round_half_away_from_zero_ratio,
    reciprocal_oracle,
    max_abs_reduce_oracle,
    normalize_scale_oracle,
    requant_code_oracle,
)
from silu_lut_ref import (
    build_sigmoid_q15_table_oracle,
    silu_sigmoid_q15_oracle,
)

__all__ = [
    "srdhm_oracle",
    "rdbpot_oracle",
    "mbqm_oracle",
    "canonical_scale_oracle",
    "scale_mul_oracle",
    "fold_channel_oracle",
    "fold_row_oracle",
    "quantize_row_oracle",
    "residual_reconcile_oracle",
    "bias_reconcile_oracle",
    "bias_reconcile_sign_inverted",
    "bias_emission_oracle",
    "iexp_constants_oracle",
    "iexp_constants_exact_twin",
    "FineScaleRejected",
]


class FineScaleRejected(ValueError):
    """C30's fine-scale (overflow-end) rejection: a derivation shift went negative."""


# --- C1-C3: the gemmlowp reference primitives, from the documented definitions ---


def srdhm_oracle(a: int, b: int) -> int:
    """C2, from D-SLM36's derivation: the WHOLE primitive is floor((a*b + 2^30) / 2^31),
    ties toward +infinity for positive and negative alike; (INT32_MIN, INT32_MIN) is the
    one pair exceeding INT32_MAX and saturates."""
    result = (a * b + (1 << 30)) // (1 << 31)
    return INT32_MAX if result > INT32_MAX else result


def rdbpot_oracle(x: int, exponent: int) -> int:
    """C3: x / 2^exponent, ties away from zero, exponent in [0, 31]."""
    if not 0 <= exponent <= 31:
        raise ValueError(f"RoundingDivideByPOT exponent must be in [0, 31]; got {exponent}")
    return round_half_away_from_zero_ratio(x, 1 << exponent)


def mbqm_oracle(x: int, quantized_multiplier: int, shift: int) -> int:
    """C1: the two primitives composed, both roundings (the pinned requant composite)."""
    return rdbpot_oracle(srdhm_oracle(x, quantized_multiplier), shift)


# --- C26: the canonical carried scale (m, e) ------------------------------------


def _round_half_even_ratio(numerator: int, denominator: int) -> int:
    """round-half-even(numerator/denominator), exact, denominator > 0."""
    if denominator <= 0:
        raise ValueError("denominator must be > 0")
    q, r = divmod(numerator, denominator)
    twice = 2 * r
    if twice > denominator or (twice == denominator and q % 2 == 1):
        q += 1
    return q


def canonical_scale_oracle(value: Fraction) -> tuple[int, int]:
    """A positive real scale as canonical (m, e): value ~= m * 2^e with m in [2^30, 2^31),
    m rounded HALF-EVEN (C14's offline-emission precedent, cited by C26 for the offline
    constants). Exact Fraction arithmetic; Fraction(float) is exact, so a float constant
    converts losslessly before this is called."""
    value = Fraction(value)
    if value <= 0:
        raise ValueError(f"canonical scale must be positive; got {value}")
    # e such that value / 2^e lands in [2^30, 2^31).
    e = value.numerator.bit_length() - value.denominator.bit_length() - 31
    while value / Fraction(2) ** (e + 31) >= 1:          # value >= 2^(e+31) -> m too big
        e += 1
    while value / Fraction(2) ** (e + 30) < 1:           # value < 2^(e+30) -> m too small
        e -= 1
    scaled = value / Fraction(2) ** e                    # in [2^30, 2^31)
    m = _round_half_even_ratio(scaled.numerator, scaled.denominator)
    if m == 1 << 31:                                     # rounded up to the ceiling
        m >>= 1
        e += 1
    assert (1 << 30) <= m < (1 << 31)
    return m, e


def scale_mul_oracle(m_a: int, e_a: int, m_b: int, e_b: int) -> tuple[int, int]:
    """C26's runtime scale product: the mantissas through ONE C2-class high-mul
    (ties toward +infinity), renormalized into [2^30, 2^31) by an EXACT left shift
    (renormalization moves whole bits; only the high-mul rounds).

    value = m_a*2^e_a * m_b*2^e_b; high = srdhm(m_a, m_b) ~= m_a*m_b/2^31 in [2^29, 2^31),
    so value ~= high * 2^(e_a + e_b + 31)."""
    high = srdhm_oracle(m_a, m_b)
    e = e_a + e_b + 31
    if high < (1 << 30):
        high <<= 1
        e -= 1
    assert (1 << 30) <= high < (1 << 31)
    return high, e


# --- C24/C25: the projection weight fold -----------------------------------------


def fold_channel_oracle(acc: int, fold: tuple[int, int] | None) -> int:
    """One folded channel: the identity (max) channel is a TRUE PASS-THROUGH (C24 --
    fold is None); a non-identity channel is one mbqm rounding at its offline
    (Mw[j], shw[j]) constant (C25)."""
    if fold is None:
        return acc
    multiplier, shift = fold
    return mbqm_oracle(acc, multiplier, shift)


def fold_row_oracle(acc_row: Sequence[int], folds: Sequence[tuple[int, int] | None]) -> list[int]:
    """C25 across a projection's output row. `folds[j] is None` marks the identity channel."""
    if len(acc_row) != len(folds):
        raise ValueError("fold table must match the accumulator row")
    return [fold_channel_oracle(int(a), f) for a, f in zip(acc_row, folds)]


def quantize_row_oracle(row: Sequence[int]) -> tuple[list[int], int, int, int, int]:
    """The C19-C22 chain over one wide row, all via rung1_ref: (codes, D', Dn, s, R).
    The composition cells' one-stop recomputation of 'then the chain quantizes it'."""
    d_prime = max_abs_reduce_oracle(row)
    dn, s = normalize_scale_oracle(d_prime)
    r = reciprocal_oracle(dn)
    codes = [requant_code_oracle(int(x), r, s) for x in row]
    return codes, d_prime, dn, s, r


# --- C26: the residual-add reconciliation -----------------------------------------


def residual_reconcile_oracle(branch_code: int, m_b: int, r_h: int, e_b: int, e_h: int) -> int:
    """C26's per-element rescale of the branch into the stream's scale:
    round_half_away_from_zero((branch_code * m_b * R_h) / 2^(62 - (e_b - e_h))),
    R_h = C19 over the stream mantissa m_h (pass rung1_ref.reciprocal_oracle(m_h)).
    A negative composite exponent is an exact left shift (no rounding)."""
    numerator = branch_code * m_b * r_h
    k = 62 - (e_b - e_h)
    if k >= 0:
        return round_half_away_from_zero_ratio(numerator, 1 << k)
    return numerator << (-k)


# --- C28: the projection bias reconciliation --------------------------------------


def bias_emission_oracle(value, s_ref, q_b: int) -> int:
    """C28's offline STORAGE emission: B[j] = round_half_even(b[j] / S_ref * 2^q_B) --
    the exact integer stored at scale S_ref in Q-format q_B. Rounding mode adopted from
    C14's offline half-even precedent (flagged in A-8 for the C28 row's converter fold;
    the row pins the storage form, this pins the emission). Exact Fraction arithmetic;
    Fraction(float) is exact for float inputs."""
    scaled = Fraction(value) / Fraction(s_ref) * (1 << q_b)
    return _round_half_even_ratio(scaled.numerator, scaled.denominator)


def bias_reconcile_oracle(b: int, q_b: int, r_a: int, e_a: int) -> int:
    """C28, the CORRECTED formula (second-pass S2-1):
    b' = round_half_away_from_zero((B * R_a) / 2^(q_B + 62 + e_a)) -- the exponent ADDS
    e_a and CARRIES the stored Q-format (C30's q_b convention). B is signed; the
    away-from-zero tie rule is load-bearing."""
    numerator = b * r_a
    k = q_b + 62 + e_a
    if k >= 0:
        return round_half_away_from_zero_ratio(numerator, 1 << k)
    return numerator << (-k)


def bias_reconcile_sign_inverted(b: int, q_b: int, r_a: int, e_a: int) -> int:
    """THE NEGATIVE-CONTROL REFERENCE, NEVER A CONFORMANCE TARGET. The pre-correction
    transcription C28 carried before second-pass finding S2-1: exponent sign-inverted
    (2^(62 - e_a)) and the Q-format q_B dropped. G-8 item 15: an implementation computing
    this form is wrong by orders of magnitude at real exponents and the suite must FAIL it."""
    del q_b  # the old form dropped the Q-format -- reproduced faithfully
    numerator = b * r_a
    k = 62 - e_a
    if k >= 0:
        return round_half_away_from_zero_ratio(numerator, 1 << k)
    return numerator << (-k)


# --- C30: the per-token i-exp constant derivation ---------------------------------


def iexp_constants_oracle(m: int, e: int, ln2_q: int, ln2_fmt: int,
                          b_q: int, b_fmt: int, ca_q: int, ca_fmt: int
                          ) -> tuple[int, int, int]:
    """C30's R-COMPOSED derivation -- the pinned DEFINITION (D-SLM52's precedent):
        R_m   = C19 over m (recomputed via rung1_ref.reciprocal_oracle)
        q_ln2 = (LN2_Q * R_m)    >> (ln2_fmt + 62 + e)
        q_b   = (B_Q   * R_m)    >> (b_fmt   + 62 + e)
        q_c   = (CA_Q  * R_m^2)  >> (ca_fmt  + 124 + 2e)
    Truncating shifts; every operand positive, so the shift IS floor -- no tie rule.
    A negative shift raises FineScaleRejected (the fine-scale / overflow-end rejection;
    q_c's threshold is the earliest and sets the derivation's domain). The coefficient
    VALUES ride C7/C8's S2 pin and are inputs here; the FORMULA is what this recomputes."""
    if not all(v > 0 for v in (ln2_q, b_q, ca_q)):
        raise ValueError("C30's floor-shift claim holds on positive coefficients only (N2-5)")
    shift_ln2 = ln2_fmt + 62 + e
    shift_b = b_fmt + 62 + e
    shift_c = ca_fmt + 124 + 2 * e
    if min(shift_ln2, shift_b, shift_c) < 0:
        raise FineScaleRejected(
            f"negative derivation shift at e={e}: (ln2 {shift_ln2}, b {shift_b}, c {shift_c})")
    r_m = reciprocal_oracle(m)
    return ((ln2_q * r_m) >> shift_ln2,
            (b_q * r_m) >> shift_b,
            (ca_q * r_m * r_m) >> shift_c)


def iexp_constants_exact_twin(m: int, e: int, ln2_q: int, ln2_fmt: int,
                              b_q: int, b_fmt: int, ca_q: int, ca_fmt: int
                              ) -> tuple[int, int, int]:
    """THE NEGATIVE-CONTROL FIXTURE, NEVER THE EQUALITY TARGET (second-pass W2-4).
    The faithful exact-division twin: floor(C / S) by exact integer division of the same
    offline constants, never consulting R_m. Diverges from the pinned R-composed form at
    fine-scale exponents (executed: agreement at operating e, divergence toward e = -77) --
    which is exactly why conformance recomputes the R-composed side and this twin exists
    only to prove the suite can SEE the difference."""
    if min(-ln2_fmt - e, -b_fmt - e, -ca_fmt - 2 * e) < 0:
        raise ValueError("twin form requires e below each format's magnitude")
    return ((ln2_q << (-ln2_fmt - e)) // m,
            (b_q << (-b_fmt - e)) // m,
            (ca_q << (-ca_fmt - 2 * e)) // (m * m))


# ==============================================================================
# The full-forward logits oracle (E9) -- the C23-C30 composition recomputed
# ==============================================================================
#
# Authored 2026-07-16 under D-SLM56/57/58 (the complete composition pin set), per the
# paired-session commission and design-record amendments A-5/A-6/A-7. Independence
# boundary, stated exactly (A-5 s14.4): the oracle READS the model's raw artifact data
# (int8 weight codes, raw per-tensor/per-channel float scales, per-head KV landing
# scales, RoPE tables, config) and RECOMPUTES every formula -- the C23-C30 composition,
# the C19-C22 chain (via rung1_ref), and the S6.3 kernel constructions -- importing
# nothing from intmath/pipeline. Kernel micro-steps (i-exp polynomial, rmsnorm format,
# softmax/sigmoid ratios, RoPE rotation) are transcribed from their PINNED constructions
# (S6.3 / S6.8 C4-C14, I-BERT by reference); the composition -- the thing E9 tests --
# is derived from the C23-C30 rows alone.
#
# OFFLINE-EMISSION EXCEPTION (C14's rule, S6.8): offline constants are emitted by the
# pinned procedures, which run in float by design ("offline-derived tables pin the
# emitted integers plus the emitting procedure"). The float-touching helpers below
# (_canonical of a float via exact Fraction(float); _quantize_multiplier_offline) are
# offline emissions; no float touches a runtime-path computation.
#
# C7/C8 fixture constants (values ride the S2 pin; shared inputs with the
# implementation -- a coefficient change fails E9 loudly, which is the freeze working):

import math as _math

IEXP_LN2_Q30 = 744261117            # floor(ln2 * 2^30), Q0.30
IEXP_B_Q30 = 1452772687             # floor(1.353 * 2^30), I-BERT B, Q1.30
IEXP_CA_Q30 = 1030312935            # floor((0.344/0.3585) * 2^30), C/A, Q0.30
IEXP_CLIP_N = 30                    # I-BERT's shift clip (C8's current candidate)

_NORM_FRAC_BITS = 16                # S6.3 norm Q-format (C23's 2^16)
_PROB_FRAC_BITS = 15                # softmax probability Q-format
_SIGMOID_FRAC_BITS = 15             # sigmoid Q-format
_I8 = 127

# C10's offline table, built once at import from silu_lut_ref's independent
# arbitrary-precision derivation (F-S3-1) -- the same offline/runtime split C10's own
# construction uses, mirrored here rather than rebuilt per forward call.
_SILU_LUT_TABLE = build_sigmoid_q15_table_oracle()


def carried_scale_product_oracle(factors):
    """D-SLM57: the multi-factor runtime scale product -- LEFT-ASSOCIATED in
    composition order, incoming carried scale first, one C1/C2 high-mul per
    multiplication, exact single-shift renormalization (scale_mul_oracle per step)."""
    factors = list(factors)
    m, e = factors[0]
    for m_b, e_b in factors[1:]:
        m, e = scale_mul_oracle(m, e, m_b, e_b)
    return m, e


def _quantize_multiplier_offline(factor: float) -> tuple[int, int]:
    """The pinned gemmlowp QuantizeMultiplier emission (offline; C25/C1). Mirrors the
    documented procedure exactly, including round()'s half-even at the emission (an
    offline-procedure detail the emission rule carries)."""
    if not 0.0 < factor < 1.0:
        raise ValueError(f"fold multipliers live in (0, 1); got {factor}")
    fraction, exponent = _math.frexp(factor)
    multiplier = int(round(fraction * (1 << 31)))
    if multiplier >= 1 << 31:
        multiplier >>= 1
        exponent += 1
    shift = -exponent
    if shift < 0:
        return (1 << 31) - 1, 0
    if shift > 31:
        raise ValueError(f"factor {factor} needs shift {shift} outside [0, 31]")
    return multiplier, shift


def _fold_table(weight_scales) -> list:
    """C24/C25: per-channel offline folds to S_ref = max_j S_w[j]; every channel AT the
    max is the identity -- a true pass-through (None)."""
    s_ref = max(weight_scales)
    return [None if s == s_ref else _quantize_multiplier_offline(s / s_ref)
            for s in weight_scales]


def _i_sqrt_oracle(n: int) -> int:
    """C4-C6: floor(sqrt(n)) by restoring shift-and-subtract, 32 unconditional digits."""
    if n < 0:
        raise ValueError("i_sqrt is defined on n >= 0")
    remainder, root, bit = n, 0, 1 << 62
    for _ in range(32):
        trial = root + bit
        if remainder >= trial:
            remainder -= trial
            root = (root >> 1) + bit
        else:
            root >>= 1
        bit >>= 2
    return root


def _rmsnorm_rows(rows, hidden: int):
    """S6.3's integer RMSNorm in the scalar-normative Q16 form: per row,
    root = max(i_sqrt((sum v^2 << 32) // hidden), 1); out_i = (v_i << 32) // root."""
    out = []
    two_f = 2 * _NORM_FRAC_BITS
    for row in rows:
        total = sum(int(v) * int(v) for v in row)
        root = max(_i_sqrt_oracle((total << two_f) // hidden), 1)
        out.append([(int(v) << two_f) // root for v in row])
    return out


def _iexp_kernel(q: int, q_ln2: int, q_b: int, q_c: int) -> int:
    """The I-BERT second-order kernel (S6.3; C7/C8 constants supplied): on a max-shifted
    logit q <= 0, clipped at -N*q_ln2; z = -clipped // q_ln2; p = clipped + z*q_ln2;
    ((p + q_b)^2 + q_c) >> z. The kernel is UNCHANGED under C30 -- only its constants'
    derivation moved to the pinned R-composed rule."""
    if q > 0:
        raise ValueError("i_exp is defined on max-shifted logits (q <= 0)")
    if q_ln2 < 1:
        raise ValueError("coarse-scale rejection: q_ln2 < 1 (i-exp's existing bound)")
    clipped = max(q, -IEXP_CLIP_N * q_ln2)
    z = -clipped // q_ln2
    q_p = clipped + z * q_ln2
    return ((q_p + q_b) ** 2 + q_c) >> z


def _iexp_constants_from_scale(m: int, e: int) -> tuple[int, int, int]:
    """C30 over the oracle's C7 fixture constants."""
    return iexp_constants_oracle(m, e, IEXP_LN2_Q30, 30, IEXP_B_Q30, 30, IEXP_CA_Q30, 30)


def _softmax_row(scores, width: int, q_ln2: int, q_b: int, q_c: int):
    """S6.3's softmax over one causal row: max-shift, i-exp, Q15 ratio. out_scale
    cancels in the ratio (C30's out_scale-never-computed rule)."""
    present = [int(scores[j]) for j in range(width)]
    peak = max(present)
    exps = [_iexp_kernel(v - peak, q_ln2, q_b, q_c) for v in present]
    total = sum(exps)
    return [(e_v << _PROB_FRAC_BITS) // max(total, 1) for e_v in exps]


def _clamp8(v: int) -> int:
    return -_I8 if v < -_I8 else (_I8 if v > _I8 else v)


def _chain(row):
    """C19-C22 over one wide row: (codes, (Dn, -s)) -- D' in exact canonical form."""
    codes, d_prime, dn, s, r = quantize_row_oracle(row)
    return codes, (dn, -s)


def _rope_rotate(rows, cos_table, sin_table, head_dim: int):
    """C13: pairwise rotation with the single RoundingDivideByPOT rounding (Q2.30),
    position = row index; clamped back to int8 (the activation format)."""
    pairs = head_dim // 2
    out = []
    for position, row in enumerate(rows):
        new = []
        for p in range(pairs):
            x, y = int(row[2 * p]), int(row[2 * p + 1])
            c, s_ = int(cos_table[position][p]), int(sin_table[position][p])
            new.append(_clamp8(rdbpot_oracle(x * c - y * s_, 30)))
            new.append(_clamp8(rdbpot_oracle(x * s_ + y * c, 30)))
        out.append(new)
    return out


def _matmul(rows, weight_t):
    """Exact integer matmul: rows x weight_t (weight_t indexed [in][out])."""
    n_in = len(weight_t)
    n_out = len(weight_t[0])
    return [[sum(int(row[i]) * int(weight_t[i][o]) for i in range(n_in))
             for o in range(n_out)] for row in rows]


def _transpose_weight(w):
    return [[int(w[r][c]) for r in range(len(w))] for c in range(len(w[0]))]


def forward_dynamic_logits_oracle(model, tokens) -> list:
    """int32 logits of the W8A8-dynamic forward, recomputed at arbitrary precision from
    the C23-C30 composition rows (D-SLM56/57/58) -- the E9 bit-exact oracle.

    Reads (data): model.weights (int8 codes), model.weight_scales (raw per-tensor /
    per-channel float scales), the per-head KV landing entries in
    model.scales.nonlinear ("layer{L}.k_head{h}.scale" / "layer{L}.v_head{h}.scale" --
    the A-3 pinned surface), model.rope_tables, model.config. model.biases must be
    empty: the C28 dynamic-arm bias artifact (B[j] at S_ref in q_B) is a converter
    surface the fixture does not carry; a biased model raises rather than silently
    skipping the C28 term.

    Recomputes (formulas): everything -- the site walk of A-5 s14.4 as completed by
    D-SLM57 (left-associated incoming-first scale products; the attn_ctx per-head
    pre-fold) and D-SLM58 (the k/v landing composite with the OFFLINE reciprocal).

    Two adopted micro-readings, recorded in A-7 (each a one-line reconcile if the
    builder read differently): (1) landed k/v codes and rotated q/k codes clamp to
    +/-127 (C22's clamp precedent -- codes are int8); (2) at sites with no incoming
    carried factor (embed, the norms, attn_ctx) the product order is
    offline-constant x (Dn, -s).
    """
    cfg = model.config
    dynamic_biases = getattr(model, "dynamic_biases", None) or {}
    if getattr(model, "biases", None) and not dynamic_biases:
        raise NotImplementedError(
            "the C28 dynamic-arm bias artifact (B[j] at S_ref, q_B) is a converter "
            "surface this oracle reads as data; the model carries biases but no "
            "dynamic_biases surface exists (A-8 pin: model.dynamic_biases[site] = "
            "(q_B, codes))")

    hd = cfg.head_dim
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    group = n_heads // n_kv
    hidden_size = cfg.hidden_size
    steps = len(tokens)

    cos_rows, sin_rows = model.rope_tables
    cos_table = [list(map(int, r)) for r in list(cos_rows)[:steps]]
    sin_table = [list(map(int, r)) for r in list(sin_rows)[:steps]]

    w = {name: (arr.tolist() if hasattr(arr, "tolist") else arr)
         for name, arr in model.weights.items()}
    raw = model.weight_scales

    # ---- offline emissions ----
    inv127 = Fraction(1, 127)
    c_embed = canonical_scale_oracle(Fraction(raw["embed"][0]) * inv127)
    c_residual = canonical_scale_oracle(inv127)
    c_mlp_act = canonical_scale_oracle(inv127 / (1 << _SIGMOID_FRAC_BITS))

    def norm_const(gain_name):
        return canonical_scale_oracle(
            Fraction(raw[gain_name][0]) / (1 << _NORM_FRAC_BITS) * inv127)

    def proj_constants(prefix, name):
        scales = raw[f"{prefix}.{name}"]
        s_ref = max(scales)
        return (_fold_table(scales), s_ref,
                canonical_scale_oracle(Fraction(s_ref) * inv127))

    def kv_raw_scale(prefix, kind, h):
        try:
            return model.scales.scale(f"{prefix}.{kind}_head{h}.scale")
        except KeyError:
            raise AssertionError(
                f"red-unimplemented: the artifact carries no "
                f"'{prefix}.{kind}_head{h}.scale' nonlinear entry — the A-3-pinned "
                f"per-head KV landing surface (calibration emits it; C27/D-SLM58)"
            ) from None

    def kv_targets(prefix, kind, s_ref):
        """D-SLM58 landing constants per head: (m_t, e_t, R_t) over canonical(S_kh/S_ref)."""
        out = []
        for h in range(n_kv):
            s_kh = kv_raw_scale(prefix, kind, h)
            m_t, e_t = canonical_scale_oracle(Fraction(s_kh) / Fraction(s_ref))
            out.append((m_t, e_t, reciprocal_oracle(m_t)))
        return out

    # ---- the entry (C23 embed: scale-carrying; subsumes the A-3 chain) ----
    hidden_codes, hidden_scale = [], []
    for t in tokens:
        codes, dfac = _chain([int(v) for v in w["embed"][int(t)]])
        hidden_codes.append(codes)
        hidden_scale.append(carried_scale_product_oracle([c_embed, dfac]))

    def land_kv(acc_folded, m_a, e_a, targets):
        """D-SLM58: per element, rhafz((acc' * m_a * R_t) / 2^(62-(e_a-e_t))), clamped."""
        return [_clamp8(residual_reconcile_oracle(
                    int(a), m_a, targets[j // hd][2], e_a, targets[j // hd][1]))
                for j, a in enumerate(acc_folded)]

    def residual_add(h_codes, h_scale, b_codes, b_scale):
        """C26: branch reconciled into the stream's domain, summed wide, re-chained."""
        out_codes, out_scale = [], []
        for t in range(steps):
            m_h, e_h = h_scale[t]
            m_b, e_b = b_scale[t]
            r_h = reciprocal_oracle(m_h)
            wide = [int(h_codes[t][i]) +
                    residual_reconcile_oracle(int(b_codes[t][i]), m_b, r_h, e_b, e_h)
                    for i in range(hidden_size)]
            codes, dfac = _chain(wide)
            out_codes.append(codes)
            out_scale.append(carried_scale_product_oracle([(m_h, e_h), c_residual, dfac]))
        return out_codes, out_scale

    def norm_site(h_codes, name):
        """C23 scale-killing: rmsnorm x gain codes, chained; carried scale GAIN-derived."""
        gain = [int(v) for v in w[name + ".gain"]]
        c_norm = norm_const(name + ".gain")
        normed_rows = _rmsnorm_rows(h_codes, hidden_size)
        out_codes, out_scale = [], []
        for t in range(steps):
            codes, dfac = _chain([normed_rows[t][i] * gain[i] for i in range(hidden_size)])
            out_codes.append(codes)
            out_scale.append(carried_scale_product_oracle([c_norm, dfac]))
        return out_codes, out_scale

    def project_dynamic(prefix, name, in_codes, in_scale):
        """Type A (C24/C25 + the chain): fold, chain; carried = incoming x c_proj x D'."""
        folds, _s_ref, c_proj = proj_constants(prefix, name)
        bias = dynamic_biases.get(f"{prefix}.{name}")
        acc = _matmul(in_codes, _transpose_weight(w[f"{prefix}.{name}"]))
        out_codes, out_scale = [], []
        for t in range(steps):
            folded = fold_row_oracle(acc[t], folds)
            if bias is not None:
                q_b, b_codes = bias
                m_a, e_a = in_scale[t]
                r_a = reciprocal_oracle(m_a)     # C28: runtime, per token (denominator per-token)
                folded = [f + bias_reconcile_oracle(int(b), q_b, r_a, e_a)
                          for f, b in zip(folded, b_codes)]
            codes, dfac = _chain(folded)
            out_codes.append(codes)
            out_scale.append(carried_scale_product_oracle([in_scale[t], c_proj, dfac]))
        return out_codes, out_scale

    def project_kv(prefix, name, kind, in_codes, in_scale):
        """C27 as corrected by D-SLM58: fold, then land at the static per-head scales."""
        folds, s_ref, _c = proj_constants(prefix, name)
        targets = kv_targets(prefix, kind, s_ref)
        bias = dynamic_biases.get(f"{prefix}.{name}")
        acc = _matmul(in_codes, _transpose_weight(w[f"{prefix}.{name}"]))
        out = []
        for t in range(steps):
            folded = fold_row_oracle(acc[t], folds)
            if bias is not None:
                q_b, b_codes = bias
                m_a, e_a = in_scale[t]
                r_a = reciprocal_oracle(m_a)
                folded = [f + bias_reconcile_oracle(int(b), q_b, r_a, e_a)
                          for f, b in zip(folded, b_codes)]
            out.append(land_kv(folded, in_scale[t][0], in_scale[t][1], targets))
        return out

    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"

        normed, normed_scale = norm_site(hidden_codes, f"{prefix}.attn_norm")
        q_codes, q_scale = project_dynamic(prefix, "q_proj", normed, normed_scale)
        k_codes = project_kv(prefix, "k_proj", "k", normed, normed_scale)
        v_codes = project_kv(prefix, "v_proj", "v", normed, normed_scale)

        def split(rows, head):
            return [row[head * hd:(head + 1) * hd] for row in rows]

        q_heads = [_rope_rotate(split(q_codes, h), cos_table, sin_table, hd)
                   for h in range(n_heads)]
        k_heads = [_rope_rotate(split(k_codes, h), cos_table, sin_table, hd)
                   for h in range(n_kv)]
        v_heads = [split(v_codes, h) for h in range(n_kv)]

        # softmax.input is per QUERY (C27/C30): S_q(i) x [S_k_head / sqrt(hd)] offline,
        # incoming carried q scale first (D-SLM57), then the C30 derivation.
        sm_consts = []
        for h in range(n_heads):
            kvh = h // group
            s_kh = kv_raw_scale(prefix, "k", kvh)
            c_sm = canonical_scale_oracle(Fraction(s_kh) / Fraction(_math.sqrt(hd)))
            sm_consts.append([
                _iexp_constants_from_scale(
                    *carried_scale_product_oracle([q_scale[i], c_sm]))
                for i in range(steps)])

        context = [[0] * (n_heads * hd) for _ in range(steps)]
        for h in range(n_heads):
            kvh = h // group
            scores = _matmul(q_heads[h], _transpose_weight(k_heads[kvh]))
            for i in range(steps):
                q_ln2, q_b, q_c = sm_consts[h][i]
                probs = _softmax_row(scores[i], i + 1, q_ln2, q_b, q_c)
                for d in range(hd):
                    context[i][h * hd + d] = sum(
                        probs[j] * int(v_heads[kvh][j][d]) for j in range(i + 1))

        # attn_ctx (C23 + D-SLM57): per-head pre-fold to max_head S_v, then the chain.
        s_v = [kv_raw_scale(prefix, "v", h) for h in range(n_kv)]
        s_v_max = max(s_v)
        seg_folds = []
        for h in range(n_heads):
            f_v = s_v[h // group]
            entry = None if f_v == s_v_max else _quantize_multiplier_offline(f_v / s_v_max)
            seg_folds.extend([entry] * hd)
        c_ctx = canonical_scale_oracle(
            Fraction(s_v_max) / (1 << _PROB_FRAC_BITS) * inv127)
        ctx_codes, ctx_scale = [], []
        for t in range(steps):
            codes, dfac = _chain(fold_row_oracle(context[t], seg_folds))
            ctx_codes.append(codes)
            ctx_scale.append(carried_scale_product_oracle([c_ctx, dfac]))

        attn_codes, attn_scale = project_dynamic(prefix, "o_proj", ctx_codes, ctx_scale)
        hidden_codes, hidden_scale = residual_add(
            hidden_codes, hidden_scale, attn_codes, attn_scale)

        normed, normed_scale = norm_site(hidden_codes, f"{prefix}.mlp_norm")
        gate_codes, gate_scale = project_dynamic(prefix, "gate_proj", normed, normed_scale)
        up_codes, up_scale = project_dynamic(prefix, "up_proj", normed, normed_scale)

        # mlp_act (C23 scale-carrying; C10's fixed-point LUT sigmoid from S_gate(token),
        # not i-exp-sigmoid -- F-S3-1, reconciled 2026-07-27).
        act_codes, act_scale = [], []
        for t in range(steps):
            m_g, e_g = gate_scale[t]
            wide = [int(gate_codes[t][i]) *
                    silu_sigmoid_q15_oracle(_SILU_LUT_TABLE, int(gate_codes[t][i]), m_g, e_g) *
                    int(up_codes[t][i])
                    for i in range(cfg.intermediate_size)]
            codes, dfac = _chain(wide)
            act_codes.append(codes)
            act_scale.append(carried_scale_product_oracle(
                [gate_scale[t], up_scale[t], c_mlp_act, dfac]))

        down_codes, down_scale = project_dynamic(prefix, "down_proj", act_codes, act_scale)
        hidden_codes, hidden_scale = residual_add(
            hidden_codes, hidden_scale, down_codes, down_scale)

    normed, _ = norm_site(hidden_codes, "final_norm")
    head_name = "embed" if cfg.tie_word_embeddings else "lm_head"
    logits = _matmul(normed, _transpose_weight(w[head_name]))
    for row in logits:
        for v in row:
            if abs(v) > INT32_MAX:
                raise OverflowError(
                    f"a logit of magnitude {abs(v)} exceeds the int32 selection width")
    return logits
