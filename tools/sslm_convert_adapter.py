"""sslm_convert_adapter.py — B0: converter-side delta-fold derivation (T-2021/T-2029, design
`Claude/Vitruvius/t1977-v5-lora-composition-reconciliation-design-2026-08-13.md` Sec4/Sec11 B0,
Wizard repo).

Ports the arithmetic `tests/t2018-slora-serial/t2018_offline_red.cpp`'s B0 cells already prove
correct (`DeriveTripleNew`, `RealizedRatio`, the WIRING-AND-DERIVATION gate) into the real Python
converter target `SuperSLM_Plan.md` Sec11 names — "the converter's adapter mode." Offline, no
engine change (design Sec11 B0's own scope): this module derives the two new per-channel triples
a runtime-additive LoRA adapter's artifact section carries —

  - `delta_identity`/`delta_mult`/`delta_exponent`, one triple per adapted-projection output
    channel (design Sec9's `DeltaFoldScales` array), converting the delta's own raw-accumulator
    scale into the base's post-WSC1-fold currency;
  - `u_identity`/`u_mult`/`u_exponent`, one triple per rank index (design Sec9's `UFoldScales`
    array), converting the intermediate rank-`r` product into the engine's int8 activation-code
    domain before it feeds the second low-rank GEMM.

Both triples are consumed at runtime by `ApplyAmplifyingWeightScaleFold` (design Sec4, D-SLM2915),
a NEW engine primitive (B1a/B2's own build obligation) — this module derives the triples but does
not implement or call that primitive; it derives the SAME triple the primitive is proven to apply
correctly (`t2018_offline_red.cpp`'s own `DeriveTripleNew`/`ApplyAmplifyingWeightScaleFold` pair),
so B1a/B2's C++ implementation and this module's Python derivation are two independent
implementations of the same specification (design Sec4), not one re-derived from the other.

WIRING-AND-DERIVATION (design Sec11 B0, D-SLM2979/D-SLM3018/D-SLM3023): this module's own
`gate_check` reproduces the two-part acceptance gate the real converter runs on its own output
before emitting an artifact section — WIRING (branch-selection: the applied triple matches the one
independently re-derived from the same required ratio) and DERIVATION (ratio-self-consistency:
the applied triple's algebraically-decoded realized ratio agrees with `T*beta[i]/S`, recomputed
from the primary calibration artifacts, within the triple format's own constructive 2^-31
precision). Per the design's own final rescoping (D-SLM3018/D-SLM3023/D-SLM3025), this gate proves
branch-selection and ratio-self-consistency GIVEN the converter's own `T` — never composed-quality
sufficiency, which is B3's alone.

This module writes no artifact bytes — `DeltaFoldScales`/`UFoldScales` do not exist as on-disk
section types until B0b lands (design Sec11 B0b, a build blocker between B0 and B1a; T-2018/T-2027
Sec18.6 states why B0b is engine work, not this file's). B0's own red-suite cells (§4, T-2018) are
all pure arithmetic with no artifact-writing assertion, which is why B0 can open and green before
B0b exists — this module's public surface is the derivation, not the section writer.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

# Domain constants, mirrored from the design (Sec4, D-SLM2915/D-SLM2943) and from
# `t2018_offline_red.cpp`'s own `kAmpMin`/`kAmpMax` — never re-derived independently, so a drift on
# either side is a single-source diff.
AMPLIFYING_SCALE_EXPONENT_MIN = -31
AMPLIFYING_SCALE_EXPONENT_MAX = 31
# Constructive quantization bound (design Sec4/Sec18): half a unit in the last place of a 31-bit
# unsigned mantissa field — 2^-31, not fitted to any measured cell.
DERIVATION_REL_BOUND = 1.0 / 2147483648.0  # 2^-31


@dataclass(frozen=True)
class AmplifyingTriple:
    """One `(identity, mult, exponent)` triple — the on-disk row shape design Sec9 specifies for
    both `DeltaFoldScales` and `UFoldScales`. `identity in {0,1}`; `exponent` is SIGNED, in
    `[AMPLIFYING_SCALE_EXPONENT_MIN, AMPLIFYING_SCALE_EXPONENT_MAX]` — the widened domain
    `ApplyAmplifyingWeightScaleFold` (design Sec4) consumes, distinct from the base WSC1 triple's
    unsigned `[0,31]` domain (`ApplyWeightScaleFold`'s own, unmodified)."""

    identity: int
    mult: int
    exponent: int


def derive_amplifying_triple(rho: float) -> Optional[AmplifyingTriple]:
    """Design Sec4's own derivation: the fixed-point ratio between a raw accumulator's native
    scale and the target it must land in, represented as `(identity, mult, exponent)` for
    `ApplyAmplifyingWeightScaleFold` (real primitive, B1a/B2's own build). Port of
    `t2018_offline_red.cpp`'s `DeriveTripleNew` — same `frexp`-based mantissa/exponent
    decomposition, same rounding, same domain check, so the two are bit-for-bit the same
    derivation over the same inputs (verified by this module's own test suite reproducing every
    B0 cell `t2018_offline_red.cpp` already proves).

    Returns `None` when `rho` is non-positive or the required exponent falls outside
    `[AMPLIFYING_SCALE_EXPONENT_MIN, AMPLIFYING_SCALE_EXPONENT_MAX]` — design Sec6 item 2's
    `AmplifyingScaleRatioOutOfDomain` signal, B0's own explicit-infeasibility disposition
    (D-SLM2917): a genuinely infeasible ratio is signaled explicitly, never silently clamped to
    `identity=1`.
    """
    # T-2041 (Poirot c81e48c review, Minor 4): a non-finite `rho` (NaN or +/-inf -- reachable from
    # malformed calibration input upstream, e.g. a zero `s_value`/`t_value` before Minor 4's own
    # zero-guards below existed) must reach this function's own `AmplifyingScaleRatioOutOfDomain`
    # signal (`None`) like any other out-of-domain ratio, never an uncaught OverflowError (inf) or
    # ValueError (NaN) from `round()` a few lines down.
    if not math.isfinite(rho):
        return None
    if rho == 1.0:
        return AmplifyingTriple(identity=1, mult=0, exponent=0)
    if rho <= 0.0:
        return None
    m, e2 = math.frexp(rho)  # rho == m * 2**e2, 0.5 <= |m| < 1.0
    q = round(m * 2147483648.0)  # round to the nearest 31-bit unsigned mantissa
    if q == 2147483648:  # rounding pushed the mantissa to exactly 2^31 -- renormalize
        q //= 2
        e2 += 1
    exponent = -e2
    if exponent < AMPLIFYING_SCALE_EXPONENT_MIN or exponent > AMPLIFYING_SCALE_EXPONENT_MAX:
        return None
    if q < 1 or q > 2147483647:
        return None
    return AmplifyingTriple(identity=0, mult=q, exponent=exponent)


def realized_ratio(triple: AmplifyingTriple) -> float:
    """DERIVATION's own reference (design Sec18/Sec19, D-SLM2979/D-SLM3018): pure algebraic decode
    of a fixed-point triple back into the ratio it represents. Port of `t2018_offline_red.cpp`'s
    `RealizedRatio`, verbatim arithmetic."""
    if triple.identity != 0:
        return 1.0
    return (triple.mult / 2147483648.0) * math.exp2(-triple.exponent)


def required_ratio_independent(t_value: float, beta_i: float, s_value: float) -> float:
    """DERIVATION's own independent recombination (design Sec18/Sec19): the required ratio
    recomputed from the base's and adapter's own primary calibration artifacts — `T`, `beta[i]`,
    `S` — never from a mechanism's own combined `rho` local. Port of `t2018_offline_red.cpp`'s
    `RequiredRatioIndependent`, verbatim arithmetic.

    Per the design's own final rescoping (D-SLM3018/D-SLM3023): `beta[i]` (the adapter's own
    per-channel calibrated static scale) and `S` (the base's shared reference scale) are genuinely
    primary — calibration outputs fixed before this derivation runs. `T` is NOT primary — it is
    itself a converter-derived output (`compute_t`, below) — so this function establishes
    ratio-self-consistency GIVEN the converter's own `T`, never independence of `T` itself. A
    mis-derived `T` moves both sides of `gate_check`'s comparison by the identical factor and
    leaves no footprint this gate can find (design Sec4, D-SLM3018) — that residual is B3's alone
    (the baked-adapter comparator), not this module's to close.
    """
    return (t_value * beta_i) / s_value


@dataclass(frozen=True)
class GateResult:
    """WIRING-AND-DERIVATION's own verdict for one applied triple (design Sec11 B0)."""

    wiring_matches: bool
    derivation_rel_err: float
    derivation_matches: bool  # derivation_rel_err <= DERIVATION_REL_BOUND
    gate_passes: bool  # wiring_matches AND derivation_matches


def gate_check(applied: AmplifyingTriple, required_rho: float, t_value: float, beta_i: float,
               s_value: float) -> GateResult:
    """The two-part acceptance gate (design Sec11 B0, D-SLM2979/D-SLM3018/D-SLM3023) a real
    converter runs on every channel of its own derived output before emitting an artifact section.

    WIRING — branch-selection only: the applied triple must be bit-identical to the triple
    independently re-derived from the SAME `required_rho` the mechanism under test used. This
    refuses a partial repair (some channels left on a different fold) by construction, but it
    cannot see a defect in `required_rho`'s own derivation (design Sec18's own T-2004 finding).

    DERIVATION — ratio-self-consistency given `T`: the applied triple's algebraically-decoded
    realized ratio is compared against `T*beta[i]/S`, recomputed independently from the primary
    calibration artifacts, within the constructive `DERIVATION_REL_BOUND` (2^-31). This refuses a
    constant-factor or off-by-one mis-derivation of `required_rho` itself (T-2004's own class), but
    — per the design's own final correction (T-2007/D-SLM3006/D-SLM3018) — it CANNOT see a
    mis-derived `T`, because `T` cancels out of the composed ratio in exact arithmetic and both
    sides of this comparison read the SAME (possibly wrong) `T` local. This is a documented,
    accepted scope limit, not a defect in this gate — sufficiency against a mis-derived `T` is
    B3's alone (the baked-adapter comparator).
    """
    ref = derive_amplifying_triple(required_rho)
    wiring_matches = (
        ref is not None
        and applied.identity == ref.identity
        and applied.mult == ref.mult
        and applied.exponent == ref.exponent
    )
    rho_indep = required_ratio_independent(t_value, beta_i, s_value)
    realized = realized_ratio(applied)
    rel_err = abs(realized - rho_indep) / rho_indep if rho_indep != 0.0 else float("inf")
    derivation_matches = rel_err <= DERIVATION_REL_BOUND
    return GateResult(
        wiring_matches=wiring_matches,
        derivation_rel_err=rel_err,
        derivation_matches=derivation_matches,
        gate_passes=wiring_matches and derivation_matches,
    )


def compute_t(alpha: List[float], u_acc_magnitudes: List[float]) -> float:
    """Design Sec4/D-SLM2916's own definition, the amplifying primitive's revision
    (`T` chosen solely to fill the intermediate INT8 domain, unconstrained by any `rho_u <= 1`
    legality requirement that no longer applies once the amplifying primitive represents any
    resulting ratio exactly): `T = max_k(alpha_k * max|u_acc[k]| / 127)`.

    `alpha` is the adapter's own per-rank calibrated static scale (a primary input, known at
    conversion time from A's own calibration, `SuperSLM_Plan.md` Sec11(iii)). `u_acc_magnitudes`
    is `max|u_acc[k]|` over the calibration corpus, one entry per rank index `k` — a genuinely
    converter-derived quantity (the corpus is what makes `T` an output of calibration, not a
    primary input; design Sec18/Sec19's own correction, D-SLM3018).
    """
    if not alpha:
        return 1.0
    t_value = max(
        (a * m / 127.0 for a, m in zip(alpha, u_acc_magnitudes)),
        default=0.0,
    )
    return t_value if t_value > 0.0 else 1.0


def derive_delta_fold_triples(s_value: float, adapter_beta: List[float], t_value: float,
                               ) -> Tuple[List[Optional[AmplifyingTriple]], List[float]]:
    """One `delta_identity`/`delta_mult`/`delta_exponent` triple per adapted-projection output
    channel (design Sec9's `DeltaFoldScales` array; design Sec4's `rho_delta[i] = T*beta[i]/S`).

    `s_value` is the base's own shared reference scale (`S = max_i base_channel_scales[i]`,
    computed by the CALLER from the base artifact's own WSC1 arrays per design Sec2/Sec4) — this
    function takes the already-reduced scalar, never the per-channel array it was reduced from.
    T-2041 (Poirot c81e48c review, Minor 5): a prior `base_channel_scales: List[float]` parameter
    was `del`-ed on this function's own first line, unread — dropped entirely rather than kept for
    "call-site symmetry" that no call site actually needed.

    Returns `(triples, required_ratios)` — the per-channel derived triple (`None` where
    `AmplifyingScaleRatioOutOfDomain` fires, design Sec6 item 2) and the required ratio each triple
    was derived from, so a caller can run `gate_check` per channel without recomputing `rho`.
    """
    # T-2041 (Minor 4): a zero (or non-finite) `s_value` must reach `AmplifyingScaleRatioOutOfDomain`
    # (every entry `None`) like any other infeasible ratio, never an uncaught ZeroDivisionError --
    # `derive_amplifying_triple` itself now guards non-finite `rho`, but the division below runs
    # BEFORE that guard ever sees the result.
    if s_value == 0.0 or not math.isfinite(s_value):
        return [None] * len(adapter_beta), [float("nan")] * len(adapter_beta)
    required = [t_value * b / s_value for b in adapter_beta]
    triples = [derive_amplifying_triple(rho) for rho in required]
    return triples, required


def derive_u_fold_triples(adapter_alpha: List[float], t_value: float,
                           ) -> Tuple[List[Optional[AmplifyingTriple]], List[float]]:
    """One `u_identity`/`u_mult`/`u_exponent` triple per rank index (design Sec9's `UFoldScales`
    array; design Sec4's extension, `rho_u[k] = alpha[k]/T`)."""
    # T-2041 (Minor 4): same zero/non-finite guard as derive_delta_fold_triples, for `t_value`.
    if t_value == 0.0 or not math.isfinite(t_value):
        return [None] * len(adapter_alpha), [float("nan")] * len(adapter_alpha)
    required = [a / t_value for a in adapter_alpha]
    triples = [derive_amplifying_triple(rho) for rho in required]
    return triples, required


# =================================================================================================
# B7 — Rejection/fallback wiring (design Sec7/Sec11 B7, D-SLM3095, T-2022 fold of Dan's ruling
# D-SLM3066 item 4). Offline, depends on B3/B6. Ported from the proven dispatcher shape
# `t2018_offline_red.cpp`'s own `Dispatch`/`DispatchResult` already establishes (T-2027 Sec18.4) —
# the real converter CLI's own branch-selection logic, per this design's own text ("the CLI emits
# a merge+quantize artifact, with a diagnostic naming which check failed").
# =================================================================================================

class RejectionBranch(Enum):
    """Design Sec7's own named diagnostic taxonomy — checked in this priority order (a row that
    cannot even be represented is a harder failure than a fidelity or saturation measurement,
    matching Sec7's own reject-over-degrade discipline: fail on the most fundamental violation
    first)."""

    NONE = 0
    DOMAIN_REJECTION_TRIP = 1
    RUNTIME_VS_BAKED_MARGIN_EXCEEDED = 2  # renamed from "composed parity miss", D-SLM3047
    SATURATION_RATE_ELEVATION = 3


class ArtifactOutcome(Enum):
    """Design Sec7 (D-SLM3095): the amended, EXPLICIT-ONLY fallback contract — a validation
    failure emits NO artifact by default; producing a merge+quantize artifact instead requires
    the caller's own explicit opt-in (`--fallback=merge`), never an automatic substitution."""

    RUNTIME_ADDITIVE = 0
    NO_ARTIFACT_EMITTED = 1
    MERGE_QUANTIZE_EMITTED = 2


def dispatch_conversion_outcome(domain_trip: bool, margin_exceeded: bool, saturation_elevated: bool,
                                 fallback_flag_present: bool) -> Tuple[RejectionBranch, ArtifactOutcome]:
    """The real converter CLI's own branch-selection logic (design Sec7): given the three
    validation verdicts B3/B6 already compute, names which named diagnostic branch (if any) fired,
    and — per the amended, explicit-only fallback (D-SLM3095) — whether an artifact is emitted at
    all. A clean input (no branch fires) always emits the runtime-additive artifact, regardless of
    the flag; a failing input emits nothing unless `fallback_flag_present`, in which case it emits
    a merge+quantize artifact instead — never a partial or degraded runtime-additive one.
    """
    if domain_trip:
        branch = RejectionBranch.DOMAIN_REJECTION_TRIP
    elif margin_exceeded:
        branch = RejectionBranch.RUNTIME_VS_BAKED_MARGIN_EXCEEDED
    elif saturation_elevated:
        branch = RejectionBranch.SATURATION_RATE_ELEVATION
    else:
        branch = RejectionBranch.NONE

    if branch == RejectionBranch.NONE:
        return branch, ArtifactOutcome.RUNTIME_ADDITIVE
    if fallback_flag_present:
        return branch, ArtifactOutcome.MERGE_QUANTIZE_EMITTED
    return branch, ArtifactOutcome.NO_ARTIFACT_EMITTED
