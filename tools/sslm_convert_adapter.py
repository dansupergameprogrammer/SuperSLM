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

T-2046 (design §24, D-SLM3157-3172, T-2043 fold of D-SLM3154's routed blocker): this module now
ALSO writes artifact bytes. B0b landed the two on-disk section types (T-2021/T-2029); §24.2 pins
the full standalone adapter-artifact byte format around them (a new `Provenance`/`ADP1`
sub-format, `Weights`/`WGT1` for the adapter's own A/B, `DeltaFoldScales`/`UFoldScales` via the
pre-existing `write_tensor_manifest`) and §24.3 pins the `--fallback=merge` numerical procedure —
both specified for the first time by that fold, closing Poirot's `c81e48c` review Significant 6.
The writer/CLI additions live in this file's own bottom section, below `dispatch_conversion_outcome`.
"""

import json
import math
import os
import shutil
import struct
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np  # already a hard dependency of sslm_model_writer.py, this file's own sibling

# T-2046 (design §24.2): both siblings live in this same tools/ directory and carry only
# standard-library/numpy dependencies (no cross-repo spike import) -- eager, not lazy, matching
# sslm_model_writer.py's own eager `import numpy as np`. Python auto-adds this file's own
# directory to sys.path whether it is run directly (`python tools/sslm_convert_adapter.py`) or
# imported by a test collected from the same directory (pytest's own default import mode), so no
# explicit sys.path insert is needed here -- convert_model.py's own bare
# `import sslm_convert_manifest as M` is the same convention.
import sslm_format as sf
import sslm_model_writer as smw

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


# =================================================================================================
# T-2046 -- design §24 (D-SLM3157-3172, T-2043 fold of D-SLM3154's routed blocker): the
# runtime-additive adapter artifact writer (§24.2) and the merge+quantize fallback (§24.3), both
# specified for the first time by that fold. Closes Poirot c81e48c review Significant 6.
#
# NAMED RESIDUAL, stated once here rather than scattered per function: B6's own K/V saturation-
# rate metric (design §6 item 3) is proven, real, and discriminating in
# `tests/t2018-slora-serial/t2018_offline_red.cpp`'s own suite (`MeasureKvSaturationRate`,
# executed) -- but its ACCEPTANCE THRESHOLD is explicitly named UNDERIVED at design §13 ("§6's
# composed-parity margin and K/V saturation-rate acceptance threshold are underived... routed to
# Charpy's temper for a bar recommendation once measured"), with no bootstrap-self-calibration
# procedure specified for it the way §6 item 1/D-SLM3101 specifies one for B3's own margin. Unlike
# B3 (below), there is no ratified way to turn a real saturation-rate measurement into an
# accept/reject boolean today, and porting `LandingRescale`'s own fixed-point saturation-counting
# arithmetic to Python — a real primitive this build has never independently re-implemented and
# would have no cross-check for — is exactly the kind of undesigned scope D-SLM3154 already routed
# rather than improvised once this fold. `saturation_elevated` is therefore wired to `False`
# below: not "always passes" (a lenient number), but "no ratified bar exists to compare a real
# measurement against" -- the honest reading of reject-over-degrade when the reject SIDE of the
# comparison does not yet exist. Domain-trip and B3's own margin check ARE fully computed, real,
# and gate for real (below).
# =================================================================================================

import re
import sys

# T-2046 (design §24.2 D-SLM3158): kPeftAdaptableProjections' own fixed 7-entry order
# (model.h:433-435) -- the ONLY source of ADP1's target_modules_mask bit order. PEFT's own on-disk
# key naming nests each leaf name under its real transformer submodule; named once here.
PEFT_ADAPTABLE_PROJECTIONS = ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "k_proj", "v_proj")
_PROJECTION_FULL_PATH = {
    "q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj", "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj", "up_proj": "mlp.up_proj", "down_proj": "mlp.down_proj",
}


def _load_spike():
    """Lazy import (T-2123/T-2137: `reference_pipeline`, vendored in-tree at
    tools/reference_pipeline/ -- no longer a cross-tree import), matching `convert_model.py`'s
    own `_load_spike` -- keeps this module importable (and its 22 existing B0 unit tests
    fast/dependency-free) on a bare checkout; only the writer/CLI functions below ever call
    this."""
    from reference_pipeline import pipeline  # noqa: E402
    return pipeline


class AdapterRejected(ValueError):
    """§11's existing reject-over-degrade allow-list (DoRA, base-mutating inits,
    `fan_in_fan_out=True`, a trained bias, an unrepresented rank_pattern/alpha_pattern), raised
    with a named diagnostic -- shared machinery both the runtime-additive writer and
    `--fallback=merge` depend on (D-SLM3165's own "rejected upstream of either fallback path"
    text), rather than a bare AssertionError."""


def check_adapter_allowlist(adapter_config: dict) -> None:
    """Mirrors `tools/t2029_b3_execute.py`'s own `load_adapter()` asserts, promoted to a raising,
    reusable function (its own ad hoc form was a bootstrap script's, not a real CLI's)."""
    if adapter_config.get("init_lora_weights") is not True:
        raise AdapterRejected(
            f"init_lora_weights={adapter_config.get('init_lora_weights')!r}: only an "
            f"additive-safe LoRA init (init_lora_weights=True) is supported")
    if adapter_config.get("use_dora"):
        raise AdapterRejected("use_dora=True: DoRA is base-mutating, not additive -- not supported")
    if adapter_config.get("bias", "none") != "none" or adapter_config.get("lora_bias"):
        raise AdapterRejected(
            f"bias={adapter_config.get('bias')!r} lora_bias={adapter_config.get('lora_bias')!r}: "
            f"a trained bias is not this design's additive delta -- not supported")
    if adapter_config.get("fan_in_fan_out"):
        raise AdapterRejected("fan_in_fan_out=True -- not supported")
    if adapter_config.get("rank_pattern") or adapter_config.get("alpha_pattern"):
        raise AdapterRejected(
            "rank_pattern/alpha_pattern (PEFT per-module rank/alpha overrides) is non-empty -- "
            "named as an explicit, undesigned residual (design §24.6 D-SLM3171's own third item): "
            "ADP1's rank field carries only the adapter's single declared rank, not a per-module "
            "override table; such an adapter needs that extension before it can convert, not a "
            "best-effort guess at this format's own still-open shape")


@dataclass(frozen=True)
class AdapterMeta:
    """One PEFT adapter's declared metadata (`adapter_config.json`), post-allowlist."""
    rank: int
    lora_alpha: float
    use_rslora: bool
    target_modules: Tuple[str, ...]  # short names, already checked against PEFT_ADAPTABLE_PROJECTIONS
    scaling: float  # lora_alpha / (sqrt(r) if use_rslora else r) -- SuperSLM_Plan.md §11(b)(ii)


def load_adapter_meta(adapter_dir: Path) -> AdapterMeta:
    with open(Path(adapter_dir) / "adapter_config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    check_adapter_allowlist(cfg)
    unknown = sorted(set(cfg["target_modules"]) - set(PEFT_ADAPTABLE_PROJECTIONS))
    if unknown:
        raise AdapterRejected(
            f"target_modules names {unknown!r}, outside the seven PEFT-adaptable projections "
            f"{PEFT_ADAPTABLE_PROJECTIONS!r} (model.h:433-435) -- not supported")
    r = int(cfg["r"])
    lora_alpha = float(cfg["lora_alpha"])
    use_rslora = bool(cfg.get("use_rslora", False))
    scaling = lora_alpha / (math.sqrt(r) if use_rslora else r)
    return AdapterMeta(rank=r, lora_alpha=lora_alpha, use_rslora=use_rslora,
                       target_modules=tuple(sorted(cfg["target_modules"])), scaling=scaling)


def target_modules_mask(target_modules) -> int:
    """ADP1's bit `i` set iff `PEFT_ADAPTABLE_PROJECTIONS[i]` is targeted (design §24.2 D-SLM3158)."""
    targeted = set(target_modules)
    mask = 0
    for i, name in enumerate(PEFT_ADAPTABLE_PROJECTIONS):
        if name in targeted:
            mask |= (1 << i)
    return mask


def _resolve_base_checkpoint_dir(adapter_dir: Path) -> Path:
    """The RAW HF checkpoint directory this adapter was trained against, read from the adapter's
    own `adapter_config.json` (`base_model_name_or_path`) -- distinct from `--base <base.sslm>`
    (design §24.2 D-SLM3163), which names the already-converted artifact whose CFG1/hash this
    writer copies (D-SLM3161). Needed because the `.sslm` carries only quantized int8 weights;
    the B3 float reference and the merge formula both need the checkpoint's own float tensors."""
    with open(Path(adapter_dir) / "adapter_config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise AdapterRejected("adapter_config.json carries no base_model_name_or_path")
    p = Path(base)
    if not (p / "config.json").exists():
        raise AdapterRejected(f"base_model_name_or_path {base!r} has no config.json there")
    return p


def read_peft_lora_pair(adapter_dir: Path, layer: int, proj_short: str, meta: AdapterMeta):
    """One (layer, projection)'s own A/B float64 tensors, PEFT's on-disk orientation (A: [r, in],
    B: [out, r]) -- `SuperSLM_Plan.md` §11(b)(i) -- returned as `(A_float, B_scaled_float)`, B
    already carrying PEFT's own scaling folded in (§11(b)(ii)), matching
    `tools/t2029_b3_execute.py`'s own `load_adapter()`.

    T-2194: bf16 is the prevailing PEFT/LoRA training default, and bf16 is not a numpy dtype --
    the `safetensors` library's own `framework="numpy"` binding has no bfloat16 representation and
    raises `TypeError: data type bfloat16 not understood` out of its own `get_tensor()` before this
    function's `.astype(np.float64)` ever ran. Read through `_SafeTensors` instead (this module's
    own sibling, `reference_pipeline.pipeline`) -- the SAME manual safetensors parser and exact
    bit-shift widening (`raw.view(uint16).astype(uint32) << 16`, reinterpreted as float32) the base
    converter's own `_open_checkpoint_tensors` path already uses to read the base checkpoint
    (pipeline.py `_SafeTensors.tensor`), rather than inventing a second bf16 convention here.
    bf16->float32 widening is LOSSLESS: bf16 is float32's top 16 bits, so the shift zero-fills the
    dropped mantissa bits and no rounding occurs. `_SafeTensors.tensor()` also raises for any
    dtype it does not widen exactly (`UnsupportedOpSet`), sweeping in the same class of defect for
    fp16 and any other on-disk dtype this reader might otherwise silently mis-cast, not only bf16.
    Numpy-only and dependency-free -- no torch import, matching `requirements.txt`'s own
    declaration that the vendored reference pipeline never imports torch.
    """
    pipeline = _load_spike()
    full_path = _PROJECTION_FULL_PATH[proj_short]
    key_a = f"base_model.model.model.layers.{layer}.{full_path}.lora_A.weight"
    key_b = f"base_model.model.model.layers.{layer}.{full_path}.lora_B.weight"
    reader = pipeline._SafeTensors(Path(adapter_dir) / "adapter_model.safetensors")
    a_f = reader.tensor(key_a)
    b_f = reader.tensor(key_b)
    return a_f, b_f * meta.scaling


def read_base_projection_weight(tensors, layer: int, proj_short: str) -> np.ndarray:
    """One projection's `[out, in]` float64 weight, read directly from the base checkpoint's own
    open tensor source -- UNPERMUTED (PEFT trains against the checkpoint's own raw orientation;
    RoPE's pair-permutation is applied only inside `pipeline.load_model`'s own processing loop,
    never at this raw-tensor level -- matching `t2029_b3_execute.py`'s own `load_base_weight()`,
    which reads the identical raw key)."""
    full_path = _PROJECTION_FULL_PATH[proj_short]
    return tensors.tensor(f"model.layers.{layer}.{full_path}.weight")


_ADP1_MAGIC = b"ADP1"
_ADP1_VERSION = 1
_ADP1_FIXED_BYTES = 68


def write_adp1(*, rank: int, target_modules, base_artifact_hash: bytes, lora_alpha: float,
               use_rslora: bool, source_adapter_name: str) -> bytes:
    """Design §24.2 D-SLM3158's own byte layout, exactly: magic(4)+version(4)+rank(4)+
    target_modules_mask(4)+base_artifact_hash(32)+lora_alpha(8,f64)+use_rslora(4)+reserved(4)+
    source_adapter_name_len(4)+name -- 68-byte fixed prefix, little-endian throughout."""
    if len(base_artifact_hash) != 32:
        raise ValueError(f"base_artifact_hash must be 32 bytes, got {len(base_artifact_hash)}")
    name_bytes = source_adapter_name.encode("utf-8")
    buf = bytearray()
    buf += _ADP1_MAGIC
    buf += struct.pack("<I", _ADP1_VERSION)
    buf += struct.pack("<I", int(rank))
    buf += struct.pack("<I", target_modules_mask(target_modules))
    buf += bytes(base_artifact_hash)
    buf += struct.pack("<d", float(lora_alpha))
    buf += struct.pack("<I", 1 if use_rslora else 0)
    buf += struct.pack("<I", 0)  # reserved -- == 0
    buf += struct.pack("<I", len(name_bytes))
    buf += name_bytes
    assert len(buf) == _ADP1_FIXED_BYTES + len(name_bytes), \
        f"ADP1 size {len(buf)} != expected {_ADP1_FIXED_BYTES + len(name_bytes)}"
    return bytes(buf)


# --- design §6 item 1 / D-SLM3101: B3's own bootstrap self-calibration, ported verbatim (same
# statistic, same PILOT/VALIDATION split, same 1.5x safety inflation, same z=1.645) from
# `tools/t2029_b3_execute.py`'s own `run_token`/`collect_gaps`/`stat`, generalized from the one
# representative (layer, projection) pair that first proved the machinery to every adapted pair
# this writer derives -- "the first adapter converted through this pipeline serves as its own
# reference at bootstrap" (design's own words), applied per-pair. ---

_B3_Z_95_ONE_SIDED = 1.645
_B3_SAFETY_INFLATION = 1.5
_B3_PILOT_N = 200


def _b3_is_pilot_item(seed: int) -> bool:
    """Deterministic 80/20 validation/pilot split by item-hash, verbatim from
    `t2029_b3_execute.py`'s own `is_pilot_item`."""
    z = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return (z % 10) < 2


def _b3_stat(gaps: List[float]) -> dict:
    n = len(gaps)
    mean = float(np.mean(gaps)) if n else 0.0
    sd = float(np.std(gaps, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    upper_ci = mean + _B3_Z_95_ONE_SIDED * se
    p95 = float(np.percentile(gaps, 95)) if n else 0.0
    return {"n": n, "mean": mean, "se": se, "upper_ci": upper_ci, "p95": p95}


def _derive_amplifying_triples_vectorized(rho):
    """Numpy-array form of `derive_amplifying_triple`, element-for-element identical -- same
    branches, same order (non-finite -> invalid; `rho==1.0` -> identity; `rho<=0.0` -> invalid;
    else the `frexp`-based derivation and its own domain check), restated as array ops purely for
    per-draw performance in `run_b3_bootstrap_check`'s own hot path (T-2046: a per-channel Python
    loop over up to 8960 channels x 400 draws x 196 adapted pairs was the actual measured cost,
    not the numpy matmuls beside it). Returns `(identity, mult, exponent, valid)`, all arrays the
    same shape as `rho`; `valid[i] is False` is this array form's own `None` (`identity`/`mult`/
    `exponent` at an invalid index are 0 and MUST NOT be read).
    """
    rho = np.asarray(rho, dtype=np.float64)
    n = rho.shape[0]
    identity = np.zeros(n, dtype=np.int64)
    mult = np.zeros(n, dtype=np.int64)
    exponent = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)

    finite = np.isfinite(rho)
    is_one = finite & (rho == 1.0)
    identity[is_one] = 1
    valid[is_one] = True

    general_mask = finite & ~is_one & (rho > 0.0)
    safe_rho = np.where(general_mask, rho, 1.0)  # placeholder for frexp on non-general entries
    with np.errstate(invalid="ignore", over="ignore"):
        m, e2 = np.frexp(safe_rho)
        q = np.round(m * 2147483648.0)
        overflow = general_mask & (q == 2147483648.0)
        q = np.where(overflow, q / 2.0, q)
        e2 = np.where(overflow, e2 + 1, e2)
    exp_ = (-e2).astype(np.int64)
    q_i = q.astype(np.int64)
    dom_ok = general_mask & (exp_ >= -31) & (exp_ <= 31) & (q_i >= 1) & (q_i <= 2147483647)
    mult = np.where(dom_ok, q_i, mult)
    exponent = np.where(dom_ok, exp_, exponent)
    valid = valid | dom_ok
    return identity, mult, exponent, valid


def _realized_ratio_vectorized(identity, mult, exponent):
    """Numpy-array form of `realized_ratio`, element-for-element identical."""
    is_id = identity != 0
    r = (mult.astype(np.float64) / 2147483648.0) * np.exp2(-exponent.astype(np.float64))
    return np.where(is_id, 1.0, r)


def _b3_collect_pair_raw_draws(w_f, a_f, b_scaled, Wc, w, S, Ac, alpha, Bc, beta, T_honest,
                               pilot_n: int = _B3_PILOT_N, seed_base: int = 0x1000,
                               validation_seed_base: int = 0x9000) -> dict:
    """One (layer, proj) pair's own raw per-draw `composed_gap`/`effect_distance_runtime` arrays,
    split by the deterministic PILOT/VALIDATION partition -- the shared computation both the
    per-pair diagnostic (`run_b3_bootstrap_check`, below) and the pooled corpus-level report
    (`run_b3_pooled_report`, T-2065 design §25.5, retired to a report by T-2213/D-SLM3783) are
    built from, so the two never risk independently drifting from each other's own arithmetic.

    T-2065 (design §25, D-SLM3201-3208, T-2058 disposition of D-SLM3185's own routed 196-pair
    sweep interpretation): extracted, unchanged in every formula/branch, from what was
    `run_b3_bootstrap_check`'s own inline `collect`/`run_token` closures before this fold -- byte-
    for-byte the same computation `Claude/Vitruvius/t2058-sweep-interpretation/
    t2058_full_sweep_stats.py`'s own read-only `collect_raw` already reproduced and cross-checked
    bit-exact against the original T-2046 sweep's own D-SLM3128 figures. Returns
    `{"composed_pilot", "composed_val", "effect_pilot", "effect_val"}`, each a 1-D `np.ndarray`,
    plus `"delta_norm_sq"` (T-2213, D-SLM3783): this pair's own squared Frobenius norm of its
    actual composed LoRA delta (`b_scaled @ a_f`), summed across pairs and square-rooted by
    `run_b3_pooled_report` into the candidate's pooled `delta_norm` -- the magnitude sanity
    check's own input.

    PERFORMANCE NOTE (T-2046, not present in `t2029_b3_execute.py`'s own single-pair probe): the
    four per-channel triple sets (u-fold, base-fold, delta-fold, baked-fold) depend only on this
    PAIR's own fixed `alpha`/`w`/`S`/`beta`/`wp`/`Sp`/`T_honest` -- never on the per-draw synthetic
    token -- so they are derived ONCE here, outside the 2*`pilot_n`-draw loop below, via
    `_derive_amplifying_triples_vectorized`/`_realized_ratio_vectorized` (this module's own
    `derive_amplifying_triple`/`realized_ratio`, restated as numpy array ops for real per-draw
    performance across every adapted (layer, proj) pair -- cross-checked element-for-element
    against the scalar functions by this file's own new pin cell). Only the RAW accumulators
    (`u_acc`/`acc`/`delta_raw`/`bacc`) genuinely vary per draw; multiplying by a precomputed
    realized-ratio array is the only per-draw cost this function pays for the fold step.
    """
    pipeline = _load_spike()
    d_out, d_in = w_f.shape
    r = a_f.shape[0]
    lora_delta = b_scaled @ a_f  # T-2213 (D-SLM3783): this pair's own actual composed LoRA delta --
                                 # reused below as `delta_norm_sq`, the magnitude sanity check's
                                 # input (tools/sslm_convert_adapter.py's own `run_b3_pooled_report`),
                                 # rather than a fresh computation.
    w_prime = w_f + lora_delta
    Wpc, wp_scales = pipeline.quantize_weight_per_channel(w_prime, output_axis=0)
    wp = np.asarray(wp_scales, dtype=np.float64)
    Sp = float(np.max(wp))

    # Per-pair-constant triple derivation, hoisted out of the draw loop (see docstring above).
    u_id, u_mult, u_exp, u_valid = _derive_amplifying_triples_vectorized(alpha / T_honest)
    u_realized = np.where(u_valid, _realized_ratio_vectorized(u_id, u_mult, u_exp), 1.0)
    base_id, base_mult, base_exp, base_valid = _derive_amplifying_triples_vectorized(w / S)
    base_realized = np.where(base_valid, _realized_ratio_vectorized(base_id, base_mult, base_exp), 1.0)
    rho_delta = T_honest * beta / S
    delta_id, delta_mult, delta_exp, delta_valid = _derive_amplifying_triples_vectorized(rho_delta)
    delta_realized = np.where(delta_valid, _realized_ratio_vectorized(delta_id, delta_mult, delta_exp), 1.0)
    baked_id, baked_mult, baked_exp, baked_valid = _derive_amplifying_triples_vectorized(wp / Sp)
    baked_realized = np.where(baked_valid, _realized_ratio_vectorized(baked_id, baked_mult, baked_exp), 1.0)

    def _fold(realized, valid, acc):
        return np.where(valid, np.round(realized * acc).astype(np.int64), acc)

    def run_token(seed):
        g = np.random.default_rng(seed)
        xf = g.standard_normal(d_in)
        xmax = float(np.max(np.abs(xf)))
        X = xmax / 127.0 if xmax > 0.0 else 1.0
        xc = np.clip(np.round(xf / X), -127, 127).astype(np.int64)

        u_acc = Ac.astype(np.int64) @ xc
        u_wide = _fold(u_realized, u_valid, u_acc)
        u_i8 = np.clip(u_wide, -127, 127)

        acc = Wc.astype(np.int64) @ xc
        acc_wide = _fold(base_realized, base_valid, acc)

        delta_raw = Bc.astype(np.int64) @ u_i8
        delta_wide = _fold(delta_realized, delta_valid, delta_raw)

        scale = X * S
        scale_baked = X * Sp
        runtime_composed = (acc_wide + delta_wide) * scale
        runtime_base = acc_wide * scale

        bacc = Wpc.astype(np.int64) @ xc
        bacc_wide = _fold(baked_realized, baked_valid, bacc)
        baked_composed = bacc_wide * scale_baked

        yb = w_f @ xf
        yd = b_scaled @ (a_f @ xf)
        ref = yb + yd

        composed_l2_runtime = math.sqrt(np.sum((runtime_composed - ref) ** 2)) / math.sqrt(np.sum(ref ** 2))
        composed_l2_baked = math.sqrt(np.sum((baked_composed - ref) ** 2)) / math.sqrt(np.sum(ref ** 2))

        effect_runtime = runtime_composed - runtime_base
        shared_base = acc_wide * scale
        effect_baked = baked_composed - shared_base
        eff_den = math.sqrt(np.sum(yd ** 2))
        if eff_den == 0.0:
            eff_den = 1.0
        effect_l2_runtime = math.sqrt(np.sum((effect_runtime - yd) ** 2)) / eff_den
        effect_l2_baked = math.sqrt(np.sum((effect_baked - yd) ** 2)) / eff_den
        return composed_l2_runtime, composed_l2_baked, effect_l2_runtime, effect_l2_baked

    def collect(n_items, seed_base_local):
        composed_pilot, composed_val, effect_rt_pilot, effect_rt_val = [], [], [], []
        for i in range(n_items):
            seed = (seed_base_local + i * 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF
            cl_rt, cl_bk, el_rt, _el_bk = run_token(seed)
            composed_gap = cl_rt - cl_bk
            if _b3_is_pilot_item(seed):
                composed_pilot.append(composed_gap)
                effect_rt_pilot.append(el_rt)
            else:
                composed_val.append(composed_gap)
                effect_rt_val.append(el_rt)
        return composed_pilot, composed_val, effect_rt_pilot, effect_rt_val

    c_pilot, _c_val1, e_pilot, _e_val1 = collect(pilot_n, seed_base)
    _c_pilot2, c_val, _e_pilot2, e_val = collect(pilot_n, validation_seed_base)

    return {"composed_pilot": np.asarray(c_pilot, dtype=np.float64),
           "composed_val": np.asarray(c_val, dtype=np.float64),
           "effect_pilot": np.asarray(e_pilot, dtype=np.float64),
           "effect_val": np.asarray(e_val, dtype=np.float64),
           # T-2213 fix round (D-SLM3787 finding C1): a plain Python `float` has no `.tolist()`,
           # and every other value in this dict is an `np.ndarray` -- `build_runtime_additive_
           # sections`'s checkpoint writer serializes the whole dict via
           # `{k: v.tolist() for k, v in raw.items()}`, so a bare `float` here raised
           # `AttributeError` on the first checkpointed pair. `np.sum(...)` already returns an
           # `np.float64` scalar, which carries `.tolist()` (-> a plain Python float on decode)
           # exactly like an `np.ndarray` does -- dropping the `float()` cast is the fix.
           "delta_norm_sq": np.sum(lora_delta ** 2)}


def run_b3_bootstrap_check(w_f, a_f, b_scaled, Wc, w, S, Ac, alpha, Bc, beta, T_honest,
                           pilot_n: int = _B3_PILOT_N, seed_base: int = 0x1000,
                           validation_seed_base: int = 0x9000) -> dict:
    """One (layer, proj) pair's own PILOT partition calibrates the four Deltas (composed
    mean/tail, effect mean/tail); the SAME pair's own VALIDATION partition is graded against its
    own just-frozen Deltas. Returns a dict with `accepted` and the measured figures.

    T-2065 (design §25.5 item 2, D-SLM3205): this per-pair self-calibrated check was never the
    PRIMARY accept/reject gate after that fold (`run_b3_multi_pair_check` was, T-2065 through
    T-2212; T-2213/D-SLM3783 retired that pooled gate's own arithmetic entirely, leaving no
    pooled accept/reject gate at all) -- it is the DIAGNOSTIC layer: its own `accepted`/`Delta`/
    validation figures are what `_b3_pair_diagnostic` (below) reads to decide whether a pair's own
    margin against ITS OWN self-calibrated Delta exceeds the 2x-bootstrap-SE review threshold
    (§25.5 item 2's own "matching §25.2's own margin analysis"), which is now this tool's primary
    B3 signal (`run_b3_pooled_report`'s own `per_pair_diagnostics`). Unchanged arithmetic from the
    pre-T-2065 form -- same Deltas, same accept/reject shape -- only its ROLE changed.
    """
    raw = _b3_collect_pair_raw_draws(w_f, a_f, b_scaled, Wc, w, S, Ac, alpha, Bc, beta, T_honest,
                                     pilot_n=pilot_n, seed_base=seed_base,
                                     validation_seed_base=validation_seed_base)
    c_pilot_stat = _b3_stat(raw["composed_pilot"])
    e_pilot_stat = _b3_stat(raw["effect_pilot"])
    delta_composed_mean = _B3_SAFETY_INFLATION * (c_pilot_stat["mean"] + _B3_Z_95_ONE_SIDED * c_pilot_stat["se"])
    delta_composed_tail = _B3_SAFETY_INFLATION * c_pilot_stat["p95"]
    delta_effect_mean = _B3_SAFETY_INFLATION * (e_pilot_stat["mean"] + _B3_Z_95_ONE_SIDED * e_pilot_stat["se"])
    delta_effect_tail = _B3_SAFETY_INFLATION * e_pilot_stat["p95"]

    c_val_stat = _b3_stat(raw["composed_val"])
    e_val_stat = _b3_stat(raw["effect_val"])

    composed_mean_accepts = c_val_stat["upper_ci"] < delta_composed_mean
    composed_tail_accepts = c_val_stat["p95"] < delta_composed_tail
    effect_mean_accepts = e_val_stat["upper_ci"] < delta_effect_mean
    effect_tail_accepts = e_val_stat["p95"] < delta_effect_tail
    accepted = composed_mean_accepts and composed_tail_accepts and effect_mean_accepts and effect_tail_accepts

    return {"accepted": accepted,
           "delta_composed_mean": delta_composed_mean, "delta_composed_tail": delta_composed_tail,
           "delta_effect_mean": delta_effect_mean, "delta_effect_tail": delta_effect_tail,
           "validation_composed_upper_ci": c_val_stat["upper_ci"], "validation_composed_p95": c_val_stat["p95"],
           "validation_effect_upper_ci": e_val_stat["upper_ci"], "validation_effect_p95": e_val_stat["p95"],
           "raw": raw}


def _bootstrap_se(arr, stat_fn, n_resamples: int, rng) -> float:
    """Nonparametric bootstrap SE of `stat_fn` over `arr` (resample-with-replacement, `n_resamples`
    draws) -- design §25.5 item 3's own resolving-power term for the P95 (tail) estimator, which
    (unlike the mean conjuncts' own parametric `z*SE`) has no closed-form sampling-SE formula.
    Verbatim from `Claude/Vitruvius/t2058-sweep-interpretation/t2058_full_sweep_stats.py`'s own
    already-executed, already-cross-checked construction (Wizard repo, T-2058)."""
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return 0.0
    vals = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        vals[i] = stat_fn(arr[idx])
    return float(np.std(vals, ddof=1))


def _p95(arr) -> float:
    return float(np.percentile(arr, 95))


_B3_REVIEW_MARGIN_SE = 2.0  # design §25.5 item 2's own stated multiple, matching §25.2's analysis.

# T-2213 (D-SLM3783): the pooled ACCEPT/REJECT arithmetic this notice used to describe is
# RETIRED, not repaired -- five adversary strikes (Claude/Loki/t2205, t2207, t2209, t2210, t2211,
# Claude/Vitruvius/t2204's own fold round 4, Wizard repo) established that its accept boundary was
# one frozen reference adapter's own idiosyncrasy (honest in-domain adapters reject on magnitude
# alone, 9 of 18 at >=2.5x the reference, and on pure structure alone, 20 of 60 at the reference's
# own magnitude) and that no in-band corruption construction ever elevated the statistic once
# magnitude was intercepted. Dan's ruling: delete the arithmetic, ship what the arc actually
# proved -- the per-pair diagnostics below (bias-corrected, D-SLM3221/T-2201) and the magnitude
# sanity check beneath them. This notice states what each can and cannot do; neither is a verdict.
_B3_POOLED_GATE_STATUS_NOTICE = (
    "B3 pooled gate status: the whole-adapter accept/reject statistic this tool once computed "
    "here has been retired -- it never discriminated a healthy adapter from a corrupted one on "
    "its own merits, only from one frozen reference adapter's own idiosyncratic scale. Nothing "
    "below blocks writing an artifact based on adapter quality; only a domain trip (an "
    "unrepresentable ratio) can do that. The per-pair diagnostics are this tool's primary B3 "
    "review signal: a pair named 'flagged' is worth a closer look, not a confirmed finding, and "
    "an EMPTY per-pair list is not proof the adapter is sound -- these diagnostics have not been "
    "shown able to flag every real magnitude error, so zero pairs named here means the "
    "diagnostics found nothing to name, not that nothing is wrong. The magnitude sanity check "
    "below is a coarse, wide-tolerance distance check against a reference adapter's own composed "
    "delta norm, when one is configured -- it never refuses an artifact either; a candidate "
    "outside its tolerance is a named WARNING with an UNRESOLVED disposition, worth review, not "
    "a finding of corruption."
)


def _b3_pair_diagnostic(name: str, raw: dict, own_check: dict, *, n_bootstrap_resamples: int,
                        rng) -> dict:
    """One pair's own diagnostic record (design §25.5 item 2): for each of the four conjuncts,
    the margin between this pair's own VALIDATION-partition point estimate and its OWN
    self-calibrated `Delta` (from `run_b3_bootstrap_check`, unchanged arithmetic), in units of that
    conjunct's own resolving power -- the PARAMETRIC `upper_CI`-implied SE for the mean conjuncts
    (already carried by `_b3_stat`), the BOOTSTRAP SE of the VALIDATION-partition P95 for the tail
    conjuncts (§25.2's own dual-track construction, restated verbatim here). A conjunct-instance
    with `margin_se > 2.0` is a NAMED FINDING (never gates); `0 <= margin_se <= 2.0` is
    UNRESOLVED-band noise (D-SLM2846's own discipline: never rounded to a pass or a fail);
    `margin_se < 0` clears with margin. Diagnostic only -- `flagged` never sets `margin_exceeded`.
    """
    c_val = raw["composed_val"]
    e_val = raw["effect_val"]
    c_val_stat = _b3_stat(c_val)
    e_val_stat = _b3_stat(e_val)
    se_composed_tail = _bootstrap_se(c_val, _p95, n_bootstrap_resamples, rng)
    se_effect_tail = _bootstrap_se(e_val, _p95, n_bootstrap_resamples, rng)
    se_composed_mean = c_val_stat["se"]
    se_effect_mean = e_val_stat["se"]

    def _margin(point, delta, se):
        return (point - delta) / se if se > 0.0 else float("inf") if point > delta else float("-inf")

    # D-SLM3221 (design §26.9): the mean conjuncts' margin is graded against the VALIDATION
    # partition's own point estimate (`mean`), matching the tail conjuncts' use of a raw `p95`
    # point estimate immediately below -- never `upper_ci`, which already carries its own
    # `+1.645*se` term and, substituted here, added a second, spurious `+1.645` SE to every
    # mean-conjunct margin relative to the tail conjuncts' correctly-paired form.
    margins = {
        "composed_mean": _margin(c_val_stat["mean"], own_check["delta_composed_mean"], se_composed_mean),
        "composed_tail": _margin(c_val_stat["p95"], own_check["delta_composed_tail"], se_composed_tail),
        "effect_mean": _margin(e_val_stat["mean"], own_check["delta_effect_mean"], se_effect_mean),
        "effect_tail": _margin(e_val_stat["p95"], own_check["delta_effect_tail"], se_effect_tail),
    }
    flagged = [conjunct for conjunct, m in margins.items() if m > _B3_REVIEW_MARGIN_SE]
    return {"name": name, "own_accepted": own_check["accepted"], "margins_se": margins, "flagged": flagged}


_B3_MAGNITUDE_WARN_RATIO = 10.0
# T-2213 fix round (D-SLM3787 item 1, superseding D-SLM3785): NOT a derived boundary. The shipped
# 10.0x is an arbitrary, coarse, one-decade-wide sanity constant, kept only because the FULL
# executed census this constant could be checked against shows no ratio -- on either side -- that
# separates honest candidates from the arc's own "corrupted" construction on this axis.
#
# The check's own axis is `delta_norm`'s ratio to a fixed reference. Read directly, without
# excluding any candidate, from `Claude/Vitruvius/t2204-fold-round4-probe/
# t2204r4_magnitude_domain_output.json` (Wizard repo; both drawing `a_f` and `b_scaled` from one
# shared `scale` parameter, so `delta_norm` scales QUADRATICALLY in `scale` for the honest set,
# and the "corrupt" set is that same honest baseline with `b_scaled` further multiplied by a
# constant `k`): the 18 honest candidates' `ratio_to_ref` spans 0.0099x-100.0x (the earlier
# ~6.2-6.3x figure came from "honest scale=0.05" alone -- 3 of the 18 honest rows; the fuller
# 6-of-18 in-band slice examined at the time also included "scale=0.02", omitting the "scale=0.1"
# (~25x) and "scale=0.2" (~100x) rows the SAME probe run and file also report); the 8 corruption
# candidates' `ratio_to_ref` spans 0.1x-100000.0x, with two readings at EXACTLY 10.0x and one at
# 0.5x. Corroborated row-for-row at the shipped band by the SAME probe's
# `t2204r4_magnitude_domain_output_band0.1-10.json` sibling (`band` field `[0.1, 10.0]`, identical
# ratios for every one of the 26 labels). `Claude/Loki/t2210-probe/t2210_magnitude_axis_output.json`
# replays the SAME 26 labels and confirms the constructions correspond, but its own rows carry no
# `delta_norm`/`ratio_to_ref` field and cannot corroborate the axis this check gates. On the high
# side the honest population's own max (100.0x) sits TEN TIMES ABOVE the lowest corruption reading
# that side has (10.0x) -- not a gap, an inversion: some "corrupted" candidates read a smaller
# ratio than some honest ones. On the low side a corruption (0.5x) sits well inside the honest
# range (which itself reaches down to 0.0099x). No threshold, symmetric or one-sided, separates
# these two populations on this axis, because the "corruption" construction here (an honest
# baseline with `B` uniformly rescaled) and ordinary honest scale variation are the same
# transformation of the same underlying weights -- the metric cannot tell training-time magnitude
# choice apart from post-hoc rescaling, which is the same conclusion the whole B3 pooled-gate arc
# reached about raw magnitude generally (this ticket's own commission, above). D-SLM3787's own
# "25x honest max to 41.6x corruption floor" does not survive this recomputation: the 25x figure
# silently excluded the census's own ~100x rows, and 41.6x is
# `Claude/Loki/t2207-t2204-restrike-2026-08-20.md`'s composed-STATISTIC margin (D-SLM3750) --
# a different quantity, never measured on this ratio axis at all, repeating exactly the
# quantity-confusion `Claude/Poirot/8af620a-t2214-gate-retirement-confirmation.md` finding S2
# raised against the figure this comment previously shipped.
#
# 10.0x therefore ships as a plain, non-calibrated sanity distance -- present for visibility,
# never a discriminator, never a REJECT (see below). On the executed census it warns on 12 of the
# 18 honest candidates -- 6 rows at ratio > 10x ("scale=0.1", ~25x, and "scale=0.2", ~100x) and 6
# rows at ratio < 0.1x ("scale=0.002", ~0.01x, and "scale=0.005", ~0.06x); the check's own
# inclusive lower band catches the second six -- and stays silent on 3 of the 8 corruptions (the
# two at exactly 10.0x, inclusive compare, and the one at 0.5x). A caller who wires a real
# reference adapter should expect the warning to fire on honest conversions at either extreme of
# scale and should not read its absence as a health signal.


def run_b3_pooled_report(pair_draws, *, reference_delta_norm: Optional[float] = None,
                         magnitude_warn_ratio: float = _B3_MAGNITUDE_WARN_RATIO,
                         n_bootstrap_resamples: int = 2000,
                         bootstrap_seed: int = 0xB007, verbose: bool = False) -> dict:
    """T-2213 (D-SLM3783): the whole-adapter pooled REPORT -- no longer a gate. The pooled
    ACCEPT/REJECT arithmetic this function used to compute (design §25.5, D-SLM3205) is RETIRED:
    five adversary strikes (Claude/Loki/t2205, t2207, t2209, t2210, t2211,
    Claude/Vitruvius/t2204's own fold round 4, Wizard repo) established its accept boundary was
    one frozen reference adapter's own idiosyncrasy, not a property of adapter health, and Dan's
    ruling (D-SLM3783) retired the arithmetic rather than continuing to repair it.

    What ships instead:

    - **`per_pair_diagnostics`** (`_b3_pair_diagnostic`, design §25.5 item 2, arithmetic itself
      unchanged) -- T-2213 fix round (D-SLM3787 finding S3): this function's own numbers ARE
      affected by the retirement even though the diagnostic's own arithmetic is not. The two
      `_bootstrap_se` calls this function used to make against the pooled population, before this
      per-pair loop, are deleted along with the retired verdict; they used to consume the shared
      `rng` first, so every pair's own bootstrap draws below shifted to a different point in the
      stream the moment those two calls were removed (executed on a constructed pair:
      `composed_tail` margin 2.4227 -> 2.4316). The loop now seeds its own `rng` at its own point
      of first use (below), so this cannot happen again from an unrelated change elsewhere in this
      function. Reports every pair's own bias-corrected margin against its OWN self-calibrated
      Delta, never gating, and is the primary B3 review signal this tool reports
      (`_B3_POOLED_GATE_STATUS_NOTICE`).
    - **`delta_norm`** (T-2213, new): the candidate's pooled composed LoRA delta norm --
      `sqrt(sum over pairs of delta_norm_sq)`, reusing `_b3_collect_pair_raw_draws`'s own
      already-computed `b_scaled @ a_f` product per pair, no fresh computation.
    - **`magnitude_warning`** (T-2213, new): `None` unless `reference_delta_norm` is supplied AND
      `delta_norm` sits outside `[reference_delta_norm / magnitude_warn_ratio,
      reference_delta_norm * magnitude_warn_ratio]` (derivation above `_B3_MAGNITUDE_WARN_RATIO`).
      When present it is a named finding with `disposition: "unresolved"` -- it NEVER sets
      anything an emission decision reads; `build_runtime_additive_sections`'s own
      `margin_exceeded` is unconditionally `False` now that this function computes no verdict.

    `pair_draws`: an ordered list of `(name, raw)` pairs, `raw` being `_b3_collect_pair_raw_draws`'s
    own return shape -- one entry per adapted (layer, projection) pair.

    Returns `{"delta_norm", "magnitude_warning", "n_pairs", "n_pilot_pooled", "n_val_pooled",
    "per_pair_diagnostics": [...]}`.
    """
    if not pair_draws:
        raise ValueError("run_b3_pooled_report: pair_draws is empty -- no pairs to pool")
    # T-2213 fix round (D-SLM3787 finding M3): a ratio <= 1.0 collapses the tolerance band to a
    # point (1.0) or inverts it (< 1.0, so `lo > hi` and every candidate warns).
    if magnitude_warn_ratio <= 1.0:
        raise ValueError(f"magnitude_warn_ratio must be > 1.0 (got {magnitude_warn_ratio}) -- a "
                         "ratio <= 1.0 collapses or inverts the [reference/ratio, reference*ratio] "
                         "tolerance band")

    # T-2213 fix round (D-SLM3787 finding M4): `pooled_composed_pilot`/`pooled_composed_val` used
    # to be concatenated here and read nowhere except `.shape[0]` below -- a full copy of the
    # whole pooled population (196 pairs in production) to compute a length. `sum(len(...))` is
    # exact and free.
    n_pilot_pooled = sum(raw["composed_pilot"].shape[0] for _name, raw in pair_draws)
    n_val_pooled = sum(raw["composed_val"].shape[0] for _name, raw in pair_draws)

    # T-2213 fix round (D-SLM3787 finding S1): every value in `raw` is required, not defaulted --
    # `_b3_collect_pair_raw_draws` always sets `delta_norm_sq`, and the resume path above now
    # recomputes it for any checkpointed pair that predates T-2213 rather than omitting it, so a
    # missing key here means a `raw` dict from somewhere else entirely and should raise, not
    # silently read as 0.
    delta_norm = math.sqrt(sum(raw["delta_norm_sq"] for _name, raw in pair_draws))
    magnitude_warning = None
    if reference_delta_norm is not None and reference_delta_norm > 0.0:
        ratio = delta_norm / reference_delta_norm
        lo, hi = 1.0 / magnitude_warn_ratio, magnitude_warn_ratio
        if not (lo <= ratio <= hi):
            magnitude_warning = {
                "candidate_delta_norm": delta_norm, "reference_delta_norm": reference_delta_norm,
                "ratio_to_reference": ratio, "tolerance": [lo, hi],
                "disposition": "unresolved",
                "reason": f"pooled composed LoRA delta norm is {ratio:.3g}x the reference's, "
                          f"outside the [{lo:.3g}x, {hi:.3g}x] wide-tolerance sanity band "
                          f"(_B3_MAGNITUDE_WARN_RATIO={magnitude_warn_ratio}) -- a named WARNING "
                          f"for review, never a REJECT",
            }
        if verbose and magnitude_warning is not None:
            print(f"  [B3 MAGNITUDE WARNING] {magnitude_warning['reason']}")

    # T-2213 fix round (D-SLM3787 finding S3, item 2): this loop's own rng, instantiated at its
    # own point of first use rather than shared with anything computed above it in this function.
    # Before this fix `rng` was created once at the top of the function and threaded through
    # every `_b3_pair_diagnostic` call below; two `_bootstrap_se` calls that used to run against
    # the pooled population, before this loop, were deleted by T-2213's own retirement (D-SLM3783)
    # -- so every pair's diagnostic silently began drawing from a different point in the stream
    # than it did before the retirement, with no diagnostic code itself touched (executed:
    # `composed_tail` margin 2.4227 -> 2.4316, `effect_tail` 1.3834 -> 1.4165 on a constructed
    # pair). Creating the generator here, at the loop that is its only consumer, means a future
    # change anywhere else in this function -- adding, removing, or reordering a pooled
    # computation -- cannot perturb these numbers again the way the retirement itself just did.
    rng = np.random.default_rng(bootstrap_seed)

    per_pair_diagnostics = []
    for name, raw in pair_draws:
        own_check_pilot_c = _b3_stat(raw["composed_pilot"])
        own_check_pilot_e = _b3_stat(raw["effect_pilot"])
        own_check = {
            "delta_composed_mean": _B3_SAFETY_INFLATION * (own_check_pilot_c["mean"] +
                                                            _B3_Z_95_ONE_SIDED * own_check_pilot_c["se"]),
            "delta_composed_tail": _B3_SAFETY_INFLATION * own_check_pilot_c["p95"],
            "delta_effect_mean": _B3_SAFETY_INFLATION * (own_check_pilot_e["mean"] +
                                                          _B3_Z_95_ONE_SIDED * own_check_pilot_e["se"]),
            "delta_effect_tail": _B3_SAFETY_INFLATION * own_check_pilot_e["p95"],
        }
        own_val_c = _b3_stat(raw["composed_val"])
        own_val_e = _b3_stat(raw["effect_val"])
        own_check["accepted"] = (
            own_val_c["upper_ci"] < own_check["delta_composed_mean"]
            and own_val_c["p95"] < own_check["delta_composed_tail"]
            and own_val_e["upper_ci"] < own_check["delta_effect_mean"]
            and own_val_e["p95"] < own_check["delta_effect_tail"])
        per_pair_diagnostics.append(_b3_pair_diagnostic(name, raw, own_check,
                                                        n_bootstrap_resamples=n_bootstrap_resamples, rng=rng))
        if verbose:
            d = per_pair_diagnostics[-1]
            if d["flagged"]:
                print(f"  [B3 review-flag] {name}: {d['flagged']} margins_se={d['margins_se']}")

    return {
        "delta_norm": delta_norm,
        "magnitude_warning": magnitude_warning,
        "n_pairs": len(pair_draws),
        "n_pilot_pooled": int(n_pilot_pooled), "n_val_pooled": int(n_val_pooled),
        "per_pair_diagnostics": per_pair_diagnostics,
    }


# =============================================================================================
# Design §26.15, AS CORRECTED (T-2099) -- the freeze-time conversion-health gate.
#
# WHAT THIS GATES. At freeze time a CANDIDATE adapter's conversion health is compared against a
# matched REFERENCE adapter's, per tier: Tier 1 over the pooled draw population, Tier 2 once per
# projection type present. The decided quantity is a RATIO -- `stat(candidate) / stat(reference)`
# for `stat` in {sample sd, p95} -- checked against a band, and the verdict is three-valued
# (D-SLM3288): UNRESOLVED first, else PASS if in band, else REFUSE. `freeze_allowed` is true only
# on an overall PASS.
#
# WHY THE ORIGINAL §26.15 REPAIR WAS ONE-SIDED, and what changed (the external review's finding,
# 2026-08-14; D-SLM3295). §26.15 derived resolving power as `band / relSE(stat(REFERENCE))` --
# the bootstrap relative SE of the reference population ALONE (`margin_of(ref_pooled, band)` in
# `Claude/Vitruvius/t2096-verdict-repair-probe/t2096_verdict_repair_probe.py`) -- and its
# escalation instruction re-collected the REFERENCE only, leaving the candidate at the production
# default `pilot_n=200`. Statistical power for a ratio depends on BOTH samples. Two consequences,
# and the second is the one that makes the gate unsafe rather than merely imprecise:
#
#   1. The margin is overstated. With equal-sized samples it is too large by a factor of about
#      sqrt(2); as the reference is escalated it grows without bound while the candidate's own
#      contribution to the ratio's uncertainty is untouched.
#   2. **Escalating the reference alone can drive ANY cell to RESOLVED.** As
#      `relSE(reference) -> 0`, the reported margin -> infinity, while `relSE(ratio)` floors at
#      `relSE(candidate)`, which never moved. The specified escalation procedure was therefore a
#      false-RESOLVED generator: spending collector time on the reference buys a RESOLVED verdict
#      on a ratio whose real uncertainty is unchanged.
#
# The corrected form below computes the ratio's own relative SE jointly,
# `relSE(R)^2 = relSE(stat(cand))^2 + relSE(stat(ref))^2` (first-order propagation for the ratio
# of two independent estimates -- the candidate and reference populations are collected from
# disjoint seed streams, `_b3_collect_pair_raw_draws`'s own `seed_base`/`validation_seed_base`
# construction), and escalates BOTH sides together at the same geometric step. That also makes the
# must-reject construction production-feasible for the first time: §26.15's own executed REFUSE
# re-collected the CORRUPTED CANDIDATE at the escalated count, which the specified production
# escalation never did, so the control exercised a data path production could not produce
# (`StandardsDocument.md` §5.4). Under joint escalation the candidate IS re-collected, so the
# construction and the production path are the same path.
#
# A SECOND ONE-SIDEDNESS, closed here rather than left for a later round to find: each tier gates
# on TWO statistics (sd and p95) and §26.15 derived resolving power from the p95 alone. A tier
# whose sd ratio is unresolvable cannot honestly PASS on the p95's resolving power, so the tier's
# achieved margin is the MINIMUM over the statistics that gate it. The p95-only figure is still
# reported, as `margin_p95_only`, so §26.15's own published table stays reproducible.
# =============================================================================================

_FREEZE_BANDS = {"sd_pooled": 0.20, "p95_pooled": 0.10, "sd_type": 0.25, "p95_type": 0.12}
_FREEZE_MARGIN_SE = 2.0            # design §26.15.6 / §26.5's own established precedent.
_FREEZE_N_BOOTSTRAP = 2000
_FREEZE_BOOTSTRAP_SEED = 0xB007
# Executed/fitted single-pair, single-side collector cost law, §26.15.5 (three executed points at
# pilot_n 200/800/3200, cross-validated to 0.7% against a fourth at 4800).
_FREEZE_COST_A, _FREEZE_COST_B = 0.0359, 0.985
# ADAPTER-LEVEL TOTAL escalation budget (D-SLM3296). §26.15.3 derived a 30-minute ceiling for ONE
# pair on ONE side and stated it per-pair; a 196-pair adapter escalating two sides against a
# per-pair ceiling authorises up to 196*2*30 minutes = 196 hours, which is not a ceiling. The
# budget is therefore charged against the WHOLE conversion: 30 minutes of escalation collector
# time total, across every pair and both sides. A per-pair cap survives only as the pilot_n
# ceiling below, which bounds any single step.
_FREEZE_ESCALATION_BUDGET_SECONDS = 1800.0
_FREEZE_PILOT_N_CEILING = 60000    # §26.15.3's own derived per-pair pilot_n ceiling.
_FREEZE_ESCALATION_FACTOR = 2      # geometric; §26.15.3 specifies doubling or similar.

FREEZE_PASS, FREEZE_REFUSE, FREEZE_UNRESOLVED = "PASS", "REFUSE", "UNRESOLVED"


def _freeze_sd(arr) -> float:
    return float(np.std(np.asarray(arr, dtype=np.float64), ddof=1))


def _freeze_p95(arr) -> float:
    return float(np.percentile(np.asarray(arr, dtype=np.float64), 95))


_FREEZE_STATS = (("sd", _freeze_sd), ("p95", _freeze_p95))


def freeze_relative_bootstrap_se(arr, stat_fn, *, n_resamples: int = _FREEZE_N_BOOTSTRAP,
                                 seed: int = _FREEZE_BOOTSTRAP_SEED) -> float:
    """The bootstrap SE of `stat_fn` over `arr`, divided by the point estimate -- one sample's own
    RELATIVE sampling uncertainty, the term that composes into the ratio's below. Returns
    `float('inf')` when the point estimate is zero (no relative uncertainty is defined, and the
    caller must not read that as perfect resolution)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size < 2:
        return float("inf")
    point = stat_fn(arr)
    se = _bootstrap_se(arr, stat_fn, n_resamples, np.random.default_rng(seed))
    if point == 0.0:
        return float("inf")
    return abs(se / point)


def freeze_ratio_relative_se(candidate, reference, stat_fn, *,
                             n_resamples: int = _FREEZE_N_BOOTSTRAP,
                             seed: int = _FREEZE_BOOTSTRAP_SEED) -> float:
    """The relative SE of `stat(candidate) / stat(reference)` -- the quantity the band is actually
    checked against.

    `relSE(R)^2 = relSE(stat(cand))^2 + relSE(stat(ref))^2`, first-order propagation for a ratio of
    two INDEPENDENT estimates. Independence is a property of the collection, not an assumption:
    candidate and reference draws come from disjoint seed streams.

    This is the correction the external review of 2026-08-14 required (D-SLM3295). §26.15 used
    `relSE(stat(reference))` alone, which understates the ratio's uncertainty always, and without
    bound once the reference is escalated and the candidate is not."""
    rel_c = freeze_relative_bootstrap_se(candidate, stat_fn, n_resamples=n_resamples, seed=seed)
    rel_r = freeze_relative_bootstrap_se(reference, stat_fn, n_resamples=n_resamples, seed=seed + 1)
    if math.isinf(rel_c) or math.isinf(rel_r):
        return float("inf")
    return math.sqrt(rel_c * rel_c + rel_r * rel_r)


def _freeze_in_band(value: float, ref: float, band: float) -> bool:
    return bool(ref * (1.0 - band) <= value <= ref * (1.0 + band))


def freeze_tier_verdict(candidate, reference, *, band_sd: float, band_p95: float,
                        n_resamples: int = _FREEZE_N_BOOTSTRAP,
                        seed: int = _FREEZE_BOOTSTRAP_SEED) -> dict:
    """One tier's own three-valued verdict (D-SLM3288), with UNRESOLVED checked FIRST.

    The tier gates on two statistics, so its achieved margin is the MINIMUM of the two -- a tier
    cannot be better resolved than the worst statistic that can reject it. `margin_p95_only` is
    reported alongside for continuity with §26.15's published table, and is never the deciding
    number."""
    bands = {"sd": band_sd, "p95": band_p95}
    per_stat = {}
    for stat_name, stat_fn in _FREEZE_STATS:
        rel = freeze_ratio_relative_se(candidate, reference, stat_fn,
                                       n_resamples=n_resamples, seed=seed)
        band = bands[stat_name]
        # A ratio relative SE of exactly 0 (a zero-dispersion population, where the bootstrap
        # returns 0 under ANY statistic) or of infinity (a zero point estimate, where no relative
        # uncertainty is defined) scores margin 0 and therefore UNRESOLVED -- never "infinitely
        # resolved". `StandardsDocument.md` §5.4: a population with zero dispersion establishes
        # that the instrument is deterministic and nothing about whether it can discriminate, so
        # the conservative direction is the only honest one here.
        margin = (band / rel) if rel > 0.0 and not math.isinf(rel) else 0.0
        c_point, r_point = stat_fn(candidate), stat_fn(reference)
        per_stat[stat_name] = {
            "band": band, "ratio_relative_se": rel, "margin": margin,
            "candidate": c_point, "reference": r_point,
            "ratio": (c_point / r_point) if r_point != 0.0 else float("inf"),
            "in_band": _freeze_in_band(c_point, r_point, band),
        }
    margin = min(s["margin"] for s in per_stat.values())
    in_band = all(s["in_band"] for s in per_stat.values())
    if margin < _FREEZE_MARGIN_SE:
        verdict = FREEZE_UNRESOLVED
    else:
        verdict = FREEZE_PASS if in_band else FREEZE_REFUSE
    return {"verdict": verdict, "margin": margin,
            "margin_p95_only": per_stat["p95"]["margin"],
            "limiting_stat": min(per_stat, key=lambda s: per_stat[s]["margin"]),
            "in_band": in_band, "per_stat": per_stat,
            "n_candidate": int(np.asarray(candidate).size),
            "n_reference": int(np.asarray(reference).size)}


def compose_freeze_verdict(tier_verdicts) -> dict:
    """D-SLM3288's across-tier composition: REFUSE dominates UNRESOLVED dominates PASS, and
    `freeze_allowed` (boolean) is true only on an overall PASS. REFUSE and UNRESOLVED both read
    `freeze_allowed=False` and are NOT the same disposition downstream -- REFUSE is terminal,
    UNRESOLVED is an instruction to the collector -- so the three-valued verdict travels beside
    the boolean and the operator-facing log carries it, never only the boolean."""
    verdicts = list(tier_verdicts)
    if FREEZE_REFUSE in verdicts:
        overall = FREEZE_REFUSE
    elif FREEZE_UNRESOLVED in verdicts:
        overall = FREEZE_UNRESOLVED
    else:
        overall = FREEZE_PASS
    return {"overall": overall, "freeze_allowed": overall == FREEZE_PASS}


def freeze_predicted_step_seconds(pilot_n: int, n_pairs: int) -> float:
    """Predicted collector cost of one escalation step: `n_pairs` pairs, BOTH sides, at `pilot_n`.
    The factor of two is the correction's own direct cost -- joint escalation collects the
    candidate as well as the reference, and a budget that priced only the reference would
    under-charge every step by half."""
    return 2.0 * n_pairs * _FREEZE_COST_A * (float(pilot_n) ** _FREEZE_COST_B)


def run_freeze_health_gate(collect_pair, pair_names, *, projection_type_of,
                           pilot_n: int = _B3_PILOT_N,
                           budget_seconds: float = _FREEZE_ESCALATION_BUDGET_SECONDS,
                           pilot_n_ceiling: int = _FREEZE_PILOT_N_CEILING,
                           escalation_factor: int = _FREEZE_ESCALATION_FACTOR,
                           n_resamples: int = _FREEZE_N_BOOTSTRAP,
                           seed: int = _FREEZE_BOOTSTRAP_SEED,
                           time_fn=None, verbose: bool = False) -> dict:
    """The complete corrected gate: evaluate, and while any tier is UNRESOLVED and budget remains,
    escalate BOTH sides together and re-evaluate the ACHIEVED joint margin.

    `collect_pair(name, pilot_n) -> (candidate_draws, reference_draws)` is the production
    collector, called once per pair per step and returning both sides at the same `pilot_n`.
    Escalating both sides at the same geometric step is the whole correction: escalating the
    reference alone drives the reported margin up without moving the ratio's real uncertainty.

    Stopping, in the order checked: every tier RESOLVED; the next step's PREDICTED cost would
    exceed the remaining ADAPTER-LEVEL budget; or `pilot_n` reaches `pilot_n_ceiling`. On either
    of the latter two the tier's disposition is a terminal, honestly-logged UNRESOLVED rather than
    an unbounded loop -- and the reason is recorded, so "we ran out of budget" is never reported as
    "no difference detected."

    The budget is charged with MEASURED elapsed time, not the fitted prediction; the prediction is
    used only to decide whether to START a step. §26.15.2's own executed counter-example (a
    2-3-point log-log fit that undershot in practice) is the reason the fit never decides an
    outcome here."""
    if time_fn is None:
        time_fn = time.time
    pair_names = list(pair_names)
    if not pair_names:
        raise ValueError("run_freeze_health_gate: pair_names is empty -- no pairs to gate. An "
                         "adapter with no adapted pairs has no conversion health to compare, and "
                         "returning PASS for one would be a gate that cannot fire")
    history = []
    spent = 0.0
    current_pilot_n = int(pilot_n)
    stop_reason = "resolved"
    while True:
        t0 = time_fn()
        draws = {name: collect_pair(name, current_pilot_n) for name in pair_names}
        step_seconds = time_fn() - t0
        spent += step_seconds

        cand_pooled = np.concatenate([np.asarray(draws[n][0], dtype=np.float64) for n in pair_names])
        ref_pooled = np.concatenate([np.asarray(draws[n][1], dtype=np.float64) for n in pair_names])
        tier1 = freeze_tier_verdict(cand_pooled, ref_pooled,
                                    band_sd=_FREEZE_BANDS["sd_pooled"],
                                    band_p95=_FREEZE_BANDS["p95_pooled"],
                                    n_resamples=n_resamples, seed=seed)
        by_type = {}
        for name in pair_names:
            by_type.setdefault(projection_type_of(name), []).append(name)
        tier2 = {}
        for proj_type, names_t in sorted(by_type.items()):
            c_t = np.concatenate([np.asarray(draws[n][0], dtype=np.float64) for n in names_t])
            r_t = np.concatenate([np.asarray(draws[n][1], dtype=np.float64) for n in names_t])
            tier2[proj_type] = freeze_tier_verdict(c_t, r_t,
                                                   band_sd=_FREEZE_BANDS["sd_type"],
                                                   band_p95=_FREEZE_BANDS["p95_type"],
                                                   n_resamples=n_resamples, seed=seed)
        all_tiers = [tier1] + [tier2[t] for t in sorted(tier2)]
        composed = compose_freeze_verdict([t["verdict"] for t in all_tiers])
        step = {"pilot_n": current_pilot_n, "step_seconds": step_seconds,
                "cumulative_seconds": spent, "tier1": tier1, "tier2": tier2,
                "overall": composed["overall"], "freeze_allowed": composed["freeze_allowed"]}
        history.append(step)
        if verbose:
            print(f"  [freeze] pilot_n={current_pilot_n:6d} n_cand={tier1['n_candidate']:6d} "
                  f"T1 margin={tier1['margin']:.3f}x ({tier1['limiting_stat']}) "
                  f"-> {composed['overall']}  [{step_seconds:.1f}s, {spent:.1f}s total]")

        if composed["overall"] != FREEZE_UNRESOLVED:
            stop_reason = "resolved"
            break
        next_pilot_n = current_pilot_n * escalation_factor
        if next_pilot_n > pilot_n_ceiling:
            stop_reason = "pilot_n_ceiling"
            break
        predicted = freeze_predicted_step_seconds(next_pilot_n, len(pair_names))
        if spent + predicted > budget_seconds:
            stop_reason = "adapter_budget_exhausted"
            break
        current_pilot_n = next_pilot_n

    final = history[-1]
    return {"overall": final["overall"], "freeze_allowed": final["freeze_allowed"],
            "stop_reason": stop_reason, "escalation_steps": len(history),
            "final_pilot_n": final["pilot_n"], "seconds_spent": spent,
            "budget_seconds": budget_seconds, "history": history,
            "tier1": final["tier1"], "tier2": final["tier2"]}


def build_runtime_additive_sections(adapter_dir, base_sslm_path, *,
                                    source_adapter_name: Optional[str] = None, verbose: bool = True,
                                    checkpoint_path=None, reference_delta_norm: Optional[float] = None):
    """Design §24.2: assembles the six pinned sections (Config, SigmoidLut, Provenance/ADP1,
    Weights/WGT1, DeltaFoldScales/DFS1, UFoldScales/UFS1) for EVERY adapted (layer, projection)
    pair the adapter's own `target_modules` declares, across every layer the bound base checkpoint
    names. Runs, per pair: the real domain-trip check (this module's own B0 derivation) and the
    real B3 raw-draw collection (`_b3_collect_pair_raw_draws`) -- `saturation_elevated` is the
    named residual stated in this file's own §24 banner comment, never computed here.

    T-2213 (D-SLM3783): B3 no longer has a pooled accept/reject gate. `run_b3_pooled_report` is
    called ONCE after this loop, over every pair's own raw draws, and returns a REPORT (per-pair
    diagnostics, the candidate's own pooled `delta_norm`, and an optional `magnitude_warning`) --
    never a verdict. `margin_exceeded` (below) is therefore unconditionally `False`; nothing B3
    computes can refuse to write an artifact any more (design §25.3's original per-pair AND-gate,
    and design §25.5's pooled replacement, are both retired -- the AND-gate for being structurally
    broken independent of true adapter quality, the pooled replacement per T-2213's own retirement
    lineage above `run_b3_pooled_report`). `run_b3_bootstrap_check`'s own per-pair self-calibrated
    verdict survives only as `run_b3_pooled_report`'s own per-pair DIAGNOSTIC layer (flagged,
    never gating, §25.5 item 2) -- this tool's primary B3 review signal.

    `reference_delta_norm`: optional, passed through to `run_b3_pooled_report` unchanged -- when
    supplied, the candidate's pooled `delta_norm` is compared against it (see
    `_B3_MAGNITUDE_WARN_RATIO`'s own derivation comment for the threshold). `None` (the default)
    means no reference is configured for this conversion and the magnitude sanity check is
    reported but never fires -- this project's own real reference adapter's raw checkpoint no
    longer exists on disk (O1, `Claude/Poirot/6ac7b84-t2209-pooled-gate-confirmation.md`, Wizard
    repo), so no production-calibrated reference value is fabricated here; a caller with a real
    reference adapter available supplies its own delta norm.

    `sections` is `None` only when `domain_trip` fires somewhere (a `None` triple is not
    serializable at all -- structurally excluded, design §6 item 2). Every other pair is derivable
    and assembled: §7 distinguishes "cannot even represent the ratio" (structural, `domain_trip`)
    from "represents it" (nothing else gates whether a pair's own content can be assembled, now
    that B3's pooled quality gate is retired) -- whether the assembled artifact is actually
    WRITTEN to `--out` is `dispatch_conversion_outcome`'s own call (`main()`, below), never this
    function's; this function's only job is "can it be built," never "should it ship."

    Returns `(sections_or_None, verdict, round_trip)`. `verdict` feeds
    `dispatch_conversion_outcome` directly (`domain_trip`/`margin_exceeded`/`saturation_elevated`)
    and carries the pooled report's own full result (`verdict["pooled"]`, including
    `per_pair_diagnostics`, `delta_norm`, `magnitude_warning`) plus each pair's own domain-trip
    status. `round_trip` is this build's own record of what was derived and written, per
    `"layer{L}.{proj}"` key -- what the round-trip proof (T-2046 handoff) compares the REAL C++
    loader's own output against,
    bit-for-bit.

    `checkpoint_path` (T-2046, added after a real 196-pair sweep was killed mid-run by an
    external process-management event, not a crash -- 2026-08-14; format changed under T-2065 to
    persist each pair's own RAW draws rather than only its old per-pair `accepted` boolean, since
    the pooled gate needs the raw per-item populations, not per-pair summaries): if given, each
    pair's own raw draw arrays (the ~6.5s/pair cost; every other step -- reading tensors,
    quantizing, deriving triples -- is ~0.1s/pair and simply re-run) are appended as one JSON line
    per pair to this path as soon as computed, and on entry, any pair already recorded there has
    its SAVED raw draws reused instead of re-running `_b3_collect_pair_raw_draws`. A resumed run
    therefore re-pays only the cheap per-pair steps for already-checkpointed pairs.
    """
    adapter_dir = Path(adapter_dir)
    meta = load_adapter_meta(adapter_dir)
    checkpoint_dir = _resolve_base_checkpoint_dir(adapter_dir)
    pipeline = _load_spike()
    cfg = pipeline.load_config(checkpoint_dir / "config.json")
    tensors = pipeline._open_checkpoint_tensors(checkpoint_dir)

    base_cfg1 = sf.read_section_bytes(str(base_sslm_path), sf.SectionType.CONFIG)
    if base_cfg1 is None:
        raise ValueError(f"{base_sslm_path}: no Config section -- not a valid base artifact")
    base_hash = sf.raw_integrity_hash(str(base_sslm_path))

    weights_tensors: Dict[str, np.ndarray] = {}
    dfs1_tensors: Dict[str, np.ndarray] = {}
    ufs1_tensors: Dict[str, np.ndarray] = {}
    round_trip: Dict[str, dict] = {}
    pair_diagnostics = []
    pair_draws_for_pooling = []  # [(name, raw_draws_dict), ...] -- T-2065's own pooled-gate input
    domain_trip = False
    calibration_seed = 0xC0FFEE  # the SAME synthetic-Gaussian single-draw convention T_honest
                                  # already uses throughout this project (t2018_offline_red.cpp's
                                  # BuildAdapter, t2029_b3_execute.py) -- a real-text corpus for T
                                  # remains the SAME already-tracked residual named since B3's
                                  # first execution (this build does not newly open that gap).

    saved_draws: Dict[str, dict] = {}
    checkpoint_file = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            with open(checkpoint_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    saved_draws[rec["name"]] = {k: np.asarray(v, dtype=np.float64)
                                                for k, v in rec["draws"].items()}
            if verbose and saved_draws:
                print(f"resuming: {len(saved_draws)} pairs' raw draws already checkpointed at {checkpoint_path}")
        checkpoint_file = open(checkpoint_path, "a", encoding="utf-8")

    for layer in range(cfg.num_hidden_layers):
        for proj in PEFT_ADAPTABLE_PROJECTIONS:
            if proj not in meta.target_modules:
                continue
            name = f"layer{layer}.{proj}"
            w_f = read_base_projection_weight(tensors, layer, proj)
            a_f, b_scaled = read_peft_lora_pair(adapter_dir, layer, proj, meta)
            d_out, d_in = w_f.shape
            r = a_f.shape[0]
            if a_f.shape != (r, d_in) or b_scaled.shape != (d_out, r):
                raise AdapterRejected(
                    f"{name}: A/B shape {a_f.shape}/{b_scaled.shape} does not match the base "
                    f"weight's own ({r},{d_in})/({d_out},{r})")

            Wc, w_scales = pipeline.quantize_weight_per_channel(w_f, output_axis=0)
            w = np.asarray(w_scales, dtype=np.float64)
            S = float(np.max(w))
            Ac, alpha_scales = pipeline.quantize_weight_per_channel(a_f, output_axis=0)
            alpha = np.asarray(alpha_scales, dtype=np.float64)
            Bc, beta_scales = pipeline.quantize_weight_per_channel(b_scaled, output_axis=0)
            beta = np.asarray(beta_scales, dtype=np.float64)

            g = np.random.default_rng(calibration_seed)
            xf = g.standard_normal(d_in)
            xmax = float(np.max(np.abs(xf)))
            X = xmax / 127.0 if xmax > 0.0 else 1.0
            xc = np.clip(np.round(xf / X), -127, 127).astype(np.int64)
            u_acc = Ac.astype(np.int64) @ xc
            T_value = compute_t(list(alpha), list(np.abs(u_acc)))

            delta_triples, _ = derive_delta_fold_triples(S, list(beta), T_value)
            u_triples, _ = derive_u_fold_triples(list(alpha), T_value)
            pair_domain_trip = any(t is None for t in delta_triples) or any(t is None for t in u_triples)
            domain_trip = domain_trip or pair_domain_trip

            if not pair_domain_trip:
                if name in saved_draws:
                    raw = saved_draws[name]
                    if "delta_norm_sq" not in raw:
                        # T-2213 fix round (D-SLM3787 finding S1): a checkpoint written before
                        # T-2213 (D-SLM3783) has no `delta_norm_sq` key at all -- `raw.get(...,
                        # 0.0)` at this function's own pooling call previously turned that
                        # absence into a silent, plausible-looking `delta_norm=0` for the resumed
                        # pair. Recompute it directly from this pair's own current A/B rather than
                        # refusing the resume: it is the cheap ~0.1s/pair product
                        # `_b3_collect_pair_raw_draws` already derives unconditionally
                        # (`b_scaled @ a_f`), never the ~6.5s/pair draw cost the checkpoint exists
                        # to avoid re-paying, so recomputing it here loses none of the resume's
                        # own performance benefit.
                        raw = dict(raw)
                        raw["delta_norm_sq"] = np.sum((b_scaled @ a_f) ** 2)
                else:
                    raw = _b3_collect_pair_raw_draws(w_f, a_f, b_scaled, Wc, w, S, Ac, alpha, Bc, beta, T_value)
                    if checkpoint_file is not None:
                        checkpoint_file.write(json.dumps(
                            {"name": name, "draws": {k: v.tolist() for k, v in raw.items()}}) + "\n")
                        checkpoint_file.flush()
                pair_draws_for_pooling.append((name, raw))

            pair_diagnostics.append({"layer": layer, "proj": proj, "domain_trip": pair_domain_trip})
            if verbose:
                print(f"  {name}: r={r} d_in={d_in} d_out={d_out} domain_trip={pair_domain_trip}")

            if pair_domain_trip:
                continue  # a None triple is not serializable at all -- structurally excluded.
                          # T-2213: B3 no longer has a pooled quality gate, so `domain_trip` (this
                          # branch) is now the ONLY thing that excludes a pair's own content from
                          # what CAN be assembled -- §7's "cannot even represent the ratio"
                          # (structural) case. Whether the assembled artifact is actually WRITTEN
                          # to --out is still dispatch_conversion_outcome's own call (main(),
                          # below), fed by saturation_elevated/domain_trip; margin_exceeded is
                          # unconditionally False now (below) and never excludes anything here.

            weights_tensors[f"{name}.lora_A"] = Ac
            weights_tensors[f"{name}.lora_B"] = Bc
            dfs1_tensors[name] = np.asarray([[t.identity, t.mult, t.exponent] for t in delta_triples], dtype=np.int32)
            ufs1_tensors[name] = np.asarray([[t.identity, t.mult, t.exponent] for t in u_triples], dtype=np.int32)
            round_trip[name] = {
                "delta": [(t.identity, t.mult, t.exponent) for t in delta_triples],
                "u": [(t.identity, t.mult, t.exponent) for t in u_triples],
                "a_codes": Ac.tolist(), "b_codes": Bc.tolist(),
            }

    if checkpoint_file is not None:
        checkpoint_file.close()

    # T-2213 (D-SLM3783): B3's pooled ACCEPT/REJECT gate is retired -- `run_b3_pooled_report`
    # computes a REPORT (per-pair diagnostics, the candidate's pooled delta_norm, an optional
    # magnitude_warning), never a verdict, called ONCE over every pair's own raw draws.
    # `margin_exceeded` is unconditionally False: nothing B3 computes can refuse to write an
    # artifact any more.
    pooled = None
    margin_exceeded = False
    if pair_draws_for_pooling:
        pooled = run_b3_pooled_report(pair_draws_for_pooling, reference_delta_norm=reference_delta_norm,
                                      verbose=verbose)
        if verbose:
            n_flagged = sum(1 for d in pooled["per_pair_diagnostics"] if d["flagged"])
            print(f"  POOLED REPORT: n_pairs={pooled['n_pairs']} n_pilot_pooled={pooled['n_pilot_pooled']} "
                 f"n_val_pooled={pooled['n_val_pooled']} delta_norm={pooled['delta_norm']:.6e} "
                 f"({n_flagged} pair(s) flagged for review)")
            print(f"  {_B3_POOLED_GATE_STATUS_NOTICE}")

    verdict = {"domain_trip": domain_trip, "margin_exceeded": margin_exceeded,
              "saturation_elevated": False, "pairs": pair_diagnostics, "pooled": pooled}

    sections = None
    if not domain_trip:
        adp1 = write_adp1(rank=meta.rank, target_modules=meta.target_modules,
                          base_artifact_hash=base_hash, lora_alpha=meta.lora_alpha,
                          use_rslora=meta.use_rslora,
                          source_adapter_name=source_adapter_name or adapter_dir.name)
        sections = [
            sf.Section(sf.SectionType.CONFIG, base_cfg1),
            sf.Section(sf.SectionType.SIGMOID_LUT, smw.write_sil1()),
            sf.Section(sf.SectionType.PROVENANCE, adp1),
            sf.Section(sf.SectionType.WEIGHTS, smw.write_tensor_manifest(smw.WGT1, np.int8, weights_tensors)),
            sf.Section(sf.SectionType.DELTA_FOLD_SCALES, smw.write_tensor_manifest(smw.DFS1, np.int32, dfs1_tensors)),
            sf.Section(sf.SectionType.U_FOLD_SCALES, smw.write_tensor_manifest(smw.UFS1, np.int32, ufs1_tensors)),
        ]
    return sections, verdict, round_trip


def build_merged_checkpoint(adapter_dir, out_dir, *, verbose: bool = True) -> Path:
    """Design §24.3 D-SLM3164/D-SLM3165: materializes a patched checkpoint directory -- every
    tensor copied through from the real base checkpoint EXCEPT the adapter's own adapted
    projections, which are replaced by `W_merged_float = W_base_float + scaling*(B_float@A_float)`
    (the SAME formula and scaling `SuperSLM_Plan.md` §11(b)(ii) already pins for the
    runtime-additive delta). `pipeline.load_model`/`convert_model.build_sections`/
    `sslm_convert_validate.validate_model`/`sslm_format.write_artifact`/
    `sslm_convert_manifest.verify_and_merge` are then run completely UNCHANGED against this
    directory (D-SLM3164's own "option (b)": a materialized patched checkpoint, chosen over a
    tensor-source-protocol wrapper class because it needs zero changes to `pipeline.py`'s own
    `load_model` signature).

    Every tensor is written as float32 (safetensors has no numpy bf16 dtype) -- LOSSLESS for the
    untouched tensors: `_SafeTensors`'s own comment states bf16->float32 widening is "a
    reinterpretation... no rounding happens at all," so re-encoding an EXACTLY-bf16-widened
    float64 value back to float32 recovers the identical bit pattern `_SafeTensors.tensor()` would
    have produced reading the original bf16 file.
    """
    from safetensors.numpy import save_file
    adapter_dir = Path(adapter_dir)
    out_dir = Path(out_dir)
    pipeline = _load_spike()
    meta = load_adapter_meta(adapter_dir)
    checkpoint_dir = _resolve_base_checkpoint_dir(adapter_dir)
    cfg = pipeline.load_config(checkpoint_dir / "config.json")
    tensors = pipeline._open_checkpoint_tensors(checkpoint_dir)
    names = pipeline._upstream_names(cfg)  # upstream -> ours

    adapted_upstream = {}
    for layer in range(cfg.num_hidden_layers):
        for proj in PEFT_ADAPTABLE_PROJECTIONS:
            if proj in meta.target_modules:
                adapted_upstream[f"model.layers.{layer}.{_PROJECTION_FULL_PATH[proj]}.weight"] = (layer, proj)

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_tensors = {}
    n_merged = 0
    for upstream in sorted(names):
        values = tensors.tensor(upstream)
        pair = adapted_upstream.get(upstream)
        if pair is not None:
            layer, proj = pair
            a_f, b_scaled = read_peft_lora_pair(adapter_dir, layer, proj, meta)
            values = values + (b_scaled @ a_f)
            n_merged += 1
        merged_tensors[upstream] = np.ascontiguousarray(values, dtype=np.float32)

    save_file(merged_tensors, str(out_dir / "model.safetensors"))
    if verbose:
        print(f"merged checkpoint: {n_merged} adapted tensors merged (scaling*(B@A) added), "
             f"{len(merged_tensors) - n_merged} tensors copied through unchanged -> "
             f"{out_dir / 'model.safetensors'}")

    for item in checkpoint_dir.iterdir():
        if item.name in ("model.safetensors", "model.safetensors.index.json"):
            continue
        if item.is_file():
            shutil.copy2(item, out_dir / item.name)
    return out_dir


def run_merge_quantize_conversion(adapter_dir, out_path, *, verifier=None, manifest_out=None,
                                  skip_verify: bool = False, verbose: bool = True,
                                  merged_dir=None) -> dict:
    """Design §24.3: `--fallback=merge`'s own output. Materializes the merged checkpoint
    (`build_merged_checkpoint`), then runs `pipeline.load_model`/`convert_model.build_sections`/
    `sslm_convert_validate.validate_model`/`sslm_format.write_artifact`/
    `sslm_convert_manifest.verify_and_merge` COMPLETELY UNCHANGED -- the same two-phase checked
    transaction any base artifact passes (D-SLM3168), never an adapter-specific relaxation.

    `merged_dir` (T-2046, added after a real `pipeline.load_model` calibration run was killed
    mid-flight by an external process-management event, not a crash, 2026-08-14): if given, an
    ALREADY-materialized merged checkpoint directory (`build_merged_checkpoint`'s own prior
    output) is used directly and `tempfile.TemporaryDirectory`'s own auto-cleanup is skipped --
    lets a retry reuse a merge step that already completed and survived the kill (a hard process
    termination does not run a `with`-block's `__exit__`, so the temp directory is not deleted),
    rather than re-paying that cost. Default (`None`): materialize fresh into a real temp
    directory, cleaned up on exit, exactly as before this parameter existed.
    """
    import convert_model as cm
    import sslm_convert_manifest as scm
    import sslm_convert_validate as scv

    pipeline = _load_spike()
    adapter_dir = Path(adapter_dir)

    def _run(merged_dir_path):
        if verbose:
            print("running pipeline.load_model against the merged checkpoint (fresh activation "
                 "calibration over the pinned corpus -- this is the real cost; T-1953 measured "
                 "~57min for a full 600-record calibration)...")
        model = pipeline.load_model(merged_dir_path)

        scv.validate_model(model, fold_ops_tensor=cm._fold_ops_tensor, ctx_fold_tensor=cm._ctx_fold_tensor,
                           unicode_major=scv.PINNED_UNICODE_VERSION[0],
                           unicode_minor=scv.PINNED_UNICODE_VERSION[1],
                           unicode_patch=scv.PINNED_UNICODE_VERSION[2])
        sections, fold_approximation_error = cm.build_sections(model)
        fingerprint = sf.write_artifact(str(out_path), sections)
        if verbose:
            print(f"wrote {out_path}\nfingerprint {fingerprint}\nsections {len(sections)}: " +
                 ", ".join(str(s.type) for s in sections))

        if skip_verify:
            return {"fingerprint": fingerprint, "verified": False}

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        verifier_cmd = [verifier] if verifier else None
        manifest = scm.verify_and_merge(repo_root, str(out_path), str(merged_dir_path),
                                        verifier_cmd=verifier_cmd, manifest_out_path=manifest_out,
                                        model=model, fold_approximation_error=fold_approximation_error)
        if verbose:
            print("verified: independent loader accepted the merge+quantize artifact")
        return {"fingerprint": fingerprint, "verified": True, "manifest": manifest}

    if merged_dir is not None:
        return _run(Path(merged_dir))
    with tempfile.TemporaryDirectory(prefix="sslm_merge_") as tmp:
        built_dir = build_merged_checkpoint(adapter_dir, Path(tmp) / "merged", verbose=verbose)
        return _run(built_dir)


def _print_pooled_b3_report(pooled, *, file=None):
    """T-2213 fix round (D-SLM3787 finding S4): the accept path and the reject path both print the
    same pooled B3 report when one exists (`pooled is not None`) -- factored out so the reject
    path (a partial domain trip) is not left silently printing nothing about the pairs that did
    not trip, which is what the deleted print block's own false justification comment claimed
    could not happen."""
    if pooled is None:
        return
    file = sys.stdout if file is None else file
    print(f"  pooled B3 report: n_pairs={pooled['n_pairs']} delta_norm={pooled['delta_norm']:.6e}",
         file=file)
    print(f"  {_B3_POOLED_GATE_STATUS_NOTICE}", file=file)
    if pooled["magnitude_warning"] is not None:
        print(f"  MAGNITUDE WARNING (unresolved, not a rejection): "
             f"{pooled['magnitude_warning']['reason']}", file=file)
    flagged = [d["name"] for d in pooled["per_pair_diagnostics"] if d["flagged"]]
    if flagged:
        print(f"  {len(flagged)} pair(s) flagged for review (diagnostic only, never gates): "
             f"{flagged}", file=file)
    else:
        print("  0 pair(s) flagged for review -- an empty list is not evidence this "
             "adapter is sound; see the notice above.", file=file)


def main():
    """Design §24.2 D-SLM3163's own CLI/dispatch contract."""
    import argparse
    ap = argparse.ArgumentParser(
        description="Convert a PEFT LoRA adapter into an .sslm artifact -- runtime-additive by "
                    "default, merge+quantize on --fallback=merge (design §24.2/§24.3).")
    ap.add_argument("--adapter", required=True, help="PEFT adapter directory (adapter_config.json "
                    "+ adapter_model.safetensors)")
    ap.add_argument("--base", required=True, help="the already-converted base .sslm artifact this "
                    "adapter binds to (CFG1/hash source, design §24.2 D-SLM3161)")
    ap.add_argument("--out", required=True, help="output .sslm path")
    ap.add_argument("--fallback", choices=["none", "merge"], default="none",
                    help="on a rejection: 'none' (default) emits no artifact; 'merge' emits a "
                         "merge+quantize artifact instead (design §7, D-SLM3095)")
    ap.add_argument("--verifier", default=None,
                    help="path to the compiled sslm_verify binary (default: searched under build/)")
    ap.add_argument("--manifest-out", default=None,
                    help="path for the combined proof manifest (default: <out>.manifest.json)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip invoking the independent C++ verifier (debugging only)")
    ap.add_argument("--reference-delta-norm", type=float, default=None,
                    help="a reference adapter's own pooled composed LoRA delta norm (T-2213, "
                         "D-SLM3783), for the B3 magnitude sanity check -- optional; when omitted "
                         "(the default) no reference is configured and the check is reported but "
                         "never fires, never blocking artifact emission either way. This is a raw "
                         "Frobenius norm, not normalized against the geometry it was measured on "
                         "-- a value taken from a different base checkpoint, rank, or "
                         "target_modules set produces a meaningless ratio with no error (D-SLM3787 "
                         "finding O4). Whoever records a reference value should record alongside "
                         "it which base-artifact hash and rank it was computed from.")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter)
    base_path = Path(args.base)
    out_path = Path(args.out)
    fallback_flag_present = (args.fallback == "merge")

    if out_path.exists():
        out_path.unlink()  # never a stale artifact from a prior run masking this run's own outcome

    print(f"converting adapter {adapter_dir} against base {base_path}")
    sections, verdict, _round_trip = build_runtime_additive_sections(
        adapter_dir, base_path, reference_delta_norm=args.reference_delta_norm)

    branch, outcome = dispatch_conversion_outcome(
        domain_trip=verdict["domain_trip"], margin_exceeded=verdict["margin_exceeded"],
        saturation_elevated=verdict["saturation_elevated"], fallback_flag_present=fallback_flag_present)

    if outcome == ArtifactOutcome.RUNTIME_ADDITIVE:
        fingerprint = sf.write_artifact(str(out_path), sections)
        print(f"wrote {out_path}\nfingerprint {fingerprint}\nsections {len(sections)}: " +
             ", ".join(str(s.type) for s in sections))
        _print_pooled_b3_report(verdict["pooled"])
        if not args.skip_verify:
            import sslm_convert_manifest as scm
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            verifier_cmd = [args.verifier] if args.verifier else None
            scm.verify_and_merge(repo_root, str(out_path), str(adapter_dir), verifier_cmd=verifier_cmd,
                                 manifest_out_path=args.manifest_out)
            print("verified: independent loader accepted the artifact")
        return 0

    # T-2213 fix round (D-SLM3787 finding S4): B3's pooled report can no longer produce
    # `margin_exceeded=True` -- this branch is reached only via `domain_trip` or
    # `saturation_elevated`. `domain_trip` is a per-pair boolean, OR-accumulated across
    # `build_runtime_additive_sections`'s own loop over every adapted pair -- a PARTIAL domain
    # trip (some pairs trip, others do not) still leaves `verdict["pooled"]` a full report over
    # every pair that did not trip; only a domain trip on EVERY pair leaves it `None`. The
    # previous comment here claimed a domain trip always means no pair was pooled, which is true
    # only when every pair trips. `saturation_elevated` is B6's own named residual, hardcoded
    # `False` and never computed by this module, so it is not a live route to this branch today.
    print(f"REJECTED: {branch.name} -- no runtime-additive artifact emitted (domain_trip="
         f"{verdict['domain_trip']} margin_exceeded={verdict['margin_exceeded']} "
         f"saturation_elevated={verdict['saturation_elevated']})", file=sys.stderr)
    # A partial domain trip still leaves a pooled report over the pairs that did not trip -- print
    # it here so a consumer refused an artifact is not shown nothing about the pairs that were
    # fine.
    _print_pooled_b3_report(verdict["pooled"], file=sys.stderr)

    if outcome == ArtifactOutcome.NO_ARTIFACT_EMITTED:
        if out_path.exists():
            out_path.unlink()
        print("no artifact written (pass --fallback=merge to emit a merge+quantize artifact instead)",
             file=sys.stderr)
        return 1

    assert outcome == ArtifactOutcome.MERGE_QUANTIZE_EMITTED
    print(f"--fallback=merge present: falling back to merge+quantize (diagnostic: {branch.name})")
    result = run_merge_quantize_conversion(adapter_dir, out_path, verifier=args.verifier,
                                           manifest_out=args.manifest_out, skip_verify=args.skip_verify)
    print(f"wrote merge+quantize artifact {out_path}\nfingerprint {result['fingerprint']}\n"
         f"diagnostic that triggered the fallback: {branch.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
