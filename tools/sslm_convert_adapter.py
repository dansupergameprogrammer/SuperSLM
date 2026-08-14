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

_SPIKE_ROOT = r"D:\Wizard\Tools"


def _load_spike():
    """Lazy cross-tree import, matching `convert_model.py`'s own `_load_spike` -- keeps this
    module importable (and its 22 existing B0 unit tests fast/dependency-free) on a bare
    checkout; only the writer/CLI functions below ever call this."""
    if _SPIKE_ROOT not in sys.path:
        sys.path.insert(0, _SPIKE_ROOT)
    from superslm_spike import pipeline  # noqa: E402
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
    `tools/t2029_b3_execute.py`'s own `load_adapter()`."""
    from safetensors import safe_open  # lazy: not needed by the 22 existing B0 unit tests
    full_path = _PROJECTION_FULL_PATH[proj_short]
    key_a = f"base_model.model.model.layers.{layer}.{full_path}.lora_A.weight"
    key_b = f"base_model.model.model.layers.{layer}.{full_path}.lora_B.weight"
    with safe_open(Path(adapter_dir) / "adapter_model.safetensors", framework="numpy") as f:
        a_f = f.get_tensor(key_a).astype(np.float64)
        b_f = f.get_tensor(key_b).astype(np.float64)
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


def run_b3_bootstrap_check(w_f, a_f, b_scaled, Wc, w, S, Ac, alpha, Bc, beta, T_honest,
                           pilot_n: int = _B3_PILOT_N, seed_base: int = 0x1000,
                           validation_seed_base: int = 0x9000) -> dict:
    """One (layer, proj) pair's own PILOT partition calibrates the four Deltas (composed
    mean/tail, effect mean/tail); the SAME pair's own VALIDATION partition is graded against its
    own just-frozen Deltas. Returns a dict with `accepted` (bool, feeding `margin_exceeded`) and
    the measured figures.

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
    w_prime = w_f + b_scaled @ a_f
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
    c_pilot_stat = _b3_stat(c_pilot)
    e_pilot_stat = _b3_stat(e_pilot)
    delta_composed_mean = _B3_SAFETY_INFLATION * (c_pilot_stat["mean"] + _B3_Z_95_ONE_SIDED * c_pilot_stat["se"])
    delta_composed_tail = _B3_SAFETY_INFLATION * c_pilot_stat["p95"]
    delta_effect_mean = _B3_SAFETY_INFLATION * (e_pilot_stat["mean"] + _B3_Z_95_ONE_SIDED * e_pilot_stat["se"])
    delta_effect_tail = _B3_SAFETY_INFLATION * e_pilot_stat["p95"]

    _c_pilot2, c_val, _e_pilot2, e_val = collect(pilot_n, validation_seed_base)
    c_val_stat = _b3_stat(c_val)
    e_val_stat = _b3_stat(e_val)

    composed_mean_accepts = c_val_stat["upper_ci"] < delta_composed_mean
    composed_tail_accepts = c_val_stat["p95"] < delta_composed_tail
    effect_mean_accepts = e_val_stat["upper_ci"] < delta_effect_mean
    effect_tail_accepts = e_val_stat["p95"] < delta_effect_tail
    accepted = composed_mean_accepts and composed_tail_accepts and effect_mean_accepts and effect_tail_accepts

    return {"accepted": accepted,
           "delta_composed_mean": delta_composed_mean, "delta_composed_tail": delta_composed_tail,
           "delta_effect_mean": delta_effect_mean, "delta_effect_tail": delta_effect_tail,
           "validation_composed_upper_ci": c_val_stat["upper_ci"], "validation_composed_p95": c_val_stat["p95"],
           "validation_effect_upper_ci": e_val_stat["upper_ci"], "validation_effect_p95": e_val_stat["p95"]}


def build_runtime_additive_sections(adapter_dir, base_sslm_path, *,
                                    source_adapter_name: Optional[str] = None, verbose: bool = True):
    """Design §24.2: assembles the six pinned sections (Config, SigmoidLut, Provenance/ADP1,
    Weights/WGT1, DeltaFoldScales/DFS1, UFoldScales/UFS1) for EVERY adapted (layer, projection)
    pair the adapter's own `target_modules` declares, across every layer the bound base checkpoint
    names. Runs, per pair: the real domain-trip check (this module's own B0 derivation) and the
    real B3 bootstrap self-calibration (above) -- `saturation_elevated` is the named residual
    stated in this file's own §24 banner comment, never computed here.

    `sections` is `None` only when `domain_trip` fires somewhere (a `None` triple is not
    serializable at all -- structurally excluded, design §6 item 2). A pair that merely fails B3
    (contributes to `margin_exceeded`) is still derivable and still assembled: §7 distinguishes
    "cannot even represent the ratio" (structural) from "represents it, but the composed quality
    is not good enough" (a QUALITY gate on an otherwise well-formed artifact) -- whether an
    assembled-but-quality-failing artifact is actually WRITTEN to `--out` is
    `dispatch_conversion_outcome`'s own call (`main()`, below), never this function's; this
    function's only job is "can it be built," never "should it ship" (design §7/D-SLM2864/
    D-SLM3095's own "never a partial or degraded runtime-additive artifact" governs the LATTER
    question, at the CLI's dispatch point, not by silently omitting content here).

    Returns `(sections_or_None, verdict, round_trip)`. `verdict` feeds
    `dispatch_conversion_outcome` directly (`domain_trip`/`margin_exceeded`/`saturation_elevated`)
    and also carries every pair's own diagnostic. `round_trip` is this build's own record of what
    was derived and written, per `"layer{L}.{proj}"` key -- what the round-trip proof (T-2046
    handoff) compares the REAL C++ loader's own output against, bit-for-bit.
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
    domain_trip = False
    margin_exceeded = False
    calibration_seed = 0xC0FFEE  # the SAME synthetic-Gaussian single-draw convention T_honest
                                  # already uses throughout this project (t2018_offline_red.cpp's
                                  # BuildAdapter, t2029_b3_execute.py) -- a real-text corpus for T
                                  # remains the SAME already-tracked residual named since B3's
                                  # first execution (this build does not newly open that gap).

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

            b3 = None
            if not pair_domain_trip:
                b3 = run_b3_bootstrap_check(w_f, a_f, b_scaled, Wc, w, S, Ac, alpha, Bc, beta, T_value)
                if not b3["accepted"]:
                    margin_exceeded = True

            pair_diagnostics.append({
                "layer": layer, "proj": proj, "domain_trip": pair_domain_trip,
                "b3_accepted": (b3["accepted"] if b3 else None),
            })
            if verbose:
                print(f"  {name}: r={r} d_in={d_in} d_out={d_out} domain_trip={pair_domain_trip} "
                     f"b3_accepted={(b3['accepted'] if b3 else 'N/A (domain-tripped)')}")

            if pair_domain_trip:
                continue  # a None triple is not serializable at all -- structurally excluded.
                          # A pair that merely fails B3 (margin_exceeded) IS still derivable and
                          # IS still included below: §7 distinguishes "cannot even represent the
                          # ratio" (domain-rejection-trip, structural) from "represents it, but the
                          # composed quality is not good enough" (margin/saturation, a QUALITY
                          # gate applied to an otherwise-well-formed artifact) -- only the former
                          # excludes a pair's own content from what CAN be assembled. Whether the
                          # assembled artifact is actually WRITTEN to --out is dispatch_conversion_
                          # outcome's own call (main(), below), fed by margin_exceeded/
                          # saturation_elevated exactly as domain_trip -- keeping the STRUCTURAL
                          # question (can this be built) and the QUALITY question (should it ship)
                          # in the two places design §7 itself keeps them.

            weights_tensors[f"{name}.lora_A"] = Ac
            weights_tensors[f"{name}.lora_B"] = Bc
            dfs1_tensors[name] = np.asarray([[t.identity, t.mult, t.exponent] for t in delta_triples], dtype=np.int32)
            ufs1_tensors[name] = np.asarray([[t.identity, t.mult, t.exponent] for t in u_triples], dtype=np.int32)
            round_trip[name] = {
                "delta": [(t.identity, t.mult, t.exponent) for t in delta_triples],
                "u": [(t.identity, t.mult, t.exponent) for t in u_triples],
                "a_codes": Ac.tolist(), "b_codes": Bc.tolist(),
            }

    verdict = {"domain_trip": domain_trip, "margin_exceeded": margin_exceeded,
              "saturation_elevated": False, "pairs": pair_diagnostics}

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
                                  skip_verify: bool = False, verbose: bool = True) -> dict:
    """Design §24.3: `--fallback=merge`'s own output. Materializes the merged checkpoint
    (`build_merged_checkpoint`), then runs `pipeline.load_model`/`convert_model.build_sections`/
    `sslm_convert_validate.validate_model`/`sslm_format.write_artifact`/
    `sslm_convert_manifest.verify_and_merge` COMPLETELY UNCHANGED -- the same two-phase checked
    transaction any base artifact passes (D-SLM3168), never an adapter-specific relaxation.
    """
    import convert_model as cm
    import sslm_convert_manifest as scm
    import sslm_convert_validate as scv

    pipeline = _load_spike()
    adapter_dir = Path(adapter_dir)
    with tempfile.TemporaryDirectory(prefix="sslm_merge_") as tmp:
        merged_dir = build_merged_checkpoint(adapter_dir, Path(tmp) / "merged", verbose=verbose)
        if verbose:
            print("running pipeline.load_model against the merged checkpoint (fresh activation "
                 "calibration over the pinned corpus -- this is the real cost; T-1953 measured "
                 "~57min for a full 600-record calibration)...")
        model = pipeline.load_model(merged_dir)

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
        manifest = scm.verify_and_merge(repo_root, str(out_path), str(merged_dir),
                                        verifier_cmd=verifier_cmd, manifest_out_path=manifest_out,
                                        model=model, fold_approximation_error=fold_approximation_error)
        if verbose:
            print("verified: independent loader accepted the merge+quantize artifact")
        return {"fingerprint": fingerprint, "verified": True, "manifest": manifest}


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
    args = ap.parse_args()

    adapter_dir = Path(args.adapter)
    base_path = Path(args.base)
    out_path = Path(args.out)
    fallback_flag_present = (args.fallback == "merge")

    if out_path.exists():
        out_path.unlink()  # never a stale artifact from a prior run masking this run's own outcome

    print(f"converting adapter {adapter_dir} against base {base_path}")
    sections, verdict, _round_trip = build_runtime_additive_sections(adapter_dir, base_path)

    branch, outcome = dispatch_conversion_outcome(
        domain_trip=verdict["domain_trip"], margin_exceeded=verdict["margin_exceeded"],
        saturation_elevated=verdict["saturation_elevated"], fallback_flag_present=fallback_flag_present)

    if outcome == ArtifactOutcome.RUNTIME_ADDITIVE:
        fingerprint = sf.write_artifact(str(out_path), sections)
        print(f"wrote {out_path}\nfingerprint {fingerprint}\nsections {len(sections)}: " +
             ", ".join(str(s.type) for s in sections))
        if not args.skip_verify:
            import sslm_convert_manifest as scm
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            verifier_cmd = [args.verifier] if args.verifier else None
            scm.verify_and_merge(repo_root, str(out_path), str(adapter_dir), verifier_cmd=verifier_cmd,
                                 manifest_out_path=args.manifest_out)
            print("verified: independent loader accepted the artifact")
        return 0

    print(f"REJECTED: {branch.name} -- no runtime-additive artifact emitted (domain_trip="
         f"{verdict['domain_trip']} margin_exceeded={verdict['margin_exceeded']} "
         f"saturation_elevated={verdict['saturation_elevated']})", file=sys.stderr)

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
