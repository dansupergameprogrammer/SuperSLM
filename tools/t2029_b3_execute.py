"""t2029_b3_execute.py -- B3's own first real execution (T-2021/T-2029, design Sec11 B3,
`Claude/Vitruvius/t1977-...-design-2026-08-13.md` Sec6 item 1/1a).

Runs the baked-adapter comparator (design Sec6 item 1, D-SLM3019/3085/3086) for the FIRST time
against a REAL PEFT LoRA adapter -- the shopkeeper adapter D-SLM2845 names
(`D:\\hf_cache\\superslm_artifacts\\qwen2.5-1.5b-shopkeeper-lora-v1\\final\\`), read directly
from its own `adapter_config.json`/`adapter_model.safetensors`, against the real base checkpoint
(`D:\\hf_cache\\hub\\models--Qwen--Qwen2.5-1.5B-Instruct\\...\\model.safetensors`). This is the
adapter's own bootstrap conversion (design Sec6 item 1: "the first adapter converted through this
pipeline serves as its own reference at bootstrap, since no other reference yet exists at B3's
first execution") -- it calibrates the four `Delta`s from its own PILOT partition and grades
itself on its own VALIDATION partition against the now-frozen `Delta`s, exactly as every future
adapter will be graded.

Scope of this first execution (stated explicitly, not left implicit): ONE representative
(layer, projection) pair -- layer 0, q_proj (in=out=1536, r=16, matching the real adapter's own
`adapter_config.json`: r=16, alpha=32, target_modules includes q_proj) -- not all seven
projections across all 28 layers. This is the minimal real, non-synthetic slice that proves the
comparator machinery end to end against real trained weights; extending to every adapted
(layer, projection) pair is a follow-on execution, not a change to the machinery itself. The
CALIBRATION CORPUS is synthetic (Gaussian-drawn per-token activations, matching this project's
own established convention for a representative RMSNorm-output distribution -- the SAME
convention `tests/t2018-slora-serial/t2018_offline_red.cpp`'s own `BuildAdapter`/`RunToken` use)
-- no real-text calibration corpus exists yet in this repository (SuperSLM_Plan.md Sec11's own
open obligation); this is named as an explicit, honest scope limit of this first execution, not
hidden.
"""

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
import sslm_convert_adapter as A  # noqa: E402

ADAPTER_DIR = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-shopkeeper-lora-v1\final")
BASE_MODEL_PATH = Path(
    r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306\model.safetensors"
)
LAYER = 0
PROJECTION = "self_attn.q_proj"

Z_95_ONE_SIDED = 1.645
SAFETY_INFLATION = 1.5
PILOT_N = 200
VALIDATION_SEED_BASE = 0x9000
PILOT_SEED_BASE = 0x1000


def load_adapter():
    with open(ADAPTER_DIR / "adapter_config.json") as fh:
        cfg = json.load(fh)
    assert cfg["init_lora_weights"] is True, "adapter must be additive-safe (init_lora_weights=True)"
    assert cfg["use_dora"] is False, "adapter must not be DoRA"
    assert cfg["bias"] == "none" and cfg["lora_bias"] is False, "adapter must carry no trained bias"
    assert cfg["fan_in_fan_out"] is False
    r = cfg["r"]
    scaling = cfg["lora_alpha"] / (math.sqrt(r) if cfg.get("use_rslora") else r)
    key_a = f"base_model.model.model.layers.{LAYER}.{PROJECTION}.lora_A.weight"
    key_b = f"base_model.model.model.layers.{LAYER}.{PROJECTION}.lora_B.weight"
    with safe_open(ADAPTER_DIR / "adapter_model.safetensors", framework="numpy") as f:
        a_f = f.get_tensor(key_a).astype(np.float64)  # [r, in]
        b_f = f.get_tensor(key_b).astype(np.float64)  # [out, r]
    b_scaled = b_f * scaling  # PEFT scaling folded into B (SuperSLM_Plan.md Sec11(ii))
    return r, a_f, b_scaled, cfg["target_modules"]


def load_base_weight():
    import torch  # noqa: PLC0415  (local import: only this function needs bf16 support)

    key = f"model.layers.{LAYER}.{PROJECTION}.weight"
    with safe_open(BASE_MODEL_PATH, framework="pt") as f:
        w = f.get_tensor(key)  # bf16, [out, in]
    return w.float().numpy().astype(np.float64)


def per_channel_scale(mat, axis_reduce):
    """max|.|/127 over `axis_reduce`, the standard int8 per-channel calibration this project's
    own converter text specifies (SuperSLM_Plan.md Sec11(iii): "quantizes A and B to int8 with
    their own static scales")."""
    mx = np.max(np.abs(mat), axis=axis_reduce)
    mx = np.where(mx <= 0.0, 1.0, mx)  # degenerate channel guard
    return mx / 127.0


def quantize(mat, scale, axis_broadcast):
    s = np.expand_dims(scale, axis_broadcast)
    q = np.round(mat / s)
    return np.clip(q, -127, 127).astype(np.int8)


class Fixture:
    pass


def build_fixture():
    r, a_f, b_scaled, target_modules = load_adapter()
    w_f = load_base_weight()  # [out, in]
    d_in = w_f.shape[1]
    d_out = w_f.shape[0]
    assert a_f.shape == (r, d_in), f"A shape {a_f.shape} != ({r},{d_in})"
    assert b_scaled.shape == (d_out, r), f"B shape {b_scaled.shape} != ({d_out},{r})"

    fx = Fixture()
    fx.r, fx.d_in, fx.d_out = r, d_in, d_out
    fx.target_modules = target_modules

    # Base: per-output-channel scale w[i] = max_j|W[i,j]|/127, S = max_i w[i] (design Sec2/Sec4).
    fx.w = per_channel_scale(w_f, axis_reduce=1)  # [out]
    fx.S = float(np.max(fx.w))
    fx.Wc = quantize(w_f, fx.w, axis_broadcast=1)  # [out, in] int8

    # Adapter: per-rank alpha[k] (A's own scale), per-output-channel beta[i] (B's own scale).
    fx.alpha = per_channel_scale(a_f, axis_reduce=1)  # [r]
    fx.Ac = quantize(a_f, fx.alpha, axis_broadcast=1)  # [r, in] int8
    fx.beta = per_channel_scale(b_scaled, axis_reduce=1)  # [out]
    fx.Bc = quantize(b_scaled, fx.beta, axis_broadcast=1)  # [out, r] int8

    fx.W_f, fx.A_f, fx.B_f = w_f, a_f, b_scaled

    # BAKED arm (design Sec6 item 1): W' = W + B.A, merged in float ONCE, quantized ONCE via the
    # base's own unmodified per-channel derivation applied to W'.
    w_prime = w_f + b_scaled @ a_f  # [out, in]
    fx.wp = per_channel_scale(w_prime, axis_reduce=1)
    fx.Sp = float(np.max(fx.wp))
    fx.Wpc = quantize(w_prime, fx.wp, axis_broadcast=1)
    fx.W_prime_f = w_prime

    # T calibration (design Sec4/D-SLM2916): T = max_k(alpha_k * max|u_acc[k]| / 127), measured
    # over a representative calibration draw -- one Gaussian token, matching BuildAdapter's own
    # construction (t2018_offline_red.cpp).
    g = np.random.default_rng(0xC0FFEE)
    xf = g.standard_normal(d_in)
    xmax = np.max(np.abs(xf))
    X = xmax / 127.0
    xc = np.clip(np.round(xf / X), -127, 127).astype(np.int8)
    u_acc = fx.Ac.astype(np.int64) @ xc.astype(np.int64)  # [r]
    fx.T_honest = float(np.max(fx.alpha * np.abs(u_acc) / 127.0))
    if fx.T_honest <= 0.0:
        fx.T_honest = 1.0
    return fx


def run_token(fx, seed, t_scale=1.0):
    g = np.random.default_rng(seed)
    xf = g.standard_normal(fx.d_in)
    xmax = np.max(np.abs(xf))
    X = xmax / 127.0
    xc = np.clip(np.round(xf / X), -127, 127).astype(np.int8)

    t_value = fx.T_honest * t_scale

    # u-fold.
    u_acc = fx.Ac.astype(np.int64) @ xc.astype(np.int64)
    u_i8 = np.zeros(fx.r, dtype=np.int64)
    for k in range(fx.r):
        rho_u = fx.alpha[k] / t_value
        triple = A.derive_amplifying_triple(rho_u)
        u_wide = round(A.realized_ratio(triple) * u_acc[k]) if triple is not None else u_acc[k]
        u_i8[k] = max(-127, min(127, u_wide))

    # base accumulator (real WSC1-style fold).
    acc = fx.Wc.astype(np.int64) @ xc.astype(np.int64)  # [out]
    base_triples = [A.derive_amplifying_triple(fx.w[i] / fx.S) for i in range(fx.d_out)]
    acc_wide = np.array(
        [round(A.realized_ratio(t) * acc[i]) if t is not None else acc[i] for i, t in enumerate(base_triples)],
        dtype=np.int64,
    )

    # delta.
    delta_raw = fx.Bc.astype(np.int64) @ u_i8
    rho_delta = t_value * fx.beta / fx.S
    delta_triples = [A.derive_amplifying_triple(r) for r in rho_delta]
    delta_wide = np.array(
        [
            round(A.realized_ratio(t) * delta_raw[i]) if t is not None else delta_raw[i]
            for i, t in enumerate(delta_triples)
        ],
        dtype=np.int64,
    )

    scale = X * fx.S
    scale_baked = X * fx.Sp

    runtime_composed = (acc_wide + delta_wide) * scale
    runtime_base = acc_wide * scale

    bacc = fx.Wpc.astype(np.int64) @ xc.astype(np.int64)
    baked_triples = [A.derive_amplifying_triple(fx.wp[i] / fx.Sp) for i in range(fx.d_out)]
    bacc_wide = np.array(
        [round(A.realized_ratio(t) * bacc[i]) if t is not None else bacc[i] for i, t in enumerate(baked_triples)],
        dtype=np.int64,
    )
    baked_composed = bacc_wide * scale_baked
    baked_base = bacc_wide * scale_baked  # baked has no separate "base" arm -- same weight quantized

    # composed float reference.
    yb = fx.W_f @ xf
    yd = fx.B_f @ (fx.A_f @ xf)
    ref = yb + yd

    composed_l2_runtime = math.sqrt(np.sum((runtime_composed - ref) ** 2)) / math.sqrt(np.sum(ref**2))
    composed_l2_baked = math.sqrt(np.sum((baked_composed - ref) ** 2)) / math.sqrt(np.sum(ref**2))

    effect_runtime = runtime_composed - runtime_base
    effect_baked = baked_composed - baked_base  # baked's "effect" is its own composed minus base-only (unadapted W)
    # baked's own "base" reference for effect isolation is the UNADAPTED base's own composed
    # output (design Sec6 item 1a: "the SAME base-only run" for both arms) -- recomputed here via
    # the base arm (acc_wide*scale is the runtime arm's own base; the baked arm's effect is its
    # own composed minus that SAME shared base-only value, per design's own "runtime_base[i] and
    # baked_base[i] are the same base-only run" text).
    shared_base = acc_wide * scale
    effect_baked = baked_composed - shared_base

    eff_den = math.sqrt(np.sum(yd**2))
    effect_l2_runtime = math.sqrt(np.sum((effect_runtime - yd) ** 2)) / eff_den
    effect_l2_baked = math.sqrt(np.sum((effect_baked - yd) ** 2)) / eff_den

    return composed_l2_runtime, composed_l2_baked, effect_l2_runtime, effect_l2_baked


def is_pilot_item(seed):
    z = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return (z % 10) < 2


def collect_gaps(fx, n_items, seed_base, t_scale=1.0):
    composed_pilot, composed_val, effect_pilot, effect_val = [], [], [], []
    for i in range(n_items):
        seed = (seed_base + i * 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF
        cl_rt, cl_bk, el_rt, el_bk = run_token(fx, seed, t_scale)
        composed_gap = cl_rt - cl_bk
        effect_gap = el_rt - el_bk
        if is_pilot_item(seed):
            composed_pilot.append(composed_gap)
            effect_pilot.append(effect_gap)
        else:
            composed_val.append(composed_gap)
            effect_val.append(effect_gap)
    return composed_pilot, composed_val, effect_pilot, effect_val


def stat(gaps, z=Z_95_ONE_SIDED):
    n = len(gaps)
    mean = statistics.fmean(gaps)
    sd = statistics.stdev(gaps) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    upper_ci = mean + z * se
    p95 = float(np.percentile(gaps, 95))
    sign_pos = sum(1 for g in gaps if g > 0) / n
    return {"n": n, "mean": mean, "se": se, "upper_ci": upper_ci, "p95": p95, "sign_frac_positive": sign_pos}


def main():
    print(f"B3 first real execution -- adapter: {ADAPTER_DIR}")
    print(f"base checkpoint: {BASE_MODEL_PATH}")
    print(f"layer={LAYER} projection={PROJECTION}")

    fx = build_fixture()
    print(f"r={fx.r} d_in={fx.d_in} d_out={fx.d_out} S={fx.S:.6e} Sp={fx.Sp:.6e} T_honest={fx.T_honest:.6e}")
    print(f"target_modules (real adapter): {fx.target_modules}")

    # WIRING-AND-DERIVATION precondition (R7, D-SLM3100): the reference must pass B0's gate.
    # Spot-checked on channel 0 -- the derivation is a pure function of (T, beta, S), already
    # exercised exhaustively by this build's own B0 test suite (tools/test_sslm_convert_adapter.py).
    rho0 = fx.T_honest * fx.beta[0] / fx.S
    triple0 = A.derive_amplifying_triple(rho0)
    gate0 = A.gate_check(triple0, rho0, fx.T_honest, fx.beta[0], fx.S) if triple0 else None
    print(f"WIRING-AND-DERIVATION precondition, channel 0: rho={rho0:.6f} gate_passes={gate0.gate_passes if gate0 else False}")

    print(f"\nCalibrating Delta on PILOT (n_target={PILOT_N})...")
    composed_pilot, _, effect_pilot, _ = collect_gaps(fx, PILOT_N, PILOT_SEED_BASE, t_scale=1.0)
    composed_pilot_stat = stat(composed_pilot)
    effect_pilot_stat = stat(effect_pilot)

    delta_composed_mean = SAFETY_INFLATION * (composed_pilot_stat["mean"] + Z_95_ONE_SIDED * composed_pilot_stat["se"])
    delta_composed_tail = SAFETY_INFLATION * composed_pilot_stat["p95"]
    delta_effect_mean = SAFETY_INFLATION * (effect_pilot_stat["mean"] + Z_95_ONE_SIDED * effect_pilot_stat["se"])
    delta_effect_tail = SAFETY_INFLATION * effect_pilot_stat["p95"]

    print(f"PILOT (n_composed={composed_pilot_stat['n']}, n_effect={effect_pilot_stat['n']}):")
    print(f"  composed: mean={composed_pilot_stat['mean']:.6f} se={composed_pilot_stat['se']:.6f} p95={composed_pilot_stat['p95']:.6f}")
    print(f"  effect:   mean={effect_pilot_stat['mean']:.6f} se={effect_pilot_stat['se']:.6f} p95={effect_pilot_stat['p95']:.6f}")
    print(f"FROZEN Delta: composed_mean={delta_composed_mean:.6f} composed_tail={delta_composed_tail:.6f} "
          f"effect_mean={delta_effect_mean:.6f} effect_tail={delta_effect_tail:.6f}")

    print(f"\nGrading the SAME reference adapter on VALIDATION (n_target={PILOT_N}) against its own frozen Delta...")
    _, composed_val, _, effect_val = collect_gaps(fx, PILOT_N, VALIDATION_SEED_BASE, t_scale=1.0)
    composed_val_stat = stat(composed_val)
    effect_val_stat = stat(effect_val)

    composed_mean_accepts = composed_val_stat["upper_ci"] < delta_composed_mean
    composed_tail_accepts = composed_val_stat["p95"] < delta_composed_tail
    effect_mean_accepts = effect_val_stat["upper_ci"] < delta_effect_mean
    effect_tail_accepts = effect_val_stat["p95"] < delta_effect_tail
    all_accept = composed_mean_accepts and composed_tail_accepts and effect_mean_accepts and effect_tail_accepts

    # UNRESOLVED-never-PASSED discipline (D-SLM2846): a conjunct whose validation upper_ci sits
    # inside its own achieved resolving power of Delta is reported as unresolved, not silently
    # rounded to accept, even when the raw comparison happens to clear the bar.
    def verdict_word(accepts, upper_ci, delta, resolving_power):
        if not accepts:
            return "REJECT"
        if abs(delta - upper_ci) < resolving_power:
            return "UNRESOLVED"
        return "ACCEPT"

    composed_mean_word = verdict_word(composed_mean_accepts, composed_val_stat["upper_ci"], delta_composed_mean,
                                       composed_val_stat["se"] * Z_95_ONE_SIDED)
    effect_mean_word = verdict_word(effect_mean_accepts, effect_val_stat["upper_ci"], delta_effect_mean,
                                     effect_val_stat["se"] * Z_95_ONE_SIDED)

    print(f"\nVALIDATION (n_composed={composed_val_stat['n']}, n_effect={effect_val_stat['n']}):")
    print(f"  composed: mean_gap={composed_val_stat['mean']:.6f} SE={composed_val_stat['se']:.6f} "
          f"upper_CI={composed_val_stat['upper_ci']:.6f} p95={composed_val_stat['p95']:.6f} "
          f"sign+={composed_val_stat['sign_frac_positive']:.2f}")
    print(f"    Delta_composed_mean={delta_composed_mean:.6f} -> mean conjunct: {composed_mean_word} "
          f"(accepts={composed_mean_accepts})")
    print(f"    Delta_composed_tail={delta_composed_tail:.6f} -> tail conjunct accepts={composed_tail_accepts}")
    print(f"  effect:   mean_gap={effect_val_stat['mean']:.6f} SE={effect_val_stat['se']:.6f} "
          f"upper_CI={effect_val_stat['upper_ci']:.6f} p95={effect_val_stat['p95']:.6f} "
          f"sign+={effect_val_stat['sign_frac_positive']:.2f}")
    print(f"    Delta_effect_mean={delta_effect_mean:.6f} -> mean conjunct: {effect_mean_word} "
          f"(accepts={effect_mean_accepts})")
    print(f"    Delta_effect_tail={delta_effect_tail:.6f} -> tail conjunct accepts={effect_tail_accepts}")

    overall = "ACCEPT" if all_accept else "REJECT"
    if all_accept and (composed_mean_word == "UNRESOLVED" or effect_mean_word == "UNRESOLVED"):
        overall = "UNRESOLVED"
    print(f"\nOVERALL VERDICT: {overall}")


if __name__ == "__main__":
    main()
