#!/usr/bin/env python3
"""Popper probe 5 -- guard-vitality attack on the T-1690 red-first proof.

The packet claims "19/19 red-first checks, each of seven composed steps
paired with a deliberate mutation confirmed to diverge". Every mutation in
that suite is one the suite's own author wrote and aimed at. This probe
brings its own independently-chosen mutations of the REAL composition (not
the WRONG_* helper functions the suite already knows about) and asks whether
the 19 checks go red. A mutation that survives is a defect class the suite
does not cover.
"""
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(r"D:\SuperSLM\.worktrees\t1690-third-witness\tools")
WORK = Path(r"D:\SuperSLM\.worktrees\popper-t1802\out\mutation_work")

MUTATIONS = [
    ("M1_attention_scale_removed",
     "    scale = 1.0 / (cfg.head_dim ** 0.5)",
     "    scale = 1.0"),
    ("M2_gqa_tile_instead_of_interleave",
     "    return torch.repeat_interleave(x, n_rep, dim=0)",
     "    return x.repeat(n_rep, 1, 1)"),
    ("M3_qkv_bias_dropped",
     "    q = (x @ wq.T + bq).view(cfg.num_attention_heads, cfg.head_dim)",
     "    q = (x @ wq.T).view(cfg.num_attention_heads, cfg.head_dim)"),
    ("M4_rotate_half_interleaved_convention",
     "    x1, x2 = x[..., : d // 2], x[..., d // 2 :]",
     "    x1, x2 = x[..., 0::2], x[..., 1::2]"),
    ("M5_rope_not_applied_to_k",
     "    k = apply_rope(k, cos, sin)",
     "    k = k"),
    ("M6_cache_read_excludes_own_position",
     "    k_all, v_all = cache.read(layer, pos + 1)",
     "    k_all, v_all = cache.read(layer, max(pos, 1))"),
    ("M7_layernorm_weights_swapped",
     '    normed = rms_norm(hidden, weights.get(layer, "input_layernorm.weight"), cfg.rms_norm_eps)',
     '    normed = rms_norm(hidden, weights.get(layer, "post_attention_layernorm.weight"), cfg.rms_norm_eps)'),
    ("M8_mlp_gate_and_up_swapped",
     "    gate = x @ w_gate.T\n    up = x @ w_up.T\n    return (silu(gate) * up) @ w_down.T",
     "    gate = x @ w_up.T\n    up = x @ w_gate.T\n    return (silu(gate) * up) @ w_down.T"),
    ("M9_softmax_no_fp32_upcast",
     "    weights_sm = torch.softmax(scores.to(torch.float32), dim=-1).to(scores.dtype)",
     "    weights_sm = torch.softmax(scores, dim=-1)"),
    ("M10_rms_norm_eps_inside_sqrt_of_sum_not_mean",
     "    variance = x.pow(2).mean(dim=-1, keepdim=True)\n    normed = x * torch.rsqrt(variance + eps)\n    return normed * weight",
     "    variance = x.pow(2).sum(dim=-1, keepdim=True)\n    normed = x * torch.rsqrt(variance + eps)\n    return normed * weight"),
]


def run_suite(work: Path) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(work / "test_independent_layer_reference.py")],
                       capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    shutil.copy(SRC / "independent_layer_reference.py", WORK)
    shutil.copy(SRC / "test_independent_layer_reference.py", WORK)
    original = (WORK / "independent_layer_reference.py").read_text(encoding="utf-8")

    rc, out = run_suite(WORK)
    print(f"baseline (unmutated): exit={rc}  {'ALL CHECKS PASSED' in out}")
    print()

    survivors = []
    for name, old, new in MUTATIONS:
        if old not in original:
            print(f"  {name:<45} SKIPPED -- anchor text not found")
            continue
        if original.count(old) != 1:
            print(f"  {name:<45} anchor appears {original.count(old)} times; mutating all")
        (WORK / "independent_layer_reference.py").write_text(original.replace(old, new), encoding="utf-8")
        rc, out = run_suite(WORK)
        failed = [l for l in out.splitlines() if l.startswith("[FAIL]")]
        killed = rc != 0
        status = "KILLED " if killed else "SURVIVED"
        print(f"  {name:<45} {status}  exit={rc}  failing checks={len(failed)}")
        for l in failed[:4]:
            print(f"        {l}")
        if not killed:
            survivors.append(name)
    (WORK / "independent_layer_reference.py").write_text(original, encoding="utf-8")

    print(f"\nsurvivors ({len(survivors)}/{len(MUTATIONS)}): {survivors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
