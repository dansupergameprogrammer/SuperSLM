# T-1409 Popper probe — product-code mutants.
#
# DISPOSABLE. Applies one defect at a time to the SHIPPED forward/matmul source,
# rebuilds the solver's own harness, and re-runs the solver's own Experiment C
# protocol. The question: does the construction's discrimination property
# (weight mutation moves decode output) still report 40 MOVES / 16 SILENT /
# 0 REFUSES when the product code under it is defective?
#
# Each mutant is reverted with `git checkout --` before the next is applied.
import subprocess, sys, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = os.path.join(ROOT, "src", "forward", "forward_sites.cpp")
MM = os.path.join(ROOT, "src", "matmul.cpp")

# (name, file, [(old, new), ...]) — every `old` must occur exactly once.
MUTANTS = [
    ("M1 GEMM weight stride transposed", MM, [
        ("out_acc[j] = DotRow(activations, weights + j * in_channels, in_channels);",
         "out_acc[j] = DotRow(activations, weights + j, in_channels);"),
    ]),
    ("M2 argmax tie-break last-wins", FS, [
        ("\t\tif (logits[i] > best_value) {", "\t\tif (logits[i] >= best_value) {"),
    ]),
    ("M3 RMSNorm denominator off-by-one", FS, [
        ("ISqrt(FloorDivI64(sumsq << (2 * kNormFracBits), static_cast<int64_t>(hidden_size)));",
         "ISqrt(FloorDivI64(sumsq << (2 * kNormFracBits), static_cast<int64_t>(hidden_size - 1)));"),
    ]),
    ("M4 gate/up weights swapped", FS, [
        ("lw.gate_weight, hidden_size,", "lw.POPPERTMP_weight, hidden_size,"),
        ("lw.up_weight, hidden_size,", "lw.gate_weight, hidden_size,"),
        ("lw.POPPERTMP_weight, hidden_size,", "lw.up_weight, hidden_size,"),
    ]),
    ("M5 mlp_residual reads seq, not the staged stream", FS, [
        ("ResidualReconcileSite(down_codes.data(), down_scale, attn_stream.data(),",
         "ResidualReconcileSite(down_codes.data(), down_scale, seq.hidden_codes,"),
        ("                           attn_stream_scale, hidden_size, lw.mlp_residual_site_constant,",
         "                           seq.hidden_scale, hidden_size, lw.mlp_residual_site_constant,"),
    ]),
    ("M8 o_proj uses v_weight", FS, [
        ("lw.o_weight, hidden_size, hidden_size,", "lw.v_weight, hidden_size, hidden_size,"),
    ]),
    ("M9 mlp_act ignores the up branch", FS, [
        ("MlpActSite(gate_codes.data(), gate_scale, up_codes.data(), up_scale, intermediate_size,",
         "MlpActSite(gate_codes.data(), gate_scale, gate_codes.data(), gate_scale, intermediate_size,"),
    ]),
    ("M6 attn residual connection dropped", FS, [
        ("ResidualReconcileSite(o_codes.data(), o_scale, seq.hidden_codes, seq.hidden_scale,",
         "ResidualReconcileSite(o_codes.data(), o_scale, o_codes.data(), o_scale,"),
    ]),
    ("M7 RMSNorm gain ignored", FS, [
        ("wide[i] = FloorDivI64(hi << (2 * kNormFracBits), root) * static_cast<int64_t>(g[i]);",
         "wide[i] = FloorDivI64(hi << (2 * kNormFracBits), root);"),
    ]),
]


def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, **kw)


def build():
    env = dict(os.environ)
    env["SOLVER_DEFINES"] = "/DSOLVER_TRACE_WIRED"
    r = subprocess.run("experiments\\build_experiment.bat", shell=True, cwd=ROOT,
                       capture_output=True, text=True, env=env)
    return r.returncode == 0, r.stdout + r.stderr


def revert():
    run("git checkout -- src/forward/forward_sites.cpp src/matmul.cpp")


def summarize(mutate_out):
    m = re.search(r"SUMMARY weight-tensor elements: total=(\d+) MOVES=(\d+) REFUSES=(\d+) SILENT=(\d+)",
                  mutate_out)
    if not m:
        first = mutate_out.strip().splitlines()[:2]
        return "NO SUMMARY (" + " | ".join(first) + ")"
    return "total=%s MOVES=%s REFUSES=%s SILENT=%s" % m.groups()


def main():
    revert()
    ok, log = build()
    if not ok:
        print("baseline build FAILED\n" + log[-2000:])
        return 1
    base_cand = run("out\\solver_experiment.exe candidate").stdout
    base_mut = run("out\\solver_experiment.exe mutate").stdout
    print("BASELINE (unmutated product code)")
    print("  mutation matrix: " + summarize(base_mut))
    print()

    for name, path, edits in MUTANTS:
        revert()
        with open(path, "r", encoding="utf-8", newline="") as f:
            src = f.read()
        bad = False
        for old, new in edits:
            if src.count(old) != 1:
                print("%-46s PATCH SITE COUNT=%d — SKIPPED" % (name, src.count(old)))
                bad = True
                break
            src = src.replace(old, new)
        if bad:
            continue
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(src)
        ok, log = build()
        if not ok:
            print("%-46s BUILD FAILED" % name)
            print(log[-1200:])
            revert()
            continue
        cand = run("out\\solver_experiment.exe candidate").stdout
        mut = run("out\\solver_experiment.exe mutate").stdout
        changed = "BEHAVIOUR CHANGED" if cand != base_cand else "behaviour identical (equivalent mutant)"
        print("%-46s %-34s | matrix: %s" % (name, changed, summarize(mut)))
        revert()

    revert()
    ok, _ = build()
    print("\nreverted and rebuilt clean: %s" % ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
