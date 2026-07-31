# T-1421 reconciliation run.
#
# Not a population. Four named defects the prior debunking pass reported as
# leaving the discrimination matrix at exactly 40/16/0, re-measured under THIS
# pass's instrument definitions so the two results can be compared on the same
# axis. The question being settled is narrow: is "invisible" a property of the
# matrix's COUNTS or of its per-element signature?

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import popper2_run as R  # noqa: E402

FS = "src/forward/forward_sites.cpp"

DEFECTS_ALL = [
    # name, file, old text, new text
    ("P-A RMSNorm denominator off by one", FS,
     "ISqrt(FloorDivI64(sumsq << (2 * kNormFracBits), static_cast<int64_t>(hidden_size)))",
     "ISqrt(FloorDivI64(sumsq << (2 * kNormFracBits), static_cast<int64_t>(hidden_size + 1)))"),
    ("P-B gate and up projection weights swapped", FS,
     "\t\tst = ProjectAndFunnel(normed.data(), mlp_normed_scale, lw.gate_weight, hidden_size,\n"
     "\t\t                      intermediate_size, lw.proj_identity, lw.proj_mult, lw.proj_shift,\n"
     "\t\t                      lw.gate_site_constant, gate_codes.data(), &gate_scale,\n"
     "\t\t                      LayerSite(site_prefix, l, \"gate_proj.requant\"), token_index,\n"
     "\t\t                      trace_hook_state);\n"
     "\t\tif (st != SslmForwardStatus::Ok) return st;\n"
     "\t\tst = ProjectAndFunnel(normed.data(), mlp_normed_scale, lw.up_weight, hidden_size,",
     "\t\tst = ProjectAndFunnel(normed.data(), mlp_normed_scale, lw.up_weight, hidden_size,\n"
     "\t\t                      intermediate_size, lw.proj_identity, lw.proj_mult, lw.proj_shift,\n"
     "\t\t                      lw.gate_site_constant, gate_codes.data(), &gate_scale,\n"
     "\t\t                      LayerSite(site_prefix, l, \"gate_proj.requant\"), token_index,\n"
     "\t\t                      trace_hook_state);\n"
     "\t\tif (st != SslmForwardStatus::Ok) return st;\n"
     "\t\tst = ProjectAndFunnel(normed.data(), mlp_normed_scale, lw.gate_weight, hidden_size,"),
    ("P-C mlp_residual reconciles against seq, not the staged stream", FS,
     "\t\tst = ResidualReconcileSite(down_codes.data(), down_scale, attn_stream.data(),\n"
     "\t\t                           attn_stream_scale, hidden_size, lw.mlp_residual_site_constant,",
     "\t\tst = ResidualReconcileSite(down_codes.data(), down_scale, seq.hidden_codes,\n"
     "\t\t                           seq.hidden_scale, hidden_size, lw.mlp_residual_site_constant,"),
    ("P-D attention residual connection dropped", FS,
     "\t\tst = ResidualReconcileSite(o_codes.data(), o_scale, seq.hidden_codes, seq.hidden_scale,",
     "\t\tst = ResidualReconcileSite(o_codes.data(), o_scale, o_codes.data(), o_scale,"),
]

DEFECTS = [d for d in DEFECTS_ALL if not os.environ.get("ONLY") or os.environ["ONLY"] in d[0]]

OUTFILE = os.path.join(R.ROOT, "experiments", "popper2_prior_results.jsonl")


def main():
    R.ENV = R.msvc_env()
    R.CL = R.find_cl(R.ENV)
    with open(os.path.join(R.ROOT, "experiments", "popper2_reference.json"),
              encoding="utf-8") as fh:
        ref = json.load(fh)

    for name, rel, old, new in DEFECTS:
        path = os.path.join(R.ROOT, rel.replace("/", os.sep))
        with open(path, "r", encoding="utf-8", newline="") as fh:
            text = fh.read()
        if "\r\n" in text:  # the working tree is CRLF; the anchors are written LF
            old = old.replace("\n", "\r\n")
            new = new.replace("\n", "\r\n")
        if text.count(old) != 1:
            print(f"{name}: PATCH ANCHOR count={text.count(old)} -- SKIPPED", flush=True)
            continue
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text.replace(old, new))
        t0 = time.time()
        try:
            obs = R.measure()
        finally:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        rec = {"name": name, "secs": round(time.time() - t0, 1)}
        if obs.get("harness_build") != "OK" or obs.get("suite_build") != "OK":
            rec["status"] = "BUILD_FAIL"
            rec["log"] = (obs.get("harness_build_log", "")
                          or obs.get("suite_build_log", ""))[-600:]
        else:
            rec["status"] = "MEASURED"
            rec["null_changed"] = obs["null"] != ref["null"]
            rec["candidate_changed"] = obs["candidate"] != ref["candidate"]
            rec["mutate_changed"] = obs["mutate"] != ref["mutate"]
            import re
            rec["mutate_summary"] = re.search(r"SUMMARY.*", obs["mutate"]).group(0)
            rec["mutate_summary_changed"] = rec["mutate_summary"] != re.search(
                r"SUMMARY.*", ref["mutate"]).group(0)
            # Which per-element verdict lines changed.
            a = ref["mutate"].splitlines()
            b = obs["mutate"].splitlines()
            rec["changed_lines"] = [f"{x} -> {y}" for x, y in zip(a, b) if x != y][:20]
            rec["suite"] = obs["suite"]
            rec["suite_rc"] = obs["suite_rc"]
        with open(OUTFILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{name}: {rec['status']} counts_changed="
              f"{rec.get('mutate_summary_changed')} signature_changed="
              f"{rec.get('mutate_changed')} pin_geom={rec.get('null_changed')} "
              f"suite_fail={rec.get('suite', {}).get('failures')} "
              f"({rec['secs']}s)", flush=True)


if __name__ == "__main__":
    main()
