#!/usr/bin/env python3
"""route_e_check.py -- the decision half of cell 10.0's route E leg (plan rev 5; E3 rewritten at rev 6, rev 7, rev 8 and rev 9).

Grades one leg built through plan §11.R against a REFERENCE written before the route ran. Field provenance
(reference.json "provenance"): commit and version READ from the candidate source (git rev-parse of the tag;
project(... VERSION) in its CMakeLists.txt); the -std READ from its CMAKE_CXX_STANDARD; the per-config define
lists DECLARED by the harness author; the per-config macro sets DERIVED by the compiler over a git archive of
the candidate commit with only the declared defines, in a clean environment (engine_macros.py reference).

Checks, each named so a reading site can cite it (plan §9 10.0):
  E1 pin line   configure log carries "SuperSLM engine pin verified: <version> @ <reference commit>"
  E2 lock       the lock the pin tool wrote has commit=<reference commit>
  E3 code      rev 9 (the conductor's call): code identity. The engine library each consumer binary actually
                linked -- semb, resolved from the consumer's build tree (build.ninja's link statement), and the 10.0
                tool, resolved from the harness's own link command (codeid.json) -- must have the code_sha256 of the
                REFERENCE engine: built by the harness from the candidate commit in a clean env -i build with the
                declared Release flags and the config's declared defines (codeid.py build-reference). It closes the
                class "the engine actually linked is not the reference engine" -- launchers, CXX, toolchain files,
                -march, -O levels, library swaps -- without a channel census. The build-configuration record (rev 7,
                rev 8) is printed beside it as "E3-record (diagnosis)": why two engines differ, never the gate.
  E4 seam       provenance.json's compile closure lists tests/support/matmul_dispatch_instrument.h
  E5 opens      the tool opened the artifact and both encodes returned (28 and 276 tokens)
  E6 reach      tiled_entries > 0 at both lengths
"""
import json, re, sys
from pathlib import Path

def main():
    ref = json.loads(Path(sys.argv[1]).read_text())
    leg = Path(sys.argv[2]); config = sys.argv[3]
    res = {}
    log = (leg / "configure.log").read_text(errors="replace")
    m = re.search(r"SuperSLM engine pin verified: (\S+) @ ([0-9a-f]{40})", log)
    res["E1 pin line"] = (bool(m) and m.group(1) == ref["version"] and m.group(2) == ref["commit"],
                          m.group(0) if m else "no pin-verified line")
    lock = (leg / "semb" / "superslm.lock").read_text()
    lc = re.search(r"^commit=([0-9a-f]{40})", lock, re.M)
    res["E2 lock"] = (bool(lc) and lc.group(1) == ref["commit"], f"commit={lc.group(1) if lc else None}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import buildcfg_record
    cw = leg / "codeid.json"
    if cw.exists():
        cid = json.loads(cw.read_text())
        import codeid
        want = ref["code"][config]
        bad = [f"{b}: {e.get('error', 'no library resolved')}" for b, e in cid.items() if not e.get("code_sha256")]
        if not cid:
            res["E3 code"] = (False, "no linked binaries in the code witness")
        elif bad:
            res["E3 code"] = (False, "; ".join(bad))
        else:
            diffs = [(b, codeid.compare(e, want)) for b, e in cid.items()]
            off = [f"{b} links {'/'.join(Path(cid[b]['library']).parts[-4:])}, {why}" for b, (ok, why) in diffs if not ok]
            res["E3 code"] = (not off, "; ".join(off) if off else
                              f"the engine linked by {', '.join(cid)} is the reference engine (code_sha256 {want['code_sha256'][:16]})")
    else:
        res["E3 code"] = (False, "no code witness (codeid.json)")
    diag = None
    wit = leg / "buildcfg.json"
    if wit.exists():
        _, diag = buildcfg_record.verdict(json.loads(wit.read_text()), ref["record"][config])
    pj = leg / "eng" / "share" / "superslm-provenance" / "provenance.json"
    ptxt = pj.read_text() if pj.exists() else ""
    res["E4 seam"] = ("tests/support/matmul_dispatch_instrument.h" in ptxt,
                      "seam header in provenance" if "matmul_dispatch_instrument.h" in ptxt else "seam header absent")
    run = (leg / "tool.out").read_text() if (leg / "tool.out").exists() else ""
    enc = re.findall(r"tokens_reaching_gemm=(\d+) status=(\d+) tiled_entries=(-?\d+)", run)
    res["E5 opens"] = (run.startswith("open: ok") and len(enc) == 2 and all(s in ("0", "7") for _, s, _ in enc),
                       run.splitlines()[0] if run else "tool did not build or run")
    res["E6 reach"] = (len(enc) == 2 and all(int(t) > 0 for _, _, t in enc),
                       ", ".join(f"M={n}: {t}" for n, _, t in enc) or "no reading")
    ok = all(v[0] for v in res.values())
    for k, (p, why) in res.items():
        print(f"  {k:<12} {'pass' if p else 'FAIL'}  {why}")
        if k == "E3 code":
            print(f"  E3-record    (diagnosis, not graded)  {diag or 'no record witness (buildcfg.json)'}")
    print(f"  => graded as '{config}': {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
