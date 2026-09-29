#!/usr/bin/env python3
"""q1_record.py -- Q1's B5 build record (plan rev 6; rev 7; rev 8; rev 9). One record per timing build.

Rev 9 (the conductor's call; coverage-mutants-rev8 4c, 4d, 4e, 2e): B5 compares CODE IDENTITY, not configuration.
  * Each timing build's record carries "_engine_code": the code_sha256 (codeid.py) of the engine library its semb
    actually linked, resolved from the consumer's build tree (build.ninja's link statement for semb), and
    "_pin_verified": the commit the consumer's configure guard printed ("engine pin verified: <v> @ <commit>").
  * A pair is EQUAL (timing valid) only when EACH side's engine is its own reference engine -- the library the
    harness builds from that side's commit in a clean env -i build with the declared Release flags -- and, when
    the expected commits are given, each side's pin-verified commit is the expected one. Otherwise REFUSED.
  * That closes the class "the engine actually linked is not the reference engine" (launchers by path or by
    content, CXX, CMAKE_TOOLCHAIN_FILE by path or by content, -march, -O0/-Og/-O1/-O2/-Os, include-path
    shadows, a library swapped under a pinned prefix) without a channel list.
  * Everything rev 6-8 recorded is kept and printed as DIAGNOSIS -- the engine cache's flags and launchers (by
    path and hash), the compiler version, the captured pin-tool environment, the embedded build-configuration
    record against the release record, and whether a launcher was modified after the link -- so a REFUSED pair
    says why the engines differ. None of it is a gate any more.
Exit codes of --compare: 0 EQUAL (timing valid); 2 REFUSED (a side's engine is not its reference engine, has no
code witness, or its pin-verified commit is not the expected one).
Threat model (plan C11): a cooperative builder's accidents, not tampering.
"""
import hashlib, json, re, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import buildcfg_record, codeid
KEYS = ["CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE", "CMAKE_CXX_COMPILER_LAUNCHER",
        "CMAKE_C_COMPILER_LAUNCHER", "CMAKE_GENERATOR"]

def launcher(v):
    if not v:
        return ""
    p = Path(v)
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else "unreadable"
    return f"{p} sha256:{h}"

def record(d):
    d = Path(d); b = d / "work" / "build"
    cache = (b / "CMakeCache.txt").read_text() if (b / "CMakeCache.txt").exists() else ""
    rec = {k: (re.search(rf"^{k}:[A-Z]+=(.*)$", cache, re.M) or [None, ""])[1] for k in KEYS}
    for k in ("CMAKE_CXX_COMPILER_LAUNCHER", "CMAKE_C_COMPILER_LAUNCHER"):
        rec[k] = launcher(rec[k])
    comp = rec["CMAKE_CXX_COMPILER"] or "c++"
    rec["compiler_version"] = subprocess.run([comp, "-dumpfullversion"], capture_output=True, text=True).stdout.strip()
    env = d / "pin-env.json"
    rec["pin_env"] = json.loads(env.read_text()) if env.exists() else None
    semb = d / "build" / "semb"
    recs = buildcfg_record.extract([str(semb)])["files"][str(semb)]
    rec["embedded_buildcfg_record"] = recs if isinstance(recs, list) else []
    late = []
    if semb.exists():
        for k in ("CMAKE_CXX_COMPILER_LAUNCHER", "CMAKE_C_COMPILER_LAUNCHER"):
            lp = Path(rec[k].split(" sha256:")[0]) if rec[k] else None
            if lp and lp.is_file() and lp.stat().st_mtime > semb.stat().st_mtime:
                late.append(str(lp))
    rec["_launcher_modified_after_link"] = late
    log = (d / "configure.log").read_text(errors="replace") if (d / "configure.log").exists() else ""
    m = re.search(r"engine pin verified: (\S+) @ ([0-9a-f]{40})", log)
    rec["_pin_verified"] = m.group(2) if m else None
    try:
        rec["_engine_code"] = codeid.linked(str(d / "build"), "semb")
    except (OSError, subprocess.CalledProcessError) as e:
        rec["_engine_code"] = {"error": str(e)}
    return rec

def code_ref(path):
    j = json.loads(Path(path).read_text())
    return j["code"]["release"] if "code" in j else j

def gate(rec, ref, commit):
    why = []
    ec = rec.get("_engine_code") or {}
    if not ec.get("code_sha256"):
        why.append(f"no code witness for the engine semb linked ({ec.get('error', 'absent')})")
    else:
        ok, d = codeid.compare(ec, ref)
        if not ok:
            why.append(f"the engine semb linked is not the reference engine for its commit: {d}")
    if commit and rec.get("_pin_verified") != commit:
        why.append(f"pin-verified commit is {str(rec.get('_pin_verified'))[:12]}, expected {commit[:12]}")
    return why

def diagnosis(rec, release_record):
    out = []
    pe = rec.get("pin_env")
    if pe is None:
        out.append("no captured pin-tool environment")
    else:
        out += [f"pin env {k}={v!r}" for k, v in sorted(pe.items()) if v]
    if rec.get("CMAKE_CXX_FLAGS"):
        out.append(f"CMAKE_CXX_FLAGS={rec['CMAKE_CXX_FLAGS']!r}")
    if rec.get("CMAKE_CXX_COMPILER_LAUNCHER"):
        out.append(f"launcher {rec['CMAKE_CXX_COMPILER_LAUNCHER']}")
    out += [f"launcher modified after the link: {x}" for x in rec.get("_launcher_modified_after_link") or []]
    r = rec.get("embedded_buildcfg_record") or []
    if len(r) != 1:
        out.append(f"{len(r)} build-configuration records in semb")
    elif release_record and r[0] != release_record:
        a = dict(x.split("=", 1) for x in r[0][16:-1].split(";") if x)
        b = dict(x.split("=", 1) for x in release_record[16:-1].split(";") if x)
        out.append("record: " + "; ".join(f"{k}={a.get(k)} (release {b.get(k)})" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)))
    elif release_record:
        out.append("record equals the release record")
    return out

def arg(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None

if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        a, b = (json.loads(Path(p).read_text()) for p in sys.argv[2:4])
        rb = code_ref(arg("--code-ref-before")); ra = code_ref(arg("--code-ref-after"))
        rel = json.loads(Path(arg("--record-reference")).read_text())["record"]["release"] if arg("--record-reference") else None
        refused = False
        for name, r, ref, c in (("before", a, rb, arg("--commit-before")), ("after", b, ra, arg("--commit-after"))):
            w = gate(r, ref, c)
            for x in w:
                print(f"  REFUSED ({name}): {x}"); refused = True
            if not w:
                print(f"  {name}: the engine semb linked is its reference engine (code_sha256 {ref['code_sha256'][:16]})"
                      + (f"; pin verified @ {c[:12]}" if c else ""))
            for x in diagnosis(r, rel):
                print(f"    diagnosis ({name}): {x}")
        print("  => B5:", "REFUSED (timing void)" if refused else "EQUAL (timing valid)")
        sys.exit(2 if refused else 0)
    print(json.dumps(record(sys.argv[1]), indent=1, sort_keys=True))
