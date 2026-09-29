#!/usr/bin/env python3
# REPORT-ONLY from plan rev 6: no gate reads this. Its regex misreads H1-H5 (adversary round 4, F3;
# probes/route_p_hook/route-p-hook-witness-output.txt). Cell 10.0's H comes from the runtime witness.
"""hook_state.py -- cell 10.0's hook-state print (plan rev 5, K1 closure, item 4).

Reads each existing route's PREFILL hook state from the consumer's own source at the commit the
plan's route table names, so the slice-2 unpark sites have a real source. Slice 2 (plan §5.1,
opt-in) threads prefill only when a hook is installed AND `reserved` carries the prefill bit
(route P, workspace hook) or when the caller passes `pf` to RunLayerLoopChunkBatched (route E).
usage: hook_state.py <plugin Source dir> <SuperEmbedder checkout> <SuperEmbedder commit>
"""
import re, subprocess, sys
from pathlib import Path

def route_p(src: Path):
    sites = []
    for f in sorted(src.rglob("*.cpp")):
        rel = f.relative_to(src).as_posix()
        if "/Tests/" in f"/{rel}" or "ThirdParty/" in rel:
            continue
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if "sslm_workspace_set_parallel_for(" in line and "TEXT(" not in line:
                sites.append(f"{rel}:{i}")
    mk = next(src.rglob("SuperSLMFinishHook.cpp"))
    txt = mk.read_text(errors="replace")
    m = re.search(r"Hook\.reserved\s*=\s*([^;]+);", txt)
    line = txt[: m.start()].count("\n") + 1 if m else None
    reserved = m.group(1).strip() if m else "?"
    prefill_bit = reserved not in ("0", "0u", "0U")
    print(f"route P: workspace hook installed at {len(sites)} site(s): {', '.join(sites)}")
    print(f"route P: hook.reserved = {reserved} ({mk.relative_to(src).as_posix()}:{line}) -> prefill opt-in bit "
          f"{'SET' if prefill_bit else 'clear'}; prefill hooked: {'yes' if prefill_bit and sites else 'no'}")

def route_e(repo: Path, commit: str):
    files = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", commit, "src", "include", "tools"],
                           capture_output=True, text=True, check=True).stdout.split()
    refs, calls = [], []
    for f in files:
        if not f.endswith((".cpp", ".h", ".hpp", ".cc")):
            continue
        txt = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{f}"], capture_output=True,
                             text=True, errors="replace").stdout
        for i, line in enumerate(txt.splitlines(), 1):
            code = line.split("//")[0]
            if re.search(r"\bparallel_for\b|sslm_parallel_for|set_parallel_for", code):
                refs.append(f"{f}:{i}")
        for m in re.finditer(r"RunLayerLoopChunkBatched\s*\(", txt):
            depth, j = 1, m.end()
            while depth and j < len(txt):
                depth += {"(": 1, ")": -1}.get(txt[j], 0); j += 1
            args = txt[m.end(): j - 1]
            calls.append((f"{f}:{txt[: m.start()].count(chr(10)) + 1}", "pf" in re.sub(r"/\*.*?\*/", "", args) or "parallel" in args))
    print(f"route E @ {commit}: hook references in code (src, include, tools): {len(refs)} {refs}")
    for site, has_pf in calls:
        print(f"route E @ {commit}: RunLayerLoopChunkBatched call at {site} passes a hook: {'yes' if has_pf else 'no'}")
    print(f"route E: prefill hooked: {'yes' if refs and any(h for _, h in calls) else 'no'}")

if __name__ == "__main__":
    route_p(Path(sys.argv[1]))
    route_e(Path(sys.argv[2]), sys.argv[3])
