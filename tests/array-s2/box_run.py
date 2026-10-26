"""The S2 box run, as one resumable script: build, attach, every runtime cell on CPU and GPU, the
mutants, the timing, and the raw logs that back every figure in RESULTS.md.

    python tests\\array-s2\\box_run.py [--model-dir D:\\_artifacts\\SuperSLM\\aex] [--base X.sslm] [--ckpt DIR]
    python tests\\array-s2\\box_run.py --list            # the steps, their commands and state
    python tests\\array-s2\\box_run.py --redo <step>     # forget this step and every later one, then run

Each step is short (a few minutes at most), writes its raw output under tests/array-s2/results/ as it
runs, and records `results/box/<step>.done` when it ends as expected. The record carries the git HEAD
and, for every step after `build_harness`, the SHA-256 of both harness binaries: a rerun skips a step
only when its record matches the current HEAD and binaries, so a crash costs only the step in
progress, and a pull or a rebuild reruns what it invalidates. A `--base` or `--ckpt` that differs
from the one `locate` recorded stops the run. A step that ends otherwise records
`<step>.failed` and the run stops there; the log names what failed. A mutant step passes only when
its log shows the mutant applied and caught by the specific expected detection (an accepting-flag,
walk-state or token mismatch, or the map refusal), with no setup failure, missing row or restore error.

Before every timed step the script checks that no other harness, test binary or attach is running,
and stops if one is. Nothing here commits: the box session reviews results/ and RESULTS.md and commits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
BOX = RES / "box"
V2 = REPO / "tests" / "t3094-multicommand-s0" / "v2"
ROWS = ["ex005", "ex006", "ex033", "ex034", "ex035", "ex041", "ex042", "ex051", "ex052", "ex053"]
TF_DOCS = ["one_intent_0_conditions", "one_intent_1_condition", "three_intents_0_2_1_conditions"]
TF_PROMPT = "ex041"
BUSY_NAMES = ("array_s2", "superslm_tests", "sslm_verify", "t2132_", "schema_fast_decode")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class Box:
    def __init__(self, args):
        self.args = args
        self.exe = REPO / "out" / "array_s2.exe"
        self.seam = REPO / "out" / "array_s2_seam.exe"
        self.verifier = REPO / "out" / "sslm_verify.exe"
        self.art_dir = Path(args.artifact_dir)
        self.scratch = Path(args.scratch)
        self.artifact = self.art_dir / "qwen05-intentdoc-v2.sslm"
        self.fixtures = RES / "fixtures.txt"
        self.loc = {}
        if (BOX / "locate.json").is_file():
            self.loc = json.loads((BOX / "locate.json").read_text(encoding="utf-8"))

    # --- helpers -------------------------------------------------------------------------------
    def run(self, log: Path, cmd: list, env_extra: dict | None = None, append: bool = False) -> int:
        env = dict(os.environ)
        env.update(env_extra or {})
        cmd = [str(c) for c in cmd]
        with open(log, "a" if append else "w", encoding="utf-8") as out:
            out.write("# " + " ".join(cmd) + (f"   (env {env_extra})" if env_extra else "") + "\n")
            out.flush()
            p = subprocess.run(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, env=env)
            out.write(f"# exit {p.returncode}\n")
        return p.returncode

    def harness(self, log: Path, *a, exe: Path | None = None, env_extra=None, append=False) -> int:
        return self.run(log, [exe or self.exe, *a], env_extra, append)

    def py(self, log: Path, *a, append=False, env_extra=None) -> int:
        return self.run(log, [sys.executable, *a], env_extra, append)

    @property
    def base(self) -> Path:
        return Path(self.loc["base"])

    @property
    def ckpt(self) -> Path:
        return Path(self.loc["ckpt"])

    def busy(self) -> list[str]:
        try:
            import psutil
        except ImportError:
            return ["psutil is not installed, so the busy check cannot run (pip install psutil)"]
        me = os.getpid()
        hits = []
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if p.info["pid"] == me:
                    continue
                name = (p.info["name"] or "").lower()
                cmd = " ".join(p.info["cmdline"] or []).lower()
                if any(n in name for n in BUSY_NAMES) or ("sslm_attach_schema" in cmd) or \
                        ("box_run.py" in cmd and p.info["pid"] != me and "python" in name):
                    hits.append(f"{p.info['pid']} {name} {cmd[:160]}")
            except psutil.Error:
                continue
        return hits

    # --- steps ---------------------------------------------------------------------------------
    def locate(self, log: Path) -> int:
        """Finds the base artifact (an .sslm with no SchemaMasks section) and the tokenizer checkpoint."""
        sys.path[:0] = [str(REPO / "tools")]
        import sslm_format as SF
        lines, model_dir = [], Path(self.args.model_dir)
        base = Path(self.args.base) if self.args.base else None
        if base is None:
            cands = []
            for f in sorted(model_dir.rglob("*.sslm")):
                h = SF.read_header(str(f))
                has_scm1 = any(s["type"] == SF.SectionType.SCHEMA_MASKS for s in h["sections"])
                has_tok = any(s["type"] == SF.SectionType.TOKENIZER for s in h["sections"])
                lines.append(f"{f}: schema_masks={has_scm1} tokenizer={has_tok}")
                if not has_scm1 and has_tok:
                    cands.append(f)
            if len(cands) != 1:
                lines.append(f"base: {len(cands)} candidate(s) under {model_dir}; pass --base <the Qwen2.5-0.5B int8 "
                             "artifact with a TOKENIZER section and no SchemaMasks section>")
                log.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return 2
            base = cands[0]
        ckpt = Path(self.args.ckpt) if self.args.ckpt else None
        if ckpt is None:
            cands = sorted({p.parent for p in model_dir.rglob("tokenizer.json") if (p.parent / "config.json").is_file()})
            lines += [f"tokenizer dir: {c}" for c in cands]
            if len(cands) != 1:
                lines.append(f"ckpt: {len(cands)} candidate(s) under {model_dir}; pass --ckpt <the Qwen2.5-0.5B-Instruct "
                             "snapshot 7ae5576 directory the base was converted from>")
                log.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return 2
            ckpt = cands[0]
        self.loc = {"base": str(base), "ckpt": str(ckpt), "base_bytes": base.stat().st_size, "base_sha256": sha256(base)}
        (BOX / "locate.json").write_text(json.dumps(self.loc, indent=1), encoding="utf-8")
        lines.append(json.dumps(self.loc))
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0

    def python_deps(self, log: Path) -> int:
        """tokenizers (fixtures), jsonschema (the output check) and psutil (measure.py, the busy check)."""
        import importlib.util
        missing = [m for m in ("tokenizers", "jsonschema", "psutil") if importlib.util.find_spec(m) is None]
        log.write_text(f"python {sys.version.split()[0]} at {sys.executable}; missing: {missing or 'none'}\n"
                       + (f"install with: {sys.executable} -m pip install {' '.join(missing)}\n" if missing else ""),
                       encoding="utf-8")
        return 1 if missing else 0

    def vsdevcmd(self) -> dict:
        if os.environ.get("VSDEVCMD"):
            return {}
        vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
            "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.is_file():
            p = subprocess.run([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                               capture_output=True, text=True)
            path = Path(p.stdout.strip()) / "Common7" / "Tools" / "VsDevCmd.bat"
            if p.stdout.strip() and path.is_file():
                return {"VSDEVCMD": f'"{path}"'}
        return {}

    def build_verify(self, log: Path) -> int:
        rc = self.run(log, ["cmd", "/c", REPO / "tools" / "build_verify.bat"], self.vsdevcmd())
        return rc if rc else (0 if self.verifier.is_file() else 1)

    def shaders(self, log: Path) -> int:
        """Compiles every GPU shader exactly as build.bat does, into out/shaders beside the harness."""
        kits = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits" / "10" / "bin"
        pinned = kits / "10.0.19041.0" / "x64" / "dxc.exe"
        found = sorted(kits.glob("10.*/x64/dxc.exe"))
        dxc = pinned if pinned.is_file() else (found[-1] if found else None)
        if dxc is None:
            log.write_text(f"dxc.exe not found under {kits}\n", encoding="utf-8")
            return 2
        out = REPO / "out" / "shaders"
        out.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as f:
            f.write(f"dxc: {dxc}{'' if dxc == pinned else '  (NOT the pinned 10.0.19041.0 build.bat uses)'}\n")
        for hlsl in sorted((REPO / "src" / "gpu" / "shaders").glob("*.hlsl")):
            rc = self.run(log, [dxc, "-T", "cs_6_2", "-E", "main", "-Fo", out / (hlsl.stem + ".cso"), hlsl,
                                "-O3", "-HV", "2018", "-WX"], append=True)
            if rc:
                return 1
        return 0

    def build_harness(self, log: Path) -> int:
        rc = self.run(log, ["cmd", "/c", HERE / "build_harness.bat"], self.vsdevcmd())
        return rc if rc else (0 if self.exe.is_file() and self.seam.is_file() else 1)

    def attach(self, log: Path, out: Path, schemas: list[str], measure: bool = True) -> int:
        cmd = [REPO / "tools" / "sslm_attach_schema.py", "--base", self.base, "--ckpt", self.ckpt]
        for s in schemas:
            cmd += ["--schema", s]
        cmd += ["--out", out, "--verifier", self.verifier, "--force"]
        if measure:
            return self.py(log, HERE / "measure.py", log.with_suffix(".measure.log"), "--", sys.executable, *cmd)
        return self.py(log, *cmd)

    def schemas(self, only_intents=False) -> list[str]:
        s = [f"intents:{V2 / 'intentdoc-v2.schema.json'}"]
        return s if only_intents else s + [f"intent_gate:{V2 / 'g1.schema.json'}"]

    def attach_main(self, log: Path) -> int:
        self.art_dir.mkdir(parents=True, exist_ok=True)
        rc = self.attach(log, self.artifact, self.schemas())
        if rc == 0:
            with open(RES / "artifacts.txt", "w", encoding="utf-8") as f:
                f.write(f"base {self.base} bytes={self.base.stat().st_size} sha256={self.loc['base_sha256']}\n")
                f.write(f"attached {self.artifact} bytes={self.artifact.stat().st_size} sha256={sha256(self.artifact)}\n")
                f.write(f"ckpt {self.ckpt}\n")
        return rc

    def reattach(self, log: Path) -> int:
        self.scratch.mkdir(parents=True, exist_ok=True)
        again = self.scratch / "reattach.sslm"
        rc = self.attach(log, again, self.schemas())
        if rc:
            return rc
        a, b = sha256(self.artifact), sha256(again)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"sha256 first={a}\nsha256 again={b}\nidentical={a == b}\n")
        return 0 if a == b else 1

    def attach_intents_only(self, log: Path) -> int:
        self.scratch.mkdir(parents=True, exist_ok=True)
        return self.attach(log, self.scratch / "intents-only.sslm", self.schemas(only_intents=True))

    def mutant_artifacts(self, log: Path) -> int:
        """Both mutants: the patched attach copies (whose own verifier gate is the fixed verifier), and
        the same mutations applied directly to the clean artifact, so the loaders can be run on both."""
        mdir = self.scratch / "mutants"
        mdir.mkdir(parents=True, exist_ok=True)
        ok = self.py(log, HERE / "make_mutants.py", REPO / "tools", mdir) == 0
        lines = []
        for mode in ("accept_shift", "offset_swap"):
            out = mdir / f"{mode}-attach.sslm"
            out.unlink(missing_ok=True)
            arc = self.run(log, [sys.executable, mdir / mode / "sslm_attach_schema.py", "--base", self.base, "--ckpt",
                                 self.ckpt, *sum((["--schema", s] for s in self.schemas()), []), "--out", out,
                                 "--verifier", self.verifier, "--force"],
                           env_extra={"PYTHONPATH": str(REPO / "tools")}, append=True)
            written = out.is_file()
            # The fixed verifier accepts accept_shift (structurally valid) and refuses offset_swap.
            ok_attach = (arc == 0 and written) if mode == "accept_shift" else (arc != 0 and not written)
            lines.append(f"attach_copy {mode} exit={arc} written={written} as_expected={ok_attach}")
            direct = mdir / f"{mode}.sslm"
            mark = len(log.read_text(encoding="utf-8", errors="replace"))
            drc = self.py(log, HERE / "scm1_mutants.py", self.artifact, mode, direct, append=True)
            applied = drc == 0 and f"MUTANT {mode}:" in log.read_text(encoding="utf-8", errors="replace")[mark:]
            differs = applied and direct.is_file() and sha256(direct) != sha256(self.artifact)
            vrc = self.run(log, [self.verifier, direct, mdir / f"{mode}.verify.json"], append=True) if applied else -1
            ok_verify = vrc == (0 if mode == "accept_shift" else 1)
            lines.append(f"direct {mode} applied={applied} differs_from_clean={differs} "
                         f"sha256={sha256(direct) if direct.is_file() else '-'} verifier_exit={vrc} as_expected={ok_verify}")
            ok = ok and ok_attach and applied and differs and ok_verify
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + f"\nmutant_artifacts ok={ok}\n")
        (RES / "mutants_artifacts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0 if ok else 1

    def refused_at_map(self, log: Path, artifact: Path, backend: str) -> int:
        """0 only when the harness exits 2 with the backend's own map call named as the refusal."""
        rc = self.harness(log, "walks", artifact, "intents", self.fixtures, backend)
        call = "sslm_model_map" if backend == "cpu" else "sslm_gpu_model_map"
        text = log.read_text(encoding="utf-8", errors="replace")
        ok = rc == 2 and f"setup failed: error:{call} " in text
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"refused_at_map={ok} ({call})\n")
        return 0 if ok else 1

    KILL_CHECKS = ("tokens", "accepting", "states", "restored_state", "restored_accepting", "end",
                   "forced_end", "forced_tokens", "forced_accepting", "forced_states")

    def planned(self, mode: str, key: str) -> int:
        if mode == "tfrestore":
            return sum(1 for l in self.fixtures.read_text(encoding="ascii").splitlines()
                       if l.startswith(f"tfsave {key}:"))
        return sum(1 for l in (RES / "saveplan.txt").read_text(encoding="ascii").splitlines()
                   if l.split()[1:2] == [key])

    def verdict(self, log: Path, ok: bool, why: str) -> int:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"mutant_killed={ok} ({why})\n")
        return 0 if ok else 1

    def state_zero_kill(self, log: Path, mode: str, backend: str, key: str) -> int:
        """The restore-at-state-zero mutant is killed only when the harness exits 1, every planned save
        point produced a result with the mutant applied (`mutated=1`), the unbroken trace or walk passed,
        no setup or restore error occurred, and at least one point failed on a trace check."""
        A, S, F = self.artifact, "intents", self.fixtures
        args = (["tfrestore", A, S, F, backend, TF_PROMPT, key] if mode == "tfrestore" else
                ["restore", A, S, F, RES / "gen.txt", RES / "saveplan.txt", backend, key])
        rc = self.harness(log, *args, "--state-zero")
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        results = [l for l in lines if l.startswith(("PASS ", "FAIL ")) and " after=" in l]
        traces = [l for l in lines if l.startswith(("PASS ", "FAIL ")) and (" walk tokens=" in l or " trace tokens=" in l)]
        bad = [l for l in lines if " setup:" in l or "check=error" in l or "no save point" in l or
               "no teacher-forced" in l or "setup failed" in l]
        kills = [l for l in results if l.startswith("FAIL ") and
                 (re.search(r" check=(\w+)", l) or [None, None])[1] in self.KILL_CHECKS]
        want = self.planned(mode, key)
        why = (f"exit={rc} results={len(results)}/{want} applied={sum(' mutated=1' in l for l in results)} "
               f"trace_pass={bool(traces) and all(l.startswith('PASS') for l in traces)} errors={len(bad)} kills={len(kills)}")
        ok = (rc == 1 and want > 0 and len(results) == want and all(" mutated=1" in l for l in results) and
              bool(traces) and all(l.startswith("PASS") for l in traces) and not bad and kills)
        return self.verdict(log, bool(ok), why)

    def accept_shift_walks_kill(self, log: Path, backend: str) -> int:
        """Killed when a walk admits every token and the accepting check (after the last token, or after
        any token) fails, with no setup failure."""
        rc = self.harness(log, "walks", self.scratch / "mutants" / "accept_shift.sslm", "intents", self.fixtures, backend)
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        def field(l, k):
            for f in l.split():
                if f.startswith(k + "="):
                    return f.split("=", 1)[1]
            return None
        walks = [l for l in lines if l.startswith(("PASS ", "FAIL ")) and f" {backend} walk " in l]
        kills = [l for l in walks if l.startswith("FAIL ") and field(l, "admitted") == field(l, "tokens") and
                 (field(l, "accepting_end") == "0" or field(l, "bad_accepting_at") not in (None, "-1"))]
        bad = [l for l in lines if "setup failed" in l or "no walk or reject" in l]
        return self.verdict(log, rc == 1 and bool(kills) and not bad,
                            f"exit={rc} walks={len(walks)} accepting_kills={len(kills)} errors={len(bad)}")

    def accept_shift_gen_kill(self, log: Path, backend: str) -> int:
        """Killed when generation runs and ends at the dead end with accepting=0 (the shifted accept set)."""
        rows = ["ex005", "ex041"]
        rc = self.harness(log, "gen", self.scratch / "mutants" / "accept_shift.sslm", "intents", self.fixtures,
                          backend, "plain", *rows)
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        recs = [l for l in lines if l.startswith(f"# {backend} ") and " end=" in l]
        kills = [l for l in recs if " end=dead_end " in l and " accepting=0 " in l]
        bad = [l for l in lines if "setup failed" in l or "# FAIL" in l]
        return self.verdict(log, rc == 1 and len(recs) == len(rows) and bool(kills) and not bad,
                            f"exit={rc} records={len(recs)}/{len(rows)} dead_end_not_accepting={len(kills)} errors={len(bad)}")

    def canon_info(self, log: Path) -> int:
        """Canonical jump-forward (being removed in 1.18), information only: plain CPU against canonical,
        recorded with validity and the first difference. Passes when the check ran to its end, whatever it
        found; a crash or a missing stream fails."""
        self.concat(RES / "gen_canon.txt", [BOX / "gen_cpu_plain.log", BOX / "gen_cpu_canon.log"])
        rc = self.py(log, HERE / "fixtures.py", "check", "--ckpt", self.ckpt, "--gen", RES / "gen_canon.txt",
                     "--json", RES / "gen_check_canon.json")
        text = log.read_text(encoding="utf-8", errors="replace")
        rows = sum(1 for l in text.splitlines() if l.startswith("ex") and ": " in l)
        ok = rc in (0, 1) and "failures:" in text and "Traceback" not in text and rows == len(ROWS)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"canon_info completed={ok} rows={rows} (information only)\n")
        return 0 if ok else 1

    def concat(self, out: Path, parts: list[Path]) -> int:
        with open(out, "w", encoding="ascii") as f:
            for p in parts:
                f.write(p.read_text(encoding="ascii", errors="replace"))
        return 0

    def same_gen(self, log: Path, a: Path, b: Path) -> int:
        def ids(p):
            return {tuple(l.split()[1:3]): l.split()[3:] for l in p.read_text(encoding="ascii").splitlines()
                    if l.startswith("gen ")}
        x, y = ids(a), ids(b)
        bad = [k for k in x if x[k] != y.get(k)] + [k for k in y if k not in x]
        log.write_text(f"{a.name} vs {b.name}: {len(x)} streams, identical={not bad and bool(x)} differing={bad}\n",
                       encoding="ascii")
        return 0 if x and not bad else 1

    # --- the step list ---------------------------------------------------------------------------
    def steps(self):
        """(name, timed, expected exits, callable(log) -> exit)."""
        A, S, F = self.artifact, "intents", self.fixtures
        st = [
            ("python_deps", False, (0,), self.python_deps),
            ("locate", False, (0,), self.locate),
            ("build_verify", False, (0,), self.build_verify),
            ("shaders", False, (0,), self.shaders),
            ("build_harness", False, (0,), self.build_harness),
            ("fixtures", False, (0,), lambda L: self.py(L, HERE / "fixtures.py", "build", "--ckpt", self.ckpt,
                                                         "--out", F)),
            ("precheck", False, (0,), lambda L: self.py(L, HERE / "fixtures.py", "precheck", "--ckpt", self.ckpt,
                                                         "--fixtures", F, "--out", RES / "precheck.txt")),
            ("verify_check", False, (0,), lambda L: self.py(L, HERE / "verify_schema_masks_check.py", self.verifier,
                                                             env_extra={"T3094_TOKENIZER_DIR": str(self.ckpt)})),
            ("attach", True, (0,), self.attach_main),
            ("reattach", True, (0,), self.reattach),
            ("attach_intents_only", True, (0,), self.attach_intents_only),
        ]
        for k in (1, 2, 3):
            st.append((f"map_{k}", True, (0,), lambda L: self.harness(L, "map", self.base, A, S)))
        st.append(("map", False, (0,), lambda L: self.concat(RES / "map.txt", [BOX / f"map_{k}.log" for k in (1, 2, 3)])))
        for b in ("cpu", "gpu"):
            st.append((f"walks_{b}", False, (0,), lambda L, b=b: self.harness(L, "walks", A, S, F, b)))
        for run in (1, 2):
            sfx = "" if run == 1 else "_run2"
            variants = [("cpu", "plain", self.exe), ("gpu", "plain", self.exe), ("cpu", "canon", self.exe),
                        ("cpu", "exact", self.exe), ("cpu", "full", self.seam)]
            for b, v, exe in variants:
                name = f"gen_{b}_{v}{sfx}"
                st.append((name, True, (0,), lambda L, b=b, v=v, exe=exe: self.harness(L, "gen", A, S, F, b, v, exe=exe)))
            parts = [BOX / f"gen_{b}_{v}{sfx}.log" for b, v, _ in variants]
            st.append((f"gen{sfx}", False, (0,), lambda L, sfx=sfx, parts=parts:
                       self.concat(RES / f"gen{sfx}.txt", parts)))
        # Must pass: plain CPU, plain GPU, exact jump-forward and the full finish -- every document valid
        # and every stream identical to plain CPU. Any difference, invalid document or crash fails it.
        st.append(("gen_musteq", False, (0,), lambda L: self.concat(
            RES / "gen_musteq.txt", [BOX / f"gen_{v}.log" for v in ("cpu_plain", "gpu_plain", "cpu_exact", "cpu_full")])))
        st.append(("gen_check", False, (0,), lambda L: self.py(
            L, HERE / "fixtures.py", "check", "--ckpt", self.ckpt, "--gen", RES / "gen_musteq.txt",
            "--json", RES / "gen_check.json")))
        # Information only: canonical jump-forward, which 1.18 removes.
        st.append(("gen_canon_info", False, (0,), self.canon_info))
        st.append(("gen_repeat", False, (0,), lambda L: self.same_gen(L, RES / "gen.txt", RES / "gen_run2.txt")))
        st.append(("saves", False, (0,), lambda L: self.py(L, HERE / "fixtures.py", "saves", "--ckpt", self.ckpt,
                                                          "--gen", RES / "gen.txt", "--out", RES / "saveplan.txt")))
        for b in ("cpu", "gpu"):
            for row in ROWS:
                st.append((f"restore_{b}_{row}", False, (0,), lambda L, b=b, row=row: self.harness(
                    L, "restore", A, S, F, RES / "gen.txt", RES / "saveplan.txt", b, row)))
            st.append((f"restore_{b}", False, (0,), lambda L, b=b: self.concat(
                RES / f"restore_{b}.txt", [BOX / f"restore_{b}_{r}.log" for r in ROWS])))
            for doc in TF_DOCS:
                st.append((f"tfrestore_{b}_{doc}", False, (0,), lambda L, b=b, doc=doc: self.harness(
                    L, "tfrestore", A, S, F, b, TF_PROMPT, doc)))
            st.append((f"tfrestore_{b}", False, (0,), lambda L, b=b: self.concat(
                RES / f"tfrestore_{b}.txt", [BOX / f"tfrestore_{b}_{d}.log" for d in TF_DOCS])))
            # The restore-at-state-zero mutant: applied at every planned point and caught by a trace check.
            st.append((f"state_zero_tf_{b}", False, (0,), lambda L, b=b: self.state_zero_kill(
                L, "tfrestore", b, "three_intents_0_2_1_conditions")))
            st.append((f"state_zero_gen_{b}", False, (0,), lambda L, b=b: self.state_zero_kill(L, "restore", b, "ex041")))
        st.append(("state_zero", False, (0,), lambda L: self.concat(
            RES / "state_zero.txt", [BOX / f"state_zero_{k}_{b}.log" for b in ("cpu", "gpu") for k in ("tf", "gen")])))
        # The two SCM1 mutants.
        st.append(("mutant_artifacts", False, (0,), self.mutant_artifacts))
        mdir = self.scratch / "mutants"
        for b in ("cpu", "gpu"):
            st.append((f"mutant_accept_shift_walks_{b}", False, (0,), lambda L, b=b: self.accept_shift_walks_kill(L, b)))
            st.append((f"mutant_accept_shift_gen_{b}", False, (0,), lambda L, b=b: self.accept_shift_gen_kill(L, b)))
            st.append((f"mutant_offset_swap_walks_{b}", False, (0,), lambda L, b=b: self.refused_at_map(
                L, mdir / "offset_swap.sslm", b)))
        st.append(("mutants", False, (0,), lambda L: self.concat(RES / "mutants.txt", [
            BOX / "mutant_artifacts.log"] + [BOX / f"mutant_{m}_{b}.log" for b in ("cpu", "gpu") for m in
                                            ("accept_shift_walks", "accept_shift_gen", "offset_swap_walks")])))
        # GPU decode at one slice per token, head on the host and on the device: wall and GPU-busy
        # (layer-loop timestamps) ms per step, and per prompt against token position.
        st.append(("gputime", True, (0,), lambda L: self.harness(
            L, "gputime", A, S, F, RES / "gen.txt", env_extra={"ARRAY_S2_GPU_LAYERS_PER_SLICE": "24"})))
        st.append(("gputime_copy", False, (0,), lambda L: self.concat(RES / "gputime.txt", [BOX / "gputime.log"])))
        return st

    def head(self) -> str:
        # The commit only: the run itself rewrites tracked files under results/.
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
        return p.stdout.strip() or "unknown"

    def binaries(self) -> dict:
        return {"exe": sha256(self.exe) if self.exe.is_file() else None,
                "seam": sha256(self.seam) if self.seam.is_file() else None}

    def stamp(self, after_build: bool) -> dict:
        return {"head": self.head(), **(self.binaries() if after_build else {})}

    def state(self, name: str, after_build: bool) -> str:
        """done (record matches HEAD and, after build_harness, both binaries), stale, or pending."""
        done = BOX / f"{name}.done"
        if not done.is_file():
            return "pending"
        try:
            rec = json.loads(done.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "stale"
        cur = self.stamp(after_build)
        return "done" if all(rec.get(k) == v for k, v in cur.items()) else "stale"

    def inputs_agree(self) -> bool:
        for key in ("base", "ckpt"):
            given = getattr(self.args, key)
            if given and self.loc.get(key) and Path(given).resolve() != Path(self.loc[key]).resolve():
                print(f"STOP: --{key} {given} differs from the {key} this run recorded, {self.loc[key]} "
                      f"(results/box/locate.json). To change inputs, start a fresh run: --redo locate.")
                return False
        return True

    def main(self) -> int:
        BOX.mkdir(parents=True, exist_ok=True)
        steps = self.steps()
        names = [s[0] for s in steps]
        build_idx = names.index("build_harness")
        if self.args.redo:
            if self.args.redo not in names:
                print(f"unknown step {self.args.redo}; --list shows them")
                return 2
            for n in names[names.index(self.args.redo):]:
                for ext in (".done", ".failed"):
                    (BOX / (n + ext)).unlink(missing_ok=True)
            if self.args.redo == "locate":
                (BOX / "locate.json").unlink(missing_ok=True)
                self.loc = {}
        if not self.inputs_agree():
            return 2
        for idx, (name, timed, expect, fn) in enumerate(steps):
            done = BOX / f"{name}.done"
            state = self.state(name, idx > build_idx)
            if self.args.list:
                print(f"{state:7s} {name}{'  (timed)' if timed else ''}")
                continue
            if state == "done":
                continue
            if timed:
                hits = self.busy()
                if hits and not self.args.ignore_busy:
                    print(f"STOP before {name}: another run is using the box:\n  " + "\n  ".join(hits))
                    return 3
            (BOX / f"{name}.failed").unlink(missing_ok=True)
            log = BOX / f"{name}.log"
            t0 = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
            try:
                rc = fn(log)
            except Exception as error:  # a crash is recorded as this step's failure, never skipped
                with open(log, "a", encoding="utf-8") as f:
                    f.write(f"# exception: {error!r}\n")
                rc = -1
            wall = time.perf_counter() - t0
            record = {"step": name, "exit": rc, "expected": list(expect), "wall_s": round(wall, 1),
                      "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **self.stamp(idx > build_idx)}
            if rc in expect:
                done.write_text(json.dumps(record) + "\n", encoding="utf-8")
                print(f"    ok (exit {rc}, {wall:.0f} s)")
            else:
                (BOX / f"{name}.failed").write_text(json.dumps(record) + "\n", encoding="utf-8")
                print(f"    FAILED: exit {rc}, expected {'/'.join(map(str, expect))}; see {log}")
                return 1
        if not self.args.list:
            print(f"all {len(steps)} steps done; raw outputs in {RES}, step logs in {BOX}")
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", default=r"D:\_artifacts\SuperSLM\aex",
                    help="searched for the base .sslm and the tokenizer checkpoint unless --base/--ckpt are given")
    ap.add_argument("--base", help="the base artifact: Qwen2.5-0.5B int8 .sslm, TOKENIZER section, no SchemaMasks")
    ap.add_argument("--ckpt", help="the tokenizer checkpoint directory the base was converted from")
    ap.add_argument("--artifact-dir", default=r"D:\_artifacts\SuperSLM\array-s2")
    ap.add_argument("--scratch", default=r"D:\_scratch\array-s2")
    ap.add_argument("--list", action="store_true", help="print the steps and their state, run nothing")
    ap.add_argument("--redo", help="forget this step's result and every later step's before running")
    ap.add_argument("--ignore-busy", action="store_true", help="run timed steps even if another run is seen")
    return Box(ap.parse_args(argv)).main()


if __name__ == "__main__":
    sys.exit(main())
