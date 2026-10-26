"""The S2 box run, as one resumable script: build, attach, every runtime cell on CPU and GPU, the
mutants, the timing, and the raw logs that back every figure in RESULTS.md.

    python tests\\array-s2\\box_run.py [--model-dir D:\\_artifacts\\SuperSLM\\aex] [--base X.sslm] [--ckpt DIR]
    python tests\\array-s2\\box_run.py --list            # the steps, their commands and state
    python tests\\array-s2\\box_run.py --redo <step>     # forget one step's result and run from there

Each step is short (a few minutes at most), writes its raw output under tests/array-s2/results/ as it
runs, and records `results/box/<step>.done` when it ends as expected. A rerun skips every step
already done, so a crash costs only the step in progress. A step that ends otherwise records
`<step>.failed` and the run stops there; the log names what failed. Steps marked `expect=1` are
mutant runs: they pass only when the harness fails.

Before every timed step the script checks that no other harness, test binary or attach is running,
and stops if one is. Nothing here commits: the box session reviews results/ and RESULTS.md and commits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
        rc = self.py(log, HERE / "make_mutants.py", REPO / "tools", mdir)
        lines = []
        for mode in ("accept_shift", "offset_swap"):
            out = mdir / f"{mode}-attach.sslm"
            arc = self.run(log, [sys.executable, mdir / mode / "sslm_attach_schema.py", "--base", self.base, "--ckpt",
                                 self.ckpt, *sum((["--schema", s] for s in self.schemas()), []), "--out", out,
                                 "--verifier", self.verifier, "--force"],
                           env_extra={"PYTHONPATH": str(REPO / "tools")}, append=True)
            lines.append(f"attach_copy {mode} exit={arc} written={out.is_file()}")
            direct = mdir / f"{mode}.sslm"
            drc = self.py(log, HERE / "scm1_mutants.py", self.artifact, mode, direct, append=True)
            if drc:
                return drc
            vrc = self.run(log, [self.verifier, direct, mdir / f"{mode}.verify.json"], append=True)
            lines.append(f"direct {mode} sha256={sha256(direct)} verifier_exit={vrc}")
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        (RES / "mutants_artifacts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return rc

    def refused_at_map(self, log: Path, artifact: Path, backend: str) -> int:
        """0 only when the harness exits 2 with the backend's own map call named as the refusal."""
        rc = self.harness(log, "walks", artifact, "intents", self.fixtures, backend)
        call = "sslm_model_map" if backend == "cpu" else "sslm_gpu_model_map"
        text = log.read_text(encoding="utf-8", errors="replace")
        ok = rc == 2 and f"setup failed: error:{call} " in text
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"refused_at_map={ok} ({call})\n")
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
        # Plain CPU and GPU: every document valid and the two streams identical (must pass).
        st.append(("gen_plain", False, (0,), lambda L: self.concat(
            RES / "gen_plain.txt", [BOX / "gen_cpu_plain.log", BOX / "gen_gpu_plain.log"])))
        st.append(("gen_check_plain", False, (0,), lambda L: self.py(
            L, HERE / "fixtures.py", "check", "--ckpt", self.ckpt, "--gen", RES / "gen_plain.txt",
            "--json", RES / "gen_check_plain.json")))
        # Every variant against plain CPU: canonical jump-forward differs by design (RESULTS.md), so this
        # report may exit 1; it is the record of which variant differs and where.
        st.append(("gen_check", False, (0, 1), lambda L: self.py(
            L, HERE / "fixtures.py", "check", "--ckpt", self.ckpt, "--gen", RES / "gen.txt",
            "--json", RES / "gen_check.json")))
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
            # The restore-at-state-zero mutant: each run must fail (exit 1).
            st.append((f"state_zero_tf_{b}", False, (1,), lambda L, b=b: self.harness(
                L, "tfrestore", A, S, F, b, TF_PROMPT, "three_intents_0_2_1_conditions", "--state-zero")))
            st.append((f"state_zero_gen_{b}", False, (1,), lambda L, b=b: self.harness(
                L, "restore", A, S, F, RES / "gen.txt", RES / "saveplan.txt", b, "ex041", "--state-zero")))
        st.append(("state_zero", False, (0,), lambda L: self.concat(
            RES / "state_zero.txt", [BOX / f"state_zero_{k}_{b}.log" for b in ("cpu", "gpu") for k in ("tf", "gen")])))
        # The two SCM1 mutants.
        st.append(("mutant_artifacts", False, (0,), self.mutant_artifacts))
        mdir = self.scratch / "mutants"
        for b in ("cpu", "gpu"):
            st.append((f"mutant_accept_shift_walks_{b}", False, (1,), lambda L, b=b: self.harness(
                L, "walks", mdir / "accept_shift.sslm", S, F, b)))
            st.append((f"mutant_accept_shift_gen_{b}", False, (1,), lambda L, b=b: self.harness(
                L, "gen", mdir / "accept_shift.sslm", S, F, b, "plain", "ex005", "ex041")))
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

    def main(self) -> int:
        BOX.mkdir(parents=True, exist_ok=True)
        steps = self.steps()
        names = [s[0] for s in steps]
        if self.args.redo:
            if self.args.redo not in names:
                print(f"unknown step {self.args.redo}; --list shows them")
                return 2
            for n in names[names.index(self.args.redo):] if self.args.redo_after else [self.args.redo]:
                for ext in (".done", ".failed"):
                    (BOX / (n + ext)).unlink(missing_ok=True)
        for name, timed, expect, fn in steps:
            done = BOX / f"{name}.done"
            if self.args.list:
                print(f"{'done   ' if done.is_file() else 'pending'} {name}{'  (timed)' if timed else ''}"
                      f"{'  expect=' + '/'.join(map(str, expect)) if expect != (0,) else ''}")
                continue
            if done.is_file():
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
                      "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
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
    ap.add_argument("--redo", help="forget this step's result before running")
    ap.add_argument("--redo-after", action="store_true", help="with --redo: also forget every later step")
    ap.add_argument("--ignore-busy", action="store_true", help="run timed steps even if another run is seen")
    return Box(ap.parse_args(argv)).main()


if __name__ == "__main__":
    sys.exit(main())
