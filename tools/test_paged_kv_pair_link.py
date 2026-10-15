"""Cells 79-16 and 79-18 (paged-KV plan rev 16.29: §29.4, §29.5, §30.1, §30.2, §30.8, §31; D-SLM8472, D-SLM8476,
D-SLM8477, D-SLM8484), written by the test author blind to tools/paged_kv_pair_ma.cpp, against the plan's text.

  79-16 (a)  tools/check_paged_kv_pair_link.py passes on the build's MA: exit 0, `LINKED <exe sha256> superslm.lib
             <library sha256>`; MA's manifest names the production library with the production pair's digest and no
             superslm_mr_opt library; a fixture whose link step carries /implib:superslm_pkv_pair_ma.lib passes (check 4
             counts no /implib: operand)
  79-16 (b)  `superslm_pkv_pair_ma identity ma1` and `identity ma2` print exactly PKV_PAIR_MA1_V1 / PKV_PAIR_MA2_V1
             defs=none (written here literally) and exit 0; any other argument exits non-zero
  79-16 (c)  fixture manifests copied from the build's, each refused (exit 3) at its own check: the link step's engine
             library rewritten to superslm_mr_opt.lib (check 4); MA's library digest differing from the production pair's
             (check 5); /DSUPERSLM_MUTANT_PKV_MR_OPT added to a compile command (check 6); invoked[exe] differing from the
             file (check 1); a second .obj on the link step (check 8)
  79-16 (d)  a manifest whose commands entry is null refuses (check 2)
  79-16 (e)  with a fixture registry: a map R79A digest equal to MA's passes, a different one refuses (check 7); the
             tie-only mode does the same for MR-OPT at R79MR
  79-18      the production pair's manifest lists tools/run_paged_kv_rows.py in `sources` and in `invoked`;
             verify_paged_kv_binding.bound() is true for a fixture entry whose instrument is the driver and whose
             source_binding is taken from that manifest, on a clean checkout at the manifest's head; the same entry
             against a manifest without the driver is false; (T-3212 C-7, R0(c) §30.1) the same entry with its
             instrument_sha256 the digest of a different driver, everything else equal, is false

Environment (all required; unset or missing fails the cells, never skips them):
  PKV_BFR              the build directory: B-FR, the Ninja Release build of the fix-round tip (plan §30.3)
  PKV_BFR_REPO         a clean checkout of SuperSLM at the manifests' git_head (79-18)
  PKV_COMMISSION_TOOL  the registrar of the Wizard tree under test, instrument-commission.ps1 (79-18; T-3210 O-5).
                       bound() grades the registrar beside the registry it is handed (SuperSLM cb7dd42, no fallback),
                       so 79-18 copies this file beside each fixture registry
  PKV_BINDING_VERIFIER optional: the verify_paged_kv_binding.py under test (default this checkout's; the test author
                       runs the pre-cb7dd42 verifier through it)
The real link-line mutant and the check-8 mutant (§30.2, §31) are executed once and cited in the test author's record
(T-3189 §17); never committed.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parent
CHECKER = TOOLS / "check_paged_kv_pair_link.py"
BFR = os.environ.get("PKV_BFR", "")
REPO = os.environ.get("PKV_BFR_REPO", "")
REGISTRAR = os.environ.get("PKV_COMMISSION_TOOL", "")
# The main checkout's registrar: the path the verifier before cb7dd42 read whatever registry it was handed. 79-18's two
# sibling pins claim these bytes, so a verifier that still reads this path reads them bound and fails the pins.
MAIN_REGISTRAR = Path("D:/Wizard/Claude/Tooling/instrument-commission.ps1")
MA, PAIR, MR = "superslm_pkv_pair_ma.exe", "superslm_pkv_pair.exe", "superslm_pkv_pair_mr_opt.exe"
LIB = "superslm.lib"
IDENTITY = {"ma1": "PKV_PAIR_MA1_V1 defs=none", "ma2": "PKV_PAIR_MA2_V1 defs=none"}

_spec = importlib.util.spec_from_file_location(
    "pkv_binding", os.environ.get("PKV_BINDING_VERIFIER") or TOOLS / "verify_paged_kv_binding.py")
binding = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(binding)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Build(unittest.TestCase):
    def setUp(self):
        if not BFR:
            self.fail("79-16/79-18 need the build: set PKV_BFR to B-FR's directory (plan §30.3); there is no default")
        self.build = Path(BFR)
        if not (self.build / MA).is_file():
            self.fail(f"PKV_BFR names {self.build}, which holds no {MA}")
        self.temp = tempfile.TemporaryDirectory(prefix="t3189-link-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self, name):
        return json.loads((self.build / f"{name}.source-set.json").read_text(encoding="utf-8"))

    def checker(self, *args):
        done = subprocess.run([sys.executable, "-I", str(CHECKER), *map(str, args)], capture_output=True, text=True,
                              timeout=300)
        return done.returncode, (done.stdout + done.stderr).strip()

    def check_ma(self, exe, registry=None):
        args = ["--exe", exe, "--library-name", LIB, "--reference-manifest", self.build / f"{PAIR}.source-set.json"]
        if registry:
            args += ["--registry", registry, "--id", "SUCCESSOR", "--slot", "R79A"]
        return self.checker(*args)

    def fixture(self, edit):
        """A copy of the build's MA, its library and its manifest in a fresh directory, the manifest changed by `edit`
        (a function of the manifest dict, returning it)."""
        d = Path(tempfile.mkdtemp(dir=self.root))
        for name in (MA, LIB):
            shutil.copyfile(self.build / name, d / name)
        m = edit(self.manifest(MA))
        (d / f"{MA}.source-set.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        return d / MA

    def link_line(self, m):
        lines = [l for l in m["build_recipe"]["commands"][MA].splitlines() if f"/out:{MA}" in l]
        self.assertEqual(len(lines), 1, "the build's MA link step is not one line")
        return lines[0]

    def rewrite(self, old, new, line_of=None):
        def edit(m):
            cmds = m["build_recipe"]["commands"][MA]
            line = line_of(m) if line_of else cmds
            self.assertEqual(line.count(old), 1, f"{old!r} in {line[:200]!r}")
            m["build_recipe"]["commands"][MA] = cmds.replace(line, line.replace(old, new))
            return m
        return edit

    def refused_at(self, exe, check):
        code, out = self.check_ma(exe)
        self.assertEqual(code, 3, out)
        self.assertTrue(out.startswith(f"REFUSED check {check}:"), out)


class Cell7916(Build):
    # (a) the build's MA passes; its manifest ties the production library and names no mutant library
    def test_79_16a_checker_passes_on_the_build(self):
        code, out = self.check_ma(self.build / MA)
        self.assertEqual((code, out), (0, f"LINKED {sha(self.build / MA)} {LIB} {sha(self.build / LIB)}"))
        ma, pair = self.manifest(MA), self.manifest(PAIR)
        self.assertEqual(ma["invoked"][LIB], pair["invoked"][LIB])
        self.assertEqual(ma["invoked"][LIB], sha(self.build / LIB))
        self.assertFalse([k for k in ma["invoked"] if "superslm_mr_opt" in k])
        self.assertNotIn("superslm_mr_opt", self.link_line(ma))

    # (a) D-SLM8484: an /implib: operand naming a superslm-prefixed .lib is not a library input
    def test_79_16a_implib_operand_is_not_a_library_input(self):
        def ensure_implib(m):
            line = self.link_line(m)
            if "/implib:superslm_pkv_pair_ma.lib" not in line:
                cmds = m["build_recipe"]["commands"][MA]
                m["build_recipe"]["commands"][MA] = cmds.replace(
                    line, line.replace(f"/out:{MA}", f"/out:{MA} /implib:superslm_pkv_pair_ma.lib"))
            return m
        exe = self.fixture(ensure_implib)
        self.assertIn("/implib:superslm_pkv_pair_ma.lib", self.link_line(json.loads(
            exe.with_name(f"{MA}.source-set.json").read_text(encoding="utf-8"))))
        code, out = self.check_ma(exe)
        self.assertEqual(code, 0, out)
        self.assertTrue(out.startswith(f"LINKED {sha(exe)} {LIB} "), out)

    # (b) identity mode prints the driver's strings, and only for ma1 and ma2
    def test_79_16b_identity_mode(self):
        for arg, line in IDENTITY.items():
            with self.subTest(arg=arg):
                done = subprocess.run([str(self.build / MA), "identity", arg], capture_output=True, text=True,
                                      timeout=60)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual(done.stdout.strip(), line)
        for args in ([], ["identity"], ["identity", "ma3"], ["identity", "MA1"], ["identity", "production"],
                     ["identity", "mr-opt"], ["identity", "ma1", "extra"], ["ma1"]):
            with self.subTest(args=args):
                done = subprocess.run([str(self.build / MA), *args], capture_output=True, text=True, timeout=60)
                self.assertNotEqual(done.returncode, 0, f"{args}: {done.stdout}")

    # (c) each fixture is refused at its own check
    def test_79_16c_check4_engine_library_rewritten(self):
        self.refused_at(self.fixture(self.rewrite(f" {LIB} ", " superslm_mr_opt.lib ", self.link_line)), 4)

    def test_79_16c_check5_library_digest_differs_from_the_production_pair(self):
        def edit(m):
            m["invoked"][LIB] = "0" * 64
            return m
        self.refused_at(self.fixture(edit), 5)

    def test_79_16c_check6_mutant_definition_on_a_compile(self):
        def compile_line(m):
            # MA's own compile: names its source, compiles (cl /c or -c), and is not the link step
            lines = [l for l in m["build_recipe"]["commands"][MA].splitlines()
                     if "paged_kv_pair_ma.cpp" in l and f"/out:{MA}" not in l and (" /c " in f" {l} " or " -c " in f" {l} ")]
            self.assertTrue(lines, "no compile command for paged_kv_pair_ma.cpp")
            return lines[0]
        self.refused_at(self.fixture(self.rewrite("/nologo", "/nologo /DSUPERSLM_MUTANT_PKV_MR_OPT", compile_line)), 6)

    def test_79_16c_check1_invoked_exe_differs(self):
        def edit(m):
            m["invoked"][MA] = "0" * 64
            return m
        self.refused_at(self.fixture(edit), 1)

    def test_79_16c_check8_second_object_on_the_link_step(self):
        def edit(m):
            line = self.link_line(m)
            obj = next(t for t in line.split() if t.endswith("paged_kv_pair_ma.cpp.obj"))
            return self.rewrite(obj, f"{obj} CMakeFiles\\superslm.dir\\src\\trace_hook.cpp.obj", self.link_line)(m)
        self.refused_at(self.fixture(edit), 8)

    # (d) a null commands entry refuses
    def test_79_16d_null_commands_entry(self):
        def edit(m):
            m["build_recipe"]["commands"][MA] = None
            return m
        self.refused_at(self.fixture(edit), 2)

    # (e) the registry tie, and the tie-only mode for MR-OPT
    def test_79_16e_registry_tie(self):
        registry = self.root / "registry.json"
        for name, r79a, r79mr, want in (("equal", sha(self.build / MA), sha(self.build / MR), 0),
                                        ("different", "1f" * 32, "2e" * 32, 3)):
            with self.subTest(map=name):
                registry.write_text(json.dumps({"SUCCESSOR": {"source_binding": {
                    "construction_executable_sha256": {"R79A": r79a, "R79MR": r79mr}}}}))
                code, out = self.check_ma(self.build / MA, registry)
                self.assertEqual(code, want, out)
                if want:
                    self.assertTrue(out.startswith("REFUSED check 7:"), out)
                code, out = self.checker("--tie-only", "--exe", self.build / MR, "--registry", registry,
                                         "--id", "SUCCESSOR", "--slot", "R79MR")
                self.assertEqual(code, want, out)
                if want:
                    self.assertTrue(out.startswith("REFUSED check 7:"), out)


class Cell7918(Build):
    def setUp(self):
        super().setUp()
        if not REPO or not (Path(REPO) / ".git").exists():
            self.fail("79-18 needs PKV_BFR_REPO: a clean checkout of SuperSLM at the manifest's git_head")
        self.repo = Path(REPO)
        if not REGISTRAR or not Path(REGISTRAR).is_file():
            self.fail("79-18 needs PKV_COMMISSION_TOOL: the instrument-commission.ps1 of the Wizard tree under test "
                      "(T-3210 O-5); there is no default")
        self.registrar = Path(REGISTRAR)

    def bound(self, binary, registry):
        return binding.bound(binary, self.repo, registry, "SUCCESSOR")[0]

    def entry(self, manifest_path, claim=None, sibling=b"registrar", instrument_sha256=None):
        """A fixture registry in its own directory naming the driver, its source_binding taken from `manifest_path`, its
        verifier_sha256 the hash of `claim` (default PKV_COMMISSION_TOOL). Beside it, instrument-commission.ps1 holds
        `sibling`: the claimed registrar's bytes by default, other bytes when given, or nothing when None. Its
        instrument_sha256 is the manifest's digest for the driver unless `instrument_sha256` is given."""
        claim = Path(claim or self.registrar)
        m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        driver = self.repo / "tools" / "run_paged_kv_rows.py"
        d = Path(tempfile.mkdtemp(dir=self.root))
        registry = d / "registry.json"
        registry.write_text(json.dumps({"SUCCESSOR": {
            "instrument": str(driver),
            "instrument_sha256": instrument_sha256 or m["sources"].get("tools/run_paged_kv_rows.py")
                                 or binding.sha(driver, normalized=True),
            "source_binding": {"manifest_sha256": sha(manifest_path), "git_head": m["git_head"],
                               "git_tree": m["git_tree"], "build_recipe_sha256": binding.canonical(m["build_recipe"]),
                               "measurement_env": binding.measurement_env(),
                               "verifier_sha256": sha(claim)}}}))
        if sibling == b"registrar":
            sibling = claim.read_bytes()
        if sibling is not None:
            (d / "instrument-commission.ps1").write_bytes(sibling)
        return registry

    def test_79_18_driver_in_the_production_manifest(self):
        m = self.manifest(PAIR)
        self.assertIn("tools/run_paged_kv_rows.py", m["sources"])
        self.assertIn("run_paged_kv_rows.py", m["invoked"])

    # cb7dd42: bound() grades the registrar beside the registry it is handed, with no fallback. Both members claim the
    # main checkout's registrar, the one the earlier verifier read whatever the registry: that verifier reads them bound
    def test_79_18_registrar_is_the_registrys_sibling(self):
        if not MAIN_REGISTRAR.is_file():
            self.fail(f"{MAIN_REGISTRAR} is the claim these members name; it does not exist here")
        manifest = self.build / f"{PAIR}.source-set.json"
        with self.subTest(sibling="one byte different"):
            registry = self.entry(manifest, MAIN_REGISTRAR, MAIN_REGISTRAR.read_bytes() + b"#")
            self.assertFalse(self.bound(self.build / PAIR, registry),
                             "bound() read a registrar other than the registry's sibling")
        with self.subTest(sibling="missing"):
            registry = self.entry(manifest, MAIN_REGISTRAR, None)
            self.assertFalse(self.bound(self.build / PAIR, registry),
                             "bound() fell back to a registrar when none sits beside the registry")

    def test_79_18_bound_with_the_driver_and_not_without(self):
        head = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"], capture_output=True, text=True)
        self.assertEqual(head.stdout.strip(), self.manifest(PAIR)["git_head"], "PKV_BFR_REPO is not at the head")
        manifest = self.build / f"{PAIR}.source-set.json"
        self.assertTrue(self.bound(self.build / PAIR, self.entry(manifest)),
                        "bound() is false for the driver entry with the tree's registrar beside the registry")
        d = Path(tempfile.mkdtemp(dir=self.root))
        for name in (PAIR, LIB):
            shutil.copyfile(self.build / name, d / name)
        m = self.manifest(PAIR)
        m["sources"].pop("tools/run_paged_kv_rows.py", None)
        m["invoked"].pop("run_paged_kv_rows.py", None)
        (d / f"{PAIR}.source-set.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        self.assertFalse(self.bound(d / PAIR, self.entry(d / f"{PAIR}.source-set.json")),
                         "bound() is true against a manifest without the driver")

    # T-3212 C-7 (R0(c), plan §30.1): the registry entry's instrument_sha256 must equal the manifest's digest for the
    # instrument. The member differs from the bound entry in that one field: the registered digest of a different
    # driver (this driver's normalized bytes with one line appended). It kills the comparison dropped with the lookup
    # kept (D:/_scratch/T-3212/vmut/verify_drop_cmp.py), which every other cell passes
    def test_79_18_instrument_digest_differing_from_the_manifest_is_unbound(self):
        manifest = self.build / f"{PAIR}.source-set.json"
        listed = self.manifest(PAIR)["sources"]["tools/run_paged_kv_rows.py"]
        driver = self.repo / "tools" / "run_paged_kv_rows.py"
        other = hashlib.sha256(driver.read_bytes().replace(b"\r\n", b"\n") + b"# a different driver\n").hexdigest()
        self.assertNotEqual(other, listed)
        with self.subTest(instrument_sha256="the manifest's (control)"):
            self.assertTrue(self.bound(self.build / PAIR, self.entry(manifest, instrument_sha256=listed)),
                            "bound() is false for the entry whose instrument digest is the manifest's")
        with self.subTest(instrument_sha256="a different driver's"):
            self.assertFalse(self.bound(self.build / PAIR, self.entry(manifest, instrument_sha256=other)),
                             "bound() is true for an entry whose instrument_sha256 is not the manifest's digest for "
                             "the instrument")


if __name__ == "__main__":
    unittest.main()
