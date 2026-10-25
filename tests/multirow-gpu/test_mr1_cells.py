"""Multi-row GPU plan (rev 1.3.2), slice MR1, cloud cells 4.3, 4.4, 4.5 and 4.7, and the mutants the plan maps
onto them (MX14, MX15 -> 4.7). Host-only: Linux dxc, python3, mingw-w64 g++.

Run:  python3 -m unittest tests/multirow-gpu/test_mr1_cells.py -v     (or pytest on the same file)
Env:  DXC (default: dxc on PATH, else /opt/dxc/bin/dxc), MR_MINGW_CXX, MR1_SCRATCH (temp root).

Red state before the port: every cell fails in discovery with "red-unimplemented: no gen_wide_shaders.py under
src/gpu" -- the generator, the widened shaders and the host recorder do not exist yet. The instrument checks in
test_mr1_lint_population.py are green before the port (they grade the lint, not the port).

Builder-facing notes:
  - widened files live beside gen_wide_shaders.py, outside src/gpu/shaders (CMake globs that directory);
  - a generated file carries GENERATED in its first three lines; proj_gemm_wide.hlsl is hand-written;
  - variants compile with -D SSLM_WIDE_H=2|4|8, and files mentioning SSLM_GEMM_SITE also with SSLM_GEMM_SITE=1..6.
"""
from __future__ import annotations

import filecmp
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mr1_wide as m  # noqa: E402


class _Base(unittest.TestCase):
    ws: m.WideSet

    @classmethod
    def setUpClass(cls):
        cls.ws = None
        cls.err = None
        try:
            cls.ws = m.discover()
        except m.MR1Red as e:
            cls.err = str(e)

    def need(self) -> m.WideSet:
        if self.ws is None:
            self.fail(self.err)
        return self.ws


class Cell_4_3_Compile(_Base):
    """4.3 [MR1 cloud]: every widened shader compiles at cs_6_2 -O3 -HV 2018 -WX for H = 2/4/8, and the D3D12 C++
    type-checks under mingw-w64."""

    def test_4_3a_every_site_has_a_widened_shader(self):
        ws = self.need()
        missing = m.expected_names_missing(ws)
        self.assertEqual(missing, [], f"widened shaders missing for: {missing}")

    def test_4_3b_dxc_cs_6_2_WX_at_H_2_4_8(self):
        ws = self.need()
        out = m.scratch_dir("mr1-4.3-")
        res = m.compile_all(ws, out, disasm=False)
        bad = [f"{c.src.name} [{c.tag}]: {c.log.strip()[:400]}" for c in res if not c.ok]
        self.assertTrue(res, "nothing compiled")
        self.assertEqual(bad, [], "\n".join(bad))
        # Every H variant was compiled for every file (no silent skip).
        for f in ws.files:
            tags = {c.tag for c in res if c.src == f}
            for h in m.VARIANTS:
                self.assertTrue(any(t.startswith(f"H{h}") for t in tags), f"{f.name}: H={h} not compiled")

    def test_4_3c_d3d12_cpp_typechecks_under_mingw(self):
        ws = self.need()
        tus = m.host_translation_units(ws.gpu_root)
        self.assertTrue(tus, "red-unimplemented: no host translation unit binds proj_gemm_wide (the recorder that "
                             "dispatches the widened PSOs)")
        for tu in tus:
            ok, log = m.mingw_typecheck(tu)
            self.assertTrue(ok, f"{tu}: mingw-w64 -fsyntax-only failed:\n{log[:3000]}")


class Cell_4_4_GeneratorDeterminism(_Base):
    """4.4 [MR1 cloud]: two regenerations give byte-identical .cso, and the source diff shows only the substitutions.
    Also §3.1: the committed output equals a fresh regeneration (never hand-edited)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.copies = None
        if cls.ws is not None:
            try:
                a = m.regenerate(m.copy_gpu_tree(cls.ws, m.scratch_dir("mr1-4.4a-")))
                b = m.regenerate(m.copy_gpu_tree(cls.ws, m.scratch_dir("mr1-4.4b-")))
                cls.copies = (a, b)
            except m.MR1Red as e:
                cls.err = str(e)

    def pair(self):
        self.need()
        if self.copies is None:
            self.fail(self.err)
        return self.copies

    def test_4_4a_generator_recreates_every_generated_file(self):
        ws = self.need()
        self.assertTrue(ws.generated, "no committed file carries the GENERATED header: nothing is generated")
        a, _ = self.pair()
        names = sorted(p.name for p in a.generated)
        self.assertEqual(names, sorted(p.name for p in ws.generated),
                         "regeneration does not produce exactly the committed generated set")

    def test_4_4b_committed_output_equals_a_regeneration(self):
        ws = self.need()
        a, _ = self.pair()
        diff = [p.name for p in ws.generated if not filecmp.cmp(p, a.wide_dir / p.name, shallow=False)]
        self.assertEqual(diff, [], f"committed generated files differ from a fresh regeneration (hand-edited, or the "
                                   f"generator is stale against the base shaders): {diff}")

    def test_4_4c_two_regenerations_give_identical_sources_and_cso(self):
        a, b = self.pair()
        src_diff = [p.name for p in a.generated if not filecmp.cmp(p, b.wide_dir / p.name, shallow=False)]
        self.assertEqual(src_diff, [], f"regenerated sources differ between runs: {src_diff}")
        ca = m.compile_all(a, a.gpu_root.parent.parent / "cso", disasm=False)
        cb = m.compile_all(b, b.gpu_root.parent.parent / "cso", disasm=False)
        self.assertTrue(ca and all(c.ok for c in ca) and all(c.ok for c in cb),
                        "a regenerated shader does not compile (4.3 names which)")
        by = {(c.src.name, c.tag): c for c in cb}
        bad = [f"{c.src.name} [{c.tag}]" for c in ca
               if c.cso.read_bytes() != by[(c.src.name, c.tag)].cso.read_bytes()]
        self.assertEqual(bad, [], f".cso differs between two regenerations: {bad}")

    def test_4_4d_source_diff_shows_only_the_substitutions(self):
        ws = self.need()
        report = []
        for g in ws.generated:
            base = ws.base_dir / g.name.replace("_wide.hlsl", ".hlsl")
            if not base.is_file():
                report.append(f"{g.name}: no base shader {base.name}")
                continue
            v = m.substitution_violations(base.read_text(encoding="utf-8"), g.read_text(encoding="utf-8"))
            report += [f"{g.name}: {x}" for x in v]
        self.assertEqual(report, [], "\n".join(report[:60]))


class Cell_4_5_Groupshared(_Base):
    """4.5 [MR1 cloud]: groupshared at H = 8 is <= 32,768 bytes, read from the compiled DXIL."""

    def test_4_5_groupshared_at_H8(self):
        ws = self.need()
        out = m.scratch_dir("mr1-4.5-")
        over, seen = [], 0
        for f in ws.files:
            for tag, extra in m.variants_of(f):
                if not tag.startswith("H8"):
                    continue
                c = m.compile_one(ws, f, tag, extra, out)
                self.assertTrue(c.ok, f"{f.name} [{tag}] does not compile: {c.log[:400]}")
                seen += 1
                n = m.groupshared_bytes(c.disasm)
                if n > m.GROUPSHARED_LIMIT:
                    over.append(f"{f.name} [{tag}]: {n} bytes")
        self.assertGreater(seen, 0, "no H = 8 variant compiled")
        self.assertEqual(over, [], f"groupshared over {m.GROUPSHARED_LIMIT} bytes at H = 8: {over}")


class Cell_4_7_WaveLint(_Base):
    """4.7 [MR1 cloud]: wave lint over every widened shader (and its includes) and over the host lane plan."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rep = None
        if cls.ws is not None:
            compiled = m.compile_all(cls.ws, m.scratch_dir("mr1-4.7-"))
            cls.compile_failures = [f"{c.src.name} [{c.tag}]" for c in compiled if not c.ok]
            cls.rep = m.lint_all(cls.ws, compiled)

    def report(self) -> m.LintReport:
        self.need()
        return self.rep

    def test_4_7a_no_wave_or_quad_intrinsic(self):
        r = self.report()
        self.assertEqual(r.wave_calls, [], "\n".join(r.wave_calls))

    def test_4_7b_no_lane_count_literal(self):
        r = self.report()
        self.assertEqual(r.lane_literals, [], "\n".join(r.lane_literals))

    def test_4_7c_barrier_after_every_tree_step(self):
        r = self.report()
        self.assertEqual(r.barriers, [], "\n".join(r.barriers))

    def test_4_7d_lanes_only_from_ComputeGpuGemmSiteGroupPlan(self):
        r = self.report()
        self.assertEqual(r.host, [], "\n".join(r.host))

    def test_4_7e_dxc_reflection_shows_no_wave_feature(self):
        r = self.report()
        self.assertEqual(self.compile_failures, [], f"not compiled, so not reflected: {self.compile_failures}")
        self.assertEqual(r.flags, [], "\n".join(r.flags))


class Mutants_4_7(_Base):
    """Guard vitality for 4.7 (plan §7 dim 11): MX14 and MX15, applied to the port's widened GEMM, turn the lint red
    through the rule each targets."""

    def test_MX14_WaveActiveSum_in_a_widened_tree_is_red(self):
        ws = self.need()
        d = m.scratch_dir("mr1-mx14-")
        c = m.apply_mutant(ws, "MX14", d)
        gemm = c.wide_dir / m.WIDE_GEMM
        comp = [m.compile_one(c, gemm, t, e, d / "cso") for t, e in m.variants_of(gemm)]
        rep = m.lint_all(c, comp)
        self.assertTrue(rep.wave_calls, "MX14 survived the source lint (4.7a)")
        self.assertTrue(rep.flags, "MX14 survived the dxc reflection check (4.7e)")

    def test_MX15_barriers_removed_from_last_log2_32_steps_is_red(self):
        ws = self.need()
        c = m.apply_mutant(ws, "MX15", m.scratch_dir("mr1-mx15-"))
        rep = m.lint_all(c, None)
        self.assertTrue(any(m.WIDE_GEMM in b for b in rep.barriers), "MX15 survived the barrier lint (4.7c)")


if __name__ == "__main__":
    unittest.main()
