"""The 4.7 wave lint's own population check (StandardsDocument §4: a new structure is validated against a
population before it is trusted). Green before the port: it grades the instrument, not the port.

  must-accept  every shipped one-row shader and header (E21: no wave op, barriered trees) lints clean;
               the spike's widened set (origin/brunel/t2790-prefill-spike) lints clean, shader and host rules;
  must-reject  MX14 and MX15 applied to the spike's proj_gemm_wide.hlsl turn the lint red, each through its rule.

The spike legs skip when the spike ref is not fetched (git fetch origin brunel/t2790-prefill-spike).
The lint was written by the test author; the plan's commissioning rule wants these constructions re-authored by an
independent seat before the lint's verdicts are load-bearing (recorded in the MR1 red-suite record).
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
import io
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mr1_wide as m  # noqa: E402

SPIKE_REF = "origin/brunel/t2790-prefill-spike"


def spike_tree() -> Path | None:
    r = subprocess.run(["git", "archive", SPIKE_REF, "src/gpu"], cwd=m.REPO, capture_output=True)
    if r.returncode != 0:
        return None
    d = m.scratch_dir("mr1-spike-")
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:
        t.extractall(d, filter="data")
    return d / "src" / "gpu"


class MustAcceptShipped(unittest.TestCase):
    def test_shipped_shaders_have_no_wave_op_and_barriered_trees(self):
        files = sorted(m.BASE_SHADERS.glob("*.hlsl")) + sorted(m.BASE_SHADERS.glob("*.hlsli"))
        self.assertGreater(len(files), 30)
        bad = []
        parsed = 0
        for f in files:
            text = f.read_text(encoding="utf-8", errors="replace")
            parsed += len(m.functions(m.strip_comments(text)))
            bad += [f"{f.name}: {v}" for v in m.barrier_violations(text)]
            bad += [f"{f.name}:{no}: wave" for no, ln in enumerate(m.strip_comments(text).split("\n"), 1)
                    if m.WAVE_CALL.search(ln)]
        self.assertGreater(parsed, 100, "the HLSL function parser found too few bodies to be reading the files")
        self.assertEqual(bad, [], "\n".join(bad))


class SpikePopulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = spike_tree()

    def setUp(self):
        if self.root is None:
            self.skipTest(f"{SPIKE_REF} not fetched")
        self.ws = m.discover(self.root)

    def test_must_accept_spike_widened_set(self):
        self.assertEqual(m.expected_names_missing(self.ws), [])
        comp = m.compile_all(self.ws, m.scratch_dir("mr1-spike-cso-"))
        self.assertTrue(all(c.ok for c in comp), [c.src.name for c in comp if not c.ok])
        rep = m.lint_all(self.ws, comp)
        self.assertEqual(rep.red(), [], "\n".join(rep.red()))
        over = [c.src.name for c in comp if c.tag.startswith("H8") and m.groupshared_bytes(c.disasm) > m.GROUPSHARED_LIMIT]
        self.assertEqual(over, [])
        # The generator's diff against its base is only substitutions.
        bad = []
        for g in self.ws.generated:
            base = self.ws.base_dir / g.name.replace("_wide.hlsl", ".hlsl")
            bad += m.substitution_violations(base.read_text(encoding="utf-8"), g.read_text(encoding="utf-8"))
        self.assertEqual(bad, [], "\n".join(bad[:20]))

    def test_must_reject_MX14(self):
        d = m.scratch_dir("mr1-spike-mx14-")
        c = m.apply_mutant(self.ws, "MX14", d)
        gemm = c.wide_dir / m.WIDE_GEMM
        comp = [m.compile_one(c, gemm, t, e, d / "cso") for t, e in m.variants_of(gemm)]
        self.assertTrue(all(x.ok for x in comp), "the MX14 mutant must compile, or the flag leg proves nothing")
        rep = m.lint_all(c, comp)
        self.assertTrue(rep.wave_calls)
        self.assertTrue(rep.flags)
        self.assertEqual(rep.barriers, [])

    def test_must_reject_MX15(self):
        d = m.scratch_dir("mr1-spike-mx15-")
        c = m.apply_mutant(self.ws, "MX15", d)
        gemm = c.wide_dir / m.WIDE_GEMM
        comp = [m.compile_one(c, gemm, t, e, d / "cso") for t, e in m.variants_of(gemm)]
        self.assertTrue(all(x.ok for x in comp), "the MX15 mutant must compile (it is a real kernel a driver could run)")
        rep = m.lint_all(c, comp)
        self.assertTrue(any(m.WIDE_GEMM in b for b in rep.barriers))
        self.assertEqual(rep.wave_calls, [])

    def test_must_reject_L1_literal_lanes(self):
        for lit in m.HOST_LANE_LITERALS:
            with self.subTest(literal=lit):
                c = m.apply_host_lanes_mutant(self.ws, lit, m.scratch_dir("mr1-spike-l1-"))
                self.assertTrue(any("lanes slot" in h for h in m.lint_host(c.gpu_root)))

    def test_must_reject_G_foreign_macros(self):
        g = m.apply_generator_macro_mutant(self.ws, m.scratch_dir("mr1-spike-g-"))
        bad = []
        for f in g.generated:
            base = g.base_dir / f.name.replace("_wide.hlsl", ".hlsl")
            bad += m.substitution_violations(base.read_text(encoding="utf-8"), f.read_text(encoding="utf-8"))
        self.assertTrue(all(any(d in b for b in bad) for d in m.FOREIGN_MACROS))


if __name__ == "__main__":
    unittest.main()
