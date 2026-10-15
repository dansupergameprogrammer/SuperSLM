"""Executed subprocess constructions for 7.4 infrastructure refusals."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
import csv
import json
import statistics

# Required, with no default (the 1.12.0 review M-2): a default naming an old scratch build ran these cells against
# that stale build whenever it was present. Unset or missing fails every cell, never skips it, as
# .github/workflows/tests.yml says of the paged-KV tool tests it leaves to the build box.
BUILD_DIR = os.environ.get("PKV_C6_BUILD", "")
FIXTURES_DIR = os.environ.get("PKV_C6_FIXTURES", "")
BUILD = Path(BUILD_DIR)
FIXTURES = Path(FIXTURES_DIR)

class RefusalCells(unittest.TestCase):
    def setUp(self):
        if not BUILD_DIR:
            self.fail("PKV_C6_BUILD is unset: name the paged-KV C6 build's directory; there is no default")
        if not (BUILD / "superslm_pkv_c6.exe").is_file():
            self.fail(f"PKV_C6_BUILD names {BUILD_DIR}, which holds no superslm_pkv_c6.exe")
        if not FIXTURES_DIR or not FIXTURES.is_dir():
            self.fail("PKV_C6_FIXTURES is unset or not a directory: name the generated fixture directory "
                      "(tools/gen_paged_kv_fixture.py); there is no default")

    def call(self, binary="superslm_pkv_c6.exe", cell="7.4/C6:fixtures", **settings):
        env = os.environ.copy()
        env["SUPERSLM_PAGED_KV_FIXTURE_DIR"] = str(FIXTURES)
        for key, value in settings.items():
            env["SUPERSLM_PAGED_KV_74_" + key] = value
        return subprocess.run([str(BUILD / binary), "=" + cell], env=env,
                              capture_output=True, text=True, timeout=90)

    def check_refusal(self, code, message, **settings):
        result = self.call(**settings)
        self.assertNotEqual(result.returncode, 76,
                            "7.4t diagnostic refusal blocked the direct 7.4 cell")
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)

    def test_74_5_geometry_refuses_before_cap(self):
        self.check_refusal(71, "geometry mismatch", FIXTURE_LARGE="pkv_qk")

    def test_74_6_wrong_cap(self):
        # pkv_def as the large side: the same geometry, so it passes 71; pinnable to 16 (16 divides 4,096), so it passes
        # the pin; and its cap is 4,096, not 32,768, so it reaches 72. pkv_odd (cap 4,100) is refused at the pin (71)
        # since T1, before the cap check (T-3198 §15).
        self.check_refusal(72, "wrong cap", FIXTURE_LARGE="pkv_def")

    def test_74_7_cpuid_and_pin_refusals(self):
        result = self.call(binary="superslm_pkv_74_no_tsc.exe")
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertIn("invariant TSC absent", result.stderr)
        self.check_refusal(73, "cannot pin timing thread", FORCE_PIN_FAILURE="1")

    def test_74_8_corrected_median_floor(self):
        self.check_refusal(74, "corrected median under 100 ticks", FORCE_FLOOR="1")

    def test_74_10_floor_at_99_and_control_at_101(self):
        below = self.call(FLOOR_CONTROL_TICKS="99")
        self.assertEqual(below.returncode, 74, below.stdout + below.stderr)
        self.assertIn("corrected median under 100 ticks", below.stderr)
        above = self.call(FLOOR_CONTROL_TICKS="101")
        self.assertNotIn(above.returncode, (74, 76), above.stdout + above.stderr)
        self.assertIn("7.4 budget", above.stdout)

    def test_74_11_reset_hook_has_sevenfold_extra_cost(self):
        result = self.call(RESET_MUTANT="1")
        self.assertNotEqual(result.returncode, 76, result.stdout + result.stderr)
        self.assertIn("7.4 budget reset r(86)=", result.stdout)
        rows = re.findall(r"7\.4 hook reset base_ticks=([0-9.]+) hooked_ticks=([0-9.]+)",
                          result.stdout)
        self.assertEqual(len(rows), 2, result.stdout)
        for base, hooked in rows:
            base, hooked = float(base), float(hooked)
            self.assertGreater(base, 0)
            self.assertGreaterEqual((hooked - base) / base, 5.0,
                                    "reset hook must inject the specified 7× extra cost")
            self.assertLessEqual((hooked - base) / base, 9.0)

    def test_74_12_pair_order_random_under_first_call_load(self):
        with tempfile.TemporaryDirectory(prefix="pkv-74-order-") as temp:
            trace = Path(temp) / "order.csv"
            result = self.call(ORDER_TRACE=str(trace), ORDER_BIAS_TICKS="1000")
            self.assertNotEqual(result.returncode, 76, result.stdout + result.stderr)
            self.assertTrue(trace.is_file(), result.stdout + result.stderr)
            with trace.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for holder in ("budget", "whole_reserve"):
                arm = [row for row in rows if row["holder"] == holder]
                self.assertEqual(len(arm), 205)
                self.assertEqual({int(row["pair"]) for row in arm}, set(range(205)))
                self.assertEqual({int(row["bias_ticks"]) for row in arm}, {1000})
                counts = {first: sum(row["first"] == first for row in arm)
                          for first in ("small", "large")}
                self.assertGreaterEqual(min(counts.values()), 60, counts)
                self.assertLessEqual(max(counts.values()), 145, counts)

    def test_74_12_order_changes_measured_adopt_ratio(self):
        # The trace alone can lie: a fixed-call-order mutant can still print
        # the RNG choice. The grader's own rank interval must see both sides
        # of the first-call bias in the measured pair ratios.
        with tempfile.TemporaryDirectory(prefix="pkv-74-order-effect-") as temp:
            trace = Path(temp) / "order.csv"
            result = self.call(ORDER_TRACE=str(trace), ORDER_BIAS_TICKS="1000000")
            self.assertNotIn(result.returncode, (71, 73, 74, 76), result.stdout + result.stderr)
            self.assertTrue(trace.is_file(), result.stdout + result.stderr)
            with trace.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for holder in ("budget", "whole_reserve"):
                arm = [row for row in rows if row["holder"] == holder]
                self.assertEqual(len(arm), 205)
                choices = {name: sum(row["first"] == name for row in arm)
                           for name in ("small", "large")}
                self.assertGreaterEqual(min(choices.values()), 60, choices)
                match = re.search(rf"7\.4 {holder} adopt r\(86\)=([0-9.]+) "
                                  rf"r\(120\)=([0-9.]+) bound=2\.0 (\S+(?: RESULT)?)",
                                  result.stdout)
                self.assertIsNotNone(match, result.stdout + result.stderr)
                lower, upper = float(match.group(1)), float(match.group(2))
                print(f"P-13 {holder}: first={choices} r(86)={lower} r(120)={upper}", flush=True)
                self.assertLess(lower, 0.5, "large-first bias must not own every pair")
                self.assertGreater(upper, 4.0, "small-first bias must not own every pair")

    def test_74_13_every_timed_call_subtracts_independent_empty_median(self):
        with tempfile.TemporaryDirectory(prefix="pkv-74-calls-") as temp:
            trace = Path(temp) / "calls.json"
            result = self.call(CALL_TRACE=str(trace))
            self.assertNotEqual(result.returncode, 76, result.stdout + result.stderr)
            self.assertTrue(trace.is_file(), result.stdout + result.stderr)
            data = json.loads(trace.read_text(encoding="utf-8"))
            empty = data["raw_empty_ticks"]
            self.assertEqual(len(empty), 21)
            reference = statistics.median(empty)
            self.assertGreater(reference, 0)
            calls = [row for row in data["calls"] if row["pair"] >= 0]
            self.assertEqual(len(calls), 2 * 205 * 2 * 2)
            self.assertEqual({row["verb"] for row in calls}, {"reset", "adopt"})
            self.assertEqual({row["holder"] for row in calls}, {"budget", "whole_reserve"})
            for row in calls:
                self.assertAlmostEqual(row["corrected_ticks"], row["raw_ticks"] - reference,
                                       delta=1, msg=row)

    def test_74_14_direct_hooked_cell_declares_quarantine(self):
        for hook in ("ADOPT", "RESET"):
            with self.subTest(hook=hook):
                result = self.call(**{hook + "_MUTANT": "1"})
                self.assertNotEqual(result.returncode, 76, result.stdout + result.stderr)
                self.assertIn(f"7.4 hook_state={hook.lower()}=1", result.stdout)
                self.assertIn("publication=quarantined", result.stdout)
                self.assertIn("7.4 budget", result.stdout)

    def test_timer_vitality_and_unresolved_refusals(self):
        self.check_refusal(75, "sign interval did not resolve", FORCE_UNRESOLVED="1")

    def test_p2_fail_stands_when_later_interval_is_unresolved(self):
        # A real Run74 fixture invocation produces FAIL rows, then reaches
        # Refuse(75). Its verdict must remain the already-observed cell FAIL.
        result = self.call(RESET_MUTANT="1", FORCE_UNRESOLVED="1")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        for holder in ("budget", "whole_reserve"):
            self.assertRegex(result.stdout,
                             rf"7\.4 {holder} reset r\(86\)=[0-9.]+ r\(120\)=[0-9.]+ bound=2\.0 FAIL")
        self.assertIn("[C6] 7.4/C6 RED", result.stdout)
        self.assertRegex(result.stdout, r"1 cells, 1 red; [0-9]+ checks, [1-9][0-9]* failures")
        self.assertIn("sign interval did not resolve", result.stderr)

    def test_4t_diagnostic_failure_is_reported_without_gating_74(self):
        result = self.call(FORCE_VITALITY_FAILURE="1")
        self.assertNotEqual(result.returncode, 76, result.stdout + result.stderr)
        self.assertIn("7.4t:", result.stdout)
        self.assertIn("INFRASTRUCTURE", result.stdout)
        for holder in ("budget", "whole_reserve"):
            for verb in ("reset", "adopt"):
                self.assertIn(f"7.4 {holder} {verb} r(86)=", result.stdout)

    def test_4t_live_no_subtraction_mutant(self):
        healthy = [self.call(cell="7.4t/C6:live") for _ in range(20)]
        passed = sum(result.returncode == 0 for result in healthy)
        print(f"7.4t diagnostic fixed-L live mechanism: {passed}/20 accepted; no 7.4 gate", flush=True)
        mutant = self.call(binary="superslm_pkv_74_no_subtraction.exe", cell="7.4t/C6:live")
        self.assertEqual(mutant.returncode, 1, mutant.stdout + mutant.stderr)
        self.assertIn("clock.okay", mutant.stderr)

    def test_74_2_and_74_3_per_call_reset_and_adopt_mutants(self):
        for verb in ("ADOPT", "RESET"):
            with self.subTest(verb=verb):
                for attempt in range(3):
                    result = self.call(**{verb + "_MUTANT": "1"})
                    self.assertNotEqual(result.returncode, 76,
                                        "7.4t diagnostic refusal blocked the direct 7.4 cell")
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    for holder in ("budget", "whole_reserve"):
                        self.assertRegex(result.stdout, rf"7\.4 {holder} {verb.lower()} r\(86\)=[0-9.]+ r\(120\)=[0-9.]+ bound=2\.0 FAIL")
                    break
                else:
                    self.fail(f"{verb} hook never reached a graded fixture call in three attempts")

if __name__ == "__main__":
    unittest.main()
