"""C6 7.9/7.9b construction and refusal cells on synthetic pair records."""
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("run_paged_kv_pairs.py")
spec = importlib.util.spec_from_file_location("paged_kv_pairs", SCRIPT)
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


class PairCells(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="t3112-pairs-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def rows(self, count=508, token_mismatch=False):
        path = self.root / "pairs.csv"
        path.write_text("".join(f"{i},{'A' if i % 2 else 'B'},100,100,12,{13 if token_mismatch and i == 0 else 12}\n"
                                for i in range(count)))
        return path

    def test_79_and_79b_must_accept_and_reject_statistic(self):
        healthy = pair.grade([0.0] * 508)
        self.assertEqual((healthy["effect"], healthy["bar"]), ("NO RESULT", "PASS"))
        mutant = pair.grade([.15] * 508)
        self.assertEqual((mutant["effect"], mutant["bar"]), ("RESOLVED", "FAIL"))
        self.assertEqual(mutant["sign_interval"], [.15, .15])
        self.assertEqual(pair.grade([0.0] * 2)["status"], "MECHANISM_ONLY")

    def test_79_12_boundary_and_exact_rank_intervals(self):
        next_bar = math.nextafter(.05, math.inf)
        for value, bar in ((0.0, "PASS"), (.04, "PASS"), (.05, "PASS"),
                           (next_bar, "FAIL"), (.06, "FAIL"), (.15, "FAIL")):
            self.assertEqual(pair.grade([value] * 508)["bar"], bar)
        # These vectors distinguish ranks 229 and 280 from min/max, median,
        # neighbouring ranks, and an interval that only touches the bar.
        values = [.04] * 228 + [.05] * 52 + [.06] * 228
        self.assertEqual(pair.grade(values)["sign_interval"], [.05, .05])
        self.assertEqual(pair.grade(values)["bar"], "PASS")
        self.assertEqual(pair.grade([.04] * 228 + [.06] * 280)["bar"], "FAIL")
        self.assertEqual(pair.grade([.04] * 229 + [.06] * 279)["bar"], "NO RESULT")
        self.assertEqual(pair.grade([-.02] * 280 + [.01] * 228)["effect"], "RESOLVED")
        self.assertEqual(pair.grade([-.02] * 279 + [.01] * 229)["effect"], "NO RESULT")

    def test_79_1_to_79_5_seeded_populations_use_actual_distribution(self):
        # Reproducible per-pair variation, including a population that straddles
        # the product bar. The oracle derives each order statistic independently.
        for seed, center, spread, expected in (
            (7901, 0.00, .01, ("PASS", "NO RESULT")),
            (7902, 0.04, .005, ("PASS", "RESOLVED")),
            (7903, 0.05, .02, ("NO RESULT", "RESOLVED")),
            (7904, 0.06, .005, ("FAIL", "RESOLVED")),
            (7905, -.02, .005, ("PASS", "RESOLVED")),
        ):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                values = [center + rng.uniform(-spread, spread) for _ in range(508)]
                ordered = sorted(values)
                result = pair.grade(values)
                self.assertEqual(result["sign_interval"], [ordered[228], ordered[279]])
                self.assertEqual((result["bar"], result["effect"]), expected)

    def test_79_orientation_from_nonconstant_pair_file(self):
        rows = self.root / "orientation.csv"
        rows.write_text("".join(f"{i},{'A' if i & 1 else 'B'},100,{95 if i & 1 else 90},12,12\n"
                                for i in range(508)))
        values, _ = pair.parse_pairs(rows, 508)
        self.assertTrue(all(value > 0 for value in values))
        self.assertEqual(pair.grade(values)["bar"], "FAIL")
        rows.write_text("".join(f"{i},{'A' if i & 1 else 'B'},100,{105 if i & 1 else 110},12,12\n"
                                for i in range(508)))
        values, _ = pair.parse_pairs(rows, 508)
        self.assertTrue(all(value < 0 for value in values))
        self.assertEqual(pair.grade(values)["effect"], "RESOLVED")

    def test_79_6_and_79b_5_corrupt_manifest_or_blob_refused(self):
        model = self.root / "model.sslm"
        model.write_bytes(b"model")
        blob = self.root / "piece.blob"
        blob.write_bytes(b"blob")
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"complete": True, "model_sha256": pair.sha(model),
            "stream": {"seed": 79, "width": 530, "prompt_tokens": 18},
            "pieces": [{"file": blob.name, "start": 0, "count": 18, "input_sha256": "-", "sha256": pair.sha(blob)}],
            "final_blob": blob.name, "final_sha256": pair.sha(blob)}))
        self.assertEqual(pair.checked_blob(manifest, model, 530)[0], blob)
        blob.write_bytes(b"flipped")
        with self.assertRaisesRegex(pair.Refusal, "mismatch"):
            pair.checked_blob(manifest, model, 530)
        manifest.write_text('{"complete":')
        with self.assertRaisesRegex(pair.Refusal, "torn"):
            pair.checked_blob(manifest, model, 530)

    def test_79_6_every_multi_piece_manifest_binding(self):
        model = self.root / "model.sslm"
        model.write_bytes(b"model")
        pieces = []
        previous = "-"
        for start in (0, 6, 12):
            blob = self.root / f"piece-{start}.blob"
            blob.write_bytes(f"blob-{start}".encode())
            digest = pair.sha(blob)
            pieces.append({"start": start, "count": 6, "file": blob.name,
                           "input_sha256": previous, "sha256": digest})
            previous = digest
        clean = {"complete": True, "model_sha256": pair.sha(model),
                 "stream": {"seed": 79, "width": 530, "prompt_tokens": 18},
                 "geometry": [4096, 2, 2, 48], "pieces": pieces,
                 "final_blob": pieces[-1]["file"], "final_sha256": previous}
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(clean))
        self.assertEqual(pair.checked_blob(manifest, model, 530, clean["geometry"])[0].name, pieces[-1]["file"])
        for label, change in (
            ("model", lambda m: m.update(model_sha256="bad")),
            ("seed", lambda m: m["stream"].update(seed=80)),
            ("width", lambda m: m["stream"].update(width=531)),
            ("geometry", lambda m: m.update(geometry=[4095, 2, 2, 48])),
            ("chain", lambda m: m["pieces"][1].update(input_sha256="bad")),
            ("previous", lambda m: m["pieces"][2].update(input_sha256="bad")),
            ("complete", lambda m: m.update(complete=False)),
            ("final", lambda m: m.update(final_sha256="bad")),
        ):
            with self.subTest(label=label):
                changed = json.loads(json.dumps(clean))
                change(changed)
                manifest.write_text(json.dumps(changed))
                with self.assertRaisesRegex(pair.Refusal, "mismatch"):
                    pair.checked_blob(manifest, model, 530, clean["geometry"])
        manifest.write_text(json.dumps(clean))
        first_blob = self.root / pieces[0]["file"]
        first_blob.write_bytes(b"corrupt interior blob")
        with self.assertRaisesRegex(pair.Refusal, "mismatch"):
            pair.checked_blob(manifest, model, 530, clean["geometry"])

    def test_prepare_timeout_halves_and_resumes_piece_chain(self):
        model = self.root / "model.sslm"
        model.write_bytes(b"model")
        directory = self.root / "prepared"
        args = SimpleNamespace(exe=self.root / "superslm_pkv_pair.exe", model=model,
                               directory=directory, width=530, cap=4096, layers=2,
                               heads=2, dim=48, piece_limit=8, cap_seconds=1)
        calls = []
        def fake_run(command, log, cap, env):
            count = int(command[9])
            calls.append(count)
            if len(calls) == 1:
                return None
            Path(command[-1]).write_bytes(f"piece-{command[8]}-{count}".encode())
            return 0
        with patch.object(pair, "production_library", return_value=("exe", "lib")), \
             patch.object(pair, "run_logged", side_effect=fake_run):
            manifest = pair.prepare(args)
            self.assertTrue(manifest["complete"])
            self.assertEqual(calls[:2], [8, 4])
            self.assertTrue(all(item["count"] <= 4 for item in manifest["pieces"]))
            self.assertEqual(pair.validate_manifest(manifest, pair.sha(model), 530, directory,
                                                    [4096, 2, 2, 48]), 18)
            self.assertEqual(pair.prepare(args)["final_sha256"], manifest["final_sha256"])

    def test_79_7_and_79b_6_seam_executable_refused(self):
        seam = self.root / "superslm_test_injection.exe"
        seam.write_bytes(b"binary")
        with self.assertRaisesRegex(pair.Refusal, "production"):
            pair.production_library(seam)

    def test_79_7_renamed_injection_binary_with_library_refused(self):
        build = Path("D:/_scratch/T-3112/build-c6")
        fake = self.root / "superslm_pkv_pair.exe"
        shutil.copy2(build / "superslm_pkv_c6.exe", fake)
        shutil.copy2(build / "superslm.lib", self.root / "superslm.lib")
        with self.assertRaisesRegex(pair.Refusal, "identity or build definitions"):
            pair.production_library(fake)
        other = self.root / "renamed_production.exe"
        shutil.copy2(build / "superslm_pkv_pair.exe", other)
        with self.assertRaisesRegex(pair.Refusal, "expected the production"):
            pair.production_library(other)

    def test_79_8_and_79b_7_token_mismatch_refused(self):
        with self.assertRaisesRegex(pair.Refusal, "token-stream mismatch"):
            pair.parse_pairs(self.rows(token_mismatch=True), 508)

    def test_79_invalid_pair_durations_refused(self):
        rows = self.rows()
        for invalid in ("0", "-1", "nan", "inf"):
            with self.subTest(duration=invalid):
                lines = rows.read_text().splitlines()
                lines[0] = f"0,B,{invalid},100,12,12"
                rows.write_text("\n".join(lines) + "\n")
                with self.assertRaisesRegex(pair.Refusal, "invalid duration"):
                    pair.parse_pairs(rows, 508)
                rows = self.rows()

    def test_79_9_and_79b_8_incomplete_pairs_refused(self):
        with self.assertRaisesRegex(pair.Refusal, "incomplete pair set"):
            pair.parse_pairs(self.rows(507), 508)

    def test_79_10_and_79b_9_live_locks_refused(self):
        for name in ("79", "79b"):
            lock = self.root / f"{name}.lock"
            with pair.PidLock(lock):
                with self.assertRaisesRegex(pair.Refusal, "second runner"):
                    with pair.PidLock(lock):
                        pass

    def args(self, result):
        model = self.root / "model.sslm"
        model.write_bytes(b"model")
        return SimpleNamespace(exe=self.root / "superslm_pkv_pair.exe", kind="79b", model_a=model,
                               model_b=None, manifest_a=None, manifest_b=None, result=result,
                               cap_a=4096, cap_b=None, layers=2, heads=2, dim=48, width=1712,
                               pairs=508, seed=7904, cap_seconds=10, mechanism=False)

    def test_79_10_box_global_lock_refuses_different_result_path_and_stale_recovers(self):
        lock = self.root / "global.lock"
        with patch.object(pair, "BOX_LOCK", lock), patch.object(pair, "production_library", return_value=("exe", "lib")):
            with pair.PidLock(lock):
                for name in ("target-a.json", "target-b.json", "shared.json"):
                    with self.assertRaisesRegex(pair.Refusal, "second runner"):
                        pair.run(self.args(self.root / name))
            lock.write_text("99999999")
            with pair.PidLock(lock):
                self.assertTrue(lock.exists())
            self.assertFalse(lock.exists())

    def test_79_9_507_pair_graded_run_refuses(self):
        args = self.args(self.root / "result.json")
        args.pairs = 507
        with self.assertRaisesRegex(pair.Refusal, "exactly 508"):
            pair.run(args)

    def test_79_11_torn_record_is_rerun_and_valid_record_is_read(self):
        result = self.root / "result.json"
        args = self.args(result)
        result.write_text('{"complete":')
        def fake_run(command, log, cap, env):
            Path(command[-1]).write_text("".join(f"{i},A,100,100,12,12\n" for i in range(508)))
            return 0
        with patch.object(pair, "BOX_LOCK", self.root / "global.lock"), \
             patch.object(pair, "production_library", return_value=("exe", "lib")), \
             patch.object(pair, "box_observation", side_effect=[{"cpu_percent": 31, "offenders": []},
                                                           {"cpu_percent": 84, "offenders": ["cl:42"]},
                                                           {"cpu_percent": 32, "offenders": []},
                                                           {"cpu_percent": 85, "offenders": ["cl:43"]}]), \
             patch.object(pair, "run_logged", side_effect=fake_run) as execute:
            first = pair.run(args)
            self.assertEqual(first["status"], "QUARANTINED")
            self.assertEqual((first["load_before"]["cpu_percent"], first["load_after"]["cpu_percent"]), (31, 84))
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(pair.run(args)["status"], "QUARANTINED")
            self.assertEqual(execute.call_count, 1)
            result.with_suffix(".pairs.csv").write_text("torn")
            self.assertEqual(pair.run(args)["status"], "QUARANTINED")
            self.assertEqual(execute.call_count, 2)

    def test_79_run_refuses_507_output_even_when_requested_508(self):
        args = self.args(self.root / "short-output.json")
        def fake_run(command, log, cap, env):
            Path(command[-1]).write_text("".join(f"{i},A,100,100,12,12\n" for i in range(507)))
            return 0
        with patch.object(pair, "BOX_LOCK", self.root / "global.lock"), \
             patch.object(pair, "production_library", return_value=("exe", "lib")), \
             patch.object(pair, "run_logged", side_effect=fake_run):
            with self.assertRaisesRegex(pair.Refusal, "incomplete pair set"):
                pair.run(args)

    def test_mutant_hook_is_recorded_and_cannot_publish_a_graded_verdict(self):
        args = self.args(self.root / "mutant.json")
        def fake_run(command, log, cap, env):
            Path(command[-1]).write_text("".join(f"{i},A,100,90,12,12\n" for i in range(508)))
            return 0
        with patch.object(pair, "BOX_LOCK", self.root / "global.lock"), \
             patch.object(pair, "production_library", return_value=("exe", "lib")), \
             patch.object(pair, "run_logged", side_effect=fake_run), \
             patch.dict(os.environ, {"SUPERSLM_PAGED_KV_79_SLOW_ARM": "A"}):
            record = pair.run(args)
        self.assertEqual((record["hook"], record["status"]), ("A", "MUTANT_CONSTRUCTION"))

    def test_independent_default_seeds_are_recorded(self):
        args = self.args(self.root / "seeded.json")
        args.seed = None
        def fake_run(command, log, cap, env):
            Path(command[-1]).write_text("".join(f"{i},A,100,100,12,12\n" for i in range(508)))
            return 0
        with patch.object(pair, "BOX_LOCK", self.root / "global.lock"), \
             patch.object(pair, "production_library", return_value=("exe", "lib")), \
             patch.object(pair, "run_logged", side_effect=fake_run), \
             patch.object(pair, "box_observation", return_value={"cpu_percent": 1}), \
             patch.object(pair.secrets, "randbits", side_effect=[7904, 7905]):
            first, second = pair.run(args), pair.run(args)
        self.assertEqual((first["seed"], second["seed"]), (7904, 7905))

    def test_load_generator_record_hash_invalidates_reuse(self):
        args = self.args(self.root / "loaded.json")
        args.load_record = self.root / "load.json"
        load_state = {"form": "square-0.1", "pid": os.getpid(),
                      "worker_pids": [os.getpid()], "phase": "on", "complete": False,
                      "started_monotonic": 1.0, "calibration": None}
        args.load_record.write_text(json.dumps(load_state), encoding="utf-8")
        def fake_run(command, log, cap, env):
            Path(command[-1]).write_text("".join(f"{i},A,100,100,12,12\n" for i in range(508)))
            return 0
        with patch.object(pair, "BOX_LOCK", self.root / "global.lock"), \
             patch.object(pair, "production_library", return_value=("exe", "lib")), \
             patch.object(pair, "run_logged", side_effect=fake_run) as execute, \
             patch.object(pair, "box_observation", return_value={"cpu_percent": 1}):
            first = pair.run(args)
            self.assertEqual(first["load_sha256"], pair.sha(args.load_record))
            self.assertEqual(first["status"], "QUARANTINED")
            load_state["form"] = "square-1"
            args.load_record.write_text(json.dumps(load_state), encoding="utf-8")
            second = pair.run(args)
            self.assertNotEqual(first["load_sha256"], second["load_sha256"])
            self.assertEqual(second["load_record_after_sha256"], pair.sha(args.load_record))
            self.assertEqual(execute.call_count, 2)

    def test_79_14_dead_or_inactive_load_generator_cannot_authorize(self):
        args = self.args(self.root / "load-bound.json")
        args.load_record = self.root / "load.json"
        for owner, phase in ((99999999, "on"), (os.getpid(), "off")):
            with self.subTest(owner=owner, phase=phase):
                args.load_record.write_text(json.dumps({
                    "form": "square-0.1", "pid": owner, "phase": phase,
                    "complete": True}), encoding="utf-8")
                with patch.object(pair, "BOX_LOCK", self.root / "global.lock"), \
                     patch.object(pair, "production_library", return_value=("exe", "lib")), \
                     patch.object(pair, "box_observation", return_value={"cpu_percent": 1}), \
                     patch.object(pair, "run_logged", side_effect=AssertionError(
                         "stale or inactive load record authorized a production pair")):
                    with self.assertRaisesRegex(pair.Refusal, "load generator.*(dead|inactive|phase)"):
                        pair.run(args)

    def test_79_11_and_79b_10_torn_results_are_missing(self):
        for name in ("79", "79b"):
            result = self.root / f"{name}.json"
            result.write_text('{"complete":')
            self.assertIsNone(pair.load_record(result))
            result.write_text(json.dumps({"complete": True, "kind": name}))
            self.assertIsNone(pair.load_record(result))


if __name__ == "__main__":
    unittest.main()
