"""Paged-KV plan rev 16.21 §23.8 step T4: red cells for the five-process 7.9 driver (tools/run_paged_kv_rows.py).

Cells (cell 7.9's text; the 79-series rows of the instrument table):
  79-12   the per-process row rule's exact bar: five process medians at exactly 0.05 PASS, at
          nextafter(0.05, +inf) FAIL; 0.04 PASS and 0.06 FAIL; the mixed cases NO RESULT and FAIL
  79-9    a process with fewer (or more) than 104 pairs is refused
  79-8    a token mismatch in any pair stops the process
  79-10   a second runner is refused while the lock owner lives, with a different result path
  79-11   a torn record reads as missing
  boot    a process on another boot is discarded
  guard   the bound table (n, P, M99) at B = 64 and at 7.9-MR-OPT's B = 16; M's definition; the
          vitality constructions at both page sizes (the T-3146 shuffle passes; ascending ids, the pair
          churn, interleave-5 and the set-concentrating order are rejected); the five id lists agree
  79-6    (rev 16.23, T-3189 G-4) a blob or manifest mismatch in any process: the restored blob's digest, the
          manifest chain, the model binding; refused before restore, its own exit code, no record, row ungraded
  79-7    a seam-library build, checked once per process before timing
  per-process refusals: each kind its own non-zero exit code; a row is never graded on four processes
  T6      (T-3202 remedies, SuperSLM 1478be7; class T6Remedies) F-1 the row record's status and exe_sha256, NO RESULT
          standing, a missing pair executable; F-12 a refused run supersedes an existing record, the rerun limit too;
          F-2 a crash after a seam identity is rerun, a transient crash is rerun; F-11 the RESERVE line recorded; F-9 an
          OS-lock holder mid-recovery refuses a second runner
  T7      (T-3203 remedies, SuperSLM 371dc34; class T7Remedies) N-1 (D-SLM8463) the --construction label in the
          process and row records, a construction at another page size refused before any run, the aggregate slot
          record (R79A from MA-1 R2b/R4a/R4b and MA-2 R4a; R79MR from 7.9-MR-OPT R2b/R4a/R4b) STANDING only when every
          run stands, each of its refusals, a REFUSED run making it REFUSED, its argument errors, and the superseding
          record's construction and page positions; N-8 a malformed PAIR line exits 64, a malformed IDS or RESERVE line
          70, an unexpected driver fault 70 (never a raw crash), each superseding an existing record and writing none
          on a fresh path; D-SLM8465 (class R3Slot) the 7.9b record names slot R3 and an unslotted record is rerun
  T8      (T-3204 remedies, D-SLM8466, SuperSLM 4156b30; class T8Remedies) the stand-in prints each construction's own
          identity, and no IDS line for MA-1 or MA-2; an MA run printing page ids refuses 62; a paged run (production,
          7.9-MR-OPT) with no ids, or an empty IDS line, refuses 62; every construction refuses every other's identity
          (62); MA rows record guard "not applicable (no paged arm)" and grade by the row rule, R79A's aggregate over
          the four standing; argv and identity in every row and process record
  T10     (T-3210 O-4, D-SLM8494, SuperSLM 89d70ec; class T8Remedies, test_t10_*) a production or 7.9-MR-OPT run with
          no RESERVE line refuses 62 with no record; an MA run with none stands. The stand-in's paged constructions
          print RESERVE by default; T6Remedies.test_f11's no-RESERVE member is superseded and now asserts the refusal
  T9      (T-3206 N-2, SuperSLM c4e3bf1; class T8Remedies, test_t9_*) an MA-1 or MA-2 run printing a RESERVE line
          refuses 62 with no record; production and 7.9-MR-OPT runs printing one record it
  loop    (T-3196 F-6) 79-6 to 79-9 through the driver's process loop, as a process: five processes against a
          stand-in pair executable, each refusal injected into process 3 (a blob changed after process 2, a seam
          identity line, a token mismatch at pair 57, 103 pairs); the exit status is EXIT_CODES[kind], the refused
          process writes no record, no row record or verdict exists, processes 1 and 2 keep their records, and the
          blob refusal fires before process 3 runs;
          a healthy row exits 0 with five process records and a graded row record

PKV_ROWS_DRIVER, when set, names the driver file under test instead of tools/run_paged_kv_rows.py (the test author's
scratch reference and its mutants run through it); unset, the shipped driver is what every cell grades. PKV_PAIRS_RUNNER
does the same for tools/run_paged_kv_pairs.py (class R3Slot).
"""
import collections
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(os.environ.get("PKV_ROWS_DRIVER") or Path(__file__).with_name("run_paged_kv_rows.py"))
spec = importlib.util.spec_from_file_location("paged_kv_rows", SCRIPT)
rows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rows)

# Cell 7.9's tables, transcribed (rev 16.21): slot -> (n, P, M99).
PRODUCTION = {"R2a": (21, 64, 3), "R2c": (34, 64, 4), "R2b": (64, 64, 6),
              "R4c": (130, 512, 4), "R4a": (250, 512, 5), "R4b": (294, 512, 5)}
MR_OPT = {"R2b": (256, 256, 6), "R4a": (999, 2048, 5), "R4b": (1175, 2048, 6)}
# Each row's geometry for the set-concentrating order: (runs per page, bytes per position per run).
GEOMETRY = {"R2a": (96, 64), "R2c": (96, 64), "R2b": (96, 64), "R4c": (112, 128), "R4a": (112, 128), "R4b": (112, 128)}
WAY = 1 << 20


# The plan's vitality constructions (Claude/Plans/superslm-paged-kv/probes/scatter_guard.py), as test data.
def proto_shuffle(pool):
    res = list(range(pool))
    x = 3146
    for i in range(pool, 1, -1):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        k = (x >> 8) % i
        res[i - 1], res[k] = res[k], res[i - 1]
    return list(reversed(res))


def pair_churn(pool, seed=7):
    pairs = [(2 * i, 2 * i + 1) for i in range(pool // 2)]
    order = list(range(len(pairs)))
    x = seed
    for i in range(len(order), 1, -1):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        k = (x >> 8) % i
        order[i - 1], order[k] = order[k], order[i - 1]
    stack = []
    for h in order:
        stack.extend(pairs[h])
    return [stack.pop() for _ in range(pool)]


def interleave(pool, k=5):
    return [p for r in range(k) for p in range(r, pool, k)]


def set_concentrating(n, runs, run_bytes, page_bytes, pool):
    mods = collections.defaultdict(list)
    for pid in range(pool):
        mods[(pid * page_bytes) % WAY].append(pid)
    out, used = [], set()
    for j in range(n):
        want = (-(j % runs) * run_bytes) % WAY
        best = min(mods, key=lambda m: (m - want) % WAY)
        cand = [p for p in mods[best] if p not in used] or [p for m in mods for p in mods[m] if p not in used]
        out.append(cand[0])
        used.add(cand[0])
    return out


def own_mode_step(ids):
    steps = collections.Counter(ids[j + 1] - ids[j] for j in range(len(ids) - 1))
    return steps.most_common(1)[0][1] if steps else 0


def pairs_at(s, count=104):
    """count pairs whose slowdown is exactly s (contiguous 1,000,000 ns, paged (1 + s) of it)."""
    return [(1_000_000.0 * (1.0 + s), 1_000_000.0, 7, 7) for _ in range(count)]


class RowRule(unittest.TestCase):
    def test_79_12_exact_bar(self):
        self.assertEqual(rows.row_verdict([0.05] * 5), "PASS")
        self.assertEqual(rows.row_verdict([math.nextafter(0.05, math.inf)] * 5), "FAIL")
        self.assertEqual(rows.row_verdict([0.04] * 5), "PASS")
        self.assertEqual(rows.row_verdict([0.06] * 5), "FAIL")

    def test_79_12_mixed(self):
        # largest > 0.05 but the median <= 0.05: neither PASS nor FAIL
        self.assertEqual(rows.row_verdict([0.04, 0.04, 0.04, 0.06, 0.06]), "NO RESULT")
        # the median > 0.05: FAIL, whatever the smallest
        self.assertEqual(rows.row_verdict([0.06, 0.06, 0.06, 0.00, 0.00]), "FAIL")
        # the median exactly at the bar with a larger maximum: neither PASS (the max is above) nor FAIL (the median is
        # not above). Kills "FAIL uses median >= 0.05" (T-3198 §15's survivor; T-3189 §11)
        self.assertEqual(rows.row_verdict([0.05, 0.05, 0.05, 0.06, 0.06]), "NO RESULT")
        with self.assertRaises(rows.Refusal):
            rows.row_verdict([0.0] * 4)

    def test_slowdown_is_the_time_ratio(self):
        self.assertAlmostEqual(rows.slowdown(105.0, 100.0), 0.05)
        self.assertAlmostEqual(rows.slowdown(100.0, 100.0), 0.0)

    def test_process_statistic_is_the_median(self):
        rng = random.Random(79)
        values = [rng.uniform(-0.03, 0.08) for _ in range(104)]
        pairs = [(1e6 * (1 + v), 1e6, 3, 3) for v in values]
        self.assertAlmostEqual(rows.process_statistic(pairs), sorted(values)[51] / 2 + sorted(values)[52] / 2, places=9)
        self.assertAlmostEqual(rows.process_statistic(pairs_at(0.02)), 0.02, places=9)

    def test_79_9_incomplete_process(self):
        for count in (103, 105, 0):
            with self.assertRaises(rows.Refusal):
                rows.process_statistic(pairs_at(0.0, count))

    def test_79_8_token_mismatch(self):
        bad = pairs_at(0.0)
        bad[57] = (1e6, 1e6, 7, 8)
        with self.assertRaises(rows.Refusal):
            rows.process_statistic(bad)

    def test_nonfinite_or_nonpositive_duration(self):
        for broken in ((math.nan, 1e6, 1, 1), (1e6, 0.0, 1, 1), (-1.0, 1e6, 1, 1)):
            bad = pairs_at(0.0)
            bad[3] = broken
            with self.assertRaises(rows.Refusal):
                rows.process_statistic(bad)

    def test_constants(self):
        self.assertEqual((rows.PAIRS_PER_PROCESS, rows.PROCESSES_PER_ROW, rows.WARMUP_TOKENS, rows.BAR), (104, 5, 4, 0.05))
        self.assertEqual(str(rows.LOCK_PATH), r"D:\_scratch\SuperSLM\pkv-c6-timing.lock")


class Guard(unittest.TestCase):
    def test_bound_table_both_page_sizes(self):
        self.assertEqual({k: (v["n"], v["P"], v["m99"]) for k, v in rows.ROWS.items()}, PRODUCTION)
        self.assertEqual({k: (v["n"], v["P"], v["m99"]) for k, v in rows.MR_OPT_ROWS.items()}, MR_OPT)
        for slot, (n, pool, m99) in PRODUCTION.items():
            self.assertEqual(rows.guard_bound(slot, 64, n), m99)
        for slot, (n, pool, m99) in MR_OPT.items():
            self.assertEqual(rows.guard_bound(slot, 16, n), m99)

    def test_mode_step_definition(self):
        for ids in ([0, 1, 2, 3], [5, 3, 9, 7, 13], [4, 2, 0, 9, 7, 5], list(range(40, 0, -3)), [1]):
            self.assertEqual(rows.mode_step(ids), own_mode_step(ids))

    def vitality(self, table, page_positions):
        for slot, (n, pool, m99) in table.items():
            runs, dpe = GEOMETRY[slot]
            run_bytes = page_positions * dpe
            orders = {"shuffle": proto_shuffle(pool)[:n], "ascending": list(range(n)), "pair churn": pair_churn(pool)[:n],
                      "interleave-5": interleave(pool)[:n],
                      "set-concentrating": set_concentrating(n, runs, run_bytes, runs * run_bytes, 4 * pool)}
            for name, ids in orders.items():
                with self.subTest(slot=slot, B=page_positions, order=name):
                    self.assertEqual(rows.guard_admits(slot, page_positions, ids), name == "shuffle",
                                     f"M = {own_mode_step(ids)}, M99 = {m99}")

    def test_vitality_at_b64(self):
        self.vitality(PRODUCTION, 64)

    def test_vitality_at_b16(self):
        self.vitality(MR_OPT, 16)

    def test_bound_is_inclusive(self):
        # n ids whose most common step occurs exactly k times: k steps of +1, the rest distinct and larger
        def with_mode(n, k):
            steps = [1] * k + list(range(2, 2 + n - 1 - k))
            ids = [0]
            for s in steps:
                ids.append(ids[-1] + s)
            return ids
        for table, b in ((PRODUCTION, 64), (MR_OPT, 16)):
            for slot, (n, pool, m99) in table.items():
                with self.subTest(slot=slot, B=b):
                    self.assertEqual(own_mode_step(with_mode(n, m99)), m99)
                    self.assertTrue(rows.guard_admits(slot, b, with_mode(n, m99)))
                    self.assertFalse(rows.guard_admits(slot, b, with_mode(n, m99 + 1)))

    def test_five_lists_agree(self):
        ids = proto_shuffle(64)[:64]
        self.assertTrue(rows.id_lists_agree([ids] * 5))
        other = list(ids)
        other[3], other[4] = other[4], other[3]
        self.assertFalse(rows.id_lists_agree([ids] * 4 + [other]))


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_inputs(root):
    """A model file and a two-piece blob chain with its manifest (restore 1,200), healthy. Returns (model, manifest,
    manifest data)."""
    model = root / "model.sslm"
    model.write_bytes(b"model bytes")
    prev, at, pieces = "-", 0, []
    for i, count in enumerate((700, 500)):
        f = root / f"piece{i}.ssb"
        f.write_bytes(f"blob after {at + count}".encode())
        pieces.append({"file": f.name, "start": at, "count": count, "input_sha256": prev, "sha256": sha(f)})
        prev, at = sha(f), at + count
    data = {"complete": True, "model_sha256": sha(model), "stream": {"seed": 79, "restore": 1200},
            "pieces": pieces, "final_blob": pieces[-1]["file"], "final_sha256": pieces[-1]["sha256"]}
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(data))
    return model, manifest, data


class Refusals(unittest.TestCase):
    """79-6, 79-7 and the per-process refusal rule (cell 7.9, rev 16.23)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="t3189-r9-refusals-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model, self.manifest, self.data = row_inputs(self.root)

    def write(self, data):
        self.data = data
        self.manifest.write_text(json.dumps(data))

    def test_79_6_healthy_blob_accepted(self):
        blob, digest = rows.check_row_blob(self.manifest, self.model, 1200)
        self.assertEqual(Path(blob).name, "piece1.ssb")
        self.assertEqual(digest, sha(self.manifest))

    def test_79_6_each_mismatch_refused(self):
        cases = {
            "blob digest": lambda: (self.root / "piece1.ssb").write_bytes(b"changed blob"),
            "chain": lambda: self.write({**self.data, "pieces": [self.data["pieces"][0],
                                                                  {**self.data["pieces"][1], "input_sha256": "0" * 64}]}),
            "model binding": lambda: self.model.write_bytes(b"another model"),
            "stream": lambda: self.write({**self.data, "stream": {"seed": 79, "restore": 1204}}),
            "torn manifest": lambda: self.manifest.write_text('{"complete": tr'),
            "final digest alone": lambda: self.write({**self.data, "final_sha256": "0" * 64}),
        }
        for name, damage in cases.items():
            with self.subTest(case=name):
                self.setUp()
                damage()
                with self.assertRaises(rows.Refusal):
                    rows.check_row_blob(self.manifest, self.model, 1200)

    def test_79_7_seam_build_refused(self):
        self.assertIsNone(rows.check_build_identity("PKV_PAIR_PRODUCTION_V2 defs=none"))
        for line in ("PKV_PAIR_PRODUCTION_V2 defs=PKV_C6_IDENTITY_PROBE", "PKV_PAIR_PRODUCTION_V2 defs=SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS", ""):
            with self.subTest(line=line):
                with self.assertRaises(rows.Refusal):
                    rows.check_build_identity(line)

    def test_exit_codes_distinct(self):
        self.assertEqual(set(rows.EXIT_CODES), {"blob", "seam", "token", "pairs"})
        self.assertEqual(len(set(rows.EXIT_CODES.values())), 4)
        self.assertNotIn(0, rows.EXIT_CODES.values())

    def test_row_graded_only_on_five_complete_processes(self):
        ok = [{"status": "ok", "median": 0.01}] * 5
        graded = rows.grade_row(ok)
        self.assertTrue(graded["graded"])
        self.assertEqual(graded["verdict"], "PASS")
        self.assertIsNotNone(graded["record"])
        for kind in ("blob", "seam", "token", "pairs"):
            with self.subTest(kind=kind):
                bad = list(ok)
                bad[2] = {"status": "refused", "kind": kind}
                g = rows.grade_row(bad)
                self.assertEqual((g["graded"], g["verdict"], g["record"]), (False, None, None))
                self.assertEqual(g["exit_code"], rows.EXIT_CODES[kind])
        g = rows.grade_row(ok[:4])  # never graded on four
        self.assertEqual((g["graded"], g["verdict"]), (False, None))


# The stand-in pair executable: prints its construction's identity line (D-SLM8466 (2); PKV_STANDIN_CONSTRUCTION,
# default production; PKV_STANDIN_IDENTITY overrides it), 104 timed pairs (slowdown 1%, tokens equal) and, for the
# constructions with a paged arm only (D-SLM8466 (1)), the measured holder's ids (PKV_STANDIN_IDS_MODE: print, none or
# empty overrides that). A paged construction prints the RESERVE line by default (D-SLM8494, T-3210 O-4: a paged run
# without one refuses), written here literally; PKV_STANDIN_RESERVE replaces it, or withholds it with "none"; PKV_STANDIN_INJECT="<kind>@<k>" injects one refusal into process k. A "blob" injection
# changes the blob's bytes at the end of process k - 1, so process k's pre-restore check is what must see it.
STANDIN = r"""
import argparse, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser()
for flag in ("--blob", "--model", "--warmup"):
    ap.add_argument(flag)
ap.add_argument("--process", type=int)
ap.add_argument("--pairs", type=int)
a = ap.parse_args()
IDENTITY = {"production": "PKV_PAIR_PRODUCTION_V2 defs=none", "7.9-MR-OPT": "PKV_PAIR_MR_OPT_V1 defs=none",
            "MA-1": "PKV_PAIR_MA1_V1 defs=none", "MA-2": "PKV_PAIR_MA2_V1 defs=none"}
construction = os.environ.get("PKV_STANDIN_CONSTRUCTION") or "production"
identity = os.environ.get("PKV_STANDIN_IDENTITY") or IDENTITY[construction]
ids_mode = os.environ.get("PKV_STANDIN_IDS_MODE") or ("print" if construction in ("production", "7.9-MR-OPT") else "none")
kind, _, at = os.environ.get("PKV_STANDIN_INJECT", "").partition("@")
at = int(at) if at else 0
with open(os.environ["PKV_STANDIN_LOG"], "a") as log:
    log.write(f"{a.process}\n")
mine = a.process == at
print("PKV_PAIR_PRODUCTION_V2 defs=PKV_C6_IDENTITY_PROBE" if kind in ("seam", "crash-seam") and mine else identity)
reserve = os.environ.get("PKV_STANDIN_RESERVE", "")
if reserve == "none":
    pass
elif reserve:
    print(reserve)
elif construction in ("production", "7.9-MR-OPT"):
    print("RESERVE seed=1 singles=63 source=tests/paged-kv/pkv_scatter_reserve.h")
for i in range(a.pairs - 1 if kind == "pairs" and mine else a.pairs):
    if kind == "badpair" and mine and i == 40:
        print("PAIR 1010000 1.0e6x 7 7")
        continue
    print(f"PAIR 1010000 1000000 7 {8 if kind == 'token' and mine and i == 57 else 7}")
if kind == "undecodable" and mine:
    sys.stdout.flush()
    sys.stdout.buffer.write(bytes([78, 79, 84, 69, 32, 0x81, 0x8d, 10]))
    sys.stdout.buffer.flush()
if ids_mode == "print":
    print("IDS " + " ".join(str(i) for i in json.loads(os.environ["PKV_STANDIN_IDS"])))
elif ids_mode == "empty":
    print("IDS")
if kind == "blob" and a.process == at - 1:
    Path(a.blob).write_bytes(Path(a.blob).read_bytes() + b"changed")
if mine and kind in ("crash", "crash-seam"):
    sys.exit(5)
once = Path(os.environ["PKV_STANDIN_LOG"] + ".once")
if mine and kind == "crash-once" and not once.exists():
    once.write_text("crashed")
    sys.exit(5)
"""


class Loop(unittest.TestCase):
    """79-6 to 79-9 through the driver's process loop, as a process (T-3196 F-6)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="t3189-r9-loop-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model, self.manifest, _ = row_inputs(self.root)
        self.standin = self.root / "standin_pair.py"
        self.standin.write_text(STANDIN)
        self.log = self.root / "standin.log"
        self.out = self.root / "out"
        self.out.mkdir()
        self.result = self.out / "R2a-run1.json"

    def run_row(self, inject="", ids=None, reserve="", pair_cmd=None):
        env = dict(os.environ, PKV_STANDIN_INJECT=inject, PKV_STANDIN_LOG=str(self.log),
                   PKV_STANDIN_IDS=json.dumps(ids if ids is not None else proto_shuffle(64)[:21]),
                   PKV_STANDIN_RESERVE=reserve, PKV_STANDIN_CONSTRUCTION="production", PKV_STANDIN_IDENTITY="",
                   PKV_STANDIN_IDS_MODE="")
        cmd = [sys.executable, "-I", str(SCRIPT), "--pair-cmd", json.dumps(pair_cmd or [sys.executable, "-I", str(self.standin)]),
               "--manifest", str(self.manifest), "--model", str(self.model), "--restore", "1200", "--slot", "R2a",
               "--result", str(self.result), "--lock", str(self.root / "pkv-c6-timing.lock")]
        done = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        ran = [int(x) for x in self.log.read_text().split()] if self.log.exists() else []
        return done, ran

    def verdicts_written(self):
        found = []
        for f in self.out.rglob("*"):
            if f.is_file():
                try:
                    data = json.loads(f.read_text())
                except ValueError:
                    continue
                if isinstance(data, dict) and data.get("verdict") is not None:
                    found.append(f.name)
        return found

    def test_loop_healthy_row_is_graded(self):
        done, ran = self.run_row()
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual(ran, [1, 2, 3, 4, 5])
        self.assertEqual((rows.read_record(self.result) or {}).get("verdict"), "PASS")
        for k in range(1, 6):
            self.assertIsNotNone(rows.read_record(self.out / f"R2a-run1.p{k}.json"), f"process {k}'s record")

    def test_loop_each_refusal_in_process_3(self):
        for kind in ("blob", "seam", "token", "pairs"):
            with self.subTest(kind=kind):
                self.setUp()
                done, ran = self.run_row(f"{kind}@3")
                self.assertEqual(done.returncode, rows.EXIT_CODES.get(kind), f"{kind}: exit {done.returncode}")
                self.assertNotEqual(done.returncode, 0)
                self.assertFalse(self.result.exists(), f"{kind}: a row record was written")
                self.assertFalse((self.out / "R2a-run1.p3.json").exists(), f"{kind}: the refused process wrote a record")
                self.assertEqual(self.verdicts_written(), [], f"{kind}: a verdict was written")
                # processes 1 and 2 completed on a blob checked before each restore: their records stand
                for k in (1, 2):
                    self.assertIsNotNone(rows.read_record(self.out / f"R2a-run1.p{k}.json"),
                                         f"{kind}: process {k}'s record (a check after the restore refuses process 2)")
                if kind == "blob":
                    # checked before process 3 restores: process 3 never ran
                    self.assertEqual(ran, [1, 2], "the blob check ran after the restore, or not per process")
                else:
                    self.assertIn(3, ran)


# ---- T-3202 (code review) remedies, SuperSLM 1478be7: one pin cell per landed remedy. Each fails with that remedy
# reverted (the test author's record, T-3189 §12, runs each revert) and passes on the built driver.
RESERVE_LINE = "RESERVE seed=1 singles=63 source=tests/paged-kv/pkv_scatter_reserve.h"


class T6Remedies(unittest.TestCase):
    setUp = Loop.setUp
    run_row = Loop.run_row

    def record(self, path=None):
        return json.loads((path or self.result).read_text(encoding="utf-8"))

    # F-1 (D-SLM8461): the row record carries status STANDING and the driver's own hash of the pair executable,
    # the verdict in its own field; every process record carries the hash too.
    def test_f1_row_record_status_and_exe_sha256(self):
        done, _ = self.run_row()
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        rec = self.record()
        exe = Path(sys.executable).resolve()
        self.assertEqual(rec.get("status"), "STANDING")
        self.assertEqual(rec.get("verdict"), "PASS")
        self.assertEqual(rec.get("exe_sha256"), sha(exe))
        self.assertEqual(Path(rec.get("exe", "")).resolve(), exe)
        for k in range(1, 6):
            self.assertEqual(self.record(self.out / f"R2a-run1.p{k}.json").get("exe_sha256"), sha(exe), f"process {k}")

    # F-1 (D-SLM8461): a NO RESULT row stands as a reading of no result.
    def test_f1_no_result_row_stands(self):
        done, _ = self.run_row(ids=list(range(21)))  # ascending: the guard rejects, so the row reads NO RESULT
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        rec = self.record()
        self.assertEqual((rec.get("status"), rec.get("verdict")), ("STANDING", "NO RESULT"))

    # F-1: a pair executable that does not exist refuses (exit 70) before any run, writing nothing.
    def test_f1_missing_pair_executable_refuses(self):
        done, ran = self.run_row(pair_cmd=[str(self.root / "no-such-pair.exe")])
        self.assertEqual(done.returncode, 70, done.stderr[-2000:])
        self.assertEqual(ran, [])
        self.assertFalse(self.result.exists())

    # F-12 (with D-SLM8461): a refused run replaces an earlier row record at its path with a REFUSED record.
    def test_f12_refused_run_supersedes_an_existing_record(self):
        self.assertEqual(self.run_row()[0].returncode, 0)
        self.assertEqual(self.record().get("status"), "STANDING")
        self.log.unlink()
        done, _ = self.run_row("seam@3")
        self.assertEqual(done.returncode, 62, done.stderr[-2000:])
        rec = self.record()
        self.assertEqual((rec.get("status"), rec.get("exit_code"), rec.get("verdict")), ("REFUSED", 62, None))
        self.assertIn("superseded", rec)
        self.assertEqual(rec.get("exe_sha256"), sha(Path(sys.executable).resolve()))

    # F-12: the rerun limit is a refusal, so it supersedes too.
    def test_f12_rerun_limit_supersedes(self):
        self.assertEqual(self.run_row()[0].returncode, 0)
        self.log.unlink()
        done, ran = self.run_row("crash@3")
        self.assertEqual(done.returncode, 70, done.stderr[-2000:])
        self.assertEqual(ran.count(3), 3)
        rec = self.record()
        self.assertEqual((rec.get("status"), rec.get("exit_code"), rec.get("verdict")), ("REFUSED", 70, None))

    # F-2: the exit status is read before the identity. A process that prints a seam identity and then crashes is
    # rerun whole (three attempts, then 70), never graded 62; a transient crash is rerun and the row is graded.
    def test_f2_crash_after_a_seam_identity_is_rerun_not_graded(self):
        done, ran = self.run_row("crash-seam@3")
        self.assertEqual(done.returncode, 70, done.stderr[-2000:])
        self.assertEqual(ran.count(3), 3)

    def test_f2_transient_crash_is_rerun(self):
        done, ran = self.run_row("crash-once@3")
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual(ran.count(3), 2)
        self.assertEqual(self.record().get("status"), "STANDING")

    # F-11: the RESERVE line is recorded in every process record and in the row record; absent, it records null.
    def test_f11_reserve_recorded(self):
        done, _ = self.run_row(reserve=RESERVE_LINE)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        want = {"seed": 1, "singles": 63, "source": "tests/paged-kv/pkv_scatter_reserve.h"}
        for k in range(1, 6):
            self.assertEqual(self.record(self.out / f"R2a-run1.p{k}.json").get("reserve"), want, f"process {k}")
        self.assertEqual(self.record().get("reserve"), want)
        # Superseded (D-SLM8494, T-3210 O-4): this member read "absent, it records null". A production run with no
        # RESERVE line now refuses as 79-7, exit 62, with no record (T8Remedies.test_t10_* pins it at both paged
        # constructions)
        self.setUp()
        done, _ = self.run_row(reserve="none")
        self.assertEqual(done.returncode, 62, done.stderr[-2000:])
        self.assertIn("no RESERVE line", done.stderr)
        self.assertFalse(self.result.exists())

    # F-9: the lock is taken by an OS byte lock before the owner is read. A runner that holds the byte lock while the
    # content still names a dead owner (mid-recovery) refuses a second runner, which must not also recover.
    def test_f9_os_lock_holder_refuses_a_second_recovery(self):
        lock = self.root / "pkv-c6-timing.lock"
        dead = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
        dead.wait()
        holder_src = (
            "import msvcrt, os, sys, time\n"
            "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT | os.O_BINARY, 0o600)\n"
            "os.write(fd, sys.argv[2].encode())\n"
            "os.lseek(fd, 1 << 20, os.SEEK_SET)\n"
            "msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)\n"
            "print('held', flush=True)\n"
            "time.sleep(60)\n") if os.name == "nt" else (
            "import fcntl, os, sys, time\n"
            "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
            "os.write(fd, sys.argv[2].encode())\n"
            "fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, 1 << 20)\n"
            "print('held', flush=True)\n"
            "time.sleep(60)\n")
        holder = subprocess.Popen([sys.executable, "-I", "-c", holder_src, str(lock), str(dead.pid)],
                                  stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: (holder.kill(), holder.wait(), holder.stdout.close()))
        self.assertEqual(holder.stdout.readline().strip(), "held")
        with self.assertRaises(rows.Refusal):
            with rows.timing_lock(self.root / "b.json", lock_path=lock):
                pass


# ---- T-3203 (confirmation review) remedies, SuperSLM 371dc34: one pin cell per landed remedy. Each fails with that
# remedy reverted (the test author's record, T-3189 §14, runs each revert) and passes on the built driver.
POSITIONS = {"production": 64, "MA-1": 64, "MA-2": 64, "7.9-MR-OPT": 16}
R79A_RUNS = (("MA-1", "R2b"), ("MA-1", "R4a"), ("MA-1", "R4b"), ("MA-2", "R4a"))
R79MR_RUNS = (("7.9-MR-OPT", "R2b"), ("7.9-MR-OPT", "R4a"), ("7.9-MR-OPT", "R4b"))


def ids_for(slot, positions):
    """An id list of the n the row reads (in any order: an ascending list reads NO RESULT, which still stands)."""
    n = (PRODUCTION if positions == 64 else MR_OPT)[slot][0]
    return list(range(n))


class T7Remedies(unittest.TestCase):
    built = None  # the healthy row records the aggregate cells start from, written once by the driver itself

    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory(prefix="t3189-r9-t7-")
        root = Path(cls.shared.name)
        cls.model, cls.manifest, _ = row_inputs(root)
        cls.standin = root / "standin_pair.py"
        cls.standin.write_text(STANDIN)
        cls.built = {}
        for construction, row in R79A_RUNS + R79MR_RUNS + (("production", "R2b"),):
            result = root / "runs" / f"{construction}-{row}.json"
            done = cls.drive(root, row, result, construction)
            if done.returncode != 0:
                cls.shared.cleanup()
                raise AssertionError(f"{construction} {row}: exit {done.returncode}\n{done.stderr[-2000:]}")
            cls.built[(construction, row)] = json.loads(result.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    @classmethod
    def drive(cls, root, row, result, construction="production", positions=None, inject="", ids=None, reserve="",
              drop=(), log=None, extra=(), identity="", ids_mode=""):
        positions = POSITIONS[construction] if positions is None else positions
        env = dict(os.environ, PKV_STANDIN_INJECT=inject, PKV_STANDIN_LOG=str(log or root / "standin.log"),
                   PKV_STANDIN_IDS=json.dumps(ids if ids is not None else ids_for(row, POSITIONS[construction])),
                   PKV_STANDIN_RESERVE=reserve, PKV_STANDIN_CONSTRUCTION=construction, PKV_STANDIN_IDENTITY=identity,
                   PKV_STANDIN_IDS_MODE=ids_mode)
        flags = {"--pair-cmd": json.dumps([sys.executable, "-I", str(cls.standin)]), "--manifest": str(cls.manifest),
                 "--model": str(cls.model), "--restore": "1200", "--slot": row, "--result": str(result),
                 "--lock": str(root / "pkv-c6-timing.lock"), "--page-positions": str(positions),
                 "--construction": construction}
        cmd = [sys.executable, "-I", str(SCRIPT)]
        for flag, value in flags.items():
            if flag not in drop:
                cmd += [flag, value]
        cmd += list(extra)
        Path(result).parent.mkdir(parents=True, exist_ok=True)
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="t3189-r9-t7case-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "standin.log"
        self.result = self.root / "row.json"

    def ran(self):
        return [int(x) for x in self.log.read_text().split()] if self.log.exists() else []

    def record(self, path=None):
        return json.loads(Path(path or self.result).read_text(encoding="utf-8"))

    def runs(self, keys, edit=None):
        """Copies of the built row records for `keys` in this case's directory; `edit` maps a key to a change (None:
        no file; a string: the file's raw text)."""
        paths = []
        for i, key in enumerate(keys):
            record = dict(self.built[key])
            if edit and key in edit:
                record = edit[key](record)
            path = self.root / "runs" / f"{i}-{key[0]}-{key[1]}.json"
            path.parent.mkdir(exist_ok=True)
            if isinstance(record, str):
                path.write_text(record, encoding="utf-8")
            elif record is not None:
                path.write_text(json.dumps(record), encoding="utf-8")
            paths.append(path)
        return paths

    def aggregate(self, slot, runs, result=None, flag=True):
        cmd = [sys.executable, "-I", str(SCRIPT)] + (["--aggregate"] if flag else []) + [
            "--slot", slot, "--result", str(result or self.result)]
        if runs is not None:
            cmd += ["--runs"] + [str(r) for r in runs]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # N-1 (D-SLM8463): --construction is recorded in every process record and in the row record, with its page size
    def test_n1_construction_is_recorded(self):
        for (construction, row), rec in self.built.items():
            with self.subTest(construction=construction, row=row):
                self.assertEqual((rec.get("construction"), rec.get("page_positions"), rec.get("status")),
                                 (construction, POSITIONS[construction], "STANDING"))
        for key in (("MA-2", "R4a"), ("7.9-MR-OPT", "R4b")):
            for name in self.built[key]["processes"]:
                proc = self.record(Path(self.shared.name) / "runs" / name)
                self.assertEqual((proc.get("construction"), proc.get("page_positions")), (key[0], POSITIONS[key[0]]),
                                 name)
        done = self.drive(self.root, "R2a", self.result, drop=("--construction",), log=self.log)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual(self.record().get("construction"), "production")

    # N-1: a construction at another page size refuses (70) before any process runs, writing nothing
    def test_n1_construction_at_another_page_size_refused(self):
        for construction, positions, row in (("7.9-MR-OPT", 64, "R2b"), ("MA-1", 16, "R2b"), ("production", 16, "R4a"),
                                             ("MA-2", 16, "R4b")):
            with self.subTest(construction=construction, positions=positions):
                self.log.unlink(missing_ok=True)
                done = self.drive(self.root, row, self.result, construction, positions, log=self.log)
                self.assertEqual(done.returncode, 70, done.stderr[-2000:])
                self.assertEqual(self.ran(), [])
                self.assertEqual(sorted(p.name for p in self.root.iterdir()), [])

    # N-1: R79A's slot record from its four runs, R79MR's from its three; STANDING (exit 0) when every run stands
    def test_n1_aggregate_standing(self):
        for slot, keys in (("R79A", R79A_RUNS), ("R79MR", R79MR_RUNS)):
            with self.subTest(slot=slot):
                result = self.root / f"{slot}.json"
                done = self.aggregate(slot, self.runs(keys), result)
                self.assertEqual(done.returncode, 0, done.stderr[-2000:])
                rec = self.record(result)
                self.assertEqual((rec.get("slot"), rec.get("aggregate"), rec.get("status")), (slot, True, "STANDING"))
                self.assertEqual(sorted((r["construction"], r["row"]) for r in rec["runs"]), sorted(keys))
                self.assertTrue(all(r["status"] == "STANDING" for r in rec["runs"]))
                self.assertEqual(rec.get("exe_sha256"), sha(Path(sys.executable).resolve()))

    # N-1: each refusal of the aggregate exits 70; on a fresh path it writes nothing, and over an existing record it
    # supersedes it with a REFUSED record
    def test_n1_aggregate_refusals(self):
        A = R79A_RUNS
        cases = {
            "missing run": (A, {A[1]: lambda r: None}, "missing or torn"),
            "torn run": (A, {A[1]: lambda r: '{"slot": "R4a", "stat'}, "missing or torn"),
            "status not STANDING or REFUSED": (A, {A[2]: lambda r: {**r, "status": "QUARANTINED"}}, "status"),
            "status absent": (A, {A[2]: lambda r: {k: v for k, v in r.items() if k != "status"}}, "status"),
            "page positions not the construction's": (A, {A[0]: lambda r: {**r, "page_positions": 16}},
                                                      "page positions"),
            "a run short": (A[:3], None, "owed"),
            "a run duplicated": (A[:3] + (A[1],), None, "owed"),
            "an extra run": (A + (("production", "R2b"),), None, "owed"),
            "another slot's runs": (R79MR_RUNS, None, "owed"),
            "a row relabelled": (A, {A[3]: lambda r: {**r, "slot": "R4b"}}, "owed"),
            "a construction relabelled": (A, {A[3]: lambda r: {**r, "construction": "MA-1"}}, "owed"),
            "exe_sha256 differs": (A, {A[2]: lambda r: {**r, "exe_sha256": "0" * 64}}, "differ"),
            "exe_sha256 missing on one": (A, {A[2]: lambda r: {**r, "exe_sha256": None}}, "differ"),
            "exe_sha256 missing on all": (A, {k: (lambda r: {**r, "exe_sha256": None}) for k in A}, "differ"),
        }
        for name, (keys, edit, why) in cases.items():
            with self.subTest(case=name):
                for f in self.root.glob("runs/*"):
                    f.unlink()
                runs = self.runs(keys, edit)
                fresh = self.root / "fresh.json"
                fresh.unlink(missing_ok=True)  # each case starts on a fresh path
                done = self.aggregate("R79A", runs, fresh)
                self.assertEqual(done.returncode, 70, f"{name}: exit {done.returncode}\n{done.stderr[-1500:]}")
                self.assertIn(why, done.stderr)
                self.assertFalse(fresh.exists(), f"{name}: a refused aggregate wrote a record on a fresh path")
                standing = self.root / "standing.json"
                standing.write_text(json.dumps({"slot": "R79A", "aggregate": True, "status": "STANDING", "runs": []}))
                done = self.aggregate("R79A", runs, standing)
                self.assertEqual(done.returncode, 70)
                rec = self.record(standing)
                self.assertEqual((rec.get("status"), rec.get("exit_code"), rec.get("verdict")), ("REFUSED", 70, None),
                                 f"{name}: the earlier aggregate record still stands")
                self.assertIn("superseded", rec)

    # N-1: a run the driver refused (its record superseded REFUSED) makes the aggregate REFUSED, exit 70; the
    # superseding record carries its construction and page positions
    def test_n1_refused_run_makes_the_aggregate_refused(self):
        for slot, keys, refused in (("R79A", R79A_RUNS, ("MA-2", "R4a")),
                                    ("R79MR", R79MR_RUNS, ("7.9-MR-OPT", "R4b"))):
            with self.subTest(slot=slot):
                runs = self.runs(keys)
                target = runs[keys.index(refused)]
                done = self.drive(self.root, refused[1], target, refused[0], inject="seam@2", log=self.log)
                self.assertEqual(done.returncode, 62, done.stderr[-2000:])
                sup = self.record(target)
                self.assertEqual((sup.get("status"), sup.get("construction"), sup.get("page_positions")),
                                 ("REFUSED", refused[0], POSITIONS[refused[0]]))
                result = self.root / f"{slot}.json"
                done = self.aggregate(slot, runs, result)
                self.assertEqual(done.returncode, 70, done.stderr[-2000:])
                rec = self.record(result)
                self.assertEqual((rec.get("aggregate"), rec.get("status")), (True, "REFUSED"))
                self.assertEqual([r["status"] for r in rec["runs"]].count("REFUSED"), 1)

    # N-1: the argument errors exit 70 and write nothing; a row run missing an input never starts a process
    def test_n1_argument_errors(self):
        runs = self.runs(R79A_RUNS)
        for slot in ("R2b", "R4a", "R3", "R79"):
            with self.subTest(aggregate_slot=slot):
                self.assertEqual(self.aggregate(slot, runs).returncode, 70)
                self.assertFalse(self.result.exists())
        with self.subTest(case="--aggregate without --runs"):
            self.assertEqual(self.aggregate("R79A", None).returncode, 70)
            self.assertFalse(self.result.exists())
        with self.subTest(case="--runs without --aggregate"):
            # a complete row run otherwise: only --runs makes it an argument error
            self.log.unlink(missing_ok=True)
            done = self.drive(self.root, "R2a", self.result, log=self.log, extra=["--runs"] + [str(r) for r in runs])
            self.assertEqual(done.returncode, 70, done.stderr[-2000:])
            self.assertEqual(self.ran(), [])
            self.assertFalse(self.result.exists())
        for flag in ("--pair-cmd", "--manifest", "--model", "--restore"):
            with self.subTest(missing=flag):
                self.log.unlink(missing_ok=True)
                self.result.unlink(missing_ok=True)
                done = self.drive(self.root, "R2a", self.result, drop=(flag,), log=self.log)
                self.assertEqual(done.returncode, 70, done.stderr[-2000:])
                # the argument check's own refusal, before the lock and the executable's hash (not the catch-all's)
                self.assertIn("a row run needs " + flag, done.stderr)
                self.assertEqual(self.ran(), [])
                self.assertFalse(self.result.exists())

    def standing_row(self):
        done = self.drive(self.root, "R2a", self.result, log=self.log)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual(self.record().get("status"), "STANDING")
        self.log.unlink()

    def refusal(self, code, **kw):
        """The same refusal on a fresh path (no record) and over a standing record (superseded REFUSED)."""
        fresh = self.root / "fresh.json"
        done = self.drive(self.root, "R2a", fresh, log=self.log, **kw)
        self.assertEqual(done.returncode, code, done.stderr[-2000:])
        self.assertIn("REFUSED", done.stderr)
        self.assertFalse(fresh.exists())
        self.log.unlink(missing_ok=True)
        self.standing_row()
        done = self.drive(self.root, "R2a", self.result, log=self.log, **kw)
        self.assertEqual(done.returncode, code, done.stderr[-2000:])
        rec = self.record()
        self.assertEqual((rec.get("status"), rec.get("exit_code"), rec.get("verdict")), ("REFUSED", code, None))
        self.assertIn("superseded", rec)
        return done, rec

    # N-8: a PAIR line that does not parse refuses as the pairs kind, 64
    def test_n8_malformed_pair_line_exits_64(self):
        _, rec = self.refusal(64, inject="badpair@2")
        self.assertIn("malformed pair line", rec.get("reason", ""))

    # N-8: an IDS line that does not parse refuses as infrastructure, 70
    def test_n8_malformed_ids_line_exits_70(self):
        _, rec = self.refusal(70, ids=["7", "x9"] + list(range(19)))
        self.assertIn("malformed IDS line", rec.get("reason", ""))  # its own refusal, not the catch-all's

    # N-8: a RESERVE line whose seed or singles is not an integer refuses as infrastructure, 70
    def test_n8_malformed_reserve_line_exits_70(self):
        for line in ("RESERVE seed=x1 singles=63 source=h", "RESERVE seed=1 singles=6.3 source=h"):
            with self.subTest(line=line):
                self.setUp()
                _, rec = self.refusal(70, reserve=line)
                self.assertIn("malformed RESERVE line", rec.get("reason", ""))  # its own refusal, not the catch-all's

    # N-8: a fault the driver did not anticipate (here, process output that does not decode) refuses through the
    # driver's own code: 70, never Python's 1, and the earlier record is superseded
    def test_n8_driver_fault_exits_70_not_1(self):
        done, rec = self.refusal(70, inject="undecodable@1")
        self.assertIn("driver fault", rec.get("reason", ""))


# ---- T-3204 (second confirmation review) remedies, D-SLM8466, SuperSLM 4156b30: one pin cell per landed remedy. Each
# fails with its remedy reverted (the test author's record, T-3189 §15, runs each revert) and passes on the built driver.
# The identity strings and the paged-arm set are written here literally, never read from the driver under test.
IDENTITIES = {"production": "PKV_PAIR_PRODUCTION_V2 defs=none", "7.9-MR-OPT": "PKV_PAIR_MR_OPT_V1 defs=none",
              "MA-1": "PKV_PAIR_MA1_V1 defs=none", "MA-2": "PKV_PAIR_MA2_V1 defs=none"}
WITH_PAGED_ARM = ("production", "7.9-MR-OPT")
NOT_APPLICABLE = "not applicable (no paged arm)"
ROW_FOR = {"production": "R2a", "7.9-MR-OPT": "R2b", "MA-1": "R2b", "MA-2": "R4a"}


class T8Remedies(unittest.TestCase):
    setUpClass = T7Remedies.__dict__["setUpClass"]
    tearDownClass = T7Remedies.__dict__["tearDownClass"]
    drive = T7Remedies.__dict__["drive"]
    setUp = T7Remedies.setUp
    ran = T7Remedies.ran
    record = T7Remedies.record
    runs = T7Remedies.runs
    aggregate = T7Remedies.aggregate

    def refused_62(self, construction, why, **kw):
        """The run refuses 79-7 (62) in process 1, before any record: none on a fresh path."""
        result = self.root / "fresh.json"
        result.unlink(missing_ok=True)
        self.log.unlink(missing_ok=True)
        done = self.drive(self.root, ROW_FOR[construction], result, construction, log=self.log, **kw)
        self.assertEqual(done.returncode, 62, f"{why}: exit {done.returncode}\n{done.stderr[-1500:]}")
        self.assertIn("79-7", done.stderr)
        self.assertFalse(result.exists(), f"{why}: a record was written")
        self.assertEqual(self.ran(), [1], f"{why}: the refusal was not in process 1")
        return done

    # D-SLM8466 (1): MA-1 and MA-2 have no paged arm; a page-id line from one is refused (62)
    def test_t8_ma_run_printing_ids_refuses(self):
        for construction in ("MA-1", "MA-2"):
            with self.subTest(construction=construction):
                done = self.refused_62(construction, f"{construction} with an IDS line", ids_mode="print")
                self.assertIn("page-id line", done.stderr)

    # D-SLM8466 (1), the mirror: a construction with a paged arm that prints no ids, or an empty IDS line, refuses (62)
    def test_t8_paged_run_without_ids_refuses(self):
        for construction in WITH_PAGED_ARM:
            for mode in ("none", "empty"):
                with self.subTest(construction=construction, ids=mode):
                    done = self.refused_62(construction, f"{construction} with ids {mode}", ids_mode=mode)
                    self.assertIn("no page ids", done.stderr)

    # D-SLM8466 (2): each construction's identity line is its own; any other construction's refuses (62), so a
    # production-shaped run cannot be labelled MA-1 (the casebook's owed case) and MR-OPT cannot pass as production
    def test_t8_identity_must_be_the_constructions_own(self):
        for construction in IDENTITIES:
            for other, line in IDENTITIES.items():
                if other == construction:
                    continue
                with self.subTest(construction=construction, prints=other):
                    # the ids follow the label, so only the identity is wrong
                    done = self.refused_62(construction, f"{construction} printing {other}'s identity", identity=line)
                    self.assertIn("identity", done.stderr)
        with self.subTest(case="production-shaped run labelled MA-1"):
            self.refused_62("MA-1", "production's identity and ids under MA-1", identity=IDENTITIES["production"],
                            ids_mode="print")

    # D-SLM8466 (1): MA rows record the guard as not applicable and grade by the row rule alone; the paged rows keep the
    # guard; R79A's aggregate over the four MA runs stands
    def test_t8_ma_rows_guard_not_applicable_and_r79a_stands(self):
        for (construction, row), rec in self.built.items():
            with self.subTest(construction=construction, row=row):
                self.assertEqual(rec.get("status"), "STANDING")
                if construction in WITH_PAGED_ARM:
                    self.assertIsInstance(rec.get("guard"), dict)
                    self.assertEqual(set(rec["guard"]), {"n", "M", "M99", "admits"})
                else:
                    self.assertEqual(rec.get("guard"), NOT_APPLICABLE)
                    self.assertEqual(rec.get("verdict"), "PASS")  # five medians at 1%: the row rule's PASS
                    self.assertNotIn("reason", rec)
        result = self.root / "R79A.json"
        done = self.aggregate("R79A", self.runs(R79A_RUNS), result)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual(self.record(result).get("status"), "STANDING")

    # T-3206 N-2: a reserve is the paged arm's; an MA run printing a RESERVE line refuses (62) with no record, while
    # the constructions with a paged arm record theirs
    def test_t9_ma_run_printing_reserve_refuses_and_paged_runs_record_it(self):
        for construction in ("MA-1", "MA-2"):
            with self.subTest(construction=construction):
                done = self.refused_62(construction, f"{construction} with a RESERVE line", reserve=RESERVE_LINE)
                self.assertIn("RESERVE line", done.stderr)
        want = {"seed": 1, "singles": 63, "source": "tests/paged-kv/pkv_scatter_reserve.h"}
        for construction in WITH_PAGED_ARM:
            with self.subTest(construction=construction):
                result = self.root / f"{construction}.json"
                done = self.drive(self.root, ROW_FOR[construction], result, construction, log=self.log,
                                  reserve=RESERVE_LINE)
                self.assertEqual(done.returncode, 0, done.stderr[-2000:])
                rec = self.record(result)
                self.assertEqual((rec.get("status"), rec.get("reserve")), ("STANDING", want))

    # T-3210 O-4 (D-SLM8494): a construction with a paged arm that prints no RESERVE line refuses (62), the mirror of the
    # MA rule; an MA run with no RESERVE line stands
    def test_t10_paged_run_without_reserve_refuses(self):
        for construction in WITH_PAGED_ARM:
            with self.subTest(construction=construction):
                done = self.refused_62(construction, f"{construction} without RESERVE", reserve="none")
                self.assertIn("no RESERVE line", done.stderr)
        with self.subTest(construction="MA-1"):
            result = self.root / "ma1.json"
            done = self.drive(self.root, ROW_FOR["MA-1"], result, "MA-1", log=self.log)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            rec = self.record(result)
            self.assertEqual((rec.get("status"), rec.get("reserve")), ("STANDING", None))

    # D-SLM8466 (2): argv and identity in every row and process record. The row's argv is the driver's own command line;
    # each process's is the exact pair command it ran
    def test_t8_argv_and_identity_recorded(self):
        root = Path(self.shared.name)
        for (construction, row), rec in self.built.items():
            with self.subTest(construction=construction, row=row):
                self.assertEqual(rec.get("identity"), IDENTITIES[construction])
                argv = rec.get("argv")
                self.assertIsInstance(argv, list)
                self.assertEqual(Path(argv[1]).resolve(), SCRIPT.resolve())
                self.assertEqual(argv[argv.index("--construction") + 1], construction)
                self.assertEqual(argv[argv.index("--slot") + 1], row)
                for k, name in enumerate(rec["processes"], 1):
                    proc = self.record(root / "runs" / name)
                    self.assertEqual(proc.get("identity"), IDENTITIES[construction], name)
                    pargv = proc.get("argv")
                    self.assertIsInstance(pargv, list, name)
                    self.assertEqual(pargv[:3], [sys.executable, "-I", str(self.standin)], name)
                    self.assertEqual(pargv[pargv.index("--process") + 1], str(k), name)
                    self.assertEqual(pargv[pargv.index("--pairs") + 1], "104", name)


class R3Slot(unittest.TestCase):
    """D-SLM8465: the 7.9b leg's record names its run-list slot, R3, and a record without it is rerun, not reused."""

    def setUp(self):
        from types import SimpleNamespace
        from unittest import mock
        self.temp = tempfile.TemporaryDirectory(prefix="t3189-r9-r3-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        path = Path(os.environ.get("PKV_PAIRS_RUNNER") or Path(__file__).with_name("run_paged_kv_pairs.py"))
        spec = importlib.util.spec_from_file_location("paged_kv_pairs_r3", path)
        self.pair = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.pair)
        model = self.root / "model.sslm"
        model.write_bytes(b"model")
        self.args = SimpleNamespace(exe=self.root / "superslm_pkv_pair.exe", kind="79b", model_a=model, model_b=None,
                                    manifest_a=None, manifest_b=None, result=self.root / "r3.json", cap_a=4096,
                                    cap_b=None, layers=2, heads=2, dim=48, width=1712, pairs=508, seed=7904,
                                    cap_seconds=10, mechanism=False)
        self.calls = 0

        def fake_run(command, log, cap, env):
            self.calls += 1
            Path(command[-1]).write_text("".join(f"{i},A,100,101,12,12\n" for i in range(508)))
            return 0
        for name, kw in (("BOX_LOCK", {"new": self.root / "global.lock"}),
                         ("production_library", {"return_value": ("exe", "lib")}),
                         ("box_observation", {"return_value": {"cpu_percent": 1}}),
                         ("run_logged", {"side_effect": fake_run})):
            patcher = mock.patch.object(self.pair, name, **kw)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_r3_record_names_its_slot(self):
        record = self.pair.run(self.args)
        self.assertEqual(record.get("slot"), "R3")
        self.assertEqual(json.loads(self.args.result.read_text(encoding="utf-8")).get("slot"), "R3")

    def test_r3_slotted_record_is_reused_and_unslotted_record_is_rerun(self):
        self.pair.run(self.args)
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.pair.run(self.args).get("slot"), "R3")
        self.assertEqual(self.calls, 1, "a slotted record was not reused")
        on_disk = json.loads(self.args.result.read_text(encoding="utf-8"))
        del on_disk["slot"]
        self.args.result.write_text(json.dumps(on_disk), encoding="utf-8")
        record = self.pair.run(self.args)
        self.assertEqual(self.calls, 2, "a record without its slot was reused")
        self.assertEqual(record.get("slot"), "R3")
        self.assertEqual(json.loads(self.args.result.read_text(encoding="utf-8")).get("slot"), "R3")


class Process(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="t3189-rows-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_boot(self):
        self.assertTrue(rows.same_boot("20261009081500.000000+000", "20261009081500.000000+000"))
        self.assertFalse(rows.same_boot("20261009081500.000000+000", "20261009093000.000000+000"))

    def test_79_10_second_runner_refused_with_another_result_path(self):
        lock = self.root / "pkv-c6-timing.lock"
        holder = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"])
        self.addCleanup(holder.kill)
        lock.write_text(str(holder.pid))  # D-SLM8464: the lock holds the bare pid
        with self.assertRaises(rows.Refusal):
            with rows.timing_lock(self.root / "b.json", lock_path=lock):
                pass

    def test_79_10_dead_owner_recovered(self):
        lock = self.root / "pkv-c6-timing.lock"
        dead = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
        dead.wait()
        lock.write_text(str(dead.pid))  # D-SLM8464: the lock holds the bare pid
        with rows.timing_lock(self.root / "a.json", lock_path=lock):
            self.assertTrue(lock.exists())

    def test_79_11_torn_record_is_missing(self):
        path = self.root / "r.json"
        rows.write_record(path, {"row": "R4a", "medians": [0.01] * 5})
        self.assertEqual(rows.read_record(path)["row"], "R4a")
        path.write_text('{"row": "R4a", "medi')
        self.assertIsNone(rows.read_record(path))
        self.assertIsNone(rows.read_record(self.root / "missing.json"))


if __name__ == "__main__":
    unittest.main()
