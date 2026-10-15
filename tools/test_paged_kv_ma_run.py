"""Cell 79-17 (paged-KV plan rev 16.28 §30.3, D-SLM8478): MA's run mode through the real driver, off the box, on B-FR.
Written by the test author blind to tools/paged_kv_pair_ma.cpp, against §29.4, §30.3 and §30.5.

MA-1 and MA-2 each run as one driver row at R2b on the 0.5B cap-4,096 artifact (--construction MA-1 / MA-2,
--page-positions 64), on the R2b blob chain (restore 3,988) made by make_paged_kv_row_chain.py on B-FR. Asserted:
exit 0; five process records, each with `identity` the construction's string (written here literally) and `argv` the
command run; 104 pairs per process, each pair's two tokens equal; no page ids and no reserve; `guard: "not applicable (no
paged arm)"`; the row record STANDING. The verdict is not asserted: off-box timing is recorded, never graded (MA-2 reads
FAIL at R2b by construction, its offset being about 9% of the 0.5B token).

Environment (all required except PKV_7917_MA and PKV_7917_OUT; missing fails the cell, never skips it):
  PKV_BFR          B-FR's directory (superslm_pkv_pair_ma.exe)
  PKV_7917_MODEL   the 0.5B cap-4,096 artifact
  PKV_7917_CHAIN   the directory holding manifest_r3988.json (the R2b chain)
  PKV_7917_MA      an executable to run in MA's place (the test author's output mutants; default B-FR's MA)
  PKV_7917_OUT     a directory to keep the records in (default: a temporary directory)
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parent
DRIVER = TOOLS / "run_paged_kv_rows.py"
IDENTITY = {"MA-1": "PKV_PAIR_MA1_V1 defs=none", "MA-2": "PKV_PAIR_MA2_V1 defs=none"}
ARG = {"MA-1": "ma1", "MA-2": "ma2"}
GEOMETRY = ["4096", "24", "2", "64"]  # the 0.5B: cap, layers, KV heads, head dimension
NOT_APPLICABLE = "not applicable (no paged arm)"


class Cell7917(unittest.TestCase):
    def setUp(self):
        need = {k: os.environ.get(k, "") for k in ("PKV_BFR", "PKV_7917_MODEL", "PKV_7917_CHAIN")}
        missing = [k for k, v in need.items() if not v]
        if missing:
            self.fail(f"79-17 needs {', '.join(missing)}; there is no default")
        self.ma = Path(os.environ.get("PKV_7917_MA") or Path(need["PKV_BFR"]) / "superslm_pkv_pair_ma.exe")
        self.model = Path(need["PKV_7917_MODEL"])
        self.manifest = Path(need["PKV_7917_CHAIN"]) / "manifest_r3988.json"
        for path in (self.ma, self.model, self.manifest):
            self.assertTrue(path.is_file(), f"{path} does not exist")
        out = os.environ.get("PKV_7917_OUT")
        if out:
            self.out = Path(out)
            self.out.mkdir(parents=True, exist_ok=True)
        else:
            temp = tempfile.TemporaryDirectory(prefix="t3189-7917-")
            self.addCleanup(temp.cleanup)
            self.out = Path(temp.name)

    def run_row(self, construction):
        pair_cmd = [str(self.ma), ARG[construction], *GEOMETRY]
        result = self.out / f"{construction}-R2b.json"
        cmd = [sys.executable, "-I", str(DRIVER), "--pair-cmd", json.dumps(pair_cmd), "--manifest", str(self.manifest),
               "--model", str(self.model), "--restore", "3988", "--slot", "R2b", "--construction", construction,
               "--page-positions", "64", "--result", str(result), "--lock", str(self.out / "pkv-c6-timing.lock")]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=45 * 60)
        (self.out / f"{construction}-R2b.driver.log").write_text(done.stdout + done.stderr, encoding="utf-8")
        return done, pair_cmd, result

    def test_79_17_ma_rows_through_the_driver(self):
        for construction in ("MA-1", "MA-2"):
            with self.subTest(construction=construction):
                done, pair_cmd, result = self.run_row(construction)
                self.assertEqual(done.returncode, 0, done.stderr[-3000:])
                row = json.loads(result.read_text(encoding="utf-8"))
                self.assertEqual((row.get("status"), row.get("construction"), row.get("page_positions"),
                                  row.get("identity"), row.get("guard")),
                                 ("STANDING", construction, 64, IDENTITY[construction], NOT_APPLICABLE))
                self.assertIn(row.get("verdict"), ("PASS", "FAIL", "NO RESULT"))  # recorded, never graded here
                self.assertEqual(len(row.get("processes", [])), 5)
                for k, name in enumerate(row["processes"], 1):
                    proc = json.loads((result.parent / name).read_text(encoding="utf-8"))
                    self.assertEqual(proc.get("identity"), IDENTITY[construction], name)
                    argv = proc.get("argv")
                    self.assertEqual(argv[:len(pair_cmd)], pair_cmd, name)
                    self.assertEqual(argv[argv.index("--process") + 1], str(k), name)
                    self.assertEqual(argv[argv.index("--pairs") + 1], "104", name)
                    pairs = proc.get("pairs")
                    self.assertEqual(len(pairs), 104, name)
                    self.assertTrue(all(p[2] == p[3] for p in pairs), f"{name}: a pair's two tokens differ")
                    self.assertIsNone(proc.get("ids"), name)
                    self.assertIsNone(proc.get("reserve"), name)


if __name__ == "__main__":
    unittest.main()
