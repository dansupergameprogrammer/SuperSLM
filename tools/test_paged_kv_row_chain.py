"""Cells for tools/make_paged_kv_row_chain.py's resume (T-3202 F-6, SuperSLM 1478be7; the test author's record T-3189
§12): each piece records its maker's SHA-256, and a chain resumed by another executable refuses rather than relabel
pieces it did not make.

A fake pair executable stands in for superslm_pkv_pair's `prefill` mode: a .cmd (Windows) or shell wrapper around a
Python script that writes a deterministic blob for (start, count) to the output path. No model is loaded.

PKV_ROW_CHAIN_MAKER, when set, names the maker under test (the test author runs the remedy reverted through it).

  F-6  make_chain records exe_sha256 on every piece; resuming the same chain with the same executable reuses every
       piece; resuming with another executable refuses (Refusal naming the piece); a piece recorded without
       exe_sha256 refuses; a piece that no longer verifies is dropped from the reusable prefix as before
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPT = Path(os.environ.get("PKV_ROW_CHAIN_MAKER") or Path(__file__).with_name("make_paged_kv_row_chain.py"))
spec = importlib.util.spec_from_file_location("row_chain", SCRIPT)
chain_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain_mod)

FAKE_PAIR = r'''
import sys
from pathlib import Path
args = sys.argv[1:]
assert args[0] == "prefill"
start, count, out = int(args[7]), int(args[8]), Path(args[10])
out.write_bytes(f"blob {start} {count}\n".encode())
print("PREFILL_OK")
'''


class RowChain(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="t3189-chain-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "fake_pair.py").write_text(FAKE_PAIR)
        self.model = self.root / "model.sslm"
        self.model.write_bytes(b"model bytes")
        self.chain_dir = self.root / "chain"

    def wrapper(self, name, tag):
        """A distinct executable per `tag`: same behaviour, different bytes, so a different SHA-256."""
        if os.name == "nt":
            path = self.root / f"{name}.cmd"
            path.write_text(f'@rem {tag}\n@"{sys.executable}" -I "{self.root / "fake_pair.py"}" %*\n')
        else:
            path = self.root / name
            path.write_text(f'#!/bin/sh\n# {tag}\nexec "{sys.executable}" -I "{self.root / "fake_pair.py"}" "$@"\n')
            path.chmod(0o755)
        return path

    def make(self, exe, restores=(300, 700)):
        return chain_mod.make_chain(exe, self.model, 4096, 2, 2, 48, list(restores), self.chain_dir,
                                    cap_seconds=60, fast_limit=256, slow_limit=256)

    def chain(self):
        return json.loads((self.chain_dir / "chain.json").read_text(encoding="utf-8"))

    def test_f6_pieces_record_their_maker(self):
        a = self.wrapper("pair_a", "a")
        self.make(a)
        pieces = self.chain()["pieces"]
        self.assertGreater(len(pieces), 1)
        for p in pieces:
            self.assertEqual(p.get("exe_sha256"), chain_mod.sha(a), p)

    def test_f6_same_executable_reuses_every_piece(self):
        a = self.wrapper("pair_a", "a")
        self.make(a, (300,))
        made = self.chain()["pieces"]
        self.make(a, (300, 700))  # resumes: the 300-token prefix is reused, the rest is made
        again = self.chain()["pieces"]
        self.assertEqual(again[:len(made)], made)

    def test_f6_another_executable_refuses_to_resume(self):
        a, b = self.wrapper("pair_a", "a"), self.wrapper("pair_b", "b")
        self.make(a, (300,))
        with self.assertRaises(chain_mod.Refusal) as caught:
            self.make(b, (300, 700))
        self.assertIn("was made by executable", str(caught.exception))

    def test_f6_piece_without_a_maker_refuses(self):
        a = self.wrapper("pair_a", "a")
        self.make(a, (300,))
        data = self.chain()
        for p in data["pieces"]:
            p.pop("exe_sha256", None)
        (self.chain_dir / "chain.json").write_text(json.dumps(data))
        with self.assertRaises(chain_mod.Refusal):
            self.make(a, (300, 700))

    def test_f6_a_piece_that_no_longer_verifies_is_dropped(self):
        a = self.wrapper("pair_a", "a")
        self.make(a, (300,))
        first = self.chain()["pieces"][0]
        (self.chain_dir / first["file"]).write_bytes(b"damaged")
        self.make(a, (300,))  # nothing reusable: the chain is remade from 0, no refusal
        self.assertEqual(self.chain()["pieces"][0]["start"], 0)


if __name__ == "__main__":
    unittest.main()
