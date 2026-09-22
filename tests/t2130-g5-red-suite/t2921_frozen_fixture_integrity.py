"""T-2921: historical compiler fixtures stay byte-stable and self-contained."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2913_common as common


class T2921_FrozenHistoricalFixtureIntegrity(unittest.TestCase):
    def test_every_historical_module_is_present_and_hash_pinned(self) -> None:
        inventory = common.frozen_fixture_inventory()
        self.assertEqual(len(inventory), 4)
        for (commit, source_path), (fixture, expected_sha256) in inventory.items():
            with self.subTest(commit=commit, source_path=source_path):
                common._require(fixture)
                common._verify_sha256(
                    fixture, expected_sha256, f"frozen fixture {commit}:{source_path}"
                )
