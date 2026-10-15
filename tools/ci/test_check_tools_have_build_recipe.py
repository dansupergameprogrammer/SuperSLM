"""A tool built by a reachable CMake include has a committed recipe."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import check_tools_have_build_recipe as gate


class IncludedCmakeRecipe(unittest.TestCase):
    def test_reachable_include_counts_and_dormant_fragment_does_not(self):
        with tempfile.TemporaryDirectory(prefix="tool-recipe-") as folder:
            root = Path(folder)
            (root / "cmake").mkdir()
            (root / "CMakeLists.txt").write_text("include(cmake/active.cmake)\n", encoding="utf-8")
            (root / "cmake/active.cmake").write_text(
                "add_executable(active tools/active.cpp)\n", encoding="utf-8")
            (root / "cmake/dormant.cmake").write_text(
                "add_executable(dormant tools/dormant.cpp)\n", encoding="utf-8")
            with patch.object(gate, "REPO_ROOT", root), patch.object(gate, "TOOLS_DIR", root / "tools"):
                recipe = "\n".join(gate.recipe_texts())
            self.assertIn("tools/active.cpp", recipe)
            self.assertNotIn("tools/dormant.cpp", recipe)


if __name__ == "__main__":
    unittest.main()
