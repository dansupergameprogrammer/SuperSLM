"""T-2703 F5 — keep the normative header-mask row synchronized with C++."""

from __future__ import annotations

import re
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _documented_header_mask() -> int:
    """Return the mask from the flags row in the format document's header table."""
    header = (_ROOT / "docs" / "sslm_format.md").read_text(encoding="utf-8")
    header_table = header.split("### Section table", maxsplit=1)[0]
    match = re.search(r"\|\s*16\s*\|\s*`u32`\s*\|\s*`flags`\s*\|\s*`& ~(0x[0-9a-fA-F]+) == 0`", header_table)
    assert match, "the normative header flags row must declare an explicit known-bit mask"
    return int(match.group(1), 16)


def _compiled_header_mask() -> int:
    """Evaluate the named flag constants used by kKnownArtifactFlagsMask."""
    source = (_ROOT / "include" / "superslm" / "artifact.h").read_text(encoding="utf-8")
    values = {
        name: int(value, 16)
        for name, value in re.findall(
            r"inline constexpr uint32_t (k\w+Flag) = (0x[0-9a-fA-F]+)u;", source)
    }
    expression = re.search(
        r"inline constexpr uint32_t kKnownArtifactFlagsMask\s*=\s*(.*?);",
        source,
        flags=re.DOTALL,
    )
    assert expression, "artifact.h must define kKnownArtifactFlagsMask"
    names = re.findall(r"k\w+Flag", expression.group(1))
    assert names, "kKnownArtifactFlagsMask must name its component flags"
    mask = 0
    for name in names:
        assert name in values, f"kKnownArtifactFlagsMask references undefined flag {name}"
        mask |= values[name]
    return mask


def test_documented_header_known_flag_mask_matches_compiled_contract():
    assert _documented_header_mask() == _compiled_header_mask()
