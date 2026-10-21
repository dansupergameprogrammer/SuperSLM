"""Must-accept and must-rejects of check_api_ordinals.py's widened locator (precache plan 7.17, rev 1.18.4,
suite audit G11).

Built from the tree's own docs/api.md and include/superslm/sslm_abi.h, edited in a temporary copy:
  must-accept  the tree's docs/api.md as it stands;
  must-reject  the paired sentence (the "ordinal 30" whose status name is on the next line) edited to 31;
  must-reject  an "ordinal N" with no backticked status name on its line or the next;
  must-reject  the paired sentence reworded past the locator, which drops SSLM_RESTORE_PRODUCER_MISMATCH below
               its floor of 2.
The independent constructions of plan section 9, E5 stay the debunker's; these pin the widening itself.
"""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_api_ordinals import check  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/api.md"
HEADER = ROOT / "include/superslm/sslm_abi.h"
PAIRED = "One covers precached prefixes (1.16.0, unreleased; ordinal 30):"


def refuses(text: str, what: str, directory: Path) -> bool:
    doc = directory / "api.md"
    doc.write_text(text, encoding="utf-8")
    try:
        check(doc, HEADER)
    except ValueError as exc:
        print(f"must-reject ({what}): refused: {exc}")
        return True
    print(f"must-reject ({what}): ACCEPTED", file=sys.stderr)
    return False


def main() -> int:
    text = DOC.read_text(encoding="utf-8")
    if text.count(PAIRED) != 1:
        print(f"the paired sentence is not in {DOC} exactly once: {PAIRED!r}", file=sys.stderr)
        return 1
    ok = True
    found = check(DOC, HEADER)
    paired = [f for f in found if f[1] == "SSLM_RESTORE_PRODUCER_MISMATCH"]
    if len(paired) < 2:
        print(f"must-accept: the paired sentence is not read ({len(paired)} statement(s) of the new status)",
              file=sys.stderr)
        ok = False
    else:
        print(f"must-accept: {len(found)} statements, {len(paired)} of SSLM_RESTORE_PRODUCER_MISMATCH")
    with TemporaryDirectory() as d:
        directory = Path(d)
        ok &= refuses(text.replace(PAIRED, PAIRED.replace("ordinal 30", "ordinal 31")), "paired sentence at 31",
                      directory)
        ok &= refuses(text + "\nA status of no name here (ordinal 12).\n\nAnd none on this line either.\n",
                      "an ordinal with no name on its line or the next", directory)
        ok &= refuses(text.replace(PAIRED, PAIRED.replace("ordinal 30", "the next free value")),
                      "paired sentence reworded past the locator", directory)
    print("test_check_api_ordinals: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
