"""Plan 7.17 [E5]: the test author's dry run of the section 8 constructions, ready for the debunker.

The debunker authors and commissions 7.17's constructions independently (plan 7.17, section 8, section 9 E5).
This script is the test author's half: it builds the anchor doc and the section 8 must-accept and must-rejects
from git, runs tests/ci/check_api_ordinals.py's check() on each against this tree's X-macro, and prints one
verdict per construction. Nothing here is the commissioning.

  anchor              docs/api.md at 5d5ca58, as is: 7.17 is red on it for its own reason (the new status
                      is not stated).
  accept-one          section 8's must-accept as worded: the anchor plus ONE correct statement of
                      SSLM_RESTORE_PRODUCER_MISMATCH at 30. Rev 1.18.4 raised that name's floor to 2, so this is
                      refused by the floor: section 8's must-accept and 7.17's rev 1.18.4 floor disagree.
  accept-two          the anchor plus the two statement forms rev 1.18.4 reads: an own-line statement and a
                      paired one ("ordinal 30" with the name on the next line). Accepted.
  reject-kv-30        accept-two with one SSLM_KV_BUDGET_EXCEEDED statement edited to 30.
  reject-new-31       accept-two with the own-line SSLM_RESTORE_PRODUCER_MISMATCH statement at 31.
  reject-paired-31    accept-two with the paired statement at 31 (rev 1.18.4, G11).
  reject-kv-reworded  accept-two with one SSLM_KV_BUDGET_EXCEEDED statement reworded past the locator.
  reject-unpaired     accept-two plus an "ordinal N" with no status name on its line or the next.

Usage: python3 run_717_constructions.py [REPO]   (REPO defaults to the repository holding this file)
"""
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo / "tests/ci"))
from check_api_ordinals import check  # noqa: E402

HEADER = repo / "include/superslm/sslm_abi.h"
ANCHOR = subprocess.run(["git", "-C", str(repo), "show", "5d5ca58:docs/api.md"], check=True,
                        capture_output=True, text=True).stdout
KV_LINE = "declared budget exceeded below the cap (`SSLM_KV_BUDGET_EXCEEDED`, ordinal 29,"
OWN = "A restore of a blob another engine version wrote returns `SSLM_RESTORE_PRODUCER_MISMATCH` (ordinal 30)."
PAIRED = "One status covers precached prefixes (ordinal 30):\n`SSLM_RESTORE_PRODUCER_MISMATCH`, a blob from another producer."
assert ANCHOR.count(KV_LINE) == 1


def with_statements(text: str, *added: str) -> str:
    return text.rstrip("\n") + "\n\n" + "\n\n".join(added) + "\n"


two = with_statements(ANCHOR, OWN, PAIRED)
cases = [
    ("anchor", ANCHOR, False),
    ("accept-one", with_statements(ANCHOR, OWN), True),
    ("accept-two", two, True),
    ("reject-kv-30", two.replace(KV_LINE, KV_LINE.replace("ordinal 29", "ordinal 30")), False),
    ("reject-new-31", two.replace(OWN, OWN.replace("ordinal 30", "ordinal 31")), False),
    ("reject-paired-31", two.replace(PAIRED, PAIRED.replace("ordinal 30", "ordinal 31")), False),
    ("reject-kv-reworded", two.replace(KV_LINE, KV_LINE.replace("ordinal 29", "the value after 28")), False),
    ("reject-unpaired", with_statements(two, "A status of no name (ordinal 12).\n\nNone here either."), False),
]
with TemporaryDirectory() as d:
    for name, text, want_accept in cases:
        doc = Path(d) / f"{name}.md"
        doc.write_text(text, encoding="utf-8")
        try:
            found = check(doc, HEADER)
            verdict, detail = "ACCEPTED", f"{len(found)} statements"
        except ValueError as exc:
            verdict, detail = "REFUSED", str(exc).replace(str(doc), name + ".md")
        note = "" if (verdict == "ACCEPTED") == want_accept else "  <- differs from the section 8 wording"
        print(f"{name:20s} {verdict:8s} {detail}{note}")
