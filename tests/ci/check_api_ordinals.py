"""Require every status ordinal docs/api.md states to be that status's ordinal (precache plan cell 7.17).

Every "ordinal N" in docs/api.md is read (rev 1.18.4, suite audit G11): the integer immediately
following the word "ordinal" ("ordinals 24 to 29" is a range, not a statement, and is not read). It
pairs with the one backticked status name of the SSLM_STATUS_ENUM_LIST X-macro on its own line; a line
with no such name pairs with the first one on the next line of the same paragraph; with none on either
line the statement is unpaired and the check refuses. Each stated integer must equal the name's
position in the X-macro. SSLM_RESTORE_PRODUCER_MISMATCH must be stated, at 30. A name with
fewer statements than its recorded floor refuses, so a statement reworded past the locator drops out
loudly instead of letting the check pass on fewer statements.

The independent must-accept and must-reject constructions are the debunker's (plan section 9, E5).
"""

import argparse
from pathlib import Path
import re
import sys

# Recorded floors (plan 7.17): the statements counted at the anchor, plus the new status once.
# Rev 1.18.4 (G11): the new status's floor is 2, its paired sentence and its own-line statement.
FLOORS = {"SSLM_KV_BUDGET_EXCEEDED": 2, "SSLM_RESTORE_PRODUCER_MISMATCH": 2}
# The new status and the ordinal the release states it at.
REQUIRED = {"SSLM_RESTORE_PRODUCER_MISMATCH": 30}

_ORDINAL = re.compile(r"\bordinal\s+(\d+)\b", re.I)
_BACKTICKED = re.compile(r"`(SSLM_[A-Z0-9_]+)`")


def x_macro_ordinals(header: Path) -> dict:
    source = header.read_text(encoding="utf-8")
    start = re.search(r"^\s*#\s*define\s+SSLM_STATUS_ENUM_LIST\s*\(\s*X\s*\)", source, re.M)
    if start is None:
        raise ValueError(f"no SSLM_STATUS_ENUM_LIST(X) definition in {header}")
    # Comments first: a comment spanning lines inside the macro needs no continuation backslash.
    rest = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)),
                  source[start.end():], flags=re.S)
    body = []
    for line in rest.splitlines():
        body.append(line)
        if line.strip() and not line.rstrip().endswith("\\"):
            break
    text = "\n".join(body)
    names = re.findall(r"\bX\s*\(\s*([A-Z][A-Z0-9_]*)\s*\)", text)
    if not names or names[0] != "SSLM_OK":
        raise ValueError("SSLM_STATUS_ENUM_LIST must start with X(SSLM_OK)")
    if len(set(names)) != len(names):
        raise ValueError("SSLM_STATUS_ENUM_LIST names a status twice")
    return {name: index for index, name in enumerate(names)}


def statements(doc: Path, ordinals: dict) -> list:
    found = []
    lines = doc.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        stated = _ORDINAL.findall(line)
        if not stated:
            continue
        names = sorted({n for n in _BACKTICKED.findall(line) if n in ordinals})
        if not names:
            # The next line of the same paragraph: its first backticked status name.
            following = lines[number] if number < len(lines) else ""
            nxt = [n for n in _BACKTICKED.findall(following) if n in ordinals] if following.strip() else []
            if not nxt:
                raise ValueError(f"{doc}:{number}: \"ordinal {stated[0]}\" with no backticked status name on "
                                 f"its line or the next line of its paragraph: an unpaired statement")
            names = [nxt[0]]
        if len(names) != 1 or len(stated) != 1:
            raise ValueError(f"{doc}:{number}: a line pairing {len(names)} status names with "
                             f"{len(stated)} ordinals is not one statement; split it")
        found.append((number, names[0], int(stated[0])))
    return found


def check(doc: Path, header: Path) -> list:
    ordinals = x_macro_ordinals(header)
    for name, value in REQUIRED.items():
        if ordinals.get(name) != value:
            raise ValueError(f"{name} is {ordinals.get(name)} in the X-macro, not {value}")
    found = statements(doc, ordinals)
    for number, name, value in found:
        if value != ordinals[name]:
            raise ValueError(f"{doc}:{number}: {name} stated at ordinal {value}; the X-macro has "
                             f"{ordinals[name]}")
    for name, floor in FLOORS.items():
        count = sum(1 for _, n, _ in found if n == name)
        if count < floor:
            raise ValueError(f"{name}: {count} statement(s) found, floor {floor} (a statement reworded "
                             f"past the locator, or never written)")
    return found


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=root / "docs/api.md")
    parser.add_argument("--header", type=Path, default=root / "include/superslm/sslm_abi.h")
    args = parser.parse_args()
    try:
        found = check(args.doc, args.header)
    except (OSError, ValueError) as exc:
        print(f"check_api_ordinals: {exc}", file=sys.stderr)
        return 1
    for number, name, value in found:
        print(f"check_api_ordinals: {args.doc.name}:{number} {name} = {value}")
    print(f"check_api_ordinals: {len(found)} stated ordinal(s) match the X-macro")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
