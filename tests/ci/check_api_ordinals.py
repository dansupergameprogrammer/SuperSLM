"""Require every status ordinal docs/api.md states to be that status's ordinal (precache plan cell 7.17).

A statement is one line of docs/api.md that pairs one backticked status name of the
SSLM_STATUS_ENUM_LIST X-macro with the word "ordinal" and the integer immediately following it
("ordinals 24 to 29" is a range, not a statement, and is not read). Each stated integer must equal the
name's position in the X-macro. SSLM_RESTORE_PRODUCER_MISMATCH must be stated, at 30. A name with
fewer statements than its recorded floor refuses, so a statement reworded past the locator drops out
loudly instead of letting the check pass on fewer statements.

The independent must-accept and must-reject constructions are the debunker's (plan section 9, E5).
"""

import argparse
from pathlib import Path
import re
import sys

# Recorded floors (plan 7.17): the statements counted at the anchor, plus the new status once.
FLOORS = {"SSLM_KV_BUDGET_EXCEEDED": 2, "SSLM_RESTORE_PRODUCER_MISMATCH": 1}
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
    for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        stated = _ORDINAL.findall(line)
        if not stated:
            continue
        names = sorted({n for n in _BACKTICKED.findall(line) if n in ordinals})
        if not names:
            continue
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
