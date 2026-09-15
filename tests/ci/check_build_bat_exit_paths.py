"""Reject batch-script exits that can escape a nested cmd.exe block.

``cmd.exe`` can make ``exit /b`` inside a parenthesized block return from an
unexpected batch-call frame.  build.bat therefore has exactly three legal
exit forms: its pre-pushd toolchain guard, its normal final status return,
and the top-level :hard_fail epilogue.  Every operational hard stop must
jump to that epilogue instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


_EXIT_RE = re.compile(r"\bexit\s+/b\b", re.IGNORECASE)
_LABEL_RE = re.compile(r"^\s*(:[^\s]+)\s*$")
_COMMENT_RE = re.compile(r"^\s*(?:rem\b|::)", re.IGNORECASE)


def _structural_parens(text: str, stop: int | None = None) -> int:
    """Count cmd grouping parens, ignoring caret escapes and quoted strings."""
    depth = 0
    quoted = False
    escaped = False
    for char in text[:stop]:
        if escaped:
            escaped = False
        elif char == "^":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char == "(":
            depth += 1
        elif not quoted and char == ")":
            depth -= 1
    return depth


def check_text(text: str) -> list[str]:
    """Return every prohibited build.bat ``exit /b`` occurrence."""
    errors: list[str] = []
    depth = 0
    seen_pushd = False
    current_label: str | None = None
    ordinary_exit_count = 0

    for number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if _COMMENT_RE.match(raw_line):
            continue

        label = _LABEL_RE.match(raw_line)
        if label:
            current_label = label.group(1).lower()

        for match in _EXIT_RE.finditer(raw_line):
            exit_depth = depth + _structural_parens(raw_line, match.start())
            normalized = stripped.lower()
            if normalized == "exit /b 1" and number == 15 and not seen_pushd:
                # The sole early toolchain guard is necessarily parenthesized.
                continue
            if exit_depth:
                errors.append(f"line {number}: exit /b is inside parenthesized block")
                continue

            if normalized == "exit /b 1" and current_label == ":hard_fail":
                continue
            if normalized == "exit /b %ec%":
                ordinary_exit_count += 1
                continue
            errors.append(f"line {number}: unapproved top-level exit /b: {stripped}")

        if re.search(r"\bpushd\b", raw_line, re.IGNORECASE):
            seen_pushd = True
        depth += _structural_parens(raw_line)
        if depth < 0:
            errors.append(f"line {number}: unmatched closing parenthesis")
            depth = 0

    if depth:
        errors.append(f"unclosed parenthesized block(s): depth {depth}")
    if ordinary_exit_count != 1:
        errors.append(f"expected exactly one final exit /b %ec%, found {ordinary_exit_count}")
    return errors


def _read_input(path: Path, revision: str | None) -> str:
    if revision is None:
        return path.read_text(encoding="utf-8")
    return subprocess.check_output(
        ["git", "show", f"{revision}:build.bat"], text=True, encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-bat", type=Path,
                        default=Path(__file__).resolve().parents[2] / "build.bat")
    parser.add_argument("--git-revision", help="check build.bat from this git revision")
    args = parser.parse_args(argv)
    try:
        errors = check_text(_read_input(args.build_bat, args.git_revision))
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: cannot read build.bat: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("FAIL: build.bat exit-path policy violated", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("PASS: build.bat exit paths use only approved top-level exits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
