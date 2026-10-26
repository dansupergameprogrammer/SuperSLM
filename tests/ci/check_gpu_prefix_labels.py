"""Require this plan's GPU prefix twins to be labelled released in 1.16.0 (precache plan cell 7.22's inverse).

Until 1.16.0 this check required the label "unreleased until 1.16.0" (cell 7.22, run at M1, E5 and the 1.15.0
release). Plan section 9, F4 lifts that label and runs the inverse; this is the inverse. Two rules:

1. No mention of sslm_gpu_prefix_save or sslm_gpu_prefix_restore in include/superslm/gpu_1p0.h or
   docs/api.md, and not docs/api.md's status-30 sentence for the GPU twins, still calls them
   "unreleased". The unit is the one the mention sits in: in the header, the comment holding the
   mention, or for a declaration the run of non-blank lines holding it (its doc comment and the
   declarations); in docs/api.md, the list item or paragraph. Text is compared with comment leaders
   and line breaks folded to single spaces, so a wrapped word counts. The status-30 sentence must
   still exist and name 1.16.0.

2. No release number other than 1.16.0 stands beside this plan's GPU code: in each file the section 4
   preamble lists, and in any other tracked file the sweep finds, a label site is a line naming
   "precached prefixes" or one of the two GPU verbs, in GPU code (the file is a listed one, its path
   names gpu, or a line within two of it names a GPU identifier); in docs/api.md it is the list item
   or paragraph naming a GPU verb. Every release number at a label site must be 1.16.0, and each listed
   file must keep at least the label sites the preamble counted, so a label deleted rather than
   relabelled drops out loudly.

The independent must-accept and must-reject constructions are the debunker's (plan sections 8, 9).
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys

STALE = re.compile(r"\bunreleased\b|\bnot (?:yet )?released\b|\buntil 1\.16\.0\b", re.I)
RELEASE = "1.16.0"
VERBS = re.compile(r"\bsslm_gpu_prefix_(?:save|restore)\b")
PRECACHED = re.compile(r"precached[ -]prefix", re.I)
VERSION = re.compile(r"(?<![\w.])(\d+\.\d+\.\d+)(?![\w.]*\d)")
GPU_IDENT = re.compile(r"SslmGpu|sslm_gpu_|SSLM_GPU_")
DOC = "docs/api.md"
HEADER = "include/superslm/gpu_1p0.h"
# The section 4 preamble's list at 83a82c2: file -> label sites counted there.
LISTED = {
    "include/superslm/gpu_1p0.h": 2,
    "include/superslm/gpu_1p0_bench_bridge.h": 1,
    "include/superslm/gpu_port.h": 1,
    "src/gpu/gpu_1p0.cpp": 2,
    "docs/api.md": 3,
    "tests/t2112-gpu-1p0-red-suite/sslm_gpu_1p0.h": 1,
    "tests/t2791-gpu-prefill-read-red-suite/cell_status_ordinals.cpp": 1,
    "tests/t2956-token-finish-red-suite/cell_status_ordinals.cpp": 1,
}
TEXT_SUFFIXES = {".h", ".hpp", ".c", ".cc", ".cpp", ".inc", ".md", ".txt", ".py", ".cmake", ".ps1",
                 ".bat", ".comp", ".glsl", ".hlsl", ".json"}


def fold(text: str) -> str:
    text = re.sub(r"(?m)^\s*(?:/\*+|\*+/?|//+)", " ", text)
    text = text.replace("*/", " ").replace("/*", " ")
    return re.sub(r"\s+", " ", text)


def doc_blocks(lines: list) -> list:
    """docs/api.md units: paragraphs, with a list item starting a new unit. (first, last) 0-based."""
    blocks, start = [], None
    for i, line in enumerate(lines):
        item = re.match(r"\s*(?:[-*+]|\d+\.)\s", line) is not None
        if not line.strip():
            if start is not None:
                blocks.append((start, i - 1))
            start = None
        elif start is None:
            start = i
        elif item:
            blocks.append((start, i - 1))
            start = i
    if start is not None:
        blocks.append((start, len(lines) - 1))
    return blocks


def comment_spans(source: str) -> list:
    """(first, last) 0-based line spans of each /* */ comment and each run of // lines."""
    spans = []
    for m in re.finditer(r"/\*.*?\*/", source, re.S):
        spans.append((source.count("\n", 0, m.start()), source.count("\n", 0, m.end())))
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("//"):
            j = i
            while j + 1 < len(lines) and lines[j + 1].lstrip().startswith("//"):
                j += 1
            spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


def header_unit(lines: list, spans: list, index: int) -> tuple:
    for first, last in spans:
        if first <= index <= last and "/*" in "\n".join(lines[first:last + 1]):
            # The mention is inside a comment only if it is not in the code after the closing */.
            closing = lines[last].find("*/")
            if index < last or closing < 0 or lines[index].find("sslm_gpu_prefix") < closing:
                return first, last
    first = index
    while first > 0 and lines[first - 1].strip():
        first -= 1
    last = index
    while last + 1 < len(lines) and lines[last + 1].strip():
        last += 1
    return first, last


def rule_one(root: Path, problems: list) -> int:
    checked = 0
    header = (root / HEADER).read_text(encoding="utf-8")
    hlines = header.splitlines()
    spans = comment_spans(header)
    for i, line in enumerate(hlines):
        if VERBS.search(line):
            first, last = header_unit(hlines, spans, i)
            checked += 1
            if STALE.search(fold("\n".join(hlines[first:last + 1]))):
                problems.append(f"{HEADER}:{i + 1}: mention still labelled \"unreleased\" in lines "
                                f"{first + 1}-{last + 1}")
    dlines = (root / DOC).read_text(encoding="utf-8").splitlines()
    status_sentences = 0
    for first, last in doc_blocks(dlines):
        text = fold("\n".join(dlines[first:last + 1]))
        mentions = VERBS.search(text) is not None
        status30 = mentions and re.search(r"\bordinal\s+30\b", text, re.I) is not None
        status_sentences += status30
        if status30 and RELEASE not in text:
            problems.append(f"{DOC}:{first + 1}: the status-30 sentence does not name {RELEASE}")
        if mentions:
            checked += 1
            if STALE.search(text):
                what = "the status-30 sentence" if status30 else "mention"
                problems.append(f"{DOC}:{first + 1}: {what} still labelled \"unreleased\" in lines "
                                f"{first + 1}-{last + 1}")
    if status_sentences < 1:
        problems.append(f"{DOC}: no paragraph pairs a GPU prefix verb with ordinal 30 (the status-30 "
                        "sentence was reworded past the locator, or removed)")
    return checked


def tracked_files(root: Path) -> list:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files"], check=True, capture_output=True,
                             text=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        out = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    return sorted(set(out) | set(LISTED))


def label_sites(path: str, lines: list) -> list:
    """(line number, release numbers) of each label site in one file."""
    sites = []
    if path == DOC:
        for first, last in doc_blocks(lines):
            text = "\n".join(lines[first:last + 1])
            if VERBS.search(text):
                for k in range(first, last + 1):
                    found = VERSION.findall(lines[k])
                    if found:
                        sites.append((k + 1, found))
        return sites
    gpu_file = path in LISTED or "gpu" in path.lower()
    for k, line in enumerate(lines):
        if not (PRECACHED.search(line) or VERBS.search(line)):
            continue
        found = VERSION.findall(line)
        if not found:
            continue
        near = "\n".join(lines[max(0, k - 2):k + 3])
        if gpu_file or GPU_IDENT.search(near):
            sites.append((k + 1, found))
    return sites


def rule_two(root: Path, problems: list) -> int:
    total = 0
    for path in tracked_files(root):
        full = root / path
        if path not in LISTED and full.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            lines = full.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            if path in LISTED:
                problems.append(f"{path}: listed in the section 4 preamble but unreadable")
            continue
        sites = label_sites(path, lines)
        total += len(sites)
        for number, found in sites:
            stale = [v for v in found if v != RELEASE]
            if stale:
                problems.append(f"{path}:{number}: {', '.join(stale)} beside this plan's GPU code "
                                f"(only {RELEASE} may stand there)")
        floor = LISTED.get(path, 0)
        if len(sites) < floor:
            problems.append(f"{path}: {len(sites)} label site(s), floor {floor} (a label deleted "
                            "rather than relabelled)")
    return total


def check(root: Path) -> tuple:
    problems = []
    mentions = rule_one(root, problems)
    sites = rule_two(root, problems)
    if problems:
        raise ValueError("\n  ".join([f"{len(problems)} problem(s):"] + problems))
    return mentions, sites


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        mentions, sites = check(args.root.resolve())
    except (OSError, ValueError) as exc:
        print(f"check_gpu_prefix_labels: {exc}", file=sys.stderr)
        return 1
    print(f"check_gpu_prefix_labels: {mentions} mention unit(s), none labelled unreleased; "
          f"{sites} label site(s) read {RELEASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
