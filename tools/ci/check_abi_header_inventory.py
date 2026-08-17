#!/usr/bin/env python3
"""T-2139 sixth confirmation review, M3's structural half (Claude/Poirot/
5fbd04d-t2139-sixth-confirmation-review.md item 5; ruled in
Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md Sec8, the "structural half" paragraph).

Three consecutive confirmation rounds found a macro landed on the frozen public ABI header
(include/superslm/sslm_abi.h) with no Sec8 inventory line naming it -- SSLM_STATUS_ENUM_LIST
(fourth pass), SUPERSLM_ABI_ENUM_ONLY (fifth pass), SUPERSLM_ABI_FUNCTIONS_INCLUDED_ (sixth pass,
inside the very fix that closed the second one). Prose review has failed this class three times in
a row; per StandardsDocument Sec4 the fourth careful edit is not the fix. This script closes it
structurally: extract every #define / #ifndef / #if !defined identifier from the header, and fail
the build when one has no corresponding line in Sec8's prose (an appearance ONLY inside a Sec8
fenced code block does not count -- that is the exact shape all three prior recurrences took: the
macro was quoted in a code snippet and never given its own inventory sentence).

The design document lives in a DIFFERENT repository (the Wizard records tree), not this one. This
check's whole population source is therefore an optional, cross-repo input:
  - SSLM_ABI_DESIGN_DOC env var, if set, is used verbatim.
  - Otherwise, a handful of conventional sibling-checkout locations are tried (this repo and the
    Wizard records repo are normally sibling checkouts under the same parent, or the design doc's
    worktree is a sibling of this one's parent .claude/worktrees/ directory).
  - If none resolve to a real file, this check SKIPS -- loudly, with the paths it tried named --
    and exits 0. A check that hard-fails a code-repo-only CI leg (or any environment that has not
    also checked out the records repo) on a missing cross-repo path is a check that will get
    silently deleted the first time it blocks someone; disclosing the skip honestly, every run, is
    the shape that survives.

Allowlist: tools/ci/abi_header_inventory_allowlist.txt, one `<IDENTIFIER>  # <reason>` line per
entry, same convention and same escape-hatch philosophy as
tools/ci/check_tools_have_build_recipe.py's own allowlist -- an allowlisted identifier is still
printed every run (never silently dropped), it just does not fail the build.
"""
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HEADER_PATH = REPO_ROOT / "include" / "superslm" / "sslm_abi.h"
ALLOWLIST_PATH = REPO_ROOT / "tools" / "ci" / "abi_header_inventory_allowlist.txt"

DESIGN_DOC_RELATIVE = "Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md"

DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)")
IFNDEF_RE = re.compile(r"^\s*#\s*ifndef\s+([A-Za-z_]\w*)")
IF_NOT_DEFINED_RE = re.compile(r"!defined\s*\(\s*([A-Za-z_]\w*)\s*\)")
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
SECTION_8_RE = re.compile(r"^##\s*8\.", re.MULTILINE)
SECTION_9_RE = re.compile(r"^##\s*9\.", re.MULTILINE)


def find_design_doc():
    import os

    env = os.environ.get("SSLM_ABI_DESIGN_DOC")
    tried = []
    if env:
        p = pathlib.Path(env)
        tried.append(p)
        if p.is_file():
            return p, tried

    # Conventional sibling-checkout guesses: REPO_ROOT is normally D:\SuperSLM or a
    # D:\SuperSLM\.worktrees\<name> worktree; the records repo is normally a sibling of the
    # top-level SuperSLM checkout (D:\Wizard). Prefer an in-flight Wizard WORKTREE copy (a
    # records fold this session made that has not reached the Wizard main checkout's develop
    # yet -- exactly this check's own first real run, T-2139 sixth confirmation fold) over the
    # main checkout, since a worktree copy is never older than the ratified record: newest
    # mtime among any .claude/worktrees/*/Claude/Vitruvius/<doc> match wins.
    worktree_matches = []
    for p in REPO_ROOT.parents:
        wizard_root = p / "Wizard"
        if not wizard_root.is_dir():
            continue
        worktrees_dir = wizard_root / ".claude" / "worktrees"
        if worktrees_dir.is_dir():
            for wt in worktrees_dir.iterdir():
                candidate = wt / DESIGN_DOC_RELATIVE
                tried.append(candidate)
                if candidate.is_file():
                    worktree_matches.append(candidate)
        main_candidate = wizard_root / DESIGN_DOC_RELATIVE
        tried.append(main_candidate)
        if worktree_matches:
            break
    if worktree_matches:
        worktree_matches.sort(key=lambda c: c.stat().st_mtime, reverse=True)
        return worktree_matches[0], tried

    # This repo itself, if it is ever vendored/checked out together (defensive, cheap to try).
    self_candidate = REPO_ROOT / DESIGN_DOC_RELATIVE
    tried.append(self_candidate)

    seen = []
    for c in tried:
        if c not in seen:
            seen.append(c)
    for c in seen:
        if c.is_file():
            return c, seen
    return None, seen


def load_allowlist():
    allow = {}
    if not ALLOWLIST_PATH.exists():
        return allow
    for line in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" not in line:
            print(f"ALLOWLIST FORMAT ERROR: entry with no reason comment: {line!r}")
            sys.exit(1)
        name, reason = line.split("#", 1)
        name = name.strip()
        reason = reason.strip()
        if not reason:
            print(f"ALLOWLIST FORMAT ERROR: entry with an empty reason: {line!r}")
            sys.exit(1)
        allow[name] = reason
    return allow


def extract_header_identifiers(header_text):
    found = []
    for line in header_text.splitlines():
        m = DEFINE_RE.match(line)
        if m:
            found.append(m.group(1))
            continue
        m = IFNDEF_RE.match(line)
        if m:
            found.append(m.group(1))
            continue
        for m in IF_NOT_DEFINED_RE.finditer(line):
            found.append(m.group(1))
    # de-dup, preserve first-seen order
    seen = set()
    ordered = []
    for name in found:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def section_8_prose(design_text):
    start_m = SECTION_8_RE.search(design_text)
    if not start_m:
        return None
    end_m = SECTION_9_RE.search(design_text, start_m.end())
    section = design_text[start_m.start():end_m.start() if end_m else len(design_text)]
    # Strip fenced code blocks: an identifier appearing ONLY inside a quoted code snippet is
    # exactly the recurring failure this checker exists to close (M3, three consecutive rounds).
    return CODE_FENCE_RE.sub("", section)


def main():
    if not HEADER_PATH.is_file():
        print(f"check_abi_header_inventory.py: {HEADER_PATH} not found -- skipping (non-fatal)")
        return 0

    design_doc, tried = find_design_doc()
    if design_doc is None:
        print("check_abi_header_inventory.py: SKIPPING -- no design document found. This check's "
              "population source (Sec8 of the T-2133 design) lives in a different repository "
              "(the Wizard records tree) and is not available in this checkout. Tried:")
        for c in tried:
            print(f"  {c}")
        print("Set SSLM_ABI_DESIGN_DOC to the design document's path to run this check for real. "
              "Non-fatal: a code-repo-only build (or CI leg) has no way to satisfy this input.")
        return 0

    header_text = HEADER_PATH.read_text(encoding="utf-8", errors="replace")
    identifiers = extract_header_identifiers(header_text)

    design_text = design_doc.read_text(encoding="utf-8", errors="replace")
    prose = section_8_prose(design_text)
    if prose is None:
        print(f"check_abi_header_inventory.py FAILED -- {design_doc} has no '## 8.' section "
              "heading; cannot locate the ABI-surface inventory to check against.")
        return 1

    allow = load_allowlist()

    missing = [name for name in identifiers if name not in prose]
    unallowlisted = [n for n in missing if n not in allow]
    allowlisted = [n for n in missing if n in allow]

    print(f"check_abi_header_inventory.py: {len(identifiers)} #define/#ifndef/#if !defined "
          f"identifier(s) extracted from {HEADER_PATH.relative_to(REPO_ROOT)}: "
          f"{', '.join(identifiers)}")
    print(f"  design doc: {design_doc}")

    if allowlisted:
        print("check_abi_header_inventory.py: allowlisted (no Sec8 inventory line outside a code "
              "block, justified below):")
        for n in allowlisted:
            print(f"  {n}  -- {allow[n]}")

    if unallowlisted:
        print("check_abi_header_inventory.py FAILED -- identifier(s) on "
              f"{HEADER_PATH.relative_to(REPO_ROOT)} with no Sec8 inventory line (outside a code "
              "block) in the design document, and not in "
              f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)}:")
        for n in unallowlisted:
            print(f"  {n}")
        print("Add a Sec8 inventory line naming the identifier (design doc, ruled public or "
              "private), or add a justified allowlist entry "
              f"({ALLOWLIST_PATH.relative_to(REPO_ROOT)}, `<IDENTIFIER>  # <reason>`).")
        return 1

    print(f"check_abi_header_inventory.py: {len(identifiers)} identifiers checked, "
          f"{len(allowlisted)} allowlisted, 0 unaccounted for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
