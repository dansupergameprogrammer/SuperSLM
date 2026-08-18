#!/usr/bin/env python3
"""T-2139 sixth confirmation review, M3's structural half (Claude/Poirot/
5fbd04d-t2139-sixth-confirmation-review.md item 5; ruled in
Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md Sec8, the "structural half" paragraph).

Three consecutive confirmation rounds found a macro landed on the frozen public ABI header
(include/superslm/sslm_abi.h) with no Sec8 inventory line naming it -- SSLM_STATUS_ENUM_LIST
(fourth pass), SUPERSLM_ABI_ENUM_ONLY (fifth pass), SUPERSLM_ABI_FUNCTIONS_INCLUDED_ (sixth pass,
inside the very fix that closed the second one). Prose review has failed this class three times in
a row; per StandardsDocument Sec4 the fourth careful edit is not the fix. This script closes it
structurally: extract every #define / #ifndef / #if !defined identifier from the header (and any
first-party .inc it transitively #includes -- M2, Claude/Poirot/
aea6116-t2139-seventh-confirmation-review.md), and fail the build when one has no corresponding
line in Sec8's prose (an appearance ONLY inside a Sec8 fenced code block does not count -- that is
the exact shape all three prior recurrences took: the macro was quoted in a code snippet and never
given its own inventory sentence).

The design document lives in a DIFFERENT repository (the Wizard records tree), not this one. This
check's whole population source is therefore an optional, cross-repo input, resolved from a
COMMITTED GIT REF rather than the filesystem (S1, Claude/Poirot/
aea6116-t2139-seventh-confirmation-review.md):
  - SSLM_ABI_DESIGN_DOC_REPO env var, if set, names the records repo's working-copy path (any
    checkout or worktree of it -- worktrees of the same repo share one object store, so which
    particular checkout is used does not change what a given ref resolves to). Otherwise, a
    handful of conventional sibling-checkout locations are tried (this repo and the Wizard
    records repo are normally sibling checkouts under the same parent).
  - SSLM_ABI_DESIGN_DOC_REF env var, if set, names the git ref to read the document AT (a branch,
    a tag, a commit). Defaults to "develop" -- the shared, ratified line -- documented here so the
    default is never a silent guess.
  - The document is read with `git -C <repo> show <ref>:<path>` -- the committed object store,
    never a working-tree file's mtime. This is why a worktree copy no longer needs to be "newer":
    every worktree of the same repo resolves the identical ref to the identical committed bytes,
    so there is no filesystem race to win in the first place. (Previously this script walked
    every sibling .claude/worktrees/* directory and took whichever had the newest st_mtime --
    closed here: st_mtime tracks when a file was last WRITTEN to a working tree, never whether its
    content was ratified, and the newest write is exactly as likely to be an unmerged draft as the
    settled record. Executed against all four on-disk copies at the time of that finding: three
    failed with "no Sec8 section", one -- this session's own worktree -- passed only because nothing
    else on the machine had been touched more recently. That is a check whose pass/fail depended on
    which worktree gets deleted first, not on the tree it was built to guard.)
  - If the repo cannot be found, or `git show` fails (bad ref, path never existed at that ref),
    this check SKIPS -- loudly, with what was tried named -- and exits 0. A check that hard-fails a
    code-repo-only CI leg (or any environment that has not also checked out the records repo) on a
    missing cross-repo input is a check that will get silently deleted the first time it blocks
    someone; disclosing the skip honestly, every run, is the shape that survives.
  - If the document resolves but carries NO '## 8.' section heading, this is ALSO a SKIP (S1's own
    fix), not a hard failure: a document that has not yet gained the section this check depends on
    is the SAME epistemic state as no document at all -- the population source simply is not
    available yet at the ref being read (true of `develop` itself as of this fix: Sec8 has not
    reached it). Failing the build on that state would make it impossible to land Sec8 at all
    without first disabling the check that depends on it.

Allowlist: tools/ci/abi_header_inventory_allowlist.txt, one `<IDENTIFIER>  # <reason>` line per
entry, same convention and same escape-hatch philosophy as
tools/ci/check_tools_have_build_recipe.py's own allowlist -- an allowlisted identifier is still
printed every run (never silently dropped), it just does not fail the build.
"""
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HEADER_PATH = REPO_ROOT / "include" / "superslm" / "sslm_abi.h"
HEADER_DIR = HEADER_PATH.parent
ALLOWLIST_PATH = REPO_ROOT / "tools" / "ci" / "abi_header_inventory_allowlist.txt"

DESIGN_DOC_RELATIVE = "Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md"
DEFAULT_DESIGN_DOC_REF = "develop"

DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)")
IFNDEF_RE = re.compile(r"^\s*#\s*ifndef\s+([A-Za-z_]\w*)")
IF_NOT_DEFINED_RE = re.compile(r"!defined\s*\(\s*([A-Za-z_]\w*)\s*\)")
INCLUDE_INC_RE = re.compile(r'^\s*#\s*include\s+"(superslm/[^"]+\.inc)"', re.MULTILINE)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
SECTION_8_RE = re.compile(r"^##\s*8\.", re.MULTILINE)
SECTION_9_RE = re.compile(r"^##\s*9\.", re.MULTILINE)

# "Cannot check" is a distinct outcome from both PASS and FAIL: the population source is not
# available (no repo, no ref, no file at that ref, or a file with no Sec8 section yet). It always
# exits 0 -- a code-repo-only environment, or a records repo whose ratified line has not yet
# gained Sec8, must never be blocked by this check -- but it is never silently indistinguishable
# from a real pass either; every SKIP prints exactly why.


class CannotCheck(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def find_design_doc_repo():
    import os

    env = os.environ.get("SSLM_ABI_DESIGN_DOC_REPO")
    if env:
        p = pathlib.Path(env)
        if (p / ".git").exists():
            return p, [p]
        raise CannotCheck(
            f"SSLM_ABI_DESIGN_DOC_REPO={env!r} is not a git checkout (no .git found there)."
        )

    tried = []
    # Conventional sibling-checkout guess: REPO_ROOT is normally D:\SuperSLM or a
    # D:\SuperSLM\.worktrees\<name> worktree; the records repo is normally a sibling of the
    # top-level SuperSLM checkout, at a directory named "Wizard". Any checkout or worktree of
    # that repo works identically here -- git show reads the ref from the shared object store,
    # not from that particular working tree's files.
    for p in REPO_ROOT.parents:
        wizard_root = p / "Wizard"
        tried.append(wizard_root)
        if (wizard_root / ".git").exists():
            return wizard_root, tried
        worktrees_dir = wizard_root / ".claude" / "worktrees"
        if worktrees_dir.is_dir():
            for wt in sorted(worktrees_dir.iterdir()):
                tried.append(wt)
                if (wt / ".git").exists():
                    return wt, tried

    raise CannotCheck(
        "no checkout of the Wizard records repo found (no .git in any candidate). Tried:\n"
        + "\n".join(f"  {c}" for c in tried)
    )


def git_show(repo, ref, relative_path):
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{relative_path}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise CannotCheck(
            f"`git -C {repo} show {ref}:{relative_path}` failed (ref does not exist, or the "
            f"path is not present at that ref):\n{result.stderr.strip()}"
        )
    return result.stdout


def find_design_doc_text():
    import os

    repo, tried_repos = find_design_doc_repo()
    ref = os.environ.get("SSLM_ABI_DESIGN_DOC_REF", DEFAULT_DESIGN_DOC_REF)
    text = git_show(repo, ref, DESIGN_DOC_RELATIVE)
    return text, repo, ref


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


def extract_identifiers_from_text(text):
    found = []
    for line in text.splitlines():
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
    return found


def extract_header_identifiers(header_path):
    """M2 (Claude/Poirot/aea6116-t2139-seventh-confirmation-review.md): the frozen consumer
    surface a caller sees after `#include "superslm/sslm_abi.h"` is that file PLUS every
    first-party .inc it transitively includes (sslm_abi.h includes sslm_abi_functions.inc, which
    itself includes sslm_abi_functions_g5_comparable.inc) -- a macro landed in one of those .inc
    files would be shipped, consumer-visible surface this checker previously could not see at
    all, because it only ever read sslm_abi.h's own text. Follows `#include "superslm/*.inc"`
    transitively (first-party only -- system/library includes are not this checker's population
    source), de-duplicating both files visited and identifiers found."""
    found = []
    visited = set()
    queue = [header_path]
    while queue:
        path = queue.pop(0)
        resolved = path.resolve()
        if resolved in visited:
            continue
        visited.add(resolved)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        found.extend(extract_identifiers_from_text(text))
        for m in INCLUDE_INC_RE.finditer(text):
            # "superslm/foo.inc" is always resolved relative to the include/ directory that
            # contains the superslm/ package, i.e. HEADER_DIR's own parent -- matching how the
            # real compiler resolves it via -Iinclude.
            queue.append(HEADER_DIR.parent / m.group(1))
    # de-dup, preserve first-seen order
    seen = set()
    ordered = []
    for name in found:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered, sorted(p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents else p
                            for p in visited if p.is_file())


def section_8_prose(design_text):
    start_m = SECTION_8_RE.search(design_text)
    if not start_m:
        return None
    end_m = SECTION_9_RE.search(design_text, start_m.end())
    section = design_text[start_m.start():end_m.start() if end_m else len(design_text)]
    # Strip fenced code blocks: an identifier appearing ONLY inside a quoted code snippet is
    # exactly the recurring failure this checker exists to close (M3, three consecutive rounds).
    return CODE_FENCE_RE.sub("", section)


def identifier_present(name, prose):
    # S2 (Claude/Poirot/aea6116-t2139-seventh-confirmation-review.md): a bare substring test
    # ("name in prose") reports a false PASS the moment a new macro's name is a PREFIX of any
    # already-narrated identifier -- executed against the real header/prose: appending
    # SSLM_ABI_ALIGNMENT, SSLM_STATUS_ENUM, or SUPERSLM_ABI_ENUM (each a genuine prefix of a real
    # identifier already narrated in Sec8) all passed as "accounted for" while never themselves
    # appearing anywhere in the document. This header's own naming convention is families of
    # shared prefixes (SSLM_ABI_ALIGNMENT_BYTES, SSLM_STATUS_ENUM_LIST/VALUE_,
    # SUPERSLM_ABI_ENUM_ONLY/SUPERSLM_ABI_FUNCTIONS_INCLUDED_), which is exactly the shape this
    # bug is invisible on until the next name that happens to collide. Word-boundary match
    # instead: the identifier must appear as a complete token, not as an infix of some longer
    # narrated name.
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, prose) is not None


def main():
    if not HEADER_PATH.is_file():
        print(f"check_abi_header_inventory.py: {HEADER_PATH} not found -- skipping (non-fatal)")
        return 0

    try:
        design_text, repo, ref = find_design_doc_text()
    except CannotCheck as e:
        print("check_abi_header_inventory.py: CANNOT-CHECK -- SKIPPING (non-fatal). This check's "
              "population source (Sec8 of the T-2133 design) could not be resolved from a "
              "committed git ref of the Wizard records repo:")
        print(f"  {e.message}")
        print("Set SSLM_ABI_DESIGN_DOC_REPO to that repo's path and/or SSLM_ABI_DESIGN_DOC_REF "
              f"to the ref to read it at (default: {DEFAULT_DESIGN_DOC_REF!r}) to run this check "
              "for real. Non-fatal: a code-repo-only build (or CI leg, or a records line that has "
              "not yet gained the section this check depends on) has no way to satisfy this "
              "input, and a document in that state is the same epistemic state as no document.")
        return 0

    identifiers, files_read = extract_header_identifiers(HEADER_PATH)

    prose = section_8_prose(design_text)
    if prose is None:
        print(f"check_abi_header_inventory.py: CANNOT-CHECK -- SKIPPING (non-fatal). "
              f"{DESIGN_DOC_RELATIVE} at {repo}@{ref} has no '## 8.' section heading; the "
              "ABI-surface inventory this check compares against does not exist at that ref yet. "
              "A document with no Sec8 is the same epistemic state as no document -- this is a "
              "SKIP (S1, Claude/Poirot/aea6116-t2139-seventh-confirmation-review.md), not a hard "
              "failure, so landing Sec8 for the first time is never blocked by the check that "
              "depends on it existing.")
        return 0

    allow = load_allowlist()

    missing = [name for name in identifiers if not identifier_present(name, prose)]
    unallowlisted = [n for n in missing if n not in allow]
    allowlisted = [n for n in missing if n in allow]

    print(f"check_abi_header_inventory.py: {len(identifiers)} #define/#ifndef/#if !defined "
          f"identifier(s) extracted from {', '.join(str(p) for p in files_read)}: "
          f"{', '.join(identifiers)}")
    print(f"  design doc: {repo}@{ref}:{DESIGN_DOC_RELATIVE}")

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
