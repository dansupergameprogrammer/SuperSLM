"""Shared clang++ discovery for tests/ci's clang-gated checks (D-SLM528).

D-SLM522's ratified text says the bad-alloc membership derivation and the
forward-leaf-call checker "run by default" once a matching-version clang++
is installed on a machine, rather than depending on someone remembering to
export SUPERSLM_CLANGXX. What D-SLM522 shipped as (`Claude/Tooling/
superslm-clang-tests.ps1`, a wrapper script in a different repository that a
caller must remember to invoke instead of `pytest`) does not deliver that:
`python -m pytest tests/ci/` still skips every clang-gated cell unless the
variable happens to be set, which is the same "remember to do a thing"
shape the decision names as vigilance and rejects (Poirot casebook
8ce6aa7-t1475-t1480-clang-unlock-confirmation.md, Significant 2, T-1494).

This module moves the discovery this repository's own toolchain resolution
needs -- so every invocation picks a usable clang++ up on its own: a bare
`pytest tests/ci/`, the CI job's own `python -m pytest tests/ci/ -v` step
(`.github/workflows/tests.yml`), a future seat's ad-hoc run, and the
wrapper script alike (which still has a role: it prints what it found and
lets a caller point at an explicit path or exact command). It also removes
Significant 1's defect class outright (closed here -- same casebook, T-1494) (a PowerShell scalar-indexing bug
that corrupted the exported path to a single character): there is nothing
here for a wrapper to corrupt, because nothing external sets the variable
that decides.

An already-set SUPERSLM_CLANGXX is always honored unchanged and first --
this module never second-guesses an explicit override, on this machine or
on GitHub's `ubuntu-latest` runners, which set it explicitly to a pinned
`clang++-18` (`.github/workflows/tests.yml`).

Every discovered (non-override) candidate's major version is checked against
the CI pin before it is returned (T-1508, Poirot casebook 76512b8-t1493-
t1499-clang-discovery-confirmation.md, Significant 1). Before this check
existed, discovery took the first `clang++` it found with no version gate at
all -- unlike Claude/Tooling/superslm-clang-tests.ps1's own discovery, which
has always preferred a version-matching candidate. A host whose only
findable clang++ does not match the pin is not rejected outright (a
version-mismatched-but-working toolchain is still useful, per the wrapper
script's own tolerance -- see its `-MajorVersion` doc comment); it is
returned only after every other candidate in the preference order failed to
match, and only with a warning on stderr, so a version mismatch is never a
silent substitution. What makes a mismatched candidate safe to run at all is
`derive_bad_alloc_membership.run_clang_ast_dump` and
`check_no_forward_leaf_calls._run_clang_ast_dump` no longer classifying a
timeout as `ClangUnavailable` -- a toolchain that hangs instead of failing
fast (as a real major-version mismatch can, e.g. inside MSVC's `<xstring>`
under `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`) now raises loudly instead
of being indistinguishable from a genuine absence.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

# LLVM's own Windows installer default location, and its %ProgramFiles%
# equivalent (usually the same path). Mirrors Claude/Tooling/
# superslm-clang-tests.ps1's discovery order for the two locations that
# script and this module both need to agree on; the third location that
# script also searches (Visual Studio's bundled LLVM component, globbed by
# version-suffixed directory) is left to the script, since a directory glob
# is not worth the cost of running on every pytest collection when the two
# fixed locations below cover the common case. This is a single fixed path,
# not two: the %ProgramFiles%-relative form below is computed in
# discover_clangxx() itself, at call time, because it depends on an
# environment variable that must not be frozen at import.
_KNOWN_WINDOWS_PATHS = (
    r"C:\Program Files\LLVM\bin\clang++.exe",
)

# The clang major version .github/workflows/tests.yml's CI jobs pin
# (`clang++-18`). Mirrors Claude/Tooling/superslm-clang-tests.ps1's
# `-MajorVersion` default.
_PINNED_MAJOR_VERSION = 18

_CLANG_VERSION_RE = re.compile(r"clang version (\d+)\.")


def _clang_major_version(clangxx_path: str) -> int | None:
    """Return the major version `clangxx_path --version` reports, or None if
    the binary cannot be invoked, times out, or its output does not match
    clang's own version-string format. Mirrors Claude/Tooling/
    superslm-clang-tests.ps1's Get-ClangMajorVersion. A 10s timeout is ample
    for `--version`, which does not touch the STL headers where a genuine
    major-version mismatch can hang (T-1508) -- this probe is not the
    operation that motivated the loud-timeout fix in run_clang_ast_dump."""
    try:
        proc = subprocess.run(
            [clangxx_path, "--version"], capture_output=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = proc.stdout.decode("utf-8", errors="replace")
    match = _CLANG_VERSION_RE.search(text)
    return int(match.group(1)) if match else None


def discover_clangxx(default: str = "clang++",
                      pinned_major: int = _PINNED_MAJOR_VERSION) -> str:
    """Return a clangxx invocation string to try, in preference order:

    1. An already-set SUPERSLM_CLANGXX -- honored unchanged, unvalidated;
       never version-checked, matching the wrapper script's own tolerance
       for an explicit override.
    2. `clang++` resolved on PATH.
    3. A short list of known Windows install locations.
    4. `default`, unresolved -- a genuinely absent toolchain still fails the
       same honest way it always has (FileNotFoundError raised by the
       subprocess call -> ClangUnavailable -> the clang-gated cells SKIP,
       which is a real skip, not a false capability claim).

    Among candidates 2 and 3, one whose `--version` major matches
    `pinned_major` is preferred over one found earlier in the search order
    but not matching -- the same preference Claude/Tooling/
    superslm-clang-tests.ps1 already applies. When at least one candidate
    was found but none matches, the first candidate found is still returned
    (a working, version-mismatched clang++ still runs the suite -- see the
    module docstring), but only after every candidate in the search order is
    exhausted, and with a warning on stderr naming the mismatch, so the
    substitution is never silent.
    """
    env_value = os.environ.get("SUPERSLM_CLANGXX")
    if env_value:
        return env_value

    candidates: list[str] = []

    on_path = shutil.which("clang++")
    if on_path:
        candidates.append(on_path)

    for candidate in _KNOWN_WINDOWS_PATHS:
        if os.path.isfile(candidate):
            candidates.append(candidate)

    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidate = os.path.join(program_files, "LLVM", "bin", "clang++.exe")
        if os.path.isfile(candidate):
            candidates.append(candidate)

    if not candidates:
        return default

    for candidate in candidates:
        if _clang_major_version(candidate) == pinned_major:
            return candidate

    # Exhausted: at least one clang++ was found, but none reports the pinned
    # major version. Fall back to the first one found (matching the wrapper
    # script's own tolerance for a mismatched-but-working toolchain), but
    # say so on stderr rather than silently substituting it as if it had
    # matched.
    fallback = candidates[0]
    print(
        f"_clang_discovery: no clang++ with major version {pinned_major} "
        f"found among {candidates}; falling back to {fallback} "
        f"(major {_clang_major_version(fallback)}), version-unpinned.",
        file=sys.stderr,
    )
    return fallback
