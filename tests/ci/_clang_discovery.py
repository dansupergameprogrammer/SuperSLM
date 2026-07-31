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
Significant 1's defect class outright (a PowerShell scalar-indexing bug
that corrupted the exported path to a single character): there is nothing
here for a wrapper to corrupt, because nothing external sets the variable
that decides.

An already-set SUPERSLM_CLANGXX is always honored unchanged and first --
this module never second-guesses an explicit override, on this machine or
on GitHub's `ubuntu-latest` runners, which set it explicitly to a pinned
`clang++-18` (`.github/workflows/tests.yml`).
"""
from __future__ import annotations

import os
import shutil

# LLVM's own Windows installer default location, and its %ProgramFiles%
# equivalent (usually the same path; kept as two entries in case
# %ProgramFiles% is redirected on a given machine). Mirrors
# Claude/Tooling/superslm-clang-tests.ps1's discovery order for the two
# locations that script and this module both need to agree on; the third
# location that script also searches (Visual Studio's bundled LLVM
# component, globbed by version-suffixed directory) is left to the script,
# since a directory glob is not worth the cost of running on every pytest
# collection when the two fixed locations below cover the common case.
_KNOWN_WINDOWS_PATHS = (
    r"C:\Program Files\LLVM\bin\clang++.exe",
)


def discover_clangxx(default: str = "clang++") -> str:
    """Return a clangxx invocation string to try, in preference order:

    1. An already-set SUPERSLM_CLANGXX -- honored unchanged, unvalidated.
    2. `clang++` resolved on PATH.
    3. A short list of known Windows install locations.
    4. `default`, unresolved -- a genuinely absent toolchain still fails the
       same honest way it always has (FileNotFoundError raised by the
       subprocess call -> ClangUnavailable -> the clang-gated cells SKIP,
       which is a real skip, not a false capability claim).
    """
    env_value = os.environ.get("SUPERSLM_CLANGXX")
    if env_value:
        return env_value

    on_path = shutil.which("clang++")
    if on_path:
        return on_path

    for candidate in _KNOWN_WINDOWS_PATHS:
        if os.path.isfile(candidate):
            return candidate

    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidate = os.path.join(program_files, "LLVM", "bin", "clang++.exe")
        if os.path.isfile(candidate):
            return candidate

    return default
