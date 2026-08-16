#!/usr/bin/env bash
# t2139_count_abi_verbs.sh -- design Sec4's own verb-count derivation script, wired here
# (Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md Sec4: "count_abi_verbs.sh (built
# into tests/t2130-g5-red-suite/tools/ or an equivalent project Tools/ location -- the red
# suite's own choice, since it is the natural place a build-time check like this lives)"). The
# red suite itself is read-only to this build (T-2139's own brief); tools/ is this project's own
# equivalent location, matching every other ticket-numbered standing tool already here
# (t2113_*, t2116_*, t2124_*). Never hand-recount the header below; run this and cite its
# output.
#
# Usage: t2139_count_abi_verbs.sh <path-to-header>
#   e.g. t2139_count_abi_verbs.sh include/superslm/sslm_abi_functions.inc
set -euo pipefail
TARGET="${1:?usage: t2139_count_abi_verbs.sh <path-to-header-or-inc-file>}"
grep -oE '\bsslm_[a-z0-9_]+\s*\(' "$TARGET" | sed -E 's/\s*\($//' | sort -u | wc -l
