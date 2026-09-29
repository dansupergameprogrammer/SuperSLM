#!/usr/bin/env bash
# pin_record_lib.sh -- the pin step's library record (plan rev 9; coverage-mutants-rev8 5d). Run immediately after
# the pin tool installs a prefix: writes the sha256 of the installed <prefix>/lib/libsuperslm.a into
# <prefix>/share/superslm-provenance/installed-lib.sha256, beside the provenance the pin tool wrote. Stand-in for
# the pin tool writing the same value into provenance.json (a consumer-side change; not made here).
set -eu
P=$1; sha256sum "$P/lib/libsuperslm.a" | cut -d' ' -f1 > "$P/share/superslm-provenance/installed-lib.sha256"
