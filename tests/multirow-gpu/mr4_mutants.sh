#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), slice MR4: the MR4 red suite's mutants, each a one-site edit of a copy of the engine,
# rebuilt with mr4_cells' objects and flags; each must turn its cell red on F-SFD.
# Usage: tests/multirow-gpu/mr4_mutants.sh [--out=DIR] [--jobs=N] [--only=MX16,...]
#   MX16  FIRST_PREWALKED ignored: the flagged row 0 is matched against the forced chain (and walked) like any row  -> 8.10
# The site: the bool embed_rows derives from SSLM_GPU_ROWS_FIRST_PREWALKED for admission (its name contains "prewalk";
# the bad-spec check that refuses the flag with PROMPT is not the site). One edit here if the engine names it otherwise.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CXX="${CXX:-$HERE/mingw_cxx.sh}"
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"
OUT="$REPO/out/mr4/mutants"
JOBS=2
ONLY=""
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--only=*) ONLY="${a#--only=}" ;;
		*) sed -n '2,8p' "$0"; exit 2 ;;
	esac
done
cd "$REPO"
CELLS="$(dirname "$OUT")/cells"
F4="${MR4_FIXTURES:-$(dirname "$OUT")/fixtures}"
bash "$HERE/mr4_wine.sh" --out="$(dirname "$OUT")" --only=cells --build-only --jobs="$JOBS" >/dev/null
want() { [ -z "$ONLY" ] && return 0; case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
FAILED=()
mkdir -p "$OUT"
export WINEPREFIX="${WINEPREFIX_MR4:-$OUT/wineprefix-$$}"
if [ -z "${WINEPREFIX_MR4:-}" ]; then
	trap 'for p in "$WINEPREFIX" "$WINEPREFIX"-*; do [ -d "$p" ] && { WINEPREFIX="$p" /usr/lib/wine/wineserver -k 2>/dev/null || true; rm -rf "$p"; }; done; true' EXIT
fi
start_xvfb; wine_env
CFLAGS="$(cat "$CELLS/obj.flags")"

# host_mutant NAME FILE COUNT PATTERN REPLACEMENT: FILE in a copy of the tree with every match of the Python regex
# PATTERN (exactly COUNT of them) replaced by REPLACEMENT; its translation unit rebuilt with the cells' flags; relinked.
host_mutant() {
	local name="$1" file="$2" count="$3" d="$OUT/$1"
	rm -rf "$d"; mkdir -p "$d/tree"
	cp "$CELLS/d3d12.dll" "$CELLS/d3d12core.dll" "$d/"; cp -r "$CELLS/shaders" "$d/shaders"; cp -a "$CELLS/obj" "$d/obj"
	cp -a include src tests "$d/tree/"
	if ! MX_PAT="$4" MX_NEW="$5" MX_N="$count" python3 -I -c "import os,re,sys; p=sys.argv[1]; s=open(p).read()
s2,n=re.subn(os.environ['MX_PAT'], os.environ['MX_NEW'], s, flags=re.M)
if n!=int(os.environ['MX_N']): sys.exit('%d matches of the site in %s, want %s' % (n, p, os.environ['MX_N']))
open(p,'w').write(s2)" "$d/tree/$file"; then
		echo "FAIL mutant $name: its site is not in $file as stated (the engine changed; update the applicator)"; FAILED+=("$name"); return 1
	fi
	# shellcheck disable=SC2086
	if ! (cd "$d/tree" && "$CXX" $CFLAGS -w -c "$file" -o "$d/obj/$(echo "$file" | tr / _).o") >"$d/build.log" 2>&1 ||
		! "$CXX" -O2 "$d/obj"/*.o -o "$d/mr4_cells.exe" -ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi -static -static-libgcc \
			-static-libstdc++ >>"$d/build.log" 2>&1; then
		echo "FAIL mutant $name: build ($d/build.log)"; FAILED+=("$name"); return 1
	fi
	rm -rf "$d/tree" "$d/obj"
}
must_red() {
	local name="$1" cells="$2" c rc ok=1
	set +e
	(cd "$OUT/$name" && WINEPREFIX="$WINEPREFIX-m" "$WINE" ./mr4_cells.exe --fixtures="$F4" --cells="$cells") >"$OUT/$name/run.log" 2>&1
	rc=$?
	set -e
	for c in ${cells//,/ }; do
		if ! tr -d '\r' <"$OUT/$name/run.log" | grep -q "^RED cell $c:"; then ok=0; echo "    $name: cell $c stayed green"; fi
	done
	[ $rc != 0 ] || ok=0
	if [ $ok = 1 ]; then echo "PASS mutant $name -> red in $cells"; else echo "FAIL mutant $name survived ($cells)"; FAILED+=("$name"); fi
}
G=src/gpu/gpu_1p0.cpp
if want MX16; then host_mutant MX16 $G 1 \
	'^(\s*)(const\s+)?bool\s+(\w*prewalk\w*)\s*=\s*[^;]*SSLM_GPU_ROWS_FIRST_PREWALKED[^;]*;' \
	'\1\2bool \3 = false;  // MX16: FIRST_PREWALKED ignored' && must_red MX16 8.10; fi
if [ ${#FAILED[@]} -eq 0 ]; then echo "mutants: PASS"; exit 0; fi
echo "mutants: FAIL (${FAILED[*]})"
exit 1
