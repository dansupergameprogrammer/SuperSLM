#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), slice MR4: the MR4 red suite's mutants, each a one-site edit of a copy of the engine,
# rebuilt with mr4_cells' objects and flags; each must turn its cell red on F-SFD.
# Usage: tests/multirow-gpu/mr4_mutants.sh [--out=DIR] [--jobs=N] [--only=MX16,...]
#   MX16  FIRST_PREWALKED ignored: the flagged row 0 is matched against the forced chain (and walked) like any row  -> 8.10
#   MXK1  the table check before the token-range loop (ruling 16 swapped)                                          -> 2.3
#   MXK2  the walk over the refused row j at a SCHEMA_FORCED chunk's refusal (M >= 2)                              -> s1
#   MXK2b the same at the M = 1 refusal                                                                            -> s1
#   MXK3  jump_forward's chunk-open guard removed                                                                  -> 4.6j
#   MXK4  embed_schema_token's chunk-open guard removed                                                            -> 4.6e
#   MXK5  forced_tokens' chunk-open guard removed                                                                  -> 4.6f
#   MXK6  the jump no longer restores last_rows_committed                                                         -> n1
# (MXK1-MXK6 are the code review's K1-K6, ruling 18.) A mutant that crashes the cells (no verdict line) counts as red.
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
		*) sed -n '2,16p' "$0"; exit 2 ;;
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
	(cd "$OUT/$name" && WINEPREFIX="$WINEPREFIX-m" timeout 1800 "$WINE" ./mr4_cells.exe --fixtures="$F4" --cells="$cells") >"$OUT/$name/run.log" 2>&1
	rc=$?
	set -e
	if tr -d '\r' <"$OUT/$name/run.log" | grep -q '^cells: '; then
		for c in ${cells//,/ }; do
			if ! tr -d '\r' <"$OUT/$name/run.log" | grep -q "^RED cell $c:"; then ok=0; echo "    $name: cell $c stayed green"; fi
		done
	else
		echo "    $name: the cells crashed (no verdict line; exit $rc): red"
	fi
	[ $rc != 0 ] || ok=0
	if [ $ok = 1 ]; then echo "PASS mutant $name -> red in $cells"; else echo "FAIL mutant $name survived ($cells)"; FAILED+=("$name"); fi
}
G=src/gpu/gpu_1p0.cpp
if want MX16; then host_mutant MX16 $G 1 \
	'^(\s*)(const\s+)?bool\s+(\w*prewalk\w*)\s*=\s*[^;]*SSLM_GPU_ROWS_FIRST_PREWALKED[^;]*;' \
	'\1\2bool \3 = false;  // MX16: FIRST_PREWALKED ignored' && must_red MX16 8.10; fi
# Ruling 18 (the MR4 code review's K1-K6).
if want MXK1; then host_mutant MXK1 $G 1 \
	'^(\s*// 5\. A token outside the vocabulary\.\n\s*for \(int32_t r = 0; r < spec->count; \+\+r\) \{\n[^\n]*\n\s*\}\n)(\s*const bool forced = spec->kind == SSLM_GPU_ROWS_SCHEMA_FORCED;\n[^\n]*\n\s*if \(forced && !model->forced\.present\) return SSLM_GPU_FORCED_TABLE_ABSENT;\n)' \
	'\2\1' && must_red MXK1 2.3; fi
if want MXK2; then host_mutant MXK2 $G 1 '^(\s*)if \(k > 0 && !c\.walk_after\.empty\(\)\) seq->dfa_walk_state = c\.walk_after\[k - 1\];' \
	'\1if (!c.walk_after.empty()) seq->dfa_walk_state = c.walk_after[f];  // MXK2: the walk over row j' && must_red MXK2 s1; fi
if want MXK2b; then host_mutant MXK2b $G 1 '^(\s*)(// The refusal: CommitPrefix\(0, 0\)\.[^\n]*\n\s*seq->last_rows_committed = 0;\n\s*RowsDropChunkImpl\(seq\);)' \
	'\1if (!c.walk_after.empty()) seq->dfa_walk_state = c.walk_after[0];  // MXK2b: the walk over the refused row\n\1\2' && must_red MXK2b s1; fi
if want MXK3; then host_mutant MXK3 $G 1 '^(\s*)if \(RowsChunkOpen\(seq\)\) return SSLM_BUSY;(  // multi-row plan §4\.6 \(R-17\): a chunk is open\n\s*// \(6\) The entry shape)' \
	'\1(void)0;  // MXK3: the jump guard removed\2' && must_red MXK3 4.6j; fi
if want MXK4; then host_mutant MXK4 $G 1 '^(\s*)if \(RowsChunkOpen\(seq\)\) return SSLM_BUSY;(  // multi-row plan §4\.6 \(R-17\): a chunk is open\n\s*SslmGpuModelHandle\* model = seq->model;\n\s*if \(token < 0)' \
	'\1(void)0;  // MXK4: the embed_schema_token guard removed\2' && must_red MXK4 4.6e; fi
if want MXK5; then host_mutant MXK5 $G 1 '^(\s*)if \(RowsChunkOpen\(seq\)\) return SSLM_BUSY;(  // multi-row plan §4\.6 \(R-17\): the walk the chunk will advance is not at rest)' \
	'\1(void)0;  // MXK5: the forced_tokens guard removed\2' && must_red MXK5 4.6f; fi
if want MXK6; then host_mutant MXK6 $G 1 '~LastRowsKeep\(\) \{ s->last_rows_committed = v; \}' \
	'~LastRowsKeep() { (void)s; (void)v; }  // MXK6: no restore' && must_red MXK6 n1; fi
if [ ${#FAILED[@]} -eq 0 ]; then echo "mutants: PASS"; exit 0; fi
echo "mutants: FAIL (${FAILED[*]})"
exit 1
