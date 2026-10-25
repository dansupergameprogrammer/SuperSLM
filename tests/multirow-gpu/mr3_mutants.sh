#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), slice MR3: guard vitality for MR3's mutants (plan §7 dim 11). Each mutant is one edit
# to a copy of the engine's C++, rebuilt into a copy of mr3_cells.exe, and must turn its named cell red on the fixtures
# it names. The sites are matched by pattern, so the builder's MR3 status changes (BUSY -> CHUNK_OPEN at save, 1.20's
# mid-token status at R-20) keep them applicable; a site that matches no line, or more lines than stated, fails the
# mutant instead of skipping it.
#
# Usage: tests/multirow-gpu/mr3_mutants.sh [--out=DIR] [--jobs=N] [--only=MX9,...]
#   MX9   save admitted mid-chunk (sslm_gpu_seq_save's chunk-open refusal removed)              -> 7.6
#   MX10  an adapter change admitted on a later slice (both chunk adapter checks removed)        -> 7.7 (pkv_def + r4)
#   MX17  the close copies only the last committed row into the mirror                          -> 7.16
#   MX18  embed_rows leaves ready_for_logits set                                                 -> 7.14
#   MXB5  the batch skips the M = 1 chunk hook after a slot's submission (review N5)             -> 5.4
#   MXS1  widened names no longer map to their shaders_wide sources (review S1, N7)              -> stale
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CXX="${CXX:-$HERE/mingw_cxx.sh}"
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"
OUT="$REPO/out/mr3/mutants"
JOBS=2
ONLY=""
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--only=*) ONLY="${a#--only=}" ;;
		*) sed -n '2,15p' "$0"; exit 2 ;;
	esac
done
cd "$REPO"
CELLS="$(dirname "$OUT")/cells"
FIX="${MR3_FIXTURES:-$REPO/out/mr2/fixtures}"
bash "$HERE/mr3_wine.sh" --out="$(dirname "$OUT")" --only=cells --build-only --jobs="$JOBS" >/dev/null
want() { [ -z "$ONLY" ] && return 0; case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
FAILED=()
mkdir -p "$OUT"
export WINEPREFIX="${WINEPREFIX_MR3:-$OUT/wineprefix-$$}"
if [ -z "${WINEPREFIX_MR3:-}" ]; then
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
		! "$CXX" -O2 "$d/obj"/*.o -o "$d/mr3_cells.exe" -ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi -static -static-libgcc \
			-static-libstdc++ >>"$d/build.log" 2>&1; then
		echo "FAIL mutant $name: build ($d/build.log)"; FAILED+=("$name"); return 1
	fi
	rm -rf "$d/tree" "$d/obj"
}
must_red() {
	local name="$1" cells="$2" m c rc ok=1
	shift 2
	for m in "$@"; do
		set +e
		WINEPREFIX="$WINEPREFIX-$m" "$WINE" "$OUT/$name/mr3_cells.exe" --model="$FIX/$m.sslm" --fixtures="$FIX" --adapter="$FIX/pkv_def_r4.adp" \
			--cells="$cells" >"$OUT/$name/$m.log" 2>&1
		rc=$?
		set -e
		for c in ${cells//,/ }; do
			if ! tr -d '\r' <"$OUT/$name/$m.log" | grep -q "^RED cell $c:"; then ok=0; echo "    $name on $m: cell $c stayed green"; fi
		done
		[ $rc != 0 ] || ok=0
	done
	if [ $ok = 1 ]; then echo "PASS mutant $name -> red in $cells ($*)"; else echo "FAIL mutant $name survived ($cells)"; FAILED+=("$name"); fi
}
G=src/gpu/gpu_1p0.cpp
# MX9: the save's chunk-open refusal (the line before its Paged KV comment; an #endif between them, gated or not), whatever status it returns.
if want MX9; then host_mutant MX9 $G 1 '^(\s*)if \(RowsChunkOpen\(seq\)\) return SSLM_[A-Z_]+;(\n(?:#endif\n)?\s*// Paged KV \(G5)' \
	'\1(void)0;  // MX9: save admitted mid-chunk\2' && must_red MX9 7.6 pkv_def pkv_qk; fi
# MX10: both adapter checks (the M = 1 path's and the widened path's).
if want MX10; then host_mutant MX10 $G 2 \
	'^(\s*)if \((?:seq->rows_chunk->adapter_fixed && seq->rows_chunk->adapter != adapter_or_null|c\.adapter_fixed && adapter != c\.adapter)\) return SSLM_[A-Z_]+;' \
	'\1(void)0;  // MX10: adapter change admitted' && must_red MX10 7.7 pkv_def; fi
if want MX17; then host_mutant MX17 $G 1 '^(\s*)for \(uint32_t r = 0; r < k; \+\+r\) copy_row\(r, L\);' \
	'\1if (k > 0) copy_row(k - 1u, L);  // MX17: only the last committed row' && must_red MX17 7.16 pkv_def pkv_qk; fi
if want MX18; then host_mutant MX18 $G 1 '^(\s*)seq->ready_for_logits = false;(\n\s*seq->bind_eligible = false;\n\s*if \(M == 1\))' \
	'\1// MX18: ready_for_logits left set\2' && must_red MX18 7.14 pkv_def pkv_qk; fi
if want MXB5; then host_mutant MXB5 $G 1 \
	'^(\s*)if \(seq->rows_chunk != nullptr\) RowsAfterOneRowSubmit\(seq, adapter_or_null, layers_to_issue, out_statuses\[i\]\);' \
	'\1(void)0;  // MXB5: the batch skips the M = 1 chunk hook' && must_red MXB5 5.4 pkv_def pkv_qk; fi
if want MXS1; then host_mutant MXS1 src/gpu/superslm_gpu.cpp 1 \
	'^(\s*)if \(WidenedShaderSources\(shader_source_dir, shader_name, &wide_dir, &sources\)\) \{' \
	'\1if (false) {  // MXS1: widened names unmapped' && must_red MXS1 stale pkv_def; fi
if [ ${#FAILED[@]} -eq 0 ]; then echo "mutants: PASS"; exit 0; fi
echo "mutants: FAIL (${FAILED[*]})"
exit 1
