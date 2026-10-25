#!/usr/bin/env bash
# Multi-row GPU plan (rev 1.3.2), slice MR2: guard vitality (plan §7 dim 11). Each mutant is built in the cloud and
# must turn its named cell red, through that cell's own check, on the fixtures it names.
#
# Usage: tests/multirow-gpu/mr2_mutants.sh [--out=DIR] [--jobs=N] [--only=MX2,MX3a,...]
#   needs mr2_wine.sh's cells build (<out>/../cells/mr2_cells.exe); builds it when absent.
#
# Shader mutants (applied to copies of src/gpu/shaders_wide, the cells exe run against the mutated .cso):
#   MX1s  every row at the chunk's widest causal width (`g_width + SSLM_WIDE_H - 1`)      -> 6.1
#   MX2   no `+ r` in the row position/address (the g_position row macro dropped)          -> 6.1, 4.6
#   MX3a  the padded rows' sticky gate removed in every site                               -> 6.4, 4.6b
#   MX3b  the padded rows' sticky gate removed in commit_site only                         -> 6.4, 4.6b
#   MX6   the adapter tails read row 0's activations (ApplyFusedAdapterDeltaGpu on row 0)  -> 4.1 (pkv_def_r4.adp)
#   MX12  at H = 2, row b's GEMM sum stored to row (b + 1) mod 2                           -> 6.1
#   MX15  barriers removed from the widened GEMM tree's last log2(32) steps: 4.7's lint (MR1 suite) is its
#         cell; here it is EXECUTED under 6.1/4.10 and the result is recorded, red or not (plan: "the record
#         names the machine (if any) that turns it red")
# Host mutants (one exact edit to a copy of the recorder's C++, gpu_1p0.cpp or multirow_wide.inl; a site that is
# no longer there fails the mutant rather than skipping it):
#   MX1h  attention groups sized for row 0 (`max_width = width0`)                          -> 6.1 (MX1-host cases)
#   MX4   the close picks the LAST refused row instead of the first                        -> 6.6 (two-refusal)
#   MX5   counters summed over all rows                                                    -> 6.6 (every j < 7)
#   MX7   the close copies a row above f into the mirror                                   -> 6.6
#   MX8   rollback skips HolderUnmapTail                                                   -> 1.1r (MR3's 1.1)
#   MX11  ctxlen advanced once per chunk                                                   -> 6.1
#   MX20  the refused row's counter deltas dropped (rev 1.1's rule)                        -> 6.6
#   MX21  the refused row's mirror rows not copied                                         -> 6.6
# Exit 0 only when every mutant run turned its cell red (MX15's executed leg is report-only).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CXX="${CXX:-$HERE/mingw_cxx.sh}"
WINE_GPU_SOURCED=1 . "$REPO/tools/cloud_gpu/wine_gpu.sh"
OUT="$REPO/out/mr2/mutants"
JOBS=2
ONLY=""
for a in "$@"; do
	case "$a" in
		--out=*) OUT="${a#--out=}" ;;
		--jobs=*) JOBS="${a#--jobs=}" ;;
		--only=*) ONLY="${a#--only=}" ;;
		*) sed -n '2,33p' "$0"; exit 2 ;;
	esac
done
cd "$REPO"
CELLS="$(dirname "$OUT")/cells"
FIX="$(dirname "$OUT")/fixtures"
# Always through mr2_wine.sh's incremental build: an exe left from a stub build (before the recorder) is not the cells.
bash "$HERE/mr2_wine.sh" --out="$(dirname "$OUT")" --only=cells --build-only --jobs="$JOBS" >/dev/null
want() { [ -z "$ONLY" ] && return 0; case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
FAILED=()
if ! grep -rqs "sslm_gpu_seq_embed_rows" src/gpu/*.cpp src/gpu/*.inl; then
	echo "RED mutants: src/ defines no recorder entry, so every cell is red-unimplemented and no mutant can be graded"
	exit 1
fi
mkdir -p "$OUT"
export WINEPREFIX="${WINEPREFIX_MR2:-$OUT/wineprefix-$$}"
start_xvfb; wine_env

# shader_mutant NAME SED-SCRIPT FILE-GLOB: copy the cells' exe and shaders, apply SED to the widened sources the glob
# names, recompile those variants into the copy.
shader_mutant() {
	local name="$1" script="$2" glob="$3" d="$OUT/$1" f h g gi o changed=0
	rm -rf "$d"; mkdir -p "$d/src"
	cp "$CELLS/mr2_cells.exe" "$CELLS/d3d12.dll" "$CELLS/d3d12core.dll" "$d/"
	cp -r "$CELLS/shaders" "$d/shaders"
	for f in src/gpu/shaders_wide/$glob; do
		sed "$script" "$f" >"$d/src/$(basename "$f")"
		cmp -s "$f" "$d/src/$(basename "$f")" && continue
		changed=$((changed + 1))
		for h in 2 4 8; do
			if [ "$(basename "$f")" = proj_gemm_wide.hlsl ]; then
				gi=1
				for g in q_proj kv_proj o_proj gate_proj up_proj down_proj; do
					"$DXC" -T cs_6_2 -E main -Fo "$d/shaders/${g}_gemm_wide_h$h.cso" "$d/src/proj_gemm_wide.hlsl" -O3 -HV 2018 \
						-WX -D SSLM_WIDE_H=$h -D SSLM_GEMM_SITE=$gi -I src/gpu/shaders >/dev/null
					gi=$((gi + 1))
				done
			else
				o="$d/shaders/$(basename "$f" .hlsl)_h$h.cso"
				"$DXC" -T cs_6_2 -E main -Fo "$o" "$d/src/$(basename "$f")" -O3 -HV 2018 -WX -D SSLM_WIDE_H=$h -I src/gpu/shaders >/dev/null
			fi
		done
	done
	[ "$changed" -gt 0 ] || { echo "FAIL mutant $name changed nothing"; FAILED+=("$name"); return 1; }
}
# must_red NAME CELLS FIXTURES...: the mutated exe must report RED for every named cell on every fixture.
must_red() {
	local name="$1" cells="$2" m c rc ok=1
	shift 2
	for m in "$@"; do
		set +e
		"$WINE" "$OUT/$name/mr2_cells.exe" --model="$FIX/$m.sslm" --fixtures="$FIX" --adapter="$FIX/pkv_def_r4.adp" \
			--cells="$cells" --partitions=5 >"$OUT/$name/$m.log" 2>&1
		rc=$?
		set -e
		for c in ${cells//,/ }; do
			if ! tr -d '\r' <"$OUT/$name/$m.log" | grep -q "^RED cell $c:"; then ok=0; echo "    $name on $m: cell $c stayed green"; fi
		done
		[ $rc != 0 ] || ok=0
	done
	if [ $ok = 1 ]; then echo "PASS mutant $name -> red in $cells ($*)"; else echo "FAIL mutant $name survived ($cells)"; FAILED+=("$name"); fi
}

if want MX1s; then shader_mutant MX1s 's/^#define g_width (g_width + g_row)$/#define g_width (g_width + SSLM_WIDE_H - 1)/' '*_site_wide.hlsl' && must_red MX1s 6.1 pkv_def pkv_qk; fi
if want MX2; then shader_mutant MX2 '/^#define g_position (g_position + g_row)$/d' '*_site_wide.hlsl' && must_red MX2 6.1,4.6 pkv_def pkv_qk; fi
if want MX3a; then shader_mutant MX3a 's/^\(\s*\)if (sticky != kTagOk) return;/\1\/\/ MX3a: padded gate removed/; s/^\(\s*\)if (SeqState.Load<int64_t>(sticky_off) != kTagOk) return;/\1\/\/ MX3a: padded gate removed/' '*_site_wide.hlsl' && must_red MX3a 6.4,4.6b pkv_def pkv_qk; fi
if want MX3b; then shader_mutant MX3b 's/^\(\s*\)if (sticky != kTagOk) return;/\1\/\/ MX3b: padded gate removed/' 'commit_site_wide.hlsl' && must_red MX3b 6.4,4.6b pkv_def pkv_qk; fi
if want MX6; then shader_mutant MX6 's/ApplyFusedAdapterDeltaGpu(t, WorkScratch, LayerScratch,/ApplyFusedAdapterDeltaGpu(t, WorkScratch, LayerScratchRows[0],/' '*_site_wide.hlsl' && must_red MX6 4.1 pkv_def; fi
if want MX12; then shader_mutant MX12 's/WorkScratchRows\[b0 + b\]\.Store<int64_t>/WorkScratchRows[SSLM_WIDE_H == 2 ? ((b0 + b + 1u) % 2u) : (b0 + b)].Store<int64_t>/' 'proj_gemm_wide.hlsl' && must_red MX12 6.1 pkv_def pkv_qk; fi
if want MX15; then
	# The MR1 suite's own MX15 construction, applied to a copy, executed under 6.1 (report-only).
	d="$OUT/MX15"; rm -rf "$d"; mkdir -p "$d/src"
	cp "$CELLS/mr2_cells.exe" "$CELLS/d3d12.dll" "$CELLS/d3d12core.dll" "$d/"; cp -r "$CELLS/shaders" "$d/shaders"
	python3 -I -c "import sys; sys.path.insert(0, '$HERE'); import mr1_wide as m; p='src/gpu/shaders_wide/proj_gemm_wide.hlsl'; open('$d/src/proj_gemm_wide.hlsl','w').write(m.mutate_mx15(open(p).read()))"
	for h in 2 4 8; do gi=1; for g in q_proj kv_proj o_proj gate_proj up_proj down_proj; do
		"$DXC" -T cs_6_2 -E main -Fo "$d/shaders/${g}_gemm_wide_h$h.cso" "$d/src/proj_gemm_wide.hlsl" -O3 -HV 2018 -WX \
			-D SSLM_WIDE_H=$h -D SSLM_GEMM_SITE=$gi -I src/gpu/shaders >/dev/null; gi=$((gi + 1)); done; done
	for m in pkv_def pkv_qk; do
		set +e; "$WINE" "$d/mr2_cells.exe" --model="$FIX/$m.sslm" --cells=6.1 >"$d/$m.log" 2>&1; rc=$?; set -e
		echo "REPORT MX15 executed on this machine ($m, 6.1): $([ $rc = 0 ] && echo 'green (lockstep holds here)' || echo 'RED')"
	done
fi
# host_mutant NAME FILE OLD NEW: the cells' objects with FILE's one exact OLD replaced by NEW (must occur once) in a
# copy of the tree, the translation unit that compiles FILE rebuilt with mr2_wine.sh's flags, relinked.
CFLAGS="-std=c++20 -O2 -Iinclude -Isrc -Isrc/gpu -Itests -Itests/multirow-gpu -DNOMINMAX -DWIN32_LEAN_AND_MEAN -include d3d12.h -include d3d12sdklayers.h -DSUPERSLM_MULTIROW_WIDE"
host_mutant() {
	local name="$1" file="$2" d="$OUT/$1" tu
	rm -rf "$d"; mkdir -p "$d/tree"
	cp "$CELLS/d3d12.dll" "$CELLS/d3d12core.dll" "$d/"; cp -r "$CELLS/shaders" "$d/shaders"; cp -a "$CELLS/obj" "$d/obj"
	cp -a include src tests "$d/tree/"
	if ! MX_OLD="$3" MX_NEW="$4" python3 -I -c "import os,sys; p=sys.argv[1]; s=open(p).read(); o=os.environ['MX_OLD']; n=s.count(o)
if n!=1: sys.exit('%s occurs %d times in %s' % (o.splitlines()[0], n, p))
open(p,'w').write(s.replace(o, os.environ['MX_NEW']))" "$d/tree/$file"; then
		echo "FAIL mutant $name: its site is not in $file (the recorder changed; update the applicator)"; FAILED+=("$name"); return 1
	fi
	case "$file" in src/gpu/multirow_wide.inl) tu=src/gpu/superslm_gpu.cpp ;; *) tu="$file" ;; esac
	# shellcheck disable=SC2086
	if ! (cd "$d/tree" && "$CXX" $CFLAGS -w -c "$tu" -o "$d/obj/$(echo "$tu" | tr / _).o") >"$d/build.log" 2>&1 ||
		! "$CXX" -O2 "$d/obj"/*.o -o "$d/mr2_cells.exe" -ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi -static -static-libgcc \
			-static-libstdc++ >>"$d/build.log" 2>&1; then
		echo "FAIL mutant $name: build ($d/build.log)"; FAILED+=("$name"); return 1
	fi
	rm -rf "$d/tree" "$d/obj"
}
INL=src/gpu/multirow_wide.inl; G=src/gpu/gpu_1p0.cpp
if want MX1h; then host_mutant MX1h $INL 'const uint32_t max_width = width0 + (variant_h - 1u);' \
	'const uint32_t max_width = width0;  // MX1h' && must_red MX1h 6.1 pkv_def pkv_qk; fi
if want MX4; then host_mutant MX4 $G $'\t\t\tj = static_cast<int32_t>(r);\n\t\t\tbreak;' \
	$'\t\t\tj = static_cast<int32_t>(r);  // MX4: no break, the last refused row' && must_red MX4 6.6 pkv_def pkv_qk; fi
if want MX5; then host_mutant MX5 $G 'for (uint32_t r = 0; r <= f; ++r) counters[i] += rows[r].counters[i];' \
	'for (uint32_t r = 0; r < M; ++r) counters[i] += rows[r].counters[i];  // MX5' && must_red MX5 6.6 pkv_def pkv_qk; fi
if want MX7; then host_mutant MX7 $G 'for (uint32_t r = 0; r < k; ++r) copy_row(r, L);' \
	'for (uint32_t r = 0; r < M; ++r) copy_row(r, L);  // MX7' && must_red MX7 6.6 pkv_def pkv_qk; fi
if want MX8; then host_mutant MX8 $G 'const SslmGpuStatus unmap = RowsUnmapTailTo(seq, e.holder_mapped);' \
	'const SslmGpuStatus unmap = SSLM_OK;  // MX8: rollback skips HolderUnmapTail' && must_red MX8 1.1r pkv_def pkv_qk; fi
if want MX11; then host_mutant MX11 $G 'st.context_length = c.p0 + static_cast<int64_t>(r);' \
	'st.context_length = c.p0;  // MX11' && must_red MX11 6.1 pkv_def pkv_qk; fi
if want MX20; then host_mutant MX20 $G 'for (uint32_t r = 0; r <= f; ++r) counters[i] += rows[r].counters[i];' \
	'for (uint32_t r = 0; r < k; ++r) counters[i] += rows[r].counters[i];  // MX20' && must_red MX20 6.6 pkv_def pkv_qk; fi
if want MX21; then host_mutant MX21 $G 'copy_row(static_cast<uint32_t>(j), refusing_end);' \
	'(void)refusing_end;  // MX21' && must_red MX21 6.6 pkv_def pkv_qk; fi
if [ ${#FAILED[@]} -eq 0 ]; then echo "mutants: PASS"; exit 0; fi
echo "mutants: FAIL (${FAILED[*]})"
exit 1
