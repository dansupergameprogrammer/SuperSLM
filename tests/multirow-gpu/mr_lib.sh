# Multi-row GPU plan (rev 1.3.2): the build helpers the MR2 and MR3 cloud runners share (sourced, not run).
# The caller sets REPO, OUT, JOBS, CXX, FLAGS and has sourced tools/cloud_gpu/wine_gpu.sh (DXC, VKP_DIR, WINE).
# shaders TREE DEST [wide]: the tree's one-row shaders (and with `wide`, its widened variants as build.bat names
# them) compiled into DEST.
shaders() {
	local tree="$1" dest="$2" f o h g gi
	mkdir -p "$dest"
	for f in "$tree"/src/gpu/shaders/*.hlsl; do
		o="$dest/$(basename "$f" .hlsl).cso"
		[ -f "$o" ] && [ "$o" -nt "$f" ] || "$DXC" -T cs_6_2 -E main -Fo "$o" "$f" -O3 -HV 2018 -WX >/dev/null
	done
	[ "${3:-}" = wide ] || return 0
	for h in 2 4 8; do
		for f in "$tree"/src/gpu/shaders_wide/*_site_wide.hlsl; do
			o="$dest/$(basename "$f" .hlsl)_h$h.cso"
			[ -f "$o" ] && [ "$o" -nt "$f" ] || "$DXC" -T cs_6_2 -E main -Fo "$o" "$f" -O3 -HV 2018 -WX -D SSLM_WIDE_H=$h \
				-I "$tree/src/gpu/shaders" >/dev/null
		done
		gi=1
		for g in q_proj kv_proj o_proj gate_proj up_proj down_proj; do
			o="$dest/${g}_gemm_wide_h$h.cso"
			[ -f "$o" ] && [ "$o" -nt "$tree/src/gpu/shaders_wide/proj_gemm_wide.hlsl" ] || "$DXC" -T cs_6_2 -E main -Fo "$o" \
				"$tree/src/gpu/shaders_wide/proj_gemm_wide.hlsl" -O3 -HV 2018 -WX -D SSLM_WIDE_H=$h -D SSLM_GEMM_SITE=$gi \
				-I "$tree/src/gpu/shaders" >/dev/null
			gi=$((gi + 1))
		done
	done
}

# build TREE OBJDIR "EXTRA" EXE CELL...: the tree's engine sources (wine_gpu.sh's SRCS, src/ entries) plus the
# cell sources, compiled in TREE (cell paths relative to it) and linked to EXE.
build() {
	local tree="$1" obj="$2" extra="$3" exe="$4"
	shift 4
	local engine
	engine="$(cd "$tree" && WINE_GPU_SOURCED=1 bash -c '. tools/cloud_gpu/wine_gpu.sh; for s in $SRCS; do case "$s" in src/*) echo "$s" ;; esac; done')"
	mkdir -p "$obj" "$(dirname "$exe")"
	( cd "$tree" && printf '%s\n' $engine "$@" | xargs -P "$JOBS" -I{} bash -c \
		'o="$1/$(echo "$2" | tr / _).o"; [ "$o" -nt "$2" ] && exit 0; '"$CXX $FLAGS $extra"' -w -c "$2" -o "$o" || { echo "COMPILE FAILED: $2"; exit 1; }' _ "$obj" {} )
	"$CXX" -O2 "$obj"/*.o -o "$exe" -ld3d12 -ldxgi -ldxguid -lshlwapi -lpsapi -static -static-libgcc -static-libstdc++
	cp "$VKP_DIR/d3d12.dll" "$VKP_DIR/d3d12core.dll" "$(dirname "$exe")/"
}

# Fixtures: tools/gen_paged_kv_fixture.py's (slow: it writes every variant, ~30 min of CPU), reused from MR_FIXTURES
# or the MR1 smoke's directory when they are already there.
FIX="$OUT/fixtures"
fixtures() {
	mkdir -p "$FIX"
	local src
	for src in "${MR_FIXTURES:-}" "$REPO/out/mr1-smoke/fixtures"; do
		[ -n "$src" ] && [ -f "$src/pkv_def.sslm" ] && [ -f "$src/pkv_qk.sslm" ] && [ ! -f "$FIX/pkv_def.sslm" ] && cp "$src"/pkv_*.sslm "$FIX/"
	done
	[ -f "$FIX/pkv_def.sslm" ] && [ -f "$FIX/pkv_qk.sslm" ] || python3 tools/gen_paged_kv_fixture.py "$FIX" >/dev/null
}
WINE_UP=0
wine_up() { [ "$WINE_UP" = 1 ] || { start_xvfb; wine_env; WINE_UP=1; }; }

