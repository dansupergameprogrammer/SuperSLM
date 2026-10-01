#!/usr/bin/env bash
# Paged-KV plan (rev 16.1) step C1's gate: "the suite builds and is red for the stated reason".
#
# For each per-step executable (tests/paged-kv/paged_kv.cmake) this builds it in BUILD_DIR and
# classifies the result:
#   red by link   every compile succeeds, and the link fails ONLY on symbols the test headers
#                 declare as owed (pkv_abi.h's verbs and seams, pkv_engine_api.h, pkv_module_api.h);
#   green-able    it links (its step's builder has defined everything it calls);
#   BROKEN        a compile error, or an undefined symbol no test header declares as owed.
# Exit status is non-zero when any target is BROKEN. GNU toolchains (the cloud leg); the box's MSVC
# leg reads LNK2019 lines the same way by hand.
#
# Usage: tools/check_paged_kv_red.sh BUILD_DIR [TARGET ...]
set -uo pipefail
build="${1:?usage: check_paged_kv_red.sh BUILD_DIR [TARGET ...]}"
shift
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(superslm_pkv_c2 superslm_pkv_c3 superslm_pkv_c4 superslm_pkv_c5 superslm_pkv_c6)
owed='^(sslm_kv_page_positions|sslm_kv_page_size|sslm_kv_pages_for_budget|sslm_kv_page_pool_overhead_size|sslm_kv_page_pool_create|sslm_seq_create_budgeted|sslm_prefix_begin_budgeted|sslm_prefix_begin_from|sslm_kv_pool_stats|sslm_seq_kv_stats|sslm_seq_restore_shared|sslm_pkv_test_only_[a-z_]+|superslm::kv_pages::[A-Za-z]+\(.*|superslm::(KeyRow|ValueRow|MutableKeyRow|MutableValueRow)\(superslm::KvPageView const&.*|superslm::RunLayerLoop(ChunkBatched)?\(.*superslm::KvPageView const&.*|superslm::GemmProbQ15AccumulateInto\(.*)$'
broken=0
for t in "${targets[@]}"; do
	if ! cmake --build "$build" --target help 2>/dev/null | grep "\.\.\. $t$" >/dev/null; then
		echo "$t: no such target (no cells for this step yet)"
		continue
	fi
	log="$(cmake --build "$build" --target "$t" -j"$(nproc)" 2>&1)"
	if [ $? -eq 0 ]; then
		echo "$t: links (green-able)"
		continue
	fi
	if grep -qE "error: |fatal error" <<<"$log" && ! grep -q "undefined reference" <<<"$log"; then
		echo "$t: BROKEN (compile error)"
		grep -E "error" <<<"$log" | head -20
		broken=1
		continue
	fi
	syms="$(grep -oE "undefined reference to \`[^']+'" <<<"$log" | sed -E "s/undefined reference to \`(.*)'/\1/" | sort -u)"
	bad="$(grep -vE "$owed" <<<"$syms" || true)"
	n="$(grep -c . <<<"$syms")"
	if [ -n "$bad" ]; then
		echo "$t: BROKEN (undefined symbols no test header owes)"
		echo "$bad" | sed 's/^/    /'
		broken=1
	else
		echo "$t: red by link on $n owed symbols"
		echo "$syms" | sed 's/^/    /'
	fi
done
exit $broken
