// T-2138 (Curie) -- Dim 5 (Failure and rejection paths), design Sec6/Sec10 dim5. This file's own
// dedicated cells target the base rejection families this suite's sslm_abi.h declares (18 base
// enumerators as of the padded-vocabulary fold, SSLM_OK(1) + argument/precondition(3) +
// artifact/content(4) + lifecycle/precondition-on-state(7) + numeric/domain(3) = 18) -- the
// registry has since grown to 26 entries total across both real headers (Sec6's GOVERNANCE
// RULING, design commit 4f4eb23896): G5's own 7 schema statuses (18-24), SSLM_ALLOCATION_FAILED
// (25), and the SSLM_STATUS_NEXT_FREE sentinel are covered by Gate C's own construction, not by
// this dimension's cells, which stay scoped to the base family this suite's own sslm_abi.h
// declares functionally. Cells NOT already exercised elsewhere are authored here; the rest are
// cross-cited so no enumerator is asserted twice under a different name:
//   SSLM_BUFFER_TOO_SMALL, SSLM_MISALIGNED_BUFFER -- dim2 M1/M2/M3.
//   SSLM_ARTIFACT_REJECTED -- dim2 M5. SSLM_ADAPTER_MODEL_MISMATCH -- dim2 M6.
//   SSLM_RESTORE_MODEL_MISMATCH, SSLM_RESTORE_KV_MISMATCH -- dim9 M2/M3.
// 11 cells here: SSLM_OK's own negative-space is implicit in every other file's non-hostile
// calls, not a dedicated cell. RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Cell 1 (SSLM_INVALID_ARGUMENT -- design Sec6's own kept broad catch-all, "a null
// required pointer, a negative count"): a null `out` pointer and a negative `count` are both
// asserted, since the design's own text names both as the SAME remedy class ("the caller's call
// shape is wrong"), not two separate enumerators. ---
static void TestDim5_C1_InvalidArgumentNullOutAndNegativeCount(sslm_model model, sslm_seq seq) {
	int32_t tokens[2] = {0, 1};
	int32_t consumed = 0;
	// Negative count.
	CHECK(sslm_prefill(model, seq, tokens, /*count=*/-1, 8, SSLM_SPAN_PROMPT, nullptr,
	                    &consumed) == SSLM_INVALID_ARGUMENT);
	// Null required out-parameter (consumed itself, on an otherwise well-formed call).
	CHECK(sslm_prefill(model, seq, tokens, 2, 8, SSLM_SPAN_PROMPT, nullptr, /*consumed=*/nullptr) ==
	      SSLM_INVALID_ARGUMENT);
}

// --- Cell 2 (SSLM_MODEL_HAS_LIVE_SEQUENCES -- design Sec6, the D-SLM32 lifecycle audit's own
// concern, enforced not merely documented): sslm_model_unmap while a live sslm_seq still
// references the model rejects, and the sequence's own state is unperturbed by the rejected
// call -- a subsequent release still succeeds. ---
static void TestDim5_C2_ModelUnmapWhileLiveSeqRejected(const void* artifact_bytes,
                                                        size_t artifact_size) {
	sslm_model model = nullptr;
	CHECK(sslm_model_map(artifact_bytes, artifact_size, &model) == SSLM_OK);
	SinglePool sp;
	CHECK(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	CHECK(sslm_model_unmap(model) == SSLM_MODEL_HAS_LIVE_SEQUENCES);
	// The rejected unmap leaves the model and sequence both usable.
	int32_t tokens[1] = {0};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, tokens, 1, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
	CHECK(sslm_model_unmap(model) == SSLM_OK);
}

// --- Cell 3 (SSLM_POOL_HAS_LIVE_HANDLES -- design Sec7.2/Sec6, symmetric with
// SSLM_MODEL_HAS_LIVE_SEQUENCES): sslm_kv_pool_destroy while a live sslm_seq/sslm_prefix still
// holds blocks from it rejects. ---
static void TestDim5_C3_KvPoolDestroyWhileLiveHandleRejected(sslm_model model,
                                                              sslm_kv_pool pool) {
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, &pool, &seq) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(pool) == SSLM_POOL_HAS_LIVE_HANDLES);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
}

// --- Cell 4 (SSLM_ADAPTER_HAS_LIVE_SEQUENCES -- design Sec6/Sec8.3's own refcount contract,
// made a rejection rather than a use-after-free): sslm_adapter_release while a live sslm_seq
// still points at it via sslm_seq_set_adapter rejects. ---
static void TestDim5_C4_AdapterReleaseWhileLiveSeqRejected(sslm_model model, sslm_seq seq,
                                                            sslm_adapter adapter) {
	CHECK(sslm_seq_set_adapter(seq, adapter) == SSLM_OK);
	CHECK(sslm_adapter_release(adapter) == SSLM_ADAPTER_HAS_LIVE_SEQUENCES);
	CHECK(sslm_seq_set_adapter(seq, /*adapter=*/nullptr) == SSLM_OK);  // unbind, plan's own
	                                                                    // NULL-adapter convention.
	CHECK(sslm_adapter_release(adapter) == SSLM_OK);
}

// --- Cell 5 (SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED -- design Sec6, plan's existing name kept):
// sslm_seq_set_adapter against a sequence with a non-zero mid-token residual marker (a partial
// token in flight, layer_index != 0) rejects. ---
static void TestDim5_C5_AdapterSwapMidTokenRejected(sslm_model model, sslm_seq seq,
                                                     sslm_adapter adapter) {
	// decode_step requires a prior prefill (a virgin sequence has no current_token to continue
	// from) -- a real, minimal prompt establishes that before entering mid-token state.
	int32_t setup_prompt[1] = {0};
	int32_t setup_consumed = 0;
	CHECK(sslm_prefill(model, seq, setup_prompt, 1, 8, SSLM_SPAN_PROMPT, nullptr,
	                    &setup_consumed) == SSLM_OK);
	CHECK(EnterMidToken(model, seq));
	CHECK(sslm_seq_set_adapter(seq, adapter) == SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED);
}

// --- Cell 6 (SSLM_SEQ_RESET_MIDTOKEN_REJECTED -- design Sec17's lifecycle addendum, this
// design's own Sec6): sslm_seq_reset against a non-zero residual marker rejects, symmetric with
// cell 5's adapter-swap rejection under the identical mid-token precondition. ---
static void TestDim5_C6_SeqResetMidTokenRejected(sslm_model model, sslm_seq seq) {
	int32_t setup_prompt[1] = {0};
	int32_t setup_consumed = 0;
	CHECK(sslm_prefill(model, seq, setup_prompt, 1, 8, SSLM_SPAN_PROMPT, nullptr,
	                    &setup_consumed) == SSLM_OK);
	CHECK(EnterMidToken(model, seq));
	CHECK(sslm_seq_reset(seq) == SSLM_SEQ_RESET_MIDTOKEN_REJECTED);
}

// --- Cell 7 (SSLM_PREFIX_FROZEN_REJECTED -- design Sec7.2/Sec6, Mendeleev audit §8.3 folded):
// sslm_prefix_prefill against an already-frozen prefix rejects, leaving the prefix's own
// frozen state untouched by the rejected call -- a subsequent sslm_seq_adopt_prefix against the
// SAME prefix still succeeds. Also exercised: the identical rejection against an
// already-RELEASED prefix is a separate, independently-checkable branch (a released handle is
// never silently reused). ---
static void TestDim5_C7_PrefixPrefillAfterFreezeOrReleaseRejected(sslm_model model,
                                                                   sslm_kv_pool* pool) {
	sslm_prefix prefix = nullptr;
	CHECK(sslm_prefix_begin(model, pool, &prefix) == SSLM_OK);
	int32_t tokens[1] = {0};
	int32_t consumed = 0;
	CHECK(sslm_prefix_prefill(model, prefix, tokens, 1, 8, SSLM_SPAN_PROMPT, nullptr,
	                           &consumed) == SSLM_OK);
	CHECK(sslm_prefix_freeze(prefix) == SSLM_OK);
	int32_t more_tokens[1] = {1};
	int32_t more_consumed = 0;
	CHECK(sslm_prefix_prefill(model, prefix, more_tokens, 1, 8, SSLM_SPAN_PROMPT, nullptr,
	                           &more_consumed) == SSLM_PREFIX_FROZEN_REJECTED);
	// The rejected call left the frozen prefix usable -- adoption still succeeds.
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_adopt_prefix(seq, prefix) == SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_prefix_release(prefix) == SSLM_OK);
	// Released, then prefilled: still rejected, never a use-after-free.
	CHECK(sslm_prefix_prefill(model, prefix, more_tokens, 1, 8, SSLM_SPAN_PROMPT, nullptr,
	                           &more_consumed) == SSLM_PREFIX_FROZEN_REJECTED);
}

// --- Cell 8 (SSLM_KV_POOL_EXHAUSTED, sequence side -- RE-DERIVED against the whole-block
// buffer model, commit fab235c1c6: "SSLM_KV_POOL_EXHAUSTED now fires at
// sslm_prefix_begin/sslm_seq_create (no free block in the pool), never mid-sslm_prefix_prefill"
// -- a block is drawn ONCE, at construction, never re-drawn mid-call, so exhaustion is
// structurally impossible mid-sslm_prefill now (content exceeding a block's own capacity
// mid-call is SSLM_CONTEXT_CAP_EXCEEDED instead, C11 below, unchanged by this fold). This cell
// exhausts the pool at sslm_seq_create itself and proves the rejected call leaves the SEQUENCE
// handle unallocated (never a torn/half-built handle) and the pool resumable once a block
// frees. ---
static void TestDim5_C8_KvPoolExhaustedSequenceResumable(sslm_model model) {
	const uint32_t block_count = 1;  // deliberately tiny -- one concurrent sequence only.
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	AlignedBuffer buf(block_count * block_bytes + overhead);
	sslm_kv_pool pool = nullptr;
	CHECK(sslm_kv_pool_create(model, buf.data(), buf.size(), block_count, &pool) == SSLM_OK);
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, &pool, &seq) == SSLM_OK);  // draws the pool's only block.

	// The pool's only block is already held -- a SECOND sslm_seq_create exhausts it.
	sslm_seq over_capacity = nullptr;
	const sslm_status exhausted_status = sslm_seq_create(model, &pool, &over_capacity);
	CHECK(exhausted_status == SSLM_KV_POOL_EXHAUSTED);
	CHECK(over_capacity == nullptr);
	// FEATURE ORACLE: the FIRST sequence is completely unaffected by the second, rejected
	// sslm_seq_create -- it still decodes correctly, proving the exhausted call never touched
	// an already-allocated handle.
	int32_t prompt[1] = {0};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, prompt, 1, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	// Releasing the first sequence frees its block -- a subsequent sslm_seq_create against the
	// now-relieved pool succeeds, proving the exhausted call left the pool's own bookkeeping in
	// a defined, resumable state rather than a torn one.
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	sslm_seq seq2 = nullptr;
	CHECK(sslm_seq_create(model, &pool, &seq2) == SSLM_OK);
	CHECK(sslm_seq_release(seq2) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
}

// --- Cell 9 (SSLM_KV_POOL_EXHAUSTED, prefix side -- RE-DERIVED against the whole-block buffer
// model, commit fab235c1c6, "now stated for both handle kinds this pool serves"): the identical
// construction-time exhaustion against sslm_prefix_begin, distinct from a sequence exhausting
// the SAME pool -- a prefix and a sequence draw from the same free list, so a pool exhausted by
// one live sequence also rejects a prefix_begin attempt, proving the two handle kinds genuinely
// share one pool's own accounting rather than each having a private, un-enforced allotment. ---
static void TestDim5_C9_KvPoolExhaustedPrefixResumable(sslm_model model) {
	const uint32_t block_count = 1;
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	AlignedBuffer buf(block_count * block_bytes + overhead);
	sslm_kv_pool pool = nullptr;
	CHECK(sslm_kv_pool_create(model, buf.data(), buf.size(), block_count, &pool) == SSLM_OK);

	// Exhaust the pool's only block with a live SEQUENCE first (the cross-handle-kind half of
	// this cell's own claim), then attempt sslm_prefix_begin against the same exhausted pool.
	sslm_seq holder = nullptr;
	CHECK(sslm_seq_create(model, &pool, &holder) == SSLM_OK);
	sslm_prefix over_capacity = nullptr;
	CHECK(sslm_prefix_begin(model, &pool, &over_capacity) == SSLM_KV_POOL_EXHAUSTED);
	CHECK(over_capacity == nullptr);

	// FEATURE ORACLE: releasing the sequence's block makes the pool resumable for a prefix --
	// sslm_prefix_begin succeeds once a block genuinely frees, and the fresh prefix is usable
	// (prefill, freeze, release all succeed) rather than permanently wedged by the earlier
	// exhaustion.
	CHECK(sslm_seq_release(holder) == SSLM_OK);
	sslm_prefix prefix = nullptr;
	CHECK(sslm_prefix_begin(model, &pool, &prefix) == SSLM_OK);
	int32_t small_tokens[1] = {0};
	int32_t small_consumed = 0;
	CHECK(sslm_prefix_prefill(model, prefix, small_tokens, 1, 8, SSLM_SPAN_PROMPT, nullptr,
	                           &small_consumed) == SSLM_OK);
	CHECK(sslm_prefix_freeze(prefix) == SSLM_OK);
	CHECK(sslm_prefix_release(prefix) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
}

// --- Cell 10 (SSLM_TOKEN_ID_OUT_OF_RANGE -- design Sec6, mirroring the GPU ABI's own B3.5
// addition, D-SLM3367): a token id outside [0, vocab_size) reaching sslm_prefill rejects. ---
static void TestDim5_C10_TokenIdOutOfRangeRejected(sslm_model model, sslm_seq seq,
                                                    int32_t vocab_size) {
	int32_t hostile_tokens[1] = {vocab_size};  // exactly one past the valid range.
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, hostile_tokens, 1, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
	      SSLM_TOKEN_ID_OUT_OF_RANGE);
	int32_t negative_token[1] = {-1};
	CHECK(sslm_prefill(model, seq, negative_token, 1, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
	      SSLM_TOKEN_ID_OUT_OF_RANGE);
}

// --- Cell 11 (SSLM_CONTEXT_CAP_EXCEEDED -- design Sec6): a prefill/decode_step call that would
// push context_length past the artifact's own context_cap rejects.
//
// GROUNDED AGAINST THE REAL IMPLEMENTATION (StandardsDocument.md Sec5.4 -- exactness verified at
// source, not by construction): the check (src/sslm_abi.cpp's own PrefillWholeTokens) is
// `if (state.context_length >= context_cap) return SSLM_CONTEXT_CAP_EXCEEDED;`, evaluated PER
// TOKEN, using context_length as it stands AT CALL ENTRY for the first token -- never a
// pre-flight check against the full requested count. A real 1.5B artifact's context_cap is
// 32768; reaching it by actually prefilling real content would mean processing ~32768 real
// per-token forward passes (this cell's own prior version tried exactly that, at a chunk_budget
// wide enough to attempt the whole span in one call, and did not return in a reasonable time --
// StandardsDocument.md Sec5.6, a ruling contradicted by measurement is re-opened, not defended).
//
// The cheap, still-real construction: save a real (tiny, non-mid-token) sequence, tamper ONLY
// the blob's own context_length field (the real implementation's exact byte offset, cited from
// src/sslm_abi.cpp's own sslm_seq_save: magic(4) + model_hash(32) + kv_precision(4) +
// schema_name_hash(8) + dfa_walk_state(4) + adapter_binding_id(8) = offset 60, 8 bytes LE) to
// context_cap itself, restore (every OTHER field, including model_hash, is untouched and still
// valid, so restore succeeds structurally), then issue ONE small prefill call -- the FIRST
// token's own check now reads context_length == context_cap and rejects immediately, no real
// per-token compute required to reach it. ---
static void TestDim5_C11_ContextCapExceededRejected(sslm_model model, sslm_seq seq,
                                                     int64_t context_cap, sslm_kv_pool* pool) {
	int32_t setup_prompt[1] = {0};
	int32_t setup_consumed = 0;
	CHECK(sslm_prefill(model, seq, setup_prompt, 1, 8, SSLM_SPAN_PROMPT, nullptr,
	                    &setup_consumed) == SSLM_OK);

	SeqBlobBuffer blob(model);
	CHECK(sslm_seq_save(seq, blob.bytes.data(), &blob.size) == SSLM_OK);
	constexpr size_t kContextLengthOffset = 60;  // cited from src/sslm_abi.cpp's own field order
	                                              // above -- 8 bytes, little-endian uint64.
	CHECK_MSG(blob.size > kContextLengthOffset + 8,
	          "blob too small (%zu bytes) to carry a context_length field at offset %zu",
	          blob.size, kContextLengthOffset);
	const uint64_t tampered_context_length = static_cast<uint64_t>(context_cap);
	for (int i = 0; i < 8; ++i) {
		blob.bytes[kContextLengthOffset + i] =
		    static_cast<uint8_t>((tampered_context_length >> (8 * i)) & 0xFF);
	}

	sslm_seq restored = nullptr;
	CHECK(sslm_seq_restore(model, pool, blob.bytes.data(), blob.size, &restored) == SSLM_OK);
	if (!restored) return;

	int32_t over_cap_tokens[2] = {0, 1};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, restored, over_cap_tokens, 2, 8, SSLM_SPAN_PROMPT, nullptr,
	                    &consumed) == SSLM_CONTEXT_CAP_EXCEEDED);
	CHECK(sslm_seq_release(restored) == SSLM_OK);
}

// REAL INVOCATION DRIVER (house pattern) -- supersedes the address-only convention. The
// builder's own ad-hoc driver crashed (STATUS_HEAP_CORRUPTION, root cause unisolated) and was
// reverted to link-only (Claude/Brunel/t2139-abi-build-2026-08-16.md S6). Authored fresh here:
// C1/C3/C4/C5/C6/C10/C11 each get their OWN dedicated, freshly-created sequence from one pool
// sized for the whole set (block_count=7), so no cell's own mid-token/hostile-input mutation
// (C5/C6 deliberately leave a sequence mid-token; C10/C11 deliberately feed hostile content)
// can be observed by a later cell reusing the same handle. C7/C8/C9 build their own dedicated
// pools (self-contained already). C2 is fully self-contained.
int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim5 C1/C3-C6/C8-C11 not run");
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}
	SslmModelView view;
	std::vector<uint8_t> bytes;
	std::string err;
	if (!LoadRealModelView(g_model_path, &view, &bytes, &err)) {
		SKIP_MSG("could not load real artifact: %s", err.c_str());
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
	if (model) {
		const int32_t vocab_size = static_cast<int32_t>(view.config.vocab_size);
		const int64_t context_cap = static_cast<int64_t>(view.config.context_cap);

		// C1, C4, C5, C6, C10 -- 5 dedicated sequences, one shared pool (C3/C11 get their own
		// pools below: C3 destroys the pool itself as part of its own test; C11 needs a SECOND
		// concurrent block for its own restored handle).
		const uint32_t bc = 5;
		const size_t block_bytes = sslm_kv_block_size(model);
		AlignedBuffer pool_buf(bc * block_bytes + sslm_kv_pool_overhead_size(model, bc));
		sslm_kv_pool pool = nullptr;
		CHECK(sslm_kv_pool_create(model, pool_buf.data(), pool_buf.size(), bc, &pool) == SSLM_OK);
		if (pool) {
			sslm_seq seq_c1 = nullptr, seq_c4 = nullptr, seq_c5 = nullptr, seq_c6 = nullptr,
			         seq_c10 = nullptr;
			CHECK(sslm_seq_create(model, &pool, &seq_c1) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_c4) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_c5) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_c6) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_c10) == SSLM_OK);

			if (seq_c1) {
				TestDim5_C1_InvalidArgumentNullOutAndNegativeCount(model, seq_c1);
				CHECK(sslm_seq_release(seq_c1) == SSLM_OK);
			}
			if (g_adapter_path.empty()) {
				SKIP_MSG("--adapter=PATH not supplied -- dim5 C4/C5 not run");
			} else {
				std::vector<uint8_t> adapter_bytes;
				CHECK(ReadFileBytes(g_adapter_path, &adapter_bytes));
				sslm_adapter adapter_c4 = nullptr, adapter_c5 = nullptr;
				CHECK(sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), model,
				                        &adapter_c4) == SSLM_OK);
				CHECK(sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), model,
				                        &adapter_c5) == SSLM_OK);
				if (seq_c4 && adapter_c4) {
					TestDim5_C4_AdapterReleaseWhileLiveSeqRejected(model, seq_c4, adapter_c4);
				} else if (adapter_c4) {
					CHECK(sslm_adapter_release(adapter_c4) == SSLM_OK);
				}
				if (seq_c5 && adapter_c5) {
					TestDim5_C5_AdapterSwapMidTokenRejected(model, seq_c5, adapter_c5);
					CHECK(sslm_adapter_release(adapter_c5) == SSLM_OK);
				} else if (adapter_c5) {
					CHECK(sslm_adapter_release(adapter_c5) == SSLM_OK);
				}
			}
			if (seq_c4) CHECK(sslm_seq_release(seq_c4) == SSLM_OK);
			if (seq_c5) CHECK(sslm_seq_release(seq_c5) == SSLM_OK);

			if (seq_c6) {
				TestDim5_C6_SeqResetMidTokenRejected(model, seq_c6);
				CHECK(sslm_seq_release(seq_c6) == SSLM_OK);
			}
			if (seq_c10) {
				TestDim5_C10_TokenIdOutOfRangeRejected(model, seq_c10, vocab_size);
				CHECK(sslm_seq_release(seq_c10) == SSLM_OK);
			}
			CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
		}

		// C11 gets its own dedicated 2-block pool -- its own seq stays live while its own
		// tampered-blob restore creates a SECOND handle, both concurrently held mid-cell.
		{
			SinglePool sp11;
			sslm_seq seq_c11 = nullptr;
			if (MakePool(model, 2, &sp11))
				CHECK(sslm_seq_create(model, &sp11.pool, &seq_c11) == SSLM_OK);
			if (seq_c11) {
				TestDim5_C11_ContextCapExceededRejected(model, seq_c11, context_cap, &sp11.pool);
				CHECK(sslm_seq_release(seq_c11) == SSLM_OK);
			}
			if (sp11.pool) CHECK(sslm_kv_pool_destroy(sp11.pool) == SSLM_OK);
		}

		// C3 gets its own dedicated pool -- it destroys the pool itself as part of its own test
		// (SSLM_POOL_HAS_LIVE_HANDLES while its own seq is live, then a real destroy after
		// release), which no other cell can share a pool with.
		{
			const uint32_t bc3 = 1;
			AlignedBuffer pool_buf3(bc3 * block_bytes + sslm_kv_pool_overhead_size(model, bc3));
			sslm_kv_pool pool3 = nullptr;
			CHECK(sslm_kv_pool_create(model, pool_buf3.data(), pool_buf3.size(), bc3, &pool3) ==
			      SSLM_OK);
			if (pool3) TestDim5_C3_KvPoolDestroyWhileLiveHandleRejected(model, pool3);
		}

		// C7, C8, C9 build their own dedicated pools internally.
		{
			const uint32_t bc7 = 2;
			AlignedBuffer pool_buf7(bc7 * block_bytes + sslm_kv_pool_overhead_size(model, bc7));
			sslm_kv_pool pool7 = nullptr;
			CHECK(sslm_kv_pool_create(model, pool_buf7.data(), pool_buf7.size(), bc7, &pool7) ==
			      SSLM_OK);
			if (pool7) {
				TestDim5_C7_PrefixPrefillAfterFreezeOrReleaseRejected(model, &pool7);
				CHECK(sslm_kv_pool_destroy(pool7) == SSLM_OK);
			}
		}
		TestDim5_C8_KvPoolExhaustedSequenceResumable(model);
		TestDim5_C9_KvPoolExhaustedPrefixResumable(model);

		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	// C2 is fully self-contained (maps/unmaps its own dedicated model handle).
	TestDim5_C2_ModelUnmapWhileLiveSeqRejected(bytes.data(), bytes.size());
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
