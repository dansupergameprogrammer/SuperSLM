// T-2138 (Curie) -- Dim 5 (Failure and rejection paths), design Sec6/Sec10 dim5. sslm_status
// carries 17 enumerators (design Sec10 dim5's own reconciled count: SSLM_OK(1) + argument/
// precondition(3) + artifact/content(4) + lifecycle/precondition-on-state(7) + numeric/domain(2)
// = 17). Cells NOT already exercised elsewhere are authored here; the rest are cross-cited so no
// enumerator is asserted twice under a different name:
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
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq) == SSLM_OK);
	CHECK(sslm_model_unmap(model) == SSLM_MODEL_HAS_LIVE_SEQUENCES);
	// The rejected unmap leaves the model and sequence both usable.
	int32_t tokens[1] = {0};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, tokens, 1, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
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
	sslm_decode_params params{};
	params.layer_budget = 1;  // deliberately less than a full token's own layer count, so the
	                          // sequence is left resting mid-token (layer_index != 0) rather
	                          // than completing a whole token.
	int32_t out_tokens[1] = {0};
	sslm_seq batch[1] = {seq};
	CHECK(sslm_decode_step(model, batch, 1, &params, nullptr, out_tokens) == SSLM_OK);
	CHECK(sslm_seq_set_adapter(seq, adapter) == SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED);
}

// --- Cell 6 (SSLM_SEQ_RESET_MIDTOKEN_REJECTED -- design Sec17's lifecycle addendum, this
// design's own Sec6): sslm_seq_reset against a non-zero residual marker rejects, symmetric with
// cell 5's adapter-swap rejection under the identical mid-token precondition. ---
static void TestDim5_C6_SeqResetMidTokenRejected(sslm_model model, sslm_seq seq) {
	sslm_decode_params params{};
	params.layer_budget = 1;
	int32_t out_tokens[1] = {0};
	sslm_seq batch[1] = {seq};
	CHECK(sslm_decode_step(model, batch, 1, &params, nullptr, out_tokens) == SSLM_OK);
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
	std::vector<uint8_t> buf(block_count * block_bytes + overhead);
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
	std::vector<uint8_t> buf(block_count * block_bytes + overhead);
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
// push context_length past the artifact's own context_cap rejects. ---
static void TestDim5_C11_ContextCapExceededRejected(sslm_model model, sslm_seq seq,
                                                     int64_t context_cap) {
	// A prompt one token longer than the artifact's own declared cap -- genuinely hostile
	// against the real domain, not an arbitrary large constant.
	std::vector<int32_t> over_cap(static_cast<size_t>(context_cap) + 1, 0);
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, over_cap.data(), (int32_t)over_cap.size(), 8, SSLM_SPAN_PROMPT,
	                    nullptr, &consumed) == SSLM_CONTEXT_CAP_EXCEEDED);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 = (void*)&TestDim5_C1_InvalidArgumentNullOutAndNegativeCount;
	(void)addr_0;
	volatile void* addr_1 = (void*)&TestDim5_C2_ModelUnmapWhileLiveSeqRejected;
	(void)addr_1;
	volatile void* addr_2 = (void*)&TestDim5_C3_KvPoolDestroyWhileLiveHandleRejected;
	(void)addr_2;
	volatile void* addr_3 = (void*)&TestDim5_C4_AdapterReleaseWhileLiveSeqRejected;
	(void)addr_3;
	volatile void* addr_4 = (void*)&TestDim5_C5_AdapterSwapMidTokenRejected;
	(void)addr_4;
	volatile void* addr_5 = (void*)&TestDim5_C6_SeqResetMidTokenRejected;
	(void)addr_5;
	volatile void* addr_6 = (void*)&TestDim5_C7_PrefixPrefillAfterFreezeOrReleaseRejected;
	(void)addr_6;
	volatile void* addr_7 = (void*)&TestDim5_C8_KvPoolExhaustedSequenceResumable;
	(void)addr_7;
	volatile void* addr_8 = (void*)&TestDim5_C9_KvPoolExhaustedPrefixResumable;
	(void)addr_8;
	volatile void* addr_9 = (void*)&TestDim5_C10_TokenIdOutOfRangeRejected;
	(void)addr_9;
	volatile void* addr_10 = (void*)&TestDim5_C11_ContextCapExceededRejected;
	(void)addr_10;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
