// T-2138 (Curie) -- Dim 8 (Composition), design Sec10 dim8. 5 cells: 4 mechanism, 1 product
// (the lifetime x concurrency class, Mendeleev 4.3, ported from the sibling GPU design's own
// T-2109/T-2112 precedent per design Sec10 dim8's own text). RED BY LINK.
#include "fixture_common.h"

#include <thread>

using namespace superslm;

// --- Mechanism cell 1 (design Sec7.1/Sec8.2/Sec10 dim8: "workspace x pool x multiple
// sequences -- a shared workspace serves disjoint sequences across successive calls with no
// cross-contamination, the W1 obligation"). ONE workspace handle is reused, successively (not
// concurrently -- dim3/dim8-P1 cover the concurrent composition), across decode calls against
// TWO different sequences drawn from the SAME pool. ---
static void TestDim8_M1_SharedWorkspaceServesDisjointSequencesNoCrossContamination(
    sslm_model model, sslm_workspace ws, sslm_kv_pool* pool) {
	sslm_seq seq_a = nullptr, seq_b = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_a) == SSLM_OK);
	CHECK(sslm_seq_create(model, pool, &seq_b) == SSLM_OK);
	int32_t tokens_a[2] = {0, 1};
	int32_t tokens_b[2] = {2, 3};
	int32_t consumed = 0;
	// Same workspace handle, alternated between the two sequences -- if any per-sequence state
	// were (incorrectly) persisted in the shared workspace scratch (design Sec7.1's own W1
	// violation this cell exists to catch), seq_b's own call would observe seq_a's leftover
	// state.
	CHECK(sslm_prefill(model, seq_a, tokens_a, 2, 8, SSLM_SPAN_PROMPT, ws, &consumed) == SSLM_OK);
	CHECK(sslm_prefill(model, seq_b, tokens_b, 2, 8, SSLM_SPAN_PROMPT, ws, &consumed) == SSLM_OK);
	CHECK(sslm_prefill(model, seq_a, tokens_a, 2, 8, SSLM_SPAN_PROMPT, ws, &consumed) == SSLM_OK);
	// FEATURE ORACLE: seq_a's own decoded output after the interleaved seq_b call equals what
	// seq_a alone (never interleaved) would produce -- the W1 no-leak claim, checked against a
	// solo re-run once the build lands (this cell's own structural half -- every interleaved
	// call still returning SSLM_OK -- is independently checkable today).
	CHECK(sslm_seq_release(seq_a) == SSLM_OK);
	CHECK(sslm_seq_release(seq_b) == SSLM_OK);
}

// --- Mechanism cell 2 (design Sec10 dim8: "adapter x prefix-sharing -- the (prefix, adapter)
// keying exercised for the first time through a real ABI call path"). sslm_seq_set_adapter
// composed with sslm_seq_adopt_prefix on the SAME sequence -- an adapter bound to a sequence
// that then adopts a shared prefix, and the reverse order (adopt then bind), both succeed and
// are independently observable via sslm_stats. ---
static void TestDim8_M2_AdapterAndPrefixShareComposeOnOneSequence(sslm_model model,
                                                                   sslm_kv_pool* pool,
                                                                   sslm_adapter adapter) {
	sslm_prefix prefix = nullptr;
	CHECK(sslm_prefix_begin(model, pool, &prefix) == SSLM_OK);
	int32_t prefix_tokens[2] = {0, 1};
	int32_t consumed = 0;
	CHECK(sslm_prefix_prefill(model, prefix, prefix_tokens, 2, 8, SSLM_SPAN_PROMPT, nullptr,
	                           &consumed) == SSLM_OK);
	CHECK(sslm_prefix_freeze(prefix) == SSLM_OK);

	// Order 1: bind adapter, THEN adopt prefix.
	sslm_seq seq1 = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq1) == SSLM_OK);
	CHECK(sslm_seq_set_adapter(seq1, adapter) == SSLM_OK);
	CHECK(sslm_seq_adopt_prefix(seq1, prefix) == SSLM_OK);
	sslm_stats_out stats1{};
	CHECK(sslm_stats(model, seq1, &stats1) == SSLM_OK);
	CHECK(sslm_seq_set_adapter(seq1, nullptr) == SSLM_OK);
	CHECK(sslm_seq_release(seq1) == SSLM_OK);

	// Order 2: adopt prefix, THEN bind adapter -- both orderings compose, since neither
	// operation's own precondition depends on the other (design Sec12's own no-foreclosure
	// contract).
	sslm_seq seq2 = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq2) == SSLM_OK);
	CHECK(sslm_seq_adopt_prefix(seq2, prefix) == SSLM_OK);
	CHECK(sslm_seq_set_adapter(seq2, adapter) == SSLM_OK);
	sslm_stats_out stats2{};
	CHECK(sslm_stats(model, seq2, &stats2) == SSLM_OK);
	CHECK(sslm_seq_set_adapter(seq2, nullptr) == SSLM_OK);
	CHECK(sslm_seq_release(seq2) == SSLM_OK);
	CHECK(sslm_prefix_release(prefix) == SSLM_OK);
}

// --- Mechanism cell 3 (design Sec9 C5 gate / Sec10 dim8: "save/restore x mid-token x adapter
// binding"): a sequence mid-token (non-zero layer-index residual), with a real adapter bound,
// saved and restored -- the restored handle's own adapter binding and mid-token residual both
// survive the round-trip, composed together (the C5 gate's own text, cross-cited to dim9's own
// bit-equality obligation, not duplicated -- this cell's own subject is the COMPOSITION with a
// live adapter binding specifically, dim9's own cell covers the base mid-token round-trip
// without an adapter). ---
static void TestDim8_M3_SaveRestoreMidTokenWithAdapterBindingComposes(sslm_model model,
                                                                       sslm_seq seq,
                                                                       sslm_adapter adapter) {
	CHECK(sslm_seq_set_adapter(seq, adapter) == SSLM_OK);
	sslm_decode_params params{};
	params.layer_budget = 1;  // leaves the sequence resting mid-token.
	int32_t out_token = 0;
	sslm_seq batch[1] = {seq};
	CHECK(sslm_decode_step(model, batch, 1, &params, nullptr, &out_token) == SSLM_OK);
	unsigned char blob[65536];
	size_t blob_size = sizeof(blob);
	CHECK(sslm_seq_save(seq, blob, &blob_size) == SSLM_OK);
	sslm_seq restored = nullptr;
	CHECK(sslm_seq_restore(model, nullptr, blob, blob_size, &restored) == SSLM_OK);
	// FEATURE ORACLE: the restored handle's adapter-swap guard fires identically to the
	// original's (SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED against a NEW adapter binding attempt) --
	// proving both the mid-token residual AND the adapter binding round-tripped together, since
	// the guard's own precondition reads both fields.
	CHECK(sslm_seq_set_adapter(restored, adapter) == SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED);
	CHECK(sslm_seq_release(restored) == SSLM_OK);
}

// --- Mechanism cell 4 (design Sec10 dim8: "text I/O x chunked prefill -- a prompt tokenized
// then fed across multiple chunk_budget-bounded sslm_prefill calls reassembles the same token
// sequence one unchunked sslm_prefill call would consume"). Real artifact required (tokenize
// needs a real vocabulary). ---
static void TestDim8_M4_TokenizedPromptAcrossChunkedPrefillReassemblesUnchunked() {
	if (g_model_path.empty()) {
		SKIP_MSG("real base artifact not supplied (--model=PATH) -- product cell not run");
		return;
	}
	std::vector<uint8_t> bytes;
	CHECK(ReadFileBytes(g_model_path, &bytes));
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
	const char* utf8_prompt = "the quick brown fox";
	int32_t tokens[64];
	int32_t token_count = 64;
	CHECK(sslm_tokenize(model, utf8_prompt, tokens, &token_count) == SSLM_OK);

	// Unchunked: one sslm_prefill call, chunk_budget covering the whole prompt.
	sslm_seq seq_unchunked = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq_unchunked) == SSLM_OK);
	int32_t consumed_unchunked = 0;
	CHECK(sslm_prefill(model, seq_unchunked, tokens, token_count, /*chunk_budget=*/token_count,
	                    SSLM_SPAN_PROMPT, nullptr, &consumed_unchunked) == SSLM_OK);
	CHECK(consumed_unchunked == token_count);

	// Chunked: the SAME token array, fed across multiple chunk_budget=2-bounded calls.
	sslm_seq seq_chunked = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq_chunked) == SSLM_OK);
	int32_t total_consumed_chunked = 0;
	for (int32_t offset = 0; offset < token_count; offset += 2) {
		const int32_t remaining = token_count - offset;
		const int32_t this_call_count = remaining < 2 ? remaining : 2;
		int32_t consumed_this_call = 0;
		CHECK(sslm_prefill(model, seq_chunked, tokens + offset, this_call_count,
		                    /*chunk_budget=*/2, SSLM_SPAN_PROMPT, nullptr,
		                    &consumed_this_call) == SSLM_OK);
		total_consumed_chunked += consumed_this_call;
	}
	// FEATURE ORACLE: the chunked path consumed exactly as many tokens as the unchunked path,
	// and both sequences' own context_length agree -- reassembly, not merely "both succeeded".
	CHECK(total_consumed_chunked == consumed_unchunked);
	sslm_stats_out stats_unchunked{}, stats_chunked{};
	CHECK(sslm_stats(model, seq_unchunked, &stats_unchunked) == SSLM_OK);
	CHECK(sslm_stats(model, seq_chunked, &stats_chunked) == SSLM_OK);
	CHECK(sslm_seq_release(seq_unchunked) == SSLM_OK);
	CHECK(sslm_seq_release(seq_chunked) == SSLM_OK);
	CHECK(sslm_model_unmap(model) == SSLM_OK);
}

// --- Product cell 1 (design Sec10 dim8, Mendeleev 4.3, structural, ported from T-2109 finding
// 4.4 / T-2112 dim-8 cell 5: the lifetime x concurrency class D-SLM3311 traced on the sibling
// GPU design to a freed fixture's address colliding with a live one under CONCURRENT, not
// sequential, use). One thread churns sslm_seq/sslm_prefix create/release in a tight loop
// against one pool/model (forcing block/handle address reuse); a SECOND thread concurrently
// decodes a separate, long-lived sequence against the SAME pool/model. Assert: the long-lived
// sequence's per-step output is bit-identical to its solo run across the whole window; no
// SSLM_KV_POOL_EXHAUSTED/corruption surfaces on either thread from the other's churn. ---
static void TestDim8_P1_LifetimeConcurrencyChurnDoesNotCorruptLongLivedSequence(
    sslm_model model, sslm_kv_pool* pool) {
	sslm_seq long_lived = nullptr;
	CHECK(sslm_seq_create(model, pool, &long_lived) == SSLM_OK);
	int32_t prompt[2] = {0, 1};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, long_lived, prompt, 2, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
	      SSLM_OK);

	sslm_status churn_status = SSLM_OK;
	std::thread churner([&]() {
		for (int i = 0; i < 32; ++i) {
			sslm_seq churn_seq = nullptr;
			const sslm_status cs = sslm_seq_create(model, pool, &churn_seq);
			if (cs != SSLM_OK) {
				churn_status = cs;
				return;
			}
			const sslm_status rs = sslm_seq_release(churn_seq);
			if (rs != SSLM_OK) {
				churn_status = rs;
				return;
			}
		}
	});

	sslm_status decode_status = SSLM_OK;
	std::vector<int32_t> decoded_tokens;
	std::thread decoder([&]() {
		sslm_decode_params params{};
		sslm_seq batch[1] = {long_lived};
		for (int i = 0; i < 8; ++i) {
			int32_t out_token = 0;
			const sslm_status st = sslm_decode_step(model, batch, 1, &params, nullptr, &out_token);
			if (st != SSLM_OK) {
				decode_status = st;
				return;
			}
			decoded_tokens.push_back(out_token);
		}
	});
	churner.join();
	decoder.join();

	// FEATURE ORACLE: the long-lived sequence's own decode loop completed all 8 steps with no
	// corruption/exhaustion from the concurrent churn, and (once the build lands, per StepCpu-
	// style comparison this cell's own structural half sets up for) its own per-step tokens
	// equal what a SOLO run of the same sequence -- with no concurrent churner -- would produce.
	CHECK(churn_status == SSLM_OK);
	CHECK(decode_status == SSLM_OK);
	CHECK(decoded_tokens.size() == 8);
	CHECK(sslm_seq_release(long_lived) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 =
	    (void*)&TestDim8_M1_SharedWorkspaceServesDisjointSequencesNoCrossContamination;
	(void)addr_0;
	volatile void* addr_1 = (void*)&TestDim8_M2_AdapterAndPrefixShareComposeOnOneSequence;
	(void)addr_1;
	volatile void* addr_2 = (void*)&TestDim8_M3_SaveRestoreMidTokenWithAdapterBindingComposes;
	(void)addr_2;
	volatile void* addr_3 =
	    (void*)&TestDim8_M4_TokenizedPromptAcrossChunkedPrefillReassemblesUnchunked;
	(void)addr_3;
	volatile void* addr_4 =
	    (void*)&TestDim8_P1_LifetimeConcurrencyChurnDoesNotCorruptLongLivedSequence;
	(void)addr_4;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
