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

// --- Mechanism cell 2 (design Sec10 dim8, RE-DERIVED against copy-on-adopt, commit
// fab235c1c6 -- the plan's own (prefix, adapter)-keyed physical-block-sharing framing does not
// survive Sec7.2's ruling; this is the design's own replacement cell, not a patch of the prior
// one). `sslm_prefix_prefill` carries no adapter parameter (base-only construction, always), so
// a copied prefix's own bytes are exactly its own base-only computation; an adapter bound via
// `sslm_seq_set_adapter` AFTER `sslm_seq_adopt_prefix` shapes generation from the copy point
// forward only -- structurally the SAME shape as the already-proven mid-sequence adapter-swap
// semantic (D-SLM28: forward-only, history preserved). THE CELL (design's own text): a sequence
// that adopts a prefix then binds an adapter reproduces, bit-for-bit, the SAME sequence built by
// PREFILLING THE PREFIX'S OWN CONTENT DIRECTLY (base-only, matching what copy-on-adopt would
// have copied -- C3's own gate: "sslm_seq_adopt_prefix's own copy reproduces a fresh independent
// prefill's output bit-for-bit at the copied positions") and then binding the SAME adapter
// immediately afterward -- proving copy-on-adopt-then-bind and prefill-then-swap are the one
// mechanism, not two. ---
static void TestDim8_M2_AdapterBoundAfterPrefixAdoptMatchesFreshPrefillThenSwap(
    sslm_model model, sslm_kv_pool* pool, sslm_adapter adapter) {
	int32_t shared_content[3] = {0, 1, 2};

	// The prefix, built and frozen the ordinary way -- its own content is base-only by
	// construction (sslm_prefix_prefill takes no adapter parameter).
	sslm_prefix prefix = nullptr;
	CHECK(sslm_prefix_begin(model, pool, &prefix) == SSLM_OK);
	int32_t prefix_consumed = 0;
	CHECK(sslm_prefix_prefill(model, prefix, shared_content, 3, 8, SSLM_SPAN_PROMPT, nullptr,
	                           &prefix_consumed) == SSLM_OK);
	CHECK(sslm_prefix_freeze(prefix) == SSLM_OK);

	// Sequence A: adopt the frozen prefix (copy-on-adopt copies shared_content's own bytes into
	// A's own block), THEN bind the adapter, THEN decode one step.
	sslm_seq seq_adopted = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_adopted) == SSLM_OK);
	CHECK(sslm_seq_adopt_prefix(seq_adopted, prefix) == SSLM_OK);
	CHECK(sslm_seq_set_adapter(seq_adopted, adapter) == SSLM_OK);
	sslm_decode_params params{};
	// T-2199 Phase D review addendum (D-SLM3797, Dan; conductor's fold-23
	// follow-on commission, item 1): struct_size is the FIRST new field,
	// caller-set, library-validated -- sslm_decode_stepImpl now rejects
	// SSLM_INVALID_ARGUMENT for any other value, checked before layer_budget.
	params.struct_size = sizeof(params);
	params.layer_budget = 1;  // any positive value -- both sides of the comparison below share
	                          // this identical params, so the comparison stays meaningful.
	int32_t token_adopted = 0;
	sslm_seq batch_adopted[1] = {seq_adopted};
	CHECK(sslm_decode_step(model, batch_adopted, 1, &params, nullptr, &token_adopted) == SSLM_OK);

	// Sequence B: prefill the IDENTICAL content directly (base-only, no adapter bound yet --
	// exactly what copy-on-adopt's own bytes are, per C3's own gate), THEN bind the SAME
	// adapter immediately (the "swap immediately post-adoption" framing), THEN decode.
	sslm_seq seq_prefilled = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_prefilled) == SSLM_OK);
	int32_t seq_consumed = 0;
	CHECK(sslm_prefill(model, seq_prefilled, shared_content, 3, 8, SSLM_SPAN_PROMPT, nullptr,
	                    &seq_consumed) == SSLM_OK);
	CHECK(sslm_seq_set_adapter(seq_prefilled, adapter) == SSLM_OK);
	int32_t token_prefilled = 0;
	sslm_seq batch_prefilled[1] = {seq_prefilled};
	CHECK(sslm_decode_step(model, batch_prefilled, 1, &params, nullptr, &token_prefilled) ==
	      SSLM_OK);

	// FEATURE ORACLE: the two decoded tokens are bit-identical -- proving copy-on-adopt-then-
	// bind and prefill-then-swap are the ONE mechanism this design's own re-derivation claims,
	// not two implementations that could silently diverge (e.g. an adapter applied only to
	// positions written AFTER adoption on one path but not the other).
	CHECK_MSG(token_adopted == token_prefilled,
	          "adopt-then-bind (%d) diverged from prefill-then-swap (%d)", token_adopted,
	          token_prefilled);

	CHECK(sslm_seq_set_adapter(seq_adopted, nullptr) == SSLM_OK);
	CHECK(sslm_seq_release(seq_adopted) == SSLM_OK);
	CHECK(sslm_seq_set_adapter(seq_prefilled, nullptr) == SSLM_OK);
	CHECK(sslm_seq_release(seq_prefilled) == SSLM_OK);
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
                                                                       sslm_adapter adapter,
                                                                       sslm_kv_pool* pool) {
	CHECK(sslm_seq_set_adapter(seq, adapter) == SSLM_OK);
	// decode_step requires a prior prefill (a virgin sequence has no current_token to continue
	// from).
	int32_t setup_prompt[1] = {0};
	int32_t setup_consumed = 0;
	CHECK(sslm_prefill(model, seq, setup_prompt, 1, 8, SSLM_SPAN_PROMPT, nullptr,
	                    &setup_consumed) == SSLM_OK);
	CHECK(EnterMidToken(model, seq));  // leaves the sequence resting mid-token.
	SeqBlobBuffer blob(model);
	CHECK(sslm_seq_save(seq, blob.bytes.data(), &blob.size) == SSLM_OK);
	sslm_seq restored = nullptr;
	CHECK(sslm_seq_restore(model, pool, blob.bytes.data(), blob.size, &restored) == SSLM_OK);
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
	// Needs a tokenizer-BEARING artifact, distinct from g_model_path (the adapter-matching base
	// M2/M3 use, which often has no tokenizer bound -- house convention, build.bat's own
	// T2139_MODEL/T2139_MODEL_TOK split).
	if (g_model_tok_path.empty()) {
		SKIP_MSG("real tokenizer-bearing artifact not supplied (--modeltok=PATH) -- product cell "
		         "not run");
		return;
	}
	std::vector<uint8_t> bytes;
	CHECK(ReadFileBytes(g_model_tok_path, &bytes));
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
	const char* utf8_prompt = "the quick brown fox";
	int32_t tokens[64];
	int32_t token_count = 64;
	CHECK(sslm_tokenize(model, utf8_prompt, tokens, &token_count) == SSLM_OK);

	SinglePool sp_unchunked, sp_chunked;
	CHECK(MakeSinglePool(model, &sp_unchunked));
	CHECK(MakeSinglePool(model, &sp_chunked));

	// Unchunked: one sslm_prefill call, chunk_budget covering the whole prompt.
	sslm_seq seq_unchunked = nullptr;
	CHECK(sslm_seq_create(model, &sp_unchunked.pool, &seq_unchunked) == SSLM_OK);
	int32_t consumed_unchunked = 0;
	CHECK(sslm_prefill(model, seq_unchunked, tokens, token_count, /*chunk_budget=*/token_count,
	                    SSLM_SPAN_PROMPT, nullptr, &consumed_unchunked) == SSLM_OK);
	CHECK(consumed_unchunked == token_count);

	// Chunked: the SAME token array, fed across multiple chunk_budget=2-bounded calls.
	sslm_seq seq_chunked = nullptr;
	CHECK(sslm_seq_create(model, &sp_chunked.pool, &seq_chunked) == SSLM_OK);
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
	CHECK(sslm_kv_pool_destroy(sp_unchunked.pool) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(sp_chunked.pool) == SSLM_OK);
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
		// T-2199 Phase D review addendum (D-SLM3797, Dan; conductor's fold-23
		// follow-on commission, item 1): struct_size is the FIRST new field,
		// caller-set, library-validated -- sslm_decode_stepImpl now rejects
		// SSLM_INVALID_ARGUMENT for any other value, checked before layer_budget.
		params.struct_size = sizeof(params);
		params.layer_budget = 1;  // any positive value -- this cell's own subject is concurrency
		                          // safety, not full-token content.
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

// REAL INVOCATION DRIVER (house pattern) -- supersedes the address-only convention. Each cell
// that needs more than one live block gets its OWN dedicated pool, sized for its own worst case,
// so no cell observes another's leftover exhaustion.
int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim8 M1-M3/P1 not run");
	} else {
		SslmModelView view;
		std::vector<uint8_t> bytes;
		std::string err;
		if (LoadRealModelView(g_model_path, &view, &bytes, &err)) {
			sslm_model model = nullptr;
			CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
			if (model) {
				const int32_t num_hidden_layers =
				    static_cast<int32_t>(view.config.num_hidden_layers);
				const size_t block_bytes = sslm_kv_block_size(model);

				// --- M1: pool(2) + workspace ---
				{
					const uint32_t bc = 2;
					AlignedBuffer pool_buf(bc * block_bytes + sslm_kv_pool_overhead_size(model, bc));
					sslm_kv_pool pool = nullptr;
					CHECK(sslm_kv_pool_create(model, pool_buf.data(), pool_buf.size(), bc, &pool) ==
					      SSLM_OK);
					const sslm_config config = ValidWorkspaceConfig(num_hidden_layers);
					AlignedBuffer ws_buf(sslm_workspace_size(model, &config));
					sslm_workspace ws = nullptr;
					CHECK(sslm_workspace_create(model, &config, ws_buf.data(), ws_buf.size(), &ws) ==
					      SSLM_OK);
					if (pool && ws) {
						TestDim8_M1_SharedWorkspaceServesDisjointSequencesNoCrossContamination(
						    model, ws, &pool);
					}
					if (ws) CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
					if (pool) CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
				}

				// --- M2/M3 need a real adapter; P1 does not ---
				if (g_adapter_path.empty()) {
					SKIP_MSG("--adapter=PATH not supplied -- dim8 M2/M3 not run");
				} else {
					std::vector<uint8_t> adapter_bytes;
					CHECK(ReadFileBytes(g_adapter_path, &adapter_bytes));
					sslm_adapter adapter_m2 = nullptr, adapter_m3 = nullptr;
					CHECK(sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), model,
					                        &adapter_m2) == SSLM_OK);
					CHECK(sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), model,
					                        &adapter_m3) == SSLM_OK);

					if (adapter_m2) {
						// M2: pool(3) -- prefix + seq_adopted + seq_prefilled.
						const uint32_t bc = 3;
						AlignedBuffer pool_buf(bc * block_bytes +
						                        sslm_kv_pool_overhead_size(model, bc));
						sslm_kv_pool pool = nullptr;
						CHECK(sslm_kv_pool_create(model, pool_buf.data(), pool_buf.size(), bc,
						                           &pool) == SSLM_OK);
						if (pool) {
							TestDim8_M2_AdapterBoundAfterPrefixAdoptMatchesFreshPrefillThenSwap(
							    model, &pool, adapter_m2);
							CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
						}
						CHECK(sslm_adapter_release(adapter_m2) == SSLM_OK);
					}

					if (adapter_m3) {
						// M3: pool(2) -- the original seq + the restored one.
						const uint32_t bc = 2;
						AlignedBuffer pool_buf(bc * block_bytes +
						                        sslm_kv_pool_overhead_size(model, bc));
						sslm_kv_pool pool = nullptr;
						CHECK(sslm_kv_pool_create(model, pool_buf.data(), pool_buf.size(), bc,
						                           &pool) == SSLM_OK);
						sslm_seq seq = nullptr;
						if (pool) CHECK(sslm_seq_create(model, &pool, &seq) == SSLM_OK);
						if (seq) {
							TestDim8_M3_SaveRestoreMidTokenWithAdapterBindingComposes(
							    model, seq, adapter_m3, &pool);
							CHECK(sslm_seq_release(seq) == SSLM_OK);
						}
						if (pool) CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
						CHECK(sslm_adapter_release(adapter_m3) == SSLM_OK);
					}
				}

				// --- P1: its own pool(2) ---
				{
					const uint32_t bc = 2;
					AlignedBuffer pool_buf(bc * block_bytes + sslm_kv_pool_overhead_size(model, bc));
					sslm_kv_pool pool = nullptr;
					CHECK(sslm_kv_pool_create(model, pool_buf.data(), pool_buf.size(), bc, &pool) ==
					      SSLM_OK);
					if (pool) {
						TestDim8_P1_LifetimeConcurrencyChurnDoesNotCorruptLongLivedSequence(model,
						                                                                     &pool);
						CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
					}
				}

				CHECK(sslm_model_unmap(model) == SSLM_OK);
			}
		} else {
			SKIP_MSG("could not load real artifact: %s", err.c_str());
		}
	}
	// M4 is self-contained (loads g_model_path itself) -- SKIPs internally if absent.
	TestDim8_M4_TokenizedPromptAcrossChunkedPrefillReassemblesUnchunked();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
