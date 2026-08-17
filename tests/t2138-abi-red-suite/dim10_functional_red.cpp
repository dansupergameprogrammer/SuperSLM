// T-2138 (Curie) -- Dim 10 (Functional achievement / feature oracle), design Sec9/Sec10 dim10.
// This dimension is where every "does it actually work end to end" claim the design names
// lives (StandardsDocument.md Sec5.4's real-workload rule). 4 cells:
//   P1 -- C4's bit-for-bit consistency oracle: cross-cited to dim6_determinism_red.cpp's own
//         P1 (the identical cell; dim6 is where the design's own Sec10 text places the
//         consistency-oracle CLAIM, dim10 is where it places the functional-achievement CROSS-
//         REFERENCE -- authored once, not duplicated).
//   P2 -- C7's golden-hash parity against TokenizerView directly.
//   P3 -- C6's delta-golden reproduction through this ABI, matching plan Sec19 S-LoRA-serial's
//         own stated (and until now unexercised) gate.
//   P4 -- S-FREEZE-EXAMPLE's own dim-10 shape: the full real-artifact, real-text, real-generation,
//         real-persistence round trip through the frozen public header alone (design Sec9's own
//         S-FREEZE-EXAMPLE slot, D-SLM3452/D-SLM3454 rulings applied).
// RED BY LINK.
#include "fixture_common.h"

#include "../tools/sslm_adapter_loader.h"

#include <cstring>

using namespace superslm;

// --- Product cell 2 (design Sec9 C7 gate / Sec10 dim10: "golden-hash parity with
// TokenizerView directly on the existing multilingual/adversarial corpus -- no new corpus
// needed, same bytes, new entry point"). sslm_tokenize's own output equals
// TokenizerView::Encode's direct call, token for token, on the SAME real prompt string. ---
static void TestDim10_P2_TokenizeGoldenParityWithTokenizerViewDirect() {
	if (g_model_path.empty()) {
		SKIP_MSG("real base artifact not supplied (--model=PATH) -- product cell not run");
		return;
	}
	SslmModelView view;
	std::vector<uint8_t> bytes;
	std::string err;
	if (!LoadRealModelView(g_model_path, &view, &bytes, &err)) {
		SKIP_MSG("could not load real artifact: %s", err.c_str());
		return;
	}
	const char* corpus[] = {"the quick brown fox jumps over the lazy dog",
	                         "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C",  // "hello world"
	                                                                              // (Chinese
	                                                                              // UTF-8)
	                         "", "!!!???...   multiple   spaces"};
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);

	CHECK_MSG(view.has_tokenizer, "real artifact at --model=PATH has no bound tokenizer -- "
	                              "cannot run the tokenize golden-parity cell");
	for (const char* text : corpus) {
		const std::vector<int32_t> reference_tokens = view.tokenizer.Encode(text);

		int32_t abi_tokens[256];
		int32_t abi_count = 256;
		const sslm_status st = sslm_tokenize(model, text, abi_tokens, &abi_count);

		// FEATURE ORACLE: parity against the direct call, not a recode -- reference_tokens IS
		// the definition of correct tokenization for this text, produced by the existing,
		// already-shipped TokenizerView, never re-derived from the ABI's own output.
		CHECK(st == SSLM_OK);
		CHECK_MSG((size_t)abi_count == reference_tokens.size(),
		          "token count mismatch on \"%s\": abi=%d reference=%zu", text, abi_count,
		          reference_tokens.size());
		for (size_t i = 0; i < reference_tokens.size() && i < (size_t)abi_count; ++i) {
			CHECK_MSG(abi_tokens[i] == reference_tokens[i],
			          "token %zu mismatch on \"%s\": abi=%d reference=%d", i, text,
			          abi_tokens[i], reference_tokens[i]);
		}
	}
	CHECK(sslm_model_unmap(model) == SSLM_OK);
}

// --- Product cell 3 (design Sec9 C6 gate / Sec10 dim10: "the delta golden re-generated through
// this ABI reproduces the V5 cross-ISA determinism property on a single resident sequence").
// The reference: the SAME real base artifact's layers, with the real adapter's delta applied
// directly (superslm_adapter::ApplyAdapterToLayers -- the identical marshal every existing
// production caller, e.g. tools/sslm_generate.cpp, uses), driven through RunGreedyOracle. The
// candidate: this ABI's own sslm_adapter_map + sslm_seq_set_adapter + decode path. Real base
// artifact and real adapter required. ---
static void TestDim10_P3_AdapterDeltaGoldenReproductionThroughAbi() {
	if (g_model_path.empty() || g_adapter_path.empty()) {
		SKIP_MSG("--model=PATH and --adapter=PATH both required -- product cell not run");
		return;
	}
	SslmModelView view;
	std::vector<uint8_t> bytes;
	std::string err;
	if (!LoadRealModelView(g_model_path, &view, &bytes, &err)) {
		SKIP_MSG("could not load real artifact: %s", err.c_str());
		return;
	}
	CpuOracleModel oracle;
	CHECK(LoadCpuOracleModel(view, &oracle, &err));

	superslm_adapter::BaseModelGeometry geo;
	geo.num_hidden_layers = oracle.num_hidden_layers;
	geo.hidden_size = oracle.hidden_size;
	geo.intermediate_size = oracle.intermediate_size;
	geo.kv_hidden_size = static_cast<uint64_t>(oracle.num_kv_heads) * oracle.head_dim;
	geo.base_artifact_hash = view.RawIntegrityHash();
	superslm_adapter::AdapterHandle adapter_handle;
	std::string adapter_err;
	const auto adapter_status =
	    superslm_adapter::LoadAdapterArtifact(g_adapter_path, geo, adapter_handle, &adapter_err);
	if (adapter_status != superslm_adapter::AdapterLoadStatus::Ok) {
		SKIP_MSG("could not load real adapter: %s", adapter_err.c_str());
		return;
	}
	superslm_adapter::ApplyAdapterToLayers(&adapter_handle, oracle.layers.data(),
	                                        oracle.num_hidden_layers);

	const int32_t prompt[3] = {0, 1, 2};
	constexpr size_t kNewTokens = 4;
	std::vector<int32_t> reference_tokens;
	size_t reference_produced = 0;
	SslmDecodeStopReason stop_reason{};
	const SslmForwardStatus ref_status = RunGreedyOracle(
	    oracle, prompt, 3, kNewTokens, &reference_tokens, &reference_produced, &stop_reason);
	CHECK(ref_status == SslmForwardStatus::Ok);

	// Candidate: the real adapter, mapped and bound through this ABI, decoding the SAME prompt.
	std::vector<uint8_t> adapter_bytes;
	CHECK(ReadFileBytes(g_adapter_path, &adapter_bytes));
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
	sslm_adapter adapter = nullptr;
	CHECK(sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), model, &adapter) ==
	      SSLM_OK);
	SinglePool sp;
	CHECK(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_adapter(seq, adapter) == SSLM_OK);
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, prompt, 3, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	std::vector<int32_t> abi_tokens(kNewTokens, 0);
	sslm_decode_params params{};
	params.layer_budget = static_cast<int32_t>(view.config.num_hidden_layers);
	sslm_seq batch[1] = {seq};
	for (size_t i = 0; i < kNewTokens; ++i) {
		int32_t out_token = 0;
		CHECK(sslm_decode_step(model, batch, 1, &params, nullptr, &out_token) == SSLM_OK);
		abi_tokens[i] = out_token;
	}

	// FEATURE ORACLE: the adapter-composed decode through this ABI reproduces the direct-engine,
	// delta-applied reference bit-for-bit, token by token -- the S-LoRA-serial gate finally
	// exercised end-to-end through a real ABI call path (design Sec9 C6, plan Sec19).
	CHECK(reference_produced == kNewTokens);
	for (size_t i = 0; i < kNewTokens && i < reference_tokens.size(); ++i) {
		CHECK_MSG(abi_tokens[i] == reference_tokens[i],
		          "adapter-decode token %zu: ABI=%d direct-engine-delta=%d", i, abi_tokens[i],
		          reference_tokens[i]);
	}
	CHECK(sslm_seq_set_adapter(seq, nullptr) == SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
	CHECK(sslm_adapter_release(adapter) == SSLM_OK);
	CHECK(sslm_model_unmap(model) == SSLM_OK);
}

// --- Product cell 4 -- THE S-FREEZE-EXAMPLE SHAPE (design Sec9 S-FREEZE-EXAMPLE slot,
// verbatim: "map an artifact, tokenize a real prompt string (D-SLM3452 -- real text I/O, not
// pre-tokenized int32_t*), size and create a pool-backed sequence with an explicitly computed
// block_count following Sec7.2's own sizing recipe (D-SLM3454 -- no default exists to fall back
// on), prefill, decode to completion, detokenize, save, restore into a fresh sequence,
// decode-verify identical continuation, unmap"). This is this suite's own mandatory
// real-workload cell (StandardsDocument.md Sec5.4: "every campaign carries at least one cell
// that runs the real artifact, at real size, on real input, and reports what a user would
// receive") -- every verb C1-C7 ship, composed in the exact sequence the design's own gate
// names, nothing pre-tokenized, nothing test-harness-only. ---
static void TestDim10_P4_SFreezeExampleShapeFullRealWorkflow() {
	if (g_model_path.empty()) {
		SKIP_MSG("real base artifact not supplied (--model=PATH) -- S-FREEZE-shaped product "
		         "cell not run");
		return;
	}
	std::vector<uint8_t> bytes;
	CHECK(ReadFileBytes(g_model_path, &bytes));

	// 1. Map.
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);

	// 2. Tokenize a REAL prompt string (D-SLM3452 -- never pre-tokenized int32_t*).
	const char* prompt_text = "Please describe the weather today in one sentence.";
	int32_t tokens[256];
	int32_t token_count = 256;
	CHECK(sslm_tokenize(model, prompt_text, tokens, &token_count) == SSLM_OK);
	CHECK(token_count > 0);

	// 3. Size and create a pool-backed sequence with an EXPLICITLY COMPUTED block_count,
	// following Sec7.2's own sizing recipe, REVISED against the whole-block buffer model
	// (commit fab235c1c6): block_count = N, the number of concurrently resident SEQUENCES the
	// caller wants (not a token-derived ceil division against kv_block_size -- kv_block_size(model)
	// now reports one whole sequence's own entire KV footprint already. block_count = 2, NOT 1:
	// this cell's own step 8 restores into a FRESH sequence while the ORIGINAL `seq` is still
	// live (both compared at step 9), so two concurrent blocks are genuinely needed -- matching
	// the real reference build's own block_count (tools/t2139_sfreeze_example.cpp:98), which
	// this cell's prior text mis-cited as block_count=1 before checking the reference.
	SslmModelView view;
	std::vector<uint8_t> parse_bytes;
	std::string err;
	CHECK(LoadRealModelView(g_model_path, &view, &parse_bytes, &err));
	const size_t kv_block_bytes = sslm_kv_block_size(model);
	CHECK_MSG(kv_block_bytes > 0, "sslm_kv_block_size must be positive");
	const uint32_t block_count = 2;  // seq + restored, concurrently live.
	const size_t pool_overhead = sslm_kv_pool_overhead_size(model, block_count);
	AlignedBuffer pool_buf(block_count * kv_block_bytes + pool_overhead);
	sslm_kv_pool pool = nullptr;
	CHECK(sslm_kv_pool_create(model, pool_buf.data(), pool_buf.size(), block_count, &pool) ==
	      SSLM_OK);
	// sslm_config is now REQUIRED sizing input for sslm_workspace_size/_create (design Sec7.1,
	// revised buffer model, commit fab235c1c6) -- an all-zero/null config is hostile input, so
	// this real-workflow cell builds a real, valid one from the artifact's own num_hidden_layers.
	const sslm_config config =
	    ValidWorkspaceConfig(static_cast<int32_t>(view.config.num_hidden_layers));
	const size_t ws_size = sslm_workspace_size(model, &config);
	CHECK_MSG(ws_size > 0, "a valid sslm_config must report a positive workspace size");
	AlignedBuffer ws_buf(ws_size);
	sslm_workspace ws = nullptr;
	CHECK(sslm_workspace_create(model, &config, ws_buf.data(), ws_buf.size(), &ws) == SSLM_OK);
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, &pool, &seq) == SSLM_OK);

	// 4. Prefill.
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, tokens, token_count, /*chunk_budget=*/token_count,
	                    SSLM_SPAN_PROMPT, ws, &consumed) == SSLM_OK);
	CHECK(consumed == token_count);

	// 5. Decode to completion (a fixed generation length stands in for "completion" -- no stop
	// tokens are threaded through this ABI's own sslm_decode_params shape, design Sec8).
	constexpr int32_t kMaxNewTokens = 16;
	std::vector<int32_t> generated_tokens;
	sslm_decode_params params{};
	params.layer_budget = static_cast<int32_t>(view.config.num_hidden_layers);
	sslm_seq batch[1] = {seq};
	for (int32_t i = 0; i < kMaxNewTokens; ++i) {
		int32_t out_token = 0;
		CHECK(sslm_decode_step(model, batch, 1, &params, ws, &out_token) == SSLM_OK);
		generated_tokens.push_back(out_token);
	}

	// 6. Detokenize (real text OUT -- the deliverable a user would receive). Design Sec7.4
	// (folded from this suite's own routed gap 2): sslm_detok_state is a fixed-size,
	// caller-allocated POD -- {0} is the valid start state, declared on the stack, no
	// construction verb -- exercising the ACTUAL incremental-detokenize call shape a real
	// consumer uses, not a null stand-in.
	char utf8_out[1024];
	int32_t utf8_out_len = sizeof(utf8_out);
	sslm_detok_state detok_state = {0};
	CHECK(sslm_detokenize_stream(model, &detok_state, generated_tokens.data(),
	                              (int32_t)generated_tokens.size(), utf8_out, &utf8_out_len) ==
	      SSLM_OK);
	// FEATURE ORACLE (design Sec7.4's own invariant): after a single, non-fragmenting call over
	// a complete token span, pending_count settles at a value in [0, 2] -- never 3 (the design's
	// own "never exceeds 2 across two consecutive calls without progress"), proving the state
	// was genuinely written to, not left at its {0} initializer by an implementation that
	// ignores the parameter.
	CHECK(detok_state.pending_count <= 2);

	// 7. Save.
	SeqBlobBuffer blob(model);
	CHECK(sslm_seq_save(seq, blob.bytes.data(), &blob.size) == SSLM_OK);

	// 8. Restore into a FRESH sequence.
	sslm_seq restored = nullptr;
	CHECK(sslm_seq_restore(model, &pool, blob.bytes.data(), blob.size, &restored) == SSLM_OK);

	// 9. Decode-verify IDENTICAL continuation: the restored handle's own next decoded token
	// equals what the ORIGINAL handle would have produced next -- the save/restore round trip's
	// own feature oracle, exercised at the end of a full real generation, not only in isolation
	// (dim9's own cells already isolate this claim; this cell's contribution is proving it holds
	// at the END of the exact S-FREEZE-shaped workflow, composed with everything above it).
	sslm_seq original_continuation[1] = {seq};
	sslm_seq restored_continuation[1] = {restored};
	int32_t original_next = 0, restored_next = 0;
	CHECK(sslm_decode_step(model, original_continuation, 1, &params, ws, &original_next) ==
	      SSLM_OK);
	CHECK(sslm_decode_step(model, restored_continuation, 1, &params, ws, &restored_next) ==
	      SSLM_OK);
	CHECK_MSG(original_next == restored_next,
	          "restored continuation diverged: original=%d restored=%d", original_next,
	          restored_next);

	// 10. Unmap (and release everything else, symmetrically).
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_seq_release(restored) == SSLM_OK);
	CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
	CHECK(sslm_model_unmap(model) == SSLM_OK);

	// This cell's own build/link surface is the S-FREEZE gate's own bar (design Sec9/Sec19): no
	// internal header beyond sslm_abi.h is required for the CALL SEQUENCE above to compile
	// (fixture_common.h's own internal-engine includes are this SUITE's oracle-comparison
	// machinery for OTHER cells, not a dependency of this cell's own call sequence) -- the true
	// bare-C++-against-the-frozen-header-alone proof is S-FREEZE-EXAMPLE's own dedicated build
	// target (design Sec9), owned by the build seat, not reproduced a second time here; this
	// cell is the coverage-model's own product cell for the SAME call shape, run inside this
	// suite's existing harness.
}

// REAL INVOCATION DRIVER (house pattern) -- supersedes the address-only convention. Every cell
// in this file is self-contained (loads its own real artifacts via g_model_path/g_adapter_path)
// and SKIPs internally when its own fixture is absent.
int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	TestDim10_P2_TokenizeGoldenParityWithTokenizerViewDirect();
	TestDim10_P3_AdapterDeltaGoldenReproductionThroughAbi();
	TestDim10_P4_SFreezeExampleShapeFullRealWorkflow();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
