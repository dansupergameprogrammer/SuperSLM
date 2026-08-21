// T-2138 (Curie) -- Dim 11 (Guard vitality), design Sec10 dim11. Every lifecycle-guard status
// in Sec6 (SSLM_MODEL_HAS_LIVE_SEQUENCES, SSLM_POOL_HAS_LIVE_HANDLES,
// SSLM_ADAPTER_HAS_LIVE_SEQUENCES) is shown ABLE TO FIRE (a mutation that violates its premise
// makes it fire) and shown to fire BEFORE the operation it guards executes -- dim5's own C2/C3/
// C4 cells already prove the unit/mutation-scope half for each guard (cross-cited here, not
// duplicated: this file's own job is naming the pre-execution-ordering half explicitly and
// authoring the PRODUCT-SCALE cell dim5's own cells do not attempt). 2 cells here: 1 mechanism
// (the ordering cross dim5's own cells assert by construction but do not narrate), 1 product
// (Mendeleev 4.3's own added obligation -- reaching a guard through ORDINARY sequencing inside
// an already-real-artifact run, not a fixture hand-built solely to trigger it). RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec10 dim11: "shown to fire BEFORE the operation it guards
// executes"). Names the ordering obligation dim5's own C2 (SSLM_MODEL_HAS_LIVE_SEQUENCES)
// already asserts by construction -- the rejected unmap must leave the model's own mapping AND
// the live sequence's own state completely untouched, proving the guard's check ran and
// rejected BEFORE any teardown step began, not that a partial teardown was rolled back after
// starting. Distinguishing "checked first" from "rolled back after" is this cell's own added
// value over dim5's own pass/fail assertion alone: the sequence's own decode continues to
// produce a well-formed result AFTER the rejected unmap, which a partial-teardown-then-rollback
// implementation could plausibly fail even while still returning the correct status code. ---
static void TestDim11_M1_ModelUnmapGuardFiresBeforeAnyTeardownStep(const void* artifact_bytes,
                                                                    size_t artifact_size) {
	sslm_model model = nullptr;
	CHECK(sslm_model_map(artifact_bytes, artifact_size, &model) == SSLM_OK);
	SinglePool sp;
	CHECK(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t prompt[2] = {0, 1};
	int32_t consumed_before = 0;
	CHECK(sslm_prefill(model, seq, prompt, 2, 8, SSLM_SPAN_PROMPT, nullptr, &consumed_before) ==
	      SSLM_OK);

	CHECK(sslm_model_unmap(model) == SSLM_MODEL_HAS_LIVE_SEQUENCES);

	// FEATURE ORACLE: the model handle is still fully live and functional after the rejected
	// unmap -- a FRESH sequence can still be created against it, and the original sequence can
	// still decode -- proving no teardown step (partial or otherwise) began before the guard's
	// own check rejected the call. A second, DEDICATED pool for the fresh sequence, since the
	// first pool (block_count=1) is already fully drawn by `seq`.
	SinglePool sp2;
	CHECK(MakeSinglePool(model, &sp2));
	sslm_seq seq2 = nullptr;
	CHECK(sslm_seq_create(model, &sp2.pool, &seq2) == SSLM_OK);
	int32_t out_token = 0;
	sslm_decode_params params{};
	// T-2199 Phase D review addendum (D-SLM3797, Dan; conductor's fold-23
	// follow-on commission, item 1): struct_size is the FIRST new field,
	// caller-set, library-validated -- sslm_decode_stepImpl now rejects
	// SSLM_INVALID_ARGUMENT for any other value, checked before layer_budget.
	params.struct_size = sizeof(params);
	params.layer_budget = 1;  // any positive value in [1, num_hidden_layers] is a valid domain
	                          // point here -- this cell's own subject is "the sequence is still
	                          // usable," not a full-token comparison.
	sslm_seq batch[1] = {seq};
	CHECK(sslm_decode_step(model, batch, 1, &params, nullptr, &out_token) == SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_seq_release(seq2) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(sp2.pool) == SSLM_OK);
	CHECK(sslm_model_unmap(model) == SSLM_OK);
}

// --- Product cell 1 (design Sec9 C6 gate / Sec10 dim11, Mendeleev 4.3's own added obligation:
// "within an already-planned real-artifact run..., reach at least one lifecycle guard through
// ORDINARY API sequencing..., arising from the test's normal call sequence rather than a
// fixture hand-built solely to trigger the guard in isolation"). Real base artifact and real
// adapter required -- the guard reached here is SSLM_ADAPTER_HAS_LIVE_SEQUENCES, arising from
// the natural sequence of "map base, map adapter, bind adapter to a sequence, attempt to
// release the adapter before unbinding" a real S-LoRA-serial consumer would plausibly write
// (not a fixture built ONLY to prove the guard fires). ---
static void TestDim11_P1_AdapterGuardReachedThroughOrdinaryRealArtifactSequencing() {
	if (g_model_path.empty() || g_adapter_path.empty()) {
		SKIP_MSG("--model=PATH and --adapter=PATH both required -- product cell not run");
		return;
	}
	std::vector<uint8_t> model_bytes, adapter_bytes;
	CHECK(ReadFileBytes(g_model_path, &model_bytes));
	CHECK(ReadFileBytes(g_adapter_path, &adapter_bytes));
	sslm_model model = nullptr;
	CHECK(sslm_model_map(model_bytes.data(), model_bytes.size(), &model) == SSLM_OK);
	sslm_adapter adapter = nullptr;
	CHECK(sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), model, &adapter) ==
	      SSLM_OK);
	SinglePool sp;
	CHECK(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	// The ordinary sequence a real consumer writes: bind, generate a little, THEN (forgetting
	// to unbind first, an ordinary caller mistake, not a hand-built violation) try to release
	// the adapter.
	CHECK(sslm_seq_set_adapter(seq, adapter) == SSLM_OK);
	int32_t prompt[2] = {0, 1};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, prompt, 2, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
	      SSLM_OK);
	// FEATURE ORACLE: the guard fires here, reached through this ORDINARY sequence (map, bind,
	// generate, release-too-early), not a fixture built solely to isolate the guard.
	CHECK(sslm_adapter_release(adapter) == SSLM_ADAPTER_HAS_LIVE_SEQUENCES);
	// The release-build status is a DEFINED rejection, not a crash -- the sequence and adapter
	// both remain usable afterward.
	CHECK(sslm_seq_set_adapter(seq, nullptr) == SSLM_OK);
	CHECK(sslm_adapter_release(adapter) == SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
	CHECK(sslm_model_unmap(model) == SSLM_OK);
}

// REAL INVOCATION DRIVER (house pattern) -- supersedes the address-only convention. Both cells
// are self-contained (M1 takes raw bytes, P1 loads its own artifacts) -- SKIPs internally when
// their own fixtures are absent.
int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim11 M1 not run");
	} else {
		std::vector<uint8_t> bytes;
		CHECK(ReadFileBytes(g_model_path, &bytes));
		TestDim11_M1_ModelUnmapGuardFiresBeforeAnyTeardownStep(bytes.data(), bytes.size());
	}
	TestDim11_P1_AdapterGuardReachedThroughOrdinaryRealArtifactSequencing();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
