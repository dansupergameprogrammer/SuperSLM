// T-2138 (Curie) -- Dim 2 (Trust boundaries and hostile inputs), design Sec10 dim2. 12 cells.
// RED BY LINK.
//
// ROUTED MODEL GAP CLOSED (design commit 41b72091c2, Sec7.1, folded from this suite's own
// first-pass finding, Claude/Curie/t2138-abi-red-suite-2026-08-16.md Sec6 Gap 1): sslm_config's
// field layout is now defined (sslm_abi.h) -- max_batch/max_chunk_budget/max_layer_budget/
// reserved, all-zero is hostile input. The hostile-sslm_config cell design Sec10 dim2's own
// text names is authored below as FIVE independent sub-cells (M8a-M8e), one per field, each
// mutating exactly ONE field hostile against an otherwise-valid baseline -- per field, not one
// bundled cell, per the design's own "five sub-cells, not one" accounting. Every sub-cell
// asserts BOTH the valid baseline (sslm_workspace_size > 0, sslm_workspace_create == SSLM_OK)
// AND the hostile mutation (sslm_workspace_size == 0, sslm_workspace_create ==
// SSLM_INVALID_ARGUMENT) -- an implementation that always returns 0/always rejects fails the
// baseline half; an implementation that never checks sslm_config at all (always accepts) fails
// the hostile half. Neither half alone would be a cell that can fail against both defect
// classes; both together are.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec7.1/Sec9 C1 gate): sslm_workspace_create with a caller buffer
// smaller than sslm_workspace_size's own reported requirement rejects SSLM_BUFFER_TOO_SMALL.
// Uses a REAL, valid sslm_config (design Sec7.1, revised buffer model, commit fab235c1c6) --
// an all-zero/null config is itself hostile input now, so this cell's own "otherwise valid
// call, only the buffer is wrong" isolation needs a real config or it would be indistinguishable
// from the hostile-config cells below (M8a-M8e). ---
static void TestDim2_M1_WorkspaceCreateTooSmallBufferRejected(sslm_model model,
                                                               int32_t num_hidden_layers) {
	const sslm_config config = ValidWorkspaceConfig(num_hidden_layers);
	const size_t required = sslm_workspace_size(model, &config);
	CHECK_MSG(required > 0, "a valid sslm_config must report a positive workspace size");
	std::vector<uint8_t> buf(required > 0 ? required - 1 : 0);
	sslm_workspace ws = nullptr;
	CHECK(sslm_workspace_create(model, &config, buf.data(), buf.size(), &ws) ==
	      SSLM_BUFFER_TOO_SMALL);
	CHECK(ws == nullptr);
}

// --- Mechanism cell 2 (design Sec7.1/Sec9 C1 gate): sslm_workspace_create with a
// misaligned caller buffer address rejects SSLM_MISALIGNED_BUFFER, never a later kernel fault.
// Real config, same reason as M1. ---
static void TestDim2_M2_WorkspaceCreateMisalignedBufferRejected(sslm_model model,
                                                                 int32_t num_hidden_layers) {
	const sslm_config config = ValidWorkspaceConfig(num_hidden_layers);
	const size_t required = sslm_workspace_size(model, &config);
	// Over-allocate by a cache line and hand back an address offset by exactly one byte from a
	// naturally-aligned base -- guaranteed misaligned against any alignment requirement the
	// artifact could plausibly declare (1-, 2-, 4-, 8-, 16-, 32-, 64-byte).
	std::vector<uint8_t> buf(required + 64);
	uint8_t* misaligned = buf.data() + 1;
	sslm_workspace ws = nullptr;
	CHECK(sslm_workspace_create(model, &config, misaligned, required + 63, &ws) ==
	      SSLM_MISALIGNED_BUFFER);
	CHECK(ws == nullptr);
}

// --- Mechanism cell 3 (design Sec7.2/Sec9 C1 gate): sslm_kv_pool_create with a caller buffer
// smaller than block_count*sslm_kv_block_size(model) plus overhead rejects SSLM_BUFFER_TOO_SMALL.
// `block_count` here is a COUNT OF CONCURRENT SEQUENCES the pool backs, and
// `sslm_kv_block_size(model)` is one whole sequence's entire KV footprint (design Sec7.2,
// revised buffer model, commit fab235c1c6) -- the formula itself (block_count * block_size +
// overhead) is unchanged, only the unit block_size measures changed. ---
static void TestDim2_M3_KvPoolCreateTooSmallBufferRejected(sslm_model model) {
	const uint32_t block_count = 4;
	const size_t required =
	    block_count * sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, block_count);
	std::vector<uint8_t> buf(required > 0 ? required - 1 : 0);
	sslm_kv_pool pool = nullptr;
	CHECK(sslm_kv_pool_create(model, buf.data(), buf.size(), block_count, &pool) ==
	      SSLM_BUFFER_TOO_SMALL);
	CHECK(pool == nullptr);
}

// --- Mechanism cell 4 (design Sec10 dim2, Mendeleev 4.2 -- the two invented sizing verbs' own
// arithmetic proven overflow-safe, distinct from and upstream of M3's buffer-vs-reported-size
// cell). `block_count` is engineered near SIZE_MAX/sslm_kv_block_size(model) so a naive
// `block_count * kv_block_size` multiply wraps. Asserts EITHER the call rejects
// SSLM_INVALID_ARGUMENT before trusting the wrapped product, OR the reported overhead/required
// size is never smaller than the true (unwrapped, widened) requirement -- a checked/saturating
// computation is an acceptable alternative to outright rejection, silent wraparound is not. ---
static void TestDim2_M4_KvPoolOverheadArithmeticOverflowSafe(sslm_model model) {
	const size_t block_bytes = sslm_kv_block_size(model);
	CHECK_MSG(block_bytes > 0, "sslm_kv_block_size must be positive for this cell to be "
	                           "meaningful");
	if (block_bytes == 0) return;
	// block_count chosen so block_count * block_bytes overflows a 32-bit computation and comes
	// close to overflowing a 64-bit one too, depending on block_bytes -- the adversarial corner
	// this cell exists to reach, per Mendeleev 4.2's own naming ("a block_count near
	// SIZE_MAX / kv_block_size(model)").
	const uint32_t hostile_block_count = 0xFFFFFFFFu;
	const size_t overhead = sslm_kv_pool_overhead_size(model, hostile_block_count);
	// The naive product this design's own construction check must not trust blindly:
	const unsigned long long naive_product =
	    (unsigned long long)hostile_block_count * (unsigned long long)block_bytes;
	const bool naive_product_wrapped =
	    (naive_product / block_bytes) != (unsigned long long)hostile_block_count;
	std::vector<uint8_t> buf(1);  // deliberately tiny -- any accept here on a wrapped-small
	                              // reported size is the exact defect this cell exists to catch.
	sslm_kv_pool pool = nullptr;
	const sslm_status st =
	    sslm_kv_pool_create(model, buf.data(), buf.size(), hostile_block_count, &pool);
	if (naive_product_wrapped) {
		// Either the call refuses to trust the wrapped arithmetic outright...
		const bool rejected_before_trusting = (st == SSLM_INVALID_ARGUMENT);
		// ...or overhead/required-size reporting is proven never smaller than the TRUE
		// (widened, unwrapped) requirement, so a tiny buf still correctly BUFFER_TOO_SMALLs
		// rather than being wrongly accepted against an understated size.
		const bool reported_size_not_understated =
		    (st == SSLM_BUFFER_TOO_SMALL) && (overhead > 0 || naive_product > buf.size());
		CHECK_MSG(rejected_before_trusting || reported_size_not_understated,
		          "hostile block_count=0xFFFFFFFF accepted a 1-byte buffer against wrapped "
		          "arithmetic (status=%d, overhead=%zu)",
		          (int)st, overhead);
	}
	CHECK(st != SSLM_OK);  // never a silent accept of a 1-byte buffer at this block_count.
}

// --- Mechanism cell 5 (design Sec3/Sec6/Sec9 C2 gate): sslm_model_map on structurally hostile
// bytes reproduces the internal SslmModelStatus rejection through SSLM_ARTIFACT_REJECTED,
// mirroring the already-shipped GPU ABI's ArtifactRejected-wraps-SslmStatus pattern
// (design Sec6, model.h:177). ---
static void TestDim2_M5_ModelMapHostileArtifactRejected() {
	// A deliberately truncated, non-magic-prefixed byte buffer -- not a legitimate .sslm
	// container by construction, distinct from a merely-empty buffer (SSLM_INVALID_ARGUMENT's
	// own null/zero-size class, exercised in dim5).
	const uint8_t hostile_bytes[16] = {'X', 'X', 'X', 'X', 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF,
	                                    0,   0,   0,   0};
	sslm_model model = nullptr;
	const sslm_status st = sslm_model_map(hostile_bytes, sizeof(hostile_bytes), &model);
	CHECK(st == SSLM_ARTIFACT_REJECTED);
	CHECK(model == nullptr);
}

// --- Mechanism cell 6 (design Sec6/Sec9 C6 gate): sslm_adapter_map against a base model the
// adapter was never compiled for rejects SSLM_ADAPTER_MODEL_MISMATCH. Real artifacts required
// (op-set/dims/base-hash mismatch is a real-content property, not constructible from bare
// bytes). ---
static void TestDim2_M6_AdapterMapBaseMismatchRejected() {
	if (g_model_path.empty() || g_foreign_model_path.empty() || g_adapter_path.empty()) {
		SKIP_MSG("--model=PATH, --foreignmodel=PATH, and --adapter=PATH all required for "
		         "the adapter/base mismatch cell -- product cell not run");
		return;
	}
	std::vector<uint8_t> base_bytes, foreign_bytes, adapter_bytes;
	CHECK(ReadFileBytes(g_model_path, &base_bytes));
	CHECK(ReadFileBytes(g_foreign_model_path, &foreign_bytes));
	CHECK(ReadFileBytes(g_adapter_path, &adapter_bytes));
	sslm_model foreign_base = nullptr;
	CHECK(sslm_model_map(foreign_bytes.data(), foreign_bytes.size(), &foreign_base) == SSLM_OK);
	sslm_adapter adapter = nullptr;
	// FEATURE ORACLE: an adapter compiled against g_model_path is rejected when mapped over the
	// DIFFERENT-shaped g_foreign_model_path base -- op-set/dims/base-hash mismatch, not a
	// bytes-level parse failure (a malformed adapter blob would instead be
	// SSLM_ARTIFACT_REJECTED, a different enumerator this cell does not exercise).
	CHECK(sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), foreign_base, &adapter) ==
	      SSLM_ADAPTER_MODEL_MISMATCH);
	CHECK(adapter == nullptr);
	CHECK(sslm_model_unmap(foreign_base) == SSLM_OK);
}

// --- Mechanism cell 7 (design Sec6/Sec10 dim2, sslm_decode_params's own hostile shape -- the
// half of the routed-gap comment's cited sentence THIS design's Sec8 does give a body for): an
// out-of-range layer_budget (negative) reaching sslm_decode_step is SSLM_INVALID_ARGUMENT. ---
static void TestDim2_M7_DecodeParamsHostileLayerBudgetRejected(sslm_model model, sslm_seq seq) {
	sslm_decode_params hostile_params{};
	// T-2199 Phase D review addendum (D-SLM3797, Dan; conductor's fold-23
	// follow-on commission, item 1): struct_size is the FIRST new field,
	// caller-set, library-validated -- sslm_decode_stepImpl now rejects
	// SSLM_INVALID_ARGUMENT for any other value, checked before layer_budget.
	hostile_params.struct_size = sizeof(hostile_params);
	hostile_params.layer_budget = -1;
	int32_t out_tokens[1] = {0};
	sslm_seq batch[1] = {seq};
	CHECK(sslm_decode_step(model, batch, 1, &hostile_params, /*ws=*/nullptr, out_tokens) ==
	      SSLM_INVALID_ARGUMENT);
}

// --- Mechanism cell 9 (D-SLM3797, T-2199 Phase D review addendum; conductor's fold-23
// follow-on commission, item 1 -- "author the ruling's own pin: wrong struct_size rejects
// loudly with the invalid-params status, correct size proceeds"). struct_size is
// sslm_decode_params' own FIRST new field, caller-set, library-validated, checked BEFORE
// layer_budget/mode/anything else (src/sslm_abi.cpp, sslm_decode_stepImpl's own comment) -- a
// version-skew caller (an old header, or a hand-rolled struct literal that has not kept up)
// gets a defined, loud rejection instead of a partial/garbage read of a struct shape the
// caller and the library disagree about. Both directions asserted, matching M7's own
// full-contrast style: a wrong struct_size rejects even with an otherwise-valid layer_budget;
// the correct size proceeds under the SAME otherwise-valid params. ---
static void TestDim2_M9_DecodeParamsStructSizeMismatchRejected(sslm_model model, sslm_seq seq) {
	int32_t out_tokens[1] = {0};
	sslm_seq batch[1] = {seq};

	// Wrong-size half: struct_size checks BEFORE layer_budget, before any per-sequence
	// readiness check (src/sslm_abi.cpp, sslm_decode_stepImpl's own comment: "struct_size is
	// validated FIRST, ahead of ... anything else") -- so an unprefilled `seq` (this cell's own
	// fixture state, matching M7's identical precondition) does not confound this half.
	sslm_decode_params wrong_size_params{};
	wrong_size_params.struct_size = sizeof(sslm_decode_params) - 1;  // deliberately wrong
	wrong_size_params.layer_budget = 1;  // otherwise valid -- isolates struct_size as the cause
	const sslm_status wrong_status =
	    sslm_decode_step_v2(model, batch, 1, &wrong_size_params, /*ws=*/nullptr, out_tokens);
	CHECK_MSG(wrong_status == SSLM_INVALID_ARGUMENT,
	          "struct_size=sizeof(sslm_decode_params)-1 (an otherwise-valid layer_budget=1) must "
	          "reject SSLM_INVALID_ARGUMENT -- got status=%d", (int)wrong_status);

	// Correct-size half: MUST actually reach the per-sequence readiness check and succeed, or
	// this half is not discriminating (an unprefilled seq would ALSO reject
	// SSLM_INVALID_ARGUMENT -- indistinguishable from the struct_size defect this cell exists
	// to catch). Prefills `seq` first, matching this suite's own established prompt fixture
	// (prompt=[0,1,2,3], SSLM_SPAN_PROMPT).
	const int32_t prompt[4] = {0, 1, 2, 3};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, prompt, 4, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	sslm_decode_params correct_size_params{};
	correct_size_params.struct_size = sizeof(correct_size_params);
	correct_size_params.layer_budget = 1;
	const sslm_status correct_status =
	    sslm_decode_step_v2(model, batch, 1, &correct_size_params, /*ws=*/nullptr, out_tokens);
	CHECK_MSG(correct_status != SSLM_INVALID_ARGUMENT,
	          "struct_size=sizeof(sslm_decode_params) with the SAME otherwise-valid layer_budget, "
	          "against a PREFILLED sequence, must NOT be rejected for its struct_size -- got "
	          "status=%d (this is the discriminating half: a build that rejects EVERY call "
	          "regardless of struct_size would still pass the wrong-size half above, which is "
	          "why both directions are asserted together, and why this half prefills first)",
	          (int)correct_status);
}

// --- Cells 8a-8e (design Sec7.1/Sec10 dim2, five independent sub-cells -- "each driven
// independently against sslm_workspace_size (== 0) and sslm_workspace_create
// (SSLM_INVALID_ARGUMENT)"). Each sub-cell mutates exactly one field of the shared
// fixture_common.h ValidWorkspaceConfig() baseline and still asserts the baseline half
// independently -- isolating the mutated field, not testing "everything hostile at once". ---
static sslm_config ValidBaselineConfig(int32_t num_hidden_layers) {
	return ValidWorkspaceConfig(num_hidden_layers);
}

// Asserts the FULL contrast for one config: baseline (valid) succeeds on both functions, then
// the SAME config with exactly one field mutated hostile fails on both -- the shape every
// M8x sub-cell below shares, factored once so each sub-cell's own body states only which field
// it mutates and to what.
static void AssertConfigHostileContrast(sslm_model model, const sslm_config& baseline,
                                         const sslm_config& hostile, const char* field_name) {
	const size_t valid_size = sslm_workspace_size(model, &baseline);
	CHECK_MSG(valid_size > 0, "baseline sslm_config (mutating %s) reported size 0 -- baseline "
	                          "itself must be accepted",
	          field_name);
	AlignedBuffer valid_buf(valid_size);
	sslm_workspace valid_ws = nullptr;
	CHECK_MSG(sslm_workspace_create(model, &baseline, valid_buf.data(), valid_buf.size(),
	                                 &valid_ws) == SSLM_OK,
	          "baseline sslm_config (mutating %s) was rejected -- baseline itself must be "
	          "accepted",
	          field_name);
	if (valid_ws) CHECK(sslm_workspace_destroy(valid_ws) == SSLM_OK);

	const size_t hostile_size = sslm_workspace_size(model, &hostile);
	CHECK_MSG(hostile_size == 0, "hostile sslm_config.%s was NOT signaled by sslm_workspace_size "
	                             "(returned %zu, expected 0)",
	          field_name, hostile_size);
	AlignedBuffer hostile_buf(valid_size > 0 ? valid_size : 4096);
	sslm_workspace hostile_ws = nullptr;
	const sslm_status hostile_status = sslm_workspace_create(
	    model, &hostile, hostile_buf.data(), hostile_buf.size(), &hostile_ws);
	CHECK_MSG(hostile_status == SSLM_INVALID_ARGUMENT,
	          "hostile sslm_config.%s was not rejected SSLM_INVALID_ARGUMENT (status=%d)",
	          field_name, (int)hostile_status);
	CHECK(hostile_ws == nullptr);
}

static void TestDim2_M8a_ConfigMaxBatchZeroRejected(sslm_model model, int32_t num_hidden_layers) {
	sslm_config baseline = ValidBaselineConfig(num_hidden_layers);
	sslm_config hostile = baseline;
	hostile.max_batch = 0;
	AssertConfigHostileContrast(model, baseline, hostile, "max_batch(=0)");
}

static void TestDim2_M8b_ConfigMaxBatchNegativeRejected(sslm_model model,
                                                          int32_t num_hidden_layers) {
	sslm_config baseline = ValidBaselineConfig(num_hidden_layers);
	sslm_config hostile = baseline;
	hostile.max_batch = -1;
	AssertConfigHostileContrast(model, baseline, hostile, "max_batch(=-1)");
}

static void TestDim2_M8c_ConfigMaxChunkBudgetZeroRejected(sslm_model model,
                                                            int32_t num_hidden_layers) {
	sslm_config baseline = ValidBaselineConfig(num_hidden_layers);
	sslm_config hostile = baseline;
	hostile.max_chunk_budget = 0;
	AssertConfigHostileContrast(model, baseline, hostile, "max_chunk_budget(=0)");
}

static void TestDim2_M8d_ConfigMaxLayerBudgetZeroAndOverCapRejected(sslm_model model,
                                                                     int32_t num_hidden_layers) {
	sslm_config baseline = ValidBaselineConfig(num_hidden_layers);
	// Two sub-values for the SAME field, both hostile for a different reason (design Sec7.1:
	// "max_layer_budget at 0 and at num_hidden_layers + 1") -- both checked here since they
	// share one field's own contrast shape.
	sslm_config hostile_zero = baseline;
	hostile_zero.max_layer_budget = 0;
	AssertConfigHostileContrast(model, baseline, hostile_zero, "max_layer_budget(=0)");
	sslm_config hostile_over_cap = baseline;
	hostile_over_cap.max_layer_budget = num_hidden_layers + 1;
	AssertConfigHostileContrast(model, baseline, hostile_over_cap,
	                             "max_layer_budget(=num_hidden_layers+1)");
}

static void TestDim2_M8e_ConfigReservedNonzeroRejected(sslm_model model,
                                                         int32_t num_hidden_layers) {
	sslm_config baseline = ValidBaselineConfig(num_hidden_layers);
	sslm_config hostile = baseline;
	hostile.reserved = 1;
	AssertConfigHostileContrast(model, baseline, hostile, "reserved(=1)");
}

// REAL INVOCATION DRIVER (house pattern) -- supersedes the address-only convention.
int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim2 M1-M4/M7-M8e not run");
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
				TestDim2_M1_WorkspaceCreateTooSmallBufferRejected(model, num_hidden_layers);
				TestDim2_M2_WorkspaceCreateMisalignedBufferRejected(model, num_hidden_layers);
				TestDim2_M3_KvPoolCreateTooSmallBufferRejected(model);
				TestDim2_M4_KvPoolOverheadArithmeticOverflowSafe(model);
				TestDim2_M8a_ConfigMaxBatchZeroRejected(model, num_hidden_layers);
				TestDim2_M8b_ConfigMaxBatchNegativeRejected(model, num_hidden_layers);
				TestDim2_M8c_ConfigMaxChunkBudgetZeroRejected(model, num_hidden_layers);
				TestDim2_M8d_ConfigMaxLayerBudgetZeroAndOverCapRejected(model, num_hidden_layers);
				TestDim2_M8e_ConfigReservedNonzeroRejected(model, num_hidden_layers);

				SinglePool sp;
				if (MakeSinglePool(model, &sp)) {
					sslm_seq seq = nullptr;
					CHECK(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
					if (seq) {
						TestDim2_M7_DecodeParamsHostileLayerBudgetRejected(model, seq);
						// M7's own rejected call (layer_budget=-1) never touches `seq` --
						// struct_size/layer_budget both check before any per-sequence access --
						// so `seq` is still pristine, unprefilled here, M9's own required start.
						TestDim2_M9_DecodeParamsStructSizeMismatchRejected(model, seq);
						CHECK(sslm_seq_release(seq) == SSLM_OK);
					}
					CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
				} else {
					SKIP_MSG("dim2 M7 needs a real pool/sequence -- not run");
				}
				CHECK(sslm_model_unmap(model) == SSLM_OK);
			}
		} else {
			SKIP_MSG("could not load real artifact: %s", err.c_str());
		}
	}
	// M5/M6 are self-contained -- M6 SKIPs internally if its own fixtures are absent.
	TestDim2_M5_ModelMapHostileArtifactRejected();
	TestDim2_M6_AdapterMapBaseMismatchRejected();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
