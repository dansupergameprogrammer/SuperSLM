// T-2138 (Curie) -- Dim 4 (Shape and platform matrices), design Sec10 dim4. 4 cells: 4
// mechanism (M4 added -- coverage gap routed from Claude/Poirot/
// 2c18dab-t2139-abi-build-review.md Sec6.3 N2, see M4's own comment below). The full
// toolchain/ISA matrix (design Sec13.3, clang-cl/Linux clang-GCC/macOS-ARM)
// is a CI-level obligation, not a single cell -- this suite's own build_link_red.bat runs on
// x64 MSVC only (matching the Gate A/C commissioning residual this design's own Sec9 already
// names, "re-run as part of every real slot's CI build across that matrix, not by the
// design-time commissioning alone"); the multi-toolchain build itself is the build seat's CI
// wiring, cross-cited here as a standing obligation this dimension names, not built by this
// suite. RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec7.1/Sec9 C1 gate: "every sizing function's returned byte
// count is sufficient... a fuzzed buffer of exactly the reported size never overflows"):
// workspace buffer-size boundary swept exact/one-byte-short/oversized. The exact-size case is
// this cell's own new claim (M1/M2 of dim2 already cover one-byte-short and misaligned as
// REJECTIONS); this cell instead confirms exact-size and oversized are BOTH accepted -- the
// positive half of the same boundary. Uses a REAL, valid sslm_config (design Sec7.1, revised
// buffer model, commit fab235c1c6) -- an all-zero/null config is hostile input now. ---
static void TestDim4_M1_WorkspaceBufferSizeBoundaryExactAndOversizedAccepted(
    sslm_model model, int32_t num_hidden_layers) {
	const sslm_config config = ValidWorkspaceConfig(num_hidden_layers);
	const size_t exact = sslm_workspace_size(model, &config);
	CHECK_MSG(exact > 0, "a valid sslm_config must report a positive workspace size");
	{
		AlignedBuffer buf(exact);
		sslm_workspace ws = nullptr;
		CHECK(sslm_workspace_create(model, &config, buf.data(), buf.size(), &ws) == SSLM_OK);
		CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
	}
	{
		AlignedBuffer buf(exact + 4096);  // oversized -- must still be accepted, never
		                                   // rejected for being "too large".
		sslm_workspace ws = nullptr;
		CHECK(sslm_workspace_create(model, &config, buf.data(), buf.size(), &ws) == SSLM_OK);
		CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
	}
}

// --- Mechanism cell 2 (design Sec7.2/Sec10 dim4): kv pool block_count swept 1 / typical / a
// large-but-legitimate CONCURRENT-SEQUENCE count, each correctly sized against
// sslm_kv_pool_overhead_size's own reported requirement for THAT block_count (the overhead is
// O(block_count), design Sec7.2, so a buffer sized for one block_count is not automatically
// sufficient for another). Sweep values revised twice against the whole-block buffer model
// (commit fab235c1c6): `block_count` is now a count of CONCURRENT SEQUENCES, each backed by one
// whole sequence's own KV footprint -- MEASURED real (Claude/Poirot/
// 2c18dab-t2139-abi-build-review.md C2): 192 MB/sequence (0.5B tier), 448 MB/sequence (1.5B
// tier), not "tens of MB" as this cell's own prior estimate assumed. 1/2/4 is the sweep that
// stays inside a few GB even on the 1.5B tier -- 1/8/64 (the design's own N=50 many-agent-cohort
// framing, taken literally) would allocate 12-28 GB for the top of the sweep alone and make this
// cell unrunnable once built rather than merely slow; the many-agent cohort SIZE claim itself is
// unaffected, since this cell's own subject is "the overhead formula is correct at more than one
// block_count," not "the design's own N=50 example is reproduced literally." ---
static void TestDim4_M2_KvPoolBlockCountSweptOneTypicalLarge(sslm_model model) {
	const uint32_t sweep[3] = {1, 2, 4};
	for (uint32_t block_count : sweep) {
		const size_t block_bytes = sslm_kv_block_size(model);
		const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
		AlignedBuffer buf(block_count * block_bytes + overhead);
		sslm_kv_pool pool = nullptr;
		CHECK_MSG(sslm_kv_pool_create(model, buf.data(), buf.size(), block_count, &pool) ==
		              SSLM_OK,
		          "block_count=%u", block_count);
		CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
	}
}

// --- Mechanism cell 4 (design Sec10 dim4, coverage gap routed from the T-2139 confirmation
// pass: Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3 N2). N2 found the workspace's
// own carved sub-region offsets (`wide_logits_offset`/`rms_wide_offset`, both `int64_t*`)
// accumulate by raw byte size with no alignment rounding: the residue is
// `4*max_chunk_budget mod 8` -- zero when `max_chunk_budget` is even, FOUR when it is odd,
// putting both pointers on a 4-mod-8 address (UB, confirmed arithmetically against the real
// carving formula, N2's own citation). N2's own text: "The suite cannot see it, by
// construction... fixture_common.h's own ValidWorkspaceConfig sets max_chunk_budget=256 and
// t2139_c2_smoke uses 64 -- both even, both aligned. The only caller in the tree with an odd
// budget is the one nothing asserts against" (the real reference consumer,
// tools/t2139_sfreeze_example.cpp, whose own prompt tokenizes to 5). This cell closes exactly
// that gap: an ODD max_chunk_budget, driven through the FULL real path (construct, prefill,
// decode -- not just construction), asserting clean execution end to end. This is the cell
// that would have caught the regression class N2 names, on any platform/toolchain where
// misaligned int64_t access actually faults (the finding's own text: "on x64 this works...
// it is still UB" -- x64 is known to tolerate this specific class silently, so a clean-
// execution assertion alone does not discriminate on x64; the cell's own value is closing the
// coverage gap for every OTHER target design Sec13.3 names, and for x64 the day this file
// builds under an alignment-checking sanitizer, exactly as design Sec9's own C1 gate already
// requires ("never overflows under ASan/UBSan across the full artifact-config matrix")). ---
static void TestDim4_M4_OddChunkBudgetWorkspaceCleanFullPathExecution(sslm_model model,
                                                                       sslm_seq seq,
                                                                       int32_t num_hidden_layers) {
	sslm_config odd_config = ValidWorkspaceConfig(num_hidden_layers);
	CHECK_MSG(odd_config.max_chunk_budget % 2 == 0,
	          "ValidWorkspaceConfig's own default (%d) was assumed even -- if this fails, the "
	          "shared default itself changed and this cell's own odd variant below needs "
	          "re-deriving from it",
	          odd_config.max_chunk_budget);
	odd_config.max_chunk_budget -= 1;  // 255 -- odd, still >= 1, still <= the shared default.
	CHECK_MSG(odd_config.max_chunk_budget % 2 == 1, "odd_config.max_chunk_budget (%d) is not odd",
	          odd_config.max_chunk_budget);

	const size_t required = sslm_workspace_size(model, &odd_config);
	CHECK_MSG(required > 0, "a valid odd-max_chunk_budget sslm_config must report a positive "
	                        "workspace size");
	AlignedBuffer buf(required);
	sslm_workspace ws = nullptr;
	CHECK(sslm_workspace_create(model, &odd_config, buf.data(), buf.size(), &ws) == SSLM_OK);
	if (!ws) return;

	// The FULL real path, not merely construction: prefill at a real, odd chunk_budget (the
	// value N2's own carving formula is keyed on, not just the sslm_config field in isolation),
	// then decode -- exercising the SAME wide_logits/rms_wide regions the carving formula
	// places at the misaligned offset.
	int32_t odd_chunk_budget = odd_config.max_chunk_budget;
	int32_t tokens[5] = {0, 1, 2, 3, 4};  // 5 tokens: the real reference consumer's own
	                                       // documented prompt length (N2's own citation,
	                                       // tools/t2139_sfreeze_example.cpp), reproduced here
	                                       // as this cell's own real workload rather than an
	                                       // arbitrary count.
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, tokens, 5, odd_chunk_budget, SSLM_SPAN_PROMPT, ws,
	                    &consumed) == SSLM_OK);
	CHECK(consumed == 5);

	sslm_decode_params params{};
	params.layer_budget = num_hidden_layers;
	int32_t out_token = 0;
	sslm_seq batch[1] = {seq};
	CHECK(sslm_decode_step(model, batch, 1, &params, ws, &out_token) == SSLM_OK);

	CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
}

// --- Mechanism cell 3 (design Sec10 dim4, this ABI's own new obligation: "this layer itself
// has never been compiled and hashed" -- the session layer's own build is proven to reach a
// real artifact's own true kv_block_size/workspace_size, not a fixed constant that happens to
// work on one artifact's dimensions). Two shape-distinct real artifacts (base tier and a
// same-tier-different-content variant, or two genuinely different tiers if both are supplied)
// must report DIFFERENT sizing outputs when their configs genuinely differ, proving the sizing
// functions are real functions of the artifact, not a memoized/hardcoded return. ---
static void TestDim4_M3_SizingFunctionsVaryWithRealArtifactShape() {
	if (g_model_path.empty() || g_foreign_model_path.empty()) {
		SKIP_MSG("--model=PATH and --foreignmodel=PATH (a genuinely different-shape artifact) "
		         "both required -- product cell not run");
		return;
	}
	// Loaded twice, deliberately: LoadRealModelView (the internal SslmModelView parse) supplies
	// each artifact's own num_hidden_layers -- the one field a valid sslm_config needs and that
	// no sslm_* ABI verb exposes from an opaque sslm_model handle directly (design Sec7.1, revised
	// buffer model, commit fab235c1c6: all-zero/null config is hostile input, so this cell cannot
	// pass nullptr and still expect a real, comparable size back). sslm_model_map (the ABI call
	// actually under test) is separate and RED BY LINK on its own.
	SslmModelView view_a, view_b;
	std::vector<uint8_t> parse_bytes_a, parse_bytes_b;
	std::string err;
	CHECK(LoadRealModelView(g_model_path, &view_a, &parse_bytes_a, &err));
	CHECK(LoadRealModelView(g_foreign_model_path, &view_b, &parse_bytes_b, &err));

	std::vector<uint8_t> a_bytes, b_bytes;
	CHECK(ReadFileBytes(g_model_path, &a_bytes));
	CHECK(ReadFileBytes(g_foreign_model_path, &b_bytes));
	sslm_model model_a = nullptr, model_b = nullptr;
	CHECK(sslm_model_map(a_bytes.data(), a_bytes.size(), &model_a) == SSLM_OK);
	CHECK(sslm_model_map(b_bytes.data(), b_bytes.size(), &model_b) == SSLM_OK);
	const sslm_config config_a =
	    ValidWorkspaceConfig(static_cast<int32_t>(view_a.config.num_hidden_layers));
	const sslm_config config_b =
	    ValidWorkspaceConfig(static_cast<int32_t>(view_b.config.num_hidden_layers));
	const size_t ws_a = sslm_workspace_size(model_a, &config_a);
	const size_t ws_b = sslm_workspace_size(model_b, &config_b);
	const size_t kv_a = sslm_kv_block_size(model_a);
	const size_t kv_b = sslm_kv_block_size(model_b);
	// FEATURE ORACLE: at least one of the two sizing outputs differs between two genuinely
	// different-shaped artifacts -- a suite that always returns the same numbers regardless of
	// input artifact would pass every OTHER sizing cell in this file while this one alone
	// catches it.
	CHECK_MSG(ws_a != ws_b || kv_a != kv_b,
	          "sizing functions returned identical outputs (ws=%zu, kv=%zu) for two "
	          "different-shaped real artifacts",
	          ws_a, kv_a);
	CHECK(sslm_model_unmap(model_a) == SSLM_OK);
	CHECK(sslm_model_unmap(model_b) == SSLM_OK);
}

// REAL INVOCATION DRIVER (house pattern) -- supersedes the address-only convention.
int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim4 M1/M2/M4 not run");
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
				TestDim4_M1_WorkspaceBufferSizeBoundaryExactAndOversizedAccepted(
				    model, num_hidden_layers);
				TestDim4_M2_KvPoolBlockCountSweptOneTypicalLarge(model);

				SinglePool sp;
				sslm_seq seq_m4 = nullptr;
				if (MakeSinglePool(model, &sp))
					CHECK(sslm_seq_create(model, &sp.pool, &seq_m4) == SSLM_OK);
				if (seq_m4) {
					TestDim4_M4_OddChunkBudgetWorkspaceCleanFullPathExecution(model, seq_m4,
					                                                          num_hidden_layers);
					CHECK(sslm_seq_release(seq_m4) == SSLM_OK);
				}
				if (sp.pool) CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);

				CHECK(sslm_model_unmap(model) == SSLM_OK);
			}
		} else {
			SKIP_MSG("could not load real artifact: %s", err.c_str());
		}
	}
	// M3 is self-contained (loads both --model=PATH and --foreignmodel=PATH itself).
	TestDim4_M3_SizingFunctionsVaryWithRealArtifactShape();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
