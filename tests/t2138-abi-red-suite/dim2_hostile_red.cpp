// T-2138 (Curie) -- Dim 2 (Trust boundaries and hostile inputs), design Sec10 dim2. 7 cells.
// RED BY LINK.
//
// ROUTED MODEL GAP (Curie's own jurisdiction -- realize the model, never invent a body it never
// gave): design Sec10 dim2's own text names "a hostile sslm_config/sslm_decode_params
// (out-of-range layer_budget, negative chunk_budget) is SSLM_INVALID_ARGUMENT" as an applicable
// cell. sslm_decode_params's body IS given (design Sec8, one int32 field, layer_budget) and its
// own hostile cell is authored below (M7). sslm_config's body is NEVER given anywhere in the
// design document -- Sec7/Sec8 pass `const sslm_config*` through sslm_workspace_size/
// sslm_workspace_create but no section defines what sslm_config contains, beyond this same
// dim2 sentence implying a layer_budget/chunk_budget-shaped pair with no confirmation those are
// the ONLY fields or their exact domain. sslm_abi.h therefore declares sslm_config as an
// INCOMPLETE type (see that header's own comment) and this file cannot construct a hostile
// instance of it -- filed here as the routed finding rather than a guessed struct body. See
// Claude/Curie/t2138-abi-red-suite-2026-08-16.md.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec7.1/Sec9 C1 gate): sslm_workspace_create with a caller buffer
// smaller than sslm_workspace_size's own reported requirement rejects SSLM_BUFFER_TOO_SMALL. ---
static void TestDim2_M1_WorkspaceCreateTooSmallBufferRejected(sslm_model model) {
	const size_t required = sslm_workspace_size(model, /*config=*/nullptr);
	std::vector<uint8_t> buf(required > 0 ? required - 1 : 0);
	sslm_workspace ws = nullptr;
	CHECK(sslm_workspace_create(model, nullptr, buf.data(), buf.size(), &ws) ==
	      SSLM_BUFFER_TOO_SMALL);
	CHECK(ws == nullptr);
}

// --- Mechanism cell 2 (design Sec7.1/Sec9 C1 gate): sslm_workspace_create with a
// misaligned caller buffer address rejects SSLM_MISALIGNED_BUFFER, never a later kernel fault.
// ---
static void TestDim2_M2_WorkspaceCreateMisalignedBufferRejected(sslm_model model) {
	const size_t required = sslm_workspace_size(model, nullptr);
	// Over-allocate by a cache line and hand back an address offset by exactly one byte from a
	// naturally-aligned base -- guaranteed misaligned against any alignment requirement the
	// artifact could plausibly declare (1-, 2-, 4-, 8-, 16-, 32-, 64-byte).
	std::vector<uint8_t> buf(required + 64);
	uint8_t* misaligned = buf.data() + 1;
	sslm_workspace ws = nullptr;
	CHECK(sslm_workspace_create(model, nullptr, misaligned, required + 63, &ws) ==
	      SSLM_MISALIGNED_BUFFER);
	CHECK(ws == nullptr);
}

// --- Mechanism cell 3 (design Sec7.2/Sec9 C1 gate): sslm_kv_pool_create with a caller buffer
// smaller than block_count*sslm_kv_block_size(model) plus overhead rejects SSLM_BUFFER_TOO_SMALL.
// ---
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
	hostile_params.layer_budget = -1;
	int32_t out_tokens[1] = {0};
	sslm_seq batch[1] = {seq};
	CHECK(sslm_decode_step(model, batch, 1, &hostile_params, /*ws=*/nullptr, out_tokens) ==
	      SSLM_INVALID_ARGUMENT);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 = (void*)&TestDim2_M1_WorkspaceCreateTooSmallBufferRejected;
	(void)addr_0;
	volatile void* addr_1 = (void*)&TestDim2_M2_WorkspaceCreateMisalignedBufferRejected;
	(void)addr_1;
	volatile void* addr_2 = (void*)&TestDim2_M3_KvPoolCreateTooSmallBufferRejected;
	(void)addr_2;
	volatile void* addr_3 = (void*)&TestDim2_M4_KvPoolOverheadArithmeticOverflowSafe;
	(void)addr_3;
	volatile void* addr_4 = (void*)&TestDim2_M5_ModelMapHostileArtifactRejected;
	(void)addr_4;
	volatile void* addr_5 = (void*)&TestDim2_M6_AdapterMapBaseMismatchRejected;
	(void)addr_5;
	volatile void* addr_6 = (void*)&TestDim2_M7_DecodeParamsHostileLayerBudgetRejected;
	(void)addr_6;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
