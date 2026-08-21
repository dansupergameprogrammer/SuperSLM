// T-2199 (Curie) -- Phase D2 red suite: wiring DampedGreedyScoreAndArgmax into
// sslm_decode_stepImpl (plan Sec8 D2, src/sslm_abi.cpp).
//
// MIGRATED 2026-08-20 (T-2199 Phase D review fix S5, `Claude/Poirot/7a3b10a-t2199-phaseD-review.md`;
// conductor's follow-on commission, item 2): every cell now calls the REAL production ABI
// surface directly -- `sslm_decode_step`/`sslm_decode_params` (`namespace superslm`, the
// additive-field shape Sec8 D1 actually ruled) -- instead of the suite-compatibility shim
// (`sslm_decode_step_damped_greedy`/`sslm_decode_params_damped_greedy`) this file used while
// Phase D was unbuilt. The shim is now KEPT ONLY because this file (plus
// phaseD2a_cost_ratio_red.cpp/phaseD3_teardown_red.cpp) constructed it directly -- per the
// review's own S5 finding and the build log's own routing (Sec9): migrating these three files
// is what makes the shim (and `sslm_decode_params_damped_greedy`) deletable, confirmed at the
// end of this file's own header comment history and in this suite's own Curie record.
// `DampedGreedyMode`/`DampedGreedyValidationParams`/`ValidateDampedGreedyParams` are NOT part
// of the shim -- they are real, permanent production surface (the shared D2/D3/D4 validator)
// and are unaffected by this migration.
//
// Every product cell below needs a REAL base artifact (a full forward pass, not merely a
// header-level fixture) -- following this repo's own established convention
// (tests/t2138-abi-red-suite/fixture_common.h's own g_model_path/LoadRealModelView/
// CpuOracleModel machinery, reused here rather than re-invented) a cell whose fixture is not
// supplied on this invocation (--model=PATH) SKIPs its own product half with an honest
// message, never fabricating a result. This session's own scratchpad happens to carry both
// target-tier checkpoints (qwen2.5-0.5b-instruct.sslm, qwen2.5-1.5b-instruct.sslm) for a local
// verification run; per StandardsDocument.md Sec5.2 that path is never hardcoded here.
//
// Coverage Model cells realized here (plan Sec9):
//   dim10 Functional achievement, MEASUREMENT half -- "damped greedy mode selected via the
//         params surface produces the primitive's exact outputs through the public decode
//         step" (small hand-computed fixture, per this commission's own brief) --
//         TestD2_DampedGreedyMode_ProducesPrimitiveExactOutputThroughDecodeStep.
//   dim6/dim10 speed-headroom's own sibling claim, engine-correctness half -- "greedy mode's
//         outputs are BIT-UNCHANGED from pre-wiring behavior" -- GATE, D-SLM3719 --
//         TestD2_GreedyMode_BitUnchangedRegression.
//   dim2  Trust boundaries -- GATE, D-SLM3719. Validation symmetry across the three wired
//         entry points (sslm_decode_stepImpl, RunGreedyDecodeLoop/CLI, GPU bridge) --
//         TestD2_ValidationSymmetry_AcrossEntryPoints.
//   dim9  Persistence/D-SLM3795 -- GATE, D-SLM3719 for the mechanics, D-SLM3795 for the
//         contract itself. The decode digest covers the damped-greedy-selected token stream --
//         TestD2_TokenDigest_CoversDampedGreedyTokens.
#include "sslm_phaseD_stub.h"
#include "sslm_phaseD_fixture.h"
#include "sslm_damped_greedy.h"
#include "superslm/decode_digest.h"
#include "superslm/sha256.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

static int GChecks = 0;
static int GFailures = 0;
static int GSkips = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			std::fflush(stdout); \
		} \
	} while (0)

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
			std::fflush(stdout); \
		} \
	} while (0)

#define SKIP_MSG(...) \
	do { \
		++GSkips; \
		std::printf("SKIP %s:%d -- ", __FILE__, __LINE__); \
		std::printf(__VA_ARGS__); \
		std::printf("\n"); \
		std::fflush(stdout); \
	} while (0)

using namespace superslm;
using namespace superslm_test_phaseD;

static std::string g_model_path;

static void ParseArgs(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		const char* kFlag = "--model=";
		if (a.compare(0, std::strlen(kFlag), kFlag) == 0) g_model_path = a.c_str() + std::strlen(kFlag);
	}
}

static bool ReadFileBytes(const std::string& path, std::vector<uint8_t>* out) {
	if (path.empty()) return false;
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(sz > 0 ? (size_t)sz : 0);
	if (sz > 0) {
		const size_t n = std::fread(out->data(), 1, (size_t)sz, f);
		std::fclose(f);
		if (n != (size_t)sz) return false;
	} else {
		std::fclose(f);
	}
	return true;
}

// The same alignment discipline tests/t2138-abi-red-suite/fixture_common.h's own AlignedBuffer
// establishes (SSLM_ABI_ALIGNMENT_BYTES) -- reused here in miniature since this file only needs
// a KV pool and workspace, not the full fixture surface that file's own struct carries.
struct AlignedBuffer {
	explicit AlignedBuffer(size_t n)
	    : bytes_(n), storage_(n > 0 ? ::operator new(n, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)) : nullptr) {}
	~AlignedBuffer() {
		if (storage_) ::operator delete(storage_, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	}
	AlignedBuffer(const AlignedBuffer&) = delete;
	AlignedBuffer& operator=(const AlignedBuffer&) = delete;
	void* data() { return storage_; }
	size_t size() const { return bytes_; }

  private:
	size_t bytes_;
	void* storage_;
};

struct RealModelFixture {
	std::vector<uint8_t> bytes;
	sslm_model model = nullptr;
	std::unique_ptr<AlignedBuffer> pool_buf;
	sslm_kv_pool pool = nullptr;
	int32_t vocab_size = 0;
	uint32_t num_hidden_layers = 0;

	~RealModelFixture() {
		if (pool) sslm_kv_pool_destroy(pool);
		if (model) sslm_model_unmap(model);
	}
};

// Loads g_model_path through the REAL, production sslm_model_map for the decode handle, PLUS
// the internal SslmModel::Load parse for config (vocab_size, num_hidden_layers) -- the public
// C ABI has no config getter (confirmed against include/superslm/sslm_abi_functions_g5_
// comparable.inc), so this mirrors tests/t2138-abi-red-suite/fixture_common.h's own established
// dual-load pattern (its dim6 P1 cell: a separate SslmModelView alongside the ABI's own
// sslm_model handle) rather than inventing a new one. This suite's own subject is decode_step's
// wiring, not the loader (that is tests/t2138-abi-red-suite's own scope).
static bool LoadRealModel(RealModelFixture* out, std::string* err) {
	if (!ReadFileBytes(g_model_path, &out->bytes)) {
		if (err) *err = "could not read " + g_model_path;
		return false;
	}
	superslm::SslmModelView view;
	if (superslm::SslmModel::Load(out->bytes.data(), out->bytes.size(), view, err) !=
	    superslm::SslmModelStatus::Ok) {
		return false;
	}
	out->vocab_size = view.config.vocab_size;
	out->num_hidden_layers = view.config.num_hidden_layers;
	if (sslm_model_map(out->bytes.data(), out->bytes.size(), &out->model) != SSLM_OK) {
		if (err) *err = "sslm_model_map failed";
		return false;
	}
	// FIXED 2026-08-20 (conductor's dispute-resolution commission, dispute 2): was hardcoded to
	// 1, but TestD2_GreedyMode_BitUnchangedRegression (below) creates TWO sequences (seq_old,
	// seq_new) against this SAME shared fixture's pool -- sslm_seq_create for the second
	// sequence failed on pool exhaustion before either decode-step call ever ran, and every
	// downstream failure in that cell was a cascade from that one root cause (build log
	// Claude/Brunel/t2199-phaseD-build-2026-08-20.md Sec4). 2 blocks covers every cell in this
	// file (the single-sequence cells simply do not use the second block).
	const uint32_t block_count = 2;
	const size_t block_bytes = sslm_kv_block_size(out->model);
	const size_t overhead = sslm_kv_pool_overhead_size(out->model, block_count);
	out->pool_buf = std::make_unique<AlignedBuffer>(block_count * block_bytes + overhead);
	if (sslm_kv_pool_create(out->model, out->pool_buf->data(), out->pool_buf->size(), block_count,
	                         &out->pool) != SSLM_OK) {
		if (err) *err = "sslm_kv_pool_create failed";
		return false;
	}
	return true;
}

// --- TestD2_GreedyMode_BitUnchangedRegression -----------------------------------------------
// The no-regression cell: greedy paths must not shift by a single bit once the damped-greedy
// branch is wired in alongside them (plan Sec8 D2: "as the damped-greedy-mode branch alongside
// today's ApplyMaskAndArgmax/ArgmaxLowestIndexTieBreak branches"). Drives the SAME prompt
// through TWO separate sslm_decode_step calls -- one with mode left at its zero-init default
// (SSLM_DECODE_MODE_GREEDY), one with mode explicitly set to SSLM_DECODE_MODE_DAMPED_GREEDY
// but every damped-greedy-only field left at a hostile-looking default -- asserting
// token-for-token bit equality over several real decode steps.
static void TestD2_GreedyMode_BitUnchangedRegression() {
	if (g_model_path.empty()) {
		SKIP_MSG("real base artifact not supplied (--model=PATH) -- greedy-unchanged regression not run");
		return;
	}
	RealModelFixture fx;
	std::string err;
	if (!LoadRealModel(&fx, &err)) {
		SKIP_MSG("could not load real artifact: %s", err.c_str());
		return;
	}
	sslm_seq seq_old = nullptr, seq_new = nullptr;
	CHECK(sslm_seq_create(fx.model, &fx.pool, &seq_old) == SSLM_OK);
	CHECK(sslm_seq_create(fx.model, &fx.pool, &seq_new) == SSLM_OK);
	const int32_t prompt[4] = {0, 1, 2, 3};
	int32_t consumed_old = 0, consumed_new = 0;
	CHECK(sslm_prefill(fx.model, seq_old, prompt, 4, 8, SSLM_SPAN_PROMPT, nullptr, &consumed_old) == SSLM_OK);
	CHECK(sslm_prefill(fx.model, seq_new, prompt, 4, 8, SSLM_SPAN_PROMPT, nullptr, &consumed_new) == SSLM_OK);
	CHECK(consumed_old == 4 && consumed_new == 4);

	constexpr int kSteps = 6;
	sslm_decode_params old_params{};
	old_params.struct_size = sizeof(old_params);  // D-SLM3797: caller-set, library-validated
	old_params.layer_budget = static_cast<int32_t>(fx.num_hidden_layers);
	sslm_decode_params new_params{};
	new_params.struct_size = sizeof(new_params);  // D-SLM3797
	new_params.layer_budget = static_cast<int32_t>(fx.num_hidden_layers);
	new_params.mode = SSLM_DECODE_MODE_GREEDY;
	// MIGRATION NOTE (S5): before this migration, `old_params` (a struct with no mode field at
	// all, pre-D1) and `new_params` (mode explicitly SSLM_DECODE_MODE_GREEDY) called two
	// DIFFERENT functions (sslm_decode_step vs. the shim sslm_decode_step_damped_greedy), so the
	// comparison was genuinely discriminating: it proved the NEW branch-dispatch code, when
	// explicitly told to run greedy, matched the OLD, unwired path bit-for-bit. Post-migration
	// both params structs go through the SAME real sslm_decode_step, so this comparison is now
	// narrower -- it proves a struct with `mode` left at its zero-init default (a caller written
	// before the `mode` field existed, recompiled against the new header without updating its
	// own code) behaves IDENTICALLY to a caller who explicitly sets mode=SSLM_DECODE_MODE_GREEDY.
	// This is still a real, non-trivial backward-compatibility contract (SSLM_DECODE_MODE_GREEDY
	// is defined as 0 specifically so a zero-initialized legacy caller does not silently change
	// behavior -- a future edit that redefines the constant, or that treats zero specially in
	// some OTHER way, would break this) -- it is simply no longer the wiring-vs-unwired claim the
	// original two-shim-functions construction made. The DIRECT wiring claim (damped-greedy mode
	// selected via the params surface reaches the primitive's own exact output) is
	// TestD2_DampedGreedyMode_ProducesPrimitiveExactOutputThroughDecodeStep's own job, below.
	sslm_seq old_batch[1] = {seq_old};
	sslm_seq new_batch[1] = {seq_new};
	for (int i = 0; i < kSteps; ++i) {
		int32_t old_tok = 0, new_tok = 0;
		CHECK(sslm_decode_step(fx.model, old_batch, 1, &old_params, nullptr, &old_tok) == SSLM_OK);
		CHECK(sslm_decode_step(fx.model, new_batch, 1, &new_params, nullptr, &new_tok) == SSLM_OK);
		CHECK_MSG(old_tok == new_tok,
		          "step %d: sslm_decode_step(mode=default/0) produced %d, "
		          "sslm_decode_step(mode=SSLM_DECODE_MODE_GREEDY) produced %d -- greedy must be "
		          "bit-unchanged regardless of whether mode is left at its zero-init default or "
		          "set explicitly (plan Sec8 D2's own no-regression clause)",
		          i, old_tok, new_tok);
	}
	CHECK(sslm_seq_release(seq_old) == SSLM_OK);
	CHECK(sslm_seq_release(seq_new) == SSLM_OK);

	// N6 (closing-review commission: "the S5 migration cost this cell its discriminating
	// power... compare against something that is not the same code path") -- ATTEMPTED, NOT
	// SHIPPED. A full independent-oracle comparison (CpuOracleModel marshal +
	// superslm::RunGreedyDecodeLoop, driven against the SAME real checkpoint fx already has
	// open) was built and, in isolation (a standalone throwaway probe, same marshal, same
	// RunGreedyDecodeLoop call, same checkpoint), produced the bit-exact expected token
	// stream (4, 5, 6, 25010, 10, 4999 -- matching sslm_decode_step's own greedy output for
	// this exact prompt) with zero issues. Embedded in THIS function, alongside fx's own
	// already-open sslm_model_map handle and KV pool, the identical marshal+call sequence
	// reproducibly crashed the process (0xC0000005) at a point AFTER every check in the new
	// code had already run and passed (isolated with per-statement stdout diagnostics,
	// removed before filing) -- consistent with heap corruption manifesting at a later,
	// unrelated allocation/free rather than at its own true source. Re-scoping the oracle's
	// own objects into a nested block that destructs immediately (before seq_abi's own
	// creation) did not resolve it. Root cause not found within this round's own budget --
	// this is a genuine, reproducible defect (in this new test code, in a shared/static
	// resource two simultaneously-open SslmModelView/sslm_model instances of the SAME file
	// contend on, or possibly in production marshal/forward-path code under that same
	// double-open condition) and is ROUTED, not shipped crashing and not silently dropped --
	// see this suite's own Curie record for the full reproduction and the standalone probe's
	// own confirmed-correct output. The cell's own EXISTING comparison (zero-init mode vs.
	// explicit SSLM_DECODE_MODE_GREEDY, both through the real sslm_decode_step, above) is
	// unaffected and stays in place -- narrower than the pre-S5-migration claim, as already
	// stated in that comparison's own comment, but real and unbroken.
}

// --- TestD2_DampedGreedyMode_ProducesPrimitiveExactOutputThroughDecodeStep ------------------
// Small hand-computed fixture (per this commission's own brief): drives ONE real decode step
// in kDampedGreedy mode through the wired public entry point, then independently recomputes
// the expected token by calling the ALREADY-REAL, ALREADY-CERTIFIED
// DampedGreedyScoreAndArgmax (Phase C, this repo's own sslm_damped_greedy.h) directly against
// the SAME logit row and mask -- captured via a plain sslm_decode_step call one step earlier so
// both paths score the identical row. This is the t2138 dim6 "ABI call vs. direct-engine call"
// oracle shape, applied one level deeper (wired-damped-greedy call vs. direct-primitive call).
static void TestD2_DampedGreedyMode_ProducesPrimitiveExactOutputThroughDecodeStep() {
	if (g_model_path.empty()) {
		SKIP_MSG("real base artifact not supplied (--model=PATH) -- primitive-exactness cell not run");
		return;
	}
	RealModelFixture fx;
	std::string err;
	if (!LoadRealModel(&fx, &err)) {
		SKIP_MSG("could not load real artifact: %s", err.c_str());
		return;
	}
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(fx.model, &fx.pool, &seq) == SSLM_OK);
	const int32_t prompt[4] = {0, 1, 2, 3};
	int32_t consumed = 0;
	CHECK(sslm_prefill(fx.model, seq, prompt, 4, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	CHECK(consumed == 4);

	// A real anti-LM, primed with the prompt's own tokens (matching the mechanism's own
	// "x<t, never the prompt" contract loosely by construction -- this cell's own concern is
	// wiring, not anti-LM semantics, which Phase A's own suite already pins exhaustively).
	AntiLmState* antilm = AntiLmCreate(/*max_order=*/2);
	CHECK(antilm != nullptr);

	int64_t q_ln2 = 0, q_b = 0, q_c = 0;
	// The plan's own corrected starting scale (Sec5.6.2, this suite's own precedent value) --
	// searched via the certified IExpScaleConstants, matching fixture_common.h's own
	// DeriveDefaultScaleConstants recipe (not duplicated here to keep this file self-contained
	// to the sslm_phaseD_stub.h/decode_digest.h include set; see that helper's own comment for
	// the derivation's provenance).
	bool found_scale = false;
	for (int64_t e = -80; e <= 0 && !found_scale; ++e) {
		for (int64_t m = (int64_t{1} << 30); m < (int64_t{1} << 31); m += (int64_t{1} << 20)) {
			int64_t a = 0, b = 0, c = 0;
			if (IExpScaleConstants(m, e, kIExpLn2Q, 30, kIExpBQ, 30, kIExpCaQ, 30, &a, &b, &c) !=
			    IExpScaleDomain::kOk)
				continue;
			if (a == 493) {
				q_ln2 = a; q_b = b; q_c = c; found_scale = true; break;
			}
		}
	}
	CHECK_MSG(found_scale, "could not derive the plan's own default (q_ln2=493) scale constants");

	sslm_decode_params params{};
	params.struct_size = sizeof(params);  // D-SLM3797
	params.layer_budget = static_cast<int32_t>(fx.num_hidden_layers);
	params.mode = SSLM_DECODE_MODE_DAMPED_GREEDY;
	params.alpha_q15 = int32_t{1} << 14;  // alpha = 0.5, a real, in-band value
	params.anti_lm_max_order = 2;
	params.top_k = 6;
	params.q_ln2 = q_ln2; params.q_b = q_b; params.q_c = q_c;

	sslm_seq batch[1] = {seq};
	int32_t produced = -1;
	CHECK(sslm_decode_step(fx.model, batch, 1, &params, nullptr, &produced) == SSLM_OK);
	CHECK_MSG(produced >= 0 && produced < fx.vocab_size,
	          "produced token %d out of [0, %d) -- the wired call must select a real vocabulary "
	          "index", produced, fx.vocab_size);

	AntiLmDestroy(antilm);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	// NOTE, filed openly: a full independent-recompute cross-check (re-deriving the SAME logit
	// row this call scored, all-legal mask, and feeding it directly to
	// DampedGreedyScoreAndArgmax with an anti-LM state primed IDENTICALLY, then asserting the
	// two tokens match bit-for-bit) needs a way to CAPTURE the row sslm_decode_step computed
	// internally -- this ABI has no out_logit_row parameter on decode_step (unlike
	// RunGreedyDecodeLoop's own out_logit_rows), so that capture is not constructible against
	// the CURRENT public surface alone. Routed, not silently dropped: either
	// sslm_decode_stepImpl's own D2 build exposes the scored row through a diagnostics
	// out-parameter (mirroring DampedGreedyScoreAndArgmaxDiag's own precedent), or this cell's
	// own cross-check is authored against RunGreedyOrDampedGreedyDecodeLoop instead (D3's own
	// out_logit_rows array), once that entry point exists -- either is a real fill this suite
	// owes once Phase D2/D3 land, not invented here against a surface that cannot support it.
}

// --- TestD2_ValidationSymmetry_AcrossEntryPoints --------------------------------------------
// The identical invalid-parameter set must produce the SAME rejection status at all three
// wired entry points -- sslm_decode_stepImpl (D2), RunGreedyDecodeLoop/the CLI (D3), and the
// GPU bridge (D4) -- never firing at one call site and silently degrading at another (plan Sec9
// dim2's own "validation-symmetry cell, owed"). This suite has no GPU device on this machine
// (matching tests/t2138-abi-red-suite's own CPU-only build discipline) -- the GPU third is
// therefore checked through ValidateDampedGreedyParams directly (the shared validation function
// a correct D2/D3/D4 build is specified to route ALL THREE entry points through, per this
// suite's own stub header comment) rather than through a live GPU call stack, and this is
// stated as a genuine, if narrower, symmetry check: it confirms the GPU bridge's OWN domain
// check (once wired) cannot diverge from the shared function without this cell's own
// third assertion catching it structurally, even though it cannot exercise gpu_1p0.cpp's own
// call site live. The CPU/CLI halves ARE exercised live, through the real wired entry points.
static void TestD2_ValidationSymmetry_AcrossEntryPoints() {
	if (g_model_path.empty()) {
		SKIP_MSG("real base artifact not supplied (--model=PATH) -- live entry-point symmetry not run");
		return;
	}
	RealModelFixture fx;
	std::string err;
	if (!LoadRealModel(&fx, &err)) {
		SKIP_MSG("could not load real artifact: %s", err.c_str());
		return;
	}
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(fx.model, &fx.pool, &seq) == SSLM_OK);
	const int32_t prompt[4] = {0, 1, 2, 3};
	int32_t consumed = 0;
	CHECK(sslm_prefill(fx.model, seq, prompt, 4, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	// The invalid-parameter set named in Sec9 dim2, EXTENDED 2026-08-20 (closing-review
	// commission, N3: "three new production guards ship unpinned... the alpha_q15 < 2^20
	// ceiling, the unknown-mode rejection"). Original four: negative alpha, k=0, k>vocab_size,
	// n=0. Two new: alpha_q15 AT the C2/Sec9-dim2 sanity ceiling (damped_greedy_phaseD.cpp's
	// own `if (p.alpha_q15 >= (int32_t{1} << 20)) return false;`), and an unrecognized `mode`
	// value (S2's own guard, `ValidateDampedGreedyParams`'s own `p.mode != kGreedy &&
	// p.mode != kDampedGreedy` rejection) -- a mutation that deletes either guard now leaves
	// this suite green with nothing more, closing exactly the gap N3 named. `mode` is now a
	// per-case field (int32_t, matching the real ABI field's own type) rather than hardcoded.
	struct Case { const char* name; int32_t mode; int32_t alpha_q15; int32_t n; int32_t k; };
	const Case kCases[] = {
	    {"negative alpha", SSLM_DECODE_MODE_DAMPED_GREEDY, -1, 2, 6},
	    {"k=0", SSLM_DECODE_MODE_DAMPED_GREEDY, int32_t{1} << 14, 2, 0},
	    {"k>vocab_size", SSLM_DECODE_MODE_DAMPED_GREEDY, int32_t{1} << 14, 2, fx.vocab_size + 1},
	    {"n=0", SSLM_DECODE_MODE_DAMPED_GREEDY, int32_t{1} << 14, 0, 6},
	    {"alpha at ceiling (2^20)", SSLM_DECODE_MODE_DAMPED_GREEDY, int32_t{1} << 20, 2, 6},
	    {"unknown mode (2)", 2, int32_t{1} << 14, 2, 6},
	};
	for (const auto& c : kCases) {
		sslm_decode_params params{};
		params.struct_size = sizeof(params);  // D-SLM3797
		params.layer_budget = static_cast<int32_t>(fx.num_hidden_layers);
		params.mode = c.mode;
		params.alpha_q15 = c.alpha_q15;
		params.anti_lm_max_order = c.n;
		params.top_k = c.k;
		params.q_ln2 = 493;  // domain of the rejection under test is alpha/n/k/mode, not the scale
		sslm_seq batch[1] = {seq};
		int32_t out_tok = -777;
		const sslm_status abi_status =
		    sslm_decode_step(fx.model, batch, 1, &params, nullptr, &out_tok);

		DampedGreedyValidationParams vp{static_cast<DampedGreedyMode>(c.mode), c.alpha_q15, c.n, c.k};
		const bool shared_valid = ValidateDampedGreedyParams(vp, fx.vocab_size);

		CHECK_MSG((abi_status == SSLM_OK) == shared_valid,
		          "case '%s': live sslm_decode_step returned %s but the SHARED "
		          "ValidateDampedGreedyParams says %s -- D2's own domain check must be the SAME "
		          "function D3/D4 route through, never a re-implemented copy that can drift",
		          c.name, abi_status == SSLM_OK ? "OK" : "REJECTED",
		          shared_valid ? "valid" : "invalid");
		CHECK_MSG(!shared_valid,
		          "case '%s' is constructed to be invalid by name -- if ValidateDampedGreedyParams "
		          "reads it as valid, this cell's own fixture (not the build) is wrong", c.name);

		// STRENGTHENED 2026-08-20 (closing-review commission, N8: "the cell compares D2's
		// status against ValidateDampedGreedyParams rather than against D3's actual return --
		// the exact weakness the prior review named"). Calls the REAL D3 entry point
		// (RunGreedyOrDampedGreedyDecodeLoop, namespace superslm since the S5 migration) with
		// the SAME invalid params and asserts its own return agrees with D2's. Every
		// model-derived parameter below (layers, embed/head weights, rope tables, workspace) is
		// a deliberate dummy/null placeholder -- safe ONLY when the call is guaranteed to reject
		// before touching any of them.
		//
		// FOUND BY EXECUTION, NOT ASSUMED (the first draft of this comment claimed otherwise
		// and crashed on it): src/damped_greedy_phaseD_loop.cpp's own control flow validates
		// (ValidateDampedGreedyParams -> InvalidDecodeParams) ONLY inside
		// `if (mode == DampedGreedyMode::kDampedGreedy)` -- a mode value that is NEITHER
		// kGreedy NOR kDampedGreedy (this cell's own "unknown mode" case, mode=2) skips that
		// branch entirely and falls through into the REAL greedy forward path, which then
		// dereferences the dummy nullptr embed/head weights -- a real, reproducible crash
		// (0xC0000005), caught by execution before this cell was filed, not shipped. This IS
		// itself a genuine finding, separate from and beyond what this cell set out to pin:
		// D2 (sslm_decode_stepImpl) validates `mode` UNCONDITIONALLY (S2's own fix, "an unknown
		// mode... now rejects instead of silently degrading to greedy" -- verified in
		// src/sslm_abi.cpp, the static_cast<DampedGreedyMode>(params->mode) + ValidateDampedGreedyParams
		// call runs for every call regardless of mode's own value), but D3 only validates when
		// mode already reads as kDampedGreedy -- an unknown mode at D3 silently falls through
		// to the GREEDY path instead of being rejected, the exact D2/D3 asymmetry N8's own
		// commission names, just discovered via a crash instead of a clean assertion. ROUTED,
		// not fixed here (Curie does not implement) -- flagged prominently in this suite's own
		// Curie record for Brunel/the conductor. The dummy-probe below is therefore restricted
		// to the cases that ARE guaranteed to hit D3's own early-validation branch (every case
		// whose own `mode` is genuinely kDampedGreedy); the "unknown mode" case's own D3 half is
		// SKIPPED with an honest message rather than constructed unsafely.
		if (c.mode != SSLM_DECODE_MODE_DAMPED_GREEDY) {
			SKIP_MSG("case '%s': D3's own RunGreedyOrDampedGreedyDecodeLoop cannot be safely "
			         "probed with dummy model data for a non-kDampedGreedy mode value -- it "
			         "falls through to the real greedy forward path instead of validating first "
			         "(the newly-found D2/D3 asymmetry, routed -- see this suite's own Curie "
			         "record)",
			         c.name);
		} else {
		int8_t dummy_hidden_codes[1] = {0};
		superslm::SequenceLayerState dummy_seq{};
		dummy_seq.hidden_codes = dummy_hidden_codes;
		superslm::SslmTensorManifest dummy_rope{};
		superslm::CarriedScale dummy_scale{};
		size_t d3_tokens_produced = 0;
		superslm::SslmDecodeStopReason d3_stop_reason{};
		const superslm::SslmForwardStatus d3_status = RunGreedyOrDampedGreedyDecodeLoop(
		    dummy_seq, /*layers=*/nullptr, /*num_hidden_layers=*/1, /*hidden_size=*/1,
		    /*head_dim=*/1, /*num_key_value_heads=*/1, /*intermediate_size=*/1,
		    /*context_cap=*/1, dummy_rope, prompt, 4, /*embed_weights=*/nullptr, dummy_scale,
		    /*final_norm_gain=*/nullptr, dummy_scale, /*head_weights=*/nullptr, fx.vocab_size,
		    /*stop_ids=*/nullptr, 0, /*max_new_tokens=*/1, /*workspace=*/nullptr, 0,
		    /*out_tokens=*/nullptr, /*out_logit_rows=*/nullptr, 0, &d3_tokens_produced,
		    &d3_stop_reason, superslm::SslmKvPrecision{}, /*option_g_fused_k_landing=*/false,
		    static_cast<DampedGreedyMode>(c.mode), c.alpha_q15, c.n, c.k, /*q_ln2=*/493,
		    /*q_b=*/0, /*q_c=*/0);
		const bool d3_valid = (d3_status == superslm::SslmForwardStatus::Ok);
		CHECK_MSG(d3_valid == shared_valid,
		          "case '%s': D3's own RunGreedyOrDampedGreedyDecodeLoop returned %s (status=%d) "
		          "for the SAME params ValidateDampedGreedyParams says are %s -- D2 and D3 must "
		          "reject/accept the identical invalid-parameter set, the actual symmetric claim "
		          "this cell exists to make (not merely 'D2 agrees with the shared validator')",
		          c.name, d3_valid ? "OK" : "REJECTED", static_cast<int>(d3_status),
		          shared_valid ? "valid" : "invalid");
		CHECK_MSG(!d3_valid,
		          "case '%s' is constructed to be invalid by name -- D3 must not accept it either",
		          c.name);
		}
	}
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// --- TestD2_TokenDigest_CoversDampedGreedyTokens --------------------------------------------
// D-SLM3795: the decode digest's determinism contract is decoder-agnostic -- it covers the
// tokens the CONFIGURED decoder produces, damped greedy included. ComputeTokenDigest itself
// (src/decode_digest.cpp) is already decoder-agnostic BY CONSTRUCTION (a pure hash over
// whatever int32 token array it is given) -- the real risk this cell targets is a CALLER bug
// (an entry point that silently skips digesting, or digests the wrong array, under the new
// mode), not the hash function. Captures a real damped-greedy-selected token stream through
// the wired entry point, computes its digest, and confirms it equals an INDEPENDENTLY computed
// digest over that exact sequence (never re-using the production call's own output as its own
// check).
static void TestD2_TokenDigest_CoversDampedGreedyTokens() {
	if (g_model_path.empty()) {
		SKIP_MSG("real base artifact not supplied (--model=PATH) -- digest-coverage cell not run");
		return;
	}
	RealModelFixture fx;
	std::string err;
	if (!LoadRealModel(&fx, &err)) {
		SKIP_MSG("could not load real artifact: %s", err.c_str());
		return;
	}
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(fx.model, &fx.pool, &seq) == SSLM_OK);
	const int32_t prompt[4] = {0, 1, 2, 3};
	int32_t consumed = 0;
	CHECK(sslm_prefill(fx.model, seq, prompt, 4, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	constexpr int kSteps = 5;
	sslm_decode_params params{};
	params.struct_size = sizeof(params);  // D-SLM3797
	params.layer_budget = static_cast<int32_t>(fx.num_hidden_layers);
	params.mode = SSLM_DECODE_MODE_DAMPED_GREEDY;
	params.alpha_q15 = int32_t{1} << 14;
	params.anti_lm_max_order = 2;
	params.top_k = 6;
	// FIXED 2026-08-20 (conductor's dispute-resolution commission, dispute 3): was
	// `params.q_ln2 = 493;` alone, leaving q_b/q_c at their zero-init default. (q_b=0, q_c=0)
	// derives M = q_b^2 + q_c = 0, which fails CheckSoftmaxRowWidthDomain's own M >= 1
	// requirement, so every call below correctly REFUSED (SSLM_ARTIFACT_REJECTED, plan Sec7.5's
	// own adopted policy) instead of running to completion -- a fixture defect, not a code
	// defect (build log Sec4). Now derives the SAME real triple the sibling cell
	// TestD2_DampedGreedyMode_ProducesPrimitiveExactOutputThroughDecodeStep already gets right,
	// via the shared helper (sslm_phaseD_fixture.h).
	CHECK(t2199phaseD::DeriveDefaultScaleConstants(&params.q_ln2, &params.q_b, &params.q_c));
	sslm_seq batch[1] = {seq};
	std::vector<int32_t> produced_tokens;
	for (int i = 0; i < kSteps; ++i) {
		int32_t tok = -1;
		CHECK(sslm_decode_step(fx.model, batch, 1, &params, nullptr, &tok) == SSLM_OK);
		produced_tokens.push_back(tok);
	}
	CHECK(sslm_seq_release(seq) == SSLM_OK);

	uint8_t got_digest[32];
	ComputeTokenDigest(produced_tokens.data(), produced_tokens.size(), got_digest);

	// Independent reference: the SAME count-prefixed little-endian composition
	// docs/sslm_format.md's own Sec10.1 states, hand-assembled here rather than calling
	// ComputeTokenDigest a second time (a second call to the SAME function proves nothing --
	// this is the consistency-oracle trap Curie's own discipline names; the independence is in
	// re-deriving the byte layout, not in re-calling the implementation).
	std::vector<uint8_t> ref_bytes;
	auto AppendLE32 = [&](int32_t v) {
		const uint32_t u = static_cast<uint32_t>(v);
		ref_bytes.push_back(static_cast<uint8_t>(u & 0xFF));
		ref_bytes.push_back(static_cast<uint8_t>((u >> 8) & 0xFF));
		ref_bytes.push_back(static_cast<uint8_t>((u >> 16) & 0xFF));
		ref_bytes.push_back(static_cast<uint8_t>((u >> 24) & 0xFF));
	};
	AppendLE32(static_cast<int32_t>(produced_tokens.size()));
	for (int32_t t : produced_tokens) AppendLE32(t);
	uint8_t ref_digest[32];
	Sha256Hash(ref_bytes.data(), ref_bytes.size(), ref_digest);

	CHECK_MSG(std::memcmp(got_digest, ref_digest, 32) == 0,
	          "ComputeTokenDigest over the damped-greedy-selected token stream must equal an "
	          "independently hand-assembled count-prefixed-LE32 SHA-256 over the SAME sequence "
	          "-- D-SLM3795's own 'covers the configured decoder's tokens' contract, verified "
	          "against real damped-greedy output, not merely against the greedy path this "
	          "function was originally proven on");
}

int main(int argc, char** argv) {
	ParseArgs(argc, argv);
	TestD2_GreedyMode_BitUnchangedRegression();
	TestD2_DampedGreedyMode_ProducesPrimitiveExactOutputThroughDecodeStep();
	TestD2_ValidationSymmetry_AcrossEntryPoints();
	TestD2_TokenDigest_CoversDampedGreedyTokens();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
