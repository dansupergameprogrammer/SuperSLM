// T-2246 (test design) -- Coverage Model row 5, Trust boundaries (audit ST-1, plan r6 SS6 row
// 5, SS3.5 guard 4, SS5 S-A trust-boundary cells): the `_v3` params are a NEW caller-facing
// input surface; every documented rejection fires per its diagnostic and entry-path
// symmetrically. The stream-identity half of ST-1's symmetry leg lives jointly in dim10 P2.
//
// RED STATUS: link-red on sslm_speculate_step_v3 / sslm_speculate_params_init until S-E.
#include "fixture_common.h"

using namespace superslm;

namespace {

// J1 -- Malformed struct_size rejects before any field is consumed (the _v2 pattern's own
// validation order): zero, undersized, and oversized all reject, leaving the sequence usable.
void TestJ1_MalformedStructSizeRejected(sslm_model model, sslm_workspace ws,
                                        const CpuOracleModel& oracle,
                                        const std::vector<int32_t>& prompt) {
	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, 4, 8, {}, &params));

	const uint32_t bad_sizes[] = {0, sizeof(params) - 4u, sizeof(params) + 4u};
	for (const uint32_t sz : bad_sizes) {
		params.struct_size = sz;
		std::vector<int32_t> tok(16, -999), rows(16 * static_cast<size_t>(oracle.vocab_size), 0);
		int32_t produced = -1, stop = -1;
		const sslm_status st =
		    sslm_speculate_step_v3(model, seq, &params, ws, tok.data(), 16, rows.data(),
		                           static_cast<int32_t>(rows.size()), &produced, &stop);
		CHECK_MSG(st == SSLM_INVALID_ARGUMENT,
		          "struct_size %u must reject with the documented argument diagnostic",
		          static_cast<unsigned>(sz));
		CHECK(produced <= 0);  // no emission escapes a rejected call
	}
	// The rejected calls left a resumable sequence: a VALID call still works.
	params.struct_size = sizeof(params);
	std::vector<int32_t> tok(16, 0), rows(16 * static_cast<size_t>(oracle.vocab_size), 0);
	int32_t produced = 0, stop = -1;
	CHECK(sslm_speculate_step_v3(model, seq, &params, ws, tok.data(), 16, rows.data(),
	                             static_cast<int32_t>(rows.size()), &produced,
	                             &stop) == SSLM_OK);
}

// J2 -- Draft-length domain rejections when caller-set: K <= 0 rejects; K exceeding the
// remaining cache capacity rejects up front (the same class the chunk guard enforces,
// forward_sites.cpp:2133-2136).
void TestJ2_KDomainRejected(sslm_model model, sslm_workspace ws,
                            const CpuOracleModel& oracle, const std::vector<int32_t>& prompt) {
	SinglePool sp;
	ASSERT_TRUE(MakePool(model, 2, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	const int32_t bad_ks[] = {0, -3};
	for (const int32_t k : bad_ks) {
		sslm_speculate_params params{};
		ASSERT_TRUE(MakeSpecParams(model, k, 8, {}, &params));
		std::vector<int32_t> tok(16, 0), rows(16 * static_cast<size_t>(oracle.vocab_size), 0);
		int32_t produced = -1, stop = -1;
		const sslm_status st =
		    sslm_speculate_step_v3(model, seq, &params, ws, tok.data(), 16, rows.data(),
		                           static_cast<int32_t>(rows.size()), &produced, &stop);
		CHECK_MSG(st == SSLM_INVALID_ARGUMENT,
		          "K = %d is outside the caller-set domain and must reject",
		          static_cast<int>(k));
	}

	// K > remaining capacity: park occupancy near cap via the tampered-restore route, then
	// ask for more drafts than positions remain. Up-front rejection, no landings.
	SeqBlobBuffer seed_blob(model);
	size_t sz = seed_blob.size;
	ASSERT_TRUE(sslm_seq_save(seq, seed_blob.bytes.data(), &sz) == SSLM_OK);
	seed_blob.bytes.resize(sz);
	sslm_seq near = nullptr;
	ASSERT_TRUE(RestoreWithTamperedContextLength(model, &sp.pool, seed_blob.bytes,
	                                             oracle.context_cap - 2, &near));
	SeqBlobBuffer pre(model);
	pre.size = pre.bytes.size();
	ASSERT_TRUE(sslm_seq_save(near, pre.bytes.data(), &pre.size) == SSLM_OK);
	pre.bytes.resize(pre.size);

	sslm_speculate_params over{};
	ASSERT_TRUE(MakeSpecParams(model, oracle.context_cap, 8, {}, &over));  // K far past room
	std::vector<int32_t> tok(64, 0), rows(64 * static_cast<size_t>(oracle.vocab_size), 0);
	int32_t produced = -1, stop = -1;
	const sslm_status st =
	    sslm_speculate_step_v3(model, near, &over, ws, tok.data(), 64, rows.data(),
	                           static_cast<int32_t>(rows.size()), &produced, &stop);
	CHECK_MSG(st != SSLM_OK, "K beyond remaining capacity must reject");
	if (st != SSLM_OK) {
		SeqBlobBuffer post(model);
		post.size = post.bytes.size();
		ASSERT_TRUE(sslm_seq_save(near, post.bytes.data(), &post.size) == SSLM_OK);
		post.bytes.resize(post.size);
		CHECK(BlobSaturationCount(post.bytes) == BlobSaturationCount(pre.bytes));  // no landings
	}
}

// J3 -- Invalid handles and null outs reject per ABI conventions (the null-required-pointer /
// invalid-handle family, design SS6), symmetric across both driving histories.
void TestJ3_NullHandleAndNullOutRejected(sslm_model model, sslm_workspace ws,
                                         const CpuOracleModel& oracle,
                                         const std::vector<int32_t>& prompt) {
	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, 2, 4, {}, &params));
	std::vector<int32_t> tok(8, 0), rows(8 * static_cast<size_t>(oracle.vocab_size), 0);
	int32_t produced = 0, stop = -1;

	CHECK(sslm_speculate_step_v3(nullptr, seq, &params, ws, tok.data(), 8, rows.data(), 8,
	                             &produced, &stop) == SSLM_INVALID_ARGUMENT);
	CHECK(sslm_speculate_step_v3(model, nullptr, &params, ws, tok.data(), 8, rows.data(), 8,
	                             &produced, &stop) == SSLM_INVALID_ARGUMENT);
	CHECK(sslm_speculate_step_v3(model, seq, nullptr, ws, tok.data(), 8, rows.data(), 8,
	                             &produced, &stop) == SSLM_INVALID_ARGUMENT);
	CHECK(sslm_speculate_step_v3(model, seq, &params, ws, nullptr, 8, rows.data(), 8,
	                             &produced, &stop) == SSLM_INVALID_ARGUMENT);
	CHECK(sslm_speculate_step_v3(model, seq, &params, ws, tok.data(), 8, rows.data(), 8,
	                             nullptr, &stop) == SSLM_INVALID_ARGUMENT);
	CHECK(sslm_speculate_step_v3(model, seq, &params, ws, tok.data(), 8, rows.data(), 8,
	                             &produced, nullptr) == SSLM_INVALID_ARGUMENT);
	(void)oracle;
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim05 J1-J3 not run");
		int ec;
		PrintSummaryAndExit(&ec);
		return ec;
	}
	SslmModelView view;
	std::vector<uint8_t> bytes;
	std::string err;
	if (!LoadRealModelView(g_model_path, &view, &bytes, &err)) {
		SKIP_MSG("could not load real artifact: %s", err.c_str());
		int ec;
		PrintSummaryAndExit(&ec);
		return ec;
	}
	CpuOracleModel oracle;
	if (!LoadCpuOracleModel(view, &oracle, &err)) {
		SKIP_MSG("could not marshal CPU oracle: %s", err.c_str());
		int ec;
		PrintSummaryAndExit(&ec);
		return ec;
	}
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
	if (model) {
		sslm_config cfg{};
		cfg.max_batch = 16;
		cfg.max_chunk_budget = 256;
		cfg.max_layer_budget = static_cast<int32_t>(oracle.num_hidden_layers);
		const size_t ws_size = sslm_workspace_size(model, &cfg);
		AlignedBuffer ws_store(ws_size ? ws_size : 1);
		sslm_workspace ws = nullptr;
		if (sslm_workspace_create(model, &cfg, ws_store.data(), ws_store.size(), &ws) !=
		    SSLM_OK) {
			SKIP_MSG("workspace create failed -- dim05 cells need a workspace");
		}
		const std::vector<int32_t> prompt = NoRepeatPrompt(oracle.vocab_size, 12);
		if (ws) {
			TestJ1_MalformedStructSizeRejected(model, ws, oracle, prompt);
			TestJ2_KDomainRejected(model, ws, oracle, prompt);
			TestJ3_NullHandleAndNullOutRejected(model, ws, oracle, prompt);
			CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		}
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	int ec;
	PrintSummaryAndExit(&ec);
	return ec;
}
