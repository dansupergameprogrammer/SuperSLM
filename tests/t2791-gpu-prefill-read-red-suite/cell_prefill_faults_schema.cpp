// T-2791 (Curie) -- plan Sec3.4 row 5 (failure and rejection paths), the members plan Sec2.6
// listed as not executable on the Qwen3 artifact: "Every prefill exit after the pre-scan other
// than kNone leaves the read refusing, one cell each", plus the schema twin's exits that return
// BEFORE the pre-scan (plan Sec3.1: those leave the snapshot as it was), and the content of the
// schema twin's kNone snapshot.
//
// Prompt twin, on every supplied artifact (compiled with /DSUPERSLM_T2169_CHUNK_RECORDING_FAULT_
// INJECTION, as tests/t2178-gpu-batched-prefill-red-suite does):
//   F1 the device-computed override: ArmT2169ChunkRecordingFaultInjection(1) on a 2-token chunk;
//   F2 / F3 / F4 the recording tail's pre-Close, Signal and bad_alloc faults;
//   F1-F4 each return exactly SSLM_DEVICE_LOST (plan Sec3.4 row 5, tightened by T-2798 / T-2801:
//   plan Sec3.6 moves only the device-guard refusal to SSLM_SEQUENCE_REJECTED; every infrastructure
//   fault stays SSLM_DEVICE_LOST, and F1 is mutant (j)'s in-suite killer);
//   F1b (T-2806 G2, T-2814) the recording seam fires in a NON-FINAL sub-chunk: after the base prefill,
//   ArmT2169ChunkRecordingFaultInjection(1) and a 6-token continuation. The seam's index counts within
//   the sub-chunk (superslm_gpu.cpp's SubmitOneSubChunkToFullDepthForG5Bridge), and sub-chunks are
//   kT2169TdrSafeMaxChunkTokens = 4 tokens, so it fires in sub-chunk 1 of 2, whose submit fails. Asserts
//   exactly SSLM_DEVICE_LOST, the context length unchanged by the call (sub-chunk 1's tokens are not
//   committed and sub-chunk 2 is never submitted), and that the read refuses. F1b0 is its must-accept
//   neighbour: the same 6 tokens unarmed read their frame. F1b is mutant (n)'s killer: F1 returns
//   through the final sub-chunk, where plan Sec3.6 item 1's flag is not consulted;
//   F6 recovery: reset + prefill afterwards reads the correct frame.
// Schema twin, on a schema-bearing artifact (--g5fixture; plan Sec3.4 row 5 names the G5
// suite's argv artifact, and the re-strike executed these members on it, Claude/Loki/te270-*):
//   S0 kNone content equals the prompt twin's frame over the same tokens;
//   S1 kDfa with >= 1 consumed -> refuse (the exit that SETS ready_for_logits; mutant (d)'s killer);
//   S2 first token unreachable (before the pre-scan) -> the prior frame;
//   S3 a schema bound across reset, SSLM_OK -> that call's frame;
//   S3b schema unbound (before the pre-scan) -> the prior frame;
//   S4 / S5 kPositionCap with 2 and with 0 admitted -> refuse;
//   F5 the device override on the schema twin -> exactly SSLM_DEVICE_LOST (tightened from `!= SSLM_OK`
//   by T-2814, plan Sec3.4 row 5 and T-2806 G1: the schema twin keeps SSLM_DEVICE_LOST, D-SLM7311, so a
//   leak of Sec3.6's classification into it that classifies by value turns F5 red), and refuse.
// Every refused read also leaves *out_required at its sentinel (plan Sec3.4 row 7, T-2806 N4).
// --g5fixture must carry a SchemaMasks section: an artifact without one fails the cell, it does not
// skip it (T-2806 M2).
// NOT A CELL, per plan Sec2.6 and Sec3.4 row 5:
//   - the schema twin's kEmbed exit. It is dead code: no loadable artifact reaches it.
//     SchemaMasksTable::Parse rejects a transition token >= vocab_size, the embed scan rejects only
//     token < 0 or >= vocab_size, and a tie between the DFA and embed counts resolves to kDfa, so
//     any token that fails embed fails DFA at the same index first (gpu_1p0.cpp
//     RunChunkAdmissionPreScan; schema_masks.h Parse). The snapshot is invalidated at the pre-scan
//     whatever the exit, so the dead exit needs no writer of its own.
//   - a std::bad_alloc unwinding a chunk (gpu_1p0.cpp's rethrow before the scale copy-back). No
//     existing seam reaches it: F4's tail bad_alloc seam is caught inside the submit primitive and
//     ends in the device override (SSLM_DEVICE_LOST), as do the allocation sites. Plan Sec3.1's
//     writer table covers the member instead -- valid = false before the pre-scan, valid = true
//     only as the kNone exit's last statement, which an unwinding exception never reaches -- and
//     the code reviewer checks that table.
//
// ORACLE: "refuse" is SSLM_PREFILL_HIDDEN_UNAVAILABLE with no frame output written; a frame is
// final_norm of the live residual right after a successful PROMPT-twin prefill over the same
// tokens (fixture_common.h, OracleFromLive) -- never the read under test. The DFA chains come from
// an independent parse of the artifact's SchemaMasks section (IndependentSchema).
//
// Red at v1.5.0 by LINK: LNK2019 on sslm_gpu_seq_read_prefill_final_hidden.
//
// Run: cell_prefill_faults_schema.exe [--qwen3=PATH] [--synthetic=PATH] [--g5fixture=PATH]
#include "fixture_common.h"

namespace {

void ExpectRefuse(const char* tag, const char* cell, const Frame& f) {
	CHECK_MSG(f.status == kPrefillHiddenUnavailable, "[%s] %s: read returned %s, want SSLM_PREFILL_HIDDEN_UNAVAILABLE",
	          tag, cell, StatusName(f.status));
	CHECK_MSG(f.OutputsUntouched(), "[%s] %s: a refused read wrote its codes or scale", tag, cell);
	CHECK_MSG(f.required == kRequiredSentinel, "[%s] %s: a check-3 refusal wrote *out_required (%zu)", tag, cell,
	          f.required);
}
void ExpectFrame(const char* tag, const char* cell, const Frame& f, const Frame& want, size_t H) {
	CHECK_MSG(f.status == SSLM_OK && f.SameFrame(want, H),
	          "[%s] %s: read returned %s, differing from the expected frame in %zu of %zu codes", tag, cell,
	          StatusName(f.status), DiffCodes(f, want, H), H);
}

// ---- prompt twin: fault seams --------------------------------------------------------------
void PromptFaults(GpuModelFixture& fx) {
	const char* tag = fx.path.c_str();
	SslmGpuContext* ctx = fx.ctx;
	SslmGpuSequenceHandle* s = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, fx.model, 64, &s) == SSLM_OK && s);
	if (!s) return;
	const std::vector<int32_t> P = fx.TokensP();
	const std::vector<int32_t> two = {P[0], P[1]};
	auto base = [&]() -> bool { return sslm_gpu_seq_reset(ctx, s) == SSLM_OK && Prefill(fx, s, P) == SSLM_OK; };

	// Must-accept neighbour: the same 2-token continuation, unarmed, succeeds and reads its frame.
	CHECK_MSG(base(), "[%s] SETUP base", tag);
	CHECK_MSG(Prefill(fx, s, two) == SSLM_OK, "[%s] SETUP unarmed continuation", tag);
	ExpectFrame(tag, "F0 unarmed 2-token continuation (must-accept)", ReadVerb(fx, s), OracleFromLive(fx, s), fx.hidden);

	CHECK_MSG(base(), "[%s] SETUP base", tag);
	superslm_gpu::ArmT2169ChunkRecordingFaultInjection(1);
	const SslmGpuStatus f1 = Prefill(fx, s, two);
	superslm_gpu::ClearT2169ChunkRecordingFaultInjection();
	// Plan Sec3.4 row 5 (T-2798): tightened from `!= SSLM_OK` to `== SSLM_DEVICE_LOST`. The recording
	// seam's status is GpuGemmGroupArithmeticInvalid, an infrastructure fault; Sec3.6 classifies by
	// provenance, so it must stay SSLM_DEVICE_LOST. Classification by value (mutant (j)) maps it to
	// SSLM_SEQUENCE_REJECTED and turns F1 red.
	CHECK_MSG(f1 == SSLM_DEVICE_LOST, "[%s] F1: the armed recording fault returned %s, want SSLM_DEVICE_LOST", tag,
	          StatusName(f1));
	std::printf("    [%s] F1 prefill status under the override: %s\n", tag, StatusName(f1));
	ExpectRefuse(tag, "F1 prompt device-computed override", ReadVerb(fx, s));

	// F1b0 / F1b (T-2806 G2): the recording fault in the FIRST of two sub-chunks of a 6-token
	// continuation. Must-accept first: the same 6 tokens, unarmed.
	const std::vector<int32_t> six = fx.Run(6, P[1]);
	CHECK_MSG(base(), "[%s] SETUP base", tag);
	CHECK_MSG(Prefill(fx, s, six) == SSLM_OK, "[%s] F1b0 SETUP: the unarmed 6-token continuation", tag);
	ExpectFrame(tag, "F1b0 unarmed 6-token continuation (must-accept)", ReadVerb(fx, s), OracleFromLive(fx, s),
	            fx.hidden);
	CHECK_MSG(base(), "[%s] SETUP base", tag);
	const int64_t open_len = ContextLength(s);
	superslm_gpu::ArmT2169ChunkRecordingFaultInjection(1);
	const SslmGpuStatus f1b = Prefill(fx, s, six);
	superslm_gpu::ClearT2169ChunkRecordingFaultInjection();
	const int64_t after_len = ContextLength(s);
	std::printf("    [%s] F1b 6-token continuation, fault in sub-chunk 1 of 2: status %s, context %lld -> %lld\n", tag,
	            StatusName(f1b), static_cast<long long>(open_len), static_cast<long long>(after_len));
	CHECK_MSG(f1b == SSLM_DEVICE_LOST,
	          "[%s] F1b: the recording fault in a non-final sub-chunk returned %s, want SSLM_DEVICE_LOST", tag,
	          StatusName(f1b));
	CHECK_MSG(after_len == open_len,
	          "[%s] F1b: the context length moved from %lld to %lld; no token of the failed call may commit", tag,
	          static_cast<long long>(open_len), static_cast<long long>(after_len));
	ExpectRefuse(tag, "F1b recording fault in a non-final sub-chunk", ReadVerb(fx, s));

	// TE-425 (SuperSLM 1.8.0 plan `te421-slm172-host-oom.md` Sec3.3, E-5): F4's seam is a host
	// allocation failure (std::bad_alloc) after Close and Signal succeeded and the work was waited out,
	// on a live device -- rule 2, SSLM_GPU_ALLOCATION_FAILED. F2's and F3's seams throw a plain
	// std::runtime_error, a non-allocation fault (rule 3), and stay SSLM_DEVICE_LOST.
	struct Tail { const char* cell; void (*arm)(); void (*clr)(); SslmGpuStatus want; } tails[] = {
	    {"F2 prompt recording-tail fault (pre-Close)", superslm_gpu::ArmT2169ChunkRecordingTailFaultInjection,
	     superslm_gpu::ClearT2169ChunkRecordingTailFaultInjection, SSLM_DEVICE_LOST},
	    {"F3 prompt recording-tail Signal fault", superslm_gpu::ArmT2169ChunkRecordingTailSignalFaultInjection,
	     superslm_gpu::ClearT2169ChunkRecordingTailSignalFaultInjection, SSLM_DEVICE_LOST},
	    {"F4 prompt recording-tail bad_alloc", superslm_gpu::ArmT2169ChunkRecordingTailBadAllocFaultInjection,
	     superslm_gpu::ClearT2169ChunkRecordingTailBadAllocFaultInjection, SSLM_GPU_ALLOCATION_FAILED},
	};
	for (const Tail& t : tails) {
		CHECK_MSG(base(), "[%s] SETUP base", tag);
		t.arm();
		SslmGpuStatus st = SSLM_OK;
		bool threw = false;
		try {
			st = Prefill(fx, s, two);
		} catch (...) {
			threw = true;  // a failed call, so the read must still refuse -- but not the status Sec3.4 requires
			std::printf("    [%s] %s: the call threw\n", tag, t.cell);
		}
		t.clr();
		// Plan Sec3.4 row 5 (T-2798): tightened from `!= SSLM_OK` to an exact status (TE-425: per tail).
		CHECK_MSG(!threw && st == t.want, "[%s] %s: the armed fault returned %s, want %s", tag,
		          t.cell, threw ? "an exception" : StatusName(st), StatusName(t.want));
		std::printf("    [%s] %s: prefill status %s\n", tag, t.cell, StatusName(st));
		ExpectRefuse(tag, t.cell, ReadVerb(fx, s));
	}

	CHECK_MSG(base(), "[%s] F6 SETUP: reset + prefill after the faults", tag);
	ExpectFrame(tag, "F6 recovery: reset + prefill after every fault", ReadVerb(fx, s), OracleFromLive(fx, s), fx.hidden);
	sslm_gpu_seq_release(ctx, s);
}

// ---- schema twin ---------------------------------------------------------------------------
void SchemaMembers(GpuModelFixture& fx) {
	const char* tag = fx.path.c_str();
	SslmGpuContext* ctx = fx.ctx;
	if (!SslmGpuModelHasSchemasForG5Bridge(fx.model)) {
		// Not a SKIP (T-2806 M2): a --g5fixture without schemas is the wrong artifact, not a missing one.
		CHECK_MSG(false, "[%s] schema members: --g5fixture carries no SchemaMasks section", tag);
		return;
	}
	IndependentSchema ds;
	if (!ds.Build(fx.bytes, g_schema_name)) {
		CHECK_MSG(false, "[%s] schema members: no schema named '%s' in an independent parse", tag, g_schema_name.c_str());
		return;
	}
	const int32_t idx = SslmGpuSchemaLookupForG5Bridge(fx.model, g_schema_name.c_str());
	CHECK_MSG(idx >= 0, "[%s] SETUP: schema lookup", tag);
	if (idx < 0) return;

	// A legal chain t0..t3 from the DFA's start state (state 0, as bound on a fresh walk).
	std::vector<int32_t> chain;
	std::vector<uint32_t> states = {0};
	for (int k = 0; k < 4; ++k) {
		const int32_t t = ds.FirstLegal(states.back());
		uint32_t nx = 0;
		if (t < 0 || !ds.Next(states.back(), t, &nx)) break;
		chain.push_back(t);
		states.push_back(nx);
	}
	CHECK_MSG(chain.size() == 4, "[%s] SETUP: a 4-token legal DFA chain from the start state", tag);
	if (chain.size() != 4) return;
	const std::vector<int32_t> P = fx.TokensP();
	SslmGpuSequenceHandle* s = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, fx.model, 64, &s) == SSLM_OK && s);
	if (!s) return;
	int32_t consumed = -1;
	auto schema_prefill = [&](const std::vector<int32_t>& t) {
		consumed = -1;
		return SslmGpuSeqPrefillSchemaContentForG5Bridge(ctx, s, t.data(), static_cast<int32_t>(t.size()),
		                                                 fx.one_layer_budget, &consumed);
	};
	auto prompt_ref = [&](const std::vector<int32_t>& t) -> Frame {  // the oracle frame for tokens t
		CHECK(sslm_gpu_seq_reset(ctx, s) == SSLM_OK);
		CHECK(Prefill(fx, s, t) == SSLM_OK);
		return OracleFromLive(fx, s);
	};
	// T-2934 reconciliation: schema binding is fresh-or-reset only. Bind before the
	// prompt; prompt spans do not advance the DFA, so the schema-content oracle is unchanged.
	// reset; bind; prompt P; schema [t0, t1] -> the schema kNone frame.
	auto base_schema = [&]() -> bool {
		return sslm_gpu_seq_reset(ctx, s) == SSLM_OK &&
		       SslmGpuSeqSetSchemaForG5Bridge(ctx, s, idx) == SSLM_OK && Prefill(fx, s, P) == SSLM_OK &&
		       schema_prefill({chain[0], chain[1]}) == SSLM_OK &&
		       consumed == 2;
	};

	// S0: the schema twin's kNone snapshot content.
	std::vector<int32_t> p_t0t1 = P;
	p_t0t1.push_back(chain[0]);
	p_t0t1.push_back(chain[1]);
	const Frame ref_s0 = prompt_ref(p_t0t1);
	CHECK_MSG(base_schema(), "[%s] S0 SETUP: schema prefill of [t0, t1]", tag);
	const Frame FS = ReadVerb(fx, s);
	ExpectFrame(tag, "S0 schema kNone frame == prompt-twin frame, same tokens", FS, ref_s0, fx.hidden);

	// S1: kDfa with 1 consumed (the exit that sets ready_for_logits).
	{
		const int32_t illegal = ds.FirstIllegal(states[3]);
		CHECK_MSG(illegal >= 0, "[%s] S1 SETUP: an illegal token after t2", tag);
		const SslmGpuStatus st = schema_prefill({chain[2], illegal});
		CHECK_MSG(st == SSLM_SEQUENCE_REJECTED && consumed == 1,
		          "[%s] S1 SETUP: kDfa with 1 consumed (got %s, consumed=%d)", tag, StatusName(st), consumed);
		ExpectRefuse(tag, "S1 schema kDfa, 1 consumed", ReadVerb(fx, s));
	}
	// S2: first token unreachable -> refused before the pre-scan -> the prior frame survives.
	{
		CHECK_MSG(base_schema(), "[%s] S2 SETUP", tag);
		const int32_t illegal = ds.FirstIllegal(states[2]);
		const SslmGpuStatus st = schema_prefill({illegal});
		CHECK_MSG(st != SSLM_OK && consumed <= 0, "[%s] S2 SETUP: first token unreachable refused (got %s)", tag,
		          StatusName(st));
		ExpectFrame(tag, "S2 schema first token unreachable (before the pre-scan)", ReadVerb(fx, s), FS, fx.hidden);
	}
	// S3: a schema bound across reset; a successful schema prefill returns THAT call's frame.
	{
		std::vector<int32_t> p_t0 = P;
		p_t0.push_back(chain[0]);
		const Frame ref_pt0 = prompt_ref(p_t0);
		CHECK(sslm_gpu_seq_reset(ctx, s) == SSLM_OK);
		CHECK(SslmGpuSeqSetSchemaForG5Bridge(ctx, s, idx) == SSLM_OK);
		CHECK(Prefill(fx, s, P) == SSLM_OK);
		const SslmGpuStatus st = schema_prefill({chain[0]});
		CHECK_MSG(st == SSLM_OK && consumed == 1, "[%s] S3 SETUP: bound-across-reset schema prefill (%s)", tag,
		          StatusName(st));
		ExpectFrame(tag, "S3 schema bound across reset, SSLM_OK", ReadVerb(fx, s), ref_pt0, fx.hidden);
	}
	// S3b: schema explicitly unbound -> refused before the pre-scan -> the prior frame survives.
	{
		CHECK(sslm_gpu_seq_reset(ctx, s) == SSLM_OK);
		CHECK(SslmGpuSeqSetSchemaForG5Bridge(ctx, s, -1) == SSLM_OK);
		CHECK(Prefill(fx, s, P) == SSLM_OK);
		const Frame FP = OracleFromLive(fx, s);
		const SslmGpuStatus st = schema_prefill({chain[0]});
		CHECK_MSG(st != SSLM_OK, "[%s] S3b SETUP: an unbound schema prefill is refused (%s)", tag, StatusName(st));
		ExpectFrame(tag, "S3b schema unbound (before the pre-scan)", ReadVerb(fx, s), FP, fx.hidden);
	}
	// S4 / S5: kPositionCap on the cap-64 sequence, 2 and 0 admitted.
	{
		CHECK(sslm_gpu_seq_reset(ctx, s) == SSLM_OK);
		CHECK(SslmGpuSeqSetSchemaForG5Bridge(ctx, s, idx) == SSLM_OK);
		const std::vector<int32_t> fill62 = fx.Run(62, P[0]);
		CHECK(Prefill(fx, s, fill62) == SSLM_OK);
		const SslmGpuStatus st = schema_prefill(chain);
		CHECK_MSG(st != SSLM_OK && consumed == 2, "[%s] S4 SETUP: kPositionCap with 2 admitted (%s, consumed=%d)", tag,
		          StatusName(st), consumed);
		ExpectRefuse(tag, "S4 schema kPositionCap, 2 admitted", ReadVerb(fx, s));

		CHECK(sslm_gpu_seq_reset(ctx, s) == SSLM_OK);
		CHECK(SslmGpuSeqSetSchemaForG5Bridge(ctx, s, idx) == SSLM_OK);
		const std::vector<int32_t> fill64 = fx.Run(64, P[0]);
		CHECK(Prefill(fx, s, fill64) == SSLM_OK);
		const SslmGpuStatus st5 = schema_prefill({chain[0]});
		CHECK_MSG(st5 != SSLM_OK && consumed == 0, "[%s] S5 SETUP: kPositionCap with 0 admitted (%s, consumed=%d)", tag,
		          StatusName(st5), consumed);
		ExpectRefuse(tag, "S5 schema kPositionCap, 0 admitted", ReadVerb(fx, s));
	}
	// F5: the device override on the schema twin.
	{
		CHECK_MSG(base_schema(), "[%s] F5 SETUP", tag);
		superslm_gpu::ArmT2169ChunkRecordingFaultInjection(1);
		const SslmGpuStatus st = schema_prefill({chain[2], chain[3]});
		superslm_gpu::ClearT2169ChunkRecordingFaultInjection();
		// T-2814 (plan Sec3.4 row 5, T-2806 G1): tightened from `!= SSLM_OK`. The schema twin's device
		// override stays SSLM_DEVICE_LOST in 1.6.0 (Sec3.3, D-SLM7311).
		CHECK_MSG(st == SSLM_DEVICE_LOST, "[%s] F5: the armed override on the schema twin returned %s, want SSLM_DEVICE_LOST",
		          tag, StatusName(st));
		ExpectRefuse(tag, "F5 schema device-computed override", ReadVerb(fx, s));
	}
	sslm_gpu_seq_release(ctx, s);
}

void RunOn(const std::string& path, const char* flag, SslmGpuContext* ctx, bool schema) {
	if (path.empty()) {
		SKIP_MSG("prefill exits: %s not supplied", flag);
		return;
	}
	GpuModelFixture fx;
	if (!fx.Open(path, ctx)) {
		CHECK_MSG(false, "could not open %s", path.c_str());
		return;
	}
	PromptFaults(fx);
	if (schema) SchemaMembers(fx);
	fx.Close();
}

}  // namespace

int main(int argc, char** argv) {
	RunAsLegDriverIfRequested(argc, argv, "cell_prefill_faults_schema");
	ParseFixtureArgs(argc, argv);
	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
		std::printf("FATAL: sslm_gpu_context_create failed\n");
		return 2;
	}
	RunOn(g_qwen3_path, "--qwen3", ctx, false);
	RunOn(g_synthetic_path, "--synthetic", ctx, false);
	RunOn(g_g5_path, "--g5fixture", ctx, true);
	if (g_g5_path.empty()) SKIP_MSG("schema members S0-S5, F5 need --g5fixture=PATH");
	sslm_gpu_context_destroy(ctx);
	return FinishSuite("cell_prefill_faults_schema");
}
