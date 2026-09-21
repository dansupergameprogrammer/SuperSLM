// T-2814 (Curie) -- plan Sec3.4 rows 5, 7 and 11, cell Q5-1 (T-2806 G1): the schema twin's device-guard
// refusal stays SSLM_DEVICE_LOST in 1.6.0 (Sec3.3; D-SLM7311, TE266-Q5 "keep"). Sec3.6 moves only the
// PROMPT twin's guard refusal to SSLM_SEQUENCE_REJECTED; the schema twin already returns
// SSLM_SEQUENCE_REJECTED from its kDfa exit ("tokens landed, ready bit set, carry on decoding"), so a guard
// refusal reported the same way would go down the continue-decoding path.
//
// FIXTURE G5-an (make_g5an_fixture.py): the G5 suite's schema-bearing artifact (--g5fixture,
// t2132_g5_fixture_1p5b.sslm) with CompositionConstants "layer0.attn_norm" at mantissa -2^28, exponent
// unchanged, integrity hash re-sealed. Both hashes are pinned. The plan's first candidate, the C39
// synthetic, cannot host this cell: its one schema has 2 states and 1 transition, so no two-token legal
// chain exists on it at any mantissa (Claude/Curie/t2814-api-and-gap-cells-2026-09-18.md).
//
// CPU ORACLE (O-CPU, cpu_oracle.h: EmbedEntry + RunLayerLoop per token over layers marshalled from this
// cell's own parse; no input from the GPU path). SETUP, before any GPU call:
//   - LABEL SWEEP over the DFA-legal ids a [t0, t1] chain can use: every id legal at the schema's start
//     state, and every id legal one step after each of them (the walk is an independent SchemaMasks parse,
//     fixture_common.h IndependentSchema). Each id is labelled by a one-token CPU forward over layer 0
//     only: passes, or refuses at layer 0. A [pass, trip] candidate is t0 passing and t1 tripping; each
//     candidate is then run at full depth as the two-token sequence, and it is a chain only if t0 passes
//     all layers and t1 refuses at index 1, layer 0. The counts are pinned (kLegalStart*, kCandidates),
//     and the cell's chain [4913, 72] must be the one found. The counts are this oracle's own, so they are
//     a regression pin on the population; the independent reading is the scratch probe q5_sweep, a
//     separate program that found the same single candidate (Claude/Curie/t2814-probe/).
//   - PRECONDITION: the CPU forward over P ++ [4913, 72] refuses at index |P|+1, layer 0, and over
//     P ++ [4913] passes. Otherwise the cell fails as SETUP.
//
// CELLS (sequence cap 64, schema "shopkeeper_intent_extraction", P = [0, 1, 3], which passes):
//   Q5-1   G5-an: reset; prompt P (the read returns P's frame, so the refusal below is discriminating);
//          bind the schema; schema-prefill [4913, 72]. Asserts exactly SSLM_DEVICE_LOST, *consumed == 1,
//          the context length |P| + 1, and that the read refuses (SSLM_PREFILL_HIDDEN_UNAVAILABLE, nothing
//          written, *out_required untouched).
//   Q5-1c  must-accept: the same calls on the unpatched --g5fixture return SSLM_OK with *consumed == 2 and
//          read the O-LIVE frame.
//
// GUARD VITALITY (plan Sec3.4 row 11): mutant (m), Sec3.6 item 3's classification also applied to the
// schema twin's override, turns Q5-1 red by status (SSLM_SEQUENCE_REJECTED where SSLM_DEVICE_LOST is
// required). cell_prefill_faults_schema's F5, tightened to == SSLM_DEVICE_LOST, catches a leak that
// classifies by value.
//
// Red at v1.5.0 by LINK: LNK2019 on sslm_gpu_seq_read_prefill_final_hidden.
//
// Run: cell_schema_guard_status.exe --g5fixture=PATH --g5an=PATH [--schema=NAME]
#include "cpu_oracle.h"

namespace {

constexpr const char* kG5Sha256 = "078df885060d5dea23a88983bb68014843d142cb6ad55c7f70ef9ff9a932a019";
constexpr const char* kG5anSha256 = "05b5d5c58ea14dbb15a3adb7f24668edcd4d7ab83344bc4381c5f7347b5a128b";
constexpr int64_t kSeqCap = 64;
constexpr int32_t kChainPass = 4913;
constexpr int32_t kChainTrip = 72;
// The label sweep's pinned population on G5-an (see the header). Filled from the reference run.
constexpr int kLegalStart = 2;          // ids legal at the start state
constexpr int kLegalStartPass = 2;      // of which pass layer 0
constexpr int kCandidates = 1;          // [pass, trip] candidates by layer-0 label, over every legal t0
constexpr int kChains = 1;              // candidates confirmed at full depth

std::vector<int32_t> LegalAt(const IndependentSchema& ds, uint32_t state) {
	std::vector<int32_t> out;
	for (int32_t t = 0; t < static_cast<int32_t>(ds.view.config.vocab_size); ++t) {
		if (ds.Next(state, t, nullptr)) out.push_back(t);
	}
	return out;
}

// Returns false on a SETUP failure.
bool LabelSweep(GpuModelFixture& g, CpuOracle& cpu, const IndependentSchema& ds) {
	const std::vector<int32_t> start = LegalAt(ds, 0);
	int start_pass = 0, candidates = 0, chains = 0;
	bool chain_found = false;
	std::string found;
	for (int32_t t0 : start) {
		const CpuResult r0 = cpu.Run({t0}, 1);
		const bool p0 = r0.st == superslm::SslmForwardStatus::Ok;
		start_pass += p0 ? 1 : 0;
		uint32_t s1 = 0;
		ds.Next(0, t0, &s1);
		const std::vector<int32_t> next = LegalAt(ds, s1);
		int np = 0, nt = 0;
		for (int32_t t1 : next) {
			const CpuResult r1 = cpu.Run({t1}, 1);
			if (r1.st == superslm::SslmForwardStatus::Ok) { ++np; continue; }
			++nt;
			if (!p0) continue;
			++candidates;
			const CpuResult rc = cpu.Run({t0, t1});
			if (rc.st != superslm::SslmForwardStatus::Ok && rc.refused_index == 1 && rc.refused_layer == 0) {
				++chains;
				found += Ids({t0, t1}) + "(" + superslm::SslmForwardStatusName(rc.st) + ") ";
				if (t0 == kChainPass && t1 == kChainTrip) chain_found = true;
			}
		}
		std::printf("  start-state id %d: layer 0 %s; %zu ids legal after it: %d pass, %d trip at layer 0\n", t0,
		            p0 ? "passes" : "trips", next.size(), np, nt);
	}
	std::printf("  label sweep: %zu legal at the start state (%d pass), [pass,trip] candidates %d, chains %d: %s\n",
	            start.size(), start_pass, candidates, chains, found.c_str());
	(void)g;
	const bool ok = static_cast<int>(start.size()) == kLegalStart && start_pass == kLegalStartPass &&
	                candidates == kCandidates && chains == kChains && chain_found;
	CHECK_MSG(ok,
	          "SETUP: G5-an's label sweep gives %zu legal / %d pass at the start state, %d candidates, %d chains "
	          "(chain [%d,%d] %s); pinned %d / %d / %d / %d with that chain",
	          start.size(), start_pass, candidates, chains, kChainPass, kChainTrip, chain_found ? "found" : "not found",
	          kLegalStart, kLegalStartPass, kCandidates, kChains);
	return ok;
}

struct SchemaRun {
	SslmGpuStatus prompt = SSLM_DEVICE_LOST;
	Frame before;  // the read after the prompt, before the schema call
	Frame live_before;
	SslmGpuStatus bind = SSLM_DEVICE_LOST;
	SslmGpuStatus st = SSLM_DEVICE_LOST;
	int32_t consumed = -1;
	int64_t ctxlen = -1;
	Frame after;
	Frame live_after;
};

SchemaRun Drive(GpuModelFixture& fx, SslmGpuSequenceHandle* s, int32_t idx, const std::vector<int32_t>& P,
                const std::vector<int32_t>& chain) {
	SchemaRun r;
	CHECK_MSG(sslm_gpu_seq_reset(fx.ctx, s) == SSLM_OK, "[%s] reset", fx.path.c_str());
	r.prompt = Prefill(fx, s, P);
	r.live_before = OracleFromLive(fx, s);
	r.before = ReadVerb(fx, s);
	r.bind = SslmGpuSeqSetSchemaForG5Bridge(fx.ctx, s, idx);
	r.st = SslmGpuSeqPrefillSchemaContentForG5Bridge(fx.ctx, s, chain.data(), static_cast<int32_t>(chain.size()),
	                                                 fx.one_layer_budget, &r.consumed);
	r.ctxlen = ContextLength(s);
	r.live_after = OracleFromLive(fx, s);
	r.after = ReadVerb(fx, s);
	std::printf("    [%s] prompt %s=%s read=%s; bind=%s; schema %s: %s consumed=%d ctxlen=%lld layer=%u read=%s\n",
	            fx.path.c_str(), Ids(P).c_str(), StatusName(r.prompt), StatusName(r.before.status),
	            StatusName(r.bind), Ids(chain).c_str(), StatusName(r.st), r.consumed,
	            static_cast<long long>(r.ctxlen), LayerIndex(s), StatusName(r.after.status));
	return r;
}

}  // namespace

int main(int argc, char** argv) {
	RunAsLegDriverIfRequested(argc, argv, "cell_schema_guard_status");
	ParseFixtureArgs(argc, argv);
	if (g_g5_path.empty() || g_g5an_path.empty()) {
		SKIP_MSG("Q5-1 needs --g5fixture=PATH and --g5an=PATH");
		return FinishSuite("cell_schema_guard_status");
	}
	std::string h;
	const bool g5_ok = FileSha256(g_g5_path, &h) && h == kG5Sha256;
	CHECK_MSG(g5_ok, "SETUP: --g5fixture is not the artifact G5-an is built from (sha256 %s, want %s)", h.c_str(),
	          kG5Sha256);
	const bool an_ok = FileSha256(g_g5an_path, &h) && h == kG5anSha256;
	CHECK_MSG(an_ok, "SETUP: --g5an is not make_g5an_fixture.py's output (sha256 %s, want %s)", h.c_str(), kG5anSha256);
	if (!g5_ok || !an_ok) return FinishSuite("cell_schema_guard_status");

	const std::vector<int32_t> P = {0, 1, 3};
	const std::vector<int32_t> chain = {kChainPass, kChainTrip};

	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
		std::printf("FATAL: sslm_gpu_context_create failed\n");
		return 2;
	}

	// ---- Q5-1 on G5-an ------------------------------------------------------------------------
	{
		GpuModelFixture g;
		if (!g.Open(g_g5an_path, ctx)) {
			CHECK_MSG(false, "G5-an did not load and map");
		} else {
			IndependentSchema ds;
			const bool schema_ok = ds.Build(g.bytes, g_schema_name);
			CHECK_MSG(schema_ok, "SETUP: no schema '%s' in an independent parse of G5-an", g_schema_name.c_str());
			const int32_t idx = SslmGpuSchemaLookupForG5Bridge(g.model, g_schema_name.c_str());
			CHECK_MSG(idx >= 0, "SETUP: schema lookup on G5-an");
			CpuOracle cpu(g, kSeqCap);
			CHECK_MSG(cpu.ok(), "SETUP: CPU oracle marshal on G5-an");
			std::printf("Q5-1 label sweep on G5-an (schema '%s')\n", g_schema_name.c_str());
			bool ready = schema_ok && idx >= 0 && cpu.ok() && LabelSweep(g, cpu, ds);
			if (ready) {
				std::vector<int32_t> head = P;
				head.push_back(kChainPass);
				std::vector<int32_t> all = head;
				all.push_back(kChainTrip);
				const CpuResult rh = cpu.Run(head);
				const CpuResult ra = cpu.Run(all);
				const bool pre = rh.st == superslm::SslmForwardStatus::Ok && ra.st != superslm::SslmForwardStatus::Ok &&
				                 ra.refused_index == static_cast<int>(P.size()) + 1 && ra.refused_layer == 0;
				CHECK_MSG(pre, "SETUP Q5-1: the CPU forward over %s refused at index %d layer %u (%s), and over %s %s; "
				          "want a refusal at index %zu layer 0, and a pass",
				          Ids(all).c_str(), ra.refused_index, ra.refused_layer, superslm::SslmForwardStatusName(ra.st),
				          Ids(head).c_str(), rh.st == superslm::SslmForwardStatus::Ok ? "passed" : "refused",
				          P.size() + 1);
				ready = pre;
			}
			SslmGpuSequenceHandle* s = nullptr;
			CHECK(sslm_gpu_seq_create(ctx, g.model, kSeqCap, &s) == SSLM_OK && s);
			if (ready && s) {
				std::printf("Q5-1 the schema twin's guard refusal\n");
				const SchemaRun r = Drive(g, s, idx, P, chain);
				CHECK_MSG(r.prompt == SSLM_OK && r.before.SameFrame(r.live_before, g.hidden) && r.bind == SSLM_OK,
				          "SETUP Q5-1: prompt %s, read before %s, bind %s -- the snapshot must be valid before the call",
				          StatusName(r.prompt), StatusName(r.before.status), StatusName(r.bind));
				CHECK_MSG(r.st == SSLM_DEVICE_LOST,
				          "Q5-1: the schema twin's device-guard refusal returned %s, want SSLM_DEVICE_LOST (Sec3.3, "
				          "D-SLM7311: Sec3.6 changes the prompt twin only)",
				          StatusName(r.st));
				CHECK_MSG(r.consumed == 1, "Q5-1: *consumed is %d, want 1 (the token before the refused one)", r.consumed);
				CHECK_MSG(r.ctxlen == static_cast<int64_t>(P.size()) + 1,
				          "Q5-1: the context length is %lld, want %zu (P plus the one token before the refused one)",
				          static_cast<long long>(r.ctxlen), P.size() + 1);
				CHECK_MSG(r.after.status == kPrefillHiddenUnavailable && r.after.OutputsUntouched() &&
				              r.after.required == kRequiredSentinel,
				          "Q5-1: the read returned %s (untouched=%d, required=%zu), want SSLM_PREFILL_HIDDEN_UNAVAILABLE "
				          "and nothing written",
				          StatusName(r.after.status), r.after.OutputsUntouched() ? 1 : 0, r.after.required);
			}
			if (s) sslm_gpu_seq_release(ctx, s);
			g.Close();
		}
	}

	// ---- Q5-1c: must-accept on the unpatched fixture --------------------------------------------
	{
		GpuModelFixture a;
		if (!a.Open(g_g5_path, ctx)) {
			CHECK_MSG(false, "--g5fixture did not load and map");
		} else {
			const int32_t idx = SslmGpuSchemaLookupForG5Bridge(a.model, g_schema_name.c_str());
			CHECK_MSG(idx >= 0, "SETUP: schema lookup on --g5fixture");
			SslmGpuSequenceHandle* s = nullptr;
			CHECK(sslm_gpu_seq_create(ctx, a.model, kSeqCap, &s) == SSLM_OK && s);
			if (s && idx >= 0) {
				std::printf("Q5-1c must-accept on the unpatched fixture\n");
				const SchemaRun r = Drive(a, s, idx, P, chain);
				CHECK_MSG(r.prompt == SSLM_OK && r.bind == SSLM_OK && r.st == SSLM_OK && r.consumed == 2 &&
				              r.after.SameFrame(r.live_after, a.hidden),
				          "Q5-1c (must-accept): prompt %s, bind %s, schema %s consumed %d, read %s; want SSLM_OK, 2 "
				          "consumed and the O-LIVE frame",
				          StatusName(r.prompt), StatusName(r.bind), StatusName(r.st), r.consumed,
				          StatusName(r.after.status));
			}
			if (s) sslm_gpu_seq_release(ctx, s);
			a.Close();
		}
	}
	CHECK_MSG(sslm_gpu_context_destroy(ctx) == SSLM_OK, "context destroy");
	return FinishSuite("cell_schema_guard_status");
}
