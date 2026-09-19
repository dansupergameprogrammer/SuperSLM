// T-2801 (Curie) -- plan Sec3.4 rows 1, 5, 7 and 11 for plan Sec3.6: a device-side guard refusal in
// the prompt prefill is SSLM_SEQUENCE_REJECTED on a live device, not SSLM_DEVICE_LOST
// (D-SLM7282, TE266-Q4 option (c); fold record Claude/Vitruvius/t2798-te266-q4-fold-2026-09-18.md Sec5).
//
// FIXTURE G-an (make_gan_fixture.py): the synthetic fused-K fixture (--synthetic, the U1 pair model)
// with CompositionConstants "layer0.attn_norm" at mantissa -2^30, exponent unchanged, integrity hash
// re-sealed. Both hashes are pinned. The plan's starting value, the loader floor -2,147,483,647,
// trips every id and gives no split; -2^30 is the plan's own "adjust m" fallback.
//
// CPU ORACLE (O-CPU, the suite's EmbedEntry + RunLayerLoop per token, marshalled here from this cell's
// own SslmModel::Load parse; no input from the GPU path). SETUP, before any GPU cell:
//   - every vocabulary id is labelled by a one-token CPU forward on G-an: passes, or refuses at layer 0
//     with CarriedScaleMantissaOutOfDomain. The split must be 120 pass / 264 trip, no id may refuse at
//     a later layer, and the label vector's SHA-256 is pinned. The unpatched fixture labels all 384 pass.
//   - per cell, the CPU forward over the cell's exact ids must refuse at the intended index, at
//     layer 0, with a guard status (plan Sec3.4 row 5's setup precondition). Otherwise SETUP.
//
// CELLS (sequence cap 64; kT2169TdrSafeMaxChunkTokens = 4 splits an admitted chunk into sub-chunks of 4):
//   Q4-1a   [pass,pass,trip], refusal in the final (only) sub-chunk -> SSLM_SEQUENCE_REJECTED
//   Q4-1a+  the same call as a continuation of a successful 1-token prefill (the snapshot was valid
//           before the call) -> SSLM_SEQUENCE_REJECTED, and the read refuses
//   Q4-1b   [pass,trip,pass x4], refusal in the first of two sub-chunks (non-final)
//   Q4-1f   [trip,pass], the refused token is the FIRST of the final (only) sub-chunk
//   Q4-1g   [p,p,p,p,trip,p,p,p,p], the refused token is the FIRST of a NON-FINAL sub-chunk
//   Q4-1h   [p,p,p,p,trip,p], the refused token is the FIRST of the final of two sub-chunks
//   Each refusal cell asserts, from the header text plan Sec3.6 item 5 specifies: the status; the
//   committed prefix ("tokens before the refused one are committed; the refused token and every later
//   one are not"), read through the bench context-length accessor; and "the prefill snapshot is empty"
//   (the read returns SSLM_PREFILL_HIDDEN_UNAVAILABLE and writes nothing). The layer index is printed,
//   not asserted: the header leaves it unspecified.
//   Q4-1c   must-accept neighbour: every id list above on the unpatched fixture returns SSLM_OK and reads
//           the O-LIVE frame, which equals the CPU frame
//   Q4-1d   Q4-1a with superslm_gpu::ArmPrefillGuardDeviceRemovedQueryInjection() armed -> SSLM_DEVICE_LOST;
//           then Q4-1a unarmed -> SSLM_SEQUENCE_REJECTED again (the seam is single-shot). This is the
//           device-alive conjunct's one killer: Q4-1e's documents never trip a guard.
//   Q4-2    (i) the sequence that refused, reset and prefilled with passing ids, reads byte-equal to a
//           control read earlier in a context that saw no refusal (and to the CPU frame);
//           (ii) a sequence on the unpatched fixture, mapped into the same context after the refusals,
//           reads byte-equal to its fresh-context control;
//           (iii) both sequences release, both models unmap and the context destroys with SSLM_OK.
//
// GUARD VITALITY (plan Sec3.4 row 11): (h) the 1.5.0 unconditional SSLM_DEVICE_LOST turns every refusal
// cell red; (i) no device-alive conjunct turns Q4-1d red; (k) final-sub-chunk provenance only turns
// Q4-1b and Q4-1g red; (l) non-final provenance only turns Q4-1a, Q4-1a+, Q4-1f and Q4-1h red.
// (j) classification by value is killed by cell_prefill_faults_schema's tightened F1, not here.
//
// Red at v1.5.0 by LINK: LNK2019 on sslm_gpu_seq_read_prefill_final_hidden and on the seam.
//
// N3 (T-2806 audit; T-2814): plan Sec3.6 item 4, "ready_for_logits stays untouched" after the guard
// refusal. The bench bridge does not expose the bit; the observable is the NEXT decode-wrapper call,
// whose ready-bit shortcut finishes the residual without embedding (the context length does not grow)
// and whose other branch embeds the given token and drives it (the context length grows by one). After
// Q4-1a (bit clear before the call) and Q4-1a+ (bit set by the prefix) the cell calls
// SslmGpuSeqDecodeStepForG5Bridge(token 0) once and asserts the status, the context-length change and
// the token that v1.5.0 gives for the same calls on the same fixture -- read from the v1.5.0 engine by
// Claude/Curie/t2814-probe/n3_wrapper_v150.cpp, never from the build under test. At v1.5.0 this refusal
// is the device override, which leaves the bit untouched, the behaviour Sec3.6 item 4 keeps.
//
// *out_required (plan Sec3.4 row 7, T-2806 N4): every refused read also leaves *out_required at its
// sentinel -- plan Sec3.1 check 4 writes it only after checks 1-3 pass.
//
// Run: cell_prompt_guard_status.exe --synthetic=PATH --gan=PATH
#include "cpu_oracle.h"

// Plan Sec3.6 item 3's seam, beside the T-2169 seams and under their macro. Declared here as the
// verb is in fixture_common.h: the cell compiles at v1.5.0 and fails to link on this symbol; once
// gpu_port.h declares it, a signature that drifts from this one is a compile error.
namespace superslm_gpu {
void ArmPrefillGuardDeviceRemovedQueryInjection();
}

namespace {

constexpr const char* kA2Sha256 = "a231d9ed9dd3944a253201aa9be418fb3b7f7ad258f79e57cf39297dc8954f70";
constexpr const char* kGanSha256 = "cf48079cd3b50eb053c8f8cd57d4feb0f102d8ec9622aaf86300a15daf96e76b";
constexpr int kGanPassCount = 120;
constexpr int kGanTripCount = 264;
// SHA-256 over the 384 one-byte labels (1 = trips at layer 0, 0 = passes), id order. Derived by this
// cell's own CPU oracle, so it is a regression pin on the labelled population, not an independent
// oracle; the independent check is the counts, which reproduce the adversary's sweep (TE-280 Sec3),
// where the GPU agreed with the CPU on the class of every id.
constexpr const char* kGanLabelSha256 = "c698f61f492976c105a1d23fb842c8d7e0360e38ec4594901b8d86fbde0020e2";
constexpr int32_t kPassA = 0;
constexpr int32_t kPassB = 1;
constexpr int32_t kTrip = 3;

// O-CPU (CpuOracle), Ids and FileSha256 live in cpu_oracle.h, shared with cell_schema_guard_status.cpp.

// The whole-vocabulary label sweep (fold record Sec5 item 1). Returns false on a SETUP failure.
bool LabelSweep(GpuModelFixture& fx, CpuOracle& cpu, bool patched, std::vector<uint8_t>* labels) {
	labels->assign(static_cast<size_t>(fx.vocab), 0);
	int pass = 0, trip0 = 0, later = 0, other_status = 0;
	for (int32_t id = 0; id < fx.vocab; ++id) {
		const CpuResult r = cpu.Run({id});
		if (r.st == superslm::SslmForwardStatus::Ok) {
			++pass;
		} else if (r.refused_layer == 0) {
			++trip0;
			(*labels)[static_cast<size_t>(id)] = 1;
			if (r.st != superslm::SslmForwardStatus::CarriedScaleMantissaOutOfDomain) ++other_status;
		} else {
			++later;
		}
	}
	const std::string h = Sha256Hex(labels->data(), labels->size());
	std::printf("  CPU label sweep %s: pass %d, trip at layer 0 %d (other than CarriedScaleMantissaOutOfDomain: %d), "
	            "refused at a later layer %d, labels sha256 %s\n",
	            patched ? "G-an" : "unpatched", pass, trip0, other_status, later, h.c_str());
	if (patched) {
		std::string trips;
		for (int32_t id = 0; id < fx.vocab; ++id)
			if ((*labels)[static_cast<size_t>(id)]) trips += std::to_string(id) + " ";
		std::printf("  G-an tripping ids: %s\n", trips.c_str());
	}
	bool ok = true;
	if (patched) {
		CHECK_MSG(pass == kGanPassCount && trip0 == kGanTripCount && later == 0 && other_status == 0,
		          "SETUP: G-an's CPU label split is %d pass / %d trip / %d later / %d other-status, want %d / %d / 0 / 0",
		          pass, trip0, later, other_status, kGanPassCount, kGanTripCount);
		ok = pass == kGanPassCount && trip0 == kGanTripCount && later == 0 && other_status == 0;
		const bool pinned = std::string(kGanLabelSha256) == h;
		CHECK_MSG(pinned, "SETUP: G-an's CPU label vector sha256 %s, pinned %s", h.c_str(), kGanLabelSha256);
		ok = ok && pinned;
		const bool chosen = (*labels)[kPassA] == 0 && (*labels)[kPassB] == 0 && (*labels)[kTrip] == 1;
		CHECK_MSG(chosen, "SETUP: the cells' ids are not labelled pass %d, pass %d, trip %d", kPassA, kPassB, kTrip);
		ok = ok && chosen;
	} else {
		CHECK_MSG(pass == fx.vocab, "SETUP: the unpatched fixture's CPU sweep passes %d of %d ids", pass, fx.vocab);
		ok = pass == fx.vocab;
	}
	return ok;
}

struct Shape {
	const char* name;
	std::vector<int32_t> prefix;  // a successful prefill before the refusing call (empty: fresh reset)
	std::vector<int32_t> ids;     // the refusing call
	int trip_index;               // index within `ids`
	const char* where;
};

std::vector<Shape> Shapes() {
	const int32_t p = kPassA, q = kPassB, t = kTrip;
	return {
	    {"Q4-1a", {}, {p, q, t}, 2, "final (only) sub-chunk, non-first token"},
	    {"Q4-1a+", {p}, {p, q, t}, 2, "final (only) sub-chunk of a continuation after a valid snapshot"},
	    {"Q4-1b", {}, {p, t, p, q, p, q}, 1, "first of two sub-chunks (non-final), non-first token"},
	    {"Q4-1f", {}, {t, p}, 0, "FIRST token of the final (only) sub-chunk"},
	    {"Q4-1g", {}, {p, q, p, q, t, p, q, p, q}, 4, "FIRST token of a non-final sub-chunk (second of three)"},
	    {"Q4-1h", {}, {p, q, p, q, t, p}, 4, "FIRST token of the final of two sub-chunks"},
	};
}

// Per plan Sec3.4 row 5: the CPU forward over the call's exact ids (after the prefix) refuses at the
// intended token, at layer 0, with a guard status.
bool Precondition(CpuOracle& cpu, const Shape& sh) {
	std::vector<int32_t> all = sh.prefix;
	all.insert(all.end(), sh.ids.begin(), sh.ids.end());
	const CpuResult r = cpu.Run(all);
	const int want = static_cast<int>(sh.prefix.size()) + sh.trip_index;
	const bool ok = r.st != superslm::SslmForwardStatus::Ok && r.refused_index == want && r.refused_layer == 0;
	CHECK_MSG(ok, "SETUP %s %s: the CPU forward refused at index %d layer %u (%s), want index %d layer 0", sh.name,
	          Ids(all).c_str(), r.refused_index, r.refused_layer, superslm::SslmForwardStatusName(r.st), want);
	return ok;
}

// One refusal cell on the given sequence. `expect` is SSLM_SEQUENCE_REJECTED, or SSLM_DEVICE_LOST for Q4-1d.
void RefusalCell(GpuModelFixture& g, SslmGpuSequenceHandle* s, const Shape& sh, SslmGpuStatus expect,
                 const char* label) {
	CHECK_MSG(sslm_gpu_seq_reset(g.ctx, s) == SSLM_OK, "%s: reset", label);
	int64_t open = 0;
	if (!sh.prefix.empty()) {
		const SslmGpuStatus ps = Prefill(g, s, sh.prefix);
		CHECK_MSG(ps == SSLM_OK, "SETUP %s: the prefix prefill returned %s", label, StatusName(ps));
		const Frame before = ReadVerb(g, s);
		CHECK_MSG(before.SameFrame(OracleFromLive(g, s), g.hidden),
		          "SETUP %s: the prefix's frame did not read (%s); the snapshot was not valid before the call", label,
		          StatusName(before.status));
		open = ContextLength(s);
	}
	const SslmGpuStatus st = Prefill(g, s, sh.ids);
	const int64_t ctxlen = ContextLength(s);
	const uint32_t layer = LayerIndex(s);
	const Frame rd = ReadVerb(g, s);
	std::printf("    %-7s %-24s %s: prefill=%s committed=%lld (open %lld) layer_index=%u read=%s\n", label,
	            Ids(sh.ids).c_str(), sh.where, StatusName(st), static_cast<long long>(ctxlen),
	            static_cast<long long>(open), layer, StatusName(rd.status));
	CHECK_MSG(st == expect, "%s %s (%s): the prompt twin returned %s, want %s", label, Ids(sh.ids).c_str(), sh.where,
	          StatusName(st), StatusName(expect));
	CHECK_MSG(ctxlen == open + sh.trip_index,
	          "%s: committed prefix is %lld tokens, want %lld (the tokens before the refused one, and none after)",
	          label, static_cast<long long>(ctxlen), static_cast<long long>(open + sh.trip_index));
	CHECK_MSG(rd.status == kPrefillHiddenUnavailable && rd.OutputsUntouched(),
	          "%s: the read returned %s (untouched=%d), want SSLM_PREFILL_HIDDEN_UNAVAILABLE and nothing written", label,
	          StatusName(rd.status), rd.OutputsUntouched() ? 1 : 0);
	CHECK_MSG(rd.required == kRequiredSentinel, "%s: a check-3 refusal wrote *out_required (%zu)", label, rd.required);
}

// N3: the next decode-wrapper call after the refusal, against v1.5.0's reading of the same calls.
struct WrapperReading {
	int ctx_delta;      // +1: the wrapper embedded token 0 (bit clear); 0: the ready-bit shortcut (bit set)
	int32_t out_token;  // v1.5.0's token
};
// Claude/Curie/t2814-probe/n3-wrapper-v150-output.txt, G-an rows R0 and R1 (both SSLM_OK at v1.5.0).
constexpr WrapperReading kV150AfterQ41a = {1, 47};
constexpr WrapperReading kV150AfterQ41aPlus = {0, 3};

void WrapperAfterRefusal(GpuModelFixture& g, SslmGpuSequenceHandle* s, const WrapperReading& want, const char* label) {
	const int64_t before = ContextLength(s);
	int32_t tok = -7;
	const SslmGpuStatus st = SslmGpuSeqDecodeStepForG5Bridge(g.ctx, s, kPassA, g.one_layer_budget, &tok);
	const int64_t delta = ContextLength(s) - before;
	std::printf("    N3 %-7s next wrapper(0): %s out_token=%d ctxlen %+lld (v1.5.0: SSLM_OK, %d, %+d)\n", label,
	            StatusName(st), tok, static_cast<long long>(delta), want.out_token, want.ctx_delta);
	CHECK_MSG(st == SSLM_OK && delta == want.ctx_delta && tok == want.out_token,
	          "N3 %s: after the guard refusal the next decode-wrapper call gave %s, token %d, context %+lld; v1.5.0 gives "
	          "SSLM_OK, token %d, context %+d (ready_for_logits %s before the call must be left untouched, Sec3.6 item 4)",
	          label, StatusName(st), tok, static_cast<long long>(delta), want.out_token, want.ctx_delta,
	          want.ctx_delta == 0 ? "set" : "clear");
}

// A fresh-context control: map, prefill `ids`, read. The frame is compared with O-LIVE and O-CPU.
Frame ControlRead(const std::string& path, const std::vector<int32_t>& ids, const char* label) {
	Frame out;
	SslmGpuContext* c = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &c) != SSLM_OK || !c) {
		CHECK_MSG(false, "SETUP %s: control context create", label);
		return out;
	}
	{
		GpuModelFixture fx;
		if (fx.Open(path, c)) {
			CpuOracle cpu(fx);
			SslmGpuSequenceHandle* s = nullptr;
			CHECK(sslm_gpu_seq_create(c, fx.model, 64, &s) == SSLM_OK && s);
			if (s) {
				CHECK(sslm_gpu_seq_reset(c, s) == SSLM_OK);
				const SslmGpuStatus ps = Prefill(fx, s, ids);
				out = ReadVerb(fx, s);
				const CpuResult r = cpu.Run(ids);
				CHECK_MSG(ps == SSLM_OK && out.SameFrame(OracleFromLive(fx, s), fx.hidden) &&
				              out.SameFrame(r.frame, fx.hidden),
				          "SETUP %s: the control %s did not read the O-LIVE and O-CPU frame (%s, read %s)", label,
				          Ids(ids).c_str(), StatusName(ps), StatusName(out.status));
				sslm_gpu_seq_release(c, s);
			}
			fx.Close();
		} else {
			CHECK_MSG(false, "SETUP %s: control could not map %s", label, path.c_str());
		}
	}
	CHECK_MSG(sslm_gpu_context_destroy(c) == SSLM_OK, "SETUP %s: control context destroy", label);
	return out;
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_synthetic_path.empty() || g_gan_path.empty()) {
		SKIP_MSG("the prompt-twin guard-status cell needs --synthetic=PATH and --gan=PATH");
		return FinishSuite("cell_prompt_guard_status");
	}
	std::string h;
	const bool a2_ok = FileSha256(g_synthetic_path, &h) && h == kA2Sha256;
	CHECK_MSG(a2_ok, "SETUP: --synthetic is not the U1 pair model G-an is built from (sha256 %s, want %s)", h.c_str(),
	          kA2Sha256);
	const bool gan_ok = FileSha256(g_gan_path, &h) && h == kGanSha256;
	CHECK_MSG(gan_ok, "SETUP: --gan is not make_gan_fixture.py's output (sha256 %s, want %s)", h.c_str(), kGanSha256);
	if (!a2_ok || !gan_ok) return FinishSuite("cell_prompt_guard_status");

	const std::vector<int32_t> pass2 = {kPassA, kPassB};

	// Q4-2's controls, read first, each in its own context that sees no refusal.
	std::printf("Q4-2 controls (fresh contexts, no refusal)\n");
	const Frame control_gan = ControlRead(g_gan_path, pass2, "control G-an");
	// GpuModelFixture::TokensP() for the pinned vocabulary of 384 (vocab/7, vocab/3, vocab-1).
	const std::vector<int32_t> a2_ids = {384 / 7, 384 / 3, 384 - 1};
	const Frame control_a2 = ControlRead(g_synthetic_path, a2_ids, "control A2");

	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
		std::printf("FATAL: sslm_gpu_context_create failed\n");
		return 2;
	}
	GpuModelFixture g;
	if (!g.Open(g_gan_path, ctx)) {
		CHECK_MSG(false, "G-an did not load and map");
		sslm_gpu_context_destroy(ctx);
		return FinishSuite("cell_prompt_guard_status");
	}
	CpuOracle cpu_g(g);
	CHECK_MSG(cpu_g.ok(), "SETUP: CPU oracle marshal on G-an");
	std::vector<uint8_t> labels;
	const bool swept = LabelSweep(g, cpu_g, true, &labels);

	SslmGpuSequenceHandle* s = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, g.model, 64, &s) == SSLM_OK && s);
	if (swept && s) {
		std::printf("Q4-1 refusal cells on G-an\n");
		for (const Shape& sh : Shapes()) {
			if (!Precondition(cpu_g, sh)) continue;
			RefusalCell(g, s, sh, SSLM_SEQUENCE_REJECTED, sh.name);
			if (std::string(sh.name) == "Q4-1a") WrapperAfterRefusal(g, s, kV150AfterQ41a, sh.name);
			if (std::string(sh.name) == "Q4-1a+") WrapperAfterRefusal(g, s, kV150AfterQ41aPlus, sh.name);
		}
		std::printf("Q4-1d the device-alive conjunct (seam armed), then single-shot\n");
		const Shape q4a = Shapes()[0];
		if (Precondition(cpu_g, q4a)) {
			superslm_gpu::ArmPrefillGuardDeviceRemovedQueryInjection();
			RefusalCell(g, s, q4a, SSLM_DEVICE_LOST, "Q4-1d");
			RefusalCell(g, s, q4a, SSLM_SEQUENCE_REJECTED, "Q4-1d'");
		}

		std::printf("Q4-2 (i) the refused sequence, reset, passing ids\n");
		CHECK(sslm_gpu_seq_reset(ctx, s) == SSLM_OK);
		const SslmGpuStatus ps = Prefill(g, s, pass2);
		const Frame reused = ReadVerb(g, s);
		const CpuResult cr = cpu_g.Run(pass2);
		std::printf("    reused %s: prefill=%s read=%s ==control=%d ==CPU=%d\n", Ids(pass2).c_str(), StatusName(ps),
		            StatusName(reused.status), reused.SameFrame(control_gan, g.hidden) ? 1 : 0,
		            reused.SameFrame(cr.frame, g.hidden) ? 1 : 0);
		CHECK_MSG(ps == SSLM_OK && reused.SameFrame(control_gan, g.hidden),
		          "Q4-2 (i): the reused sequence read %s, differing from the fresh-context control in %zu codes",
		          StatusName(reused.status), DiffCodes(reused, control_gan, g.hidden));
		CHECK_MSG(reused.SameFrame(cr.frame, g.hidden), "Q4-2 (i): the reused sequence's frame differs from O-CPU");
	}

	std::printf("Q4-2 (ii) + Q4-1c: the unpatched fixture mapped into the same context after the refusals\n");
	GpuModelFixture a;
	const bool a_ok = a.Open(g_synthetic_path, ctx);
	CHECK_MSG(a_ok, "Q4-2 (ii): the unpatched fixture did not map into the context after the refusals");
	SslmGpuSequenceHandle* s2 = nullptr;
	if (a_ok) {
		CpuOracle cpu_a(a);
		std::vector<uint8_t> a_labels;
		LabelSweep(a, cpu_a, false, &a_labels);
		CHECK(sslm_gpu_seq_create(ctx, a.model, 64, &s2) == SSLM_OK && s2);
		if (s2) {
			CHECK(sslm_gpu_seq_reset(ctx, s2) == SSLM_OK);
			const SslmGpuStatus ps = Prefill(a, s2, a2_ids);
			const Frame f2 = ReadVerb(a, s2);
			std::printf("    A2 %s: prefill=%s read=%s ==control=%d\n", Ids(a2_ids).c_str(), StatusName(ps),
			            StatusName(f2.status), f2.SameFrame(control_a2, a.hidden) ? 1 : 0);
			CHECK_MSG(ps == SSLM_OK && f2.SameFrame(control_a2, a.hidden),
			          "Q4-2 (ii): the second model's sequence read %s, differing from its fresh-context control in %zu codes",
			          StatusName(f2.status), DiffCodes(f2, control_a2, a.hidden));
			// Q4-1c: every refusal shape's ids on the unpatched fixture pass and read their frame.
			for (const Shape& sh : Shapes()) {
				std::vector<int32_t> all = sh.prefix;
				all.insert(all.end(), sh.ids.begin(), sh.ids.end());
				const CpuResult r = cpu_a.Run(all);
				CHECK_MSG(r.st == superslm::SslmForwardStatus::Ok,
				          "SETUP Q4-1c %s: the CPU forward refused on the unpatched fixture", sh.name);
				CHECK(sslm_gpu_seq_reset(ctx, s2) == SSLM_OK);
				SslmGpuStatus st = SSLM_OK;
				if (!sh.prefix.empty()) st = Prefill(a, s2, sh.prefix);
				if (st == SSLM_OK) st = Prefill(a, s2, sh.ids);
				const Frame want = OracleFromLive(a, s2);
				const Frame got = ReadVerb(a, s2);
				std::printf("    Q4-1c %-7s %-24s prefill=%s read=%s ==O-LIVE=%d ==CPU=%d\n", sh.name,
				            Ids(sh.ids).c_str(), StatusName(st), StatusName(got.status),
				            got.SameFrame(want, a.hidden) ? 1 : 0, got.SameFrame(r.frame, a.hidden) ? 1 : 0);
				CHECK_MSG(st == SSLM_OK && got.SameFrame(want, a.hidden) && got.SameFrame(r.frame, a.hidden),
				          "Q4-1c %s (must-accept) on the unpatched fixture: prefill %s, read %s", sh.name,
				          StatusName(st), StatusName(got.status));
			}
		}
	}

	std::printf("Q4-2 (iii) teardown\n");
	if (s2) CHECK_MSG(sslm_gpu_seq_release(ctx, s2) == SSLM_OK, "Q4-2 (iii): release the unpatched fixture's sequence");
	if (s) CHECK_MSG(sslm_gpu_seq_release(ctx, s) == SSLM_OK, "Q4-2 (iii): release the refused sequence");
	if (a_ok) {
		CHECK_MSG(sslm_gpu_model_unmap(ctx, a.model) == SSLM_OK, "Q4-2 (iii): unmap the unpatched fixture");
		a.model = nullptr;
	}
	CHECK_MSG(sslm_gpu_model_unmap(ctx, g.model) == SSLM_OK, "Q4-2 (iii): unmap G-an");
	g.model = nullptr;
	CHECK_MSG(sslm_gpu_context_destroy(ctx) == SSLM_OK, "Q4-2 (iii): context destroy (no wedged submitted window)");
	return FinishSuite("cell_prompt_guard_status");
}
