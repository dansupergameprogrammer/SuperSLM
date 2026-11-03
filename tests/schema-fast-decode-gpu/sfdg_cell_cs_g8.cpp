// Engine 1.19 clean stop, cell G8 (sfdg_cleanstop.h has the contract): a non-accepting dead end. The degenerate-row
// seam (ArmGpuFinishDegenerateLogitRowInjection: the next finish row is all INT32_MIN) at a state that is not
// accepting and does not admit token 0 gives -2 at SSLM_OK with accepting 0 on both queries: the defect the caller
// reports as an error, told apart from the end by the accepting read alone.
//   One row   (as G8.1) the decode bridge's ready finish after the prompt: -2, accepting 0, the walk and context
//             unchanged, ready re-armed; the seam is single-shot, so the next decode gives the unarmed token.
//   Multi-row an EXACT_DRAFT chunk [o_0, o_1, o_2] with o_0 in hand, the seam armed before the close: row 0's finish
//             dead-ends, so the close commits row 0 only (last_rows_committed 1) and rows_produced is -2 at SSLM_OK,
//             with accepting 0, the walk o_0's state, ready re-armed; the next decode gives o_1.
// The multi-row arm maps the device head (EXACT_DRAFT needs it); the one-row arm runs on both heads. Needs
// SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION. Box only. Run: sfdg_cell_cs_g8.exe [--fixture-dir=DIR]
#include "sfdg_cleanstop.h"

extern "C" void ArmGpuFinishDegenerateLogitRowInjection();

using namespace sfdg;
using namespace sfdg::cs;

namespace {

void OneRow(const char* label, GRig& rig, SslmGpuModelHandle* m, const superslm::SslmModelView& view,
            const std::vector<int32_t>& prompt) {
	GSeq ref = rig.New(kPotion, m);
	if (!ref.h || rig.Prompt(ref, prompt) != GS::SSLM_OK) return;
	GS st = GS::SSLM_OK;
	const int32_t want = Token(rig.env, ref, &st);
	GSeq s = rig.New(kPotion, m);
	if (!s.h || rig.Prompt(s, prompt) != GS::SSLM_OK) return;
	const uint32_t walk0 = SslmGpuSeqWalkStateForG5Bridge(s.h);
	const int64_t ctx0 = CtxLen(s.h);
	CHECK_MSG(Accepting(rig.ctx(), s.h) == 0 && !LegalAt(view, kPotion, walk0, 0), "[CS.G8 one row, %s] precondition: "
	          "state %u accepting %d, token 0 %s", label, walk0, Accepting(rig.ctx(), s.h),
	          LegalAt(view, kPotion, walk0, 0) ? "legal" : "illegal");
	ArmGpuFinishDegenerateLogitRowInjection();
	int32_t t = -7;
	st = SslmGpuSeqDecodeStepForG5Bridge(rig.ctx(), s.h, s.last, rig.env.full_budget, &t);
	CHECK_MSG(st == GS::SSLM_OK && t == -2, "[CS.G8 one row, %s] armed finish: %s token %d, want OK and -2", label, Name(st),
	          t);
	CHECK_MSG(Accepting(rig.ctx(), s.h) == 0, "[CS.G8 one row, %s] -2 with accepting %d, want 0 (a dead end)", label,
	          Accepting(rig.ctx(), s.h));
	CHECK_MSG(SslmGpuSeqWalkStateForG5Bridge(s.h) == walk0 && CtxLen(s.h) == ctx0 && ReadyFlag(rig.ctx(), s.h) == 1,
	          "[CS.G8 one row, %s] walk %u (was %u), context %lld (was %lld), ready %d", label,
	          SslmGpuSeqWalkStateForG5Bridge(s.h), walk0, static_cast<long long>(CtxLen(s.h)), static_cast<long long>(ctx0),
	          ReadyFlag(rig.ctx(), s.h));
	t = -7;
	st = SslmGpuSeqDecodeStepForG5Bridge(rig.ctx(), s.h, s.last, rig.env.full_budget, &t);
	CHECK_MSG(st == GS::SSLM_OK && t == want && want >= 0, "[CS.G8 one row, %s] the decode after (seam spent): %s token %d, "
	          "want %d", label, Name(st), t, want);
}

void MultiRow(GRig& rig, SslmGpuModelHandle* m, const superslm::SslmModelView& view, const std::vector<int32_t>& prompt) {
	const Off off = RunOff(rig, m, kPotion, prompt);
	CHECK_MSG(off.ok && off.toks.size() >= 3, "[CS.G8 multi-row] the Off run: ok %d, %zu tokens", off.ok, off.toks.size());
	if (!off.ok || off.toks.size() < 3) return;
	GSeq s = After(rig, m, kPotion, prompt, 1);  // o_0 in hand
	if (!s.h) return;
	const uint32_t walk0 = SslmGpuSeqWalkStateForG5Bridge(s.h);
	const int64_t ctx0 = CtxLen(s.h);
	CHECK_MSG(Accepting(rig.ctx(), s.h) == 0 && !LegalAt(view, kPotion, walk0, 0), "[CS.G8 multi-row] precondition: "
	          "state %u accepting %d, token 0 %s", walk0, Accepting(rig.ctx(), s.h),
	          LegalAt(view, kPotion, walk0, 0) ? "legal" : "illegal");
	const std::vector<int32_t> rows = {off.toks[0], off.toks[1], off.toks[2]};
	int32_t admitted = -9;
	const GS es = EmbedRows(rig, s.h, kExactDraft, rows, &admitted);
	CHECK_MSG(es == GS::SSLM_OK && admitted == 3, "[CS.G8 multi-row] embed_rows %s, admitted %d of 3", Name(es), admitted);
	if (es != GS::SSLM_OK) return;
	ArmGpuFinishDegenerateLogitRowInjection();
	const GS cs = CloseChunk(rig, s.h, rig.env.full_budget);
	int32_t tok = -77;
	const GS ps = sslm_gpu_seq_rows_produced(s.h, &tok);
	CHECK_MSG(cs == GS::SSLM_OK && ps == GS::SSLM_OK && tok == -2, "[CS.G8 multi-row] close %s, rows_produced %s token %d "
	          "(want OK, OK, -2)", Name(cs), Name(ps), tok);
	CHECK_MSG(LastCommitted(s.h) == 1, "[CS.G8 multi-row] last_rows_committed %d, want 1 (row 0 dead-ended)",
	          LastCommitted(s.h));
	CHECK_MSG(Accepting(rig.ctx(), s.h) == 0, "[CS.G8 multi-row] -2 with accepting %d, want 0 (a dead end)",
	          Accepting(rig.ctx(), s.h));
	CHECK_MSG(SslmGpuSeqWalkStateForG5Bridge(s.h) == walk0 && ReadyFlag(rig.ctx(), s.h) == 1 && CtxLen(s.h) == ctx0 + 1,
	          "[CS.G8 multi-row] walk %u (want %u), ready %d, context %lld (want %lld)", SslmGpuSeqWalkStateForG5Bridge(s.h),
	          walk0, ReadyFlag(rig.ctx(), s.h), static_cast<long long>(CtxLen(s.h)), static_cast<long long>(ctx0 + 1));
	s.ready = true;
	GS st = GS::SSLM_OK;
	const int32_t t = Token(rig.env, s, &st);
	CHECK_MSG(st == GS::SSLM_OK && t == off.toks[1], "[CS.G8 multi-row] the decode after (seam spent): %s token %d, Off's %d",
	          Name(st), t, off.toks[1]);
}

}  // namespace

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		const std::string dir = FixtureDir(argc, argv);
		CRig c(rig.raw);
		const std::vector<int32_t> prompt = c.Tokenize(kPromptPotionText);
		Mapped host, dev;
		if (MapVariant(rig, dir + "/fsfd.sslm", 0, &host)) OneRow("host head", rig, host.m, host.view, prompt);
		if (MapVariant(rig, dir + "/fsfd.sslm", SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE, &dev)) {
			OneRow("device head", rig, dev.m, dev.view, prompt);
			MultiRow(rig, dev.m, dev.view, prompt);
		}
		rig.ReleaseSeqs();
		Unmap(rig, &host);
		Unmap(rig, &dev);
	}
	return Finish("sfdg_cell_cs_g8");
}
