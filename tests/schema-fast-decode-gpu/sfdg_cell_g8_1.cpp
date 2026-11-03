// Schema fast-decode plan rev 1.4.4 §11.2, cell G8.1 (slice G1-M): the degenerate-row seam under the legal-row finish.
// ArmGpuFinishDegenerateLogitRowInjection presents an all-INT32_MIN row after the finish's logits. Under a bound
// schema the GPU finish now scores only the walk state's legal rows (S-GROWS shows n rows), and the seam still gives
// the -2 dead end: SSLM_OK, token -2, the walk state unchanged, the ready flag re-armed. The seam is single-shot, so
// the next decode call finishes the same hidden state and gives the token an unarmed sequence gives.
//   Arm 1  the head on the host (flags 0).
//   Arm 2  the head on the device, tied (rev 1.4.1): the bound finish runs on the host from embed_weights.
// The state must not admit token 0 (the masked argmax of an all-INT32_MIN row), or the dead end is a transition;
// the cell checks that from the fixture's mask page. Needs SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION.
// Box only. Needs fsfd.sslm. Run: sfdg_cell_g8_1.exe [--fixture-dir=DIR]
#include "sfdg_helpers.h"

extern "C" void ArmGpuFinishDegenerateLogitRowInjection();

using namespace sfdg;

namespace {

void Arm(const char* label, GRig& rig, SslmGpuModelHandle* m, const superslm::SslmModelView& view,
         const std::vector<int32_t>& prompt) {
	GSeq ref = rig.New(kPotion, m);
	if (!ref.h) return;
	rig.Prompt(ref, prompt);
	GS st = GS::SSLM_OK;
	const int32_t want = Token(rig.env, ref, &st);
	CHECK_MSG(st == GS::SSLM_OK && want >= 0, "[G8.1 %s] reference token: %s %d", label, Name(st), want);

	GSeq s = rig.New(kPotion, m);
	if (!s.h) return;
	rig.Prompt(s, prompt);
	const uint32_t walk0 = SslmGpuSeqWalkStateForG5Bridge(s.h);
	const uint32_t n = LegalCount(view, kPotion, walk0);
	CHECK_MSG(n >= 1 && !LegalAt(view, kPotion, walk0, 0),
	          "[G8.1 %s] precondition: state %u has %u legal rows and token 0 %s", label, walk0, n,
	          LegalAt(view, kPotion, walk0, 0) ? "legal" : "illegal");

	ArmGpuFinishDegenerateLogitRowInjection();
	sslm_gpu_test_only_rows_log_reset();
	int32_t tok = -7;
	st = SslmGpuSeqDecodeStepForG5Bridge(rig.ctx(), s.h, s.last, rig.env.full_budget, &tok);
	const std::vector<RowsEntry> l = RowsLog();
	CHECK_MSG(st == GS::SSLM_OK && tok == -2, "[G8.1 %s] armed finish: %s token %d (want OK, -2)", label, Name(st), tok);
	CHECK_MSG(SslmGpuSeqWalkStateForG5Bridge(s.h) == walk0, "[G8.1 %s] the dead end moved the walk", label);
	CHECK_MSG(!l.empty() && l.back().rows == n && l.back().device == 0,
	          "[G8.1 %s] armed finish S-GROWS: rows %llu device %d (want the %u legal rows on the host)", label,
	          l.empty() ? 0ull : static_cast<unsigned long long>(l.back().rows), l.empty() ? -1 : l.back().device, n);

	// Ready re-armed: the next call finishes the same hidden state without an embed, and the seam is spent.
	tok = -7;
	st = SslmGpuSeqDecodeStepForG5Bridge(rig.ctx(), s.h, s.last, rig.env.full_budget, &tok);
	CHECK_MSG(st == GS::SSLM_OK && tok == want, "[G8.1 %s] after the dead end: %s token %d (want %d)", label, Name(st),
	          tok, want);
}

}  // namespace

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		const std::string dir = FixtureDir(argc, argv);
		CRig c(rig.raw);
		const std::vector<int32_t> prompt = c.Tokenize(kPromptPotionText);
		Mapped host, dev;
		if (MapVariant(rig, dir + "/fsfd.sslm", 0, &host)) Arm("host head", rig, host.m, host.view, prompt);
		if (MapVariant(rig, dir + "/fsfd.sslm", SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE, &dev)) {
			CHECK_MSG(dev.view.config.tie_word_embeddings, "[G8.1] fsfd.sslm is not tied");
			Arm("tied device head", rig, dev.m, dev.view, prompt);
		}
		rig.ReleaseSeqs();
		Unmap(rig, &host);
		Unmap(rig, &dev);
	}
	return Finish("sfdg_cell_g8_1");
}
