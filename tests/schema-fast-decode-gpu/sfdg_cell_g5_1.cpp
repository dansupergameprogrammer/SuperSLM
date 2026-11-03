// Schema fast-decode plan rev 1.4.4 §11.2, cell G5.1 (slice G1-M): a hostile hook on the legal list.
// A host parallel-for hook that runs none of its tasks breaks the exactly-once contract. Under a bound schema the
// GPU finish's legal-row pass (LogitsSiteLegalRows over the walk state's CSR ids) calls it, so the finish returns
// SSLM_GPU_PARALLEL_FOR_INCOMPLETE and the sequence rests: no token, the walk state unchanged, the ready flag
// re-armed. A retry with an honest hook (and with none) gives the token an unhooked sequence gives, and S-GROWS
// shows the retry scored the state's legal rows only.
//   Arm 1  the head on the host (flags 0): the existing host-head exit, now over the legal list.
//   Arm 2  the head on the device, tied (rev 1.4.1, review L8): the bound finish now runs on the host through the
//          hook, so a device-head model can return SSLM_GPU_PARALLEL_FOR_INCOMPLETE from a finish, which it never
//          could before G1-M.
// Box only. Needs fsfd.sslm. Run: sfdg_cell_g5_1.exe [--fixture-dir=DIR]
#include "sfdg_helpers.h"

#include <atomic>

using namespace sfdg;

namespace {

struct HookState {
	std::atomic<int> calls{0};
	bool hostile = false;
};
void Run(void* host, int32_t count, sslm_task_fn task, void* ctx) {
	auto& h = *static_cast<HookState*>(host);
	h.calls.fetch_add(1);
	if (h.hostile) return;  // omits every task index
	for (int32_t i = 0; i < count; ++i) task(ctx, i);
}
sslm_parallel_for Hook(HookState* h) {
	sslm_parallel_for pf{};
	pf.run = &Run;
	pf.host_ctx = h;
	pf.max_tasks = 4;
	return pf;
}

// One arm: `m` is a model of the tied fixture with residency `label`.
void Arm(const char* label, GRig& rig, SslmGpuModelHandle* m, const superslm::SslmModelView& view,
         const std::vector<int32_t>& prompt) {
	// The reference: an unhooked sequence's first token at the potion schema's start state.
	CHECK_MSG(sslm_gpu_context_set_host_parallel_for(rig.ctx(), nullptr) == GS::SSLM_OK, "[G5.1 %s] clear hook", label);
	GSeq ref = rig.New(kPotion, m);
	if (!ref.h) return;
	rig.Prompt(ref, prompt);
	GS st = GS::SSLM_OK;
	const int32_t want = Token(rig.env, ref, &st);
	CHECK_MSG(st == GS::SSLM_OK && want >= 0, "[G5.1 %s] reference token: %s %d", label, Name(st), want);
	const uint32_t want_walk = SslmGpuSeqWalkStateForG5Bridge(ref.h);

	GSeq s = rig.New(kPotion, m);
	if (!s.h) return;
	rig.Prompt(s, prompt);  // ready: the next decode call finishes without an embed
	const uint32_t walk0 = SslmGpuSeqWalkStateForG5Bridge(s.h);
	const uint32_t n = LegalCount(view, kPotion, walk0);
	CHECK_MSG(n >= 1, "[G5.1 %s] the start state has no legal row", label);

	HookState hs;
	hs.hostile = true;
	sslm_parallel_for pf = Hook(&hs);
	CHECK_MSG(sslm_gpu_context_set_host_parallel_for(rig.ctx(), &pf) == GS::SSLM_OK, "[G5.1 %s] install", label);
	sslm_gpu_test_only_rows_log_reset();
	for (int attempt = 0; attempt < 2; ++attempt) {  // rests every time, not once
		int32_t tok = -7;
		const GS h = SslmGpuSeqFinishTokenForG5Bridge(rig.ctx(), s.h, &tok);
		CHECK_MSG(h == GS::SSLM_GPU_PARALLEL_FOR_INCOMPLETE && tok == -1,
		          "[G5.1 %s] hostile hook, attempt %d: %s token %d (want PARALLEL_FOR_INCOMPLETE, -1)", label, attempt,
		          Name(h), tok);
		CHECK_MSG(SslmGpuSeqWalkStateForG5Bridge(s.h) == walk0, "[G5.1 %s] the walk moved on a rested finish", label);
	}
	CHECK_MSG(hs.calls.load() >= 2, "[G5.1 %s] the bound finish never called the hook (%d calls)", label,
	          hs.calls.load());
	CHECK_MSG(RowsLog().empty(), "[G5.1 %s] a rested finish logged S-GROWS rows", label);

	// The retry, honest hook: the ready flag was re-armed, so the decode bridge finishes without an embed.
	hs.hostile = false;
	sslm_gpu_test_only_rows_log_reset();
	const int32_t got = Token(rig.env, s, &st);
	const std::vector<RowsEntry> l = RowsLog();
	CHECK_MSG(st == GS::SSLM_OK && got == want && SslmGpuSeqWalkStateForG5Bridge(s.h) == want_walk,
	          "[G5.1 %s] retry: %s token %d walk %u (want %d walk %u)", label, Name(st), got,
	          SslmGpuSeqWalkStateForG5Bridge(s.h), want, want_walk);
	CHECK_MSG(!l.empty() && l.back().rows == n && l.back().device == 0,
	          "[G5.1 %s] retry S-GROWS: rows %llu device %d (want %u legal rows on the host)", label,
	          l.empty() ? 0ull : static_cast<unsigned long long>(l.back().rows), l.empty() ? -1 : l.back().device, n);

	// And with no hook at all: a second rested sequence, retried unhooked.
	GSeq t = rig.New(kPotion, m);
	if (!t.h) return;
	rig.Prompt(t, prompt);
	hs.hostile = true;
	int32_t tok = -7;
	const GS h = SslmGpuSeqFinishTokenForG5Bridge(rig.ctx(), t.h, &tok);
	CHECK_MSG(h == GS::SSLM_GPU_PARALLEL_FOR_INCOMPLETE, "[G5.1 %s] second sequence: %s", label, Name(h));
	CHECK_MSG(sslm_gpu_context_set_host_parallel_for(rig.ctx(), nullptr) == GS::SSLM_OK, "[G5.1 %s] clear", label);
	const int32_t got2 = Token(rig.env, t, &st);
	CHECK_MSG(st == GS::SSLM_OK && got2 == want, "[G5.1 %s] unhooked retry: %s token %d (want %d)", label, Name(st), got2,
	          want);
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
			CHECK_MSG(dev.view.config.tie_word_embeddings, "[G5.1] fsfd.sslm is not tied");
			Arm("tied device head", rig, dev.m, dev.view, prompt);
		}
		sslm_gpu_context_set_host_parallel_for(rig.ctx(), nullptr);
		rig.ReleaseSeqs();
		Unmap(rig, &host);
		Unmap(rig, &dev);
	}
	return Finish("sfdg_cell_g5_1");
}
