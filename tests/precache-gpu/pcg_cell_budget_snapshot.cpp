// Precached-prefixes plan §3.1 on GPU [F2]: a frozen prefix is, for every purpose an adopter can observe, the
// state the blob carries. An adopter observes the prefix's prefill snapshot through
// sslm_gpu_seq_read_prefill_final_hidden (adopt copies it), and the restore rebuilds that snapshot from the
// blob's residual, valid whenever the prefix is ready. So a prefix whose last prefill landed tokens and then
// ended SSLM_GPU_KV_BUDGET_EXCEEDED (ready, at full depth, savable) must give its adopters the same snapshot
// read, original and restored alike: the same status, codes and scale, and the same greedy tokens.
//
// RED before the fix: the budget-exceeded exit set ready_for_logits but left the snapshot invalid, so an
// adopter of the original read SSLM_PREFILL_HIDDEN_UNAVAILABLE while an adopter of the restored prefix read
// the residual.
//
// Run: pcg_cell_budget_snapshot.exe --fixture-dir=DIR (pkv_def).
#include "pcg_helpers.h"

using namespace pcg;

namespace {

struct AdopterView {
	GS read = GS::SSLM_DEVICE_LOST;
	std::vector<int8_t> codes;
	int64_t m = 0, e = 0;
	std::vector<int32_t> tokens;
};

AdopterView Observe(Rig& rig, SslmGpuPrefixHandle* p, int64_t len, int32_t last_token, uint32_t hidden) {
	AdopterView v;
	GSeq s = rig.Budgeted(256);
	if (!s.h) return v;
	const GS st = Adopt(rig, s, p, len);
	CHECK_MSG(st == GS::SSLM_OK, "adopt: %s", StatusName(st));
	if (st == GS::SSLM_OK) {
		v.codes.assign(hidden, 0);
		size_t required = 0;
		v.read = sslm_gpu_seq_read_prefill_final_hidden(rig.ctx, s.h, v.codes.data(), v.codes.size(), &required, &v.m,
		                                                &v.e);
		s.last = last_token;
		v.tokens = DecodeN(rig.e, s, 8);
	}
	rig.ReleaseSeq(s.h);
	return v;
}

void CellBudgetSnapshot(const Env& e) {
	Rig rig(e, 32);
	if (!rig.pool) return;
	const int32_t budget = 20;
	const std::vector<int32_t> t = PrefixTokens(30, 47);
	SslmGpuPrefixHandle* raw = nullptr;
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, budget, &raw);
	CHECK_MSG(st == GS::SSLM_OK, "prefix_begin(%d): %s", budget, StatusName(st));
	if (st != GS::SSLM_OK) return;
	SslmGpuPrefixHandle* p = rig.Prefix(raw);
	// One call that admits every token (full_budget is one token's dispatches), so the budget stops it.
	int32_t consumed = 0;
	st = sslm_gpu_prefix_prefill(rig.ctx, p, t.data(), static_cast<int32_t>(t.size()), 0,
	                             e.full_budget * static_cast<uint32_t>(t.size()), &consumed);
	CHECK_MSG(st == GS::SSLM_GPU_KV_BUDGET_EXCEEDED && consumed > 0,
	          "the last prefill must exceed the budget with tokens landed: %s, consumed %d", StatusName(st), consumed);
	if (consumed <= 0) return;
	CHECK(sslm_gpu_prefix_freeze(rig.ctx, p) == GS::SSLM_OK);
	const std::vector<uint8_t> blob = SavePrefix(e, p);
	if (blob.empty()) return;
	const RestoredPrefix r = RestorePrefix(rig, blob);
	CHECK_MSG(r.status == GS::SSLM_OK && r.p, "restore: %s", Name(r.status));
	if (!r.p) return;
	uint32_t hidden = 0;
	CHECK(sslm_gpu_model_hidden_size(rig.model, &hidden) == GS::SSLM_OK && hidden > 0);
	const int32_t last = t[static_cast<size_t>(consumed) - 1];
	const AdopterView a = Observe(rig, p, consumed, last, hidden);
	const AdopterView b = Observe(rig, r.p, consumed, last, hidden);
	std::printf("budget snapshot: landed %d of %zu; original adopter read %s, restored adopter read %s\n", consumed,
	            t.size(), Name(a.read), Name(b.read));
	CHECK_MSG(a.read == b.read, "adopters of the original (%s) and of the restored prefix (%s) read the snapshot differently",
	          Name(a.read), Name(b.read));
	CHECK_MSG(a.read == GS::SSLM_OK, "an adopter of a ready budget-exceeded prefix reads its snapshot: %s", Name(a.read));
	if (a.read == GS::SSLM_OK && b.read == GS::SSLM_OK) {
		CHECK_MSG(a.codes == b.codes && a.m == b.m && a.e == b.e, "the two adopters' snapshot codes or scale differ");
	}
	CHECK_MSG(!a.tokens.empty() && a.tokens == b.tokens, "the two adopters' greedy tokens differ");
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) CellBudgetSnapshot(e);
	}
	return FinishSuite("pcg_cell_budget_snapshot");
}
