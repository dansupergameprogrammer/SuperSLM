// Precached-prefixes plan §7 cell 3.4 [F2]: a GPU restore issued while a sequence is Submitted returns
// SSLM_BUSY with nothing drawn (the fill probe reads the pool unchanged); after sslm_gpu_ready it succeeds.
// The save, which submits nothing, is not refused Busy (§4 item 1). Cell 11.5 [F2] is this cell run against
// an engine whose restore has no EntrySubmitGuard (tools: tests/precache-gpu/apply_gpu_mutant.py
// no-entry-guard): it must go red. Kept in its own executable so the mutant build is this cell alone.
//
// RED AT F1: the save stub returns SSLM_DEVICE_LOST, so there is no blob to restore.
//
// Run: pcg_cell_busy.exe --fixture-dir=DIR (pkv_def).
#include "pcg_helpers.h"

using namespace pcg;

namespace {

void Cell34(const Env& e) {
	Rig rig(e, 32);
	if (!rig.pool) return;
	const std::vector<int32_t> t = PrefixTokens(40, 31);
	SslmGpuPrefixHandle* p = Prefix(rig, t);
	if (!p) return;
	const std::vector<uint8_t> blob = SavePrefix(e, p);
	if (blob.empty()) return;
	SubmittedSeq sub = MakeSubmitted(rig);
	if (!sub.ok) return;
	// Pages held now: the prefix's 3 and the Submitted sequence's R(64) = 5.
	const int64_t free_before = 32 - e.geo.PrefixPages(40) - e.geo.R(64);
	{
		GS st = GS::SSLM_DEVICE_LOST;
		const std::vector<uint8_t> again = SavePrefix(e, p, &st);
		CHECK_MSG(st == GS::SSLM_OK && again == blob, "3.4: the save while a sequence is Submitted: %s (it submits nothing)",
		          Name(st));
	}
	const RestoredPrefix busy = RestorePrefix(rig, blob);
	Refused(busy, GS::SSLM_BUSY, "3.4: restore while a sequence is Submitted");
	const RestoredPrefix busy_hint = RestorePrefix(rig, blob, p);
	Refused(busy_hint, GS::SSLM_BUSY, "3.4: restore with a hint while a sequence is Submitted");
	CHECK(Drain(e, sub.h) == GS::SSLM_OK);
	{
		struct S : ProbeState {
			Rig& r;
			explicit S(Rig& rr) : r(rr) {}
			SslmGpuKvPoolHandle* Pool() override { return r.pool; }
			SslmGpuContext* Ctx() override { return r.ctx; }
			SslmGpuModelHandle* Model() override { return r.model; }
			void DrainAll() override {}
		} s(rig);
		ProbeExactlyFreeOneState(e, s, free_before, "3.4: nothing drawn by the Busy refusals");
	}
	const RestoredPrefix ok = RestorePrefix(rig, blob);
	CHECK_MSG(ok.status == GS::SSLM_OK && ok.p, "3.4: after sslm_gpu_ready the restore returns %s", Name(ok.status));
	if (ok.p) CHECK_MSG(SavePrefix(e, ok.p) == blob, "3.4: the restored prefix re-saves byte-identical");
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) Cell34(e);
	}
	return FinishSuite("pcg_cell_busy");
}
