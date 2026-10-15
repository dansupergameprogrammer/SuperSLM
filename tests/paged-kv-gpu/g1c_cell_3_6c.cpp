// Paged-KV plan (rev 16.2) §7 cell 3.6c [G3 two pools and pool create; G4 adopt, begin_from; G5
// restore-into-pool], group g1c.
//
//   "The process-wide rule. Each case is refused SSLM_BUSY with nothing drawn: after sslm_gpu_ready,
//    the GPU fill probe is unchanged, and then each call proceeds and its output is byte-equal. The
//    mutant whose guard is scoped to the pool returns SSLM_OK in every case."
//
//   [G3] (i) Holder A on pool 1 is Submitted, and holder B on pool 2 of the same context decodes.
//        (ii) A dedicated sequence is Submitted, and sslm_gpu_kv_pool_create is called: SSLM_BUSY and a
//        null out-handle. Pool create reaches only its context's own allocator (§4.4), so (ii) pins
//        the over-guard choice (decision 63), not a hazard.
//   [G4] A dedicated sequence is Submitted, and adopt and begin_from are each called.
//   [G5] The same, for restore-into-pool.
//
// (i), G4 and G5 are g1c_guard_runs.h's GUARDED RUN: (i) with A on its own pool (AKind::PooledOtherPool;
// the fill probe reads B's pool, pool 0 here), G4/G5 with A dedicated (AKind::Dedicated). The pool-scoped
// mutant sees no Submitted holder on B's pool and returns SSLM_OK: the first check of each run.
//
// (ii) has ITS OWN construction (carrier row "CLASS-B ride, registry face": "3.6c's pool-create case
// gets its own construction"), not 3.6d's: the refused create must leave a null out-handle and draw
// nothing -- no pool exists to draw from, so "nothing drawn" is the null handle -- and after the drain
// the create proceeds, and its output is graded as a pool's output can be: every page of its mirror
// reads zero through the GPU mirror peek (§4 item 3: zero-filled before it returns), and the GPU fill
// probe shows exactly `page_count` pages free (admission on this pool, refusal on a rebuild). The twin
// is the same create with nothing Submitted.
//
// RED NOW (63e7b0e): BY LINK -- sslm_gpu_kv_pool_create/destroy, sslm_gpu_seq_create_budgeted and the
// mirror peek (G3), the prefix verbs and adopt (G4), restore-into-pool (G5).
// GREEN: each part at its step, on both GPUs.
#include "g1c_guard_runs.h"
// pkvg unification: the parts after PKVG_STEP (g1c_helpers.h) are compiled out.

namespace {
using namespace g1c;

void Run(GuardedSpec g, const char* id) {
	g.name = std::string(id) + " " + g.name;
	g.resave_holder = nullptr;  // 3.6c names no re-save
	GuardedRun(g);
}

void G3TwoPools() { Run(SpecDecode(AKind::PooledOtherPool), "3.6c/G3 (i)"); }

// (ii): a dedicated D Submitted; sslm_gpu_kv_pool_create.
void G3PoolCreate() {
	Stack* s = GetStack();
	if (!s) return;
	const uint32_t pages = 24;
	const size_t pb = s->art.PageBytes();
	// The twin: the create with nothing Submitted.
	std::vector<uint8_t> twin_mirror;
	SslmGpuStatus twin_st = SSLM_DEVICE_LOST;
	{
		World w(*s);
		SslmGpuSequenceHandle* d = w.Dedicated();
		if (!d || !w.Submit(d, AToken(*s))) return;
		(void)w.Drain(d);
		twin_st = sslm_gpu_kv_pool_create(s->ctx, s->host, pages, &w.out_pool);
		G1C_CHECK(twin_st == SSLM_OK && w.out_pool, "3.6c/G3 (ii): SETUP: the twin's create -> %s", St(twin_st));
		if (w.out_pool) twin_mirror = MirrorBytes(w.out_pool, pages, pb);
		w.TakeOuts();
	}
	G1C_CHECK(twin_mirror == std::vector<uint8_t>(size_t{pages} * pb, 0),
	          "3.6c/G3 (ii): the twin's fresh pool does not read zero on every page");
	// MAIN.
	{
		World w(*s);
		SslmGpuSequenceHandle* d = w.Dedicated();
		if (!d || !w.Submit(d, AToken(*s))) return;
		const SslmGpuStatus st = sslm_gpu_kv_pool_create(s->ctx, s->host, pages, &w.out_pool);
		G1C_CHECK(st == SSLM_BUSY, "3.6c/G3 (ii): pool create while a dedicated sequence is Submitted -> %s, want "
		                           "SSLM_BUSY", St(st));
		G1C_CHECK(w.out_pool == nullptr, "3.6c/G3 (ii): the refused create left a non-null out-handle");
		w.TakeOuts();
		(void)w.Drain(d);
		// Proceeds: the twin's status, the twin's (all-zero) mirror, and exactly `pages` free.
		const int pi = w.Pool(pages);
		G1C_CHECK(w.pools[pi].h != nullptr, "3.6c/G3 (ii): after the drain the create did not proceed");
		if (!w.pools[pi].h) return;
		const std::vector<uint8_t> m = MirrorBytes(w.pools[pi].h, pages, pb);
		G1C_CHECK(m == twin_mirror, "3.6c/G3 (ii): the created pool's mirror differs from the twin's at byte %zu",
		          FirstDiff(m, twin_mirror));
		(void)AdmissionLeg(w, pi, pages, "3.6c/G3 (ii)");
	}
	// REBUILD: the refusal leg on the same state.
	{
		World w(*s);
		SslmGpuSequenceHandle* d = w.Dedicated();
		if (!d || !w.Submit(d, AToken(*s))) return;
		SslmGpuKvPoolHandle* refused = nullptr;
		const SslmGpuStatus st = sslm_gpu_kv_pool_create(s->ctx, s->host, pages, &refused);
		G1C_CHECK(st == SSLM_BUSY && !refused, "3.6c/G3 (ii): rebuild: %s", St(st));
		if (refused) w.loose_pools.push_back(refused);
		(void)w.Drain(d);
		const int pi = w.Pool(pages);
		if (w.pools[pi].h) (void)RefusalLeg(w, pi, pages, "3.6c/G3 (ii)");
	}
}

#if PKVG_STEP >= 4
void G4Adopt() { Run(SpecAdopt(AKind::Dedicated), "3.6c/G4"); }
void G4BeginFrom() { Run(SpecBeginFrom(AKind::Dedicated), "3.6c/G4"); }
#endif  // PKVG_STEP >= 4
#if PKVG_STEP >= 5
void G5RestoreIntoPool() { Run(SpecRestoreIntoPool(AKind::Dedicated), "3.6c/G5"); }
#endif  // PKVG_STEP >= 5
}  // namespace

G1C_CELL("3.6c/G3:two_pools", "G3", G3TwoPools);
G1C_CELL("3.6c/G3:pool_create", "G3", G3PoolCreate);
#if PKVG_STEP >= 4
G1C_CELL("3.6c/G4:adopt", "G4", G4Adopt);
G1C_CELL("3.6c/G4:begin_from", "G4", G4BeginFrom);
#endif
#if PKVG_STEP >= 5
G1C_CELL("3.6c/G5:restore_into_pool", "G5", G5RestoreIntoPool);
#endif
