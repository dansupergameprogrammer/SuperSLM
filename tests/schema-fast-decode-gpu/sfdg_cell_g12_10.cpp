// Schema fast-decode plan rev 1.4.4 §11.2, cell G12.10 (slice G1-J; engine code review B1): a host allocation
// failure inside the verb's pass restores the entry state. G12.7 covers a failure the pass returns as a status; this
// cell covers one it throws. S-GJUMPALLOC throws std::bad_alloc on the host at one of three sites of a canonical
// call, after the entry snapshot is taken:
//   site 0  before the chunk pre-scan, with the prefill snapshot's flag already cleared;
//   site 1  after the pooled holder's pages are mapped through the admitted rows;
//   site 2  after the pass's sub-chunks committed on the device, before the walk and the ready flag move.
// The verb must return SSLM_GPU_ALLOCATION_FAILED with nothing admitted and everything in its entry snapshot as at
// entry: context length, walk, ready flag, layer, the pooled holder's mapped page count, the blob, and
// sslm_gpu_seq_read_prefill_final_hidden (valid at entry, because the call starts after a prefill bridge). After it,
// a canonical call equals a fresh sequence's. Dedicated and pooled; LONGKEY's start span at chunk budget 8. The
// prompt is sized so that the 8 rows cross a page boundary, so site 1 leaves a page mapped that the exit must
// unmap. Before the fix the throw reached the API boundary, which returns the same status without the restore.
// Box only. Run: sfdg_cell_g12_10.exe [--fixture-dir=DIR]
// MR5 RE-POINT (multi-row GPU plan, ruling 29 M1 as widened). 1.18 removes Canonical, so the call is an Exact call:
// after the prompt (3B - 2 tokens) a token in hand, and a draft of Off's own next 7 tokens (computed by decode alone
// on a twin) at chunk budget 8: one pass of 8 rows from 3B - 2, across the page boundary. S-GJUMPALLOC's three sites
// are JumpExactInternal's (MR5): 0 after the entry snapshot, 1 after every row's page is mapped, 2 after the pass,
// before the ready flag moves. Every check above is unchanged; "a canonical call after the failure equals a fresh
// one's" is re-pointed to the same Exact call (token and draft), which also keeps Off's 7. Retired: the precondition
// "LONGKEY's start chain >= 8" (Exact keeps no start span); the draft's length is checked instead.
#include "sfdg_helpers.h"

using namespace sfdg;

namespace {

SslmGpuKvPoolHandle* g_pool = nullptr;

GSeq NewSeq(GRig& rig, bool pooled) {
	if (!pooled) return rig.New(kLongKey);
	GSeq s;
	const GS st = sslm_gpu_seq_create_budgeted(rig.ctx(), rig.model(), g_pool, 200, &s.h);
	CHECK_MSG(st == GS::SSLM_OK && s.h, "[G12.10] budgeted create: %s", Name(st));
	if (!s.h) return s;
	rig.seqs.push_back(s.h);
	SslmGpuSeqSetSchemaForG5Bridge(rig.ctx(), s.h, rig.Index(kLongKey));
	return s;
}

}  // namespace

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		CRig c(rig.raw);
		const std::vector<int32_t> base = c.Tokenize(kPromptPotionText);
		// A prompt ending two positions before a page boundary: 8 admitted rows map one more page.
		const int64_t B = rig.env.geo.B > 0 ? rig.env.geo.B : 16;
		const std::vector<int32_t> prompt = PromptOfLength(base, static_cast<size_t>(3 * B - 2));
		const GS ps = sslm_gpu_kv_pool_create(rig.ctx(), rig.model(), 64, &g_pool);
		CHECK_MSG(ps == GS::SSLM_OK && g_pool, "[G12.10] pool: %s", Name(ps));
		for (const bool pooled : {false, true}) {
			if (pooled && !g_pool) continue;
			for (const int32_t site : {0, 1, 2}) {
				char arm[64];
				std::snprintf(arm, sizeof arm, "%s, site %d", pooled ? "pooled" : "dedicated", site);
				GSeq s = NewSeq(rig, pooled), fresh = NewSeq(rig, pooled), twin = NewSeq(rig, pooled), o = NewSeq(rig, pooled);
				if (!s.h || !fresh.h || !twin.h || !o.h) continue;
				for (GSeq* q : {&s, &fresh, &twin, &o}) rig.Prompt(*q, prompt);
				GS st = GS::SSLM_OK;
				int32_t p = -1;
				for (GSeq* q : {&s, &fresh, &twin}) {
					const int32_t t = Token(rig.env, *q, &st);
					CHECK_MSG(st == GS::SSLM_OK && t >= 0 && (p < 0 || t == p), "[G12.10 %s] the token in hand: %s %d", arm, Name(st), t);
					p = t;
				}
				std::vector<int32_t> offn = OffTokensGpu(rig, o, 8);
				while (!offn.empty() && offn.back() < 0) offn.pop_back();
				CHECK_MSG(offn.size() >= 8 && offn[0] == p, "[G12.10 %s] precondition: Off's 8 tokens after the prompt (%zu)", arm,
				          offn.size());
				if (offn.size() < 8) {
					rig.ReleaseSeqs();
					continue;
				}
				const std::vector<int32_t> draft(offn.begin() + 1, offn.begin() + 8);
				if (pooled) {
					// The precondition for site 1: a successful call maps at least one page.
					const SeqObs t0 = Observe(rig, twin.h);
					const JumpOut tj = JumpDraft(rig, twin, p, 8, draft);
					CHECK_MSG(tj.st == GS::SSLM_OK && tj.forced == draft && Observe(rig, twin.h).mapped > t0.mapped,
					          "[G12.10 %s] precondition: the call keeps Off's 7 and maps a page (%s, kept %d, mapped %u -> %u)", arm,
					          Name(tj.st), tj.count, t0.mapped, Observe(rig, twin.h).mapped);
				}
				const SeqObs before = Observe(rig, s.h);
				const SnapRead snap = ReadSnapshot(rig.ctx(), s.h);
				CHECK_MSG(snap.st == GS::SSLM_OK, "[G12.10 %s] precondition: a valid snapshot at entry (%s)", arm,
				          Name(snap.st));
				CHECK_MSG(sslm_gpu_test_only_arm_jump_host_bad_alloc(site) == GS::SSLM_OK, "[G12.10] arm");
				const JumpOut j = JumpDraft(rig, s, p, 8, draft);
				sslm_gpu_test_only_arm_jump_host_bad_alloc(-1);
				CHECK_MSG(j.st == GS::SSLM_GPU_ALLOCATION_FAILED && j.count == 0,
				          "[G12.10 %s] the verb returned %s with %d kept", arm, Name(j.st), j.count);
				const SeqObs after = Observe(rig, s.h);
				CHECK_MSG(after == before, "[G12.10 %s] the entry state was not restored: context %lld -> %lld, walk %u -> "
				          "%u, ready %d -> %d, layer %u -> %u, mapped %u -> %u, blob %s", arm,
				          static_cast<long long>(before.ctx), static_cast<long long>(after.ctx), before.walk, after.walk,
				          before.ready, after.ready, before.layer, after.layer, before.mapped, after.mapped,
				          after.blob == before.blob ? "equal" : "differs");
				const SnapRead snap_after = ReadSnapshot(rig.ctx(), s.h);
				CHECK_MSG(snap_after == snap, "[G12.10 %s] the prefill snapshot differs from the entry's (%s)", arm,
				          Name(snap_after.st));
				// The sequence decodes as if the call had not been made: the same Exact call equals a fresh one's.
				const JumpOut a = JumpDraft(rig, s, p, 8, draft), b = JumpDraft(rig, fresh, p, 8, draft);
				CHECK_MSG(a.st == GS::SSLM_OK && b.st == GS::SSLM_OK && a.forced == b.forced && a.forced == draft &&
				              Observe(rig, s.h) == Observe(rig, fresh.h),
				          "[G12.10 %s] the Exact call after the failure differs from a fresh one's (%s kept %d, %s kept %d)", arm,
				          Name(a.st), a.count, Name(b.st), b.count);
				rig.ReleaseSeqs();
			}
		}
		rig.ReleaseSeqs();
		if (g_pool) sslm_gpu_kv_pool_destroy(rig.ctx(), g_pool);
	}
	return Finish("sfdg_cell_g12_10");
}
