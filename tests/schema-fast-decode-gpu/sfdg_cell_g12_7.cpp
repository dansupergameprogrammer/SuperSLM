// Schema fast-decode plan rev 1.4.4 §11.2, cell G12.7 (slice G1-J; rev 1.4.1, review L9, audit G-2): canonical
// failure. S-GCHUNKFAULT fires at sub-chunk 0 and at sub-chunk 1 of a canonical call, with
// SSLM_GPU_ALLOCATION_FAILED and with SSLM_DEVICE_LOST, on a dedicated and on a pooled sequence. The verb returns
// that status and commits nothing: everything in its entry snapshot equals the entry -- context length, walk, the
// five counters, the hidden state, layer index and ready flag (the blob holds them all, and the dedicated host
// mirror), the pooled holder's mapped pages, and sslm_gpu_seq_read_prefill_final_hidden, which is valid at entry
// because the call starts after a prefill bridge (rev 1.4.2, strike N5; XG30). The call is LONGKEY's start span at
// chunk budget 8: two sub-chunks, so a fault at sub-chunk 1 comes after sub-chunk 0 committed four rows on the
// device, which the exit must not keep (XG19). After the failure, an off decode equals GPU off's, and a canonical
// call equals a fresh one's. Box only. Run: sfdg_cell_g12_7.exe [--fixture-dir=DIR]
// MR5 RE-POINT (multi-row GPU plan, ruling 29 M1 as widened). 1.18 removes Canonical, so the failing call is an Exact
// call: LONGKEY after the prompt, a token in hand, and a draft of Off's own next 11 tokens (computed by decode alone on
// a twin) at chunk budget 12: 12 rows, two passes (8 and 4). S-GCHUNKFAULT's index is the Exact pass (MR5):
//   pass 0  the call fails before any row is kept: it returns the fault's status, kept 0, and everything in the entry
//           snapshot is as at entry (the checks above, unchanged); then the decode equals GPU off's (XG19), and an Exact
//           call (the same token and draft) equals a fresh twin's.
//   pass 1  the second pass fails after the first kept its 8 rows: Exact keeps the earlier passes and returns SSLM_OK
//           (build-mr5.md; docs/api.md). Re-pointed from 1.17's "a fault at sub-chunk 1 commits nothing": kept 7 (Off's
//           first 7 drafts), and the sequence (blob, walk, flags, context, mapped pages) and the next decode equal a
//           twin's call at budget 8 (the first pass alone): the failed pass leaves no row on the device or the host (XG19).
// The prefill snapshot is untouched by either (XG12). Retired: the precondition "LONGKEY's start chain >= 8" (Exact
// keeps no start span); the draft's length is checked instead.
#include "sfdg_helpers.h"

using namespace sfdg;

namespace {

SslmGpuKvPoolHandle* g_pool = nullptr;

GSeq NewSeq(GRig& rig, bool pooled) {
	if (!pooled) return rig.New(kLongKey);
	GSeq s;
	const GS st = sslm_gpu_seq_create_budgeted(rig.ctx(), rig.model(), g_pool, 200, &s.h);
	CHECK_MSG(st == GS::SSLM_OK && s.h, "[G12.7] budgeted create: %s", Name(st));
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
		const std::vector<int32_t> prompt = c.Tokenize(kPromptPotionText);
		const GS ps = sslm_gpu_kv_pool_create(rig.ctx(), rig.model(), 96, &g_pool);  // MR5: five sequences per arm
		CHECK_MSG(ps == GS::SSLM_OK && g_pool, "[G12.7] pool: %s", Name(ps));
		for (const bool pooled : {false, true}) {
			if (pooled && !g_pool) continue;
			for (const GS fault : {GS::SSLM_GPU_ALLOCATION_FAILED, GS::SSLM_DEVICE_LOST}) {
				for (const int32_t at : {0, 1}) {
					char arm[96];
					std::snprintf(arm, sizeof arm, "%s, %s at pass %d", pooled ? "pooled" : "dedicated", Name(fault), at);
					GSeq s = NewSeq(rig, pooled), off = NewSeq(rig, pooled), fresh = NewSeq(rig, pooled), o = NewSeq(rig, pooled);
					if (!s.h || !off.h || !fresh.h || !o.h) continue;
					for (GSeq* q : {&s, &off, &fresh, &o}) rig.Prompt(*q, prompt);
					GS st = GS::SSLM_OK;
					int32_t p = -1;
					for (GSeq* q : {&s, &off, &fresh}) {
						const int32_t t = Token(rig.env, *q, &st);
						CHECK_MSG(st == GS::SSLM_OK && t >= 0 && (p < 0 || t == p), "[G12.7 %s] the token in hand: %s %d", arm, Name(st), t);
						p = t;
					}
					// Off's stream from the same point, by decode alone: offn[0] is p, then its next tokens.
					std::vector<int32_t> offn = OffTokensGpu(rig, o, 13);
					while (!offn.empty() && offn.back() < 0) offn.pop_back();
					CHECK_MSG(offn.size() >= 13 && offn[0] == p, "[G12.7 %s] precondition: Off's 12 tokens after %d (%zu)", arm, p,
					          offn.size());
					if (offn.size() < 13) {
						rig.ReleaseSeqs();
						continue;
					}
					const std::vector<int32_t> draft(offn.begin() + 1, offn.begin() + 12);
					const SeqObs before = Observe(rig, s.h);
					const SnapRead snap = ReadSnapshot(rig.ctx(), s.h);
					CHECK_MSG(snap.st == GS::SSLM_OK, "[G12.7 %s] precondition: a valid snapshot at entry (%s)", arm,
					          Name(snap.st));
					CHECK_MSG(sslm_gpu_test_only_arm_chunk_fault(at, fault) == GS::SSLM_OK, "[G12.7] arm");
					const JumpOut j = JumpDraft(rig, s, p, 12, draft);
					sslm_gpu_test_only_arm_chunk_fault(-1, GS::SSLM_OK);
					CHECK_MSG(ReadSnapshot(rig.ctx(), s.h) == snap, "[G12.7 %s] the prefill snapshot differs from the entry's "
					          "(%s)", arm, Name(ReadSnapshot(rig.ctx(), s.h).st));
					if (at == 0) {
						CHECK_MSG(j.st == fault && j.count == 0, "[G12.7 %s] the verb returned %s with %d kept", arm, Name(j.st),
						          j.count);
						const SeqObs after = Observe(rig, s.h);
						CHECK_MSG(after == before, "[G12.7 %s] the entry state was not restored: context %lld -> %lld, walk %u -> "
						          "%u, ready %d -> %d, layer %u -> %u, mapped %u -> %u, blob %s", arm,
						          static_cast<long long>(before.ctx), static_cast<long long>(after.ctx), before.walk, after.walk,
						          before.ready, after.ready, before.layer, after.layer, before.mapped, after.mapped,
						          after.blob == before.blob ? "equal" : "differs");
						// An off decode with the same token equals GPU off's (XG19), and an Exact call a fresh one's.
						GSeq s2 = s;
						GS st_off = GS::SSLM_OK;
						const int32_t t = Token(rig.env, s2, &st), t_off = Token(rig.env, off, &st_off);
						CHECK_MSG(st == st_off && t == t_off && rig.Blob(s.h) == rig.Blob(off.h),
						          "[G12.7 %s] the decode after: %s %d; GPU off %s %d", arm, Name(st), t, Name(st_off), t_off);
						GSeq s3 = NewSeq(rig, pooled);
						if (s3.h) {
							rig.Prompt(s3, prompt);
							const int32_t p3 = Token(rig.env, s3, &st);
							CHECK_MSG(sslm_gpu_test_only_arm_chunk_fault(at, fault) == GS::SSLM_OK, "[G12.7] arm");
							(void)JumpDraft(rig, s3, p3, 12, draft);
							sslm_gpu_test_only_arm_chunk_fault(-1, GS::SSLM_OK);
							const JumpOut a = JumpDraft(rig, s3, p3, 12, draft), b = JumpDraft(rig, fresh, p, 12, draft);
							CHECK_MSG(a.st == GS::SSLM_OK && a.forced == b.forced && a.forced == draft &&
							              Observe(rig, s3.h) == Observe(rig, fresh.h),
							          "[G12.7 %s] the Exact call after the failure (%s, kept %d) differs from a fresh one's (%s, "
							          "kept %d; Off's draft %zu)", arm, Name(a.st), a.count, Name(b.st), b.count, draft.size());
						}
					} else {
						// The first pass's 8 rows: p and 7 drafts, the last checked against draft[7].
						const std::vector<int32_t> first(draft.begin(), draft.begin() + 7);
						CHECK_MSG(j.st == GS::SSLM_OK && j.forced == first, "[G12.7 %s] the verb returned %s, kept %s; want SSLM_OK "
						          "and the first pass's %s (Exact keeps the earlier passes)", arm, Name(j.st), Str(j.forced).c_str(),
						          Str(first).c_str());
						const JumpOut b = JumpDraft(rig, fresh, p, 8, first);
						const SeqObs sa = Observe(rig, s.h), sb = Observe(rig, fresh.h);
						CHECK_MSG(b.st == GS::SSLM_OK && b.forced == first && sa == sb,
						          "[G12.7 %s] the sequence after the failed second pass differs from one pass at budget 8 (%s, kept %d): "
						          "context %lld / %lld, walk %u / %u, ready %d / %d, layer %u / %u, mapped %u / %u, blob %s", arm,
						          Name(b.st), b.count, static_cast<long long>(sa.ctx), static_cast<long long>(sb.ctx), sa.walk, sb.walk,
						          sa.ready, sb.ready, sa.layer, sb.layer, sa.mapped, sb.mapped, sa.blob == sb.blob ? "equal" : "differ");
						GS sb_st = GS::SSLM_OK;
						const int32_t ta = Token(rig.env, s, &st), tb = Token(rig.env, fresh, &sb_st);
						CHECK_MSG(st == GS::SSLM_OK && ta == tb && ta == offn[8] && rig.Blob(s.h) == rig.Blob(fresh.h),
						          "[G12.7 %s] the decode after: %d; the one-pass twin's %d; Off's %d", arm, ta, tb, offn[8]);
					}
					rig.ReleaseSeqs();
				}
			}
		}
		rig.ReleaseSeqs();
		if (g_pool) sslm_gpu_kv_pool_destroy(rig.ctx(), g_pool);
	}
	return Finish("sfdg_cell_g12_7");
}
