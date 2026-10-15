// Paged-KV plan (rev 16.2) §7 cell 9.6 [G5]: "GPU: SLM6 round trip; SLM5 and SLM4 still restore."
// Group g1e, step G1 (the GPU red suite). Plan sources: §4 item 10 (SLM6, the writer rule), §3.7
// (the blob is a function of the valid state alone), §3.9 A10 and A12/A13, §9.2 (GPU cells survive
// by the writer rule).
//
// Parts, all [G5]:
//   A. SLM6 round trip, the GPU twin of 9.1's budget half: pooled budget holders in five states
//      (fresh; prefill 100 + decode 4, resting; prefill 100, mid-token; adopted 1,000, resting ready;
//      adopted 1,008 + 3 tokens, mid-token). Each saves 'SLM6' (a budget blob with the declared
//      budget and origin: the writer rule), and save -> restore_into_pool -> save is byte-identical,
//      taken twice, with a null prefix and (adopted states) with the matching one; the restored
//      holder's next tokens equal the unsaved original's. Each restore draws pages a released
//      sequence dirtied first, so a save that gathers bytes outside the valid rows [0, L') shows in
//      the second save (the GPU form of 9.1's 0x5A pool: the GPU pool has no fill argument).
//   B. SLM5 still restores (A10): a dedicated holder still saves 'SLM5' (writer rule), and
//      sslm_gpu_seq_restore of it re-saves byte-identically and continues as the unsaved original,
//      resting and mid-token.
//   C. SLM4 still restores (A10): the SLM5 blob of B's resting state rewritten to SLM4's layout
//      (g1e_helpers.h, Slm4FromSlm5) restores, re-saves as the SLM5 the original wrote (an SLM4 has
//      no tail; the original's tail is the unbound default, checked as a construction), and continues
//      as the original.
//
// Mutants killed: a pooled holder writing SLM5 (A: magic), SLM6 sized to the cap or gathering past
// L' (A: re-save differs over dirtied pages), a restore that drops kv_mode/budget/origin (A: the
// re-save's header), a dedicated holder writing SLM6 (B: magic; every existing SLM5 cell would then
// break, §9.2), a reader that stops accepting SLM5 or SLM4 (B, C: status).
//
// RED NOW: link-red on sslm_gpu_kv_pool_create/destroy, sslm_gpu_seq_create_budgeted and
// sslm_gpu_test_only_peek_pool_mirror_bytes (G3), sslm_gpu_prefix_* and sslm_gpu_seq_adopt_prefix
// (G4), sslm_gpu_seq_restore_into_pool (G5). After G4: red at run, part A, until G5's SLM6 writer
// (the pooled save is not 'SLM6'). Parts B and C use only shipped verbs and are green once linked.
// GREEN: every check passes on both GPUs, with SSLM_GPU_ENABLE_DEBUG_LAYER=1 reporting nothing.
//
// Run: g1e_cell_9_6.exe --fixture-dir=DIR   (DIR from `python tools/gen_paged_kv_fixture.py DIR`)
#include "g1e_helpers.h"

using namespace g1e;

namespace {

struct StateA {
	const char* name;
	int32_t prefix_len;  // 0: no prefix adopted
	int32_t extra_tokens;
	bool prefill100;
	bool mid;
};
const StateA kStatesA[] = {
    {"fresh", 0, 0, false, false},
    {"prefill100+decode4 resting", 0, 4, true, false},
    {"prefill100 mid-token", 0, 0, true, true},
    {"adopted 1000 resting ready", 1000, 0, false, false},
    {"adopted 1008 + 3 tokens, mid-token", 1008, 3, false, true},
};
constexpr int32_t kBudget = 512;

// Dirties pages of `rig`'s pool: a budget sequence writes `n` rows and is released, so the pages it
// drew are free again and no longer zero.
void Dirty(Rig& rig, int32_t budget, int32_t n) {
	GSeq d = rig.Budgeted(budget);
	if (!d.h) return;
	CHECK(Prefill(rig.e, d, Stream(97, n, kVocab)) == GS::SSLM_OK);
	rig.ReleaseSeq(d.h);
}

std::string Join96(const std::vector<int32_t>& t) {
	std::string o;
	for (size_t i = 0; i < t.size(); ++i) o += (i ? "," : "") + std::to_string(t[i]);
	return o;
}

// CONSTRUCTION (G5 cells fix): the "fresh" state (L = 0, never prefilled, no prefix) has nothing to decode
// from, so its continuation cannot be 8 tokens on either side -- the CPU twin (c4_dim9_persistence.cpp
// 9.1) skips the continuation for "fresh" for that reason. Its continuation here is graded the 8.16 way
// (StepOutcomes: the token, -2, or -1000 - status per step, 8 steps on each side, equal), plus the two
// holders' saves equal after those steps. The round trips (b1 == b2, twice) are unchanged and remain the
// cell's grade for "fresh"; every other state keeps the 8-real-token continuation check.
void PartA(const Env& e) {
	for (const StateA& s : kStatesA) {
		for (int with_handle = 0; with_handle <= (s.prefix_len > 0 ? 1 : 0); ++with_handle) {
			const std::string what = std::string("9.6 A ") + s.name + (with_handle ? " (matching prefix)" : " (null prefix)");
			const int64_t pp = e.geo.PrefixPages(s.prefix_len);
			// The prefix, the original, and two restores (each R + E when private) with slack.
			const int64_t per = e.geo.R(kBudget) + (with_handle ? 0 : e.geo.Materialized(s.prefix_len, kBudget));
			Rig rig(e, static_cast<uint32_t>(pp + e.geo.R(kBudget) + 2 * per + 8));
			if (!rig.pool) continue;
			SslmGpuPrefixHandle* px = s.prefix_len > 0 ? BudgetPrefix(rig, PrefixTokens(s.prefix_len)) : nullptr;
			GSeq o = rig.Budgeted(kBudget);
			if (!o.h || (s.prefix_len > 0 && !px)) continue;
			if (px) CHECK(Adopt(rig, o, px, s.prefix_len) == GS::SSLM_OK);
			if (s.prefill100) CHECK(Prefill(e, o, Stream(31, 100, kVocab)) == GS::SSLM_OK);
			if (s.extra_tokens) CHECK(static_cast<int>(DecodeN(e, o, s.extra_tokens).size()) == s.extra_tokens);
			if (s.mid) EnterMidToken(e, o);
			const std::vector<uint8_t> b1 = Save(e, o.h);
			// kills: a pooled holder writing SLM5, or an SLM6 without its budget and origin (writer rule)
			if (!Slm6SelfCheck(b1, e.geo, kBudget, s.prefix_len, ContextLength(o.h), what.c_str())) continue;
			Dirty(rig, kBudget, 300);
			std::vector<uint8_t> cur = b1;
			for (int round = 1; round <= 2; ++round) {
				const Restored r = RestoreIntoPool(rig, cur, with_handle ? px : nullptr, o);
				CHECK_MSG(r.status == GS::SSLM_OK && r.seq.h, "%s round %d: restore_into_pool: %s", what.c_str(), round,
				          StatusName(r.status));
				if (!r.seq.h) break;
				const std::vector<uint8_t> b2 = Save(e, r.seq.h);
				// kills: a save gathering past L' (the dirtied pages show), a restore dropping a field
				CHECK_MSG(b2 == b1, "%s round %d: save -> restore -> save is not byte-identical (%zu vs %zu bytes)",
				          what.c_str(), round, b2.size(), b1.size());
				cur = b2;
				if (round == 2) {
					GSeq tw = r.seq;
					if (s.prefix_len == 0 && !s.prefill100) {
						// fresh: nothing to decode from (see CONSTRUCTION above). One outcome per step on each side,
						// equal, and the two holders' states still equal after the steps (their saves).
						const std::vector<int32_t> a = StepOutcomes(e, o, 8), b = StepOutcomes(e, tw, 8);
						CHECK_MSG(a.size() == 8 && a == b, "%s: the restored holder's step outcomes [%s] differ from the original's [%s]",
						          what.c_str(), Join96(b).c_str(), Join96(a).c_str());
						CHECK_MSG(Save(e, o.h) == Save(e, tw.h), "%s: after the steps, the restored holder's save differs from the original's",
						          what.c_str());
					} else {
						const std::vector<int32_t> a = DecodeN(e, o, 8), b = DecodeN(e, tw, 8);
						CHECK_MSG(a.size() == 8 && a == b, "%s: the restored holder's next tokens differ from the original's",
						          what.c_str());
					}
				}
			}
		}
	}
}

constexpr int64_t kDedicatedCap = 256;

struct DedicatedSaved {
	std::vector<uint8_t> blob;
	std::vector<int32_t> next8;
	GSeq at_save;
};

DedicatedSaved MakeDedicated(const Env& e, bool mid) {
	DedicatedSaved d;
	SslmGpuSequenceHandle* h = nullptr;
	const GS st = sslm_gpu_seq_create(e.ctx, e.model(), kDedicatedCap, &h);
	CHECK_MSG(st == GS::SSLM_OK && h, "sslm_gpu_seq_create: %s", StatusName(st));
	if (!h) return d;
	GSeq s;
	s.h = h;
	CHECK(Prefill(e, s, Stream(31, 100, kVocab)) == GS::SSLM_OK);
	CHECK(DecodeN(e, s, 4).size() == 4);
	if (mid) EnterMidToken(e, s);
	d.blob = Save(e, h);
	d.at_save = s;
	d.next8 = DecodeN(e, s, 8);
	sslm_gpu_seq_release(e.ctx, h);
	return d;
}

void PartBC(const Env& e) {
	for (int mid = 0; mid <= 1; ++mid) {
		const char* what = mid ? "9.6 B SLM5 mid-token" : "9.6 B SLM5 resting";
		const DedicatedSaved d = MakeDedicated(e, mid != 0);
		// kills: a dedicated holder writing SLM6 (the writer rule, §4 item 10; §9.2 class D)
		CHECK_MSG(IsMagic(d.blob, "SLM5"), "%s: a dedicated holder's save is not 'SLM5'", what);
		if (!IsMagic(d.blob, "SLM5")) continue;
		SslmGpuSequenceHandle* r = OutSentinel<SslmGpuSequenceHandle>();
		const GS st = sslm_gpu_seq_restore(e.ctx, e.model(), d.blob.data(), d.blob.size(), &r);
		CHECK_MSG(st == GS::SSLM_OK && Real(r), "%s: sslm_gpu_seq_restore: %s", what, StatusName(st));
		if (!Real(r)) continue;
		CHECK_MSG(Save(e, r) == d.blob, "%s: re-save differs", what);
		GSeq t = d.at_save;
		t.h = r;
		CHECK_MSG(DecodeN(e, t, 8) == d.next8, "%s: next tokens differ from the unsaved original's", what);
		sslm_gpu_seq_release(e.ctx, r);
	}
	// C: SLM4, from B's resting state.
	{
		const char* what = "9.6 C SLM4";
		const DedicatedSaved d = MakeDedicated(e, false);
		const std::vector<uint8_t> slm4 = Slm4FromSlm5(d.blob);
		CHECK_MSG(!slm4.empty() && slm4.size() + kSlm5Tail == d.blob.size(), "%s: construction: no SLM4 from the SLM5", what);
		// Construction: the original's tail is the unbound default an SLM4 restore produces.
		const bool default_tail = static_cast<int32_t>(Le32(d.blob, 120)) == -1 && Le32(d.blob, 124) == 0xFFFFFFFFu &&
		                          Le32(d.blob, 128) == 0;
		CHECK_MSG(default_tail, "%s: construction: the SLM5 tail is not (unbound, no walk, not ready)", what);
		if (slm4.empty()) return;
		SslmGpuSequenceHandle* r = OutSentinel<SslmGpuSequenceHandle>();
		const GS st = sslm_gpu_seq_restore(e.ctx, e.model(), slm4.data(), slm4.size(), &r);
		// kills: a reader that stops accepting SLM4
		CHECK_MSG(st == GS::SSLM_OK && Real(r), "%s: sslm_gpu_seq_restore: %s", what, StatusName(st));
		if (!Real(r)) return;
		CHECK_MSG(Save(e, r) == d.blob, "%s: re-save is not the SLM5 the original wrote", what);
		GSeq t = d.at_save;
		t.h = r;
		CHECK_MSG(DecodeN(e, t, 8) == d.next8, "%s: next tokens differ from the unsaved original's", what);
		sslm_gpu_seq_release(e.ctx, r);
	}
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) {
			PartA(e);
			PartBC(e);
		}
	}
	return FinishSuite("g1e_cell_9_6");
}
