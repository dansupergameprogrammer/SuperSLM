// Schema fast-decode plan rev 1.4.4 §11.2, cell G9.1 (slice G1-J): GPU save and restore between spans continues
// identically. After every verb call of the §5.6 loop the sequence is saved and restored into a new handle, and the
// loop continues on it; the stream equals the uninterrupted one.
//   (a) SLM5: a dedicated sequence, POTION and PROMPT_RESULT at chunk budget 64 (the stream is also cell 6.6's).
//   (b) SLM6: a pooled (budgeted) sequence, the same.
//   (c) (rev 1.4.1, audit N-3) a restore between two verb calls of one chain: LONGKEY at chunk budget 4, whose
//       26-token start chain takes seven calls. After the first, the verb has set the ready flag at full depth; the
//       restored sequence's sslm_gpu_seq_ready_for_logits reads 1, and a -1 verb call continues the chain.
// MR5 RE-POINT (multi-row GPU plan, ruling 29 M1 as widened). 1.18 removes Canonical:
//   (a), (b) the loop is Exact's host loop (RunGpuJf), and the pinned digest is cell 6.X5's Off stream (kOffGoldens:
//       33 and 159 tokens), not 1.17's Canonical stream (kGoldenStreams). The restore checks are unchanged.
//   (c) a restore inside one jump loop, between two Exact passes: LONGKEY at chunk budget 4, the token in hand and a
//       draft of Off's own next three tokens (computed by decode alone), so the first call keeps 3. The restored
//       sequence's ready flag reads 1 (unchanged). Retired: "a -1 call continues the chain", which only Canonical had
//       (it admitted the rest of a forced chain without a pending token); a -1 call on the restored ready shape is
//       now Exact's no-op (OK, nothing kept, nothing changed), checked instead, and the loop continues with the next
//       decode and a second Exact call. The stream equals Off's.
// Box only. Run: sfdg_cell_g9_1.exe [--fixture-dir=DIR]
#include "sfdg_helpers.h"

#include <algorithm>

using namespace sfdg;

namespace {

SslmGpuKvPoolHandle* g_pool = nullptr;

// Release `h` and drop it from rig.seqs, so a pooled sequence gives its pages back to the pool.
void Drop(GRig& rig, SslmGpuSequenceHandle* h) {
	if (!h) return;
	rig.seqs.erase(std::remove(rig.seqs.begin(), rig.seqs.end(), h), rig.seqs.end());
	sslm_gpu_seq_release(rig.ctx(), h);
}

// Save `s` and restore it into a new handle (SLM5, or SLM6 into the pool), replacing and releasing s.h. The pool
// holds a few sequences, so a run of restores must not keep the replaced handles' pages.
bool SaveRestore(GRig& rig, GSeq& s, bool pooled, const char* cell) {
	const std::vector<uint8_t> blob = rig.Blob(s.h);
	SslmGpuSequenceHandle* r = nullptr;
	const GS st = pooled ? sslm_gpu_seq_restore_into_pool(rig.ctx(), rig.model(), g_pool, blob.data(), blob.size(),
	                                                      nullptr, &r, nullptr)
	                     : sslm_gpu_seq_restore(rig.ctx(), rig.model(), blob.data(), blob.size(), &r);
	CHECK_MSG(!blob.empty() && st == GS::SSLM_OK && r, "[%s] %s restore: %s", cell, pooled ? "SLM6" : "SLM5", Name(st));
	if (st != GS::SSLM_OK || !r) return false;
	CHECK_MSG(rig.Blob(r) == blob, "[%s] the restored sequence saves a different blob", cell);
	rig.seqs.push_back(r);
	Drop(rig, s.h);
	s.h = r;
	return true;
}

GSeq NewSeq(GRig& rig, const char* schema, bool pooled) {
	if (!pooled) return rig.New(schema);
	GSeq s;
	const GS st = sslm_gpu_seq_create_budgeted(rig.ctx(), rig.model(), g_pool, 255, &s.h);
	CHECK_MSG(st == GS::SSLM_OK && s.h, "[G9.1] budgeted create: %s", Name(st));
	if (!s.h) return s;
	rig.seqs.push_back(s.h);
	const GS b = SslmGpuSeqSetSchemaForG5Bridge(rig.ctx(), s.h, rig.Index(schema));
	CHECK_MSG(b == GS::SSLM_OK, "[G9.1] bind %s: %s", schema, Name(b));
	return s;
}

}  // namespace

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		CRig c(rig.raw);
		const GS ps = sslm_gpu_kv_pool_create(rig.ctx(), rig.model(), 64, &g_pool);
		CHECK_MSG(ps == GS::SSLM_OK && g_pool, "[G9.1] pool: %s", Name(ps));
		for (const bool pooled : {false, true}) {
			const char* cell = pooled ? "G9.1 (b) SLM6" : "G9.1 (a) SLM5";
			if (pooled && !g_pool) continue;
			for (const GoldenStream& g : kOffGoldens) {
				const bool potion = std::string(g.schema) == kPotion;
				const std::vector<int32_t> prompt = c.Tokenize(potion ? kPromptPotionText : kPromptPrText);
				GSeq plain = NewSeq(rig, g.schema, pooled);
				if (!plain.h) continue;
				rig.Prompt(plain, prompt);
				const GJfRun want = RunGpuJf(rig, plain, 64);
				Drop(rig, plain.h);
				GSeq s = NewSeq(rig, g.schema, pooled);
				if (!s.h) continue;
				rig.Prompt(s, prompt);
				int restores = 0;
				const GJfRun got = RunGpuJf(rig, s, 64, 400, [&](GSeq& q, int) {
					if (SaveRestore(rig, q, pooled, cell)) ++restores;
				});
				CHECK_MSG(got.stream == want.stream && got.dead_end && restores > 0,
				          "[%s] %s: %d restores; stream %s, uninterrupted %s", cell, g.schema, restores,
				          Str(got.stream).c_str(), Str(want.stream).c_str());
				CHECK_MSG(got.stream.size() == g.length && HexDigest(got.stream) == g.digest, "[%s] %s: %zu tokens, digest %s; "
				          "Off's pinned %zu, %s", cell, g.schema, got.stream.size(), HexDigest(got.stream).c_str(), g.length, g.digest);
				Drop(rig, s.h);
			}
		}
		// (c) a restore between two Exact calls of one loop (MR5 re-point).
		const std::vector<int32_t> prompt = c.Tokenize(kPromptPotionText);
		GSeq o = rig.New(kLongKey);
		GSeq s = rig.New(kLongKey);
		if (o.h && s.h) {
			rig.Prompt(o, prompt);
			rig.Prompt(s, prompt);
			// Off's stream, by decode alone: off[0] is the token the first decode puts in hand.
			std::vector<int32_t> off = OffTokensGpu(rig, o, 400);
			const bool dead = !off.empty() && off.back() == -2;
			if (dead) off.pop_back();
			CHECK_MSG(dead && off.size() >= 9, "[G9.1 (c)] precondition: Off's LONGKEY stream reaches its dead end after 9 or "
			          "more tokens (%zu, dead end %d)", off.size(), dead);
			if (dead && off.size() >= 9) {
				std::vector<int32_t> stream;
				GS st = GS::SSLM_OK;
				const int32_t p = Token(rig.env, s, &st);
				stream.push_back(p);
				const std::vector<int32_t> d1(off.begin() + 1, off.begin() + 4);
				const JumpOut first = JumpDraft(rig, s, p, 4, d1);
				CHECK_MSG(st == GS::SSLM_OK && p == off[0] && first.st == GS::SSLM_OK && first.forced == d1,
				          "[G9.1 (c)] the first call: the token %d (Off's %d), %s, kept %s (Off's own %s)", p, off[0],
				          Name(first.st), Str(first.forced).c_str(), Str(d1).c_str());
				stream.insert(stream.end(), first.forced.begin(), first.forced.end());
				if (SaveRestore(rig, s, false, "G9.1 (c)")) {
					CHECK_MSG(ReadyFlag(rig.ctx(), s.h) == 1, "[G9.1 (c)] the restored sequence's ready flag reads %d",
					          ReadyFlag(rig.ctx(), s.h));
					s.ready = s.primed = true;
					s.mid = false;
					const SeqObs before = Observe(rig, s.h);
					const JumpOut noop = Jump(rig, s, -1, 4);
					CHECK_MSG(noop.st == GS::SSLM_OK && noop.count == 0 && Observe(rig, s.h) == before,
					          "[G9.1 (c)] the -1 call after the restore: %s, %d kept, sequence %s (want Exact's no-op)",
					          Name(noop.st), noop.count, Observe(rig, s.h) == before ? "unchanged" : "changed");
					const int32_t p2 = Token(rig.env, s, &st);
					stream.push_back(p2);
					const std::vector<int32_t> d2(off.begin() + 5, off.begin() + 8);
					const JumpOut next = JumpDraft(rig, s, p2, 4, d2);
					CHECK_MSG(st == GS::SSLM_OK && next.st == GS::SSLM_OK && next.forced == d2,
					          "[G9.1 (c)] the second call after the restore: the token %d, %s, kept %s (Off's own %s)", p2,
					          Name(next.st), Str(next.forced).c_str(), Str(d2).c_str());
					stream.insert(stream.end(), next.forced.begin(), next.forced.end());
					std::vector<int32_t> rest = OffTokensGpu(rig, s, 400);
					const bool rest_dead = !rest.empty() && rest.back() == -2;
					if (rest_dead) rest.pop_back();
					stream.insert(stream.end(), rest.begin(), rest.end());
					CHECK_MSG(stream == off && rest_dead, "[G9.1 (c)] stream %s, Off's %s", Str(stream).c_str(), Str(off).c_str());
				}
			}
		}
		rig.ReleaseSeqs();
		if (g_pool) sslm_gpu_kv_pool_destroy(rig.ctx(), g_pool);
	}
	return Finish("sfdg_cell_g9_1");
}
