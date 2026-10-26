// Schema fast-decode plan rev 1.4.4 §11.2, cell G9.1 (slice G1-J): GPU save and restore between spans continues
// identically. After every verb call of the §5.6 loop the sequence is saved and restored into a new handle, and the
// loop continues on it; the stream equals the uninterrupted one.
//   (a) SLM5: a dedicated sequence, POTION and PROMPT_RESULT at chunk budget 64 (the stream is also cell 6.6's).
//   (b) SLM6: a pooled (budgeted) sequence, the same.
//   (c) (rev 1.4.1, audit N-3) a restore between two verb calls of one chain: LONGKEY at chunk budget 4, whose
//       26-token start chain takes seven calls. After the first, the verb has set the ready flag at full depth; the
//       restored sequence's sslm_gpu_seq_ready_for_logits reads 1, and a -1 verb call continues the chain.
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
			for (const GoldenStream& g : kGoldenStreams) {
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
				CHECK_MSG(HexDigest(got.stream) == g.digest, "[%s] %s: digest %s, pinned %s", cell, g.schema,
				          HexDigest(got.stream).c_str(), g.digest);
				Drop(rig, s.h);
			}
		}
		// (c) a restore inside one chain.
		const std::vector<int32_t> prompt = c.Tokenize(kPromptPotionText);
		GSeq plain = rig.New(kLongKey);
		GSeq s = rig.New(kLongKey);
		if (plain.h && s.h) {
			rig.Prompt(plain, prompt);
			rig.Prompt(s, prompt);
			const GJfRun want = RunGpuJf(rig, plain, 4);
			const JumpOut first = Jump(rig, s, -1, 4);
			CHECK_MSG(first.st == GS::SSLM_OK && first.count == 4 && QueryGpu(rig.ctx(), s.h, 0).count > 0,
			          "[G9.1 (c)] the first call: %s, %d forced, and the chain must go on", Name(first.st), first.count);
			if (SaveRestore(rig, s, false, "G9.1 (c)")) {
				CHECK_MSG(ReadyFlag(rig.ctx(), s.h) == 1, "[G9.1 (c)] the restored sequence's ready flag reads %d",
				          ReadyFlag(rig.ctx(), s.h));
				const JumpOut next = Jump(rig, s, -1, 4);
				CHECK_MSG(next.st == GS::SSLM_OK && next.count == 4, "[G9.1 (c)] the -1 call after the restore: %s, %d",
				          Name(next.st), next.count);
				GJfRun rest = RunGpuJf(rig, s, 4);
				std::vector<int32_t> stream = first.forced;
				stream.insert(stream.end(), next.forced.begin(), next.forced.end());
				stream.insert(stream.end(), rest.stream.begin(), rest.stream.end());
				CHECK_MSG(stream == want.stream && rest.dead_end, "[G9.1 (c)] stream %s, uninterrupted %s", Str(stream).c_str(),
				          Str(want.stream).c_str());
			}
		}
		rig.ReleaseSeqs();
		if (g_pool) sslm_gpu_kv_pool_destroy(rig.ctx(), g_pool);
	}
	return Finish("sfdg_cell_g9_1");
}
