// Engine 1.19 clean stop, cell G10 (sfdg_cleanstop.h has the contract), the GPU twin of the CPU cell CS.E8 (coverage
// audit 1.19, gap 8): jump-forward with a draft that runs past the closing token. With o_(k-1) in hand,
// sslm_gpu_seq_jump_forward under SSLM_JUMP_EXACT with the draft o_k .. o_(e-1), o_(e-1) again, <|endoftext|>. Nothing
// is legal after the closing token, so the verb returns SSLM_OK and keeps exactly the true tail o_k .. o_(e-1); the
// overrun is never committed. The sequence is ready with accepting 1 and the context the prompt plus e; the next
// decode-bridge call is -2 at SSLM_OK with accepting 1, the context unchanged, and the blob equal to the Off run's after
// its -2. Tails of 1, 2 and 5 tokens. Host head (the jump verb does not need the device head). Box only.
// Run: sfdg_cell_cs_g10.exe [--fixture-dir=DIR]
#include "sfdg_cleanstop.h"

using namespace sfdg;
using namespace sfdg::cs;

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		CRig c(rig.raw, 16);
		for (const Case& k : kCases) {
			const std::vector<int32_t> prompt = c.Tokenize(k.prompt_text);
			const Off off = RunOff(rig, rig.model(), k.schema, prompt);
			CHECK_MSG(off.ok, "[CS.G10] %s: the Off run did not reach its end", k.schema);
			if (!off.ok || off.toks.empty()) continue;
			const size_t e = off.toks.size();
			for (size_t d : {size_t(1), size_t(2), size_t(5)}) {
				if (d >= e) continue;
				const size_t kk = e - d;  // tokens decoded; o_(kk-1) in hand
				GSeq s = After(rig, rig.model(), k.schema, prompt, kk);
				if (!s.h) {
					CHECK_MSG(false, "[CS.G10] %s tail %zu: the history did not replay", k.schema, d);
					continue;
				}
				const std::vector<int32_t> tail(off.toks.begin() + static_cast<std::ptrdiff_t>(kk), off.toks.end());
				std::vector<int32_t> draft = tail;
				draft.push_back(off.toks.back());
				draft.push_back(kSpecial);
				const JumpOut j = JumpDraft(rig, s, s.last, static_cast<int32_t>(draft.size()) + 1, draft);
				CHECK_MSG(j.st == GS::SSLM_OK && j.forced == tail, "[CS.G10] %s tail %zu: the verb gave %s, kept %s from the "
				          "draft %s; want OK and exactly the tail %s", k.schema, d, Name(j.st), Str(j.forced).c_str(),
				          Str(draft).c_str(), Str(tail).c_str());
				const int64_t want_ctx = static_cast<int64_t>(prompt.size() + e);
				CHECK_MSG(ReadyFlag(rig.ctx(), s.h) == 1 && Accepting(rig.ctx(), s.h) == 1 && CtxLen(s.h) == want_ctx,
				          "[CS.G10] %s tail %zu: after the verb ready %d accepting %d context %lld; want 1, 1 and %lld",
				          k.schema, d, ReadyFlag(rig.ctx(), s.h), Accepting(rig.ctx(), s.h),
				          static_cast<long long>(CtxLen(s.h)), static_cast<long long>(want_ctx));
				GS st = GS::SSLM_OK;
				const int32_t t = Token(rig.env, s, &st);
				CHECK_MSG(st == GS::SSLM_OK && t == -2 && Accepting(rig.ctx(), s.h) == 1 && CtxLen(s.h) == want_ctx,
				          "[CS.G10] %s tail %zu: the next decode gave %s token %d accepting %d, context %lld (want %lld)",
				          k.schema, d, Name(st), t, Accepting(rig.ctx(), s.h), static_cast<long long>(CtxLen(s.h)),
				          static_cast<long long>(want_ctx));
				CHECK_MSG(rig.Blob(s.h) == off.end_blob, "[CS.G10] %s tail %zu: the end state differs from the Off run's",
				          k.schema, d);
			}
			rig.ReleaseSeqs();
		}
	}
	return Finish("sfdg_cell_cs_g10");
}
