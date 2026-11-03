// Engine 1.19 clean stop, cell G6 (sfdg_cleanstop.h has the contract): jump-forward reaching the end.
//   (a) With o_(k-1) in hand, sslm_gpu_seq_jump_forward under SSLM_JUMP_EXACT with the true tail o_k .. o_(e-1) (through
//       the closing token) as the draft: SSLM_OK, the whole tail kept, the sequence ready with accepting 1. The verb
//       never makes the -2 itself; the next decode-bridge call is -2 at SSLM_OK with accepting 1, the context unchanged
//       by it, and the blob equal to the Off run's after its -2. Tails of 1, 2 and 5 tokens.
//   (b) The NULL-options host loop (RunGpuJf) ends at the decode's -2 with every verb call SSLM_OK, accepting 1, the
//       CPU's stream, and the Off run's end blob; a verb call after the end is a no-op and the next decode is -2 again.
// Host head (the jump verb does not need the device head). Box only. Run: sfdg_cell_cs_g6.exe [--fixture-dir=DIR]
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
			CHECK_MSG(off.ok, "[CS.G6] %s: the Off run did not reach its end", k.schema);
			if (!off.ok) continue;
			const size_t e = off.toks.size();
			for (size_t d : {size_t(1), size_t(2), size_t(5)}) {
				if (d >= e) continue;
				const size_t kk = e - d;  // tokens decoded; o_(kk-1) in hand
				GSeq s = After(rig, rig.model(), k.schema, prompt, kk);
				if (!s.h) {
					CHECK_MSG(false, "[CS.G6 (a)] %s tail %zu: the history did not replay", k.schema, d);
					continue;
				}
				const std::vector<int32_t> draft(off.toks.begin() + static_cast<std::ptrdiff_t>(kk), off.toks.end());
				const JumpOut j = JumpDraft(rig, s, s.last, static_cast<int32_t>(draft.size()) + 1, draft);
				CHECK_MSG(j.st == GS::SSLM_OK && j.forced == draft, "[CS.G6 (a)] %s tail %zu: the verb gave %s, kept %s; want OK "
				          "and the tail %s", k.schema, d, Name(j.st), Str(j.forced).c_str(), Str(draft).c_str());
				CHECK_MSG(ReadyFlag(rig.ctx(), s.h) == 1 && Accepting(rig.ctx(), s.h) == 1, "[CS.G6 (a)] %s tail %zu: after the "
				          "verb ready %d accepting %d, want 1 and 1", k.schema, d, ReadyFlag(rig.ctx(), s.h),
				          Accepting(rig.ctx(), s.h));
				const int64_t ctx0 = CtxLen(s.h);
				GS st = GS::SSLM_OK;
				const int32_t t = Token(rig.env, s, &st);
				CHECK_MSG(st == GS::SSLM_OK && t == -2 && Accepting(rig.ctx(), s.h) == 1 && CtxLen(s.h) == ctx0,
				          "[CS.G6 (a)] %s tail %zu: the next decode gave %s token %d accepting %d, context %lld -> %lld", k.schema,
				          d, Name(st), t, Accepting(rig.ctx(), s.h), static_cast<long long>(ctx0),
				          static_cast<long long>(CtxLen(s.h)));
				CHECK_MSG(rig.Blob(s.h) == off.end_blob, "[CS.G6 (a)] %s tail %zu: the end state differs from the Off run's",
				          k.schema, d);
			}
			{
				const CpuRun want = RunCpu(c, k.schema, prompt, /*sliced=*/false);
				GSeq s = rig.New(k.schema);
				if (!s.h || rig.Prompt(s, prompt) != GS::SSLM_OK) continue;
				const GJfRun r = RunGpuJf(rig, s, 64);
				CHECK_MSG(r.dead_end && r.end == GS::SSLM_OK && Accepting(rig.ctx(), s.h) == 1, "[CS.G6 (b)] %s: the loop ended "
				          "with %s, dead end %d, accepting %d", k.schema, Name(r.end), r.dead_end, Accepting(rig.ctx(), s.h));
				CHECK_MSG(r.stream == want.stream && rig.Blob(s.h) == off.end_blob, "[CS.G6 (b)] %s: stream %s (the CPU's %s), "
				          "end state %s the Off run's", k.schema, Str(r.stream).c_str(), Str(want.stream).c_str(),
				          rig.Blob(s.h) == off.end_blob ? "equal to" : "different from");
				const std::vector<uint8_t> b = rig.Blob(s.h);
				const JumpOut j = Jump(rig, s, -1, 64);
				GS st = GS::SSLM_OK;
				const int32_t t = Token(rig.env, s, &st);
				CHECK_MSG(j.st == GS::SSLM_OK && j.count == 0 && st == GS::SSLM_OK && t == -2 && rig.Blob(s.h) == b,
				          "[CS.G6 (b)] %s: after the end the verb gave %s (%d kept) and the decode %s token %d, sequence %s",
				          k.schema, Name(j.st), j.count, Name(st), t, rig.Blob(s.h) == b ? "unchanged" : "changed");
			}
			rig.ReleaseSeqs();
		}
	}
	return Finish("sfdg_cell_cs_g6");
}
