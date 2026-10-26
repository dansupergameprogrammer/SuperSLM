// Schema fast-decode plan rev 1.4.4 §11.2, cell G11.1 (slice G1-J): GPU pending admission walks the pending token
// zero times, and jump-forward from a ready seed and after a dead end embeds nothing twice.
//   (a) A token in hand: the verb's walk state, context length and blob equal the probe's composition (the prompt
//       bridge for the pending token, which does not walk it, then the schema-content bridge for the chain). A verb
//       that walked the pending token again (XG2, XG8) leaves another state or admits nothing.
//   (b) A ready seed (after the prompt): the verb admits the start chain over the ready row, which it never
//       finishes or embeds again: context = prompt + chain, and the blob equals the schema-content bridge's.
//   (c) After a dead end (the -2 finish re-arms the ready flag): the verb with -1 admits nothing and changes
//       nothing, and the next decode gives -2 again at the same context length.
// MR5 RE-POINT (multi-row GPU plan, ruling 29 M1 as widened). 1.18 removes Canonical:
//   (b) a ready seed holds no pending token, so Exact is a no-op: OK, nothing kept, the context and the sequence as
//       the untouched twin's. Retired: "the verb admits the start chain over the ready row", which only Canonical did
//       (Exact keeps nothing without a pending token); its "never finishes or embeds the ready row again" is what the
//       no-op's unchanged sequence and context now check.
//   (a) a token in hand: the verb (NULL options: the forced chain as the draft) keeps Exact's prefix, the chain's
//       leading ids that equal Off's own next tokens (computed by decode alone on a third twin). The composition is
//       re-pointed to that prefix: the prompt bridge for the pending token (which does not walk it), then the
//       schema-content bridge for the kept ids; walk, context and blob equal it. A verb that walked the pending token
//       again (XG2, XG8) still leaves another state.
//   (c) unchanged.
// Box only. Run: sfdg_cell_g11_1.exe [--fixture-dir=DIR]
#include "sfdg_helpers.h"

using namespace sfdg;

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		CRig c(rig.raw);
		const std::vector<int32_t> prompt = c.Tokenize(kPromptPotionText);
		for (const char* schema : {kPotion, kUtf8Enum}) {
			// (b) the ready seed.
			GSeq a = rig.New(schema), b = rig.New(schema), o = rig.New(schema);
			if (!a.h || !b.h || !o.h) continue;
			rig.Prompt(a, prompt);
			rig.Prompt(b, prompt);
			rig.Prompt(o, prompt);
			const SeqObs b0 = Observe(rig, b.h);
			const JumpOut j0 = Jump(rig, a, -1, 64);
			CHECK_MSG(j0.st == GS::SSLM_OK && j0.count == 0 && CtxLen(a.h) == int64_t(prompt.size()),
			          "[G11.1 (b)] %s: %s, kept %d, context %lld (want Exact's no-op at %zu)", schema, Name(j0.st), j0.count,
			          static_cast<long long>(CtxLen(a.h)), prompt.size());
			CHECK_MSG(Observe(rig, a.h) == b0, "[G11.1 (b)] %s: the no-op changed the sequence (the untouched twin's differs)",
			          schema);
			// (a) a token in hand on all three.
			GS st = GS::SSLM_OK;
			const int32_t t1 = Token(rig.env, a, &st);
			const int32_t t1b = Token(rig.env, b, &st);
			const int32_t t1o = Token(rig.env, o, &st);
			CHECK_MSG(t1 >= 0 && t1 == t1b && t1 == t1o, "[G11.1 (a)] %s: the choice %d, %d and %d", schema, t1, t1b, t1o);
			if (t1 < 0) continue;
			const uint32_t walk_after_choice = SslmGpuSeqWalkStateForG5Bridge(a.h);
			const GQuery q1 = QueryGpu(rig.ctx(), a.h);
			CHECK_MSG(q1.count > 0, "[G11.1 (a)] %s: precondition: the choice %d leads to a forced state", schema, t1);
			if (q1.count <= 0) continue;
			const std::vector<int32_t> offn = OffTokensGpu(rig, o, static_cast<int64_t>(q1.ids.size()) + 1);
			const int32_t kept = ExactKeptPrefix(q1.ids, offn, 63);
			const std::vector<int32_t> want(q1.ids.begin(), q1.ids.begin() + kept);
			const int64_t ctx0 = CtxLen(a.h);
			const JumpOut j1 = Jump(rig, a, t1, 64);
			Compose(rig, b, t1, want);
			CHECK_MSG(j1.st == GS::SSLM_OK && j1.forced == want, "[G11.1 (a)] %s: %s, kept %s; the chain from state %u is %s, "
			          "Off's next %s, so Exact keeps %s", schema, Name(j1.st), Str(j1.forced).c_str(), walk_after_choice,
			          Str(q1.ids).c_str(), Str(offn).c_str(), Str(want).c_str());
			CHECK_MSG(CtxLen(a.h) == ctx0 + 1 + int64_t(want.size()), "[G11.1 (a)] %s: context %lld, want %lld", schema,
			          static_cast<long long>(CtxLen(a.h)), static_cast<long long>(ctx0 + 1 + int64_t(want.size())));
			const SeqObs oa = Observe(rig, a.h), ob = Observe(rig, b.h);
			CHECK_MSG(oa == ob, "[G11.1 (a)] %s: the verb's walk %u context %lld ready %d layer %u, the composition's walk %u "
			          "context %lld ready %d layer %u (blobs %s)", schema, oa.walk, static_cast<long long>(oa.ctx), oa.ready, oa.layer,
			          ob.walk, static_cast<long long>(ob.ctx), ob.ready, ob.layer, oa.blob == ob.blob ? "equal" : "differ");
			const int32_t na = Token(rig.env, a, &st);
			CHECK_MSG(st == GS::SSLM_OK && static_cast<size_t>(kept) < offn.size() && na == offn[static_cast<size_t>(kept)],
			          "[G11.1 (a)] %s: the next decode %d, Off's %d", schema, na,
			          static_cast<size_t>(kept) < offn.size() ? offn[static_cast<size_t>(kept)] : -9);
		}
		// (c) after a dead end.
		GSeq d = rig.New(kPotion);
		if (d.h) {
			rig.Prompt(d, prompt);
			const GJfRun r = RunGpuJf(rig, d, 64);
			CHECK_MSG(r.dead_end, "[G11.1 (c)] the loop ended with %s and no dead end", Name(r.end));
			if (r.dead_end) {
				const SeqObs before = Observe(rig, d.h);
				CHECK_MSG(before.ready == 1, "[G11.1 (c)] the dead end left the ready flag %d", before.ready);
				const JumpOut j = Jump(rig, d, -1, 64);
				CHECK_MSG(j.st == GS::SSLM_OK && j.count == 0 && Observe(rig, d.h) == before,
				          "[G11.1 (c)] the verb after the dead end: %s, %d forced, sequence %s", Name(j.st), j.count,
				          Observe(rig, d.h) == before ? "unchanged" : "changed");
				GS st = GS::SSLM_OK;
				const int32_t t = Token(rig.env, d, &st);
				CHECK_MSG(st == GS::SSLM_OK && t == -2 && CtxLen(d.h) == before.ctx,
				          "[G11.1 (c)] the decode after: %s token %d, context %lld (was %lld)", Name(st), t,
				          static_cast<long long>(CtxLen(d.h)), static_cast<long long>(before.ctx));
			}
		}
		rig.ReleaseSeqs();
	}
	return Finish("sfdg_cell_g11_1");
}
