// Schema fast-decode plan rev 1.4.4 §11.2, the verb's cells (slice G1-J): G12.1, G12.2, G12.3, G12.8 and G12.9.
//   G12.1 Fused admission. A pending token and a chain in one call make ceil(R/4) layer submissions (S-GSUB), where
//         the probe's composition (the prompt bridge for the pending token, then the schema-content bridge) makes
//         1 + ceil((R-1)/4). Tokens, walk and the SLM5 and SLM6 blobs equal the composition's; so does
//         sslm_gpu_seq_read_prefill_final_hidden after it (rev 1.4.1, audit G-7).
//   G12.2 The walked embed (Q25 A). The chain fed row by row through sslm_gpu_seq_embed_schema_token and layer
//         slices of sslm_decode_step_gpu, then one finish, gives the verb's token, walk and blob. An illegal token
//         and an unbound sequence are refused with nothing changed. From the Ready shape the walked embed clears the
//         ready flag, so the next decode bridge call embeds and finishes the new row (XG31).
//   G12.3 Entry shapes: every row of §7.3's table, the primed row included; a valid but wrong pending token is
//         embedded as given; after every refusal and after a query a fresh sequence is still bindable, and a walked
//         embed clears bind eligibility; after sslm_gpu_seq_embed_token ("embedded, no layers run") a pending token
//         re-embeds over it.
//   G12.8 The ready getter: 0 with a token in hand and in the primed row, 1 after an admitting verb call and after
//         an SLM5 or SLM6 restore of that state; KV_BUFFER_MISMATCH and BUSY arms; no submission (S-GSUB).
//   G12.9 Exact mode refused (rev 1.4.4): SSLM_JUMP_EXACT with a draft and with NULL, from each of the first three
//         shapes, is SSLM_SEQUENCE_REJECTED with the sequence unchanged (blob, walk, flags, bindability) and no
//         submission; the next canonical call equals a fresh one's (XG37).
// Box only. Run: sfdg_cell_g12_verb.exe [--fixture-dir=DIR] [--only=G12.N]
#include "sfdg_helpers.h"

using namespace sfdg;

namespace {

constexpr int32_t kExact = 1;  // SSLM_JUMP_EXACT, the CPU enum's value

bool Only(int argc, char** argv, const char* cell) {
	for (int i = 1; i < argc; ++i)
		if (std::strncmp(argv[i], "--only=", 7) == 0) return std::strcmp(argv[i] + 7, cell) == 0;
	return true;
}

SslmGpuKvPoolHandle* g_pool = nullptr;

GSeq NewSeq(GRig& rig, const char* schema, bool pooled) {
	if (!pooled) return rig.New(schema);
	GSeq s;
	const GS st = sslm_gpu_seq_create_budgeted(rig.ctx(), rig.model(), g_pool, 255, &s.h);
	CHECK_MSG(st == GS::SSLM_OK && s.h, "budgeted create: %s", Name(st));
	if (!s.h) return s;
	rig.seqs.push_back(s.h);
	if (schema) {
		const GS b = SslmGpuSeqSetSchemaForG5Bridge(rig.ctx(), s.h, rig.Index(schema));
		CHECK_MSG(b == GS::SSLM_OK, "bind %s: %s", schema, Name(b));
	}
	return s;
}

// A sequence with a produced token in hand: the prompt, the start chain by the verb, one decode. Returns the token.
int32_t TokenInHand(GRig& rig, GSeq& s, const std::vector<int32_t>& prompt) {
	rig.Prompt(s, prompt);
	Jump(rig, s, -1, 64);
	GS st = GS::SSLM_OK;
	const int32_t t = Token(rig.env, s, &st);
	CHECK_MSG(st == GS::SSLM_OK && t >= 0, "token in hand: %s %d", Name(st), t);
	return t;
}

// Drive the embedded row's layers in slices of `layers_per_call` layers, drained after each call.
GS DriveSliced(GRig& rig, GSeq& s, uint32_t layers_per_call) {
	const uint32_t L = rig.env.fx.layers;
	const uint32_t budget = rig.env.fx.one_layer_budget * layers_per_call;
	for (int guard = 0; LayerIdx(s.h) < L && guard < 64; ++guard) {
		GS st = sslm_decode_step_gpu(rig.ctx(), s.h, nullptr, budget);
		if (st != GS::SSLM_OK) return st;
		st = Drain(rig.env, s.h);
		if (st != GS::SSLM_OK) return st;
	}
	return LayerIdx(s.h) == L ? GS::SSLM_OK : GS::SSLM_SEQUENCE_REJECTED;
}

// The primed row (G30): the prompt's last row at full depth with the ready flag clear. Every prompt token is
// embedded and driven to depth by hand, as the plugin's prefill does, and the last is never finished. No bridge
// runs: the prompt bridge sets the ready flag, and sslm_gpu_seq_embed_token does not clear it.
void Primed(GRig& rig, GSeq& s, const std::vector<int32_t>& prompt) {
	for (const int32_t t : prompt) {
		if (sslm_gpu_seq_embed_token(rig.ctx(), s.h, t) != GS::SSLM_OK || DriveSliced(rig, s, 4) != GS::SSLM_OK) {
			CHECK_MSG(false, "primed: embed and drive %d", t);
			return;
		}
	}
	s.ready = false;
	s.last = prompt.back();
}

// ---- G12.1 -----------------------------------------------------------------------------------------------------
void Cell_12_1(GRig& rig, const std::vector<int32_t>& prompt) {
	for (const bool pooled : {false, true}) {
		const char* arm = pooled ? "SLM6" : "SLM5";
		GSeq a = NewSeq(rig, kPotion, pooled), b = NewSeq(rig, kPotion, pooled);
		if (!a.h || !b.h) continue;
		const int32_t t = TokenInHand(rig, a, prompt);
		const int32_t tb = TokenInHand(rig, b, prompt);
		CHECK_MSG(t == tb, "[G12.1 %s] the twins chose %d and %d", arm, t, tb);
		const GQuery q = QueryGpu(rig.ctx(), a.h);
		const uint64_t R = 1 + q.ids.size();
		CHECK_MSG(q.count > 0, "[G12.1 %s] precondition: a chain after the choice %d", arm, t);
		uint64_t n0 = Submissions(rig.ctx());
		const JumpOut j = Jump(rig, a, t, 64);
		const uint64_t verb_subs = Submissions(rig.ctx()) - n0;
		n0 = Submissions(rig.ctx());
		const GS cs = Compose(rig, b, tb, q.ids);
		const uint64_t comp_subs = Submissions(rig.ctx()) - n0;
		CHECK_MSG(j.st == GS::SSLM_OK && cs == GS::SSLM_OK && j.forced == q.ids, "[G12.1 %s] verb %s forced %s, composition %s",
		          arm, Name(j.st), Str(j.forced).c_str(), Name(cs));
		CHECK_MSG(verb_subs == (R + 3) / 4, "[G12.1 %s] R = %llu rows: the verb made %llu submissions, want ceil(R/4) = %llu",
		          arm, static_cast<unsigned long long>(R), static_cast<unsigned long long>(verb_subs),
		          static_cast<unsigned long long>((R + 3) / 4));
		CHECK_MSG(comp_subs == 1 + (R - 1 + 3) / 4, "[G12.1 %s] the composition made %llu submissions, want %llu", arm,
		          static_cast<unsigned long long>(comp_subs), static_cast<unsigned long long>(1 + (R - 1 + 3) / 4));
		CHECK_MSG(Observe(rig, a.h) == Observe(rig, b.h), "[G12.1 %s] the verb's sequence (blob, walk, flags) differs from "
		          "the composition's", arm);
		CHECK_MSG(ReadSnapshot(rig.ctx(), a.h) == ReadSnapshot(rig.ctx(), b.h) &&
		              ReadSnapshot(rig.ctx(), a.h).st == GS::SSLM_OK,
		          "[G12.1 %s] the prefill snapshot after the verb differs from the composition's (%s, %s)", arm,
		          Name(ReadSnapshot(rig.ctx(), a.h).st), Name(ReadSnapshot(rig.ctx(), b.h).st));
		GS st = GS::SSLM_OK;
		const int32_t na = Token(rig.env, a, &st), nb = Token(rig.env, b, &st);
		CHECK_MSG(na == nb && na != -1, "[G12.1 %s] the next tokens %d and %d", arm, na, nb);
		CHECK_MSG(rig.Blob(a.h) == rig.Blob(b.h), "[G12.1 %s] the blobs differ after the next decode", arm);
	}
}

// ---- G12.2 -----------------------------------------------------------------------------------------------------
void Cell_12_2(GRig& rig, const std::vector<int32_t>& prompt) {
	const superslm::SslmModelView& view = rig.env.fx.view;
	GSeq a = rig.New(kPotion), b = rig.New(kPotion);
	if (!a.h || !b.h) return;
	const int32_t t = TokenInHand(rig, a, prompt);
	TokenInHand(rig, b, prompt);
	const GQuery q = QueryGpu(rig.ctx(), a.h);
	CHECK_MSG(q.count > 1, "[G12.2] precondition: a chain of two or more after the choice (%d)", q.count);
	// The verb.
	Jump(rig, a, t, 64);
	GS st = GS::SSLM_OK;
	const int32_t next_verb = Token(rig.env, a, &st);
	// The sliced path: the pending token by the plain embed (its finish walked it), the chain by the walked embed.
	CHECK_MSG(sslm_gpu_seq_embed_token(rig.ctx(), b.h, t) == GS::SSLM_OK && DriveSliced(rig, b, 4) == GS::SSLM_OK,
	          "[G12.2] the pending row");
	for (int32_t c : q.ids) {
		const GS e = sslm_gpu_seq_embed_schema_token(rig.ctx(), b.h, c);
		CHECK_MSG(e == GS::SSLM_OK && ReadyFlag(rig.ctx(), b.h) == 0, "[G12.2] walked embed %d: %s", c, Name(e));
		CHECK_MSG(DriveSliced(rig, b, 4) == GS::SSLM_OK, "[G12.2] the row of %d", c);
	}
	int32_t next_sliced = -9;
	st = SslmGpuSeqFinishTokenForG5Bridge(rig.ctx(), b.h, &next_sliced);
	CHECK_MSG(st == GS::SSLM_OK && next_sliced == next_verb, "[G12.2] sliced %s token %d, verb %d", Name(st), next_sliced,
	          next_verb);
	CHECK_MSG(SslmGpuSeqWalkStateForG5Bridge(a.h) == SslmGpuSeqWalkStateForG5Bridge(b.h) && rig.Blob(a.h) == rig.Blob(b.h),
	          "[G12.2] after the next token the walks are %u and %u, blobs %s", SslmGpuSeqWalkStateForG5Bridge(a.h),
	          SslmGpuSeqWalkStateForG5Bridge(b.h), rig.Blob(a.h) == rig.Blob(b.h) ? "equal" : "differ");
	// Refusals, nothing changed: an illegal token at the walk state, and an unbound sequence.
	{
		const uint32_t w = SslmGpuSeqWalkStateForG5Bridge(b.h);
		int32_t illegal = -1;
		for (int32_t id = 0; id < 4096 && illegal < 0; ++id)
			if (!LegalAt(view, kPotion, w, id)) illegal = id;
		const SeqObs before = Observe(rig, b.h);
		const GS e = sslm_gpu_seq_embed_schema_token(rig.ctx(), b.h, illegal);
		CHECK_MSG(illegal >= 0 && e == GS::SSLM_SEQUENCE_REJECTED && Observe(rig, b.h) == before,
		          "[G12.2] illegal token %d at state %u: %s, sequence %s", illegal, w, Name(e),
		          Observe(rig, b.h) == before ? "unchanged" : "changed");
		GSeq u = rig.New(nullptr);
		if (u.h) {
			rig.Prompt(u, prompt);
			const SeqObs ub = Observe(rig, u.h);
			const GS eu = sslm_gpu_seq_embed_schema_token(rig.ctx(), u.h, prompt[0]);
			CHECK_MSG(eu == GS::SSLM_SEQUENCE_REJECTED && Observe(rig, u.h) == ub, "[G12.2] unbound: %s", Name(eu));
		}
	}
	// The Ready arm (rev 1.4.2, XG31): the walked embed clears the ready flag; the decode bridge then embeds and
	// finishes the new row, as the verb's one-row admission of the same token does.
	GSeq r = rig.New(kPotion), v = rig.New(kPotion);
	if (r.h && v.h) {
		rig.Prompt(r, prompt);
		rig.Prompt(v, prompt);
		const GQuery qs = QueryGpu(rig.ctx(), r.h);
		if (qs.count > 0) {
			const int32_t c0 = qs.ids[0];
			const GS e = sslm_gpu_seq_embed_schema_token(rig.ctx(), r.h, c0);
			CHECK_MSG(e == GS::SSLM_OK && ReadyFlag(rig.ctx(), r.h) == 0, "[G12.2 Ready] walked embed: %s, ready %d", Name(e),
			          ReadyFlag(rig.ctx(), r.h));
			r.last = c0;
			const int32_t tr = Token(rig.env, r, &st);
			Jump(rig, v, -1, 1);
			const int32_t tv = Token(rig.env, v, &st);
			CHECK_MSG(tr == tv && tr != -1, "[G12.2 Ready] the decode after the walked embed gave %d; the verb's one row %d",
			          tr, tv);
		}
	}
}

// ---- G12.3 -----------------------------------------------------------------------------------------------------
void Refused(const char* what, GRig& rig, GSeq& s, int32_t pending, GS want, const SslmGpuJumpOptions* o = nullptr) {
	const SeqObs before = Observe(rig, s.h);
	const JumpOut j = Jump(rig, s, pending, 8, o);
	CHECK_MSG(j.st == want && Observe(rig, s.h) == before, "[G12.3] %s: %s (want %s), sequence %s", what, Name(j.st),
	          Name(want), Observe(rig, s.h) == before ? "unchanged" : "changed");
}

void Cell_12_3(GRig& rig, const std::vector<int32_t>& prompt) {
	// Fresh and bound: refusals leave it bindable; a query leaves it bindable; -1 admits the start span.
	{
		GSeq f = rig.New(kPotion);
		if (f.h) {
			Refused("fresh, a token given", rig, f, prompt[0], GS::SSLM_SEQUENCE_REJECTED);
			SslmGpuJumpOptions bad = Opts(0);
			bad.struct_size = 1;
			Refused("fresh, malformed options", rig, f, -1, GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, &bad);
			Refused("fresh, pending out of range", rig, f, 1 << 30, GS::SSLM_TOKEN_ID_OUT_OF_RANGE);
			(void)QueryGpu(rig.ctx(), f.h);
			CHECK_MSG(rig.Bindable(f.h, kPotion), "[G12.3] fresh: not bindable after the refusals and a query");
			GSeq g = rig.New(kPotion);
			const GQuery q = QueryGpu(rig.ctx(), f.h);
			const JumpOut j = Jump(rig, f, -1, 64);
			if (g.h) rig.Content(g, q.ids);
			CHECK_MSG(j.st == GS::SSLM_OK && j.forced == q.ids && CtxLen(f.h) == int64_t(q.ids.size()) &&
			              (g.h == nullptr || Observe(rig, f.h) == Observe(rig, g.h)),
			          "[G12.3] fresh, -1: %s, forced %s (start chain %s)", Name(j.st), Str(j.forced).c_str(),
			          Str(q.ids).c_str());
			CHECK_MSG(!rig.Bindable(f.h, kPotion), "[G12.3] fresh: still bindable after an admitting call");
		}
		GSeq w = rig.New(kPotion);
		if (w.h) {
			const GQuery q = QueryGpu(rig.ctx(), w.h);
			if (q.count > 0) CHECK_MSG(sslm_gpu_seq_embed_schema_token(rig.ctx(), w.h, q.ids[0]) == GS::SSLM_OK, "[G12.3] walked embed");
			CHECK_MSG(!rig.Bindable(w.h, kPotion), "[G12.3] a walked embed left the sequence bindable");
		}
	}
	// Ready: a token given is refused.
	{
		GSeq r = rig.New(kPotion);
		if (r.h) {
			rig.Prompt(r, prompt);
			Refused("ready, a token given", rig, r, prompt[0], GS::SSLM_SEQUENCE_REJECTED);
		}
	}
	// Primed (G30): -1 admits the chain and equals the ready shape's admission; a token given is refused.
	{
		GSeq p = rig.New(kPotion), r = rig.New(kPotion);
		if (p.h && r.h) {
			Primed(rig, p, prompt);
			CHECK_MSG(ReadyFlag(rig.ctx(), p.h) == 0 && LayerIdx(p.h) == rig.env.fx.layers, "[G12.3] primed: ready %d layer %u",
			          ReadyFlag(rig.ctx(), p.h), LayerIdx(p.h));
			Refused("primed, a token given", rig, p, prompt[0], GS::SSLM_SEQUENCE_REJECTED);
			rig.Prompt(r, prompt);
			const JumpOut jp = Jump(rig, p, -1, 64), jr = Jump(rig, r, -1, 64);
			CHECK_MSG(jp.st == GS::SSLM_OK && jp.forced == jr.forced && rig.Blob(p.h) == rig.Blob(r.h),
			          "[G12.3] primed, -1: %s forced %s; ready %s", Name(jp.st), Str(jp.forced).c_str(), Str(jr.forced).c_str());
		}
	}
	// A token in hand: -1 refused; a valid but wrong token embedded as given; after sslm_gpu_seq_embed_token the
	// pending token re-embeds over it.
	{
		GSeq a = rig.New(kPotion), b = rig.New(kPotion), c = rig.New(kPotion);
		if (a.h && b.h && c.h) {
			const int32_t t = TokenInHand(rig, a, prompt);
			TokenInHand(rig, b, prompt);
			TokenInHand(rig, c, prompt);
			Refused("token in hand, -1", rig, a, -1, GS::SSLM_SEQUENCE_REJECTED);
			const int32_t wrong = t == prompt[0] ? prompt[1] : prompt[0];
			const GQuery q = QueryGpu(rig.ctx(), b.h);
			GSeq bw = rig.New(kPotion);
			if (bw.h) {
				TokenInHand(rig, bw, prompt);
				const JumpOut jw = Jump(rig, b, wrong, 64);
				Compose(rig, bw, wrong, q.ids);
				CHECK_MSG(jw.st == GS::SSLM_OK && Observe(rig, b.h) == Observe(rig, bw.h),
				          "[G12.3] a wrong pending token (%d, not %d) is not embedded as given: %s", wrong, t, Name(jw.st));
			}
			CHECK_MSG(sslm_gpu_seq_embed_token(rig.ctx(), c.h, wrong) == GS::SSLM_OK, "[G12.3] embed_token");
			GSeq d = rig.New(kPotion);
			if (d.h) {
				TokenInHand(rig, d, prompt);
				const JumpOut jc = Jump(rig, c, t, 64), jd = Jump(rig, d, t, 64);
				CHECK_MSG(jc.st == GS::SSLM_OK && jc.forced == jd.forced && rig.Blob(c.h) == rig.Blob(d.h),
				          "[G12.3] embedded, no layers run: the pending token did not re-embed over it (%s)", Name(jc.st));
			}
		}
	}
	// Mid-token: refused. Submitted: BUSY.
	{
		GSeq m = rig.New(kPotion);
		if (m.h) {
			const int32_t t = TokenInHand(rig, m, prompt);
			CHECK_MSG(sslm_gpu_seq_embed_token(rig.ctx(), m.h, t) == GS::SSLM_OK, "[G12.3] mid: embed");
			GS st = sslm_decode_step_gpu(rig.ctx(), m.h, nullptr, rig.env.fx.one_layer_budget);
			CHECK_MSG(st == GS::SSLM_OK, "[G12.3] mid: one layer: %s", Name(st));
			const SslmGpuJumpOptions none = Opts(0);
			int32_t out[8], n = -9;
			const GS busy = sslm_gpu_seq_jump_forward(rig.ctx(), m.h, t, 8, &none, out, 8, &n);
			CHECK_MSG(busy == GS::SSLM_BUSY, "[G12.3] Submitted: %s", Name(busy));
			CHECK_MSG(Drain(rig.env, m.h) == GS::SSLM_OK && LayerIdx(m.h) > 0 && LayerIdx(m.h) < rig.env.fx.layers,
			          "[G12.3] mid: layer %u", LayerIdx(m.h));
			Refused("mid-token", rig, m, t, GS::SSLM_SEQUENCE_REJECTED);
			Refused("mid-token, -1", rig, m, -1, GS::SSLM_SEQUENCE_REJECTED);
		}
	}
}

// ---- G12.8 -----------------------------------------------------------------------------------------------------
void Cell_12_8(GRig& rig, const std::vector<int32_t>& prompt) {
	for (const bool pooled : {false, true}) {
		const char* arm = pooled ? "SLM6" : "SLM5";
		GSeq a = NewSeq(rig, kPotion, pooled);
		if (!a.h) continue;
		const int32_t t = TokenInHand(rig, a, prompt);
		uint64_t n0 = Submissions(rig.ctx());
		CHECK_MSG(ReadyFlag(rig.ctx(), a.h) == 0, "[G12.8 %s] token in hand: %d", arm, ReadyFlag(rig.ctx(), a.h));
		Jump(rig, a, t, 64);
		n0 = Submissions(rig.ctx());
		CHECK_MSG(ReadyFlag(rig.ctx(), a.h) == 1, "[G12.8 %s] after an admitting call: %d", arm, ReadyFlag(rig.ctx(), a.h));
		CHECK_MSG(Submissions(rig.ctx()) == n0, "[G12.8 %s] the getter submitted", arm);
		const std::vector<uint8_t> blob = rig.Blob(a.h);
		SslmGpuSequenceHandle* r = nullptr;
		const GS rs = pooled ? sslm_gpu_seq_restore_into_pool(rig.ctx(), rig.model(), g_pool, blob.data(), blob.size(),
		                                                      nullptr, &r, nullptr)
		                     : sslm_gpu_seq_restore(rig.ctx(), rig.model(), blob.data(), blob.size(), &r);
		if (r) rig.seqs.push_back(r);
		CHECK_MSG(rs == GS::SSLM_OK && r && ReadyFlag(rig.ctx(), r) == 1, "[G12.8 %s] after the restore: %s, %d", arm, Name(rs),
		          r ? ReadyFlag(rig.ctx(), r) : -1);
	}
	GSeq p = rig.New(kPotion);
	if (p.h) {
		Primed(rig, p, prompt);
		CHECK_MSG(ReadyFlag(rig.ctx(), p.h) == 0, "[G12.8] primed: %d", ReadyFlag(rig.ctx(), p.h));
		int32_t out = -9;
		CHECK_MSG(sslm_gpu_seq_ready_for_logits(rig.ctx(), p.h, nullptr) == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH &&
		              sslm_gpu_seq_ready_for_logits(rig.ctx(), nullptr, &out) == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH &&
		              sslm_gpu_seq_ready_for_logits(nullptr, p.h, &out) == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && out == -9,
		          "[G12.8] the malformed arms");
		CHECK_MSG(sslm_gpu_seq_embed_token(rig.ctx(), p.h, prompt[0]) == GS::SSLM_OK &&
		              sslm_decode_step_gpu(rig.ctx(), p.h, nullptr, rig.env.fx.one_layer_budget) == GS::SSLM_OK,
		          "[G12.8] submit one layer");
		CHECK_MSG(sslm_gpu_seq_ready_for_logits(rig.ctx(), p.h, &out) == GS::SSLM_BUSY && out == -9, "[G12.8] Submitted: not BUSY");
		Drain(rig.env, p.h);
	}
}

// ---- G12.9 -----------------------------------------------------------------------------------------------------
void Cell_12_9(GRig& rig, const std::vector<int32_t>& prompt) {
	for (int shape = 0; shape < 3; ++shape) {
		const char* name = shape == 0 ? "token in hand" : shape == 1 ? "ready" : "fresh";
		GSeq s = rig.New(kPotion), twin = rig.New(kPotion);
		if (!s.h || !twin.h) continue;
		int32_t pending = -1;
		if (shape == 0) {
			pending = TokenInHand(rig, s, prompt);
			TokenInHand(rig, twin, prompt);
		} else if (shape == 1) {
			rig.Prompt(s, prompt);
			rig.Prompt(twin, prompt);
		}
		const int32_t draft[1] = {prompt[0]};
		for (int with_draft = 0; with_draft < 2; ++with_draft) {
			const SslmGpuJumpOptions ex = with_draft ? Opts(kExact, draft, 1) : Opts(kExact);
			const SeqObs before = Observe(rig, s.h);
			const uint64_t n0 = Submissions(rig.ctx());
			const JumpOut j = Jump(rig, s, pending, 8, &ex);
			CHECK_MSG(j.st == GS::SSLM_SEQUENCE_REJECTED && Observe(rig, s.h) == before && Submissions(rig.ctx()) == n0,
			          "[G12.9] %s, exact %s: %s, sequence %s, %llu submissions", name, with_draft ? "with a draft" : "NULL",
			          Name(j.st), Observe(rig, s.h) == before ? "unchanged" : "changed",
			          static_cast<unsigned long long>(Submissions(rig.ctx()) - n0));
		}
		if (shape == 2) CHECK_MSG(rig.Bindable(s.h, kPotion), "[G12.9] fresh: not bindable after the refusals");
		const JumpOut a = Jump(rig, s, pending, 64), b = Jump(rig, twin, pending, 64);
		CHECK_MSG(a.st == b.st && a.forced == b.forced && Observe(rig, s.h) == Observe(rig, twin.h),
		          "[G12.9] %s: the canonical call after the refusals differs from a fresh one's", name);
	}
}

}  // namespace

int main(int argc, char** argv) {
	GRig rig;
	if (rig.Open(argc, argv, /*load_ref=*/false)) {
		CRig c(rig.raw);
		const std::vector<int32_t> prompt = c.Tokenize(kPromptPotionText);
		const GS ps = sslm_gpu_kv_pool_create(rig.ctx(), rig.model(), 96, &g_pool);
		CHECK_MSG(ps == GS::SSLM_OK && g_pool, "pool: %s", Name(ps));
		if (Only(argc, argv, "G12.1") && g_pool) Cell_12_1(rig, prompt);
		if (Only(argc, argv, "G12.2")) Cell_12_2(rig, prompt);
		if (Only(argc, argv, "G12.3")) Cell_12_3(rig, prompt);
		if (Only(argc, argv, "G12.8") && g_pool) Cell_12_8(rig, prompt);
		if (Only(argc, argv, "G12.9")) Cell_12_9(rig, prompt);
		rig.ReleaseSeqs();
		if (g_pool) sslm_gpu_kv_pool_destroy(rig.ctx(), g_pool);
	}
	return Finish("sfdg_cell_g12_verb");
}
