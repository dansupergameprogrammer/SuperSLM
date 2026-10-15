// Paged-KV plan (rev 16.2), G1, group g1c: the two run shapes cells 3.6b, 3.6c and 3.6d are made of,
// and the builders of each guarded verb's construction (shared so the three cells cannot drift).
//
// GUARDED RUN (3.6b, 3.6c, 3.6d's main runs). One guarded entry point, called while a sequence A is
// Submitted (made Submitted by `sslm_decode_step_gpu`, the only way §7 makes "A is Submitted", §4.4
// "No §7 construction makes a sequence Submitted that way"):
//   TWIN    the same construction on fresh handles, A Submitted and drained, and THEN the call --
//           which succeeds. Records the whole mirror after the drain, the schema operand, the call's
//           status and its output.
//   MAIN    the construction; [G5: B's blob saved]; A Submitted; the call -> SSLM_BUSY, and every
//           other channel of the refused call (a null out-handle, SSLM_BUSY in every out_statuses
//           slot, *consumed 0); sslm_gpu_ready(A); then the operands, each against the twin:
//             - the whole mirror of every pool, byte-equal (the GPU mirror peek);
//             - the schema operand (SetSchema on B, or set_schema on a fresh prefix): equal to the
//               twin's, and to the value the plan states where it states one;
//             - [G5] B's re-saved blob, equal to the one saved before the refused call;
//             - the GPU fill probe's admission leg on the probed pool: exactly k free, k written here
//               as pool pages minus the §3.4 reservations the construction made;
//           then the call again: the twin's status, and its output byte-equal to the twin's.
//   REBUILD the construction again, A Submitted, the call refused, drained, and the fill probe's
//           refusal leg (k + 1 not all admitted).
// Under the cells' named mutants: no guard -> the refused call returns SSLM_OK; a guard at the reset
// only -> the pre-reset writes show in the mirror, the schema operand, the fill probe or the re-save,
// per verb as §7 states; a guard scoped to the pool (3.6c) -> SSLM_OK; rev 5's placement (3.6d) ->
// SSLM_DEVICE_LOST or SSLM_OK with the state changed.
//
// PRECEDENCE RUN (3.6d's precedence control, §4.4 *Status precedence*). One post-guard exit (or one
// pre-guard refusal) called twice on fresh rebuilds: with a pooled sequence A Submitted (SSLM_BUSY for
// a post-guard exit; the refusal's own status for a pre-guard one), and with nothing Submitted
// (today's status). The named mutant, the guard moved below the exit, returns today's status while A
// is Submitted.
#ifndef SSLM_G1C_GUARD_RUNS_H
#define SSLM_G1C_GUARD_RUNS_H

#include "g1c_helpers.h"

namespace g1c {

enum class AKind {
	PooledSamePool,   // A on pool 0, beside B (3.6b, 3.6d)
	PooledOtherPool,  // A on its own pool 1 of the same context (3.6c (i))
	Dedicated,        // A a dedicated sequence (3.6c G4/G5, 3.6d's scope control)
};

struct GuardedSpec {
	std::string name;
	AKind a = AKind::PooledSamePool;
	bool b_device_logits = false;  // pool 0's model
	uint32_t pool_pages = 0;       // 0: --pool-pages
	std::function<bool(World&)> build;                         // after pool 0 exists, before A
	std::function<SslmGpuStatus(World&)> call;                 // the guarded call
	std::function<void(World&)> check_refused;                 // the refused call's other channels
	std::function<std::vector<int64_t>(World&, SslmGpuStatus)> output;  // after the call succeeded
	std::function<SslmGpuStatus(World&)> schema_operand;       // optional
	bool has_expect_schema = false;
	SslmGpuStatus expect_schema = SSLM_OK;
	int probe_pool = 0;  // -1: no fill probe
	bool mirror = true;
	std::function<SslmGpuSequenceHandle*(World&)> resave_holder;  // G5 parts only
};

inline const char* AName(AKind a) {
	return a == AKind::PooledSamePool ? "pooled A, same pool"
	       : a == AKind::PooledOtherPool ? "pooled A, another pool"
	                                     : "dedicated A";
}

// Pool 0, the construction, then A (made after B, so the draws are in one order in every rebuild).
inline bool Prepare(World& w, const GuardedSpec& g) {
	w.Pool(g.pool_pages ? g.pool_pages : Opt().pool_pages, g.b_device_logits);
	if (!w.ok) return false;
	if (g.build && !g.build(w)) return false;
	switch (g.a) {
		case AKind::PooledSamePool: w.A = w.Pooled(0, 32); break;
		case AKind::PooledOtherPool: w.A = w.Pooled(w.Pool(16), 32); break;
		case AKind::Dedicated: w.A = w.Dedicated(); break;
	}
	return w.ok && w.A;
}
inline int32_t AToken(const Stack& s) { return Toks(s.art, 1, 11)[0]; }

inline void GuardedRun(const GuardedSpec& g) {
	Stack* s = GetStack();
	if (!s) return;
	const char* nm = g.name.c_str();
	std::printf("  run %s (%s)\n", nm, AName(g.a));

	// TWIN: the call made only after the drain.
	std::vector<uint8_t> twin_mirror;
	SslmGpuStatus twin_schema = SSLM_DEVICE_LOST, twin_st = SSLM_DEVICE_LOST;
	std::vector<int64_t> twin_out;
	{
		World w(*s);
		if (!Prepare(w, g) || !w.Submit(w.A, AToken(*s))) {
			G1C_CHECK(false, "%s: SETUP: twin construction failed", nm);
			return;
		}
		(void)w.Drain(w.A);
		if (g.mirror) twin_mirror = w.Mirrors();
		if (g.schema_operand) twin_schema = g.schema_operand(w);
		twin_st = g.call(w);
		twin_out = g.output ? g.output(w, twin_st) : std::vector<int64_t>{};
		w.TakeOuts();
		G1C_CHECK(twin_st == SSLM_OK, "%s: SETUP: the twin's call, with nothing Submitted, returned %s", nm, St(twin_st));
	}

	// MAIN: the call refused while A is Submitted.
	{
		World w(*s);
		if (!Prepare(w, g)) {
			G1C_CHECK(false, "%s: SETUP: construction failed", nm);
			return;
		}
		SslmGpuSequenceHandle* holder = g.resave_holder ? g.resave_holder(w) : nullptr;
		std::vector<uint8_t> blob0;
		if (holder) {
			SslmGpuStatus sv = SSLM_DEVICE_LOST;
			blob0 = Save(s->ctx, holder, &sv);
			G1C_CHECK(sv == SSLM_OK && blob0.size() >= 4 && std::memcmp(blob0.data(), "SLM6", 4) == 0,
			          "%s: SETUP: B's blob before the call: %s, %zu bytes, magic %.4s", nm, St(sv), blob0.size(),
			          blob0.size() >= 4 ? reinterpret_cast<const char*>(blob0.data()) : "----");
		}
		if (!w.Submit(w.A, AToken(*s))) return;
		const SslmGpuStatus st = g.call(w);
		G1C_CHECK(st == SSLM_BUSY, "%s: the call while A is Submitted returned %s, want SSLM_BUSY", nm, St(st));
		if (g.check_refused) g.check_refused(w);
		w.TakeOuts();
		const SslmGpuStatus dr = w.Drain(w.A);
		G1C_CHECK(dr == SSLM_OK, "%s: SETUP: sslm_gpu_ready(A) -> %s", nm, St(dr));
		if (g.mirror) {
			const std::vector<uint8_t> m = w.Mirrors();
			const size_t d = FirstDiff(m, twin_mirror);
			G1C_CHECK(d == SIZE_MAX, "%s: the pools' mirrors after the refused call differ from the twin's at byte "
			                         "%zu of %zu (page %zu)", nm, d, m.size(), d == SIZE_MAX ? 0 : d / s->art.PageBytes());
		}
		if (g.schema_operand) {
			const SslmGpuStatus sc = g.schema_operand(w);
			G1C_CHECK(sc == twin_schema, "%s: the schema operand after the drain is %s, the twin's %s", nm, St(sc),
			          St(twin_schema));
			if (g.has_expect_schema) {
				G1C_CHECK(sc == g.expect_schema, "%s: the schema operand after the drain is %s, want %s", nm, St(sc),
				          St(g.expect_schema));
			}
		}
		if (holder) {
			const std::vector<uint8_t> blob1 = Save(s->ctx, holder);
			const size_t d = FirstDiff(blob1, blob0);
			G1C_CHECK(d == SIZE_MAX, "%s: B's re-saved SLM6 blob differs from the one saved before the call at byte "
			                         "%zu (%zu vs %zu bytes)", nm, d, blob1.size(), blob0.size());
		}
		if (g.probe_pool >= 0) (void)AdmissionLeg(w, g.probe_pool, w.FreePages(g.probe_pool), nm);
		const SslmGpuStatus st2 = g.call(w);
		G1C_CHECK(st2 == twin_st, "%s: after the drain the call returned %s, the twin's %s", nm, St(st2), St(twin_st));
		const std::vector<int64_t> out = g.output ? g.output(w, st2) : std::vector<int64_t>{};
		w.TakeOuts();
		G1C_CHECK(out == twin_out, "%s: the output after proceeding differs from the twin's (%zu vs %zu values)", nm,
		          out.size(), twin_out.size());
	}

	// REBUILD: the fill probe's refusal leg on the same state.
	if (g.probe_pool >= 0) {
		World w(*s);
		if (!Prepare(w, g) || !w.Submit(w.A, AToken(*s))) return;
		const SslmGpuStatus st = g.call(w);
		G1C_CHECK(st == SSLM_BUSY, "%s: rebuild: the call while A is Submitted returned %s", nm, St(st));
		w.TakeOuts();
		(void)w.Drain(w.A);
		(void)RefusalLeg(w, g.probe_pool, w.FreePages(g.probe_pool), nm);
	}
}

struct PrecedenceSpec {
	std::string name;
	bool b_device_logits = false;
	std::function<bool(World&)> build;        // after pool 0 exists
	std::function<SslmGpuStatus(World&)> call;  // returns the graded status (the call's, or a slot's)
	SslmGpuStatus expect_submitted = SSLM_BUSY;
	std::vector<SslmGpuStatus> expect_idle;   // today's status; more than one only where the plan is open
	std::function<void(World&)> check_submitted;  // other channels of the leg with A Submitted
	std::function<void(World&)> follow_on;        // after the drain, on the leg with A Submitted
};

inline void PrecedenceRun(const PrecedenceSpec& p) {
	Stack* s = GetStack();
	if (!s) return;
	const char* nm = p.name.c_str();
	std::printf("  precedence %s\n", nm);
	GuardedSpec g;
	g.b_device_logits = p.b_device_logits;
	g.build = p.build;
	{
		World w(*s);
		if (!Prepare(w, g) || !w.Submit(w.A, AToken(*s))) {
			G1C_CHECK(false, "%s: SETUP: construction failed", nm);
			return;
		}
		const SslmGpuStatus st = p.call(w);
		G1C_CHECK(st == p.expect_submitted, "%s: with a pooled A Submitted: %s, want %s", nm, St(st),
		          St(p.expect_submitted));
		if (p.check_submitted) p.check_submitted(w);
		w.TakeOuts();
		(void)w.Drain(w.A);
		if (p.follow_on) p.follow_on(w);
	}
	{
		World w(*s);
		if (!Prepare(w, g)) {
			G1C_CHECK(false, "%s: SETUP: construction failed", nm);
			return;
		}
		const SslmGpuStatus st = p.call(w);
		bool any = false;
		std::string want;
		for (SslmGpuStatus e : p.expect_idle) {
			any = any || st == e;
			want += std::string(want.empty() ? "" : " or ") + St(e);
		}
		G1C_CHECK(any, "%s: with nothing Submitted: %s, want %s", nm, St(st), want.c_str());
		w.TakeOuts();
	}
}

// =============================================================================================
// Builders: B's resting states, each reachable at source.
// =============================================================================================
// B fresh on pool 0 (bind-eligible, not ready for logits), bound to the schema when `bind`.
inline bool BuildFreshB(World& w, bool bind = false, int32_t budget = 64) {
	w.B = w.Pooled(0, budget);
	if (!w.B) return false;
	w.binding = -1;
	if (bind) {
		const int32_t idx = w.Schema(w.pools[0].model == w.s.dev);
		if (idx < 0) {
			G1C_SKIP("--model has no schema named %s: the schema-bound run needs one", Opt().schema.c_str());
			return false;
		}
		const SslmGpuStatus st = SslmGpuSeqSetSchemaForG5Bridge(w.s.ctx, w.B, idx);
		G1C_CHECK(st == SSLM_OK, "SETUP: binding B to the schema -> %s", St(st));
		w.binding = idx;
	}
	return true;
}
// B after a prompt prefill of `n` tokens: at full depth, ready for logits, not bind-eligible.
inline bool BuildReadyB(World& w, size_t n = 16, int32_t budget = 64) {
	if (!BuildFreshB(w, false, budget)) return false;
	const std::vector<int32_t> t = Toks(w.s.art, n, 5);
	const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(w.s.ctx, w.B, t.data(), static_cast<int32_t>(t.size()),
	                                                            w.s.one_layer);
	G1C_CHECK(st == SSLM_OK, "SETUP: prompt prefill of B (%zu) -> %s", n, St(st));
	return st == SSLM_OK;
}
// `q` resting at a page boundary (context_length 16 = B for B = 16) with its next token embedded:
// prefill 16, finish the ready token through the bridge decode, embed that token. Its embed revoked
// bind_eligible, so its only pre-reset writes under a reset-only guard are ones no operand reads
// (§7 3.6b's G3 part: the reset-only mutant is equivalent for decode and batch decode).
inline bool RestAtBoundary(World& w, SslmGpuSequenceHandle* q) {
	const int64_t n = w.s.art.B() <= 64 ? w.s.art.B() : 16;
	const std::vector<int32_t> t = Toks(w.s.art, static_cast<size_t>(n), 7);
	SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(w.s.ctx, q, t.data(), static_cast<int32_t>(t.size()),
	                                                      w.s.one_layer);
	int32_t tok = -1;
	if (st == SSLM_OK) st = SslmGpuSeqDecodeStepForG5Bridge(w.s.ctx, q, t[0], w.s.full_depth, &tok);
	if (st == SSLM_OK) st = sslm_gpu_seq_embed_token(w.s.ctx, q, tok >= 0 ? tok : t[0]);
	G1C_CHECK(st == SSLM_OK, "SETUP: resting a holder at the page boundary -> %s", St(st));
	return st == SSLM_OK;
}
// The output of a holder whose guarded call drove a decode: drain, finish, then 3 more tokens.
inline std::vector<int64_t> FinishAndRun(World& w, SslmGpuSequenceHandle* q) {
	std::vector<int64_t> out;
	out.push_back(static_cast<int64_t>(Ready(w.s.ctx, q)));
	int32_t tok = -1;
	const SslmGpuStatus f = SslmGpuSeqFinishTokenForG5Bridge(w.s.ctx, q, &tok);
	out.push_back(static_cast<int64_t>(f));
	out.push_back(tok);
	const std::vector<int64_t> more = DecodeRun(w.s, q, 3, tok >= 0 ? tok : 1);
	out.insert(out.end(), more.begin(), more.end());
	return out;
}
// A pooled X adopting the frozen prefix `p` (after freezing it when asked), then 4 tokens: the output
// of the verbs that make or fill a prefix.
inline std::vector<int64_t> AdoptAndRun(World& w, SslmGpuPrefixHandle* p, bool freeze_first) {
	std::vector<int64_t> out;
	if (!p) return {-1};
	if (freeze_first) out.push_back(static_cast<int64_t>(sslm_gpu_prefix_freeze(w.s.ctx, p)));
	SslmGpuSequenceHandle* x = nullptr;
	const SslmGpuStatus c = sslm_gpu_seq_create_budgeted(w.s.ctx, w.pools[0].model, w.pools[0].h, 64, &x);
	out.push_back(static_cast<int64_t>(c));
	if (!x) return out;
	w.seqs.push_back(x);
	out.push_back(static_cast<int64_t>(sslm_gpu_seq_adopt_prefix(w.s.ctx, x, p)));
	const std::vector<int64_t> more = DecodeRun(w.s, x, 4, Toks(w.s.art, 1, 3)[0]);
	out.insert(out.end(), more.begin(), more.end());
	return out;
}
inline SslmGpuStatus SetSchemaB(World& w) { return SslmGpuSeqSetSchemaForG5Bridge(w.s.ctx, w.B, w.binding); }

// The prefix length of 3.6b's G4 part: mid-page (1,000 = 62 pages + 8 rows at B = 16), so adopt's
// tail copy shows in the mirror; at an aligned 1,008 no operand of G4 sees adopt's mutant.
constexpr int32_t kMidPagePrefix = 1000;

// =============================================================================================
// Spec factories, one per guarded verb.
// =============================================================================================
// sslm_decode_step_gpu on B resting at a page boundary.
inline GuardedSpec SpecDecode(AKind a) {
	GuardedSpec g;
	g.name = "sslm_decode_step_gpu";
	g.a = a;
	g.build = [](World& w) { return BuildFreshB(w) && RestAtBoundary(w, w.B); };
	g.call = [](World& w) { return sslm_decode_step_gpu(w.s.ctx, w.B, nullptr, w.s.full_depth); };
	g.output = [](World& w, SslmGpuStatus) { return FinishAndRun(w, w.B); };
	g.schema_operand = SetSchemaB;
	g.has_expect_schema = true;
	g.expect_schema = SSLM_SEQUENCE_REJECTED;  // the embed revoked bind_eligible in both builds
	g.resave_holder = [](World& w) { return w.B; };
	return g;
}
// sslm_decode_step_batch_gpu on [B, C], both resting at a page boundary.
inline GuardedSpec SpecBatch(AKind a) {
	GuardedSpec g;
	g.name = "sslm_decode_step_batch_gpu";
	g.a = a;
	g.build = [](World& w) {
		if (!BuildFreshB(w) || !RestAtBoundary(w, w.B)) return false;
		w.C = w.Pooled(0, 64);
		return w.C && RestAtBoundary(w, w.C);
	};
	g.call = [](World& w) {
		SslmGpuSequenceHandle* seqs[2] = {w.B, w.C};
		w.slots.assign(2, SSLM_DEVICE_LOST);
		return sslm_decode_step_batch_gpu(w.s.ctx, seqs, nullptr, 2, 2u * w.s.full_depth, w.slots.data());
	};
	g.check_refused = [](World& w) {
		for (size_t i = 0; i < w.slots.size(); ++i) {
			G1C_CHECK(w.slots[i] == SSLM_BUSY, "batch: out_statuses[%zu] is %s, want SSLM_BUSY", i, St(w.slots[i]));
		}
	};
	g.output = [](World& w, SslmGpuStatus) {
		std::vector<int64_t> out;
		for (SslmGpuStatus st : w.slots) out.push_back(static_cast<int64_t>(st));
		const std::vector<int64_t> b = FinishAndRun(w, w.B), c = FinishAndRun(w, w.C);
		out.insert(out.end(), b.begin(), b.end());
		out.insert(out.end(), c.begin(), c.end());
		return out;
	};
	g.schema_operand = SetSchemaB;
	g.has_expect_schema = true;
	g.expect_schema = SSLM_SEQUENCE_REJECTED;
	g.resave_holder = [](World& w) { return w.B; };
	return g;
}
// The prompt prefill bridge on a fresh B (bind-eligible): SetSchema stays SSLM_OK.
inline GuardedSpec SpecPromptPrefill(AKind a) {
	GuardedSpec g;
	g.name = "SslmGpuSeqPrefillPromptForG5Bridge";
	g.a = a;
	g.build = [](World& w) {
		w.toks = Toks(w.s.art, 20, 9);
		return BuildFreshB(w);
	};
	g.call = [](World& w) {
		return SslmGpuSeqPrefillPromptForG5Bridge(w.s.ctx, w.B, w.toks.data(), static_cast<int32_t>(w.toks.size()),
		                                          w.s.one_layer);
	};
	g.output = [](World& w, SslmGpuStatus) { return DecodeRun(w.s, w.B, 4, 1); };
	g.schema_operand = SetSchemaB;
	g.has_expect_schema = true;
	g.expect_schema = SSLM_OK;
	g.resave_holder = [](World& w) { return w.B; };
	return g;
}
// The schema-content prefill bridge on a fresh, schema-bound B. Its one forced token is legal at the
// schema's state 0: it is the token a fresh bound sequence emits first (masked argmax from state 0).
inline GuardedSpec SpecSchemaPrefill(AKind a) {
	GuardedSpec g;
	g.name = "SslmGpuSeqPrefillSchemaContentForG5Bridge";
	g.a = a;
	g.build = [](World& w) {
		if (!BuildFreshB(w, /*bind=*/true)) return false;
		SslmGpuSequenceHandle* z = w.Pooled(0, 16);
		if (!z) return false;
		SslmGpuStatus st = SslmGpuSeqSetSchemaForG5Bridge(w.s.ctx, z, w.binding);
		int32_t t0 = -1;
		if (st == SSLM_OK) st = SslmGpuSeqDecodeStepForG5Bridge(w.s.ctx, z, Toks(w.s.art, 1, 13)[0], w.s.full_depth, &t0);
		G1C_CHECK(st == SSLM_OK && t0 >= 0, "SETUP: the schema's first legal token: %s, token %d", St(st), t0);
		w.toks = {t0};
		return st == SSLM_OK && t0 >= 0;
	};
	g.call = [](World& w) {
		w.i32 = -1;
		return SslmGpuSeqPrefillSchemaContentForG5Bridge(w.s.ctx, w.B, w.toks.data(), static_cast<int32_t>(w.toks.size()),
		                                                 w.s.one_layer, &w.i32);
	};
	g.check_refused = [](World& w) { G1C_CHECK(w.i32 == 0, "schema prefill: *consumed %d on the refusal", w.i32); };
	g.output = [](World& w, SslmGpuStatus) {
		std::vector<int64_t> out{w.i32};
		const std::vector<int64_t> more = DecodeRun(w.s, w.B, 3, 1);
		out.insert(out.end(), more.begin(), more.end());
		return out;
	};
	g.schema_operand = SetSchemaB;
	g.has_expect_schema = true;
	g.expect_schema = SSLM_OK;
	g.resave_holder = [](World& w) { return w.B; };
	return g;
}
// GPU adopt of a frozen prefix at `len` (1,000: mid-page) onto a fresh pooled B.
inline GuardedSpec SpecAdopt(AKind a, int32_t len = kMidPagePrefix) {
	GuardedSpec g;
	g.name = "sslm_gpu_seq_adopt_prefix (prefix " + std::to_string(len) + ")";
	g.a = a;
	g.build = [len](World& w) {
		w.P = w.Prefix(0, len, Toks(w.s.art, static_cast<size_t>(len), 17), /*freeze=*/true);
		return w.P && BuildFreshB(w, false, 96);
	};
	g.call = [](World& w) { return sslm_gpu_seq_adopt_prefix(w.s.ctx, w.B, w.P); };
	g.output = [](World& w, SslmGpuStatus) { return DecodeRun(w.s, w.B, 4, 1); };
	g.schema_operand = SetSchemaB;  // fresh B: SSLM_OK in both builds; GPU adopt checks before any change
	g.has_expect_schema = true;
	g.expect_schema = SSLM_OK;
	g.resave_holder = [](World& w) { return w.B; };
	return g;
}
// begin_from of a frozen parent at 1,000: the would-be child is B. A null out-handle on the refusal.
inline GuardedSpec SpecBeginFrom(AKind a) {
	GuardedSpec g;
	g.name = "sslm_gpu_prefix_begin_from";
	g.a = a;
	g.build = [](World& w) {
		w.P = w.Prefix(0, kMidPagePrefix, Toks(w.s.art, kMidPagePrefix, 19), /*freeze=*/true);
		return w.P != nullptr;
	};
	g.call = [](World& w) {
		w.out_prefix = nullptr;
		return sslm_gpu_prefix_begin_from(w.s.ctx, w.P, 64, &w.out_prefix);
	};
	g.check_refused = [](World& w) {
		G1C_CHECK(w.out_prefix == nullptr, "begin_from: a refused call left a non-null out-handle");
	};
	g.output = [](World& w, SslmGpuStatus) {
		SslmGpuPrefixHandle* child = w.out_prefix;
		w.TakeOuts();
		std::vector<int64_t> out{child != nullptr};
		const std::vector<int64_t> more = AdoptAndRun(w, child, /*freeze_first=*/true);
		out.insert(out.end(), more.begin(), more.end());
		return out;
	};
	return g;
}
// The prefix prefill on a fresh prefix Q: set_schema on Q stays SSLM_OK after the drain.
inline GuardedSpec SpecPrefixPrefill(AKind a) {
	GuardedSpec g;
	g.name = "sslm_gpu_prefix_prefill";
	g.a = a;
	g.build = [](World& w) {
		w.Q = w.Prefix(0, 64, {}, /*freeze=*/false);
		w.toks = Toks(w.s.art, 20, 23);
		return w.Q != nullptr;
	};
	g.call = [](World& w) {
		w.i32 = -1;
		// One call that covers the whole span: per-token cost is full depth (dispatches per layer x layers).
		return sslm_gpu_prefix_prefill(w.s.ctx, w.Q, w.toks.data(), static_cast<int32_t>(w.toks.size()), 0,
		                               w.s.full_depth * static_cast<uint32_t>(w.toks.size()), &w.i32);
	};
	g.check_refused = [](World& w) { G1C_CHECK(w.i32 == 0, "prefix prefill: *consumed %d on the refusal", w.i32); };
	g.output = [](World& w, SslmGpuStatus) {
		std::vector<int64_t> out{w.i32};
		const std::vector<int64_t> more = AdoptAndRun(w, w.Q, /*freeze_first=*/true);
		out.insert(out.end(), more.begin(), more.end());
		return out;
	};
	// §4.7: set_schema binds through the prefix's internal sequence; on a fresh prefix it is SSLM_OK,
	// and SSLM_GPU_PREFIX_REJECTED once a refused prefill revoked the bind permission (the mutant).
	g.schema_operand = [](World& w) {
		return sslm_gpu_prefix_set_schema(w.s.ctx, w.Q, w.Schema(w.pools[0].model == w.s.dev));
	};
	g.has_expect_schema = true;
	g.expect_schema = SSLM_OK;
	return g;
}
// Restore-into-pool of a pooled sequence's SLM6 blob: the holder the verb would make is B.
inline GuardedSpec SpecRestoreIntoPool(AKind a) {
	GuardedSpec g;
	g.name = "sslm_gpu_seq_restore_into_pool";
	g.a = a;
	g.build = [](World& w) {
		SslmGpuSequenceHandle* src = w.Pooled(0, 64);
		if (!src) return false;
		const std::vector<int32_t> t = Toks(w.s.art, 20, 29);
		const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(w.s.ctx, src, t.data(), 20, w.s.one_layer);
		w.blob = Save(w.s.ctx, src);
		G1C_CHECK(st == SSLM_OK && w.blob.size() >= 4 && std::memcmp(w.blob.data(), "SLM6", 4) == 0,
		          "SETUP: the pooled source's SLM6 blob: prefill %s, %zu bytes", St(st), w.blob.size());
		return !w.blob.empty();
	};
	g.call = [](World& w) {
		w.u32 = 0xFFFFFFFFu;
		w.out_seq = nullptr;
		return sslm_gpu_seq_restore_into_pool(w.s.ctx, w.pools[0].model, w.pools[0].h, w.blob.data(), w.blob.size(),
		                                      nullptr, &w.out_seq, &w.u32);
	};
	g.check_refused = [](World& w) {
		G1C_CHECK(w.out_seq == nullptr, "restore-into-pool: a refused call left a non-null out-handle");
	};
	g.output = [](World& w, SslmGpuStatus) {
		SslmGpuSequenceHandle* r = w.out_seq;
		w.TakeOuts();
		std::vector<int64_t> out{r != nullptr};
		if (!r) return out;
		out.push_back(Save(w.s.ctx, r) == w.blob);
		const std::vector<int64_t> more = DecodeRun(w.s, r, 4, 1);
		out.insert(out.end(), more.begin(), more.end());
		return out;
	};
	return g;
}

}  // namespace g1c

#endif  // SSLM_G1C_GUARD_RUNS_H
