// Paged-KV plan (rev 16.2) §7 cell 3.6d [G3 existing entry points and pool create; G4 adopt,
// begin_from, prefix prefill; G5 restore-into-pool and the re-save], group g1c -- the default GPU
// build's runs. The bench prefill's runs are g1c_cell_3_6d_bench.cpp (the bench configuration build),
// and the first-touch control is g1c_cell_3_6d_first_touch.cpp (the allocation-fault build).
//
// "The guarded set of §4.4, one run per 'guarded' row. A pooled sequence A is made Submitted by
//  sslm_decode_step_gpu, and each guarded entry point is called, on a pooled holder B where it takes
//  one. Each returns SSLM_BUSY and changes nothing, and after sslm_gpu_ready(A) it proceeds. The named
//  mutant is rev 5's placement: a guard only on rev 5's verb list, with PoolSubmitAssert at every
//  reset site."
//
// MAIN RUNS (g1c_guard_runs.h's GUARDED RUN, A pooled on B's pool). The operands are the status, the
// out-handle, B's whole mirror (the GPU mirror peek, against the twin), SetSchema, and the fill probe:
//   [G3] SslmGpuSeqDecodeStepForG5Bridge on both branches: not ready for logits (embed, then drive;
//        SetSchema SSLM_OK, B bind-eligible) and ready (finish; SetSchema SSLM_SEQUENCE_REJECTED: every
//        path that sets ready_for_logits clears bind_eligible first). Under rev 5 the call returns
//        SSLM_DEVICE_LOST with the binding revoked; on the ready branch SSLM_DEVICE_LOST alone separates.
//        SslmGpuSeqFinishTokenForG5Bridge on a DEVICE-LOGITS model with B AT FULL DEPTH (a fresh B would
//        return SSLM_SEQUENCE_REJECTED first). sslm_decode_step_gpu. sslm_decode_step_batch_gpu
//        (SSLM_BUSY as the return and in every out_statuses[i]). The two prefill bridges on a fresh B,
//        SetSchema still SSLM_OK. sslm_gpu_seq_restore, sslm_gpu_seq_create, sslm_gpu_model_map,
//        sslm_gpu_adapter_map and sslm_gpu_kv_pool_create, each SSLM_BUSY and a null out-handle.
//   [G4] adopt, begin_from and prefix prefill, as the G3 part, with sslm_gpu_prefix_set_schema still
//        SSLM_OK on a fresh prefix after the refused prefix prefill.
//   [G5] restore-into-pool; and every run whose B is a sequence adds B's re-saved SLM6 blob, equal to
//        the one saved before.
// CONTROLS:
//   - precedence (§4.4 *Status precedence*): each post-guard exit returns SSLM_BUSY with a pooled A
//     Submitted and today's status with nothing Submitted; the guard-below mutant returns today's
//     status while A is Submitted, except on the bridge decode's ready branch, where its call reaches
//     the inner finish's guard -- there the separating operand is the FOLLOW-ON call (sslm_gpu_ready(A),
//     then the bridge decode on B with token -1: the correct B is still ready and returns SSLM_OK with
//     a token; the mutant's B lost ready_for_logits and its embed refuses -1). The dispatch budget below
//     one layer precedes the guard: SSLM_DISPATCH_BUDGET_TOO_SMALL either way. [G4] the prefix prefill
//     with count 0 (a must-accept: the guard-below mutant is equivalent there). [G5] restore-into-pool
//     with a bad magic.
//   - the mixed batch: a dedicated D Submitted and waited through the wait seam; [D2, P], [P, D2],
//     [P, null] and [another context's member, P] each SSLM_BUSY in the return and every slot. The
//     named mutant takes the scope from the first member only (SSLM_OK on [D2, P]).
//   - scope: D Submitted and waited -> a decode on a second dedicated sequence, seq_create and model
//     map SSLM_OK with the debug layer clean; pool create and any call on a pooled holder SSLM_BUSY.
//     A pooled sequence Submitted -> seq_create SSLM_BUSY. D Submitted and NOT waited -> the decode on
//     the second dedicated sequence SSLM_BUSY (the ledger's term). Must-reject: a guard scoped to any
//     sequence for dedicated callers (SSLM_BUSY where SSLM_OK is due).
//   - the assert's window: with nothing else Submitted, a pooled prefill returns SSLM_OK with
//     PoolSubmitAssert in place (the racing case is the model's only; 3.6e fires the assert).
//   The census is advisory and no part passes on it (rev 11).
//
// RED NOW (63e7b0e): BY LINK -- the pool verbs, sslm_gpu_seq_create_budgeted, the mirror peek and the
// wait seam sslm_gpu_test_only_wait_in_flight (G3), the prefix verbs and adopt (G4),
// restore-into-pool (G5). At run before G3 the guard does not exist: every existing entry point
// returns SSLM_OK (or its own refusal) on a pooled holder while A is Submitted.
// GREEN: each part at its step, on both GPUs.
#include "g1c_guard_runs.h"
// pkvg unification: the parts after PKVG_STEP (g1c_helpers.h) are compiled out.

#include <memory>

namespace {
using namespace g1c;

// ---------------------------------------------------------------------------------------------
// The adapter artifacts (argv), loaded once.
// ---------------------------------------------------------------------------------------------
const Artifact* AdapterOf(const std::string& path, const char* flag) {
	static std::vector<std::unique_ptr<Artifact>> cache;
	for (auto& a : cache) {
		if (a->path == path) return a->ok ? a.get() : nullptr;
	}
	if (path.empty()) {
		G1C_SKIP("%s not supplied", flag);
		return nullptr;
	}
	cache.push_back(std::make_unique<Artifact>());
	if (!cache.back()->Load(path)) {
		G1C_CHECK(false, "cannot load %s=%s", flag, path.c_str());
		return nullptr;
	}
	return cache.back().get();
}

// ---------------------------------------------------------------------------------------------
// 3.6d's own specs (the rest are shared with 3.6b/3.6c in g1c_guard_runs.h).
// ---------------------------------------------------------------------------------------------
std::vector<int64_t> TokenThenRun(World& w, SslmGpuSequenceHandle* q) {
	std::vector<int64_t> out{w.i32};
	const std::vector<int64_t> more = DecodeRun(w.s, q, 3, w.i32 >= 0 ? w.i32 : 1);
	out.insert(out.end(), more.begin(), more.end());
	return out;
}

GuardedSpec SpecBridgeNotReady() {
	GuardedSpec g;
	g.name = "SslmGpuSeqDecodeStepForG5Bridge (not ready: embed, drive)";
	g.build = [](World& w) { return BuildFreshB(w); };
	g.call = [](World& w) {
		w.i32 = -1;
		return SslmGpuSeqDecodeStepForG5Bridge(w.s.ctx, w.B, Toks(w.s.art, 1, 31)[0], w.s.full_depth, &w.i32);
	};
	g.output = [](World& w, SslmGpuStatus) { return TokenThenRun(w, w.B); };
	g.schema_operand = SetSchemaB;
	g.has_expect_schema = true;
	g.expect_schema = SSLM_OK;
	g.resave_holder = [](World& w) { return w.B; };
	return g;
}
GuardedSpec SpecBridgeReady() {
	GuardedSpec g;
	g.name = "SslmGpuSeqDecodeStepForG5Bridge (ready: finish)";
	g.b_device_logits = true;
	g.build = [](World& w) { return BuildReadyB(w); };
	g.call = [](World& w) {
		w.i32 = -1;
		return SslmGpuSeqDecodeStepForG5Bridge(w.s.ctx, w.B, Toks(w.s.art, 1, 31)[0], w.s.full_depth, &w.i32);
	};
	g.output = [](World& w, SslmGpuStatus) { return TokenThenRun(w, w.B); };
	g.schema_operand = SetSchemaB;
	g.has_expect_schema = true;
	g.expect_schema = SSLM_SEQUENCE_REJECTED;
	g.resave_holder = [](World& w) { return w.B; };
	return g;
}
GuardedSpec SpecFinishDeviceLogits() {
	GuardedSpec g;
	g.name = "SslmGpuSeqFinishTokenForG5Bridge (device-logits model, B at full depth)";
	g.b_device_logits = true;
	g.build = [](World& w) { return BuildReadyB(w); };
	g.call = [](World& w) {
		w.i32 = -1;
		return SslmGpuSeqFinishTokenForG5Bridge(w.s.ctx, w.B, &w.i32);
	};
	g.output = [](World& w, SslmGpuStatus) { return TokenThenRun(w, w.B); };
	g.schema_operand = SetSchemaB;  // "the same assertions": unchanged against the twin
	g.resave_holder = [](World& w) { return w.B; };
	return g;
}
// The make-type rows: SSLM_BUSY and a null out-handle; after the drain each proceeds.
GuardedSpec SpecSeqRestore() {
	GuardedSpec g;
	g.name = "sslm_gpu_seq_restore";
	g.build = [](World& w) {
		SslmGpuSequenceHandle* d = w.Dedicated();
		if (!d) return false;
		const std::vector<int32_t> t = Toks(w.s.art, 10, 37);
		const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(w.s.ctx, d, t.data(), 10, w.s.one_layer);
		w.blob = Save(w.s.ctx, d);
		G1C_CHECK(st == SSLM_OK && !w.blob.empty(), "SETUP: a dedicated sequence's blob: %s", St(st));
		return !w.blob.empty();
	};
	g.call = [](World& w) {
		w.out_seq = nullptr;
		return sslm_gpu_seq_restore(w.s.ctx, w.s.host, w.blob.data(), w.blob.size(), &w.out_seq);
	};
	g.check_refused = [](World& w) { G1C_CHECK(!w.out_seq, "seq_restore: a non-null out-handle on the refusal"); };
	g.output = [](World& w, SslmGpuStatus) {
		std::vector<int64_t> out{w.out_seq != nullptr};
		if (w.out_seq) out.push_back(Save(w.s.ctx, w.out_seq) == w.blob);
		return out;
	};
	return g;
}
GuardedSpec SpecSeqCreate() {
	GuardedSpec g;
	g.name = "sslm_gpu_seq_create";
	g.call = [](World& w) {
		w.out_seq = nullptr;
		return sslm_gpu_seq_create(w.s.ctx, w.s.host, 64, &w.out_seq);
	};
	g.check_refused = [](World& w) { G1C_CHECK(!w.out_seq, "seq_create: a non-null out-handle on the refusal"); };
	g.output = [](World& w, SslmGpuStatus) { return std::vector<int64_t>{w.out_seq != nullptr}; };
	return g;
}
GuardedSpec SpecModelMap() {
	GuardedSpec g;
	g.name = "sslm_gpu_model_map";
	g.call = [](World& w) {
		w.out_model = nullptr;
		return sslm_gpu_model_map(w.s.ctx, &w.s.art.view, GpuResidencyConfig{}, &w.out_model);
	};
	g.check_refused = [](World& w) { G1C_CHECK(!w.out_model, "model_map: a non-null out-handle on the refusal"); };
	g.output = [](World& w, SslmGpuStatus) { return std::vector<int64_t>{w.out_model != nullptr}; };
	return g;
}
GuardedSpec SpecAdapterMap(const Artifact* adapter) {
	GuardedSpec g;
	g.name = "sslm_gpu_adapter_map";
	g.call = [adapter](World& w) {
		w.out_adapter = nullptr;
		return sslm_gpu_adapter_map(w.s.ctx, w.s.host, &adapter->view, &w.out_adapter);
	};
	g.check_refused = [](World& w) { G1C_CHECK(!w.out_adapter, "adapter_map: a non-null out-handle on the refusal"); };
	g.output = [](World& w, SslmGpuStatus) { return std::vector<int64_t>{w.out_adapter != nullptr}; };
	return g;
}
GuardedSpec SpecPoolCreate() {
	GuardedSpec g;
	g.name = "sslm_gpu_kv_pool_create";
	g.call = [](World& w) {
		w.out_pool = nullptr;
		return sslm_gpu_kv_pool_create(w.s.ctx, w.s.host, 8, &w.out_pool);
	};
	g.check_refused = [](World& w) { G1C_CHECK(!w.out_pool, "pool_create: a non-null out-handle on the refusal"); };
	g.output = [](World& w, SslmGpuStatus) { return std::vector<int64_t>{w.out_pool != nullptr}; };
	return g;
}

void Run(GuardedSpec g, const char* id, bool resave) {
	g.name = std::string(id) + " " + g.name;
	if (!resave) g.resave_holder = nullptr;
	GuardedRun(g);
}

// ---------------------------------------------------------------------------------------------
// [G3] main runs.
// ---------------------------------------------------------------------------------------------
void G3BridgeNotReady() { Run(SpecBridgeNotReady(), "3.6d/G3", false); }
void G3BridgeReady() { Run(SpecBridgeReady(), "3.6d/G3", false); }
void G3Finish() { Run(SpecFinishDeviceLogits(), "3.6d/G3", false); }
void G3Decode() { Run(SpecDecode(AKind::PooledSamePool), "3.6d/G3", false); }
void G3Batch() { Run(SpecBatch(AKind::PooledSamePool), "3.6d/G3", false); }
void G3PromptPrefill() { Run(SpecPromptPrefill(AKind::PooledSamePool), "3.6d/G3", false); }
void G3SchemaPrefill() { Run(SpecSchemaPrefill(AKind::PooledSamePool), "3.6d/G3", false); }
void G3SeqRestore() { Run(SpecSeqRestore(), "3.6d/G3", false); }
void G3SeqCreate() { Run(SpecSeqCreate(), "3.6d/G3", false); }
void G3ModelMap() { Run(SpecModelMap(), "3.6d/G3", false); }
void G3AdapterMap() {
	const Artifact* a = AdapterOf(Opt().adapter, "--adapter");
	if (a) Run(SpecAdapterMap(a), "3.6d/G3", false);
}
void G3PoolCreate() { Run(SpecPoolCreate(), "3.6d/G3", false); }

// ---------------------------------------------------------------------------------------------
// [G3] precedence.
// ---------------------------------------------------------------------------------------------
bool FreshEmbedded(World& w) {
	if (!BuildFreshB(w)) return false;
	const SslmGpuStatus st = sslm_gpu_seq_embed_token(w.s.ctx, w.B, Toks(w.s.art, 1, 43)[0]);
	G1C_CHECK(st == SSLM_OK, "SETUP: embed on B -> %s", St(st));
	return st == SSLM_OK;
}

void G3PrecedencePromptCount0() {
	PrecedenceSpec p;
	p.name = "3.6d/G3 prompt prefill, count == 0";
	p.build = [](World& w) {
		w.toks = Toks(w.s.art, 4, 47);
		return BuildFreshB(w);
	};
	p.call = [](World& w) { return SslmGpuSeqPrefillPromptForG5Bridge(w.s.ctx, w.B, w.toks.data(), 0, w.s.one_layer); };
	p.expect_idle = {SSLM_OK};
	PrecedenceRun(p);
}
void G3PrecedenceSchemaUnbound() {
	PrecedenceSpec p;
	p.name = "3.6d/G3 schema-content prefill on an unbound B";
	p.build = [](World& w) {
		w.toks = Toks(w.s.art, 1, 53);
		return BuildFreshB(w);
	};
	p.call = [](World& w) {
		w.i32 = -1;
		return SslmGpuSeqPrefillSchemaContentForG5Bridge(w.s.ctx, w.B, w.toks.data(), 1, w.s.one_layer, &w.i32);
	};
	p.expect_idle = {SSLM_SEQUENCE_REJECTED};
	PrecedenceRun(p);
}
void G3PrecedenceFinishLayer0() {
	PrecedenceSpec p;
	p.name = "3.6d/G3 finish at layer_index 0";
	p.b_device_logits = true;
	p.build = [](World& w) { return BuildFreshB(w); };
	p.call = [](World& w) {
		w.i32 = -1;
		return SslmGpuSeqFinishTokenForG5Bridge(w.s.ctx, w.B, &w.i32);
	};
	p.expect_idle = {SSLM_SEQUENCE_REJECTED};
	PrecedenceRun(p);
}
void G3PrecedenceFinishHostLogits() {
	PrecedenceSpec p;
	p.name = "3.6d/G3 finish on a host-logits model (reaches no sink)";
	p.b_device_logits = false;
	p.build = [](World& w) { return BuildReadyB(w); };
	p.call = [](World& w) {
		w.i32 = -1;
		return SslmGpuSeqFinishTokenForG5Bridge(w.s.ctx, w.B, &w.i32);
	};
	p.expect_idle = {SSLM_OK};
	PrecedenceRun(p);
}
// Today's status of the embed's refusal of -1 is SSLM_TOKEN_ID_OUT_OF_RANGE: the code (gpu_1p0.cpp's
// embed), gpu_1p0.h (sslm_gpu_seq_embed_token: a token outside [0, vocab_size) returns TokenIdOutOfRange;
// the precedence list: the bridge decode returns "the embed's refusals"), and §7's own follow-on sentence
// for this cell. §7's bracket for this run reads SSLM_SEQUENCE_REJECTED, which disagrees with all three;
// §4.4's rule is "today's status otherwise", so the run asserts today's (notes: plan ambiguity, reported).
void G3PrecedenceBridgeEmbedRefusal() {
	PrecedenceSpec p;
	p.name = "3.6d/G3 bridge decode, the embed's refusal (token -1)";
	p.build = [](World& w) { return BuildFreshB(w); };
	p.call = [](World& w) {
		w.i32 = -7;
		return SslmGpuSeqDecodeStepForG5Bridge(w.s.ctx, w.B, -1, w.s.full_depth, &w.i32);
	};
	p.expect_idle = {SSLM_TOKEN_ID_OUT_OF_RANGE};
	PrecedenceRun(p);
}
void G3PrecedenceBridgeReadyHostLogits() {
	PrecedenceSpec p;
	p.name = "3.6d/G3 bridge decode, ready branch on a host-logits model";
	p.b_device_logits = false;
	p.build = [](World& w) { return BuildReadyB(w); };  // ready for logits, not bind-eligible
	p.call = [](World& w) {
		w.i32 = -7;
		return SslmGpuSeqDecodeStepForG5Bridge(w.s.ctx, w.B, Toks(w.s.art, 1, 59)[0], w.s.full_depth, &w.i32);
	};
	p.expect_idle = {SSLM_OK};
	// The separating operand: the follow-on with token -1 (the guard-below mutant clears
	// ready_for_logits before the inner guard refuses, so its B takes the embed branch and the embed
	// refuses -1 with SSLM_TOKEN_ID_OUT_OF_RANGE).
	p.follow_on = [](World& w) {
		int32_t tok = -7;
		const SslmGpuStatus st = SslmGpuSeqDecodeStepForG5Bridge(w.s.ctx, w.B, -1, w.s.full_depth, &tok);
		G1C_CHECK(st == SSLM_OK && tok >= 0, "ready branch follow-on: bridge decode with token -1 after the drain -> %s, "
		                                     "token %d; want SSLM_OK with a token (B still ready for logits)", St(st), tok);
	};
	PrecedenceRun(p);
}
void G3PrecedenceRestoreBadMagic() {
	PrecedenceSpec p;
	p.name = "3.6d/G3 sslm_gpu_seq_restore of a bad magic";
	p.build = [](World& w) {
		w.blob.assign(256, 'X');
		return true;
	};
	p.call = [](World& w) {
		w.out_seq = nullptr;
		const SslmGpuStatus st = sslm_gpu_seq_restore(w.s.ctx, w.s.host, w.blob.data(), w.blob.size(), &w.out_seq);
		G1C_CHECK(!w.out_seq, "seq_restore of a bad magic: a non-null out-handle");
		return st;
	};
	p.expect_idle = {SSLM_SEQUENCE_KV_BUFFER_MISMATCH};
	PrecedenceRun(p);
}
void G3PrecedenceBatchNullMember() {
	PrecedenceSpec p;
	p.name = "3.6d/G3 batch with a null member (its slot)";
	p.build = [](World& w) { return FreshEmbedded(w); };
	p.call = [](World& w) {
		SslmGpuSequenceHandle* seqs[2] = {w.B, nullptr};
		w.slots.assign(2, SSLM_DEVICE_LOST);
		const SslmGpuStatus call = sslm_decode_step_batch_gpu(w.s.ctx, seqs, nullptr, 2, w.s.full_depth, w.slots.data());
		w.i32 = static_cast<int32_t>(call);
		if (w.slots[0] == SSLM_OK) w.submitted.push_back(w.B);  // the idle leg's B decodes
		return w.slots[1];
	};
	p.check_submitted = [](World& w) {
		G1C_CHECK(w.i32 == static_cast<int32_t>(SSLM_BUSY) && w.slots[0] == SSLM_BUSY,
		          "batch [B, null] with A Submitted: call %s, slot 0 %s; want SSLM_BUSY in both",
		          St(static_cast<SslmGpuStatus>(w.i32)), St(w.slots[0]));
	};
	p.expect_idle = {SSLM_SEQUENCE_KV_BUFFER_MISMATCH};
	PrecedenceRun(p);
}
void G3PrecedenceModelMapMarshal() {
	// An artifact whose marshal fails: the model's own view with hidden_size off by one, so every
	// layer's attn_norm.gain length check in MarshalLayer refuses (layer_marshal.h). The header and
	// the loader are not involved; model map reads the view it is given.
	static Stack* s0 = GetStack();
	if (!s0) return;
	// A second load of the same artifact (the view is move-only), then the one field changed.
	static Artifact bad_art;
	static bool loaded = bad_art.Load(s0->art.path);
	G1C_CHECK(loaded, "SETUP: second load of %s failed", s0->art.path.c_str());
	if (!loaded) return;
	static superslm::SslmModelView& bad = bad_art.view;
	bad.config.hidden_size = s0->art.view.config.hidden_size + 1;
	PrecedenceSpec p;
	p.name = "3.6d/G3 model map of an artifact whose marshal fails";
	p.call = [](World& w) {
		w.out_model = nullptr;
		const SslmGpuStatus st = sslm_gpu_model_map(w.s.ctx, &bad, GpuResidencyConfig{}, &w.out_model);
		G1C_CHECK(!w.out_model, "model map of an unmarshalable view: a non-null out-handle");
		return st;
	};
	p.expect_idle = {SSLM_DEVICE_LOST};
	PrecedenceRun(p);
}
void G3PrecedenceAdapterOtherBase() {
	const Artifact* other = AdapterOf(Opt().adapter_other_base, "--adapter-other-base");
	if (!other) return;
	PrecedenceSpec p;
	p.name = "3.6d/G3 adapter map of an adapter of another base";
	p.call = [other](World& w) {
		w.out_adapter = nullptr;
		const SslmGpuStatus st = sslm_gpu_adapter_map(w.s.ctx, w.s.host, &other->view, &w.out_adapter);
		G1C_CHECK(!w.out_adapter, "adapter map of another base: a non-null out-handle");
		return st;
	};
	p.expect_idle = {SSLM_ADAPTER_BASE_HASH_MISMATCH};
	PrecedenceRun(p);
}
void G3PrecedenceBudgetTooSmall() {
	PrecedenceSpec p;
	p.name = "3.6d/G3 sslm_decode_step_gpu with a budget below one layer (precedes the guard)";
	p.build = [](World& w) { return FreshEmbedded(w); };
	p.call = [](World& w) { return sslm_decode_step_gpu(w.s.ctx, w.B, nullptr, w.s.one_layer - 1); };
	p.expect_submitted = SSLM_DISPATCH_BUDGET_TOO_SMALL;
	p.expect_idle = {SSLM_DISPATCH_BUDGET_TOO_SMALL};
	PrecedenceRun(p);
}

// ---------------------------------------------------------------------------------------------
// [G3] the mixed batch.
// ---------------------------------------------------------------------------------------------
struct OtherContext {
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	SslmGpuSequenceHandle* seq = nullptr;
	bool Open(Stack& s) {
		GpuContextConfig cfg{};
		SslmGpuStatus st = sslm_gpu_context_create(cfg, &ctx);
		if (st == SSLM_OK) st = sslm_gpu_model_map(ctx, &s.art.view, GpuResidencyConfig{}, &model);
		if (st == SSLM_OK) st = sslm_gpu_seq_create(ctx, model, 64, &seq);
		if (st == SSLM_OK) st = sslm_gpu_seq_embed_token(ctx, seq, Toks(s.art, 1, 61)[0]);
		G1C_CHECK(st == SSLM_OK, "SETUP: the other context's member -> %s", St(st));
		return st == SSLM_OK;
	}
	~OtherContext() {
		if (seq) (void)sslm_gpu_seq_release(ctx, seq);
		if (model) (void)sslm_gpu_model_unmap(ctx, model);
		if (ctx) (void)sslm_gpu_context_destroy(ctx);
	}
};

void G3MixedBatch() {
	Stack* s = GetStack();
	if (!s) return;
	OtherContext oc;
	if (!oc.Open(*s)) return;
	World w(*s);
	w.Pool(Opt().pool_pages);
	SslmGpuSequenceHandle* p = w.Pooled(0, 64);
	SslmGpuSequenceHandle* d = w.Dedicated();
	SslmGpuSequenceHandle* d2 = w.Dedicated();
	if (!w.ok) return;
	SslmGpuStatus st = sslm_gpu_seq_embed_token(s->ctx, p, Toks(s->art, 1, 67)[0]);
	if (st == SSLM_OK) st = sslm_gpu_seq_embed_token(s->ctx, d2, Toks(s->art, 1, 71)[0]);
	G1C_CHECK(st == SSLM_OK, "SETUP: embeds -> %s", St(st));
	if (!w.Submit(d, AToken(*s))) return;
	const SslmGpuStatus wt = sslm_gpu_test_only_wait_in_flight(s->ctx, d);
	G1C_CHECK(wt == SSLM_OK, "the wait seam on D -> %s", St(wt));
	struct Case {
		const char* name;
		SslmGpuSequenceHandle* m[2];
	};
	const Case cases[] = {
	    {"[D2, P]", {d2, p}},
	    {"[P, D2]", {p, d2}},
	    {"[P, null]", {p, nullptr}},
	    {"[another context's member, P]", {oc.seq, p}},
	};
	for (const Case& c : cases) {
		SslmGpuStatus slots[2] = {SSLM_DEVICE_LOST, SSLM_DEVICE_LOST};
		const SslmGpuStatus call = sslm_decode_step_batch_gpu(s->ctx, c.m, nullptr, 2, 2u * s->full_depth, slots);
		G1C_CHECK(call == SSLM_BUSY && slots[0] == SSLM_BUSY && slots[1] == SSLM_BUSY,
		          "mixed batch %s with a dedicated D Submitted and waited: call %s, slots %s %s; want SSLM_BUSY in "
		          "all three", c.name, St(call), St(slots[0]), St(slots[1]));
		for (int i = 0; i < 2; ++i) {  // a mutant's member that decoded is drained at teardown
			if (slots[i] == SSLM_OK && c.m[i] && c.m[i] != oc.seq) w.submitted.push_back(c.m[i]);
		}
	}
	const SslmGpuStatus dr = w.Drain(d);
	G1C_CHECK(dr == SSLM_OK, "sslm_gpu_ready(D) after the wait seam -> %s", St(dr));
}

// ---------------------------------------------------------------------------------------------
// [G3] scope.
// ---------------------------------------------------------------------------------------------
void G3Scope() {
	Stack* s = GetStack();
	if (!s) return;
	bool layer_on = false;
	{
		World w(*s);
		w.Pool(Opt().pool_pages);
		SslmGpuSequenceHandle* p = w.Pooled(0, 64);
		SslmGpuSequenceHandle* p2 = w.Pooled(0, 64);
		SslmGpuSequenceHandle* d = w.Dedicated();
		SslmGpuSequenceHandle* d2 = w.Dedicated();
		if (!w.ok) return;
		SslmGpuStatus st = sslm_gpu_seq_embed_token(s->ctx, p, Toks(s->art, 1, 73)[0]);
		if (st == SSLM_OK) st = sslm_gpu_seq_embed_token(s->ctx, d2, Toks(s->art, 1, 79)[0]);
		G1C_CHECK(st == SSLM_OK, "SETUP: embeds -> %s", St(st));
		(void)DebugLayerReportable(&layer_on);  // clear what earlier work queued
		if (!w.Submit(d, AToken(*s))) return;
		const SslmGpuStatus wt = sslm_gpu_test_only_wait_in_flight(s->ctx, d);
		G1C_CHECK(wt == SSLM_OK, "scope: the wait seam on D -> %s", St(wt));
		// Dedicated callers, a model: today's behaviour.
		const SslmGpuStatus c1 = sslm_gpu_seq_create(s->ctx, s->host, 64, &w.out_seq);
		G1C_CHECK(c1 == SSLM_OK && w.out_seq, "scope: seq_create with D Submitted and waited -> %s, want SSLM_OK", St(c1));
		w.TakeOuts();
		const SslmGpuStatus m1 = sslm_gpu_model_map(s->ctx, &s->art.view, GpuResidencyConfig{}, &w.out_model);
		G1C_CHECK(m1 == SSLM_OK && w.out_model, "scope: model map with D Submitted and waited -> %s, want SSLM_OK", St(m1));
		w.TakeOuts();
		// Pool create (an arc verb: any sequence) and calls on a pooled holder: refused.
		const SslmGpuStatus pc = sslm_gpu_kv_pool_create(s->ctx, s->host, 8, &w.out_pool);
		G1C_CHECK(pc == SSLM_BUSY && !w.out_pool, "scope: pool create with D Submitted -> %s, want SSLM_BUSY and null",
		          St(pc));
		w.TakeOuts();
		const SslmGpuStatus pd = sslm_decode_step_gpu(s->ctx, p, nullptr, s->full_depth);
		G1C_CHECK(pd == SSLM_BUSY, "scope: a decode on a pooled holder with D Submitted -> %s, want SSLM_BUSY", St(pd));
		if (pd == SSLM_OK) w.submitted.push_back(p);
		const std::vector<int32_t> t = Toks(s->art, 4, 83);
		const SslmGpuStatus pp = SslmGpuSeqPrefillPromptForG5Bridge(s->ctx, p2, t.data(), 4, s->one_layer);
		G1C_CHECK(pp == SSLM_BUSY, "scope: a prefill on a pooled holder with D Submitted -> %s, want SSLM_BUSY", St(pp));
		// The decode on a second dedicated sequence: today's SSLM_OK (nothing executes, nothing is
		// unobserved, no pooled sequence is Submitted).
		const SslmGpuStatus dd = sslm_decode_step_gpu(s->ctx, d2, nullptr, s->full_depth);
		G1C_CHECK(dd == SSLM_OK, "scope: a decode on a second dedicated sequence with D Submitted and waited -> %s, "
		                         "want SSLM_OK (must-reject: a guard scoped to any sequence for dedicated callers)", St(dd));
		if (dd == SSLM_OK) w.submitted.push_back(d2);
		(void)w.Drain(d2);
		(void)w.Drain(d);
		const size_t reportable = DebugLayerReportable(&layer_on);
		G1C_CHECK(layer_on && reportable == 0, "scope: %zu WARNING-or-worse debug-layer messages (layer on: %d; run "
		                                       "with SSLM_GPU_ENABLE_DEBUG_LAYER=1)", reportable, layer_on ? 1 : 0);
	}
	{
		// A pooled sequence Submitted: sslm_gpu_seq_create (a dedicated caller) is refused.
		World w(*s);
		w.Pool(Opt().pool_pages);
		SslmGpuSequenceHandle* p = w.Pooled(0, 64);
		if (!p || !w.Submit(p, AToken(*s))) return;
		const SslmGpuStatus c = sslm_gpu_seq_create(s->ctx, s->host, 64, &w.out_seq);
		G1C_CHECK(c == SSLM_BUSY && !w.out_seq, "scope: seq_create with a pooled sequence Submitted -> %s, want "
		                                        "SSLM_BUSY and null", St(c));
		w.TakeOuts();
	}
	{
		// D Submitted and NOT waited: the singleton's ledger holds an unobserved submission, so the
		// decode on the second dedicated sequence is refused (decision 68), where rev 7 reset the
		// singleton under D's list.
		World w(*s);
		SslmGpuSequenceHandle* d = w.Dedicated();
		SslmGpuSequenceHandle* d2 = w.Dedicated();
		if (!w.ok) return;
		const SslmGpuStatus e = sslm_gpu_seq_embed_token(s->ctx, d2, Toks(s->art, 1, 89)[0]);
		G1C_CHECK(e == SSLM_OK, "SETUP: embed -> %s", St(e));
		if (!w.Submit(d, AToken(*s))) return;
		const SslmGpuStatus dd = sslm_decode_step_gpu(s->ctx, d2, nullptr, s->full_depth);
		G1C_CHECK(dd == SSLM_BUSY, "scope: a decode on a second dedicated sequence with D Submitted and NOT waited -> "
		                           "%s, want SSLM_BUSY (the ledger's term)", St(dd));
		if (dd == SSLM_OK) w.submitted.push_back(d2);
	}
}

// ---------------------------------------------------------------------------------------------
// [G3] the assert's window: with nothing else Submitted, a pooled prefill returns SSLM_OK.
// ---------------------------------------------------------------------------------------------
void G3AssertWindow() {
	Stack* s = GetStack();
	if (!s) return;
	World w(*s);
	w.Pool(Opt().pool_pages);
	SslmGpuSequenceHandle* b = w.Pooled(0, 64);
	if (!b) return;
	const std::vector<int32_t> t = Toks(s->art, 20, 97);
	const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(s->ctx, b, t.data(), 20, s->one_layer);
	G1C_CHECK(st == SSLM_OK, "the assert's window: a pooled prefill with nothing else Submitted -> %s, want SSLM_OK", St(st));
}

#if PKVG_STEP >= 4
// ---------------------------------------------------------------------------------------------
// [G4]
// ---------------------------------------------------------------------------------------------
void G4Adopt() { Run(SpecAdopt(AKind::PooledSamePool), "3.6d/G4", false); }
void G4BeginFrom() { Run(SpecBeginFrom(AKind::PooledSamePool), "3.6d/G4", false); }
void G4PrefixPrefill() { Run(SpecPrefixPrefill(AKind::PooledSamePool), "3.6d/G4", false); }
void G4PrecedencePrefixCount0() {
	PrecedenceSpec p;
	p.name = "3.6d/G4 prefix prefill, count == 0 (a must-accept: the guard-below mutant is equivalent)";
	p.build = [](World& w) {
		w.Q = w.Prefix(0, 64, {}, false);
		w.toks = Toks(w.s.art, 4, 101);
		return w.Q != nullptr;
	};
	p.call = [](World& w) {
		w.i32 = -1;
		return sslm_gpu_prefix_prefill(w.s.ctx, w.Q, w.toks.data(), 0, 0, w.s.one_layer, &w.i32);
	};
	p.expect_idle = {SSLM_OK};
	PrecedenceRun(p);
}

#endif  // PKVG_STEP >= 4
#if PKVG_STEP >= 5
// ---------------------------------------------------------------------------------------------
// [G5]
// ---------------------------------------------------------------------------------------------
void G5RestoreIntoPool() { Run(SpecRestoreIntoPool(AKind::PooledSamePool), "3.6d/G5", false); }
// "Every run above adds B's re-saved SLM6 blob": every run whose B is a sequence (the make-type rows
// make no B, and begin_from's and the prefix prefill's B is a prefix, which has no save verb).
void G5ResaveBridgeNotReady() { Run(SpecBridgeNotReady(), "3.6d/G5 re-save", true); }
void G5ResaveBridgeReady() { Run(SpecBridgeReady(), "3.6d/G5 re-save", true); }
void G5ResaveFinish() { Run(SpecFinishDeviceLogits(), "3.6d/G5 re-save", true); }
void G5ResaveDecode() { Run(SpecDecode(AKind::PooledSamePool), "3.6d/G5 re-save", true); }
void G5ResaveBatch() { Run(SpecBatch(AKind::PooledSamePool), "3.6d/G5 re-save", true); }
void G5ResavePromptPrefill() { Run(SpecPromptPrefill(AKind::PooledSamePool), "3.6d/G5 re-save", true); }
void G5ResaveSchemaPrefill() { Run(SpecSchemaPrefill(AKind::PooledSamePool), "3.6d/G5 re-save", true); }
void G5ResaveAdopt() { Run(SpecAdopt(AKind::PooledSamePool), "3.6d/G5 re-save", true); }
// §4.10's order puts the magic after the guard. With nothing Submitted a blob whose magic is not 'SLM6'
// is refused SSLM_GPU_HOLDER_MODE_MISMATCH: §4 item 10's reading (anything that is not an SLM6 budget
// blob), gpu_1p0.h's step (3) for this verb, and the code (gpu_1p0.cpp's magic check). §7 3.6d's G5
// part reads SSLM_SEQUENCE_KV_BUFFER_MISMATCH; the §4.10 reading was taken over it (coverage audit 1.13 §3).
void G5PrecedenceRestoreIntoPoolBadMagic() {
	PrecedenceSpec p;
	p.name = "3.6d/G5 restore-into-pool of a bad magic";
	p.build = [](World& w) {
		w.blob.assign(256, 'X');
		return true;
	};
	p.call = [](World& w) {
		w.out_seq = nullptr;
		const SslmGpuStatus st = sslm_gpu_seq_restore_into_pool(w.s.ctx, w.pools[0].model, w.pools[0].h, w.blob.data(),
		                                                        w.blob.size(), nullptr, &w.out_seq, &w.u32);
		G1C_CHECK(!w.out_seq, "restore-into-pool of a bad magic: a non-null out-handle");
		return st;
	};
	p.expect_idle = {SSLM_GPU_HOLDER_MODE_MISMATCH};
	PrecedenceRun(p);
}
#endif  // PKVG_STEP >= 5
}  // namespace

G1C_CELL("3.6d/G3:bridge_decode_not_ready", "G3", G3BridgeNotReady);
G1C_CELL("3.6d/G3:bridge_decode_ready", "G3", G3BridgeReady);
G1C_CELL("3.6d/G3:finish_device_logits", "G3", G3Finish);
G1C_CELL("3.6d/G3:decode", "G3", G3Decode);
G1C_CELL("3.6d/G3:batch", "G3", G3Batch);
G1C_CELL("3.6d/G3:prompt_prefill", "G3", G3PromptPrefill);
G1C_CELL("3.6d/G3:schema_prefill", "G3", G3SchemaPrefill);
G1C_CELL("3.6d/G3:seq_restore", "G3", G3SeqRestore);
G1C_CELL("3.6d/G3:seq_create", "G3", G3SeqCreate);
G1C_CELL("3.6d/G3:model_map", "G3", G3ModelMap);
G1C_CELL("3.6d/G3:adapter_map", "G3", G3AdapterMap);
G1C_CELL("3.6d/G3:pool_create", "G3", G3PoolCreate);
G1C_CELL("3.6d/G3:precedence:prompt_count0", "G3", G3PrecedencePromptCount0);
G1C_CELL("3.6d/G3:precedence:schema_unbound", "G3", G3PrecedenceSchemaUnbound);
G1C_CELL("3.6d/G3:precedence:finish_layer0", "G3", G3PrecedenceFinishLayer0);
G1C_CELL("3.6d/G3:precedence:finish_host_logits", "G3", G3PrecedenceFinishHostLogits);
G1C_CELL("3.6d/G3:precedence:bridge_embed_refusal", "G3", G3PrecedenceBridgeEmbedRefusal);
G1C_CELL("3.6d/G3:precedence:bridge_ready_host_logits", "G3", G3PrecedenceBridgeReadyHostLogits);
G1C_CELL("3.6d/G3:precedence:seq_restore_bad_magic", "G3", G3PrecedenceRestoreBadMagic);
G1C_CELL("3.6d/G3:precedence:batch_null_member", "G3", G3PrecedenceBatchNullMember);
G1C_CELL("3.6d/G3:precedence:model_map_marshal", "G3", G3PrecedenceModelMapMarshal);
G1C_CELL("3.6d/G3:precedence:adapter_other_base", "G3", G3PrecedenceAdapterOtherBase);
G1C_CELL("3.6d/G3:precedence:budget_too_small", "G3", G3PrecedenceBudgetTooSmall);
G1C_CELL("3.6d/G3:mixed_batch", "G3", G3MixedBatch);
G1C_CELL("3.6d/G3:scope", "G3", G3Scope);
G1C_CELL("3.6d/G3:assert_window", "G3", G3AssertWindow);
#if PKVG_STEP >= 4
G1C_CELL("3.6d/G4:adopt", "G4", G4Adopt);
G1C_CELL("3.6d/G4:begin_from", "G4", G4BeginFrom);
G1C_CELL("3.6d/G4:prefix_prefill", "G4", G4PrefixPrefill);
G1C_CELL("3.6d/G4:precedence:prefix_count0", "G4", G4PrecedencePrefixCount0);
#endif
#if PKVG_STEP >= 5
G1C_CELL("3.6d/G5:restore_into_pool", "G5", G5RestoreIntoPool);
G1C_CELL("3.6d/G5:resave-bridge_decode_not_ready", "G5", G5ResaveBridgeNotReady);
G1C_CELL("3.6d/G5:resave-bridge_decode_ready", "G5", G5ResaveBridgeReady);
G1C_CELL("3.6d/G5:resave-finish_device_logits", "G5", G5ResaveFinish);
G1C_CELL("3.6d/G5:resave-decode", "G5", G5ResaveDecode);
G1C_CELL("3.6d/G5:resave-batch", "G5", G5ResaveBatch);
G1C_CELL("3.6d/G5:resave-prompt_prefill", "G5", G5ResavePromptPrefill);
G1C_CELL("3.6d/G5:resave-schema_prefill", "G5", G5ResaveSchemaPrefill);
G1C_CELL("3.6d/G5:resave-adopt", "G5", G5ResaveAdopt);
G1C_CELL("3.6d/G5:precedence:restore_into_pool_bad_magic", "G5", G5PrecedenceRestoreIntoPoolBadMagic);
#endif
