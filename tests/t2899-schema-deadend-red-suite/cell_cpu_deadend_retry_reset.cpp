// T-2899 (Curie) -- plan `Claude/Plans/te266-gpu-path.md` Sec3.10.5/Sec3.10.6, the CPU half of
// D-SLM3476's retry guarantee: CPU Cell 1 (iii) (retry after a dead end) and Cell 1 (iv)
// (resettability after a dead end), over routes R, D, C and D1, plus the save/restore
// round-trip cell (Sec3.10.5's own residual).
//
// ADOPTED, PER BRIEF, NOT RE-DERIVED. This cell's driving logic is TE-338's own
// `Claude/Loki/te338-probe/probe_cpu.cpp` (routes R, D, C) and TE-361's own
// `Claude/Loki/te361-probe/probe_cpu_recover.cpp` (route D1, the reset/recover/reuse checks,
// the save/restore round-trip census leg) -- both already executed against the real shipped
// library and cited by the plan (Sec3.10.5/Sec3.10.6) as the authored-cell source. Converted
// from print-and-eyeball probes into CHECK-asserted cells (this suite's own convention); the
// underlying calls, routes and fixture are unchanged.
//
// FIXTURE. --model=PATH, the C39 synthetic (`tests/t2199-damped-greedy-red-suite`'s own
// `t2199_s8_fixture.sslm`, `a8f88ab9...8097`; see cell_adopt_prefix_census.cpp's own header for
// the generator-reproducibility note this suite shares). t0 (the schema's own admitted token
// from a fresh bind) is DERIVED by construction (a throwaway probe-prefill per candidate, per
// T-2872's own correction of TE-338's original hardcoded t0=0), not assumed.
//
// ROUTES (identical to TE-338/TE-361's own construction):
//   R  -- prompt P, then a schema-content prefill of {t0} reaches the dead end from the READY
//         branch (layer_index == 0 already).
//   D  -- prompt P only; greedy decode itself produces t0 (the accepting token), so the SECOND
//         decode call reaches the dead end from the EMBED branch (layer_index ==
//         num_hidden_layers after that call's own drive).
//   C  -- prompt fills context_cap-1, schema-content {t0} fills the cap exactly (the dead end
//         coincides with a saturated context).
//   D1 -- route D at layer_budget=1 (T-2894's own added route): the dead end is reached inside
//         a partial-budget generation loop, not a single full-depth call.
//
// ORACLE. The specification (plan Sec3.10.1/Sec3.10.5), not "whatever the shipped CPU already
// does" -- TE-338 found the shipped promise held on 0 of 3 routes before the fix; this cell's
// own acceptance bar is the promise text, and its own red-now numbers (below) are read as
// confirming TE-338's own finding, not as this cell's target.
//
// RED AT a3f89cb (pristine): every repeat-decode call after route R/D's first dead end embeds
// a NEW token and advances layer_index (violating "no new token" -- Cell 1(iii)); route C's
// repeat call returns SSLM_CONTEXT_CAP_EXCEEDED, not -2/SSLM_OK; `sslm_seq_reset` after any
// route's dead end refuses SSLM_SEQ_RESET_MIDTOKEN_REJECTED on routes D/D1 (violating Cell
// 1(iv)); routes R/C's own dead end already rests at layer_index==0 so their OWN reset already
// succeeds at a3f89cb (T-2866's own noted asymmetry) -- this cell asserts (iv) on ALL FOUR
// routes and is red on exactly D and D1, matching TE-361's own finding precisely.
//
// GUARD VITALITY, two independent mutants against the reference fix
// (`Claude/Vitruvius/t2898-probe/sslm_abi_cpu_fixed_v5.cpp`, cumulative through T-2894):
//   MUT_NOREARM   : both miss sites' `seq->ready_for_logits = true;` reverted to a no-op --
//                   must turn (iii) red on every route (a repeat call embeds a new token again).
//   MUT_NORESET   : both miss sites' `seq->state.layer_index = 0;` reverted to a no-op (T-2894's
//                   own line) -- must turn (iv) red on exactly routes D and D1 while (iii) STAYS
//                   GREEN on every route (the two assertions are independent -- TE-361's own
//                   finding: a fix that held (iii) and silently failed (iv)).
//
// Run: cell_cpu_deadend_retry_reset.exe --model=PATH
#include "superslm/model.h"
#include "superslm/sslm_abi.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int GChecks = 0;
static int GFailures = 0;
static int GSkips = 0;

#define CHECK(cond)                                                     \
	do {                                                                \
		++GChecks;                                                     \
		if (!(cond)) {                                                 \
			++GFailures;                                               \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			std::fflush(stdout);                                       \
		}                                                              \
	} while (0)

#define CHECK_MSG(cond, ...)                                            \
	do {                                                                \
		++GChecks;                                                     \
		if (!(cond)) {                                                 \
			++GFailures;                                               \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__);                                  \
			std::printf("\n");                                         \
			std::fflush(stdout);                                       \
		}                                                              \
	} while (0)

namespace {

uint32_t Le32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24); }
uint64_t Le64(const uint8_t* p) { return static_cast<uint64_t>(Le32(p)) | (static_cast<uint64_t>(Le32(p + 4)) << 32); }

// Blob field offsets: TE-338/TE-361/TE-366's own established technique (`sslm_seq_save`'s v4
// layout, `sslm_abi.cpp`'s own save/restore block) -- reused here rather than re-derived, since
// it is what every prior probe's own headline numbers were captured against. Only
// context_length (offset 60) is read; it has no public accessor (`sslm_stats_out` does not
// carry it), and it is the load-bearing "no new token committed" observable this cell's own
// Cell 1(iii)/(iv) claims are graded on.
int64_t BlobContextLength(sslm_seq s) {
	size_t n = 0;
	if (sslm_seq_save(s, nullptr, &n) != SSLM_BUFFER_TOO_SMALL) return -1;
	std::vector<uint8_t> b(n);
	size_t nn = n;
	if (sslm_seq_save(s, b.data(), &nn) != SSLM_OK || nn < 68) return -1;
	return static_cast<int64_t>(Le64(&b[60]));
}

// Offset 48: dfa_walk_state, LE32 -- the SAME 'SSB3'/'SSB4' fixed-header layout
// BlobContextLength's own offset-60 read is grounded in (src/sslm_abi.cpp's own save-side
// comment: magic(4) + model_hash(32) + kv_precision(4) + schema_name_hash(8) =
// 48, then dfa_walk_state(4) at 48..52). No public accessor exists for this field either;
// the blob is the only host-visible read, exactly as for context_length above.
uint32_t BlobDfaWalkState(sslm_seq s) {
	size_t n = 0;
	if (sslm_seq_save(s, nullptr, &n) != SSLM_BUFFER_TOO_SMALL) return 0xFFFFFFFFu;
	std::vector<uint8_t> b(n);
	size_t nn = n;
	if (sslm_seq_save(s, b.data(), &nn) != SSLM_OK || nn < 52) return 0xFFFFFFFFu;
	return Le32(&b[48]);
}

struct Cpu {
	std::vector<uint8_t> bytes;
	sslm_model model = nullptr;
	std::vector<uint8_t> pool_store;
	sslm_kv_pool pool = nullptr;
	sslm_schema schema = nullptr;
	int32_t layers = 0;
	int32_t vocab = 0;
	int64_t cap = 0;

	~Cpu() {
		if (pool) sslm_kv_pool_destroy(pool);
		if (model) sslm_model_unmap(model);
	}
};

bool CpuOpen(Cpu* g, const std::string& path, std::string* err) {
	FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) {
		*err = "could not open " + path;
		return false;
	}
	std::fseek(f, 0, SEEK_END);
	long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	g->bytes.resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	if (sz > 0) std::fread(g->bytes.data(), 1, static_cast<size_t>(sz), f);
	std::fclose(f);

	superslm::SslmModelView view;
	if (superslm::SslmModel::Load(g->bytes.data(), g->bytes.size(), view, err) !=
	    superslm::SslmModelStatus::Ok) {
		return false;
	}
	g->layers = static_cast<int32_t>(view.config.num_hidden_layers);
	g->vocab = view.config.vocab_size;
	g->cap = view.config.context_cap;
	if (sslm_model_map(g->bytes.data(), g->bytes.size(), &g->model) != SSLM_OK) {
		*err = "sslm_model_map failed";
		return false;
	}
	const uint32_t blocks = 4;
	size_t bb = sslm_kv_block_size(g->model), ov = sslm_kv_pool_overhead_size(g->model, blocks);
	g->pool_store.assign(bb * blocks + ov + 63, 0);
	void* a = g->pool_store.data();
	size_t space = g->pool_store.size();
	std::align(64, bb * blocks + ov, a, space);
	if (sslm_kv_pool_create(g->model, a, bb * blocks + ov, blocks, &g->pool) != SSLM_OK) {
		*err = "sslm_kv_pool_create failed";
		return false;
	}
	size_t n = 0;
	sslm_schema_name(g->model, 0, nullptr, &n);
	std::string name(n, '\0');
	sslm_schema_name(g->model, 0, name.data(), &n);
	if (!name.empty() && name.back() == '\0') name.pop_back();
	if (sslm_schema_lookup(g->model, name.c_str(), &g->schema) != SSLM_OK) {
		*err = "sslm_schema_lookup(" + name + ") failed";
		return false;
	}
	return true;
}

sslm_seq NewSeq(Cpu& g) {
	sslm_seq s = nullptr;
	CHECK(sslm_seq_create(g.model, &g.pool, &s) == SSLM_OK && s);
	if (s) CHECK(sslm_seq_set_schema(s, g.schema) == SSLM_OK);
	return s;
}

sslm_status PrefillAll(Cpu& g, sslm_seq s, const std::vector<int32_t>& t, sslm_span_kind kind) {
	int32_t off = 0;
	while (off < static_cast<int32_t>(t.size())) {
		int32_t c = 0;
		sslm_status st =
		    sslm_prefill(g.model, s, t.data() + off, static_cast<int32_t>(t.size()) - off, 8, kind, nullptr, &c);
		if (st != SSLM_OK) return st;
		if (c <= 0) return SSLM_INVALID_ARGUMENT;
		off += c;
	}
	return SSLM_OK;
}

int32_t Decode1(Cpu& g, sslm_seq s, sslm_status* out_st, int32_t layer_budget) {
	sslm_decode_params p{};
	p.struct_size = sizeof(p);
	p.layer_budget = layer_budget > 0 ? layer_budget : g.layers;
	int32_t out = 12345;
	*out_st = sslm_decode_step(g.model, &s, 1, &p, nullptr, &out);
	return out;
}

int32_t DecodeDamped(Cpu& g, sslm_seq s, sslm_status* out_st) {
	sslm_decode_params p{};
	if (sslm_decode_params_init(g.model, SSLM_DECODE_MODE_DAMPED_GREEDY, g.layers, &p) != SSLM_OK) {
		*out_st = SSLM_INVALID_ARGUMENT;
		return 12345;
	}
	int32_t out = 12345;
	*out_st = sslm_decode_step_v2(g.model, &s, 1, &p, nullptr, &out);
	return out;
}

// Derived by construction (T-2872's own discipline): the schema's own admitted token from a
// fresh binding, found by trying candidates until one is accepted rather than assumed.
int32_t DeriveAcceptedToken(Cpu& g, const std::vector<int32_t>& P) {
	for (int32_t cand = 0; cand < g.vocab; ++cand) {
		sslm_seq s = nullptr;
		if (sslm_seq_create(g.model, &g.pool, &s) != SSLM_OK) continue;
		if (sslm_seq_set_schema(s, g.schema) != SSLM_OK) {
			sslm_seq_release(s);
			continue;
		}
		if (PrefillAll(g, s, P, SSLM_SPAN_PROMPT) != SSLM_OK) {
			sslm_seq_release(s);
			continue;
		}
		int32_t consumed = 0;
		const sslm_status st = sslm_prefill(g.model, s, &cand, 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed);
		sslm_seq_release(s);
		if (st == SSLM_OK && consumed == 1) return cand;
	}
	return -1;
}

struct DeadEndState {
	sslm_seq seq = nullptr;
	int64_t ctx_after_first_miss = -1;
};

// Drives `route` to its first dead end and returns the live handle sitting there, plus the
// context_length observed right after the FIRST -2. Cell 1(iii) is checked by the caller over
// subsequent calls.
DeadEndState ReachDeadEnd(Cpu& g, const std::string& route, const std::vector<int32_t>& P, int32_t t0) {
	DeadEndState d;
	d.seq = NewSeq(g);
	if (!d.seq) return d;
	if (route == "R") {
		CHECK_MSG(PrefillAll(g, d.seq, P, SSLM_SPAN_PROMPT) == SSLM_OK, "[R] prompt prefill");
		CHECK_MSG(PrefillAll(g, d.seq, {t0}, SSLM_SPAN_SCHEMA_CONTENT) == SSLM_OK, "[R] schema-content prefill");
		sslm_status st{};
		const int32_t out = Decode1(g, d.seq, &st, -1);
		CHECK_MSG(st == SSLM_OK && out == -2, "[R] first decode after the accepting prefill: st=%d out=%d, want -2/OK",
		          static_cast<int>(st), out);
	} else if (route == "D") {
		CHECK_MSG(PrefillAll(g, d.seq, P, SSLM_SPAN_PROMPT) == SSLM_OK, "[D] prompt prefill");
		sslm_status st1{};
		const int32_t tok1 = Decode1(g, d.seq, &st1, -1);
		CHECK_MSG(st1 == SSLM_OK && tok1 == t0, "[D] first decode should PRODUCE the accepting token %d: st=%d out=%d",
		          t0, static_cast<int>(st1), tok1);
		sslm_status st2{};
		const int32_t out2 = Decode1(g, d.seq, &st2, -1);
		CHECK_MSG(st2 == SSLM_OK && out2 == -2, "[D] second decode should dead-end: st=%d out=%d",
		          static_cast<int>(st2), out2);
	} else if (route == "C") {
		std::vector<int32_t> fill;
		for (int64_t i = 0; i < g.cap - 1; ++i) fill.push_back(P[static_cast<size_t>(i) % P.size()]);
		CHECK_MSG(PrefillAll(g, d.seq, fill, SSLM_SPAN_PROMPT) == SSLM_OK, "[C] cap-filling prompt prefill");
		CHECK_MSG(PrefillAll(g, d.seq, {t0}, SSLM_SPAN_SCHEMA_CONTENT) == SSLM_OK, "[C] schema-content prefill at the cap");
		sslm_status st{};
		const int32_t out = Decode1(g, d.seq, &st, -1);
		CHECK_MSG(st == SSLM_OK && out == -2, "[C] first decode at the cap: st=%d out=%d, want -2/OK",
		          static_cast<int>(st), out);
	}
	d.ctx_after_first_miss = BlobContextLength(d.seq);
	return d;
}

// Cell 1(iii): a repeat call after the FIRST dead end consumes no new token and does not move
// context_length. Checked for two further calls, matching TE-338/T-2866's own construction.
void CheckRetry(Cpu& g, const std::string& route, sslm_seq s, int64_t ctx0) {
	for (int call = 2; call <= 3; ++call) {
		sslm_status st{};
		const int32_t out = Decode1(g, s, &st, -1);
		const int64_t ctx = BlobContextLength(s);
		CHECK_MSG(st == SSLM_OK && out == -2,
		          "[%s] Cell1(iii) repeat call #%d: st=%d out=%d, want -2/SSLM_OK (a repeat call must dead-end "
		          "identically, never embed a new token)",
		          route.c_str(), call, static_cast<int>(st), out);
		CHECK_MSG(ctx == ctx0,
		          "[%s] Cell1(iii) repeat call #%d: context_length moved %lld -> %lld -- a repeat call must "
		          "commit no new token",
		          route.c_str(), call, static_cast<long long>(ctx0), static_cast<long long>(ctx));
	}
}

// Cell 1(iv): after the dead end, sslm_seq_reset succeeds and the sequence is fully reusable.
void CheckResettable(Cpu& g, const std::string& route, sslm_seq s, const std::vector<int32_t>& P) {
	const sslm_status rst = sslm_seq_reset(s);
	CHECK_MSG(rst == SSLM_OK, "[%s] Cell1(iv): sslm_seq_reset after the dead end returned %d, want SSLM_OK",
	          route.c_str(), static_cast<int>(rst));
	if (rst != SSLM_OK) return;
	const sslm_status us = sslm_seq_set_schema(s, nullptr);
	CHECK_MSG(us == SSLM_OK, "[%s] Cell1(iv): unbind after reset returned %d, want SSLM_OK", route.c_str(),
	          static_cast<int>(us));
	const sslm_status ua = sslm_seq_set_adapter(s, nullptr);
	CHECK_MSG(ua == SSLM_OK, "[%s] Cell1(iv): set_adapter(null) after reset returned %d, want SSLM_OK", route.c_str(),
	          static_cast<int>(ua));
	CHECK_MSG(sslm_seq_set_schema(s, g.schema) == SSLM_OK, "[%s] Cell1(iv): re-bind after reset", route.c_str());
	const sslm_status pp = PrefillAll(g, s, {P[0]}, SSLM_SPAN_PROMPT);
	CHECK_MSG(pp == SSLM_OK, "[%s] Cell1(iv): prompt prefill after reset returned %d", route.c_str(),
	          static_cast<int>(pp));
	sslm_status ds{};
	const int32_t tok = Decode1(g, s, &ds, -1);
	CHECK_MSG(ds == SSLM_OK && tok >= 0,
	          "[%s] Cell1(iv): decode after reset+reuse: st=%d out=%d, want a real, non-negative token",
	          route.c_str(), static_cast<int>(ds), tok);
}

// Sec3.10.5's own residual: save a sequence resting at the dead end BEFORE reset is ever called
// on the original handle, restore into a fresh handle, and confirm the restored copy resets and
// reuses while the original's own state is unperturbed.
void CheckSaveRestoreRoundTrip(Cpu& g, const std::string& route, const std::vector<int32_t>& P, int32_t t0) {
	DeadEndState d = ReachDeadEnd(g, route, P, t0);
	if (!d.seq) return;
	sslm_status ignore{};
	Decode1(g, d.seq, &ignore, -1);  // one repeat call, still at the dead end (Cell 1(iii))

	size_t n = 0;
	CHECK(sslm_seq_save(d.seq, nullptr, &n) == SSLM_BUFFER_TOO_SMALL);
	std::vector<uint8_t> blob(n);
	size_t written = n;
	CHECK(sslm_seq_save(d.seq, blob.data(), &written) == SSLM_OK);

	sslm_seq restored = nullptr;
	const sslm_status rs = sslm_seq_restore(g.model, &g.pool, blob.data(), written, &restored);
	CHECK_MSG(rs == SSLM_OK && restored, "[%s] save/restore: restore returned %d", route.c_str(),
	          static_cast<int>(rs));
	if (rs == SSLM_OK && restored) {
		const sslm_status rr = sslm_seq_reset(restored);
		CHECK_MSG(rr == SSLM_OK, "[%s] save/restore: reset on the RESTORED copy returned %d, want SSLM_OK",
		          route.c_str(), static_cast<int>(rr));
		CHECK(sslm_seq_set_schema(restored, g.schema) == SSLM_OK);
		CHECK(PrefillAll(g, restored, {P[0]}, SSLM_SPAN_PROMPT) == SSLM_OK);
		sslm_status ds{};
		const int32_t tok = Decode1(g, restored, &ds, -1);
		CHECK_MSG(ds == SSLM_OK && tok >= 0, "[%s] save/restore: decode on the restored, reset copy: st=%d out=%d",
		          route.c_str(), static_cast<int>(ds), tok);
		// Control: the ORIGINAL handle's own state is unperturbed by the save/restore performed
		// on the copy.
		sslm_status origst{};
		const int32_t origout = Decode1(g, d.seq, &origst, -1);
		CHECK_MSG(origst == SSLM_OK && origout == -2,
		          "[%s] save/restore control: the ORIGINAL handle's next decode changed to st=%d out=%d "
		          "(want it still -2/SSLM_OK, unperturbed by the copy's own save/restore)",
		          route.c_str(), static_cast<int>(origst), origout);
		sslm_seq_release(restored);
	}
	sslm_seq_release(d.seq);
}

// Drives a fresh sequence, at layer_budget=1, to its third finished decode call (the dead
// end -- the first two finished calls are the schema's own accepted content, matching route
// D's own single-step shape). Returns the live handle and the context_length observed there.
DeadEndState ReachD1DeadEnd(Cpu& g, const std::vector<int32_t>& P) {
	DeadEndState d;
	d.seq = NewSeq(g);
	if (!d.seq) return d;
	CHECK_MSG(PrefillAll(g, d.seq, P, SSLM_SPAN_PROMPT) == SSLM_OK, "[D1] prompt prefill");
	sslm_decode_params p{};
	p.struct_size = sizeof(p);
	p.layer_budget = 1;
	int finished = 0, calls = 0;
	int32_t last_out = 12345;
	sslm_status last_st{};
	while (finished < 3 && calls < 256) {
		int32_t out = 12345;
		last_st = sslm_decode_step(g.model, &d.seq, 1, &p, nullptr, &out);
		++calls;
		if (last_st != SSLM_OK || out != -1) {
			last_out = out;
			++finished;
		}
	}
	CHECK_MSG(finished == 3, "[D1] partial-budget generation loop did not reach 3 finished calls in %d steps",
	          calls);
	CHECK_MSG(last_st == SSLM_OK && last_out == -2,
	          "[D1] the third finished call should be the dead end: st=%d out=%d", static_cast<int>(last_st),
	          last_out);
	d.ctx_after_first_miss = BlobContextLength(d.seq);
	return d;
}

void RouteD1PartialBudget(Cpu& g, const std::vector<int32_t>& P) {
	DeadEndState d_iii = ReachD1DeadEnd(g, P);
	if (d_iii.seq) {
		CheckRetry(g, "D1", d_iii.seq, d_iii.ctx_after_first_miss);
		sslm_seq_release(d_iii.seq);
	}
	DeadEndState d_iv = ReachD1DeadEnd(g, P);
	if (d_iv.seq) {
		CheckResettable(g, "D1", d_iv.seq, P);
		sslm_seq_release(d_iv.seq);
	}
}

// The damped-greedy branch's own independent copy of the miss-and-continue block
// (`sslm_abi.cpp:2461-2467` at Stage 1) -- T-2872/T-2894's own owed cell (plan Sec3.10.6):
// "the damped-greedy-mode retry cell... and its own resettability twin, both authored as
// formal red cells." Route D only (prompt P, first damped-greedy decode call produces the
// accepting token, second dead-ends) -- matches T-2872/T-2894's own construction; routes R and
// C are not swept here because their dead end is reached from the ready branch exactly as on
// the greedy path (already covered by CheckRetry/CheckResettable's own route R/C legs), per
// the plan's own note that the damped-greedy miss site's fix is identical to the greedy site's,
// line for line.
DeadEndState ReachDampedDeadEnd(Cpu& g, const std::vector<int32_t>& P, int32_t t0) {
	DeadEndState d;
	d.seq = NewSeq(g);
	if (!d.seq) return d;
	CHECK_MSG(PrefillAll(g, d.seq, P, SSLM_SPAN_PROMPT) == SSLM_OK, "[DG-D] prompt prefill");
	sslm_status st1{};
	const int32_t tok1 = DecodeDamped(g, d.seq, &st1);
	CHECK_MSG(st1 == SSLM_OK && tok1 == t0,
	          "[DG-D] first damped-greedy decode should PRODUCE the accepting token %d: st=%d out=%d", t0,
	          static_cast<int>(st1), tok1);
	sslm_status st2{};
	const int32_t out2 = DecodeDamped(g, d.seq, &st2);
	CHECK_MSG(st2 == SSLM_OK && out2 == -2, "[DG-D] second damped-greedy decode should dead-end: st=%d out=%d",
	          static_cast<int>(st2), out2);
	d.ctx_after_first_miss = BlobContextLength(d.seq);
	return d;
}

void CheckDampedRetry(Cpu& g, sslm_seq s, int64_t ctx0) {
	for (int call = 2; call <= 3; ++call) {
		sslm_status st{};
		const int32_t out = DecodeDamped(g, s, &st);
		const int64_t ctx = BlobContextLength(s);
		CHECK_MSG(st == SSLM_OK && out == -2,
		          "[DG-D] Cell1(iii) damped-greedy repeat call #%d: st=%d out=%d, want -2/SSLM_OK", call,
		          static_cast<int>(st), out);
		CHECK_MSG(ctx == ctx0,
		          "[DG-D] Cell1(iii) damped-greedy repeat call #%d: context_length moved %lld -> %lld", call,
		          static_cast<long long>(ctx0), static_cast<long long>(ctx));
	}
}

void DampedGreedyRoute(Cpu& g, const std::vector<int32_t>& P, int32_t t0) {
	DeadEndState d_iii = ReachDampedDeadEnd(g, P, t0);
	if (d_iii.seq) {
		CheckDampedRetry(g, d_iii.seq, d_iii.ctx_after_first_miss);
		sslm_seq_release(d_iii.seq);
	}
	DeadEndState d_iv = ReachDampedDeadEnd(g, P, t0);
	if (d_iv.seq) {
		CheckResettable(g, "DG-D", d_iv.seq, P);
		sslm_seq_release(d_iv.seq);
	}
}

// CPU Cell 2 twin -- the degenerate-row construction, for real (plan Sec3.10.6, T-2872
// closing T-2870 F4; TE-365 C2: the seam was armed and returned without asserting, and the
// unconsumed arm then fired inside the NEXT cell's own route R -- both closed below).
//
// FIXTURE. A SEPARATE model from every other cell in this file: the C39 synthetic
// (`t2199_s8_fixture.sslm`) carries only `g5_minimal_one_field`, a single boolean leaf that
// compiles to 2 states and 1 transition (`tools/_t2199_s8_synthetic_full_model_fixture.py`'s
// own docstring) -- its only non-start state IS the accepting state, with no interior state
// to reach S_e from. `g` here is opened by `main` against
// `make_t2909_string_schema_fixture.py`'s own hermetic artifact instead: a real schema
// compiled by the production compiler (T-2853's own leaf, TE-365 C1) with a `Prompt_Result`
// string field, and a real vocabulary whose first ids are that generator's own
// TOK_OPEN/TOK_OPEN_QUOTE/TOK_CONTENT/TOK_BACKSLASH/TOK_ESCAPE_N/TOK_CLOSE/TOK_BRACE pieces --
// reproduced here as the `kT2909Tok*` constants below, kept in lockstep with that file's own
// module constants rather than re-derived independently (both this cell and the GPU twin,
// `cell_gpu_cell2_degenerate.cpp`, read the identical fixture and the identical ids).
//
// CONSTRUCTION. `ReachSeCpu` prefills a prompt (three filler ids, ordinary vocabulary
// pieces the schema's own DFA never transitions on), binds the schema, then drives THREE
// REAL, ADMITTED schema-content transitions in one `sslm_prefill` span: TOK_OPEN (state 0 ->
// the pre-value literal state), TOK_OPEN_QUOTE (-> S_c, T-2910's own depth-0-only opening --
// the literal object skeleton and the value's opening quote no longer share one vocabulary
// piece, T-2915) and TOK_BACKSLASH (S_c -> S_e) -- none is the degenerate-row construction;
// all three are ordinary, correct transitions the compiler's own DFA admits, exactly as the
// plan's own "a real, admitted backslash transition from S_c" specifies. The resulting
// sequence rests
// at S_e with `layer_index == 0`/`ready_for_logits == true` (the SAME resting shape Cell
// 1's own routes reach), so the next `sslm_decode_step` call reaches Finish directly.
//
// ASSERTIONS. (i)/(ii): with the seam armed, Finish presents a synthetic all-`INT32_MIN` row
// at S_e -- `-2` at `SSLM_OK` (never a produced token), and `dfa_walk_state` PINNED at S_e
// (this closes the C2 finding's own "unconsumed arm" side effect: the seam is single-shot and
// is always exercised by THIS call, on THIS handle, never left to fire unconsumed on a later,
// unrelated cell). Must-accept neighbour (plan Sec3.10.2): the IDENTICAL construction, un-
// armed -- the real, non-degenerate row lets the masked argmax select one of S_e's own three
// admitted escapes (TOK_BACKSLASH/TOK_ESCAPE_N/TOK_CLOSE, each admitted from S_e per the
// generator's own SETUP self-check) and the walk leaves S_e (advances to S_c, per the
// compiler's own construction -- every one of S_e's admitted tokens returns to S_c, since an
// escape sequence can never itself close the string).
//
// GUARD VITALITY (T-2909, closing the C2 residual the plan's own row 11 names for the GPU
// twin -- "must turn both Cell 1 and Cell 2 red" -- CPU-side: the seam's own consumption
// call is what stands in for CPU's "checked return," since CPU's pre-existing masked-argmax
// `has_transition` check was already correct at a3f89cb, D-SLM3476, Sec3.10.4's own cost
// table). `run_mutants_cpu_deadend.bat`'s own `MUT_NOSEAM` variant links this SAME cell
// against a scratch copy of `sslm_abi.cpp` with ONLY the
// `MaybeInjectCpuFinishDegenerateLogitRow(logit_row, ...)` call deleted (the seam's
// consumption site, `sslm_abi.cpp:2553`) -- the flag still arms, but nothing ever reads it,
// so Finish presents the REAL (uncorrupted) row at S_e regardless. Cell 2(i)'s own assertion
// (`out == -2`) must turn red under that mutant while every other cell in this same binary
// (Cell 1's routes, D1, damped-greedy, save/restore) stays green, since nothing else in this
// file ever arms the flag the deleted call would have consumed.
#if defined(SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION)
extern "C" void ArmCpuFinishDegenerateLogitRowInjection();

// Token ids from `make_t2909_string_schema_fixture.py` (kept in lockstep with that
// generator's own module constants TOK_OPEN/TOK_OPEN_QUOTE/TOK_CONTENT/TOK_BACKSLASH/
// TOK_ESCAPE_N/TOK_CLOSE/TOK_BRACE). T-2915 (porting the plan's final Sec3.9 design): T-2910's
// boundary discipline refuses a token that opens the string's content sub-automaton past its
// own first byte, so the generator's own literal object skeleton and the value's opening
// quote no longer share one vocabulary piece -- reaching S_e now takes three fed tokens
// (open, open-quote, backslash) rather than two.
constexpr int32_t kT2909TokOpen = 0;
constexpr int32_t kT2909TokOpenQuote = 1;
constexpr int32_t kT2909TokBackslash = 3;
constexpr int32_t kT2909TokEscapeN = 4;
constexpr int32_t kT2909TokClose = 5;
constexpr int32_t kT2909FillerBase = 7;  // "<unused-N>" pieces: harmless prompt filler.

// Drives a fresh sequence on `g` (the string-schema fixture) to S_e via three real, admitted
// schema-content transitions from a fresh bind. Returns the live handle, or nullptr on setup
// failure (already recorded via CHECK_MSG).
sslm_seq ReachSeCpu(Cpu& g) {
	sslm_seq s = NewSeq(g);
	if (!s) return nullptr;
	const std::vector<int32_t> P = {kT2909FillerBase, kT2909FillerBase + 1, kT2909FillerBase + 2};
	CHECK_MSG(PrefillAll(g, s, P, SSLM_SPAN_PROMPT) == SSLM_OK, "[CPU C2] prompt prefill");
	const std::vector<int32_t> content = {kT2909TokOpen, kT2909TokOpenQuote, kT2909TokBackslash};
	CHECK_MSG(PrefillAll(g, s, content, SSLM_SPAN_SCHEMA_CONTENT) == SSLM_OK,
	          "[CPU C2] schema-content prefill: open+open-quote+backslash to reach S_e");
	return s;
}

void CpuCell2DegenerateRowTwin(Cpu& g) {
	{
		sslm_seq s = ReachSeCpu(g);
		if (s) {
			const uint32_t s_e = BlobDfaWalkState(s);
			ArmCpuFinishDegenerateLogitRowInjection();
			sslm_status st{};
			const int32_t out = Decode1(g, s, &st, -1);
			CHECK_MSG(st == SSLM_OK && out == -2,
			          "[CPU C2] degenerate row at S_e: st=%d out=%d, want -2/SSLM_OK",
			          static_cast<int>(st), out);
			CHECK_MSG(BlobDfaWalkState(s) == s_e,
			          "[CPU C2] degenerate row at S_e: dfa_walk_state moved %u -> %u", s_e,
			          BlobDfaWalkState(s));
			sslm_seq_release(s);
		}
	}
	{
		// Must-accept neighbour: the identical construction, un-armed.
		sslm_seq s = ReachSeCpu(g);
		if (s) {
			const uint32_t s_e = BlobDfaWalkState(s);
			sslm_status st{};
			const int32_t out = Decode1(g, s, &st, -1);
			CHECK_MSG(st == SSLM_OK && out >= 0,
			          "[CPU C2] must-accept neighbour: st=%d out=%d, want a real, non-negative token",
			          static_cast<int>(st), out);
			CHECK_MSG(out == kT2909TokBackslash || out == kT2909TokEscapeN || out == kT2909TokClose,
			          "[CPU C2] must-accept neighbour: produced token %d is not one of S_e's own "
			          "admitted escapes {%d,%d,%d}",
			          out, kT2909TokBackslash, kT2909TokEscapeN, kT2909TokClose);
			const uint32_t moved = BlobDfaWalkState(s);
			CHECK_MSG(moved != s_e, "[CPU C2] must-accept neighbour: dfa_walk_state did not leave S_e (%u)",
			          s_e);
			sslm_seq_release(s);
		}
	}
}
#endif

}  // namespace

int main(int argc, char** argv) {
	std::string model_path;
	std::string string_schema_path;
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a.rfind("--model=", 0) == 0) model_path = a.substr(8);
		else if (a.rfind("--stringschema=", 0) == 0) string_schema_path = a.substr(15);
	}
	if (model_path.empty()) {
		std::printf("SKIP cell_cpu_deadend_retry_reset -- needs --model=PATH\n");
		std::printf("checks=0 failures=0 skips=1\n");
		return 0;
	}
	Cpu g;
	std::string err;
	if (!CpuOpen(&g, model_path, &err)) {
		std::printf("FAIL cell_cpu_deadend_retry_reset -- load failed: %s\n", err.c_str());
		return 1;
	}
	const std::vector<int32_t> P = {0, 1, 2, 3};
	const int32_t t0 = DeriveAcceptedToken(g, P);
	CHECK_MSG(t0 >= 0, "SETUP: no schema-admitted token found from the prompt {0,1,2,3}");
	std::printf("layers=%d vocab=%d cap=%lld t0=%d\n", g.layers, g.vocab, static_cast<long long>(g.cap), t0);
	if (t0 < 0) {
		std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
		return 1;
	}

	// Cell 1(iii) and Cell 1(iv) are checked on INDEPENDENT sequence instances, each freshly
	// driven to the SAME dead end -- (iii)'s own extra decode calls must not be allowed to move
	// the sequence further before (iv) asks whether reset succeeds "after the dead end." Chaining
	// them on one handle would silently test resettability after N erroneous retries instead of
	// after the first miss, which is a different (and, on pristine, differently-answered)
	// question from the one Cell 1(iv) and T-2894/TE-361 actually pose.
	for (const std::string& route : {std::string("R"), std::string("D"), std::string("C")}) {
		DeadEndState d_iii = ReachDeadEnd(g, route, P, t0);
		if (d_iii.seq) {
			CheckRetry(g, route, d_iii.seq, d_iii.ctx_after_first_miss);
			sslm_seq_release(d_iii.seq);
		}
		DeadEndState d_iv = ReachDeadEnd(g, route, P, t0);
		if (d_iv.seq) {
			CheckResettable(g, route, d_iv.seq, P);
			sslm_seq_release(d_iv.seq);
		}
	}
	RouteD1PartialBudget(g, P);
	DampedGreedyRoute(g, P, t0);
#if defined(SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION)
	if (string_schema_path.empty()) {
		std::printf("SKIP cell_cpu_deadend_retry_reset -- Cell 2 twin needs --stringschema=PATH\n");
		++GSkips;
	} else {
		Cpu g2;
		std::string err2;
		if (!CpuOpen(&g2, string_schema_path, &err2)) {
			std::printf("FAIL cell_cpu_deadend_retry_reset -- string-schema fixture load failed: %s\n",
			            err2.c_str());
			++GChecks;
			++GFailures;
		} else {
			CpuCell2DegenerateRowTwin(g2);
		}
	}
#endif
	for (const std::string& route : {std::string("R"), std::string("D"), std::string("C")}) {
		CheckSaveRestoreRoundTrip(g, route, P, t0);
	}

	// S1 (TE-365): a skipped cell fails the run -- an acceptance run supplies every fixture
	// flag, so skips=0 here is a genuine claim, not a summary line nobody consulted.
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return (GFailures == 0 && GSkips == 0) ? 0 : 1;
}
