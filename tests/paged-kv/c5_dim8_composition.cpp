// Paged-KV plan (rev 16.1) §7 dimension 8, the CPU composition pairs (step C5): 8.1-8.12.
//
// Each pair crosses paging (a budget holder that shares a prefix's pages) with one existing feature,
// and grades it the way the suite grades byte-equality for budget holders: tokens and K/V rows
// against a legacy twin (a legacy holder in a legacy pool doing the same calls, which C4 pins to the
// v1.11.0 reference) or against the R0 reference directly, never the whole blob (SSB6 against
// 1.9.0's SSB5 differ by design, §3.7). Page counts are graded by the fill probe (§8).

#include "pkv_budget_b_helpers.h"

#include "superslm/parallel_for.h"

#include <thread>

namespace {

using namespace pkv;
using namespace pkv::budget_b;

// A run's observable result: its tokens and its final save.
struct RunOut {
	std::vector<int32_t> tokens;
	std::vector<uint8_t> blob;
};

// Tokens equal, and the rows [0, L') of the two final saves equal.
void ExpectSameRun(const Fixture& fx, const RunOut& got, const RunOut& want, const char* what) {
	PKV_CHECK_MSG(got.tokens == want.tokens, "%s: tokens differ (%zu vs %zu)", what, got.tokens.size(), want.tokens.size());
	if (got.blob.empty() || want.blob.empty()) {
		PKV_CHECK_MSG(false, "%s: no final save", what);
		return;
	}
	const int64_t L = BlobLp(got.blob);
	PKV_CHECK_MSG(L == BlobLp(want.blob), "%s: L' %lld vs %lld", what, static_cast<long long>(L),
	              static_cast<long long>(BlobLp(want.blob)));
	PKV_CHECK_MSG(Rows(got.blob, fx, L) == Rows(want.blob, fx, L), "%s: K/V rows [0, %lld) differ", what,
	              static_cast<long long>(L));
}

using Drive = std::function<void(sslm_seq, std::vector<int32_t>*)>;

// The legacy twin: a legacy pool of 2 blocks, a legacy prefix of `prefix` (none when empty), a
// legacy sequence that copies it, `drive`, and a final save.
RunOut LegacyTwin(const Fixture& fx, const std::vector<int32_t>& prefix, const Drive& drive) {
	RunOut out;
	LegacyPool pool(fx.model, 2);
	sslm_prefix px = prefix.empty() ? nullptr : MakePrefix(fx, &pool.pool, prefix, 0);
	sslm_seq s = MakeLegacySeq(fx, &pool.pool);
	if (s) {
		if (px) PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
		drive(s, &out.tokens);
		out.blob = Save(s);
		sslm_seq_release(s);
	}
	if (px) sslm_prefix_release(px);
	return out;
}

// The budget run: a page pool, a budget prefix of `prefix` (budget = its length), a budget-b
// sequence that shares it, `drive`, and a final save.
RunOut BudgetRun(const Fixture& fx, const std::vector<int32_t>& prefix, int32_t b, const Drive& drive) {
	RunOut out;
	const int32_t plen = static_cast<int32_t>(prefix.size());
	PagePool pool(fx.model, static_cast<uint32_t>((plen ? fx.R(plen) : 0) + fx.R(b) + 2));
	sslm_prefix px = plen ? MakePrefix(fx, &pool.pool, prefix, plen) : nullptr;
	sslm_seq s = MakeBudgetSeq(fx, &pool.pool, b);
	if (s) {
		if (px) PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
		drive(s, &out.tokens);
		out.blob = Save(s);
		sslm_seq_release(s);
	}
	if (px) sslm_prefix_release(px);
	return out;
}

// One token per sequence per call, the whole layer budget each call.
std::vector<std::vector<int32_t>> BatchDecode(const Fixture& fx, std::vector<sslm_seq> seqs, int rounds) {
	std::vector<std::vector<int32_t>> out(seqs.size());
	sslm_decode_params p{};
	p.layer_budget = static_cast<int32_t>(fx.geo.layers);
	std::vector<int32_t> toks(seqs.size());
	for (int r = 0; r < rounds; ++r) {
		const sslm_status st = sslm_decode_step(fx.model, seqs.data(), static_cast<int32_t>(seqs.size()), &p, nullptr, toks.data());
		PKV_CHECK_MSG(st == SSLM_OK, "batch decode round %d: status %d", r, static_cast<int>(st));
		if (st != SSLM_OK) break;
		for (size_t i = 0; i < seqs.size(); ++i) {
			PKV_CHECK_MSG(toks[i] >= 0, "batch decode round %d: sequence %zu gave %d", r, i, toks[i]);
			out[i].push_back(toks[i]);
		}
	}
	return out;
}

const std::vector<int32_t>& World1000(const Fixture& fx) {
	static const std::vector<int32_t> t = Stream(51, 1000, fx.vocab);
	return t;
}

// ---- 8.1 [C5] paging x batch decode ------------------------------------------------------------------

// A legacy sequence and a budget sequence share one budget prefix (1,000 tokens, mid-page) in one
// page pool, each prefills its own 20-token suffix, and both decode 40 tokens in the same
// sslm_decode_step calls, across page boundaries. Each equals its solo legacy twin.
void Cell81() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	PagePool pool(fx.model, static_cast<uint32_t>(fx.R(1000) + fx.CapPages() + fx.R(512) + 2));
	sslm_prefix px = MakePrefix(fx, &pool.pool, World1000(fx), 1000);
	sslm_seq l = px ? MakeLegacySeq(fx, &pool.pool) : nullptr;
	sslm_seq b = l ? MakeBudgetSeq(fx, &pool.pool, 512) : nullptr;
	if (b) {
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(l, px), SSLM_OK);
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(b, px), SSLM_OK);
		PKV_CHECK_EQ(PrefillAll(fx.model, l, Stream(81, 20, fx.vocab), 64), SSLM_OK);
		PKV_CHECK_EQ(PrefillAll(fx.model, b, Stream(82, 20, fx.vocab), 64), SSLM_OK);
		const std::vector<std::vector<int32_t>> t = BatchDecode(fx, {l, b}, 40);
		const RunOut got_l{t[0], Save(l)}, got_b{t[1], Save(b)};
		for (int i = 0; i < 2; ++i) {
			const RunOut want = LegacyTwin(fx, World1000(fx), [&](sslm_seq s, std::vector<int32_t>* out) {
				PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(81 + static_cast<uint32_t>(i), 20, fx.vocab), 64), SSLM_OK);
				DecodeN(fx, s, 40, out);
			});
			// kills: a batch step that maps or reads through one batch member's table for another, or
			// a shared page written by the batch's legacy (copying) member
			ExpectSameRun(fx, i ? got_b : got_l, want, i ? "8.1 budget member" : "8.1 legacy member");
		}
	}
	if (b) sslm_seq_release(b);
	if (l) sslm_seq_release(l);
	if (px) sslm_prefix_release(px);
}

// ---- 8.2 [C5] x chunk budgets 1, 7 and max ---------------------------------------------------------

// A budget adopter of a 1,000-token prefix (origin mid-page) and a budget holder from origin 0
// prefill 100 tokens in chunks of 1, 7 and the whole count, every chunk path spanning page
// boundaries, then decode 8. Each equals its legacy twin with the same chunk budget.
void Cell82() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	for (int32_t chunk : {1, 7, 100}) {
		const Drive drive = [&](sslm_seq s, std::vector<int32_t>* out) {
			int64_t consumed = 0;
			PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(83, 100, fx.vocab), chunk, &consumed), SSLM_OK);
			PKV_CHECK_EQ(consumed, 100);
			DecodeN(fx, s, 8, out);
		};
		const std::string a = "8.2 adopter, chunk " + std::to_string(chunk), o = "8.2 origin 0, chunk " + std::to_string(chunk);
		// kills: a chunk that maps one page too few at a boundary inside it, or maps from the pool
		ExpectSameRun(fx, BudgetRun(fx, World1000(fx), 128, drive), LegacyTwin(fx, World1000(fx), drive), a.c_str());
		ExpectSameRun(fx, BudgetRun(fx, {}, 128, drive), LegacyTwin(fx, {}, drive), o.c_str());
	}
}

// ---- 8.3 [C5] x schema ---------------------------------------------------------------------------------

// A schema-admitted token from the start state, derived on the fixture (the census's way).
int32_t AdmittedToken(const Fixture& fx, sslm_schema schema) {
	LegacyPool pool(fx.model, 1);
	sslm_seq d = MakeLegacySeq(fx, &pool.pool);
	if (!d) return -1;
	PKV_CHECK_EQ(sslm_seq_set_schema(d, schema), SSLM_OK);
	PKV_CHECK_EQ(PrefillAll(fx.model, d, Stream(72, 20, fx.vocab), 8), SSLM_OK);
	const int32_t t = CensusGreedy(fx, d, 1)[0];
	sslm_seq_release(d);
	PKV_CHECK_MSG(t >= 0, "no schema-admitted token from the start state");
	return t;
}

// A prefix of `n0` prompt tokens, then the admitted token as SCHEMA_CONTENT (so the forced span
// sits at position n0, the last row of a page for n0 = 15 and the first of the next for n0 = 16).
void PrefillSchemaRoot(const Fixture& fx, sslm_prefix p, sslm_schema schema, int32_t n0, int32_t admitted) {
	PKV_CHECK_EQ(sslm_prefix_set_schema(p, schema), SSLM_OK);
	PKV_CHECK_EQ(PrefixPrefillAll(fx.model, p, Stream(84, n0, fx.vocab), 8), SSLM_OK);
	int32_t c = 0;
	PKV_CHECK(sslm_prefix_prefill(fx.model, p, &admitted, 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &c) == SSLM_OK && c == 1);
}

// 8.3: a budget prefix with real schema progress (a forced SCHEMA_CONTENT token at a page
// boundary); a begin_from child of it inherits the progress and prefills 20 prompt tokens across
// the next page; a schema-bound budget sequence shares the child, and its tokens, schema_accepting
// and rows equal a legacy twin (one legacy prefix with the same calls, copied). A sequence without
// the schema is still refused SSLM_PREFIX_SCHEMA_MISMATCH, unchanged, and loses its bind
// permission as today (§3.9 A18).
void Cell83() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	sslm_schema schema = nullptr;
	PKV_CHECK_EQ(sslm_schema_lookup(fx.model, "g5_minimal_one_field", &schema), SSLM_OK);
	if (!schema) return;
	const int32_t admitted = AdmittedToken(fx, schema);
	if (admitted < 0) return;
	for (int32_t n0 : {15, 16}) {
		// The budget side.
		PagePool pool(fx.model, 64);
		sslm_prefix root = nullptr, child = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin_budgeted(fx.model, &pool.pool, 32, &root), SSLM_OK);
		if (!root) continue;
		PrefillSchemaRoot(fx, root, schema, n0, admitted);
		PKV_CHECK_EQ(sslm_prefix_freeze(root), SSLM_OK);
		child = MakeChild(fx, root, Stream(85, 20, fx.vocab), 32);
		sslm_seq s = child ? MakeBudgetSeq(fx, &pool.pool, 32) : nullptr;
		RunOut got;
		sslm_stats_out gs{};
		if (s) {
			PKV_CHECK_EQ(sslm_seq_set_schema(s, schema), SSLM_OK);
			PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, child), SSLM_OK);  // kills: begin_from dropping the walk
			PKV_CHECK_EQ(sslm_stats(fx.model, s, &gs), SSLM_OK);
			got.tokens = CensusGreedy(fx, s, 6);
			got.blob = Save(s);
		}
		// The legacy twin: one legacy prefix with the same calls in the same chunks.
		LegacyPool lpool(fx.model, 2);
		sslm_prefix lp = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &lpool.pool, &lp), SSLM_OK);
		RunOut want;
		sslm_stats_out ws{};
		if (lp) {
			PrefillSchemaRoot(fx, lp, schema, n0, admitted);
			PKV_CHECK_EQ(PrefixPrefillAll(fx.model, lp, Stream(85, 20, fx.vocab), 64), SSLM_OK);
			PKV_CHECK_EQ(sslm_prefix_freeze(lp), SSLM_OK);
			sslm_seq t = MakeLegacySeq(fx, &lpool.pool);
			if (t) {
				PKV_CHECK_EQ(sslm_seq_set_schema(t, schema), SSLM_OK);
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(t, lp), SSLM_OK);
				PKV_CHECK_EQ(sslm_stats(fx.model, t, &ws), SSLM_OK);
				want.tokens = CensusGreedy(fx, t, 6);
				want.blob = Save(t);
				sslm_seq_release(t);
			}
			sslm_prefix_release(lp);
		}
		const std::string what = "8.3 forced token at " + std::to_string(n0);
		PKV_CHECK_MSG(gs.schema_accepting == ws.schema_accepting, "%s: schema_accepting %d vs %d", what.c_str(),
		              gs.schema_accepting, ws.schema_accepting);
		ExpectSameRun(fx, got, want, what.c_str());
		// A mismatch still rejects, before any state change. The sequence is fresh, so it is still
		// bindable before the adopt, and the refusal's revocation of that permission is observable.
		sslm_seq u = MakeBudgetSeq(fx, &pool.pool, 32);
		if (u && child) {
			const std::vector<uint8_t> before = Save(u);
			// kills: share adopt mapping the prefix's pages before the schema check
			PKV_CHECK_EQ(sslm_seq_adopt_prefix(u, child), SSLM_PREFIX_SCHEMA_MISMATCH);
			PKV_CHECK_MSG(Save(u) == before, "%s: a refused adopt changed the sequence", what.c_str());
			PKV_CHECK_EQ(sslm_seq_set_schema(u, schema), SSLM_SCHEMA_BIND_REJECTED);
		}
		if (u) sslm_seq_release(u);
		if (s) sslm_seq_release(s);
		if (child) sslm_prefix_release(child);
		sslm_prefix_release(root);
	}
}

// ---- 8.4 [C5] x adapter ----------------------------------------------------------------------------

// A budget adopter with an adapter bound equals legacy copy adopt with the same adapter. The
// adapter moves layer 1's V rows, so the rows written after the origin differ from an unadapted
// run's: the cell first checks the adapter changes the tokens at all.
void Cell84() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	Adapter ad(fx);
	if (!ad.a) return;
	auto drive = [&](bool bind) {
		return [&, bind](sslm_seq s, std::vector<int32_t>* out) {
			if (bind) PKV_CHECK_EQ(sslm_seq_set_adapter(s, ad.a), SSLM_OK);
			DecodeN(fx, s, 40, out);
			PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(86, 20, fx.vocab), 7), SSLM_OK);
			DecodeN(fx, s, 8, out);
		};
	};
	const RunOut got = BudgetRun(fx, World1000(fx), 128, drive(true));
	PKV_CHECK_MSG(got.tokens != BudgetRun(fx, World1000(fx), 128, drive(false)).tokens,
	              "construction: the adapter does not change the tokens, so the pair is vacuous");
	// kills: an adapter's delta applied to (or skipped on) the shared prefix span, or a shared page
	// written under the adapter
	ExpectSameRun(fx, got, LegacyTwin(fx, World1000(fx), drive(true)), "8.4 adapter");
}

// ---- 8.5 [C5] x damped-greedy mode ------------------------------------------------------------------

std::vector<int32_t> DampedN(const Fixture& fx, sslm_seq s, int n) {
	sslm_decode_params p{};
	PKV_CHECK_EQ(sslm_decode_params_init(fx.model, SSLM_DECODE_MODE_DAMPED_GREEDY, static_cast<int32_t>(fx.geo.layers), &p),
	             SSLM_OK);
	sslm_seq b[1] = {s};
	std::vector<int32_t> t;
	for (int i = 0; i < n; ++i) {
		int32_t tok = -1;
		const sslm_status st = sslm_decode_step_v2(fx.model, b, 1, &p, nullptr, &tok);
		PKV_CHECK_MSG(st == SSLM_OK && tok >= 0, "damped step %d: status %d token %d", i, static_cast<int>(st), tok);
		if (st != SSLM_OK) break;
		t.push_back(tok);
	}
	return t;
}

// A budget adopter decodes 40 damped-greedy tokens across page boundaries, is saved and restored
// (sharing the prefix), and decodes 10 more; its legacy twin does the same with a legacy restore.
// The anti-LM history travels in both blobs.
void Cell85() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	RunOut got, want;
	{
		PagePool pool(fx.model, static_cast<uint32_t>(fx.R(1000) + 2 * fx.R(128)));
		sslm_prefix px = MakePrefix(fx, &pool.pool, World1000(fx), 1000);
		sslm_seq s = px ? MakeBudgetSeq(fx, &pool.pool, 128) : nullptr;
		if (s) {
			PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
			got.tokens = DampedN(fx, s, 40);
			const std::vector<uint8_t> blob = Save(s);
			sslm_seq_release(s);
			sslm_seq r = nullptr;
			PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), px, &r, nullptr), SSLM_OK);
			if (r) {
				got.tokens = Concat(got.tokens, DampedN(fx, r, 10));
				got.blob = Save(r);
				sslm_seq_release(r);
			}
		}
		if (px) sslm_prefix_release(px);
	}
	{
		LegacyPool pool(fx.model, 3);
		sslm_prefix px = MakePrefix(fx, &pool.pool, World1000(fx), 0);
		sslm_seq s = px ? MakeLegacySeq(fx, &pool.pool) : nullptr;
		if (s) {
			PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
			want.tokens = DampedN(fx, s, 40);
			const std::vector<uint8_t> blob = Save(s);
			sslm_seq_release(s);
			sslm_seq r = nullptr;
			PKV_CHECK_EQ(sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &r), SSLM_OK);
			if (r) {
				want.tokens = Concat(want.tokens, DampedN(fx, r, 10));
				want.blob = Save(r);
				sslm_seq_release(r);
			}
		}
		if (px) sslm_prefix_release(px);
	}
	// kills: an SSB6 writer or reader that drops or reorders the anti-LM history, or damped decode
	// reading a row past the live width through a page
	ExpectSameRun(fx, got, want, "8.5 damped greedy");
}

// ---- 8.6 [C5] x parallel-for hook --------------------------------------------------------------------

// A host hook that runs odd task indices on a second thread and even ones on the calling thread,
// and returns only when both are done (parallel_for.h's contract).
struct TwoThreadHook {
	static void Run(void*, int32_t task_count, sslm_task_fn task, void* task_ctx) {
		std::thread other([&] {
			for (int32_t i = 1; i < task_count; i += 2) task(task_ctx, i);
		});
		for (int32_t i = 0; i < task_count; i += 2) task(task_ctx, i);
		other.join();
	}
};

// A budget adopter decoding through a workspace with a two-thread hook (logits and, with
// SSLM_PARALLEL_FOR_MATVEC, every one-row projection, including one-token prefill calls) equals its
// legacy twin with no hook and no workspace.
void Cell86() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const sslm_config cfg{1, 64, static_cast<int32_t>(fx.geo.layers), 0u};
	AlignedBuf mem(sslm_workspace_size(fx.model, &cfg));
	sslm_workspace ws = nullptr;
	PKV_CHECK_EQ(sslm_workspace_create(fx.model, &cfg, mem.p, mem.n, &ws), SSLM_OK);
	if (!ws) return;
	const sslm_parallel_for pf{&TwoThreadHook::Run, nullptr, 4, SSLM_PARALLEL_FOR_MATVEC};
	PKV_CHECK_EQ(sslm_workspace_set_parallel_for(ws, &pf), SSLM_OK);
	const std::vector<int32_t> extra = Stream(87, 5, fx.vocab);
	const Drive hooked = [&](sslm_seq s, std::vector<int32_t>* out) {
		sslm_decode_params p{};
		p.layer_budget = static_cast<int32_t>(fx.geo.layers);
		sslm_seq b[1] = {s};
		for (int i = 0; i < 40; ++i) {
			int32_t tok = -1;
			PKV_CHECK_EQ(sslm_decode_step(fx.model, b, 1, &p, ws, &tok), SSLM_OK);
			out->push_back(tok);
		}
		for (int32_t t : extra) {
			int32_t c = 0;
			PKV_CHECK_EQ(sslm_prefill(fx.model, s, &t, 1, 1, SSLM_SPAN_PROMPT, ws, &c), SSLM_OK);
		}
		for (int i = 0; i < 8; ++i) {
			int32_t tok = -1;
			PKV_CHECK_EQ(sslm_decode_step(fx.model, b, 1, &p, ws, &tok), SSLM_OK);
			out->push_back(tok);
		}
	};
	const Drive plain = [&](sslm_seq s, std::vector<int32_t>* out) {
		DecodeN(fx, s, 40, out);
		for (int32_t t : extra) PKV_CHECK_EQ(PrefillAll(fx.model, s, {t}, 1), SSLM_OK);
		DecodeN(fx, s, 8, out);
	};
	// kills: a page view built on the calling thread's stack that a task reads after it moved, or a
	// hooked projection reading the flat workspace instead of the page view
	ExpectSameRun(fx, BudgetRun(fx, World1000(fx), 128, hooked), LegacyTwin(fx, World1000(fx), plain), "8.6 hook");
	sslm_workspace_destroy(ws);
}

// ---- 8.7 [C5] x layer-budget micro-steps --------------------------------------------------------------

// Decodes `n` tokens one layer per call, so each token whose row opens a new page is suspended
// after layer 0 (the page mapped, layer 1 not yet written) and resumed by the next call.
std::vector<int32_t> MicroSteps(const Fixture& fx, sslm_seq s, int n, int* suspended) {
	sslm_decode_params p{};
	p.layer_budget = 1;
	sslm_seq b[1] = {s};
	std::vector<int32_t> t;
	for (int guard = 0; static_cast<int>(t.size()) < n && guard < 16 * n; ++guard) {
		int32_t tok = -1;
		const sslm_status st = sslm_decode_step(fx.model, b, 1, &p, nullptr, &tok);
		PKV_CHECK_MSG(st == SSLM_OK, "micro-step: status %d", static_cast<int>(st));
		if (st != SSLM_OK) break;
		if (tok >= 0) t.push_back(tok);
		else ++*suspended;
	}
	return t;
}

// A token at a page-opening position (16k) suspended mid-layers and resumed: budget holders whose
// next row is 16 (after a 16-token prefix), 1,008 (a 1,008-token prefix) and 1,024 (origin 0, 1,024
// prefilled) on pkv_def, and 16,384 on pkv_32k (the 1.5B cap's midpoint). Each decodes its tokens
// one layer per call and equals a legacy twin decoding whole tokens.
void Cell87() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	struct Case {
		int32_t prefix, prompt;
	};
	for (const Case& c : {Case{16, 0}, Case{1008, 0}, Case{0, 1024}}) {
		const std::vector<int32_t> px = c.prefix ? Stream(51, c.prefix, fx.vocab) : std::vector<int32_t>{};
		int suspended = 0;
		const Drive micro = [&](sslm_seq s, std::vector<int32_t>* out) {
			if (c.prompt) PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(88, c.prompt, fx.vocab), 64), SSLM_OK);
			*out = MicroSteps(fx, s, 20, &suspended);
		};
		const Drive whole = [&](sslm_seq s, std::vector<int32_t>* out) {
			if (c.prompt) PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(88, c.prompt, fx.vocab), 64), SSLM_OK);
			DecodeN(fx, s, 20, out);
		};
		const RunOut got = BudgetRun(fx, px, c.prompt + 64, micro);
		PKV_CHECK_MSG(suspended >= 19, "construction: only %d suspensions", suspended);
		const std::string what = "8.7 next row at " + std::to_string(c.prefix + c.prompt);
		// kills: a page mapped at layer 0 and mapped again (or not found) when layer 1 resumes
		ExpectSameRun(fx, got, LegacyTwin(fx, px, whole), what.c_str());
	}
	const Fixture& big = GetFixture("pkv_32k");
	if (!big.ok) return;
	const int32_t at = 16384;
	const std::vector<int32_t> prompt = Stream(89, at, big.vocab);
	RunOut got, want;
	{
		PagePool pool(big.model, static_cast<uint32_t>(big.R(at + 32)));
		sslm_seq s = MakeBudgetSeq(big, &pool.pool, at + 32);
		if (s) {
			PKV_CHECK_EQ(PrefillAll(big.model, s, prompt, 512), SSLM_OK);
			int suspended = 0;
			got.tokens = MicroSteps(big, s, 6, &suspended);
			got.blob = Save(s);
			sslm_seq_release(s);
		}
	}
	{
		LegacyPool pool(big.model, 1);
		sslm_seq s = MakeLegacySeq(big, &pool.pool);
		if (s) {
			PKV_CHECK_EQ(PrefillAll(big.model, s, prompt, 512), SSLM_OK);
			DecodeN(big, s, 6, &want.tokens);
			want.blob = Save(s);
			sslm_seq_release(s);
		}
	}
	ExpectSameRun(big, got, want, "8.7 next row at 16,384 on the 32k cap");
}

// ---- 8.8 [C5] x save/restore mid-token after a nested adopt ----------------------------------------

// A legacy prefix holding the chain's 1,200 tokens, prefilled in the chain's own chunks (the world's
// 1,000 in 64s, then the persona's 200 in 64s from 1,000), so its rows are the chain's by
// construction.
sslm_prefix LegacyChainPrefix(const Fixture& fx, sslm_kv_pool* pool) {
	sslm_prefix p = nullptr;
	PKV_CHECK_EQ(sslm_prefix_begin(fx.model, pool, &p), SSLM_OK);
	if (!p) return nullptr;
	PKV_CHECK_EQ(PrefixPrefillAll(fx.model, p, WorldTokens(fx.vocab), 64), SSLM_OK);
	PKV_CHECK_EQ(PrefixPrefillAll(fx.model, p, PersonaTokens(fx.vocab), 64), SSLM_OK);
	PKV_CHECK_EQ(sslm_prefix_freeze(p), SSLM_OK);
	return p;
}

// world -> persona (begin_from) -> a budget-512 sequence that decodes 30 tokens and is saved
// mid-token. Restored with the persona handle and without, each twin re-saves byte-equal to the
// blob and finishes with the unsaved original's tokens; the blob's rows equal a legacy twin's
// mid-token save (the zero substitution on both writers).
void Cell88() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	PagePool pool(fx.model, 600);
	Handles h;
	const Chain c = BuildChain(fx, &pool.pool, &h);
	sslm_seq s = c.persona ? MakeBudgetSeq(fx, &pool.pool, 512) : nullptr;
	if (!s) return;
	h.seqs.push_back(s);
	PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, c.persona), SSLM_OK);
	std::vector<int32_t> t;
	DecodeN(fx, s, 30, &t);
	EnterMidToken(fx, s);
	const std::vector<uint8_t> blob = Save(s);
	std::vector<int32_t> finish;
	DecodeN(fx, s, 11, &finish);
	for (int with_handle = 0; with_handle <= 1; ++with_handle) {
		sslm_seq r = nullptr;
		PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(),
		                                     with_handle ? c.persona : nullptr, &r, nullptr),
		             SSLM_OK);
		if (!r) continue;
		h.seqs.push_back(r);
		// kills: a restore that drops layer_index or rescatters the mid-token row
		PKV_CHECK_MSG(Save(r) == blob, "8.8 (%s handle): the restored holder re-saves differently", with_handle ? "with" : "without");
		std::vector<int32_t> got;
		DecodeN(fx, r, 11, &got);
		PKV_CHECK_MSG(got == finish, "8.8 (%s handle): tokens after restore differ", with_handle ? "with" : "without");
	}
	LegacyPool lpool(fx.model, 2);
	sslm_prefix lp = LegacyChainPrefix(fx, &lpool.pool);
	sslm_seq l = lp ? MakeLegacySeq(fx, &lpool.pool) : nullptr;
	if (l) {
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(l, lp), SSLM_OK);
		std::vector<int32_t> lt;
		DecodeN(fx, l, 30, &lt);
		PKV_CHECK(lt == t);
		EnterMidToken(fx, l);
		ExpectSameRun(fx, RunOut{t, blob}, RunOut{lt, Save(l)}, "8.8 mid-token rows against the legacy twin");
		sslm_seq_release(l);
	}
	if (lp) sslm_prefix_release(lp);
}

// ---- 8.9 [C5] x mixed pool ---------------------------------------------------------------------------

// Legacy and budget holders in one page pool: a legacy holder with its own prompt, a budget holder
// sharing a budget prefix, and a legacy holder copying the same budget prefix (§3.9 A20: the
// adopter's mode decides). They decode round-robin; each equals its legacy twin; after a full
// teardown the fill probe admits the whole pool.
void Cell89() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const uint32_t P = static_cast<uint32_t>(fx.R(1000) + 2 * fx.CapPages() + fx.R(128) + 2);
	struct Mixed : PageState {
		sslm_seq l1 = nullptr, b1 = nullptr, l2 = nullptr;
		Mixed(const Fixture& fx, uint32_t pages, bool decode, std::vector<RunOut>* outs) : PageState(fx.model, pages) {
			sslm_prefix px = MakePrefix(fx, Pool(), World1000(fx), 1000);
			l1 = MakeLegacySeq(fx, Pool());
			b1 = px ? MakeBudgetSeq(fx, Pool(), 128) : nullptr;
			l2 = b1 ? MakeLegacySeq(fx, Pool()) : nullptr;
			if (l1 && b1 && l2) {
				PKV_CHECK_EQ(PrefillAll(fx.model, l1, Stream(90, 100, fx.vocab), 64), SSLM_OK);
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(b1, px), SSLM_OK);
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(l2, px), SSLM_OK);
				if (decode) {
					outs->assign(3, RunOut{});
					for (int i = 0; i < 20; ++i) {
						DecodeN(fx, l1, 1, &(*outs)[0].tokens);
						DecodeN(fx, b1, 1, &(*outs)[1].tokens);
						DecodeN(fx, l2, 1, &(*outs)[2].tokens);
					}
					(*outs)[0].blob = Save(l1);
					(*outs)[1].blob = Save(b1);
					(*outs)[2].blob = Save(l2);
				}
			}
			for (sslm_seq s : {l1, b1, l2})
				if (s) sslm_seq_release(s);
			if (px) sslm_prefix_release(px);
		}
	};
	std::vector<RunOut> outs;
	{ Mixed m(fx, P, true, &outs); }
	if (outs.size() == 3) {
		const Drive d20 = [&](sslm_seq s, std::vector<int32_t>* out) { DecodeN(fx, s, 20, out); };
		ExpectSameRun(fx, outs[0], LegacyTwin(fx, {}, [&](sslm_seq s, std::vector<int32_t>* out) {
			              PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(90, 100, fx.vocab), 64), SSLM_OK);
			              DecodeN(fx, s, 20, out);
		              }),
		              "8.9 legacy holder");
		const RunOut twin = LegacyTwin(fx, World1000(fx), d20);
		ExpectSameRun(fx, outs[1], twin, "8.9 budget sharer");
		ExpectSameRun(fx, outs[2], twin, "8.9 legacy copier of a budget prefix");
	}
	// kills: a mixed-mode release path that leaks a page (a copied page counted as shared, or the
	// budget prefix's pages kept by its legacy copier)
	ProbeExactlyFree(fx, [&] { return std::unique_ptr<ProbeState>(new Mixed(fx, P, false, nullptr)); }, P);
}

// ---- 8.10 [C5] x cross-pool (2.8) --------------------------------------------------------------------

// A budget sequence in pool A adopts a frozen budget prefix in pool B: SSLM_INVALID_ARGUMENT, the
// sequence unchanged (its re-saved blob equals the one saved before, and it continues as a twin that
// never tried). A legacy sequence in A adopts the same prefix by copy, and its records equal the
// reference's prefix_lengths p1000 stages. After the prefix's release the fill probe on B admits
// all of B.
void Cell810() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const uint32_t PA = static_cast<uint32_t>(fx.R(128) + fx.CapPages()), PB = static_cast<uint32_t>(fx.R(1000) + 4);
	struct TwoPools : ProbeState {
		std::unique_ptr<PagePool> a, b;
		sslm_seq s = nullptr, l = nullptr;
		sslm_prefix px = nullptr;
		TwoPools(const Fixture& fx, uint32_t pa, uint32_t pb)
		    : a(std::make_unique<PagePool>(fx.model, pa)), b(std::make_unique<PagePool>(fx.model, pb)) {
			px = MakePrefix(fx, &b->pool, World1000(fx), 1000);
			s = MakeBudgetSeq(fx, &a->pool, 128);
			l = MakeLegacySeq(fx, &a->pool);
		}
		~TwoPools() override {
			if (s) sslm_seq_release(s);
			if (l) sslm_seq_release(l);
			if (px) sslm_prefix_release(px);
		}
		sslm_kv_pool* Pool() override { return &b->pool; }
	};
	const Drive used = [&](sslm_seq q, std::vector<int32_t>* out) {
		PKV_CHECK_EQ(PrefillAll(fx.model, q, Stream(91, 40, fx.vocab), 16), SSLM_OK);
		DecodeN(fx, q, 4, out);
	};
	{
		TwoPools st(fx, PA, PB);
		if (!st.s || !st.l || !st.px) return;
		std::vector<int32_t> t;
		used(st.s, &t);
		const std::vector<uint8_t> before = Save(st.s);
		// kills: share adopt across pools (it returns SSLM_OK)
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(st.s, st.px), SSLM_INVALID_ARGUMENT);
		PKV_CHECK_MSG(Save(st.s) == before, "8.10: the refused budget sequence changed");
		DecodeN(fx, st.s, 8, &t);
		const RunOut twin = BudgetRun(fx, {}, 128, [&](sslm_seq q, std::vector<int32_t>* out) {
			used(q, out);
			DecodeN(fx, q, 8, out);
		});
		PKV_CHECK_MSG(t == twin.tokens, "8.10: the refused budget sequence continues differently");
		// The legacy sequence copies across pools, as today; its records are the reference's.
		std::vector<Record> recs;
		std::vector<int32_t> pending;
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(st.l, st.px), SSLM_OK);
		recs.push_back(MakeRecord(fx, "p1000_adopted", st.l, &pending));
		DecodeN(fx, st.l, 8, &pending);
		recs.push_back(MakeRecord(fx, "p1000_decode8", st.l, &pending));
		PKV_CHECK_EQ(PrefillAll(fx.model, st.l, Stream(52, 20, fx.vocab), 64), SSLM_OK);
		DecodeN(fx, st.l, 4, &pending);
		recs.push_back(MakeRecord(fx, "p1000_prefill20+decode4", st.l, &pending));
		ExpectStagesMatch("v1.11.0", fx, "prefix_lengths", recs);
	}
	// kills: a cross-pool attempt that took a reference on B's pages and kept it past the release
	ProbeExactlyFree(fx, [&] {
		auto st = std::make_unique<TwoPools>(fx, PA, PB);
		if (st->s && st->l && st->px) {
			std::vector<int32_t> t;
			used(st->s, &t);
			sslm_seq_adopt_prefix(st->s, st->px);
			sslm_seq_adopt_prefix(st->l, st->px);
			sslm_prefix_release(st->px);
			st->px = nullptr;
		}
		return std::unique_ptr<ProbeState>(std::move(st));
	}, PB);
}

// ---- 8.11 [C5] x used-state adopt census in budget mode (1.10) ----------------------------------------

// The 60-row census of pkv_budget_b_helpers.h (1.10's construction): every row's adopt onto a used
// budget holder equals a fresh budget holder's, the pool's free pages are exact by the fill probe,
// and the adopter's pages read 0xCD after its release.
void Cell811() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const CensusResult r = RunBudgetCensus(fx);
	PKV_CHECK_MSG(r.mismatches == 0, "8.11: %d of %d census rows differ from a fresh budget holder", r.mismatches, r.rows);
}

// ---- 8.12 [C5] sslm_seq_restore_shared x ... ----------------------------------------------------------

// Pages the chain of world length wl and persona length pl holds with the world live, by §3.4:
// the world keeps ceil(wl/B); the persona maps ceil((wl+pl)/B) entries, floor(wl/B) of them shared.
int64_t ChainPages(const Fixture& fx, int64_t wl, int64_t pl) {
	const int64_t B = fx.B();
	return (wl + B - 1) / B + ((wl + pl + B - 1) / B - wl / B);
}

// restore_shared x a mid-token blob; x a schema-bound prefix (the walk state comes from the blob while
// the pages are shared); x an adapter bound on the restored holder; x a mixed pool. Sharing is graded
// by admission: each destination pool leaves exactly R(512) pages free for the restore, so a private
// restore (R + E) is refused there.
void Cell812() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const int64_t R512 = fx.R(512);
	// (a) x a mid-token blob, and (d) x a mixed pool (a legacy holder decoding in the same pool).
	for (int mixed = 0; mixed <= 1; ++mixed) {
		const char* what = mixed ? "8.12 mixed pool" : "8.12 mid-token blob";
		const uint32_t P = static_cast<uint32_t>(ChainPages(fx, kWorldLen, kPersonaLen) + 2 * R512 + (mixed ? fx.CapPages() : 0));
		PKV_CHECK_EQ(ChainPages(fx, kWorldLen, kPersonaLen), kChainPagesWorldLive);
		PagePool pool(fx.model, P);
		Handles h;
		sslm_seq l = mixed ? MakeLegacySeq(fx, &pool.pool) : nullptr;
		if (l) {
			h.seqs.push_back(l);
			PKV_CHECK_EQ(PrefillAll(fx.model, l, Stream(92, 60, fx.vocab), 64), SSLM_OK);
		}
		const Chain c = BuildChain(fx, &pool.pool, &h);
		sslm_seq s = c.persona ? MakeBudgetSeq(fx, &pool.pool, 512) : nullptr;
		if (!s) continue;
		h.seqs.push_back(s);
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, c.persona), SSLM_OK);
		std::vector<int32_t> t;
		DecodeN(fx, s, 20, &t);
		EnterMidToken(fx, s);
		const std::vector<uint8_t> blob = Save(s);
		sslm_seq r = nullptr;
		// kills: restore_shared refusing a mid-token blob's share, or drawing E when sharing
		PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), c.persona, &r, nullptr), SSLM_OK);
		if (!r) continue;
		h.seqs.push_back(r);
		std::vector<int32_t> a, b;
		DecodeN(fx, s, 10, &a);
		DecodeN(fx, r, 10, &b);
		PKV_CHECK_MSG(a == b, "%s: the restored holder's tokens differ from the unsaved original's", what);
		if (l) {
			std::vector<int32_t> lt;
			DecodeN(fx, l, 10, &lt);
			const RunOut twin = LegacyTwin(fx, {}, [&](sslm_seq q, std::vector<int32_t>* out) {
				PKV_CHECK_EQ(PrefillAll(fx.model, q, Stream(92, 60, fx.vocab), 64), SSLM_OK);
				DecodeN(fx, q, 10, out);
			});
			PKV_CHECK_MSG(lt == twin.tokens, "%s: the legacy holder's tokens differ from its twin's", what);
		}
	}
	// (b) x a schema-bound prefix: the world is bound and carries a forced token at 1,000, so the chain
	// is 1,001 + 200 and the adopter's origin, 1,201, is mid-page.
	{
		sslm_schema schema = nullptr;
		PKV_CHECK_EQ(sslm_schema_lookup(fx.model, "g5_minimal_one_field", &schema), SSLM_OK);
		const int32_t admitted = schema ? AdmittedToken(fx, schema) : -1;
		if (admitted >= 0) {
			const uint32_t P = static_cast<uint32_t>(ChainPages(fx, kWorldLen + 1, kPersonaLen) + 2 * R512);
			PagePool pool(fx.model, P);
			sslm_prefix world = nullptr;
			PKV_CHECK_EQ(sslm_prefix_begin_budgeted(fx.model, &pool.pool, kWorldLen + 1, &world), SSLM_OK);
			if (world) {
				PKV_CHECK_EQ(sslm_prefix_set_schema(world, schema), SSLM_OK);
				PKV_CHECK_EQ(PrefixPrefillAll(fx.model, world, WorldTokens(fx.vocab), 64), SSLM_OK);
				int32_t cc = 0;
				PKV_CHECK(sslm_prefix_prefill(fx.model, world, &admitted, 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &cc) ==
				              SSLM_OK &&
				          cc == 1);
				PKV_CHECK_EQ(sslm_prefix_freeze(world), SSLM_OK);
				sslm_prefix persona = MakeChild(fx, world, PersonaTokens(fx.vocab), kPersonaLen);
				sslm_seq s = persona ? MakeBudgetSeq(fx, &pool.pool, 512) : nullptr;
				if (s) {
					PKV_CHECK_EQ(sslm_seq_set_schema(s, schema), SSLM_OK);
					PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, persona), SSLM_OK);
					CensusGreedy(fx, s, 3);
					const std::vector<uint8_t> blob = Save(s);
					sslm_seq r = nullptr;
					PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), persona, &r, nullptr),
					             SSLM_OK);
					if (r) {
						sslm_stats_out so{}, ro{};
						PKV_CHECK_EQ(sslm_stats(fx.model, s, &so), SSLM_OK);
						PKV_CHECK_EQ(sslm_stats(fx.model, r, &ro), SSLM_OK);
						// kills: a shared restore taking the walk state from the handle instead of the blob
						PKV_CHECK_EQ(ro.schema_accepting, so.schema_accepting);
						PKV_CHECK_MSG(CensusGreedy(fx, r, 4) == CensusGreedy(fx, s, 4),
						              "8.12 schema-bound: next tokens differ after the shared restore");
						sslm_seq_release(r);
					}
					sslm_seq_release(s);
				}
				if (persona) sslm_prefix_release(persona);
				sslm_prefix_release(world);
			}
		}
	}
	// (c) x an adapter bound on the restored holder (bound after the restore, between tokens; a
	// mid-token bind is refused as today), against the unsaved original with the same adapter bound
	// at the same point.
	{
		Adapter ad(fx);
		const uint32_t P = static_cast<uint32_t>(kChainPagesWorldLive + 2 * R512);
		PagePool pool(fx.model, P);
		Handles h;
		const Chain c = BuildChain(fx, &pool.pool, &h);
		sslm_seq s = (ad.a && c.persona) ? MakeBudgetSeq(fx, &pool.pool, 512) : nullptr;
		if (s) {
			h.seqs.push_back(s);
			PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, c.persona), SSLM_OK);
			std::vector<int32_t> t;
			DecodeN(fx, s, 20, &t);
			const std::vector<uint8_t> blob = Save(s);
			sslm_seq r = nullptr;
			PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), c.persona, &r, nullptr), SSLM_OK);
			if (r) {
				h.seqs.push_back(r);
				PKV_CHECK_EQ(sslm_seq_set_adapter(s, ad.a), SSLM_OK);
				PKV_CHECK_EQ(sslm_seq_set_adapter(r, ad.a), SSLM_OK);
				std::vector<int32_t> a, b;
				DecodeN(fx, s, 12, &a);
				DecodeN(fx, r, 12, &b);
				// kills: an adapter applied to rows the restored holder shares (they are base-only)
				PKV_CHECK_MSG(a == b, "8.12 adapter: the restored holder's tokens differ from the original's");
				ExpectSameRun(fx, RunOut{a, Save(s)}, RunOut{b, Save(r)}, "8.12 adapter rows");
			}
		}
	}
}

PKV_CELL("8.1", "C5", Cell81);
PKV_CELL("8.2", "C5", Cell82);
PKV_CELL("8.3", "C5", Cell83);
PKV_CELL("8.4", "C5", Cell84);
PKV_CELL("8.5", "C5", Cell85);
PKV_CELL("8.6", "C5", Cell86);
PKV_CELL("8.7", "C5", Cell87);
PKV_CELL("8.8", "C5", Cell88);
PKV_CELL("8.9", "C5", Cell89);
PKV_CELL("8.10", "C5", Cell810);
PKV_CELL("8.11", "C5", Cell811);
PKV_CELL("8.12", "C5", Cell812);

}  // namespace
