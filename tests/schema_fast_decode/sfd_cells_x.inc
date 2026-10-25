// Schema fast-decode plan rev 1.3.3, slice J1-X (§5.10 exact jump-forward; §11.4's engine cells). Included by
// tests/test_schema_fast_decode.cpp inside its anonymous namespace, after the TF cells, so it shares the
// suite's rig, blob reader, reference tables, hooks and seams. The P1-X (plugin) and GX (GPU) cells are later
// slices and are not here.
//
// Red first: until J1-X lands, the 9-argument verb is sfd_red_stubs_x.cpp's no-behaviour overload, which
// returns SSLM_STATUS_NEXT_FREE. Every cell starts with XLive(), which makes one exact call and fails the cell
// once ("exact mode unimplemented") when it sees that sentinel, so each cell is red for that reason and for no
// other. With J1-X built (SUPERSLM_SFD_X_RED_STUBS OFF), XLive passes and the cell runs.
//
// The oracle (§11.4) is jump-forward off: the same build, prompt, schema, holder and decode parameters, with
// the verb never called. "Equals off" is checked after every decode_step, at the same output position: the
// status and the produced token, the SSB blob bytes (walk state, forced_token_count, the five saturation
// counters and the anti-LM state are all in it; SSB5 zero-fills past context_length, so C12's excluded K/V
// bytes cannot differ), sslm_seq_kv_stats, and the holder's table entries. A run compared from a restored
// blob compares the token and the blob only (a restore draws its own pages).

constexpr sslm_status kXStub = SSLM_STATUS_NEXT_FREE;

sslm_jump_options XOpts(int32_t mode, const int32_t* draft, int32_t count,
                        uint32_t size = static_cast<uint32_t>(sizeof(sslm_jump_options))) {
	sslm_jump_options o{};
	o.struct_size = size;
	o.mode = mode;
	o.draft = draft;
	o.draft_count = count;
	return o;
}
// Exact mode with `d` as the draft (NULL: the canonical chain).
sslm_jump_options XExact(const std::vector<int32_t>* d) {
	if (!d) return XOpts(SSLM_JUMP_EXACT, nullptr, 0);
	static const int32_t kNone = -7;
	return XOpts(SSLM_JUMP_EXACT, d->empty() ? &kNone : d->data(), static_cast<int32_t>(d->size()));
}

Jump XJump(sslm_model m, sslm_seq s, int32_t budget, const sslm_decode_params* params, const sslm_jump_options* opts,
           sslm_workspace w, int32_t cap = -1) {
	Jump j;
	const int32_t c = cap < 0 ? std::max(budget, 1) : cap;
	std::vector<int32_t> buf(size_t(c > 0 ? c : 1), -7);
	j.st = sfd::JumpWithOptions(m, s, budget, params, opts, w, buf.data(), c, &j.count);
	if (j.count > 0 && j.count <= c) j.forced.assign(buf.begin(), buf.begin() + j.count);
	return j;
}

// The red-first gate: one exact call on a fresh POTION sequence with a pending token.
bool XLive(const char* cell) {
	static int live = -1;
	if (live < 0) {
		live = 0;
		const Sfd& f = GetSfd();
		if (f.ok) {
			Rig rig(f.m1->m);
			sslm_seq s = rig.NewSeq(kPotion);
			int32_t t = -100;
			if (s && rig.PromptOk(s, f.prompt_potion) && rig.Decode1(s, rig.Greedy(), rig.W(), &t) == SSLM_OK) {
				const sslm_jump_options o = XExact(nullptr);
				live = XJump(rig.m, s, 8, nullptr, &o, rig.W()).st == kXStub ? 0 : 1;
			}
		}
	}
	CHECK_MSG(live == 1, "[%s] exact mode is unimplemented: the J1-X verb returned SSLM_STATUS_NEXT_FREE (the red "
	          "stub, sfd_red_stubs_x.cpp), or F-SFD did not load", cell);
	return live == 1;
}

// ---- the rig: one model (F-SFD's, S-BOUND's full-finish copy, or a page-pinned copy), a fresh pool ----

bool XPinPage(sslm_model m, int32_t page) {
#if defined(SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS)
	const sslm_status st = KvPagesTestPinPagePositions(m, page);
	CHECK_MSG(st == SSLM_OK, "KvPagesTestPinPagePositions(%d) == %d", page, static_cast<int>(st));
	return st == SSLM_OK;
#else
	(void)m;
	static bool noted = false;
	if (!noted) std::printf("NOTE [J1-X] this library has no page pin seam: page-size-%d arms run at F-SFD's own size\n", page);
	noted = true;
	return false;
#endif
}

struct XSpec {
	const char* schema = kPotion;
	const std::vector<int32_t>* prompt = nullptr;  // nullptr: the schema's suite prompt
	int32_t order = 0;                             // 0 greedy; >= 1 damped greedy at that anti-LM order
	int32_t order2 = 0;                            // from output position `switch_at` on, decode at this order
	size_t switch_at = SIZE_MAX;
	bool ws = true;
	int32_t holder_budget = 0;  // 0: sslm_seq_create (SSB5); > 0: sslm_seq_create_budgeted (SSB6)
	int32_t page = 0;           // 0: F-SFD's own page size; else pinned (2 in every page-count cell)
	bool full_finish = false;   // S-BOUND 0: the model without the exactness predicate (cell 5.X5)
	int hook = 0;               // 0 none; 1 bit 1 at max_tasks 4 (Q6 C's split); 2 bit 1 at max_tasks 1;
	                            // 3 bit 1 at max_tasks 4 on real threads; 4 the same at max_tasks 8
	sslm_adapter adapter = nullptr;
	int max_out = 320;
	std::string Name() const {
		char b[200];
		std::snprintf(b, sizeof b, "%s %s%s, %s%s%s%s%s", schema, order ? "damped@" : "greedy",
		              order ? std::to_string(order).c_str() : "", ws ? "ws" : "no ws",
		              holder_budget ? ", budget holder" : "", page ? (", page " + std::to_string(page)).c_str() : "",
		              hook ? (", hook " + std::to_string(hook)).c_str() : "", full_finish ? ", full finish" : "");
		return b;
	}
};

const std::vector<int32_t>& XPrompt(const Sfd& f, const XSpec& sp) {
	if (sp.prompt) return *sp.prompt;
	return std::string(sp.schema) == kPromptResult ? f.prompt_pr : f.prompt_potion;
}

struct XEnv {
	std::unique_ptr<Mapped> own;
	sslm_model m = nullptr;
	std::unique_ptr<Rig> rig;
	CountingHook hook;
	bool ok = false;
	XEnv(const Sfd& f, const XSpec& sp, uint32_t blocks = 8, int workspaces = 1) {
		if (sp.full_finish) {
			m = f.full->m;
		} else if (sp.page > 0) {
			own = Map(f.raw);
			m = own->m;
			if (m) XPinPage(m, sp.page);
		} else {
			m = f.m1->m;
		}
		if (!m) return;
		rig = std::make_unique<Rig>(m, workspaces, blocks);
		ok = rig->ok;
		if (ok && sp.hook) {
			hook.max_tasks = sp.hook == 2 ? 1 : sp.hook == 4 ? 8 : 4;
			hook.threads = sp.hook >= 3;
			for (int i = 0; i < workspaces && ok; ++i) ok = Install(rig->W(i), &hook, SSLM_PARALLEL_FOR_MATVEC);
		}
	}
	sslm_workspace W(const XSpec& sp, int i = 0) const { return sp.ws ? rig->W(i) : nullptr; }
	sslm_seq NewSeq(const Sfd& f, const XSpec& sp) {
		sslm_seq s = sp.holder_budget > 0 ? rig->NewBudgeted(sp.holder_budget, sp.schema) : rig->NewSeq(sp.schema);
		if (s && sp.adapter) CHECK_MSG(sslm_seq_set_adapter(s, sp.adapter) == SSLM_OK, "sslm_seq_set_adapter");
		if (s && !rig->PromptOk(s, XPrompt(f, sp))) return nullptr;
		return s;
	}
};

sslm_decode_params XParams(const Rig& rig, int32_t order) { return order > 0 ? rig.Damped(order) : rig.Greedy(); }
int32_t XOrderAt(const XSpec& sp, size_t pos) { return pos >= sp.switch_at ? sp.order2 : sp.order; }

// ---- observations ------------------------------------------------------------------------------------

struct XObs {
	int st = 0;
	int32_t tok = -100;
	std::vector<uint8_t> blob;
	sslm_seq_kv_stats_out kv{};
	std::vector<uint32_t> table;
};
XObs XObserve(Rig& rig, sslm_seq s, int st, int32_t tok) {
	XObs o;
	o.st = st;
	o.tok = tok;
	o.blob = rig.Save(s);
	o.kv.struct_size = sizeof(o.kv);
	sslm_seq_kv_stats(s, &o.kv);
	uint32_t mapped = 0;
	for (uint32_t i = 0;; ++i) {
		uint32_t page = 0;
		if (sslm_pkv_test_only_seq_table_entry(s, i, &page, &mapped) != SSLM_OK || i >= mapped) break;
		o.table.push_back(page);
	}
	return o;
}
enum : int { kXCmpTok = 1, kXCmpBlob = 2, kXCmpPages = 4, kXCmpAll = 7 };
std::string XDiff(const XObs& a, const XObs& b, int what = kXCmpAll) {
	char buf[400];
	if ((what & kXCmpTok) && (a.st != b.st || a.tok != b.tok)) {
		std::snprintf(buf, sizeof buf, "status/token %d:%d, off %d:%d", a.st, a.tok, b.st, b.tok);
		return buf;
	}
	if ((what & kXCmpBlob) && a.blob != b.blob) {
		const Blob x = ParseBlob(a.blob), y = ParseBlob(b.blob);
		size_t first = 0;
		while (first < std::min(a.blob.size(), b.blob.size()) && a.blob[first] == b.blob[first]) ++first;
		std::snprintf(buf, sizeof buf, "blob differs (sizes %zu/%zu, first byte %zu): walk %u/%u ctx %lld/%lld forced "
		              "%lld/%lld order %d/%d history %zu/%zu ready %u/%u layer %d/%d", a.blob.size(), b.blob.size(), first,
		              x.walk, y.walk, static_cast<long long>(x.ctx), static_cast<long long>(y.ctx),
		              static_cast<long long>(x.forced), static_cast<long long>(y.forced), x.order, y.order, x.hist.size(),
		              y.hist.size(), x.ready, y.ready, x.layer, y.layer);
		return buf;
	}
	if (what & kXCmpPages) {
		const sslm_seq_kv_stats_out& p = a.kv;
		const sslm_seq_kv_stats_out& q = b.kv;
		if (p.mode != q.mode || p.origin != q.origin || p.budget != q.budget || p.limit != q.limit ||
		    p.context_length != q.context_length || p.mapped_private_pages != q.mapped_private_pages ||
		    p.shared_pages != q.shared_pages || p.reserve_pages != q.reserve_pages ||
		    p.materialized_pages != q.materialized_pages) {
			std::snprintf(buf, sizeof buf, "kv stats: ctx %lld/%lld mapped %u/%u shared %u/%u reserve %u/%u materialized "
			              "%u/%u", static_cast<long long>(p.context_length), static_cast<long long>(q.context_length),
			              p.mapped_private_pages, q.mapped_private_pages, p.shared_pages, q.shared_pages, p.reserve_pages,
			              q.reserve_pages, p.materialized_pages, q.materialized_pages);
			return buf;
		}
		if (a.table != b.table) {
			std::snprintf(buf, sizeof buf, "table entries differ (%zu/%zu entries)", a.table.size(), b.table.size());
			return buf;
		}
	}
	return {};
}

// ---- jump-forward off ----------------------------------------------------------------------------------

struct XOff {
	bool ok = false;
	std::vector<XObs> obs;            // after decode i (output position i)
	std::vector<int32_t> stream;      // the produced tokens
	std::vector<uint32_t> walk_after;  // the walk state after token i
};

XOff RunOffOn(XEnv& env, const XSpec& sp, sslm_seq s) {
	XOff o;
	if (!s) return o;
	for (int i = 0; i < sp.max_out; ++i) {
		const sslm_decode_params p = XParams(*env.rig, XOrderAt(sp, size_t(i)));
		int32_t t = -100;
		const sslm_status st = env.rig->Decode1(s, p, env.W(sp), &t);
		o.obs.push_back(XObserve(*env.rig, s, static_cast<int>(st), t));
		if (st != SSLM_OK || t < 0) break;
		o.stream.push_back(t);
		o.walk_after.push_back(ParseBlob(o.obs.back().blob).walk);
	}
	o.ok = !o.obs.empty();
	return o;
}
XOff RunOff(const Sfd& f, const XSpec& sp) {
	std::unique_ptr<MinWorkScope> mw;
	if (sp.hook) mw = std::make_unique<MinWorkScope>(kSmallMinWork);
	XEnv env(f, sp);
	if (!env.ok) return XOff{};
	XOff o = RunOffOn(env, sp, env.NewSeq(f, sp));
	CHECK_MSG(o.ok, "[J1-X off %s] the off run did not start", sp.Name().c_str());
	return o;
}

// ---- drafts (§11.4 "Drafts used by the cells") ----------------------------------------------------------

enum class XDraft { kNull, kEmpty, kPerfect, kPerfectRun, kAlwaysWrong, kWrongAt, kRandom, kGreedyAtContext };
const char* XDraftName(XDraft d) {
	switch (d) {
	case XDraft::kNull: return "NULL";
	case XDraft::kEmpty: return "empty";
	case XDraft::kPerfect: return "perfect";
	case XDraft::kPerfectRun: return "perfect to the next choice";
	case XDraft::kAlwaysWrong: return "always-wrong";
	case XDraft::kWrongAt: return "wrong-at-r";
	case XDraft::kRandom: return "seeded-random";
	case XDraft::kGreedyAtContext: return "greedy-at-context";
	}
	return "?";
}

bool XForced(const Sfd& f, const char* schema, uint32_t s) {
	const RefSchema& r = f.R(schema);
	return s < r.st.size() && r.st[s].forced;
}
// A legal token at state s other than `not_this`; -1 when the row has fewer than two tokens.
int32_t XWrong(const Sfd& f, const char* schema, uint32_t s, int32_t not_this) {
	const std::vector<uint32_t> row = f.LegalIds(schema, s);
	if (row.size() < 2) return -1;
	for (uint32_t t : row)
		if (static_cast<int32_t>(t) != not_this) return static_cast<int32_t>(t);
	return -1;
}
// Continues `d` from state s with legal tokens (the row's first) up to kmax, stopping at an empty row.
void XExtend(const Sfd& f, const char* schema, uint32_t s, size_t kmax, std::vector<int32_t>* d) {
	while (d->size() < kmax) {
		const std::vector<uint32_t> row = f.LegalIds(schema, s);
		if (row.empty()) return;
		d->push_back(static_cast<int32_t>(row[0]));
		if (!f.Step(schema, s, d->back(), &s)) return;
	}
}

// The draft for a verb call whose pending token is off's token at position p.
std::vector<int32_t> XMakeDraft(const Sfd& f, const char* schema, const XOff& off, size_t p, XDraft kind, int32_t r,
                                size_t kmax, std::mt19937& g, const std::vector<int32_t>* greedy_at) {
	std::vector<int32_t> d;
	const size_t n = off.stream.size();
	if (p >= n) return d;
	uint32_t s = off.walk_after[p];
	switch (kind) {
	case XDraft::kNull:
	case XDraft::kEmpty: break;
	case XDraft::kPerfect:
		for (size_t q = p + 1; q < n && d.size() < kmax; ++q) d.push_back(off.stream[q]);
		break;
	case XDraft::kPerfectRun:
		for (size_t q = p + 1; q < n && d.size() < kmax && XForced(f, schema, off.walk_after[q - 1]); ++q)
			d.push_back(off.stream[q]);
		break;
	case XDraft::kAlwaysWrong: {
		const int32_t w = XWrong(f, schema, s, p + 1 < n ? off.stream[p + 1] : -1);
		if (w < 0 || kmax == 0) break;
		d.push_back(w);
		uint32_t nx = 0;
		if (!f.Step(schema, s, w, &nx)) break;
		// "A legal token other than the off run's, at every position with two or more legal tokens; at a
		// one-legal-token state it stops": past the first wrong token there is no off token to differ from.
		while (d.size() < kmax) {
			const std::vector<uint32_t> row = f.LegalIds(schema, nx);
			if (row.size() < 2) break;
			d.push_back(static_cast<int32_t>(row[0]));
			if (!f.Step(schema, nx, d.back(), &nx)) break;
		}
		break;
	}
	case XDraft::kWrongAt: {
		size_t q = p + 1;
		for (; q < n && d.size() < size_t(r) && d.size() < kmax; ++q) {
			d.push_back(off.stream[q]);
			s = off.walk_after[q];
		}
		if (d.size() < size_t(r) || d.size() >= kmax) break;
		const int32_t w = XWrong(f, schema, s, q < n ? off.stream[q] : -1);
		if (w < 0) break;  // one legal token here: the draft is the perfect prefix
		d.push_back(w);
		uint32_t nx = 0;
		if (f.Step(schema, s, w, &nx)) XExtend(f, schema, nx, kmax, &d);
		break;
	}
	case XDraft::kRandom:
		while (d.size() < kmax) {
			const std::vector<uint32_t> row = f.LegalIds(schema, s);
			if (row.empty()) break;
			d.push_back(static_cast<int32_t>(row[g() % row.size()]));
			if (!f.Step(schema, s, d.back(), &s)) break;
		}
		break;
	case XDraft::kGreedyAtContext:
		if (greedy_at)
			for (size_t q = p + 1; q < greedy_at->size() && d.size() < kmax; ++q) d.push_back((*greedy_at)[q]);
		break;
	}
	return d;
}

// greedy_at[i]: the greedy pick at the context of off's tokens [0, i) (a damped off run's own contexts, through
// choice states). Read by restoring off's blob after decode i - 1 and decoding greedy once.
std::vector<int32_t> XGreedyAtContext(const Sfd& f, const XSpec& sp, const XOff& off) {
	std::vector<int32_t> g;
	XSpec gs = sp;
	gs.hook = 0;
	XEnv env(f, gs);
	if (!env.ok) return g;
	for (size_t i = 0; i < off.stream.size(); ++i) {
		sslm_seq s = i == 0 ? env.NewSeq(f, gs) : env.rig->Restore(off.obs[i - 1].blob);
		int32_t t = -100;
		if (!s || env.rig->Decode1(s, env.rig->Greedy(), env.W(gs), &t) != SSLM_OK) break;
		g.push_back(t);
		env.rig->Release(s);
	}
	return g;
}

// ---- the exact run ----------------------------------------------------------------------------------------

struct XVerbRec {
	size_t pos = 0;  // the pending token's output position
	int32_t budget = 0;
	bool null_draft = false;
	std::vector<int32_t> draft;
	Jump j;
	std::vector<sfd::LayerRowsEntry> passes;
	std::vector<bool> rows_check;  // S-ROWS entries written during the call: their check tags
	std::vector<uint64_t> rows;    // and their row counts
	int64_t ctx_before = 0, ctx_after = 0;
	int64_t limit_before = 0;  // sslm_seq_kv_stats' limit before the call (the holder's K/V budget, or the cap)
	std::vector<uint8_t> blob_before, blob_after;
};

struct XPlan {
	XDraft kind = XDraft::kNull;
	int32_t r = 1;
	int32_t budget = 8;
	size_t kmax = SIZE_MAX;  // SIZE_MAX: budget - 1
	uint32_t seed = 1;
	bool heap_draft = false;  // cell 1.X1: a fresh buffer per call, overwritten and freed right after it
	// Per call, after the decode at output position `pos` (the pending token's): whether to call the verb, and
	// with what. Unset: always, with the fields above.
	std::function<bool(size_t pos, std::mt19937& g, XPlan* call)> schedule;
	// Before and after each verb call (seams, hooks, mid-run saves).
	std::function<void(size_t pos, XEnv& env, sslm_seq& s)> before_verb, after_verb;
	// Allowed non-OK verb statuses (cells 5.X1, 5.X3, 5.X4, 5.X6: a pass failure returns canonical mode's
	// status with nothing committed); the loop then retries the call once with nothing armed.
	std::function<bool(sslm_status)> allow_status;
	const std::vector<int32_t>* greedy_at = nullptr;
	int cmp = kXCmpAll;
};

struct XRun {
	bool ok = false;
	std::vector<XObs> obs;
	std::vector<int32_t> stream;
	std::vector<XVerbRec> verbs;
	std::vector<sfd::LayerRowsEntry> passes;  // every S-LAYERROWS entry after the prompt, in order
	std::string diff;                          // the first difference from off ("" when none)
	size_t diff_at = SIZE_MAX;
};

XRun RunExactOn(const Sfd& f, const XSpec& sp, const XOff& off, const XPlan& plan, XEnv& env, sslm_seq s) {
	XRun x;
	if (!s) {
		x.diff = "setup failed";
		return x;
	}
	std::mt19937 g(plan.seed);
	sfd::LayerRowsLogReset();
	size_t pos = 0;
	auto take_passes = [&](std::vector<sfd::LayerRowsEntry>* into) {
		for (size_t i = 0; i < sfd::LayerRowsLogCount(); ++i) {
			x.passes.push_back(sfd::LayerRowsLogAt(i));
			if (into) into->push_back(sfd::LayerRowsLogAt(i));
		}
		sfd::LayerRowsLogReset();
	};
	auto fail = [&](size_t at, const std::string& why) {
		if (x.diff.empty()) {
			x.diff = why;
			x.diff_at = at;
		}
	};
	for (int guard = 0; guard < 4 * sp.max_out && x.diff.empty(); ++guard) {
		// decode_step, compared with off at the same output position
		const sslm_decode_params dp = XParams(*env.rig, XOrderAt(sp, pos));
		int32_t t = -100;
		const sslm_status st = env.rig->Decode1(s, dp, env.W(sp), &t);
		take_passes(nullptr);
		x.obs.push_back(XObserve(*env.rig, s, static_cast<int>(st), t));
		if (pos >= off.obs.size()) {
			fail(pos, "exact decoded past off's end");
			break;
		}
		const std::string d = XDiff(x.obs.back(), off.obs[pos], plan.cmp);
		if (!d.empty()) {
			fail(pos, "after the decode at output position " + std::to_string(pos) + ": " + d);
			break;
		}
		++pos;
		if (st != SSLM_OK || t < 0) break;
		x.stream.push_back(t);
		// the verb
		XPlan call = plan;
		if (plan.schedule && !plan.schedule(pos - 1, g, &call)) continue;
		// An order change lands between calls (§5.10.7): off switches at output position switch_at, so a call
		// before it confirms no token at or past it (its rows are scored at the call's order).
		if (sp.order2 > 0 && pos < sp.switch_at) call.budget = std::min(call.budget, int32_t(sp.switch_at - pos) + 1);
		const size_t kmax = call.kmax != SIZE_MAX ? call.kmax : size_t(std::max(call.budget - 1, 0));
		const bool null_draft = call.kind == XDraft::kNull;
		XVerbRec v;
		v.pos = pos - 1;
		v.budget = call.budget;
		v.null_draft = null_draft;
		if (!null_draft) v.draft = XMakeDraft(f, sp.schema, off, pos - 1, call.kind, call.r, kmax, g, plan.greedy_at);
		if (plan.before_verb) plan.before_verb(pos - 1, env, s);
		v.blob_before = env.rig->Save(s);
		v.ctx_before = ParseBlob(v.blob_before).ctx;
		{
			sslm_seq_kv_stats_out kv{};
			kv.struct_size = sizeof(kv);
			if (sslm_seq_kv_stats(s, &kv) == SSLM_OK) v.limit_before = kv.limit;
		}
		const sslm_decode_params vp = XParams(*env.rig, XOrderAt(sp, pos));
		const sslm_decode_params* vpp = sp.order > 0 || sp.order2 > 0 ? &vp : nullptr;
		sfd::RowsLogReset();
		std::unique_ptr<int32_t[]> heap;
		sslm_jump_options o = XExact(null_draft ? nullptr : &v.draft);
		if (plan.heap_draft && !null_draft) {
			heap.reset(new int32_t[v.draft.size() + 1]);
			std::copy(v.draft.begin(), v.draft.end(), heap.get());
			o.draft = heap.get();
		}
		v.j = XJump(env.m, s, call.budget, vpp, &o, env.W(sp));
		if (heap) {
			std::fill(heap.get(), heap.get() + v.draft.size() + 1, -1);
			heap.reset();
		}
		for (size_t i = 0; i < sfd::RowsLogCount(); ++i) {
			v.rows.push_back(sfd::RowsLogAt(i));
			v.rows_check.push_back(sfd::RowsLogIsCheck(i));
		}
		take_passes(&v.passes);
		if (plan.after_verb) plan.after_verb(pos - 1, env, s);
		v.blob_after = env.rig->Save(s);
		v.ctx_after = ParseBlob(v.blob_after).ctx;
		char why[300];
		if (v.j.st != SSLM_OK) {
			if (!plan.allow_status || !plan.allow_status(v.j.st)) {
				std::snprintf(why, sizeof why, "the exact verb at position %zu (budget %d, %s draft of %zu) returned %d",
				              pos - 1, call.budget, XDraftName(call.kind), v.draft.size(), static_cast<int>(v.j.st));
				fail(pos - 1, why);
				x.verbs.push_back(v);
				break;
			}
			if (v.blob_after != v.blob_before) {
				fail(pos - 1, "a refused exact call (status " + std::to_string(v.j.st) + ") changed the sequence");
				x.verbs.push_back(v);
				break;
			}
			x.verbs.push_back(v);
			// the retry, with nothing armed
			v.j = XJump(env.m, s, call.budget, vpp, &o, env.W(sp));
			take_passes(&v.passes);
			v.ctx_after = env.rig->State(s).ctx;
			if (v.j.st != SSLM_OK) {
				fail(pos - 1, "the retried exact verb returned " + std::to_string(v.j.st));
				break;
			}
		}
		const bool count_ok = v.j.count >= 0 && v.j.count <= std::max(call.budget - 1, 0) &&
		                      (null_draft || size_t(v.j.count) <= v.draft.size()) &&
		                      pos + size_t(v.j.count) <= off.stream.size();
		bool ids_ok = count_ok && v.j.forced.size() == size_t(v.j.count);
		for (int32_t i = 0; ids_ok && i < v.j.count; ++i) ids_ok = v.j.forced[size_t(i)] == off.stream[pos + size_t(i)];
		if (!ids_ok) {
			std::snprintf(why, sizeof why, "the exact verb at position %zu (budget %d, %s draft %s) confirmed %d tokens %s; "
			              "off continues %s", pos - 1, call.budget, XDraftName(call.kind), Str(v.draft, 12).c_str(),
			              v.j.count, Str(v.j.forced, 12).c_str(),
			              Str(std::vector<int32_t>(off.stream.begin() + long(std::min(pos, off.stream.size())),
			                                       off.stream.begin() + long(std::min(pos + 8, off.stream.size()))))
			                  .c_str());
			fail(pos - 1, why);
			x.verbs.push_back(v);
			break;
		}
		x.stream.insert(x.stream.end(), v.j.forced.begin(), v.j.forced.end());
		pos += size_t(v.j.count);
		x.verbs.push_back(v);
	}
	if (x.diff.empty() && x.obs.size() + 0 > 0 && pos != off.obs.size())
		fail(pos, "exact ended at output position " + std::to_string(pos) + ", off at " + std::to_string(off.obs.size()));
	x.ok = x.diff.empty();
	return x;
}

XRun RunExact(const Sfd& f, const XSpec& sp, const XOff& off, const XPlan& plan) {
	std::unique_ptr<MinWorkScope> mw;
	if (sp.hook) mw = std::make_unique<MinWorkScope>(kSmallMinWork);
	XEnv env(f, sp);
	if (!env.ok) {
		XRun x;
		x.diff = "setup failed";
		return x;
	}
	return RunExactOn(f, sp, off, plan, env, env.NewSeq(f, sp));
}

std::string XWhat(const XSpec& sp, const XPlan& p) {
	char b[300];
	std::snprintf(b, sizeof b, "%s, %s draft (r %d), budget %d", sp.Name().c_str(), XDraftName(p.kind), p.r, p.budget);
	return b;
}
#define X_EQUALS_OFF(cell, run, sp, plan)                                                                         \
	CHECK_MSG((run).ok, "[%s %s] exact differs from off: %s", cell, XWhat(sp, plan).c_str(), (run).diff.c_str())

// The output positions p of `off` whose state after p is forced with at least `want` more off tokens in the
// forced run (so a verb call there drafts a chain), and those at a choice state.
std::vector<size_t> XForcedPositions(const Sfd& f, const char* schema, const XOff& off, size_t want) {
	std::vector<size_t> v;
	for (size_t p = 0; p + 1 < off.stream.size(); ++p) {
		size_t run = 0;
		for (size_t q = p + 1; q < off.stream.size() && XForced(f, schema, off.walk_after[q - 1]); ++q) ++run;
		if (run >= want) v.push_back(p);
	}
	return v;
}

// The rig for a cell that starts exact calls from off's own blobs: a fresh pool on the spec's model.
struct XRestoreRig {
	std::unique_ptr<MinWorkScope> mw;
	std::unique_ptr<XEnv> env;
	XRestoreRig(const Sfd& f, const XSpec& sp) {
		if (sp.hook) mw = std::make_unique<MinWorkScope>(kSmallMinWork);
		env = std::make_unique<XEnv>(f, sp);
	}
	bool ok() const { return env && env->ok; }
	sslm_seq At(const XOff& off, size_t p) { return env->rig->Restore(off.obs[p].blob); }
};
const sslm_decode_params* XVerbParams(const XSpec& sp, const sslm_decode_params& p) {
	return sp.order > 0 || sp.order2 > 0 ? &p : nullptr;
}

// ==========================================================================================================
// 6. Numerical and determinism: the oracle lives here.
// ==========================================================================================================

// The fixture precondition (§11.4): on each schema, off agrees with the canonical chain at one forced state at
// least and disagrees at another.
void XPrecondition(const Sfd& f, const char* schema, const XOff& off) {
	const RefSchema& r = f.R(schema);
	int agree = 0, disagree = 0;
	for (size_t i = 0; i < off.stream.size(); ++i) {
		const uint32_t s = i == 0 ? r.start : off.walk_after[i - 1];
		if (s >= r.st.size() || !r.st[s].forced || r.st[s].chain.empty()) continue;
		(r.st[s].chain[0] == off.stream[i] ? agree : disagree)++;
	}
	CHECK_MSG(agree > 0 && disagree > 0, "[6.X1 precondition] %s: off agrees with the canonical chain at %d forced "
	          "states and disagrees at %d; the oracle discriminates only if both are > 0", schema, agree, disagree);
}

// 6.X1 [J1-X; XE1, XE2, XE7, XE8, XE9, XE17, XE31] Exactness: every schema, greedy and damped at orders 2 and 3,
// every draft kind, chunk_budget 1, 2, 3, 7, 8 and 64 (matrix A, workspace, whole-reserve holder, one thread);
// and POTION across workspace / no workspace, SSB5 / SSB6 holders, page size 2 / F-SFD's own, one thread / Q6
// C's split (matrix B). Equals off.
void Cell_X_6_1() {
	if (!XLive("6.X1")) return;
	const Sfd& f = GetSfd();
	for (const char* schema : {kPotion, kPromptResult, kLongKey, kUtf8Enum}) {
		for (int32_t order : {0, 2, 3}) {
			XSpec sp;
			sp.schema = schema;
			sp.order = order;
			const XOff off = RunOff(f, sp);
			if (!off.ok) continue;
			if (order == 0) XPrecondition(f, schema, off);
			std::vector<int32_t> gac;
			if (order > 0) gac = XGreedyAtContext(f, sp, off);
			for (XDraft kind : {XDraft::kNull, XDraft::kPerfect, XDraft::kAlwaysWrong, XDraft::kWrongAt, XDraft::kRandom,
			                    XDraft::kGreedyAtContext}) {
				if (kind == XDraft::kGreedyAtContext && order == 0) continue;  // greedy-at-context is perfect there
				for (int32_t budget : {1, 2, 3, 7, 8, 64}) {
					XPlan plan;
					plan.kind = kind;
					plan.budget = budget;
					plan.seed = uint32_t(budget * 31 + order);
					plan.greedy_at = &gac;
					const XRun run = RunExact(f, sp, off, plan);
					X_EQUALS_OFF("6.X1 A", run, sp, plan);
				}
			}
		}
	}
	for (int32_t order : {0, 2}) {
		for (bool ws : {true, false}) {
			for (int32_t holder : {0, kCap}) {
				for (int32_t page : {0, 2}) {
					for (int hook : {0, 1}) {
						if (hook && !ws) continue;  // the hook lives in the workspace
						XSpec sp;
						sp.order = order;
						sp.ws = ws;
						sp.holder_budget = holder;
						sp.page = page;
						sp.hook = hook;
						const XOff off = RunOff(f, sp);
						if (!off.ok) continue;
						for (XDraft kind : {XDraft::kNull, XDraft::kWrongAt, XDraft::kPerfect}) {
							for (int32_t budget : {2, 8}) {
								XPlan plan;
								plan.kind = kind;
								plan.budget = budget;
								const XRun run = RunExact(f, sp, off, plan);
								X_EQUALS_OFF("6.X1 B", run, sp, plan);
							}
						}
					}
				}
			}
		}
	}
}

// 6.X2 [J1-X] Exact under any host schedule: 200 seeded schedules (after each decode_step the host calls the
// verb or skips it at random, with a random chunk_budget and a random draft kind). Each equals off.
void Cell_X_6_2() {
	if (!XLive("6.X2")) return;
	const Sfd& f = GetSfd();
	XOff offs[2];
	XSpec sps[2];
	for (int i = 0; i < 2; ++i) {
		sps[i].order = i == 0 ? 0 : 2;
		offs[i] = RunOff(f, sps[i]);
	}
	const XDraft kinds[] = {XDraft::kNull, XDraft::kEmpty, XDraft::kPerfect, XDraft::kAlwaysWrong, XDraft::kWrongAt,
	                        XDraft::kRandom};
	int bad = 0;
	for (uint32_t seed = 1; seed <= 200; ++seed) {
		const int i = int(seed % 2);
		if (!offs[i].ok) continue;
		XPlan plan;
		plan.seed = seed;
		plan.schedule = [&](size_t, std::mt19937& g, XPlan* call) {
			if (g() % 10 < 3) return false;
			call->budget = 1 + int32_t(g() % 16);
			call->kind = kinds[g() % (sizeof kinds / sizeof kinds[0])];
			call->r = int32_t(g() % 4);
			return true;
		};
		const XRun run = RunExact(f, sps[i], offs[i], plan);
		if (!run.ok && bad++ < 5)
			CHECK_MSG(false, "[6.X2 seed %u, %s] exact differs from off: %s", seed, sps[i].Name().c_str(), run.diff.c_str());
	}
	CHECK_MSG(bad == 0, "[6.X2] %d of 200 schedules differ from off", bad);
}

// The SSB counters (kv_saturation_count at 92, the four per-site counts at 124..155).
std::vector<uint8_t> XCounters(const std::vector<uint8_t>& b) {
	if (b.size() < 156) return {};
	std::vector<uint8_t> c(b.begin() + 92, b.begin() + 100);
	c.insert(c.end(), b.begin() + 124, b.begin() + 156);
	return c;
}

// 6.X3 [J1-X; XE6] Counter attribution: S-SATROW at a rejected row, then at a confirmed row. The counters equal
// off's, which carries the confirmed row's injected count only. Arms: one thread, batched; bit 1 at M = 8,
// batched; Q6 C's split (M <= 7), where S-SATROW names the r-th one-row pass (a rejected row's pass never
// runs there, so the seam stays armed and is disarmed after the call).
void Cell_X_6_3() {
	if (!XLive("6.X3")) return;
	const Sfd& f = GetSfd();
	struct Arm {
		const char* name;
		const char* schema;
		int hook;
		int32_t budget;  // the call's rows h = budget (a full draft of budget - 1)
	};
	const Arm arms[] = {{"one thread, batched", kPotion, 0, 5}, {"bit 1, M = 8, batched", kLongKey, 1, 8},
	                    {"Q6 C split", kPotion, 1, 5}};
	for (const Arm& a : arms) {
		XSpec sp;
		sp.schema = a.schema;
		sp.hook = a.hook;
		const XOff off = RunOff(f, sp);
		const std::vector<size_t> ps = XForcedPositions(f, a.schema, off, size_t(a.budget - 1));
		CHECK_MSG(!ps.empty(), "[6.X3 %s] no position with a forced run of %d tokens", a.name, a.budget - 1);
		if (ps.empty()) continue;
		const int32_t r = 2;  // wrong-at-2: j = 2, rows 3.. rejected
		// The position: the first whose wrong-at-2 draft is full (budget - 1 tokens), else the first with a row
		// r + 1 at all. XWrong finds no wrong token where the state has one legal token, and XMakeDraft then
		// returns the perfect prefix only, which has no row r + 1 for S-SATROW(r + 1) to name.
		size_t p = SIZE_MAX;
		std::vector<int32_t> d;
		for (int pass = 0; pass < 2 && p == SIZE_MAX; ++pass)
			for (size_t q : ps) {
				std::mt19937 gq(1);
				std::vector<int32_t> dq =
				    XMakeDraft(f, a.schema, off, q, XDraft::kWrongAt, r, size_t(a.budget - 1), gq, nullptr);
				if (pass == 0 ? dq.size() == size_t(a.budget - 1) : dq.size() > size_t(r)) {
					p = q;
					d = std::move(dq);
					break;
				}
			}
		CHECK_MSG(p != SIZE_MAX, "[6.X3 %s] no position whose wrong-at-%d draft has a row %d", a.name, r, r + 1);
		if (p == SIZE_MAX) continue;
		XRestoreRig rr(f, sp);
		if (!rr.ok()) continue;
		const sslm_jump_options o = XExact(&d);
		const bool split = a.hook && a.budget <= int32_t(kQ6cSplitMax);
		// (i) a rejected row
		{
			sslm_seq s = rr.At(off, p);
			sfd::ArmSatRow(r + 1);
			const Jump j = XJump(rr.env->m, s, a.budget, nullptr, &o, rr.env->W(sp));
			const bool left = sfd::SatRowArmed();
			sfd::DisarmSatRow();
			int32_t t = -100;
			const sslm_status st = rr.env->rig->Decode1(s, rr.env->rig->Greedy(), rr.env->W(sp), &t);
			const XObs ob = XObserve(*rr.env->rig, s, static_cast<int>(st), t);
			CHECK_MSG(j.st == SSLM_OK && j.count == r, "[6.X3 %s] the call: status %d count %d, want %d", a.name,
			          static_cast<int>(j.st), j.count, r);
			CHECK_MSG(split ? left : !left, "[6.X3 %s] S-SATROW at rejected row %d: %s", a.name, r + 1,
			          split ? "consumed under the split, where no rejected row runs" : "not consumed by the batched pass");
			const size_t at = p + size_t(j.count) + 1;
			CHECK_MSG(at < off.obs.size() && XCounters(ob.blob) == XCounters(off.obs[at].blob) &&
			              XDiff(ob, off.obs[at], kXCmpTok | kXCmpBlob).empty(),
			          "[6.X3 %s] a rejected row's injected count survived: %s", a.name,
			          at < off.obs.size() ? XDiff(ob, off.obs[at], kXCmpTok | kXCmpBlob).c_str() : "past off's end");
			rr.env->rig->Release(s);
		}
		// (ii) a confirmed row rc: off carries the same injection at the decode that embeds that token
		for (int32_t rc : {0, 1, 2}) {
			sslm_seq s = rr.At(off, p);
			sfd::ArmSatRow(rc);
			const Jump j = XJump(rr.env->m, s, a.budget, nullptr, &o, rr.env->W(sp));
			sfd::DisarmSatRow();
			int32_t t = -100;
			const sslm_status st = rr.env->rig->Decode1(s, rr.env->rig->Greedy(), rr.env->W(sp), &t);
			const XObs ob = XObserve(*rr.env->rig, s, static_cast<int>(st), t);
			rr.env->rig->Release(s);
			const size_t at = p + size_t(std::max(j.count, 0)) + 1;
			// off with the injection: from off's blob with token p + rc pending, the decode that embeds it
			sslm_seq q = rr.At(off, p + size_t(rc));
			XObs want;
			for (size_t i = p + size_t(rc) + 1; q && i <= at && i < off.obs.size(); ++i) {
				if (i == p + size_t(rc) + 1) sfd::ArmSatRow(0);
				int32_t u = -100;
				const sslm_status us = rr.env->rig->Decode1(q, rr.env->rig->Greedy(), rr.env->W(sp), &u);
				sfd::DisarmSatRow();
				want = XObserve(*rr.env->rig, q, static_cast<int>(us), u);
			}
			if (q) rr.env->rig->Release(q);
			CHECK_MSG(j.st == SSLM_OK && j.count == r && !want.blob.empty() &&
			              XDiff(ob, want, kXCmpTok | kXCmpBlob).empty() && XCounters(ob.blob) != XCounters(off.obs[at].blob),
			          "[6.X3 %s] S-SATROW at confirmed row %d: status %d count %d; against off with the same injection: %s",
			          a.name, rc, static_cast<int>(j.st), j.count, XDiff(ob, want, kXCmpTok | kXCmpBlob).c_str());
		}
	}
}

// The off stream's digest, pinned on main (SUPERSLM_SFD_PRINT_GOLDENS=1 prints it): greedy, from the suite's
// prompts, jump-forward off. Exact mode's stream must equal it on every leg (6.X5, 4.X3).
struct XOffGolden {
	const char* schema;
	size_t length;
	const char* digest;
};
constexpr XOffGolden kXOffGoldens[] = {
    {"potion_shop_order", 33, "7953dd212aa523bc1800991ac79268947b1c51df48e20ee91c0075e817244bed"},
    {"prompt_result", 159, "674e41c93d082d38d189b1a55466bb03570fc760bdc76911c80e1d9584ec236c"},
};

// 6.X5 / 4.X3 [J1-X] Golden digests: off's stream equals the pinned golden, and exact's (NULL, perfect and
// wrong-at-1 drafts, budget 8) equals it, on this leg (the scalar, AVX2 and AVX-512 suites each run it).
void Cell_X_6_5() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	for (const XOffGolden& gd : kXOffGoldens) {
		XSpec sp;
		sp.schema = gd.schema;
		const XOff off = RunOff(f, sp);
		CHECK_MSG(off.stream.size() == gd.length && HexDigest(off.stream) == gd.digest,
		          "[6.X5] %s: off's stream (%zu tokens, %s) is not the golden (%zu, %s)", gd.schema, off.stream.size(),
		          HexDigest(off.stream).c_str(), gd.length, gd.digest);
	}
	if (!XLive("6.X5")) return;
	for (const XOffGolden& gd : kXOffGoldens) {
		XSpec sp;
		sp.schema = gd.schema;
		const XOff off = RunOff(f, sp);
		for (XDraft kind : {XDraft::kNull, XDraft::kPerfect, XDraft::kWrongAt}) {
			XPlan plan;
			plan.kind = kind;
			const XRun run = RunExact(f, sp, off, plan);
			CHECK_MSG(run.ok && HexDigest(run.stream) == gd.digest, "[6.X5] %s, %s draft: exact's digest %s, golden %s (%s)",
			          gd.schema, XDraftName(kind), HexDigest(run.stream).c_str(), gd.digest, run.diff.c_str());
		}
	}
}

// 6.X6 [J1-X; XE12, XE25] Damped discrimination: the greedy-at-context draft on PROMPT_RESULT, damped at orders
// 2 and 3, through choice states. Red if no check row's drafted token differs from the damped off token.
void Cell_X_6_6() {
	if (!XLive("6.X6")) return;
	const Sfd& f = GetSfd();
	for (int32_t order : {2, 3}) {
		XSpec sp;
		sp.schema = kPromptResult;
		sp.order = order;
		const XOff off = RunOff(f, sp);
		const std::vector<int32_t> gac = XGreedyAtContext(f, sp, off);
		for (int32_t budget : {4, 8}) {
			XPlan plan;
			plan.kind = XDraft::kGreedyAtContext;
			plan.budget = budget;
			plan.greedy_at = &gac;
			const XRun run = RunExact(f, sp, off, plan);
			X_EQUALS_OFF("6.X6", run, sp, plan);
			int differs = 0;
			for (const XVerbRec& v : run.verbs)
				for (size_t i = 0; i < v.draft.size() && v.pos + 1 + i < off.stream.size(); ++i)
					if (v.draft[i] != off.stream[v.pos + 1 + i]) {
						++differs;
						break;
					}
			CHECK_MSG(differs > 0, "[6.X6 order %d budget %d] precondition: no check row where the greedy-at-context draft "
			          "differs from the damped off token", order, budget);
		}
	}
}

// 6.X7 [J1-X; XE23] Seams in the check. S-CHOICE armed before an exact call whose check rows cover the pinned
// position: the check confirms nothing (j = 0, p alone is committed), the pin survives the call, and the next
// decode consumes it where off consumes it; the same for the degenerate-row injection; and a UTF8ENUM arm that
// pins the 0xC3 byte token (cell 2.6's path). S-ROWS's check entries carry the check tag, and the untagged
// entries per decode equal off's (one per decode finish).
void Cell_X_6_7() {
	if (!XLive("6.X7")) return;
	const Sfd& f = GetSfd();
	struct Arm {
		const char* name;
		const char* schema;
		int seam;  // 0 S-CHOICE (a legal wrong token), 1 S-CHOICE 0xC3, 2 the degenerate row
	};
	const Arm arms[] = {{"S-CHOICE", kPotion, 0}, {"S-CHOICE 0xC3", kUtf8Enum, 1}, {"degenerate row", kPotion, 2}};
	for (const Arm& a : arms) {
		XSpec sp;
		sp.schema = a.schema;
		const XOff off = RunOff(f, sp);
		// p: the pending token's position, such that the next position is a choice the pin can take
		size_t p = SIZE_MAX;
		int32_t pin = -1;
		for (size_t q = 0; q + 2 < off.stream.size() && p == SIZE_MAX; ++q) {
			const uint32_t s = off.walk_after[q];
			if (a.seam == 1) {
				uint32_t nx = 0;
				const int32_t c3 = f.byte_tok[0xC3];
				if (c3 != off.stream[q + 1] && f.Step(a.schema, s, c3, &nx)) {
					p = q;
					pin = c3;
				}
			} else if (a.seam == 0) {
				const int32_t w = XWrong(f, a.schema, s, off.stream[q + 1]);
				if (w >= 0) {
					p = q;
					pin = w;
				}
			} else if (!XForced(f, a.schema, s)) {
				p = q;
			}
		}
		CHECK_MSG(p != SIZE_MAX, "[6.X7 %s] no position for the pin on %s", a.name, a.schema);
		if (p == SIZE_MAX) continue;
		auto arm_seam = [&]() {
			if (a.seam < 2) {
				sfd::ArmChoice(pin);
				return true;
			}
#if defined(SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION)
			ArmCpuFinishDegenerateLogitRowInjection();
			return true;
#else
			CHECK_MSG(false, "[6.X7] the suite's library lacks SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION");
			return false;
#endif
		};
		XRestoreRig rr(f, sp);
		if (!rr.ok()) continue;
		// off with the seam at that position
		std::vector<XObs> want;
		{
			sslm_seq q = rr.At(off, p);
			if (!q || !arm_seam()) continue;
			for (int i = 0; i < 4; ++i) {
				int32_t u = -100;
				const sslm_status us = rr.env->rig->Decode1(q, rr.env->rig->Greedy(), rr.env->W(sp), &u);
				want.push_back(XObserve(*rr.env->rig, q, static_cast<int>(us), u));
				if (us != SSLM_OK || u < 0) break;
			}
			sfd::DisarmChoice();
			rr.env->rig->Release(q);
		}
		std::mt19937 g(1);
		const std::vector<int32_t> d = XMakeDraft(f, a.schema, off, p, XDraft::kPerfect, 0, 6, g, nullptr);
		const sslm_jump_options o = XExact(&d);
		sslm_seq s = rr.At(off, p);
		if (!s || !arm_seam()) continue;
		sfd::RowsLogReset();
		const Jump j = XJump(rr.env->m, s, 7, nullptr, &o, rr.env->W(sp));
		const bool pin_left = a.seam < 2 ? sfd::ChoiceArmed() : true;
		const Blob after = rr.env->rig->State(s);
		CHECK_MSG(j.st == SSLM_OK && j.count == 0 && after.ctx == ParseBlob(off.obs[p].blob).ctx + 1 && after.ready == 1,
		          "[6.X7 %s] with the seam armed the check must confirm nothing and commit p alone: status %d count %d, "
		          "ctx %lld (before %lld), ready %u", a.name, static_cast<int>(j.st), j.count,
		          static_cast<long long>(after.ctx), static_cast<long long>(ParseBlob(off.obs[p].blob).ctx), after.ready);
		CHECK_MSG(pin_left, "[6.X7 %s] the check consumed the single-shot pin", a.name);
		size_t tagged = 0;
		for (size_t i = 0; i < sfd::RowsLogCount(); ++i) tagged += sfd::RowsLogIsCheck(i) ? 1 : 0;
		CHECK_MSG(tagged == sfd::RowsLogCount(), "[6.X7 %s] %zu of the call's %zu S-ROWS entries lack the check tag", a.name,
		          sfd::RowsLogCount() - tagged, sfd::RowsLogCount());
		for (size_t i = 0; i < want.size(); ++i) {
			sfd::RowsLogReset();
			int32_t u = -100;
			const sslm_status us = rr.env->rig->Decode1(s, rr.env->rig->Greedy(), rr.env->W(sp), &u);
			size_t untagged = 0;
			for (size_t k = 0; k < sfd::RowsLogCount(); ++k) untagged += sfd::RowsLogIsCheck(k) ? 0 : 1;
			const XObs ob = XObserve(*rr.env->rig, s, static_cast<int>(us), u);
			CHECK_MSG(XDiff(ob, want[i], kXCmpTok | kXCmpBlob).empty() && untagged == 1,
			          "[6.X7 %s] decode %zu after the call, against off with the same seam: %s; untagged S-ROWS %zu", a.name, i,
			          XDiff(ob, want[i], kXCmpTok | kXCmpBlob).c_str(), untagged);
		}
		sfd::DisarmChoice();
		rr.env->rig->Release(s);
	}
}

// ==========================================================================================================
// 1. Lifetime and reuse
// ==========================================================================================================

// 1.X1 [J1-X] The verb keeps no reference to `draft`: each call's draft is a fresh heap buffer that the host
// overwrites and frees right after the call (the ASan leg sees any later read). The run equals off.
void Cell_X_1_1() {
	if (!XLive("1.X1")) return;
	const Sfd& f = GetSfd();
	for (int32_t order : {0, 2}) {
		XSpec sp;
		sp.order = order;
		const XOff off = RunOff(f, sp);
		for (XDraft kind : {XDraft::kPerfect, XDraft::kWrongAt}) {
			XPlan plan;
			plan.kind = kind;
			plan.heap_draft = true;
			const XRun run = RunExact(f, sp, off, plan);
			X_EQUALS_OFF("1.X1", run, sp, plan);
		}
	}
}

// 1.X2 [J1-X] Exact loop, reset, bind another schema, exact loop again: equals two fresh off runs.
void Cell_X_1_2() {
	if (!XLive("1.X2")) return;
	const Sfd& f = GetSfd();
	XSpec a, b;
	a.schema = kPotion;
	b.schema = kPromptResult;
	const XOff off_a = RunOff(f, a), off_b = RunOff(f, b);
	XEnv env(f, a);
	if (!env.ok) return;
	sslm_seq s = env.NewSeq(f, a);
	XPlan plan;
	plan.kind = XDraft::kWrongAt;
	const XRun ra = RunExactOn(f, a, off_a, plan, env, s);
	X_EQUALS_OFF("1.X2 first", ra, a, plan);
	CHECK_MSG(s && sslm_seq_reset(s) == SSLM_OK && env.rig->TryBind(s, kPromptResult) == SSLM_OK &&
	              env.rig->PromptOk(s, f.prompt_pr),
	          "[1.X2] reset, rebind to PROMPT_RESULT and prompt failed");
	plan.cmp = kXCmpTok | kXCmpBlob;  // the reset sequence's pages are not a fresh holder's
	const XRun rb = RunExactOn(f, b, off_b, plan, env, s);
	X_EQUALS_OFF("1.X2 after reset", rb, b, plan);
}

// Cell 1.X3's arms; `stub_unmap` runs them with HolderUnmapTail stubbed (11.X3). Returns the number of runs
// that differed from off.
int XUnmapTailArms(const Sfd& f, bool stub_unmap, bool report) {
	int differ = 0;
	for (int hook : {0, 1}) {
		XSpec sp;
		sp.page = 2;
		sp.hook = hook;
		const XOff off = RunOff(f, sp);
		bool before = false, at = false, crossed = false;
		for (int32_t r : {0, 1, 2, 3}) {
			for (int32_t budget : {4, 8}) {
				XPlan plan;
				plan.kind = XDraft::kWrongAt;
				plan.r = r;
				plan.budget = budget;
				sfd::SetHolderUnmapTailStubbed(stub_unmap);
				const XRun run = RunExact(f, sp, off, plan);
				sfd::SetHolderUnmapTailStubbed(false);
				if (!run.ok) ++differ;
				if (report) X_EQUALS_OFF("1.X3", run, sp, plan);
				for (const XVerbRec& v : run.verbs) {
					const int64_t rows = int64_t(v.draft.size()) + 1, kept = v.ctx_after - v.ctx_before;
					if (kept < 1 || rows <= kept) continue;  // nothing rejected
					(v.ctx_after % 2 == 0 ? at : before) = true;
					if ((v.ctx_before + rows + 1) / 2 > (v.ctx_after + 1) / 2) crossed = true;
				}
			}
		}
		if (report)
			CHECK_MSG(before && at && crossed, "[1.X3 hook %d] coverage: a commit ending between pages %d, at a boundary "
			          "%d, rejected rows crossing a page %d", hook, before, at, crossed);
	}
	return differ;
}

// 1.X3 [J1-X; XE5, XE28, XE30] HolderUnmapTail, page size 2: (a) one thread and (b) bit 1 at max_tasks 4 under
// Q6 C's split, with commits that end mid-page and at a page boundary and rejected rows that cross one. The
// mapped and reserve counts and the table entries equal off's after every decode_step (so the next map draws
// the same physical page as off: LIFO). Release returns every page to the pool.
void Cell_X_1_3() {
	if (!XLive("1.X3")) return;
	const Sfd& f = GetSfd();
	XUnmapTailArms(f, false, true);
	XSpec sp;
	sp.page = 2;
	XEnv env(f, sp);
	if (!env.ok) return;
	sslm_kv_pool_stats_out ps0{};
	ps0.struct_size = sizeof ps0;
	sslm_kv_pool_stats(env.rig->pool, &ps0);
	{
		const XOff off = RunOff(f, sp);
		XPlan plan;
		plan.kind = XDraft::kWrongAt;
		sslm_seq s = env.NewSeq(f, sp);
		RunExactOn(f, sp, off, plan, env, s);
		env.rig->Release(s);
	}
	sslm_kv_pool_stats_out ps1{};
	ps1.struct_size = sizeof ps1;
	sslm_kv_pool_stats(env.rig->pool, &ps1);
	CHECK_MSG(ps1.free_pages == ps0.free_pages, "[1.X3] after release the pool has %u free pages, %u before the run",
	          ps1.free_pages, ps0.free_pages);
}

// 1.X4 [J1-X; XE22] Release poison: after an exact call whose rejected rows crossed a page boundary (page size
// 2), release the sequence at once and peek the page the rejected rows wrote: it reads 0xCD. The page is the
// one off maps next at that position (HolderUnmapTail's LIFO return, 1.X3).
void Cell_X_1_4() {
	if (!XLive("1.X4")) return;
	const Sfd& f = GetSfd();
	XSpec sp;
	sp.page = 2;
	const XOff off = RunOff(f, sp);
	// the first call position whose wrong-at-0 draft leaves two or more rejected rows
	const std::vector<size_t> ps = XForcedPositions(f, sp.schema, off, 3);
	CHECK_MSG(!ps.empty(), "[1.X4] no position with a forced run of 3");
	if (ps.empty()) return;
	// Decode to the call position with canonical-shaped budget-1 exact calls (so the holder matches off's
	// there), make the always-wrong call, read where its commit ended, and release at once: no later decode
	// re-maps the page the rejected rows wrote.
	XEnv env2(f, sp);
	if (!env2.ok) return;
	sslm_seq s2 = env2.NewSeq(f, sp);
	int64_t ctx_after = -1;
	bool released = false;
	for (size_t pos = 0; s2 && pos <= ps[0]; ++pos) {
		int32_t t = -100;
		if (env2.rig->Decode1(s2, env2.rig->Greedy(), env2.W(sp), &t) != SSLM_OK) break;
		if (pos == ps[0]) {
			std::mt19937 g(1);
			const std::vector<int32_t> d = XMakeDraft(f, sp.schema, off, pos, XDraft::kAlwaysWrong, 0, 4, g, nullptr);
			const sslm_jump_options o = XExact(&d);
			if (XJump(env2.m, s2, 5, nullptr, &o, env2.W(sp)).st == SSLM_OK) ctx_after = env2.rig->State(s2).ctx;
			env2.rig->Release(s2);
			released = true;
			break;
		}
		const sslm_jump_options o = XExact(nullptr);
		XJump(env2.m, s2, 1, nullptr, &o, env2.W(sp));
	}
	CHECK_MSG(ctx_after >= 0, "[1.X4] the exact call at position %zu did not run", ps[0]);
	if (ctx_after < 0) return;
	// The rejected rows started at position ctx_after; page ctx_after / 2 (or the next page when ctx_after is
	// mid-page) is the first page they wrote that the commit does not keep.
	const uint32_t first_page = uint32_t((ctx_after + 1) / 2);
	uint32_t phys = UINT32_MAX;
	for (const XObs& o : off.obs)
		if (o.table.size() > first_page) {
			phys = o.table[first_page];
			break;
		}
	CHECK_MSG(released && phys != UINT32_MAX, "[1.X4] setup: released %d, page %u", released, phys);
	if (!released || phys == UINT32_MAX) return;
	size_t lo = 1, hi = 1 << 20;  // the page size in bytes: the largest n the peek admits
	std::vector<uint8_t> buf(hi);
	while (lo < hi) {
		const size_t mid = (lo + hi + 1) / 2;
		if (sslm_pkv_test_only_peek_page_bytes(env2.rig->pool, phys, buf.data(), mid) == SSLM_OK) lo = mid;
		else hi = mid - 1;
	}
	const bool read = sslm_pkv_test_only_peek_page_bytes(env2.rig->pool, phys, buf.data(), lo) == SSLM_OK;
	const bool poisoned = read && std::all_of(buf.begin(), buf.begin() + long(lo), [](uint8_t b) { return b == 0xCD; });
	CHECK_MSG(poisoned, "[1.X4] page %u (the rejected rows' page) does not read 0xCD after release (%zu bytes)", phys, lo);
}

// ==========================================================================================================
// 2. Trust and hostile input
// ==========================================================================================================

// 2.X1 [J1-X; XE15, XE24] (a) Each §5.10.2 refusal returns SSLM_INVALID_ARGUMENT, writes neither output, and
// leaves the blob byte-equal. (b) On a fresh bound sequence each refusal leaves it bindable (cell 1.2's pattern).
void Cell_X_2_1() {
	if (!XLive("2.X1")) return;
	const Sfd& f = GetSfd();
	const int32_t draft[4] = {1, 2, 3, 4};
	struct Case {
		const char* name;
		sslm_jump_options o;
	};
	const uint32_t sz = sizeof(sslm_jump_options);
	const Case cases[] = {
	    {"struct_size 0", XOpts(SSLM_JUMP_EXACT, nullptr, 0, 0)},
	    {"struct_size - 4", XOpts(SSLM_JUMP_EXACT, nullptr, 0, sz - 4)},
	    {"struct_size + 4", XOpts(SSLM_JUMP_EXACT, nullptr, 0, sz + 4)},
	    {"mode 2", XOpts(2, nullptr, 0)},
	    {"mode -1", XOpts(-1, nullptr, 0)},
	    {"exact, draft_count -1", XOpts(SSLM_JUMP_EXACT, draft, -1)},
	    {"exact, NULL draft with draft_count 3", XOpts(SSLM_JUMP_EXACT, nullptr, 3)},
	    {"canonical, a draft", XOpts(SSLM_JUMP_CANONICAL, draft, 0)},
	    {"canonical, draft_count 2", XOpts(SSLM_JUMP_CANONICAL, nullptr, 2)},
	    {"canonical, draft and count", XOpts(SSLM_JUMP_CANONICAL, draft, 2)},
	};
	for (const Case& c : cases) {
		{
			Rig rig(f.m1->m);
			sslm_seq s = rig.NewSeq(kPotion);
			int32_t t = -100;
			if (!s || !rig.PromptOk(s, f.prompt_potion) || rig.Decode1(s, rig.Greedy(), rig.W(), &t) != SSLM_OK) continue;
			const std::vector<uint8_t> before = rig.Save(s);
			int32_t buf[8];
			std::fill(buf, buf + 8, kSent);
			int32_t count = kSent;
			const sslm_status st = sfd::JumpWithOptions(rig.m, s, 8, nullptr, &c.o, rig.W(), buf, 8, &count);
			CHECK_MSG(st == SSLM_INVALID_ARGUMENT && count == kSent && std::all_of(buf, buf + 8, [](int32_t v) {
				          return v == kSent;
			          }) && rig.Save(s) == before,
			          "[2.X1 (a) %s] status %d, count %d, outputs %s, blob %s", c.name, static_cast<int>(st), count,
			          std::all_of(buf, buf + 8, [](int32_t v) { return v == kSent; }) ? "unwritten" : "written",
			          rig.Save(s) == before ? "unchanged" : "changed");
		}
		{
			Rig rig(f.m1->m);
			sslm_seq s = rig.NewSeq(kPotion);
			if (!s) continue;
			int32_t buf[8], count = 0;
			sfd::JumpWithOptions(rig.m, s, 8, nullptr, &c.o, rig.W(), buf, 8, &count);
			CHECK_MSG(rig.TryBind(s, kPromptResult) == SSLM_OK, "[2.X1 (b) %s] the refused call cleared bind eligibility",
			          c.name);
		}
	}
}

// The tokens of `d` the draft walk keeps from state s: in range, not special, not byteless, legal at its walk
// state, at most kmax.
size_t XLegalPrefix(const Sfd& f, const char* schema, uint32_t s, const int32_t* d, size_t n, size_t kmax) {
	size_t k = 0;
	for (; k < n && k < kmax; ++k) {
		const int32_t t = d[k];
		uint32_t nx = 0;
		if (t < 0 || t >= kTokVocab || f.specials.count(t) || !f.Step(schema, s, t, &nx)) break;
		s = nx;
	}
	return k;
}

// A buffer of exactly n int32 elements that ends at a guard page (POSIX: PROT_NONE), so a read past n faults.
struct XGuarded {
	uint8_t* base = nullptr;
	size_t bytes = 0;
	int32_t* p = nullptr;
	explicit XGuarded(size_t n) {
#if !defined(_WIN32)
		const size_t pg = size_t(sysconf(_SC_PAGESIZE));
		bytes = 2 * pg;
		void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (m == MAP_FAILED) return;
		base = static_cast<uint8_t*>(m);
		mprotect(base + pg, pg, PROT_NONE);
		p = reinterpret_cast<int32_t*>(base + pg - n * sizeof(int32_t));
#else
		const size_t pg = 4096;
		bytes = 2 * pg;
		base = static_cast<uint8_t*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
		if (!base) return;
		DWORD old = 0;
		VirtualProtect(base + pg, pg, PAGE_NOACCESS, &old);
		p = reinterpret_cast<int32_t*>(base + pg - n * sizeof(int32_t));
#endif
	}
	~XGuarded() {
#if !defined(_WIN32)
		if (base) munmap(base, bytes);
#else
		if (base) VirtualFree(base, 0, MEM_RELEASE);
#endif
	}
	XGuarded(const XGuarded&) = delete;
	XGuarded& operator=(const XGuarded&) = delete;
};

// 2.X2 [J1-X; XE14, XE19] Hostile drafts, each against the off run: illegal at index 0 and at index k - 1; ids
// -1, V and INT32_MAX; a special id; a byteless id; a draft that runs past the accepting state; and draft_count
// far above chunk_budget - 1 with a buffer of exactly chunk_budget - 1 elements ending at a guard page. The verb
// returns SSLM_OK for every one, no illegal token is embedded (S-LAYERROWS: one pass of 1 + the legal prefix;
// an empty legal prefix leaves k = 0 after the truncation, a no-op with no pass and the blob unchanged, by the
// conductor's G-4 ruling), and the run equals off.
void Cell_X_2_2() {
	if (!XLive("2.X2")) return;
	const Sfd& f = GetSfd();
	XSpec sp;
	const XOff off = RunOff(f, sp);
	const std::vector<size_t> ps = XForcedPositions(f, sp.schema, off, 3);
	CHECK_MSG(!ps.empty() && off.stream.size() > 4, "[2.X2] no position with a forced run of 3");
	if (ps.empty() || off.stream.size() <= 4) return;
	const size_t p = ps[0];
	const uint32_t s0 = off.walk_after[p];
	int32_t illegal = -1;
	for (int32_t t = 0; t < kTokVocab && illegal < 0; ++t) {
		uint32_t nx = 0;
		if (!f.Step(sp.schema, s0, t, &nx) && !f.specials.count(t)) illegal = t;
	}
	const int32_t special = f.specials.empty() ? -1 : *f.specials.begin();
	std::vector<int32_t> perfect3(off.stream.begin() + long(p) + 1, off.stream.begin() + long(p) + 4);
	struct Case {
		const char* name;
		size_t at;  // the pending position
		std::vector<int32_t> d;
	};
	std::vector<Case> cases = {
	    {"illegal at 0", p, {illegal, perfect3[1]}},
	    {"illegal at k - 1", p, {perfect3[0], perfect3[1], illegal}},
	    {"-1 at 0", p, {-1, perfect3[1]}},
	    {"V at 0", p, {kV, perfect3[1]}},
	    {"INT32_MAX at 0", p, {INT32_MAX}},
	    {"special at 0", p, {special}},
	    {"byteless at 0", p, {kTokVocab}},
	};
	{
		// past the accepting state: off's tail, then more tokens
		const size_t q = off.stream.size() >= 3 ? off.stream.size() - 3 : 0;
		std::vector<int32_t> d(off.stream.begin() + long(q) + 1, off.stream.end());
		d.push_back(off.stream[0]);
		d.push_back(off.stream[1]);
		cases.push_back({"past the accepting state", q, d});
	}
	XRestoreRig rr(f, sp);
	if (!rr.ok()) return;
	for (const Case& c : cases) {
		sslm_seq s = rr.At(off, c.at);
		if (!s) continue;
		sfd::LayerRowsLogReset();
		const sslm_jump_options o = XExact(&c.d);
		const Jump j = XJump(rr.env->m, s, 8, nullptr, &o, rr.env->W(sp));
		const std::vector<sfd::LayerRowsEntry> passes = Q6cPasses();
		const size_t legal = XLegalPrefix(f, sp.schema, off.walk_after[c.at], c.d.data(), c.d.size(), 7);
		uint64_t rows = 0;
		for (const sfd::LayerRowsEntry& e : passes) rows += e.rows;
		// k is taken after the clips and the truncation (the conductor's G-4 ruling): an empty legal prefix is a
		// no-op, so exactly no pass runs; otherwise exactly one pass of 1 + the legal prefix.
		const bool rows_ok = legal > 0 ? rows == 1 + legal : rows == 0;
		const bool noop_ok = legal > 0 || (j.count == 0 && rr.env->rig->Save(s) == off.obs[c.at].blob);
		int32_t t = -100;
		const sslm_status st = rr.env->rig->Decode1(s, rr.env->rig->Greedy(), rr.env->W(sp), &t);
		const size_t at = c.at + size_t(std::max(j.count, 0)) + 1;
		const XObs ob = XObserve(*rr.env->rig, s, static_cast<int>(st), t);
		CHECK_MSG(j.st == SSLM_OK && rows_ok && noop_ok && size_t(std::max(j.count, 0)) <= legal && at < off.obs.size() &&
		              XDiff(ob, off.obs[at], kXCmpTok | kXCmpBlob).empty(),
		          "[2.X2 %s] status %d count %d, S-LAYERROWS rows %llu (legal prefix %zu); then %s", c.name,
		          static_cast<int>(j.st), j.count, static_cast<unsigned long long>(rows), legal,
		          at < off.obs.size() ? XDiff(ob, off.obs[at], kXCmpTok | kXCmpBlob).c_str() : "past off's end");
		rr.env->rig->Release(s);
	}
	// draft_count far above chunk_budget - 1, the buffer exactly chunk_budget - 1 long at a guard page
	{
		const int32_t budget = 4;
		XGuarded gbuf(size_t(budget - 1));
		CHECK_MSG(gbuf.p != nullptr, "[2.X2] the guard page did not map");
		sslm_seq s = rr.At(off, p);
		if (gbuf.p && s) {
			for (int32_t i = 0; i < budget - 1; ++i) gbuf.p[i] = perfect3[size_t(i)];
			const sslm_jump_options o = XOpts(SSLM_JUMP_EXACT, gbuf.p, 1 << 20);
			const Jump j = XJump(rr.env->m, s, budget, nullptr, &o, rr.env->W(sp));
			int32_t t = -100;
			const sslm_status st = rr.env->rig->Decode1(s, rr.env->rig->Greedy(), rr.env->W(sp), &t);
			const size_t at = p + size_t(std::max(j.count, 0)) + 1;
			const XObs ob = XObserve(*rr.env->rig, s, static_cast<int>(st), t);
			CHECK_MSG(j.st == SSLM_OK && j.count == budget - 1 && at < off.obs.size() &&
			              XDiff(ob, off.obs[at], kXCmpTok | kXCmpBlob).empty(),
			          "[2.X2 guard page] status %d count %d (want %d)", static_cast<int>(j.st), j.count, budget - 1);
		}
	}
}

// 2.X3 [J1-X] Exact mode on an artifact without a forced table (no TOK1), with a host draft and with NULL:
// SSLM_ARTIFACT_REJECTED, nothing changed.
void Cell_X_2_3() {
	if (!XLive("2.X3")) return;
	const Sfd& f = GetSfd();
	std::unique_ptr<Mapped> m = Map(VariantWithout(f, {kSecTokenizer, kSecUnicode}, 0));
	if (!m->m) return;
	Rig rig(m->m);
	sslm_seq s = rig.NewSeq(kPotion);
	int32_t t = -100;
	if (!s || !rig.PromptOk(s, f.prompt_potion) || rig.Decode1(s, rig.Greedy(), rig.W(), &t) != SSLM_OK) return;
	const std::vector<uint8_t> before = rig.Save(s);
	const std::vector<int32_t> d = {t, t};
	for (const std::vector<int32_t>* dd : {&d, static_cast<const std::vector<int32_t>*>(nullptr)}) {
		const sslm_jump_options o = XExact(dd);
		const Jump j = XJump(rig.m, s, 8, nullptr, &o, rig.W());
		CHECK_MSG(j.st == SSLM_ARTIFACT_REJECTED && rig.Save(s) == before, "[2.X3 %s draft] status %d, blob %s",
		          dd ? "host" : "NULL", static_cast<int>(j.st), rig.Save(s) == before ? "unchanged" : "changed");
	}
}

// ==========================================================================================================
// 3. Concurrency
// ==========================================================================================================

// 3.X1 [J1-X] Under TSan: a decode_step on the same sequence during the exact verb, and a release during it (the
// verb held inside a one-row pass by a blocking hook). No race; the surviving calls equal off.
void Cell_X_3_1() {
	if (!XLive("3.X1")) return;
	const MinWorkScope min_work(kSmallMinWork);
	const Sfd& f = GetSfd();
	XSpec sp;
	const XOff off = RunOff(f, sp);
	const std::vector<size_t> ps = XForcedPositions(f, sp.schema, off, 2);
	if (ps.empty()) return;
	const size_t p = ps[0];
	std::mt19937 g(1);
	const std::vector<int32_t> d = XMakeDraft(f, sp.schema, off, p, XDraft::kPerfect, 0, 2, g, nullptr);
	for (int arm = 0; arm < 2; ++arm) {
		Rig rig(f.m1->m);
		BlockingHook h;
		const sslm_parallel_for pf = h.Pf();
		if (!rig.ok || sslm_workspace_set_parallel_for(rig.W(), &pf) != SSLM_OK) continue;
		sslm_seq s = rig.Restore(off.obs[p].blob);
		if (!s) continue;
		Jump j;
		const sslm_jump_options o = XExact(&d);
		std::thread verb([&] { j = XJump(rig.m, s, 3, nullptr, &o, rig.W()); });
		const bool inside = h.WaitEntered(5);
		CHECK_MSG(inside, "[3.X1 arm %d] the verb never entered the hook's run", arm);
		std::thread other;
		sslm_status other_st = SSLM_OK;
		int32_t other_tok = -100;
		if (inside) {
			if (arm == 0) {
				rig.Forget(s);
				other = std::thread([&] { other_st = sslm_seq_release(s); });
			} else {
				other = std::thread([&] {
					const sslm_decode_params gp = rig.Greedy();
					other_st = sslm_decode_step_v2(rig.m, &s, 1, &gp, nullptr, &other_tok);
				});
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
		h.Release();
		verb.join();
		if (other.joinable()) other.join();
		CHECK_MSG(j.st == SSLM_OK && j.count == int32_t(d.size()), "[3.X1 arm %d] the verb: status %d count %d, want %zu",
		          arm, static_cast<int>(j.st), j.count, d.size());
		if (arm == 0) {
			CHECK_MSG(other_st == SSLM_OK, "[3.X1] the concurrent release == %d", static_cast<int>(other_st));
		} else {
			const size_t at = p + d.size() + 1;
			CHECK_MSG(other_st == SSLM_OK && at < off.stream.size() + 1 && other_tok == off.obs[at].tok,
			          "[3.X1] the concurrent decode: status %d token %d, off %d", static_cast<int>(other_st), other_tok,
			          at < off.obs.size() ? off.obs[at].tok : -1);
		}
	}
}

// 3.X2 [J1-X] Under TSan, Q6 C with real worker threads at max_tasks 4 and 8, in exact mode: equals off.
void Cell_X_3_2() {
	if (!XLive("3.X2")) return;
	const Sfd& f = GetSfd();
	for (int hook : {3, 4}) {
		XSpec sp;
		sp.hook = hook;
		sp.order = 2;
		const XOff off = RunOff(f, sp);
		for (XDraft kind : {XDraft::kPerfect, XDraft::kWrongAt}) {
			XPlan plan;
			plan.kind = kind;
			plan.budget = 6;
			const XRun run = RunExact(f, sp, off, plan);
			X_EQUALS_OFF("3.X2", run, sp, plan);
		}
	}
}

// 3.X3 [J1-X] Two sequences on two threads share one model, in exact mode with different drafts: each equals
// its off run.
void Cell_X_3_3() {
	if (!XLive("3.X3")) return;
	const Sfd& f = GetSfd();
	XSpec sp[2];
	sp[1].schema = kPromptResult;
	XOff off[2];
	for (int i = 0; i < 2; ++i) off[i] = RunOff(f, sp[i]);
	XPlan plan[2];
	plan[0].kind = XDraft::kPerfect;
	plan[1].kind = XDraft::kWrongAt;
	XRun run[2];
	std::thread th[2];
	for (int i = 0; i < 2; ++i) th[i] = std::thread([&, i] { run[i] = RunExact(f, sp[i], off[i], plan[i]); });
	for (std::thread& t : th) t.join();
	for (int i = 0; i < 2; ++i) X_EQUALS_OFF("3.X3", run[i], sp[i], plan[i]);
}

// ==========================================================================================================
// 4. Shape and platform
// ==========================================================================================================

// 4.X1 [J1-X; XE2, XE8, XE31] Mismatch at every position: for each verb position of an off run, the wrong-at-r
// draft for every r from 0 to k - 1, and the perfect draft. j = r each time (k for the perfect draft);
// S-LAYERROWS shows one pass of 1 + k rows at one thread; the next decode equals off.
void Cell_X_4_1() {
	if (!XLive("4.X1")) return;
	const Sfd& f = GetSfd();
	for (int32_t order : {0, 2}) {
		XSpec sp;
		sp.order = order;
		const XOff off = RunOff(f, sp);
		XRestoreRig rr(f, sp);
		if (!rr.ok()) continue;
		const sslm_decode_params vp = XParams(*rr.env->rig, order);
		int bad = 0;
		for (size_t p = 0; p + 2 < off.stream.size(); ++p) {
			const int32_t budget = 8;
			for (int32_t r = 0; r <= budget - 1; ++r) {
				std::mt19937 g(1);
				const bool perfect = r == budget - 1;
				const std::vector<int32_t> d = XMakeDraft(f, sp.schema, off, p, perfect ? XDraft::kPerfect : XDraft::kWrongAt,
				                                          r, size_t(budget - 1), g, nullptr);
				if (d.empty()) continue;
				// j: the prefix of d that equals off
				int32_t want = 0;
				while (size_t(want) < d.size() && p + 1 + size_t(want) < off.stream.size() &&
				       d[size_t(want)] == off.stream[p + 1 + size_t(want)])
					++want;
				sslm_seq s = rr.At(off, p);
				if (!s) continue;
				sfd::LayerRowsLogReset();
				const sslm_jump_options o = XExact(&d);
				const Jump j = XJump(rr.env->m, s, budget, XVerbParams(sp, vp), &o, rr.env->W(sp));
				const std::vector<sfd::LayerRowsEntry> passes = Q6cPasses();
				int32_t t = -100;
				const sslm_status st = rr.env->rig->Decode1(s, vp, rr.env->W(sp), &t);
				const XObs ob = XObserve(*rr.env->rig, s, static_cast<int>(st), t);
				const size_t at = p + size_t(std::max(j.count, 0)) + 1;
				const bool ok = j.st == SSLM_OK && j.count == want && (perfect || want == std::min<int32_t>(r, int32_t(d.size()))) &&
				                passes.size() == 1 && passes[0].kind == sfd::kLayerRowsBatched &&
				                passes[0].rows == 1 + d.size() && at < off.obs.size() &&
				                XDiff(ob, off.obs[at], kXCmpTok | kXCmpBlob).empty();
				if (!ok && bad++ < 6)
					CHECK_MSG(false, "[4.X1 %s, p %zu, %s r %d] status %d count %d (want %d), S-LAYERROWS %s (want one pass of "
					          "%zu); then %s", sp.Name().c_str(), p, perfect ? "perfect" : "wrong-at", r,
					          static_cast<int>(j.st), j.count, want, Q6cShape(passes).c_str(), 1 + d.size(),
					          at < off.obs.size() ? XDiff(ob, off.obs[at], kXCmpTok | kXCmpBlob).c_str() : "past off's end");
				rr.env->rig->Release(s);
				if (perfect) break;
			}
		}
		CHECK_MSG(bad == 0, "[4.X1 %s] %d calls failed", sp.Name().c_str(), bad);
	}
}

// 4.X2 [J1-X] Draft and budget extremes, each at one thread, under Q6 C's split (bit 1, max_tasks 4) and at bit
// 1 with max_tasks 1: k = 0 in both forms (a non-NULL draft with draft_count 0 is a no-op that leaves the pending
// token pending; NULL drafts the canonical chain), k = 1, k = chunk_budget - 1, chunk_budget 1 (k clipped to 0: a
// no-op, by the conductor's G-4 ruling that k is taken after the clips), chunk_budget 2, and M = 7, 8 and 9+ on
// LONGKEY. Each equals off.
void Cell_X_4_2() {
	if (!XLive("4.X2")) return;
	const Sfd& f = GetSfd();
	struct Case {
		const char* name;
		const char* schema;
		XDraft kind;
		int32_t budget;
		size_t kmax;
	};
	const Case cases[] = {
	    {"k = 0, empty draft", kPotion, XDraft::kEmpty, 8, SIZE_MAX},
	    {"k = 0, NULL draft", kPotion, XDraft::kNull, 8, SIZE_MAX},
	    {"k = 1", kPotion, XDraft::kPerfect, 8, 1},
	    {"k = chunk_budget - 1", kPotion, XDraft::kPerfect, 8, SIZE_MAX},
	    {"chunk_budget 1", kPotion, XDraft::kPerfect, 1, SIZE_MAX},
	    {"chunk_budget 2", kPotion, XDraft::kPerfect, 2, SIZE_MAX},
	    {"M = 7", kLongKey, XDraft::kPerfect, 7, SIZE_MAX},
	    {"M = 8", kLongKey, XDraft::kPerfect, 8, SIZE_MAX},
	    {"M = 9", kLongKey, XDraft::kPerfect, 9, SIZE_MAX},
	    {"M = 16", kLongKey, XDraft::kPerfect, 16, SIZE_MAX},
	};
	for (int hook : {0, 1, 2}) {
		for (const Case& c : cases) {
			XSpec sp;
			sp.schema = c.schema;
			sp.hook = hook;
			const XOff off = RunOff(f, sp);
			XPlan plan;
			plan.kind = c.kind;
			plan.budget = c.budget;
			plan.kmax = c.kmax;
			const XRun run = RunExact(f, sp, off, plan);
			CHECK_MSG(run.ok, "[4.X2 %s, hook %d] exact differs from off: %s", c.name, hook, run.diff.c_str());
			for (const XVerbRec& v : run.verbs) {
				if (c.kind == XDraft::kEmpty) {
					CHECK_MSG(v.j.count == 0 && v.blob_after == v.blob_before, "[4.X2 %s, hook %d] the empty draft at %zu was "
					          "not a no-op", c.name, hook, v.pos);
				}
				if (c.budget == 1) {
					// The conductor's G-4 ruling: k is taken after the clips, so chunk_budget 1 clips the draft to
					// k = 0 and the call is a no-op returning OK; the pending token stays pending (C13 at h = 0 alike).
					CHECK_MSG(v.j.st == SSLM_OK && v.j.count == 0 && v.blob_after == v.blob_before,
					          "[4.X2 %s, hook %d] the budget-1 call at %zu (k clipped to 0) must be a no-op: status %d count "
					          "%d, ctx +%lld, blob %s", c.name, hook, v.pos, static_cast<int>(v.j.st), v.j.count,
					          static_cast<long long>(v.ctx_after - v.ctx_before),
					          v.blob_after == v.blob_before ? "unchanged" : "changed");
				}
			}
			// Only where the draft can fill the budget: a draft capped at kmax tokens admits at most kmax + 1 rows.
			if (c.kind == XDraft::kPerfect && c.budget >= 7 && c.kmax >= size_t(c.budget - 1)) {
				bool reached = false;
				for (const XVerbRec& v : run.verbs) reached = reached || v.ctx_after - v.ctx_before == c.budget;
				CHECK_MSG(reached, "[4.X2 %s, hook %d] no call admitted M = %d rows", c.name, hook, c.budget);
			}
		}
	}
}

// ==========================================================================================================
// 5. Failure paths
// ==========================================================================================================

// 5.X1 [J1-X; XE21] An allocation failure in the pass (the pass bad-alloc slot): nothing is committed, the
// pages mapped for the call go back (page size 2), the blob is byte-equal, and the retry equals off.
void Cell_X_5_1() {
	if (!XLive("5.X1")) return;
	const Sfd& f = GetSfd();
	for (int32_t order : {0, 2}) {
		XSpec sp;
		sp.page = 2;
		sp.order = order;
		const XOff off = RunOff(f, sp);
		const std::vector<size_t> ps = XForcedPositions(f, sp.schema, off, 2);
		if (ps.empty()) continue;
		XPlan plan;
		plan.kind = XDraft::kPerfect;
		plan.budget = 4;
		bool fired = false;
		XObs before;
		plan.before_verb = [&](size_t pos, XEnv& e, sslm_seq& s) {
			if (pos != ps[0]) return;
			before = XObserve(*e.rig, s, 0, 0);
			superslm_test::g_inject_jump_forward_pass_alloc_failure = true;
		};
		plan.after_verb = [&](size_t pos, XEnv& e, sslm_seq& s) {
			if (pos != ps[0]) return;
			fired = !superslm_test::g_inject_jump_forward_pass_alloc_failure;
			superslm_test::g_inject_jump_forward_pass_alloc_failure = false;
			const XObs now = XObserve(*e.rig, s, 0, 0);
			CHECK_MSG(XDiff(now, before).empty(), "[5.X1 %s] the failed call changed the sequence: %s", sp.Name().c_str(),
			          XDiff(now, before).c_str());
		};
		plan.allow_status = [](sslm_status st) { return st == SSLM_ALLOCATION_FAILED; };
		const XRun run = RunExact(f, sp, off, plan);
		X_EQUALS_OFF("5.X1", run, sp, plan);
		CHECK_MSG(fired, "[5.X1 %s] the pass bad-alloc slot was not reached", sp.Name().c_str());
		bool failed_once = false;
		for (const XVerbRec& v : run.verbs) failed_once = failed_once || v.pos == ps[0];
		CHECK_MSG(failed_once, "[5.X1 %s] no call at position %zu", sp.Name().c_str(), ps[0]);
	}
}

// 5.X2 [J1-X; XE20] Limits inside the draft, on K/V-budget holders and on a cap holder (a long prompt), page size
// 2. The verb returns SSLM_OK wherever the limit falls (at h = 0 it is a no-op for a NULL, an empty and a
// non-empty draft, with nothing changed); the decode after it returns the limit status off returns; no check
// row reads at or past h (the check's S-ROWS entries number at most h - 1); the run, driven by a host loop that
// stops on any non-OK status, equals off.
void Cell_X_5_2() {
	if (!XLive("5.X2")) return;
	const Sfd& f = GetSfd();
	std::vector<XSpec> specs;
	for (int32_t extra : {1, 2, 3, 4, 5, 7, 10}) {
		XSpec sp;
		sp.page = 2;
		sp.holder_budget = int32_t(f.prompt_potion.size()) + extra;
		specs.push_back(sp);
	}
	static std::vector<int32_t> long_prompts[3];
	const int32_t cap_room[3] = {2, 4, 7};
	for (int i = 0; i < 3; ++i) {
		long_prompts[i].clear();
		while (long_prompts[i].size() + f.prompt_potion.size() <= size_t(kCap - cap_room[i]))
			long_prompts[i].insert(long_prompts[i].end(), f.prompt_potion.begin(), f.prompt_potion.end());
		while (long_prompts[i].size() < size_t(kCap - cap_room[i])) long_prompts[i].push_back(f.prompt_potion.back());
		XSpec sp;
		sp.page = 2;
		sp.prompt = &long_prompts[i];
		specs.push_back(sp);
	}
	for (const XSpec& sp : specs) {
		const XOff off = RunOff(f, sp);
		const bool limited = !off.obs.empty() && off.obs.back().st != SSLM_OK;
		CHECK_MSG(limited, "[5.X2 %s] setup: off did not end at a limit (last status %d)", sp.Name().c_str(),
		          off.obs.empty() ? -1 : off.obs.back().st);
		for (XDraft kind : {XDraft::kNull, XDraft::kEmpty, XDraft::kPerfect, XDraft::kWrongAt}) {
			XPlan plan;
			plan.kind = kind;
			plan.budget = 8;
			const XRun run = RunExact(f, sp, off, plan);
			X_EQUALS_OFF("5.X2", run, sp, plan);
			for (const XVerbRec& v : run.verbs) {
				size_t checks = 0;
				for (bool c : v.rows_check) checks += c ? 1 : 0;
				// h from the holder's limit and the context before the call (§5.10.3): the draft clipped to
				// chunk_budget - 1 and to the headroom less the pending row. k is taken after the clips (the
				// conductor's G-4 ruling), so k <= 0 is a no-op: no rows, no check row, the blob unchanged.
				const int64_t headroom = v.limit_before - v.ctx_before;
				const int64_t k_draft = v.null_draft ? int64_t(v.budget) - 1 : int64_t(v.draft.size());
				const int64_t k = std::min({k_draft, int64_t(v.budget) - 1, headroom - 1});
				const int64_t admitted = v.ctx_after - v.ctx_before;
				if (k <= 0) {
					CHECK_MSG(checks == 0 && admitted == 0 && v.blob_after == v.blob_before,
					          "[5.X2 %s, %s] the call at %zu (headroom %lld, k after the clips %lld) must be a no-op: %zu "
					          "check rows, %lld admitted, blob %s", sp.Name().c_str(), XDraftName(kind), v.pos,
					          static_cast<long long>(headroom), static_cast<long long>(k), checks,
					          static_cast<long long>(admitted), v.blob_after == v.blob_before ? "unchanged" : "changed");
				} else {
					CHECK_MSG(int64_t(checks) <= k && admitted <= 1 + k,
					          "[5.X2 %s, %s] the call at %zu: %zu check rows and %lld rows admitted; h = 1 + k = %lld "
					          "(headroom %lld), so at most h - 1 check rows", sp.Name().c_str(), XDraftName(kind), v.pos,
					          checks, static_cast<long long>(admitted), static_cast<long long>(1 + k),
					          static_cast<long long>(headroom));
				}
			}
		}
	}
}

// 5.X3 [J1-X; XE21] A FinishRow that fails at check row r: (i) a hostile finish hook (ParallelForIncomplete, the
// finish's legal-row split under the hook, bits 0) and (iii) a damped-scorer refusal (S-FINISHEXIT's damped
// exits, skip r). Rows 0..r are committed and the verb returns OK. S-SATROW marks a rejected row; page size 2.
// One-shot: the next decode equals off's never-failed decode. Persistent: the next decode returns the status
// off returns at that position with an equal blob (§4.5), and with the failure gone the retry equals off.
// (ii), an allocation failure in the no-workspace damped path, is in the allocation sweep (5.X3 (ii)).
void XCell53Rekey(const Sfd& f);  // the persistent arms across an anti-LM change, below
void Cell_X_5_3() {
	if (!XLive("5.X3")) return;
	const Sfd& f = GetSfd();
	for (int path = 0; path < 3; ++path) {  // 0 the hook; 1 damped refused; 2 damped !ok
		XSpec sp;
		sp.page = 2;
		sp.order = path == 0 ? 0 : 2;
		const XOff off = RunOff(f, sp);
		const std::vector<size_t> ps = XForcedPositions(f, sp.schema, off, 4);
		if (ps.empty()) continue;
		const size_t p = ps[0];
		std::mt19937 g0(1);
		const std::vector<int32_t> d = XMakeDraft(f, sp.schema, off, p, XDraft::kPerfect, 0, 4, g0, nullptr);
		for (int32_t r = 0; r < int32_t(d.size()); ++r) {
			for (bool persistent : {false, true}) {
				const std::string tag = std::string(path == 0 ? "hook" : path == 1 ? "refused" : "!ok") + ", r " +
				                        std::to_string(r) + (persistent ? ", persistent" : ", one-shot");
				// The reference, built the same way as the run (a fresh rig, off's blob at p restored, so the holder's
				// pages compare): off decodes r times to the run's position, then the decode the run makes after the
				// call (with the same failure armed when persistent), then, persistent, the retry with it gone.
				XObs ref1, ref2;
				bool ref_ok = false;
				{
					XRestoreRig r2(f, sp);
					CountingHook h2;
					h2.max_tasks = 4;
					sslm_seq q = r2.ok() ? r2.At(off, p) : nullptr;
					if (q && (path != 0 || Install(r2.env->rig->W(), &h2, 0))) {
						const sslm_decode_params vp0 = XParams(*r2.env->rig, sp.order);
						bool good = true;
						for (int32_t i = 0; i < r && good; ++i) {
							int32_t u = -100;
							good = r2.env->rig->Decode1(q, vp0, r2.env->W(sp), &u) == SSLM_OK && u >= 0;
						}
						if (good) {
							if (persistent) {
								if (path == 0) {
									h2.Clear();
									h2.hostile = Hostile::kOmit;
									h2.hostile_call = 0;
									h2.hostile_persistent = true;
								} else {
									sfd::ArmFinishExit(path == 1 ? sfd::FinishExit::kDampedRefused : sfd::FinishExit::kDampedNotOk, 0);
								}
							}
							int32_t u = -100;
							const sslm_status us = r2.env->rig->Decode1(q, vp0, r2.env->W(sp), &u);
							ref1 = XObserve(*r2.env->rig, q, static_cast<int>(us), u);
							if (persistent) {
								h2.Clear();
								h2.hostile = Hostile::kNone;
								sfd::DisarmFinishExit();
								int32_t u2 = -100;
								const sslm_status us2 = r2.env->rig->Decode1(q, vp0, r2.env->W(sp), &u2);
								ref2 = XObserve(*r2.env->rig, q, static_cast<int>(us2), u2);
							}
							ref_ok = true;
						}
						if (path == 0) Install(r2.env->rig->W(), nullptr, 0);
					}
					if (q) r2.env->rig->Release(q);
				}
				CHECK_MSG(ref_ok, "[5.X3 %s] setup: the off reference did not build", tag.c_str());
				if (!ref_ok) continue;
				XRestoreRig rr(f, sp);
				if (!rr.ok()) continue;
				CountingHook h;
				h.max_tasks = 4;
				if (path == 0 && !Install(rr.env->rig->W(), &h, 0)) continue;
				const sslm_decode_params vp = XParams(*rr.env->rig, sp.order);
				sslm_seq s = rr.At(off, p);
				if (!s) continue;
				auto arm = [&](bool on) {
					if (path == 0) {
						h.Clear();
						h.hostile = on ? Hostile::kOmit : Hostile::kNone;
						h.hostile_call = r;
						h.hostile_persistent = persistent;
					} else if (on) {
						sfd::ArmFinishExit(path == 1 ? sfd::FinishExit::kDampedRefused : sfd::FinishExit::kDampedNotOk, r);
					} else {
						sfd::DisarmFinishExit();
					}
				};
				arm(true);
				sfd::ArmSatRow(int32_t(d.size()));  // the last row: rejected (or never kept) once the check stops at r
				const sslm_jump_options o = XExact(&d);
				const Jump j = XJump(rr.env->m, s, int32_t(d.size()) + 1, XVerbParams(sp, vp), &o, rr.env->W(sp));
				sfd::DisarmSatRow();
				CHECK_MSG(j.st == SSLM_OK && j.count == r, "[5.X3 %s] the verb: status %d count %d; want OK with rows 0..%d "
				          "committed (count %d)", tag.c_str(), static_cast<int>(j.st), j.count, r, r);
				if (!persistent) arm(false);
				else if (path == 0) {
					h.Clear();  // the next decode's first finish call is call 0 again: still hostile
					h.hostile_call = 0;
				} else {
					sfd::DisarmFinishExit();
					sfd::ArmFinishExit(path == 1 ? sfd::FinishExit::kDampedRefused : sfd::FinishExit::kDampedNotOk, 0);
				}
				int32_t t = -100;
				const sslm_status st = rr.env->rig->Decode1(s, vp, rr.env->W(sp), &t);
				const XObs ob = XObserve(*rr.env->rig, s, static_cast<int>(st), t);
				// Pages and stats too (kXCmpAll): the rows the check stopped short of had pages mapped (XE21).
				if (!persistent) {
					CHECK_MSG(XDiff(ob, ref1).empty(), "[5.X3 %s] the next decode against off's never-failed decode: %s",
					          tag.c_str(), XDiff(ob, ref1).c_str());
				} else {
					CHECK_MSG(st != SSLM_OK && XDiff(ob, ref1).empty(),
					          "[5.X3 %s] the failing decode: status %d; off's at that position %d; %s", tag.c_str(),
					          static_cast<int>(st), ref1.st, XDiff(ob, ref1).c_str());
					arm(false);
					int32_t t2 = -100;
					const sslm_status st2 = rr.env->rig->Decode1(s, vp, rr.env->W(sp), &t2);
					const XObs ob2 = XObserve(*rr.env->rig, s, static_cast<int>(st2), t2);
					CHECK_MSG(XDiff(ob2, ref2).empty(), "[5.X3 %s] the retry with the failure gone: %s", tag.c_str(),
					          XDiff(ob2, ref2).c_str());
				}
				arm(false);
				if (path == 0) Install(rr.env->rig->W(), nullptr, 0);
				rr.env->rig->Release(s);
			}
		}
	}
	XCell53Rekey(f);
}

// 5.X3, the persistent arms across an anti-LM change (code review TF/J1-X round 1, finding 1). A finish that fails
// persistently from exact's check row 0, with the verb's damped params naming (a) a different order than the live
// state's, or (b) a sequence's first damped call after greedy decodes. Off makes the same decodes without the verb:
// under TF its failed decode installs nothing (C16), so right after the failed decode both arms must rest with equal
// blobs (C12), and the retry at the original order, the failure gone, must agree token for token and blob for blob.
// `hostile`: the finish fails through a hook that omits a task (bits 0: only the finish reads it, the batched layer
// pass); else through S-FINISHEXIT's damped-refused exit from finish 0. `draft` null: the forced chain (NULL draft).
void XRekeyArm(const Sfd& f, const XSpec& sp, size_t p, const std::vector<int32_t>* draft, int32_t to_order,
               bool hostile, const char* leg) {
	const std::string tag = std::string("5.X3 persistent, ") + leg + ", " + sp.Name() + ", at " + std::to_string(p) +
	                        ", verb at " + (to_order ? "damped@" + std::to_string(to_order) : std::string("greedy"));
	const XOff off = RunOff(f, sp);
	CHECK_MSG(off.ok && p < off.obs.size(), "[%s] setup: off has %zu decodes", tag.c_str(), off.obs.size());
	if (!off.ok || p >= off.obs.size()) return;
	XRestoreRig ro(f, sp), rx(f, sp);
	if (!ro.ok() || !rx.ok()) {
		CHECK_MSG(false, "[%s] setup: the rigs did not build", tag.c_str());
		return;
	}
	CountingHook ho, hx;
	ho.max_tasks = hx.max_tasks = 4;
	sslm_seq so = ro.At(off, p), sx = rx.At(off, p);
	if (!so || !sx) {
		CHECK_MSG(false, "[%s] setup: restoring off's blob failed", tag.c_str());
		if (so) ro.env->rig->Release(so);
		if (sx) rx.env->rig->Release(sx);
		return;
	}
	int32_t chain = 0, acc = 0;
	sslm_seq_forced_tokens(rx.env->m, sx, nullptr, 0, &chain, &acc);
	CHECK_MSG(draft || chain >= 1, "[%s] setup: no forced chain after the pending token (%d)", tag.c_str(), chain);
	auto arm = [&](bool on) {
		if (hostile) {
			for (CountingHook* h : {&ho, &hx}) {
				h->Clear();
				h->hostile = on ? Hostile::kOmit : Hostile::kNone;
				h->hostile_call = 0;
				h->hostile_persistent = on;
			}
			if (on) {
				Install(ro.env->rig->W(), &ho, 0);
				Install(rx.env->rig->W(), &hx, 0);
			} else {
				Install(ro.env->rig->W(), nullptr, 0);
				Install(rx.env->rig->W(), nullptr, 0);
			}
		} else if (on) {
			sfd::ArmFinishExit(sfd::FinishExit::kDampedRefused, 0);
		} else {
			sfd::DisarmFinishExit();
		}
	};
	const sslm_decode_params vto = XParams(*rx.env->rig, to_order);
	const sslm_decode_params vfrom = XParams(*rx.env->rig, sp.order);
	// The exact arm's verb, with the failure armed from its first finish (check row 0).
	arm(true);
	const sslm_jump_options o = XExact(draft);
	const int32_t budget = draft ? int32_t(draft->size()) + 1 : 8;
	const Jump j = XJump(rx.env->m, sx, budget, to_order ? &vto : nullptr, &o, rx.env->W(sp));
	CHECK_MSG(j.st == SSLM_OK && j.count == 0, "[%s] the verb: status %d count %d; want OK with nothing confirmed (the "
	          "check failed at row 0)", tag.c_str(), static_cast<int>(j.st), j.count);
	if (!hostile) {  // re-arm so the next decode's finish is finish 0 again, in both arms alike
		sfd::DisarmFinishExit();
		sfd::ArmFinishExit(sfd::FinishExit::kDampedRefused, 0);
	} else {
		ho.Clear();
		hx.Clear();
	}
	// Both arms: the failing decode at the verb's params. Off installs nothing (TF); exact must rest the same.
	int32_t to = -100, tx = -100;
	const sslm_status sto = ro.env->rig->Decode1(so, vto, ro.env->W(sp), &to);
	const sslm_status stx = rx.env->rig->Decode1(sx, vto, rx.env->W(sp), &tx);
	const XObs fo = XObserve(*ro.env->rig, so, static_cast<int>(sto), to);
	const XObs fx = XObserve(*rx.env->rig, sx, static_cast<int>(stx), tx);
	CHECK_MSG(sto != SSLM_OK, "[%s] setup: off's decode did not fail (status %d)", tag.c_str(), static_cast<int>(sto));
	const std::string fd = XDiff(fx, fo, kXCmpTok | kXCmpBlob);
	CHECK_MSG(fd.empty(), "[%s] right after the failed decode, exact against off: %s", tag.c_str(), fd.c_str());
	// The failure gone, the retry at the original order, to the end of the stream (or 24 decodes).
	arm(false);
	for (int i = 0; i < 24; ++i) {
		int32_t uo = -100, ux = -100;
		const sslm_status ao = ro.env->rig->Decode1(so, vfrom, ro.env->W(sp), &uo);
		const sslm_status ax = rx.env->rig->Decode1(sx, vfrom, rx.env->W(sp), &ux);
		const std::string rd = XDiff(XObserve(*rx.env->rig, sx, static_cast<int>(ax), ux),
		                             XObserve(*ro.env->rig, so, static_cast<int>(ao), uo), kXCmpTok | kXCmpBlob);
		CHECK_MSG(rd.empty(), "[%s] retry decode %d at the original order, exact against off: %s", tag.c_str(), i,
		          rd.c_str());
		if (!rd.empty() || ao != SSLM_OK || uo < 0) break;
	}
	arm(false);
	ro.env->rig->Release(so);
	rx.env->rig->Release(sx);
}

void XCell53Rekey(const Sfd& f) {
	// The review's failing input (rekey.cpp): potion, its prompt, a workspace, 3 damped decodes at order 2, a hook
	// that fails the finish, the verb at damped order 3 with a NULL draft, budget 8 (batched: bits 0).
	static const std::vector<int32_t> kReviewPrompt = {11, 523, 77, 1960, 300, 42};
	{
		XSpec sp;
		sp.prompt = &kReviewPrompt;
		sp.order = 2;
		XRekeyArm(f, sp, 2, nullptr, 3, true, "(a) order change, the review's input, batched");
	}
	for (int split = 0; split < 2; ++split) {
		for (int c = 0; c < 2; ++c) {  // (a) order 2 -> 3; (b) greedy -> first damped call at order 2
			XSpec sp;
			sp.order = c == 0 ? 2 : 0;
			sp.hook = split ? 1 : 0;  // hook 1: bit 1 at max_tasks 4, Q6 C's one-row split for M in [2, 7]
			const XOff off = RunOff(f, sp);
			const std::vector<size_t> ps = XForcedPositions(f, sp.schema, off, 4);
			CHECK_MSG(!ps.empty(), "[5.X3 persistent %s] no position with a forced run of 4", sp.Name().c_str());
			if (ps.empty()) continue;
			std::mt19937 g(1);
			const std::vector<int32_t> d = XMakeDraft(f, sp.schema, off, ps[0], XDraft::kPerfect, 0, 4, g, nullptr);
			CHECK_MSG(d.size() >= 1 && d.size() + 1 <= kQ6cSplitMax, "[5.X3 persistent %s] the draft has %zu tokens",
			          sp.Name().c_str(), d.size());
			const std::string leg = std::string(c == 0 ? "(a) order change" : "(b) first damped call") +
			                        (split ? ", split" : ", batched");
			XRekeyArm(f, sp, ps[0], &d, c == 0 ? 3 : 2, false, leg.c_str());
		}
	}
}

// 5.X4 [J1-X] A page-map refusal (an empty reserve, through the existing drain seam), page size 2: nothing is
// committed, the status is canonical mode's for the same call, and the pages and counters equal their values
// before the call.
void Cell_X_5_4() {
	if (!XLive("5.X4")) return;
	const Sfd& f = GetSfd();
	XSpec sp;
	sp.page = 2;
	const XOff off = RunOff(f, sp);
	const std::vector<size_t> ps = XForcedPositions(f, sp.schema, off, 2);
	size_t p = SIZE_MAX;
	for (size_t q : ps)
		if (ParseBlob(off.obs[q].blob).ctx % 2 == 0 && p == SIZE_MAX) p = q;  // the pending row needs a new page
	CHECK_MSG(p != SIZE_MAX, "[5.X4] no forced position whose pending row starts a page");
	if (p == SIZE_MAX) return;
	std::mt19937 g(1);
	const std::vector<int32_t> d = XMakeDraft(f, sp.schema, off, p, XDraft::kPerfect, 0, 3, g, nullptr);
	sslm_status canon = SSLM_OK;
	{
		XRestoreRig rr(f, sp);
		sslm_seq s = rr.ok() ? rr.At(off, p) : nullptr;
		if (s && sslm_pkv_test_only_drain_reserve(s) == SSLM_OK)
			canon = JumpForward(rr.env->m, s, int32_t(d.size()) + 1, nullptr, rr.env->W(sp)).st;
	}
	XRestoreRig rr(f, sp);
	sslm_seq s = rr.ok() ? rr.At(off, p) : nullptr;
	if (!s || sslm_pkv_test_only_drain_reserve(s) != SSLM_OK) return;
	const XObs before = XObserve(*rr.env->rig, s, 0, 0);
	const sslm_jump_options o = XExact(&d);
	const Jump j = XJump(rr.env->m, s, int32_t(d.size()) + 1, nullptr, &o, rr.env->W(sp));
	const XObs after = XObserve(*rr.env->rig, s, 0, 0);
	CHECK_MSG(canon != SSLM_OK && j.st == canon && XDiff(after, before).empty(), "[5.X4] status %d (canonical %d); %s",
	          static_cast<int>(j.st), static_cast<int>(canon), XDiff(after, before).c_str());
}

// 5.X5 [J1-X] S-BOUND clears the exactness predicate: the exact verb is a no-op (OK, count 0) and the run
// equals off.
void Cell_X_5_5() {
	if (!XLive("5.X5")) return;
	const Sfd& f = GetSfd();
	XSpec sp;
	sp.full_finish = true;
	const XOff off = RunOff(f, sp);
	XPlan plan;
	plan.kind = XDraft::kPerfect;
	const XRun run = RunExact(f, sp, off, plan);
	X_EQUALS_OFF("5.X5", run, sp, plan);
	for (const XVerbRec& v : run.verbs)
		CHECK_MSG(v.j.count == 0 && v.blob_after == v.blob_before, "[5.X5] the call at %zu was not a no-op (count %d)",
		          v.pos, v.j.count);
}

// 5.X6 [J1-X; XE21, XE27] Q6 C's split, a pass that fails at row r >= 1. Bit 1 (max_tasks 4), damped@2, page
// size 2; a hostile hook fails one hook call of the verb, swept over every call it makes. A failure in a one-row
// pass r >= 1 commits rows 0..r - 1 (r = 1: p alone) and returns OK; a failure in a check finish commits rows
// 0..r; a failure in row 0's pass commits nothing and returns canonical mode's status. Each way, the anti-LM
// history and order right after the call equal off's at that position (no token fed past the commit), and the
// run (pages and counters included) equals off.
void Cell_X_5_6() {
	if (!XLive("5.X6")) return;
	const Sfd& f = GetSfd();
	XSpec sp;
	sp.page = 2;
	sp.order = 2;
	sp.hook = 1;
	const XOff off = RunOff(f, sp);
	const std::vector<size_t> ps = XForcedPositions(f, sp.schema, off, 4);
	if (ps.empty()) return;
	const size_t p = ps[0];
	// the honest call's hook calls
	int calls = 0;
	{
		XPlan plan;
		plan.kind = XDraft::kPerfect;
		plan.budget = 5;
		plan.schedule = [&](size_t pos, std::mt19937&, XPlan*) { return pos == p; };
		plan.before_verb = [&](size_t, XEnv& e, sslm_seq&) { e.hook.Clear(); };
		plan.after_verb = [&](size_t, XEnv& e, sslm_seq&) { calls = int(e.hook.Counts().size()); };
		const XRun run = RunExact(f, sp, off, plan);
		X_EQUALS_OFF("5.X6 honest", run, sp, plan);
	}
	CHECK_MSG(calls > 0, "[5.X6] the honest call made no hook call (the split did not act)");
	std::set<int32_t> committed;
	for (int c = 0; c < calls; ++c) {
		XPlan plan;
		plan.kind = XDraft::kPerfect;
		plan.budget = 5;
		plan.schedule = [&](size_t pos, std::mt19937&, XPlan*) { return pos == p; };
		plan.before_verb = [&](size_t, XEnv& e, sslm_seq&) {
			e.hook.Clear();
			e.hook.hostile = Hostile::kOmit;
			e.hook.hostile_call = c;
		};
		plan.after_verb = [&](size_t, XEnv& e, sslm_seq& s) {
			e.hook.hostile = Hostile::kNone;
			const Blob now = e.rig->State(s);
			const int64_t kept = now.ctx - ParseBlob(off.obs[p].blob).ctx;
			if (kept >= 1) {
				committed.insert(int32_t(kept - 1));
				const Blob want = ParseBlob(off.obs[p + size_t(kept) - 1].blob);
				CHECK_MSG(now.order == want.order && now.hist == want.hist, "[5.X6 call %d] right after the call (%lld rows "
				          "kept) the anti-LM state is order %d history %zu; off's at that position %d, %zu", c,
				          static_cast<long long>(kept), now.order, now.hist.size(), want.order, want.hist.size());
			}
		};
		plan.allow_status = [](sslm_status st) { return st != SSLM_OK; };
		const XRun run = RunExact(f, sp, off, plan);
		CHECK_MSG(run.ok, "[5.X6 hostile call %d of %d] exact differs from off: %s", c, calls, run.diff.c_str());
	}
	CHECK_MSG(committed.count(0) && committed.size() >= 2, "[5.X6] the sweep did not reach both an r = 1 arm (p alone "
	          "kept) and an r >= 2 arm (%zu commit shapes)", committed.size());
}

// 5.X7 [J1-X; XE26] The anti-LM prepare fails (the verb's create bad-alloc slot) on an order change: the verb
// returns the status with nothing changed, and the retry equals off (off changes order at the same position).
void Cell_X_5_7() {
	if (!XLive("5.X7")) return;
	const Sfd& f = GetSfd();
	XSpec sp;
	sp.order = 2;
	sp.order2 = 3;
	const XOff probe = RunOff(f, sp);
	const std::vector<size_t> ps = XForcedPositions(f, sp.schema, probe, 2);
	if (ps.empty()) return;
	sp.switch_at = ps[0] + 1;
	const XOff off = RunOff(f, sp);
	bool fired = false;
	XPlan plan;
	plan.kind = XDraft::kPerfect;
	plan.budget = 4;
	plan.before_verb = [&](size_t pos, XEnv&, sslm_seq&) {
		if (pos == ps[0]) superslm_test::g_inject_anti_lm_create_alloc_failure = true;
	};
	plan.after_verb = [&](size_t pos, XEnv&, sslm_seq&) {
		if (pos != ps[0]) return;
		fired = !superslm_test::g_inject_anti_lm_create_alloc_failure;
		superslm_test::g_inject_anti_lm_create_alloc_failure = false;
	};
	plan.allow_status = [](sslm_status st) { return st == SSLM_ALLOCATION_FAILED; };
	const XRun run = RunExact(f, sp, off, plan);
	X_EQUALS_OFF("5.X7", run, sp, plan);
	CHECK_MSG(fired, "[5.X7] the create slot was not reached by the call at %zu", ps[0]);
}

// ==========================================================================================================
// 8. Composition
// ==========================================================================================================

// 8.X1 [J1-X; XE13] With Q6 C under bit 1: for M <= 7, S-LAYERROWS shows exactly j + 1 one-row passes (no row
// past the first mismatch); for M >= 8, one batched pass. The run equals off and the one-thread exact run.
void Cell_X_8_1() {
	if (!XLive("8.X1")) return;
	const Sfd& f = GetSfd();
	for (const char* schema : {kPotion, kLongKey}) {
		for (int32_t budget : {4, 7, 9}) {
			for (XDraft kind : {XDraft::kPerfect, XDraft::kWrongAt}) {
				XSpec one, split;
				one.schema = split.schema = schema;
				split.hook = 1;
				const XOff off = RunOff(f, one);
				XPlan plan;
				plan.kind = kind;
				plan.budget = budget;
				const XRun r1 = RunExact(f, one, off, plan);
				const XRun rs = RunExact(f, split, RunOff(f, split), plan);
				X_EQUALS_OFF("8.X1 split", rs, split, plan);
				CHECK_MSG(r1.stream == rs.stream, "[8.X1 %s] the split run's stream differs from one thread's",
				          XWhat(split, plan).c_str());
				for (const XVerbRec& v : rs.verbs) {
					const size_t m = 1 + v.draft.size();
					if (v.draft.empty()) continue;
					const size_t j1 = size_t(std::max(v.j.count, 0)) + 1;
					bool shape = true;
					if (m <= kQ6cSplitMax) {
						shape = v.passes.size() == j1;
						for (const sfd::LayerRowsEntry& e : v.passes) shape = shape && e.kind == sfd::kLayerRowsBatched && e.rows == 1;
					} else {
						shape = v.passes.size() == 1 && v.passes[0].rows == m;
					}
					CHECK_MSG(shape, "[8.X1 %s] the call at %zu (M %zu, j %d): S-LAYERROWS %s", XWhat(split, plan).c_str(),
					          v.pos, m, v.j.count, Q6cShape(v.passes).c_str());
				}
			}
		}
	}
}

// 8.X2 [J1-X; XE29] Save and restore between calls in each resting shape (pending after a decode; ready after an
// admitting call); prefix adoption on a whole-reserve adopter; and a budget adopter after a budget prefix adopt
// (shared leading pages, page size 2: shared_pages and the mapped counts equal off's). Each equals an
// uninterrupted off run.
void Cell_X_8_2() {
	if (!XLive("8.X2")) return;
	const Sfd& f = GetSfd();
	// save and restore between calls
	{
		XSpec sp;
		const XOff off = RunOff(f, sp);
		for (int shape = 0; shape < 2; ++shape) {
			XPlan plan;
			plan.kind = XDraft::kWrongAt;
			plan.cmp = kXCmpTok | kXCmpBlob;
			auto roundtrip = [&](size_t, XEnv& e, sslm_seq& s) {
				const std::vector<uint8_t> b = e.rig->Save(s);
				e.rig->Release(s);
				s = e.rig->Restore(b);
			};
			if (shape == 0) plan.before_verb = roundtrip;
			else plan.after_verb = roundtrip;
			const XRun run = RunExact(f, sp, off, plan);
			CHECK_MSG(run.ok, "[8.X2 save/restore %s] %s", shape == 0 ? "pending" : "ready", run.diff.c_str());
		}
	}
	// prefix adoption: the prompt as a frozen prefix, adopted, then the exact loop; off adopts the same prefix
	for (int budget_mode = 0; budget_mode < 2; ++budget_mode) {
		XSpec sp;
		sp.page = budget_mode ? 2 : 0;
		XOff off;
		XRun run;
		for (int arm = 0; arm < 2; ++arm) {
			XEnv env(f, sp);
			if (!env.ok) break;
			sslm_prefix pre = nullptr;
			const sslm_status bs = budget_mode ? sslm_prefix_begin_budgeted(env.m, &env.rig->pool, kCap, &pre)
			                                   : sslm_prefix_begin(env.m, &env.rig->pool, &pre);
			sslm_schema h = nullptr;
			int32_t consumed = 0;
			const bool built = bs == SSLM_OK && sslm_schema_lookup(env.m, kPotion, &h) == SSLM_OK &&
			                   sslm_prefix_set_schema(pre, h) == SSLM_OK &&
			                   sslm_prefix_prefill(env.m, pre, f.prompt_potion.data(), int32_t(f.prompt_potion.size()),
			                                       int32_t(f.prompt_potion.size()), SSLM_SPAN_PROMPT, nullptr,
			                                       &consumed) == SSLM_OK &&
			                   sslm_prefix_freeze(pre) == SSLM_OK;
			CHECK_MSG(built, "[8.X2 adopt, %s] the prefix did not build", budget_mode ? "budget" : "whole-reserve");
			sslm_seq s = nullptr;
			if (built) {
				s = budget_mode ? env.rig->NewBudgeted(kCap, nullptr) : env.rig->NewSeq(nullptr);
				if (s && sslm_seq_adopt_prefix(s, pre) != SSLM_OK) s = nullptr;
			}
			if (s && arm == 0) off = RunOffOn(env, sp, s);
			if (s && arm == 1) {
				XPlan plan;
				plan.kind = XDraft::kWrongAt;
				run = RunExactOn(f, sp, off, plan, env, s);
			}
			if (s) env.rig->Release(s);
			if (pre) sslm_prefix_release(pre);
		}
		CHECK_MSG(run.ok, "[8.X2 adopt, %s] exact differs from off: %s", budget_mode ? "budget, page 2" : "whole-reserve",
		          run.diff.c_str());
	}
}

// 8.X3 [J1-X] A decode batch of four: one in exact mode between verb calls, one canonical (drained), one off, one
// unbound. Each equals its solo run.
void Cell_X_8_3() {
	if (!XLive("8.X3")) return;
	const Sfd& f = GetSfd();
	XSpec sp;
	const XOff off = RunOff(f, sp);
	Rig rig(f.m1->m, 1, 8, 4);
	sslm_seq s[4] = {rig.NewSeq(kPotion), rig.NewSeq(kPotion), rig.NewSeq(kPotion), rig.NewSeq(nullptr)};
	for (sslm_seq q : s)
		if (!q || !rig.PromptOk(q, f.prompt_potion)) return;
	// the solo runs of the canonical and unbound sequences
	std::vector<int32_t> solo_canon, solo_unbound;
	{
		Rig r2(f.m1->m);
		sslm_seq c = r2.NewSeq(kPotion);
		sslm_seq u = r2.NewSeq(nullptr);
		if (!c || !u || !r2.PromptOk(c, f.prompt_potion) || !r2.PromptOk(u, f.prompt_potion)) return;
		JfOpts o = GreedyOpts(r2);
		solo_canon = RunJf(r2.m, c, o).stream;
		for (int i = 0; i < 40; ++i) {
			int32_t t = -100;
			if (r2.Decode1(u, r2.Greedy(), r2.W(), &t) != SSLM_OK || t < 0) break;
			solo_unbound.push_back(t);
		}
	}
	std::vector<int32_t> got[4];
	bool done[4] = {false, false, false, false};
	const sslm_decode_params g = rig.Greedy();
	for (int step = 0; step < 300; ++step) {
		// canonical: drain before each decode
		if (!done[1]) {
			for (int guard = 0; guard < 16; ++guard) {
				if (QueryForced(rig.m, s[1], 0).count == 0) break;
				const Jump j = JumpForward(rig.m, s[1], 64, nullptr, rig.W());
				got[1].insert(got[1].end(), j.forced.begin(), j.forced.end());
				if (j.st != SSLM_OK) {
					done[1] = true;
					break;
				}
			}
		}
		std::vector<sslm_seq> batch;
		std::vector<int> who;
		for (int i = 0; i < 4; ++i)
			if (!done[i]) {
				batch.push_back(s[i]);
				who.push_back(i);
			}
		if (batch.empty()) break;
		std::vector<int32_t> out(batch.size(), -100);
		if (sslm_decode_step_v2(rig.m, batch.data(), int32_t(batch.size()), &g, rig.W(), out.data()) != SSLM_OK) break;
		for (size_t k = 0; k < batch.size(); ++k) {
			const int i = who[k];
			if (out[k] < 0) {
				done[i] = true;  // the sequence's stream ended
				continue;
			}
			got[i].push_back(out[k]);
		}
		if (got[3].size() >= solo_unbound.size()) done[3] = true;  // the unbound run is compared over its solo length
		// exact: one verb call after its decode
		if (!done[0]) {
			const std::vector<int32_t> pos_stream = got[0];
			if (!pos_stream.empty() && pos_stream.size() <= off.stream.size()) {
				std::mt19937 gg(1);
				const std::vector<int32_t> d =
				    XMakeDraft(f, kPotion, off, pos_stream.size() - 1, XDraft::kWrongAt, 1, 4, gg, nullptr);
				const sslm_jump_options o = XExact(&d);
				const Jump j = XJump(rig.m, s[0], 5, nullptr, &o, rig.W());
				got[0].insert(got[0].end(), j.forced.begin(), j.forced.end());
			}
		}
	}
	CHECK_MSG(got[0] == off.stream, "[8.X3] the exact sequence's stream %s; its solo (off) %s", Str(got[0]).c_str(),
	          Str(off.stream).c_str());
	CHECK_MSG(got[1] == solo_canon, "[8.X3] the canonical sequence's stream differs from its solo run");
	CHECK_MSG(got[2] == off.stream, "[8.X3] the off sequence's stream differs from its solo run");
	CHECK_MSG(std::vector<int32_t>(got[3].begin(), got[3].begin() + long(std::min(got[3].size(), solo_unbound.size()))) ==
	              solo_unbound,
	          "[8.X3] the unbound sequence's stream differs from its solo run");
}

// 8.X4 [J1-X] An adapter-bound sequence equals off with the same adapter.
void Cell_X_8_4() {
	if (!XLive("8.X4")) return;
	const Sfd& f = GetSfd();
	const std::vector<uint8_t> bytes = BuildAdapterBytes(f);
	if (bytes.empty()) return;
	std::unique_ptr<Aligned> ab = std::make_unique<Aligned>(bytes.size());
	std::memcpy(ab->p, bytes.data(), bytes.size());
	sslm_adapter ad = nullptr;
	if (sslm_adapter_map(ab->p, bytes.size(), f.m1->m, &ad) != SSLM_OK || !ad) return;
	XSpec sp;
	sp.adapter = ad;
	const XOff off = RunOff(f, sp);
	for (XDraft kind : {XDraft::kPerfect, XDraft::kWrongAt}) {
		XPlan plan;
		plan.kind = kind;
		const XRun run = RunExact(f, sp, off, plan);
		X_EQUALS_OFF("8.X4", run, sp, plan);
	}
	sslm_adapter_release(ad);
}

// 8.X5 [J1-X] Modes mixed on one sequence: canonical calls first (the §5.6 drain for the first spans), then exact
// calls. From the first exact call on, the run equals jump-forward off continued from the same state.
void Cell_X_8_5() {
	if (!XLive("8.X5")) return;
	const Sfd& f = GetSfd();
	XSpec sp;
	XEnv env(f, sp);
	if (!env.ok) return;
	sslm_seq s = env.NewSeq(f, sp);
	if (!s) return;
	// canonical: drain, decode, drain
	for (int i = 0; i < 2; ++i) {
		for (int guard = 0; guard < 16 && QueryForced(env.m, s, 0).count > 0; ++guard)
			if (JumpForward(env.m, s, 64, nullptr, env.rig->W()).st != SSLM_OK) return;
		int32_t t = -100;
		if (env.rig->Decode1(s, env.rig->Greedy(), env.rig->W(), &t) != SSLM_OK || t < 0) return;
	}
	const std::vector<uint8_t> mid = env.rig->Save(s);
	// off continued from that state
	XOff off;
	{
		XEnv e2(f, sp);
		sslm_seq q = e2.rig->Restore(mid);
		off = RunOffOn(e2, sp, q);
	}
	// exact from the same state: the pending token is off's position -1, so the first exact call drafts off[0..]
	XOff shifted = off;
	{
		XEnv e3(f, sp);
		sslm_seq q = e3.rig->Restore(mid);
		int32_t pending = ParseBlob(mid).current;
		shifted.stream.insert(shifted.stream.begin(), pending);
		shifted.walk_after.insert(shifted.walk_after.begin(), ParseBlob(mid).walk);
		shifted.obs.insert(shifted.obs.begin(), XObserve(*e3.rig, q, 0, pending));
		std::mt19937 g(1);
		const std::vector<int32_t> d = XMakeDraft(f, sp.schema, shifted, 0, XDraft::kWrongAt, 1, 4, g, nullptr);
		const sslm_jump_options o = XExact(&d);
		const Jump j = XJump(e3.m, q, 5, nullptr, &o, e3.rig->W());
		CHECK_MSG(j.st == SSLM_OK, "[8.X5] the first exact call after canonical ones: status %d", static_cast<int>(j.st));
		XOff rest = off;
		rest.obs.erase(rest.obs.begin(), rest.obs.begin() + std::min<long>(std::max(j.count, 0), long(rest.obs.size())));
		rest.stream.erase(rest.stream.begin(), rest.stream.begin() + std::min<long>(std::max(j.count, 0), long(rest.stream.size())));
		rest.walk_after.erase(rest.walk_after.begin(),
		                      rest.walk_after.begin() + std::min<long>(std::max(j.count, 0), long(rest.walk_after.size())));
		XPlan plan;
		plan.kind = XDraft::kWrongAt;
		plan.cmp = kXCmpTok | kXCmpBlob;
		const XRun run = RunExactOn(f, sp, rest, plan, e3, q);
		X_EQUALS_OFF("8.X5", run, sp, plan);
	}
}

// 8.X6 [J1-X; XE10, XE11] Damped anti-LM. (a) After every decode_step the history equals off's, at one order and
// after an order change between calls (the blob carries both). (b) On a sequence holding an order-3 history, a
// refused call, a no-op call and a budget-1 call at order 2 each leave the history and order right after the verb
// equal to the history before the call. By the conductor's G-4 ruling (k is taken after the clips) the budget-1
// call is a no-op, so no exact call has h = 1; that arm is the clipped no-op, and every arm leaves the blob equal.
void Cell_X_8_6() {
	if (!XLive("8.X6")) return;
	const Sfd& f = GetSfd();
	for (int arm = 0; arm < 2; ++arm) {
		XSpec sp;
		sp.order = 2;
		if (arm == 1) {
			sp.order2 = 3;
			sp.switch_at = 6;
		}
		const XOff off = RunOff(f, sp);
		for (XDraft kind : {XDraft::kPerfect, XDraft::kWrongAt, XDraft::kNull}) {
			XPlan plan;
			plan.kind = kind;
			const XRun run = RunExact(f, sp, off, plan);
			X_EQUALS_OFF("8.X6 (a)", run, sp, plan);
		}
	}
	XSpec sp;
	sp.order = 3;
	const XOff off = RunOff(f, sp);
	const std::vector<size_t> ps = XForcedPositions(f, sp.schema, off, 2);
	if (ps.empty()) return;
	const size_t p = ps[0];
	XRestoreRig rr(f, sp);
	if (!rr.ok()) return;
	const sslm_decode_params p2 = rr.env->rig->Damped(2);
	std::mt19937 g(1);
	const std::vector<int32_t> d = XMakeDraft(f, sp.schema, off, p, XDraft::kPerfect, 0, 2, g, nullptr);
	const std::vector<int32_t> empty;
	struct Case {
		const char* name;
		sslm_jump_options o;
		int32_t budget;
	};
	const Case cases[] = {{"refused", XOpts(SSLM_JUMP_EXACT, nullptr, 0, 0), 8},
	                      {"no-op (empty draft)", XExact(&empty), 8},
	                      {"budget 1 (k clipped to 0: a no-op)", XExact(&d), 1}};
	for (const Case& c : cases) {
		sslm_seq s = rr.At(off, p);
		if (!s) continue;
		const std::vector<uint8_t> blob_before = rr.env->rig->Save(s);
		const Blob before = rr.env->rig->State(s);
		const Jump j = XJump(rr.env->m, s, c.budget, &p2, &c.o, rr.env->W(sp));
		const Blob after = rr.env->rig->State(s);
		CHECK_MSG(before.order == 3 && after.order == before.order && after.hist == before.hist,
		          "[8.X6 (b) %s] right after the call: order %d -> %d, history %s -> %s", c.name, before.order, after.order,
		          Str(before.hist).c_str(), Str(after.hist).c_str());
		// Each call changes nothing (the conductor's G-4 ruling makes the budget-1 call a no-op, as the empty draft):
		// the whole blob is byte-equal, and the two non-refused calls return OK with count 0.
		CHECK_MSG(rr.env->rig->Save(s) == blob_before && (std::string(c.name) == "refused" || (j.st == SSLM_OK && j.count == 0)),
		          "[8.X6 (b) %s] the call: status %d count %d, blob %s", c.name, static_cast<int>(j.st), j.count,
		          rr.env->rig->Save(s) == blob_before ? "unchanged" : "changed");
		rr.env->rig->Release(s);
	}
}

// 8.X7 [J1-X] The check and the decode step share FinishRow: S-ROWS records, for each check row r, n legal rows
// (the legal row's size at the walk state after row r's token), as a decode finish does there.
void Cell_X_8_7() {
	if (!XLive("8.X7")) return;
	const Sfd& f = GetSfd();
	XSpec sp;
	const XOff off = RunOff(f, sp);
	const std::vector<size_t> ps = XForcedPositions(f, sp.schema, off, 3);
	if (ps.empty()) return;
	const size_t p = ps[0];
	XRestoreRig rr(f, sp);
	if (!rr.ok()) return;
	std::mt19937 g(1);
	const std::vector<int32_t> d = XMakeDraft(f, sp.schema, off, p, XDraft::kPerfect, 0, 3, g, nullptr);
	sslm_seq s = rr.At(off, p);
	if (!s) return;
	sfd::RowsLogReset();
	const sslm_jump_options o = XExact(&d);
	const Jump j = XJump(rr.env->m, s, int32_t(d.size()) + 1, nullptr, &o, rr.env->W(sp));
	std::vector<uint64_t> want;
	for (size_t r = 0; r < d.size(); ++r) want.push_back(f.LegalIds(sp.schema, off.walk_after[p + r]).size());
	std::vector<uint64_t> got;
	for (size_t i = 0; i < sfd::RowsLogCount(); ++i)
		if (sfd::RowsLogIsCheck(i)) got.push_back(sfd::RowsLogAt(i));
	CHECK_MSG(j.st == SSLM_OK && got == want, "[8.X7] the check's S-ROWS entries %zu, want %zu (rows per check row: the "
	          "legal-row finish's n)", got.size(), want.size());
}

// ==========================================================================================================
// 9. Persistence and version
// ==========================================================================================================

// 9.X1 [J1-X] A blob saved right after an exact call restores (into this build, as the canonical build would)
// and continues by plain decode to the off stream; and a blob saved mid-generation by off continues under exact
// mode to the off stream.
// The canonical J1 build's pending-only commit at TF's N (arm (c'): N decodes, then the canonical verb at budget
// 1), pinned as a digest computed on the canonical build (the J1 tip, 2af41659), as 9.1 pins kMainBlobDigest. An
// exact j = 0 commit at N must save this very blob (§5.10.3's commit shape), so the canonical build restores it.
sslm_seq XTfBuildExact(TfRig& x, int j_want, int32_t* j_out = nullptr);  // below, with TF's (d) arms
struct XCanonPin {
	TfCfg cfg;
	size_t size;
	const char* digest;
};
constexpr XCanonPin kXCanonPins[] = {
    {TfCfg{false, false}, 32992, "9951c60aad55491745d441eda7361780c34f317097c3f64dfd7e3b75ee92059c"},
    {TfCfg{true, false}, 33008, "07808d189ed86c6c9730e07c89bc68fa16d6ccd5471e1645248edc8c44710b47"},
};

void Cell_X_9_1() {
	const Sfd& f = GetSfd();
	if (!f.ok) return;
	// The pin's own half, green on the canonical build: (c')'s blob still has the pinned digest.
	for (const XCanonPin& pin : kXCanonPins) {
		TfRig x(f, pin.cfg);
		if (!x.Setup()) continue;
		sslm_seq s = x.Build(TfArm::kCanonPending);
		const std::vector<uint8_t> b = s ? x.rig.Save(s) : std::vector<uint8_t>{};
		if (s) x.rig.Release(s);
		CHECK_MSG(!b.empty() && b.size() == pin.size && BytesDigest(b) == pin.digest,
		          "[9.X1 canonical pin %s] (c')'s blob at N: %zu bytes, digest %s; pinned %zu, %s", TfCfgName(pin.cfg).c_str(),
		          b.size(), BytesDigest(b).c_str(), pin.size, pin.digest);
	}
	if (!XLive("9.X1")) return;
	// Across builds: an exact j = 0 commit at N saves the blob the canonical build pinned.
	for (const XCanonPin& pin : kXCanonPins) {
		TfRig x(f, pin.cfg);
		if (!x.Setup()) continue;
		sslm_seq s = XTfBuildExact(x, 0);
		const std::vector<uint8_t> b = s ? x.rig.Save(s) : std::vector<uint8_t>{};
		if (s) x.rig.Release(s);
		CHECK_MSG(!b.empty() && b.size() == pin.size && BytesDigest(b) == pin.digest,
		          "[9.X1 across builds %s] the exact j = 0 commit at N saved %zu bytes, digest %s; the canonical build's "
		          "pinned (c') blob is %zu, %s", TfCfgName(pin.cfg).c_str(), b.size(), BytesDigest(b).c_str(), pin.size,
		          pin.digest);
	}
	XSpec sp;
	const XOff off = RunOff(f, sp);
	{
		XPlan plan;
		plan.kind = XDraft::kWrongAt;
		std::vector<std::pair<size_t, std::vector<uint8_t>>> saves;
		plan.after_verb = [&](size_t pos, XEnv& e, sslm_seq& s) {
			const int64_t kept = e.rig->State(s).ctx - ParseBlob(off.obs[pos].blob).ctx;
			saves.push_back({pos + size_t(std::max<int64_t>(kept - 1, 0)), e.rig->Save(s)});
		};
		RunExact(f, sp, off, plan);
		Rig rig(f.m1->m);
		int bad = 0;
		for (const auto& sv : saves) {
			sslm_seq q = rig.Restore(sv.second);
			std::vector<int32_t> tail;
			for (int i = 0; q && i < 400; ++i) {
				int32_t t = -100;
				if (rig.Decode1(q, rig.Greedy(), rig.W(), &t) != SSLM_OK || t < 0) break;
				tail.push_back(t);
			}
			if (q) rig.Release(q);
			const std::vector<int32_t> want(off.stream.begin() + long(std::min(sv.first + 1, off.stream.size())), off.stream.end());
			if (tail != want) ++bad;
		}
		CHECK_MSG(!saves.empty() && bad == 0, "[9.X1] %d of %zu blobs saved after an exact call do not continue to off's "
		          "stream by plain decode", bad, saves.size());
	}
	for (size_t at : {size_t(0), size_t(3), off.stream.size() / 2}) {
		if (at + 1 >= off.obs.size()) continue;
		XEnv env(f, sp);
		sslm_seq q = env.rig->Restore(off.obs[at].blob);
		XOff rest = off;
		rest.obs.erase(rest.obs.begin(), rest.obs.begin() + long(at) + 1);
		std::vector<int32_t> tail;
		XPlan plan;
		plan.kind = XDraft::kPerfect;
		// the first call drafts from off's own positions: run the verb by hand, then the loop
		std::mt19937 g(1);
		const std::vector<int32_t> d = XMakeDraft(f, sp.schema, off, at, XDraft::kPerfect, 0, 7, g, nullptr);
		const sslm_jump_options o = XExact(&d);
		const Jump j = XJump(env.m, q, 8, nullptr, &o, env.rig->W());
		tail.insert(tail.end(), j.forced.begin(), j.forced.end());
		for (int i = 0; q && i < 400; ++i) {
			int32_t t = -100;
			if (env.rig->Decode1(q, env.rig->Greedy(), env.rig->W(), &t) != SSLM_OK || t < 0) break;
			tail.push_back(t);
		}
		const std::vector<int32_t> want(off.stream.begin() + long(at) + 1, off.stream.end());
		CHECK_MSG(j.st == SSLM_OK && tail == want, "[9.X1] off's blob at %zu, continued under exact mode: status %d, %s; off "
		          "%s", at, static_cast<int>(j.st), Str(tail).c_str(), Str(want).c_str());
	}
}

// 9.X2 [J1-X] The ABI inventory carries sslm_jump_options and sslm_jump_mode; sslm_seq_jump_forward keeps its
// ordinal in the verb list; the status list's entry count equals the base this
// line builds on: main's 30 plus the merged precached-prefixes line's SSLM_RESTORE_PRODUCER_MISMATCH, 31 (as 9.2).
// J1 and J1-X ship together as 1.15.0, so this cell checks no version; 9.2's receipt reads it at the release commit
// (plan rev 1.3.7, section 17.13).
void Cell_X_9_2() {
	if (!XLive("9.X2")) return;
	const std::string abi = ReadRepoFile("include/superslm/sslm_abi.h");
	const std::string inc = ReadRepoFile("include/superslm/sslm_abi_functions.inc");
	const std::string verbs = ReadRepoFile("tests/t2807-api-slot/c_abi_verbs.txt");
	CHECK_MSG(abi.find("sslm_jump_options") != std::string::npos && abi.find("sslm_jump_mode") != std::string::npos,
	          "[9.X2] sslm_abi.h does not define sslm_jump_options and sslm_jump_mode");
	CHECK_MSG(inc.find("const sslm_jump_options* opts") != std::string::npos,
	          "[9.X2] sslm_abi_functions.inc's verb does not take the options pointer");
	// The ordinal: the verb's line in c_abi_verbs.txt (J1 put it at line 46, after sslm_seq_forced_tokens).
	size_t line = 0, at = 0, found = 0;
	for (size_t i = 0; i <= verbs.size(); ++i)
		if (i == verbs.size() || verbs[i] == '\n') {
			++line;
			if (verbs.compare(at, i - at, "sslm_seq_jump_forward") == 0) found = line;
			at = i + 1;
		}
	CHECK_MSG(found == 46, "[9.X2] sslm_seq_jump_forward is at line %zu of c_abi_verbs.txt; J1 put it at 46", found);
	int n = 0;
#define SFD_X_COUNT_STATUS(name) ++n;
	SSLM_STATUS_ENUM_LIST(SFD_X_COUNT_STATUS)
#undef SFD_X_COUNT_STATUS
	CHECK_MSG(n == 31, "[9.X2] the status list has %d entries; main plus precached prefixes has 31", n);
}

// ==========================================================================================================
// 10. Functional achievement
// ==========================================================================================================

// 10.X1 [J1-X; XE3, XE16] The feature oracle on POTION (greedy and damped@2, budget 64). The perfect draft (to
// the next choice, as the plugin drafts): S-LAYERROWS rows over the whole run equal the output tokens (zero
// waste), and the verb passes with M >= 2 equal the forced runs entered. The always-wrong draft: every admitting
// call admits exactly one row, and out_forced_count is 0. The NULL draft: each pass is the canonical-chain
// prediction of the reference rule (§5.9 (d)): 1 + |chain| rows, j = the chain's prefix that equals off.
void Cell_X_10_1() {
	if (!XLive("10.X1")) return;
	const Sfd& f = GetSfd();
	for (int32_t order : {0, 2}) {
		XSpec sp;
		sp.order = order;
		const XOff off = RunOff(f, sp);
		const RefSchema& ref = f.R(sp.schema);
		{
			XPlan plan;
			plan.kind = XDraft::kPerfectRun;
			plan.budget = 64;
			const XRun run = RunExact(f, sp, off, plan);
			X_EQUALS_OFF("10.X1 perfect", run, sp, plan);
			uint64_t rows = 0;
			size_t big = 0;
			for (const sfd::LayerRowsEntry& e : run.passes) rows += e.rows;
			for (const XVerbRec& v : run.verbs)
				for (const sfd::LayerRowsEntry& e : v.passes) big += e.rows >= 2 ? 1 : 0;
			size_t runs = 0;
			for (size_t i = 0; i + 1 < off.stream.size(); ++i)
				if (XForced(f, sp.schema, off.walk_after[i]) && (i == 0 || !XForced(f, sp.schema, off.walk_after[i - 1])))
					++runs;
			const bool dead_end = !off.obs.empty() && off.obs.back().tok == -2;
			CHECK_MSG(dead_end && rows == off.stream.size(), "[10.X1 %s perfect] S-LAYERROWS rows %llu, output tokens %zu "
			          "(waste must be 0)", sp.Name().c_str(), static_cast<unsigned long long>(rows), off.stream.size());
			CHECK_MSG(big == runs, "[10.X1 %s perfect] %zu verb passes with M >= 2; %zu forced runs entered",
			          sp.Name().c_str(), big, runs);
		}
		{
			XPlan plan;
			plan.kind = XDraft::kAlwaysWrong;
			plan.budget = 64;
			const XRun run = RunExact(f, sp, off, plan);
			X_EQUALS_OFF("10.X1 always-wrong", run, sp, plan);
			for (const XVerbRec& v : run.verbs)
				if (v.ctx_after != v.ctx_before)
					CHECK_MSG(v.ctx_after == v.ctx_before + 1 && v.j.count == 0, "[10.X1 %s always-wrong] the call at %zu "
					          "admitted %lld rows, count %d", sp.Name().c_str(), v.pos,
					          static_cast<long long>(v.ctx_after - v.ctx_before), v.j.count);
		}
		{
			XPlan plan;
			plan.kind = XDraft::kNull;
			plan.budget = 64;
			const XRun run = RunExact(f, sp, off, plan);
			X_EQUALS_OFF("10.X1 NULL", run, sp, plan);
			for (const XVerbRec& v : run.verbs) {
				const uint32_t s = off.walk_after[v.pos];
				const std::vector<int32_t>& chain = s < ref.st.size() ? ref.st[s].chain : std::vector<int32_t>{};
				const size_t k = std::min(chain.size(), size_t(63));
				int32_t want = 0;
				while (size_t(want) < k && v.pos + 1 + size_t(want) < off.stream.size() &&
				       chain[size_t(want)] == off.stream[v.pos + 1 + size_t(want)])
					++want;
				const bool shape = k == 0 ? v.passes.empty() : v.passes.size() == 1 && v.passes[0].rows == 1 + k;
				CHECK_MSG(shape && v.j.count == want, "[10.X1 %s NULL] the call at %zu: S-LAYERROWS %s, count %d; the "
				          "reference rule predicts one pass of %zu and count %d", sp.Name().c_str(), v.pos,
				          Q6cShape(v.passes).c_str(), v.j.count, k ? 1 + k : 0, want);
			}
		}
	}
}

// ==========================================================================================================
// 11. Guard vitality
// ==========================================================================================================

// 11.X1 [J1-X; XE1] S-VERIFY "always equal" turns 6.X1 red under the always-wrong draft (the oracle can fail),
// and "never equal" turns 10.X1's perfect-draft row count red.
void Cell_X_11_1() {
	if (!XLive("11.X1")) return;
	const Sfd& f = GetSfd();
	XSpec sp;
	const XOff off = RunOff(f, sp);
	{
		XPlan plan;
		plan.kind = XDraft::kAlwaysWrong;
		sfd::SetVerifyForce(sfd::VerifyForce::kAlwaysEqual);
		const XRun run = RunExact(f, sp, off, plan);
		sfd::SetVerifyForce(sfd::VerifyForce::kNone);
		CHECK_MSG(!run.ok, "[11.X1] S-VERIFY always-equal: the always-wrong run still equals off (the oracle cannot fail)");
	}
	{
		XPlan plan;
		plan.kind = XDraft::kPerfectRun;
		plan.budget = 64;
		sfd::SetVerifyForce(sfd::VerifyForce::kNeverEqual);
		const XRun run = RunExact(f, sp, off, plan);
		sfd::SetVerifyForce(sfd::VerifyForce::kNone);
		uint64_t rows = 0;
		for (const sfd::LayerRowsEntry& e : run.passes) rows += e.rows;
		CHECK_MSG(rows != off.stream.size(), "[11.X1] S-VERIFY never-equal: the perfect draft still shows zero waste (%llu "
		          "rows for %zu tokens)", static_cast<unsigned long long>(rows), off.stream.size());
	}
}

// 11.X2 [J1-X] The pre-scan still stops an illegal draft token when the draft walk's legality check is disabled
// through a seam: the token is not embedded (S-LAYERROWS rows <= 1 + the legal prefix) and the run continues as
// off's. The verb's status in this state is unspecified by the plan, so it is not asserted.
void Cell_X_11_2() {
	if (!XLive("11.X2")) return;
	const Sfd& f = GetSfd();
	XSpec sp;
	const XOff off = RunOff(f, sp);
	const std::vector<size_t> ps = XForcedPositions(f, sp.schema, off, 2);
	if (ps.empty()) return;
	const size_t p = ps[0];
	int32_t illegal = -1;
	for (int32_t t = 0; t < kTokVocab && illegal < 0; ++t) {
		uint32_t nx = 0;
		if (!f.Step(sp.schema, off.walk_after[p], t, &nx) && !f.specials.count(t)) illegal = t;
	}
	XRestoreRig rr(f, sp);
	if (!rr.ok()) return;
	for (size_t at_index : {size_t(0), size_t(1)}) {
		std::vector<int32_t> d = {off.stream[p + 1], off.stream[p + 2]};
		d[at_index] = illegal;
		sslm_seq s = rr.At(off, p);
		if (!s) continue;
		sfd::SetDraftWalkLegalityDisabled(true);
		sfd::LayerRowsLogReset();
		const sslm_jump_options o = XExact(&d);
		const Jump j = XJump(rr.env->m, s, 8, nullptr, &o, rr.env->W(sp));
		sfd::SetDraftWalkLegalityDisabled(false);
		uint64_t rows = 0;
		for (const sfd::LayerRowsEntry& e : Q6cPasses()) rows += e.rows;
		const Blob b = rr.env->rig->State(s);
		const int64_t kept = b.ctx - ParseBlob(off.obs[p].blob).ctx;
		CHECK_MSG(rows <= 1 + at_index && kept <= int64_t(1 + at_index), "[11.X2 illegal at %zu] status %d: %llu rows ran, "
		          "%lld kept; the pre-scan must stop the illegal token", at_index, static_cast<int>(j.st),
		          static_cast<unsigned long long>(rows), static_cast<long long>(kept));
		int32_t t = -100;
		const sslm_status st = rr.env->rig->Decode1(s, rr.env->rig->Greedy(), rr.env->W(sp), &t);
		const size_t at = p + size_t(std::max<int64_t>(kept, 1));
		const XObs ob = XObserve(*rr.env->rig, s, static_cast<int>(st), t);
		CHECK_MSG(at < off.obs.size() && XDiff(ob, off.obs[at], kXCmpTok | kXCmpBlob).empty(),
		          "[11.X2 illegal at %zu] the next decode: %s", at_index,
		          at < off.obs.size() ? XDiff(ob, off.obs[at], kXCmpTok | kXCmpBlob).c_str() : "past off's end");
		rr.env->rig->Release(s);
	}
}

// 11.X3 [J1-X] With HolderUnmapTail stubbed through a seam, 1.X3 is red.
void Cell_X_11_3() {
	if (!XLive("11.X3")) return;
	const Sfd& f = GetSfd();
	const int differ = XUnmapTailArms(f, true, false);
	CHECK_MSG(differ > 0, "[11.X3] with HolderUnmapTail stubbed, every 1.X3 run still equals off (the oracle cannot fail)");
}

// ==========================================================================================================
// TF's exact-mode arms (plan rev 1.3.3 §11.5 arm (d)), which land with J1-X
// ==========================================================================================================

// The (d) arms: after an exact call that committed j = 0 (p alone) or j >= 2 rows, the target decode runs the
// ready branch. Built on TfRig's POTION sequence at N (TF.2's construction).
sslm_seq XTfBuildExact(TfRig& x, int j_want, int32_t* j_out) {
	sslm_seq s = x.Build(TfArm::kLayer);  // N decodes: t_{N-1} pending
	if (!s) return nullptr;
	// off's never-failed stream continues with stream[N], stream[N + 1], ...
	const size_t at = size_t(x.n);
	std::vector<int32_t> d;
	if (j_want >= 2) {
		// a perfect draft: off's own next tokens (up to 7)
		for (size_t q = at; q < x.stream.size() && d.size() < 7; ++q) d.push_back(x.stream[q]);
	} else if (at < x.stream.size()) {
		// a draft the check rejects at once: a token legal at the walk state after the pending token, other than
		// off's next token (so the draft walk keeps it, k = 1, and the check confirms nothing)
		for (uint32_t t : x.f.LegalIds(kPotion, x.rig.State(s).walk))
			if (int32_t(t) != x.stream[at] && !x.f.specials.count(int32_t(t))) {
				d = {int32_t(t)};
				break;
			}
	}
	if (d.empty()) {
		CHECK_MSG(false, "[TF (d) %s] setup: no draft for j = %d at N = %d (off's stream %zu tokens)",
		          TfCfgName(x.c).c_str(), j_want, x.n, x.stream.size());
		x.rig.Release(s);
		return nullptr;
	}
	const sslm_decode_params p = x.P(2);
	const sslm_jump_options o = XExact(&d);
	const Jump j = XJump(x.rig.m, s, int32_t(d.size()) + 1, x.c.damped ? &p : nullptr, &o, x.W());
	const bool ok = j.st == SSLM_OK && (j_want >= 2 ? j.count >= 2 : j.count == 0) && !ReadShape(x.rig.m, s).pending;
	if (!ok) {
		CHECK_MSG(false, "[TF (d) %s] setup: the exact call at N committed %d drafted tokens (status %d); want %s",
		          TfCfgName(x.c).c_str(), j.count, static_cast<int>(j.st), j_want >= 2 ? ">= 2" : "0");
		x.rig.Release(s);
		return nullptr;
	}
	if (j_out) *j_out = j.count;
	return s;
}

// TF.2d [TF + J1-X; XF1] TF.2's status exits (all four, damped ones under damped) in exact mode's arms (d0) and
// (d2), greedy and damped@2, with a workspace and without one: the failed decode rests at its
// entry (byte-equal to the blob before the call), the refusal repeats on two retries with the blob unchanged,
// and disarmed, the next three decodes equal the never-failed run from the same state.
void Cell_X_TF_2d() {
	if (!XLive("TF.2d")) return;
	const Sfd& f = GetSfd();
	for (const TfCfg c : {TfCfg{false, false}, TfCfg{false, true}, TfCfg{true, false}, TfCfg{true, true}}) {
		TfRig x(f, c);
		if (!x.Setup()) continue;
		std::vector<sfd::FinishExit> exits = {sfd::FinishExit::kFinalNorm, sfd::FinishExit::kLogitsReject};
		if (c.damped) {
			exits.push_back(sfd::FinishExit::kDampedNotOk);
			exits.push_back(sfd::FinishExit::kDampedRefused);
		}
		for (int jw : {0, 2}) {
			for (const sfd::FinishExit ex : exits) {
				TfRig::Run ref;
				if (sslm_seq r = XTfBuildExact(x, jw)) {
					ref = x.Continue(r, 2, 3);
					x.rig.Release(r);
				}
				sslm_seq s = XTfBuildExact(x, jw);
				if (!s || ref.st.size() != 3) {
					if (s) x.rig.Release(s);
					continue;
				}
				const std::vector<uint8_t> before = x.rig.Save(s);
				sfd::ArmFinishExit(ex, 0);
				int32_t t = -100;
				const sslm_status st = x.Decode(s, 2, &t);
				const std::vector<uint8_t> rest = x.rig.Save(s);
				const sslm_status st2 = x.Decode(s, 2, &t);
				const sslm_status st3 = x.Decode(s, 2, &t);
				const bool same = x.rig.Save(s) == rest;
				sfd::DisarmFinishExit();
				CHECK_MSG(st != SSLM_OK && rest == before && st2 == st && st3 == st && same,
				          "[TF.2d (d%d) %s, %s] status %d, %d, %d; rest %s entry; retries %s", jw, TfCfgName(c).c_str(),
				          TfExitName(ex), static_cast<int>(st), static_cast<int>(st2), static_cast<int>(st3),
				          rest == before ? "at" : "not at", same ? "unchanged" : "changed the blob");
				const TfRig::Run cont = x.Continue(s, 2, 3);
				CHECK_MSG(cont == ref, "[TF.2d (d%d) %s, %s] disarmed: %s; never-failed %s", jw, TfCfgName(c).c_str(),
				          TfExitName(ex), Str(cont).c_str(), Str(ref).c_str());
				x.rig.Release(s);
			}
		}
	}
}

// TF.4 (b) X [TF + J1-X] At a dead end the prepared re-key is installed in exact mode as in off: the -2 decode at
// order 3 after an exact commit rests byte-equal to off's -2 decode at the same output position (off's layer
// branch, arm (a), at that position): after a j = 0 commit (d0) at N, and after a j >= 2 commit (d2) at N + j.
void Cell_X_TF_4b() {
	if (!XLive("TF.4b X")) return;
	const Sfd& f = GetSfd();
	for (const TfCfg c : {TfCfg{true, false}, TfCfg{true, true}}) {
		TfRig x(f, c);
		if (!x.Setup()) continue;
		for (int jw : {0, 2}) {
			std::vector<uint8_t> blobs[2];
			int32_t j = -1;
			sslm_seq e = XTfBuildExact(x, jw, &j);
			if (!e) continue;
			for (int arm = 0; arm < 2; ++arm) {
				sslm_seq s = arm == 0 ? x.Build(TfArm::kLayer, x.n + j) : e;
				if (!s) continue;
				const Blob b = x.rig.State(s);
				int32_t illegal = -1;
				for (int32_t t = 0; t < kTokVocab && illegal < 0; ++t) {
					uint32_t nx = 0;
					if (!f.Step(kPotion, b.walk, t, &nx)) illegal = t;
				}
				sfd::ArmChoice(illegal);
				int32_t t = -100;
				const sslm_status st = x.Decode(s, 3, &t);
				sfd::DisarmChoice();
				CHECK_MSG(st == SSLM_OK && t == -2, "[TF.4 (b) X (d%d) %s, %s] the dead end: status %d token %d", jw,
				          TfCfgName(c).c_str(), arm ? "exact" : "off", static_cast<int>(st), t);
				blobs[arm] = x.rig.Save(s);
				x.rig.Release(s);
			}
			CHECK_MSG(!blobs[0].empty() && blobs[0] == blobs[1], "[TF.4 (b) X (d%d) %s] after the -2 decode, exact %s; off %s",
			          jw, TfCfgName(c).c_str(), TfFields(blobs[1]).c_str(), TfFields(blobs[0]).c_str());
		}
	}
}
