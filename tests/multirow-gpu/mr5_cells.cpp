// Multi-row GPU plan (rev 1.3.2), slice MR5 (G1-X: Exact on the GPU, layer-sliced chunks; MR-D11): the red suite's
// cloud cells. Wine + vkd3d-proton + lavapipe, on F-SFD (tools/gen_schema_fast_decode_fixture.py: fsfd.sslm and its
// independent reference fsfd_ref.txt). Built by mr5_wine.sh with -DSUPERSLM_MULTIROW_WIDE and the test seams.
//
// THE ORACLE. Every expectation comes from the Off path (the G5 decode bridge, one token at a time) or the CPU's Off
// run, in the same process, never from an Exact build's output. A twin handle in the same context replays the same
// history; the Exact handle must then equal it byte for byte (SLM5 dedicated, SLM6 pooled) at the same token count and
// rest shape. Drafts are the model's own next tokens (from the twin), the forced chain the engine's 1.17 query gives,
// or, where a draft must be legal and wrong, a token fsfd_ref.txt shows legal at a choice state (its paths), never
// one an Exact build admitted.
//
// EXACT_DRAFT's model (plan §4.1, §4.7; ruling 26 Q1): a handle holding a pending token p (produced and walked by the
// prompt's finish, an Off finish or an Exact close) opens [p, d1..d(M-1)] with flags 0. Off with p in hand would
// produce o1, o2, ...; the close commits a rows (p and the drafts up to the first d_r != o_r), and the produced token
// is o_a, which rows_produced returns. That is exactly the Off twin after a decodes: o_a in hand, ready clear.
//
// Cells (red until MR5 builds Exact; mr5_entry.h names the verb and seams the library lacks today):
//   2.2   sslm_gpu_seq_rows_produced: null arguments, BUSY with a chunk open, the produced token after an Exact
//         close, SEQUENCE_REJECTED after a PROMPT close, an abandon, a reset, a restore and on a fresh handle.
//   2.3x  EXACT_DRAFT with an illegal draft at row k (a special id, never legal under a schema) admits k (k 1..M-1);
//         a token out of the vocabulary is TOKEN_ID_OUT_OF_RANGE; headroom caps admission; flag 0x1 is a bad spec.
//   7.10  The Exact commit's counters (S-GSATROW): row 1 saturates and a = 3, counters = entry + the deltas of rows
//         0..2 and the blob equals Off's (MX19); a saturating REJECTED row is excluded (MX23).
//   7.11e Exact as a setting, the engine's part (ruling 26 Q2): on a model mapped without a device logits head,
//         embed_rows(EXACT_DRAFT) is SEQUENCE_REJECTED with nothing changed, and the jump verb (NULL options) still
//         keeps exactly Off's tokens and bytes. (7.11 (a), (b)'s log and (c) are the plugin's, MR6.)
//   7.17  rows_produced returns Off's next token after an Exact close (the first chunk after a prompt, after an Off
//         token, after an Exact close: that chunk walks the produced token once), and -2 at a schema dead end; at
//         every outcome the whole blob equals Off's, ready word included; at the dead end one more decode-bridge call
//         returns -2 at OK and consumes nothing (MX24).
//   7.18  Exact's partial commit for every a in 1..M: last_rows_committed = a, pages and the pool's free count equal
//         Off's, the blob (mirror) equals Off's.
//   7.19  S-GNARROW at row r: the status, last_rows_committed (r + 1: the row whose finish failed is in Off's blob)
//         and the whole blob equal Off's at that token's finish failure.
//   8.4   A plugin-shaped Exact loop (drafts = the engine's forced chain) gives Off's tokens, and Off's save bytes at
//         every close; M 2, 4, 8; one-layer and full slices; dedicated and pooled; four schemas.
//   gx.1  The jump verb (NULL options) in a host loop gives GPU Off's stream, and both equal the CPU's Off stream.
//   gx.2  After every decode step of that loop the blob equals Off's (SLM5, SLM6) and the pooled table's mapped count
//         equals Off's; a rejected page crossing leaves a dirty page that release poisons (0xCD; XG17, XG18); a
//         saturating kept row's counters (XG6).
//   gx.4  The verb keeps no reference to the draft; exact loop, reset, rebind, exact loop equals two fresh Off runs.
//   gx.5  Malformed options (KV_BUFFER_MISMATCH, nothing changed) and hostile drafts (out of range, -1, a special id
//         at row k): kept <= k, and the run equals Off (XG7).
//   gx.6  Two sequences in Exact with different drafts; the claim hook asserts the second caller's BUSY; each equals
//         its Off run.
//   gx.7  The pending predicate at the GPU's resting shapes: ready and fresh without a pending token are no-ops; an
//         empty draft keeps p pending; k = 1 and k = budget - 1; a pending token at the ready shape and a mid-token
//         handle are 1.17's SEQUENCE_REJECTED.
//   gx.8  A fault in the first pass (S-GCHUNKFAULT): its status, nothing committed. A fault in a later pass and
//         S-GREADBACK: OK, the earlier passes' kept rows only, and the continuation equals Off.
//   gx.9  An Exact, an Off and an unbound sequence interleaved on one context: each equals its solo run.
//   gx.10 Save and restore between Exact calls (SLM5, SLM6), then Exact again: the Off stream.
//   gx.11 S-GVERIFY "always equal" with a legal, always-wrong draft turns gx.1's comparison red (guard vitality).
//   gx.12 The prefill snapshot reads the same after every Exact call (XG12).
//   gx.13 A mismatch in the first pass: no later pass runs (S-GEXACTPASSES; XG13).
//
// Usage: mr5_cells.exe --fixtures=DIR [--cells=2.2,7.17,...] [--show=N]
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mr2_common.h"
#include "mr5_entry.h"
#include "superslm/sslm_abi.h"

// The engine's claim seam (mr3_entry.h has the contract): the hook runs on the thread that takes a handle's claim.
void SslmGpuSeqClaimHookForTest(void (*hook)(SslmGpuSequenceHandle* seq, void* user), void* user) noexcept;

namespace {
using S = SslmGpuStatus;
using mr2::Model;

constexpr S kArg = S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
constexpr int32_t kExactDraft = 2;  // SSLM_GPU_ROWS_EXACT_DRAFT: the value stays 2 (1 is refused, E-11 (c))
constexpr int32_t kSpecial = 151643;  // <|endoftext|>: a special id, byteless, never legal under a schema mask
constexpr int32_t kNull = -1000;      // DoJump: NULL options

struct Opt {
	std::string fixtures;
	std::set<std::string> cells;
	int show = 16;
	bool want(const std::string& c) const { return cells.empty() || cells.count(c); }
};
Opt g_opt;
std::map<std::string, int> g_fail, g_checks;
std::string g_cell;
void Check(bool ok, const std::string& what) {
	++g_checks[g_cell];
	if (!ok && ++g_fail[g_cell] <= g_opt.show) std::printf("FAIL %s: %s\n", g_cell.c_str(), what.c_str());
}
std::string F(const char* fmt, ...) {
	char b[1536];
	va_list ap;
	va_start(ap, fmt);
	std::vsnprintf(b, sizeof b, fmt, ap);
	va_end(ap);
	return b;
}
const char* N(S s) {
	static thread_local char buf[8][24];
	static thread_local int at = 0;
	char* b = buf[at++ % 8];
	if (s == MR5_STUB_STATUS) return "STUB(unimplemented)";
	std::snprintf(b, 24, "%u", static_cast<unsigned>(s));
	return b;
}
void StatusIs(S got, S want, const std::string& tag) { Check(got == want, F("%s: returned %s, want %s", tag.c_str(), N(got), N(want))); }
std::string Ids(const std::vector<int32_t>& v) {
	std::string s = "{";
	for (size_t i = 0; i < v.size() && i < 16; ++i) s += (i ? "," : "") + std::to_string(v[i]);
	return s + (v.size() > 16 ? ",...}" : "}");
}
void BlobEq(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, const std::string& tag) {
	const long d = mr2::FirstDiff(got, want);
	size_t n = 0;
	for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) n += got[i] != want[i];
	Check(d < 0 && !got.empty(), F("%s: blob differs from Off's (sizes %zu/%zu, first byte %ld, %zu bytes differ)", tag.c_str(), got.size(),
	                               want.size(), d, n));
}

// ------------------------------------------------------------------------------------------------ the world
const char* kPotion = "potion_shop_order";
const char* kPromptResult = "prompt_result";
const char* kLongKey = "longkey";
const char* kUtf8 = "utf8enum";
const char* kSchemas[] = {kPotion, kPromptResult, kLongKey, kUtf8};

// mr2::World with the residency chosen: Exact on the GPU needs the device-resident logits head (§4.7 step 4).
struct World {
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	std::vector<SslmGpuKvPoolHandle*> pools;
	std::vector<SslmGpuSequenceHandle*> seqs;
	bool ok = false;
	explicit World(const Model& m, bool head_on_device = true) {
		S st = sslm_gpu_context_create(GpuContextConfig{}, &ctx);
		if (st != S::SSLM_OK) { std::printf("FAIL context create %s\n", N(st)); return; }
		GpuResidencyConfig rc{};
		rc.flags = head_on_device ? SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE : 0u;
		if ((st = sslm_gpu_model_map(ctx, &m.view, rc, &model)) != S::SSLM_OK) { std::printf("FAIL model map %s\n", N(st)); return; }
		ok = true;
	}
	World(const World&) = delete;
	~World() {
		for (auto it = seqs.rbegin(); it != seqs.rend(); ++it)
			if (*it) sslm_gpu_seq_release(ctx, *it);
		for (auto* p : pools) sslm_gpu_kv_pool_destroy(ctx, p);
		if (model) sslm_gpu_model_unmap(ctx, model);
		if (ctx) sslm_gpu_context_destroy(ctx);
	}
	SslmGpuKvPoolHandle* Pool(uint32_t pages) {
		SslmGpuKvPoolHandle* p = nullptr;
		if (sslm_gpu_kv_pool_create(ctx, model, pages, &p) != S::SSLM_OK) return nullptr;
		pools.push_back(p);
		return p;
	}
	SslmGpuSequenceHandle* Seq(bool pooled, int32_t cap, const char* schema, SslmGpuKvPoolHandle** pool_out = nullptr) {
		SslmGpuSequenceHandle* s = nullptr;
		S st;
		if (pooled) {
			SslmGpuKvPoolHandle* p = Pool(static_cast<uint32_t>(cap / 4 + 8));
			if (pool_out) *pool_out = p;
			st = p ? sslm_gpu_seq_create_budgeted(ctx, model, p, cap, &s) : S::SSLM_DEVICE_LOST;
		} else {
			st = sslm_gpu_seq_create(ctx, model, cap, &s);
		}
		if (st != S::SSLM_OK) { Check(false, F("seq create %s", N(st))); return nullptr; }
		seqs.push_back(s);
		if (schema) {
			const S b = SslmGpuSeqSetSchemaForG5Bridge(ctx, s, SslmGpuSchemaLookupForG5Bridge(model, schema));
			Check(b == S::SSLM_OK, F("bind %s: %s", schema, N(b)));
		}
		return s;
	}
	void Forget(SslmGpuSequenceHandle* s) {
		for (auto& x : seqs)
			if (x == s) x = nullptr;
	}
};

struct Fsfd {
	Model m;
	bool ok = false;
	bool Load() { return ok = m.Load(g_opt.fixtures + "/fsfd.sslm"); }
};

std::vector<int32_t> Prompt(int n, uint32_t stream) {
	std::vector<int32_t> v;
	for (int i = 0; i < n; ++i) v.push_back(mr2::StreamToken(i, 151936, stream) % 20000 + 200);  // ordinary text ids
	return v;
}
uint32_t Walk(SslmGpuSequenceHandle* s) { return SslmGpuSeqWalkStateForG5Bridge(s); }
int32_t ReadyFlag(World& w, SslmGpuSequenceHandle* s) {
	int32_t r = -1;
	return sslm_gpu_seq_ready_for_logits(w.ctx, s, &r) == S::SSLM_OK ? r : -1;
}
int64_t CtxLen(SslmGpuSequenceHandle* s) { return *SslmGpuSeqHandleContextLengthForBench(s); }
int32_t RowsOpen(SslmGpuSequenceHandle* s) {
	int32_t r = -1;
	return sslm_gpu_seq_rows_open(s, &r) == S::SSLM_OK ? r : -1;
}
int32_t LastCommitted(SslmGpuSequenceHandle* s) {
	int32_t r = -1;
	return sslm_gpu_seq_last_rows_committed(s, &r) == S::SSLM_OK ? r : -1;
}
uint32_t Mapped(SslmGpuSequenceHandle* s) {
	uint32_t page = 0, mapped = 0;
	(void)sslm_gpu_test_only_seq_table_entry(s, 0, &page, &mapped);
	return mapped;
}
uint32_t PoolFree(SslmGpuKvPoolHandle* p) {
	SslmGpuKvPoolStats st{};
	st.struct_size = sizeof st;
	return p && sslm_gpu_kv_pool_stats(p, &st) == S::SSLM_OK ? st.free_pages : 0xFFFFFFFFu;
}
std::vector<int32_t> Chain(World& w, SslmGpuSequenceHandle* s) {
	std::vector<int32_t> ids(512, -7);
	int32_t n = 0, acc = 0;
	const S st = sslm_gpu_seq_forced_tokens(w.ctx, s, ids.data(), 512, &n, &acc);
	ids.resize(st == S::SSLM_OK && n >= 0 ? static_cast<size_t>(std::min(n, 512)) : 0);
	return ids;
}
std::vector<uint8_t> Blob(World& w, SslmGpuSequenceHandle* s, const std::string& tag) {
	std::vector<uint8_t> b;
	S st = S::SSLM_OK;
	size_t need = 0;
	sslm_gpu_seq_save(w.ctx, s, nullptr, &need);
	b.assign(need ? need : 1, 0);
	size_t sz = b.size();
	st = sslm_gpu_seq_save(w.ctx, s, b.data(), &sz);
	if (st != S::SSLM_OK) {
		Check(false, F("%s: save %s", tag.c_str(), N(st)));
		return {};
	}
	b.resize(sz);
	return b;
}
struct Obs {
	std::vector<uint8_t> blob;
	uint32_t walk = 0, layer = 0, mapped = 0;
	int32_t ready = -1, open = -1;
	int64_t ctx = -1;
	bool operator==(const Obs& o) const {
		return blob == o.blob && walk == o.walk && layer == o.layer && mapped == o.mapped && ready == o.ready && open == o.open && ctx == o.ctx;
	}
	std::string Str() const {
		return F("blob %zu fnv %016llx walk %u layer %u ready %d open %d ctx %lld mapped %u", blob.size(), (unsigned long long)mr2::Fnv(blob), walk,
		         layer, ready, open, (long long)ctx, mapped);
	}
};
Obs Observe(World& w, SslmGpuSequenceHandle* s) {
	Obs o;
	o.blob = Blob(w, s, "observe");
	o.walk = Walk(s);
	o.layer = *SslmGpuSeqHandleLayerIndexForBench(s);
	o.ready = ReadyFlag(w, s);
	o.ctx = CtxLen(s);
	o.open = RowsOpen(s);
	o.mapped = Mapped(s);
	return o;
}
void Unchanged(World& w, SslmGpuSequenceHandle* s, const Obs& before, const std::string& tag) {
	const Obs after = Observe(w, s);
	Check(after == before, F("%s: changed the handle (before %s; after %s)", tag.c_str(), before.Str().c_str(), after.Str().c_str()));
}

// ------------------------------------------------------------------------------------------------ Off (the oracle)
// One decode-bridge token: embeds `tok` (unless the handle is ready, when it finishes the ready row) and finishes.
S Dec(World& w, const Model& m, SslmGpuSequenceHandle* s, int32_t tok, int32_t* out) {
	*out = -1;
	return SslmGpuSeqDecodeStepForG5Bridge(w.ctx, s, tok, m.full_budget(), out);
}
// A handle's history: the prompt, then `n` Off tokens (the first is the prompt's finish). `pend` is the last token
// produced (in hand, not embedded), or the prompt's last id when n = 0 (the ready shape).
struct Hist {
	std::vector<int32_t> prompt;
	std::vector<int32_t> toks;  // o_0 .. o_(n-1)
	int32_t pend = -1;
};
bool Replay(World& w, const Model& m, SslmGpuSequenceHandle* s, const std::vector<int32_t>& prompt, int n, Hist* h) {
	h->prompt = prompt;
	h->toks.clear();
	if (SslmGpuSeqPrefillPromptForG5Bridge(w.ctx, s, prompt.data(), static_cast<int32_t>(prompt.size()), m.full_budget()) != S::SSLM_OK) return false;
	int32_t last = prompt.back();
	for (int i = 0; i < n; ++i) {
		int32_t t = -1;
		if (Dec(w, m, s, last, &t) != S::SSLM_OK || t < 0) return false;
		h->toks.push_back(t);
		last = t;
	}
	h->pend = last;
	return true;
}
// The Off reference of a schema and prompt: up to `max` tokens, the walk after each, and the dead end (the index of
// the decode that returned -2, i.e. the stream ends after `dead` tokens), on a fresh world.
struct OffRef {
	std::vector<int32_t> toks;
	std::vector<uint32_t> walk;  // walk[i]: after o_i
	int dead = -1;
	bool ok = false;
};
OffRef RunOff(const Model& m, const char* schema, const std::vector<int32_t>& prompt, int max, bool head = true) {
	OffRef r;
	World w(m, head);
	if (!w.ok) return r;
	SslmGpuSequenceHandle* s = w.Seq(false, 240, schema);
	if (!s || SslmGpuSeqPrefillPromptForG5Bridge(w.ctx, s, prompt.data(), static_cast<int32_t>(prompt.size()), m.full_budget()) != S::SSLM_OK) return r;
	int32_t last = prompt.back();
	for (int i = 0; i < max; ++i) {
		int32_t t = -1;
		if (Dec(w, m, s, last, &t) != S::SSLM_OK) return r;
		if (t == -2) {
			r.dead = i;
			break;
		}
		if (t < 0) return r;
		r.toks.push_back(t);
		r.walk.push_back(Walk(s));
		last = t;
	}
	r.ok = true;
	return r;
}

// ------------------------------------------------------------------------------------------------ CPU Off (gx.1)
struct CpuOff {
	std::vector<int32_t> toks;
	bool ok = false;
};
CpuOff RunCpuOff(const Model& m, const char* schema, const std::vector<int32_t>& prompt, int max) {
	// The CPU ABI as mrx_cpu.cpp drives it: a workspace, a 24-block pool, greedy parameters.
	CpuOff r;
	const size_t n = m.bytes.size();
	uint8_t* mem = static_cast<uint8_t*>(::operator new((n + 63) / 64 * 64, std::align_val_t(64)));
	std::memcpy(mem, m.bytes.data(), n);
	sslm_model cm = nullptr;
	const char* failed = "sslm_model_map";
	if (sslm_model_map(mem, n, &cm) == SSLM_OK) {
		const uint32_t blocks = 24;
		const size_t size = blocks * sslm_kv_block_size(cm) + sslm_kv_pool_overhead_size(cm, blocks);
		uint8_t* pm = static_cast<uint8_t*>(::operator new((size + 63) / 64 * 64, std::align_val_t(64)));
		const sslm_config cfg{1, 64, 2, 0u};
		const size_t wsz = sslm_workspace_size(cm, &cfg);
		uint8_t* wm = static_cast<uint8_t*>(::operator new((wsz + 63) / 64 * 64, std::align_val_t(64)));
		sslm_workspace ws = nullptr;
		sslm_kv_pool pool = nullptr;
		sslm_seq s = nullptr;
		sslm_schema h = nullptr;
		if (sslm_kv_pool_create(cm, pm, size, blocks, &pool) != SSLM_OK) failed = "sslm_kv_pool_create";
		else if (sslm_workspace_create(cm, &cfg, wm, wsz, &ws) != SSLM_OK) failed = "sslm_workspace_create";
		else if (sslm_seq_create(cm, &pool, &s) != SSLM_OK) failed = "sslm_seq_create";
		else if (sslm_schema_lookup(cm, schema, &h) != SSLM_OK || sslm_seq_set_schema(s, h) != SSLM_OK) failed = "the schema";
		else {
			int32_t c = 0;
			bool ok = sslm_prefill(cm, s, prompt.data(), static_cast<int32_t>(prompt.size()), static_cast<int32_t>(prompt.size()), SSLM_SPAN_PROMPT,
			                       ws, &c) == SSLM_OK;
			failed = ok ? nullptr : "sslm_prefill";
			sslm_decode_params p{};
			sslm_decode_params_init(cm, SSLM_DECODE_MODE_GREEDY, 2, &p);
			for (int i = 0; ok && i < max; ++i) {
				int32_t out = -100;
				ok = sslm_decode_step_v2(cm, &s, 1, &p, ws, &out) == SSLM_OK;
				if (!ok) failed = "sslm_decode_step_v2";
				if (!ok || out < 0) break;
				r.toks.push_back(out);
			}
			r.ok = ok;
		}
		if (s) sslm_seq_release(s);
		if (ws) sslm_workspace_destroy(ws);
		if (pool) sslm_kv_pool_destroy(pool);
		sslm_model_unmap(cm);
		::operator delete(wm, std::align_val_t(64));
		::operator delete(pm, std::align_val_t(64));
	}
	if (failed) std::printf("CPU Off (%s): %s failed\n", schema, failed);
	::operator delete(mem, std::align_val_t(64));
	return r;
}

// ------------------------------------------------------------------------------------------------ fsfd_ref.txt
// Per schema: each state's reference path from the start; from them, the tokens known legal at a non-forced state
// (the next token of every path that runs through it), each with the rest of that path (legal from there on).
struct RefState {
	bool forced = false;
	std::vector<int32_t> chain, path;
};
struct RefSchema {
	std::vector<RefState> states;
};
std::map<std::string, RefSchema> g_ref;
bool LoadRef() {
	std::FILE* f = std::fopen((g_opt.fixtures + "/fsfd_ref.txt").c_str(), "r");
	if (!f) return false;
	char line[65536];
	RefSchema* cur = nullptr;
	while (std::fgets(line, sizeof line, f)) {
		std::istringstream in(line);
		std::string kind;
		in >> kind;
		if (kind == "schema") {
			std::string name;
			in >> name;
			cur = &g_ref[name];
		} else if (kind == "state" && cur) {
			RefState st;
			int id = 0, acc = 0, forced = 0, flen = 0, nch = 0, npa = 0;
			std::string fhex;
			in >> id >> acc >> forced >> flen >> fhex >> nch;
			st.forced = forced != 0;
			for (int i = 0; i < nch; ++i) { int32_t x; in >> x; st.chain.push_back(x); }
			in >> npa;
			for (int i = 0; i < npa; ++i) { int32_t x; in >> x; st.path.push_back(x); }
			if (static_cast<int>(cur->states.size()) <= id) cur->states.resize(static_cast<size_t>(id) + 1);
			cur->states[static_cast<size_t>(id)] = st;
		}
	}
	std::fclose(f);
	return !g_ref.empty();
}
// Tokens known legal at `state` of `schema`, each with the legal continuation that follows it on its path.
std::vector<std::pair<int32_t, std::vector<int32_t>>> LegalAt(const char* schema, uint32_t state) {
	std::vector<std::pair<int32_t, std::vector<int32_t>>> out;
	auto it = g_ref.find(schema);
	if (it == g_ref.end() || state >= it->second.states.size()) return out;
	const std::vector<int32_t>& pa = it->second.states[state].path;
	for (const RefState& t : it->second.states) {
		if (t.path.size() <= pa.size() || !std::equal(pa.begin(), pa.end(), t.path.begin())) continue;
		const int32_t x = t.path[pa.size()];
		std::vector<int32_t> tail(t.path.begin() + static_cast<long>(pa.size()) + 1, t.path.end());
		bool merged = false;
		for (auto& e : out)
			if (e.first == x) {
				if (tail.size() > e.second.size()) e.second = tail;
				merged = true;
			}
		if (!merged) out.push_back({x, tail});
	}
	return out;
}

// A choice point in an Off stream: the index c whose walk (after o_c) is a non-forced state with a known legal token
// X != o_(c+1), and X's legal continuation. Searched over prompts; the reference's state ids are the engine's walk
// states (checked: every forced state the stream passes has the reference's chain as the engine's query).
struct Choice {
	const char* schema = nullptr;
	std::vector<int32_t> prompt;
	OffRef off;
	int c = -1;
	int32_t x = -1;
	std::vector<int32_t> tail;
};
Choice FindChoice(const Model& m, int min_c, int min_tail, int min_after = 2) {
	for (const char* schema : {kLongKey, kPotion, kPromptResult, kUtf8})
		for (uint32_t stream = 1; stream < 40; ++stream) {
			Choice ch;
			ch.schema = schema;
			ch.prompt = Prompt(12, 500 + stream);
			ch.off = RunOff(m, schema, ch.prompt, 64);
			if (!ch.off.ok) continue;
			for (int c = min_c; c + min_after < static_cast<int>(ch.off.toks.size()); ++c) {
				const uint32_t s = ch.off.walk[static_cast<size_t>(c)];
				auto it = g_ref.find(schema);
				if (it == g_ref.end() || s >= it->second.states.size() || it->second.states[s].forced) continue;
				for (auto& e : LegalAt(schema, s)) {
					if (e.first == ch.off.toks[static_cast<size_t>(c) + 1] || static_cast<int>(e.second.size()) < min_tail) continue;
					ch.c = c;
					ch.x = e.first;
					ch.tail = e.second;
					return ch;
				}
			}
		}
	return Choice{};
}

// ------------------------------------------------------------------------------------------------ Exact drivers
// Drains `s`: the slice's status, or the ready call's own failure (mr2::Ready on this file's World).
S Ready(World& w, SslmGpuSequenceHandle* s) {
	int32_t ready = 0;
	S out = S::SSLM_OK;
	const S st = sslm_gpu_ready(w.ctx, s, 1, &ready, &out);
	if (st != S::SSLM_OK) return st;
	if (!ready) return S::SSLM_BUSY;
	return out;
}
S EmbedRows(World& w, SslmGpuSequenceHandle* s, int32_t kind, const std::vector<int32_t>& rows, uint32_t flags, int32_t* admitted) {
	SslmGpuRowsSpec spec{sizeof(SslmGpuRowsSpec), kind, rows.data(), static_cast<int32_t>(rows.size()), flags};
	*admitted = -9;
	return sslm_gpu_seq_embed_rows(w.ctx, s, &spec, admitted);
}
// Slices of `budget` dispatches (0: one layer) until the chunk closes; the closing ready's status.
S CloseChunk(World& w, const Model& m, SslmGpuSequenceHandle* s, uint32_t budget = 0) {
	const uint32_t b = budget ? budget : m.dpl();
	for (int i = 0; i < 128; ++i) {
		S st = sslm_decode_step_gpu(w.ctx, s, nullptr, b);
		if (st == S::SSLM_OK) st = Ready(w, s);
		if (st != S::SSLM_OK) return st;
		if (RowsOpen(s) == 0) return S::SSLM_OK;
	}
	return S::SSLM_DEVICE_LOST;
}
struct ExOut {
	S embed = S::SSLM_OK, close = S::SSLM_OK, prod = S::SSLM_OK;
	int32_t adm = -9, tok = -77;
};
ExOut RunExact(World& w, const Model& m, SslmGpuSequenceHandle* s, const std::vector<int32_t>& rows, uint32_t budget = 0) {
	ExOut o;
	o.embed = EmbedRows(w, s, kExactDraft, rows, 0, &o.adm);
	if (o.embed != S::SSLM_OK) return o;
	o.close = CloseChunk(w, m, s, budget);
	o.prod = sslm_gpu_seq_rows_produced(s, &o.tok);
	return o;
}
// Accepted rows: p plus the drafts that equal Off's next tokens (`next` = o_(j+1), o_(j+2), ...).
int32_t Accepted(const std::vector<int32_t>& rows, const std::vector<int32_t>& next) {
	int32_t a = 1;
	while (a < static_cast<int32_t>(rows.size()) && a - 1 < static_cast<int32_t>(next.size()) && rows[static_cast<size_t>(a)] == next[static_cast<size_t>(a) - 1]) ++a;
	return a;
}
// Advance the Off twin `k` decodes from its in-hand token; returns the last produced token (-2 at a dead end).
int32_t TwinDecodes(World& w, const Model& m, SslmGpuSequenceHandle* t, Hist* h, int k, S* st_out = nullptr) {
	int32_t last = h->pend, tok = -1;
	for (int i = 0; i < k; ++i) {
		const S st = Dec(w, m, t, last, &tok);
		if (st_out) *st_out = st;
		if (st != S::SSLM_OK || tok < 0) return st != S::SSLM_OK ? -1000 : tok;
		h->toks.push_back(tok);
		last = tok;
	}
	h->pend = last;
	return tok;
}

// The jump verb: `mode` kNull (NULL options) or a mode; `draft` with SSLM_JUMP_EXACT.
struct Jump {
	S st = S::SSLM_OK;
	int32_t count = -7;
	std::vector<int32_t> out;
};
Jump DoJump(World& w, SslmGpuSequenceHandle* s, int32_t pending, int32_t budget, int32_t mode, const int32_t* draft = nullptr,
            int32_t draft_count = 0, uint32_t struct_size = sizeof(SslmGpuJumpOptions)) {
	Jump j;
	j.out.assign(static_cast<size_t>(std::max(budget, 64)), -7);
	SslmGpuJumpOptions o{};
	o.struct_size = struct_size;
	o.mode = mode;
	o.draft = draft;
	o.draft_count = draft_count;
	j.st = sslm_gpu_seq_jump_forward(w.ctx, s, pending, budget, mode == kNull ? nullptr : &o, j.out.data(), static_cast<int32_t>(j.out.size()), &j.count);
	return j;
}
// Exact's kept drafts: the clipped draft's longest prefix equal to Off's next tokens.
int32_t Kept(const std::vector<int32_t>& draft, int32_t budget, const std::vector<int32_t>& next) {
	int32_t k = 0;
	const int32_t lim = std::min<int32_t>(static_cast<int32_t>(draft.size()), budget - 1);
	while (k < lim && k < static_cast<int32_t>(next.size()) && draft[static_cast<size_t>(k)] == next[static_cast<size_t>(k)]) ++k;
	return k;
}

// The Exact jump loop: from a handle holding `h->pend` (in hand after Off decodes), call the verb with `budget` and
// the options (NULL, or `draft_fn`'s draft), then one decode; until `total` tokens or a dead end. `after_decode` runs
// after every decode with the token count so far.
struct LoopOut {
	std::vector<int32_t> toks;
	bool dead = false;
	bool ok = true;
};
LoopOut ExactLoop(World& w, const Model& m, SslmGpuSequenceHandle* s, Hist* h, int32_t budget, int total,
                  const std::function<std::vector<int32_t>(SslmGpuSequenceHandle*)>& draft_fn,
                  const std::function<void(size_t)>& after_decode, const std::string& tag) {
	LoopOut o;
	o.toks = h->toks;
	int32_t pend = h->pend;
	bool in_hand = !h->toks.empty();
	for (int guard = 0; guard < 400 && static_cast<int>(o.toks.size()) < total; ++guard) {
		if (in_hand) {
			const int32_t room = total - static_cast<int32_t>(o.toks.size());
			const int32_t b = std::min(budget, std::max(room, 1));
			Jump j;
			if (draft_fn) {
				const std::vector<int32_t> d = draft_fn(s);
				j = DoJump(w, s, pend, b, SSLM_JUMP_EXACT, d.empty() ? nullptr : d.data(), static_cast<int32_t>(d.size()));
			} else {
				j = DoJump(w, s, pend, b, kNull);
			}
			if (j.st != S::SSLM_OK) {
				Check(false, F("%s: the jump at %zu tokens returned %s", tag.c_str(), o.toks.size(), N(j.st)));
				o.ok = false;
				return o;
			}
			for (int32_t i = 0; i < j.count; ++i) o.toks.push_back(j.out[static_cast<size_t>(i)]);
			if (j.count > 0) pend = j.out[static_cast<size_t>(j.count) - 1];
		}
		int32_t t = -1;
		const S st = Dec(w, m, s, pend, &t);
		if (st != S::SSLM_OK) {
			Check(false, F("%s: decode at %zu tokens returned %s", tag.c_str(), o.toks.size(), N(st)));
			o.ok = false;
			return o;
		}
		if (t == -2) {
			o.dead = true;
			break;
		}
		o.toks.push_back(t);
		pend = t;
		in_hand = true;
		if (after_decode) after_decode(o.toks.size());
	}
	return o;
}

// ------------------------------------------------------------------------------------------------ 2.2
void Cell22(Fsfd& fx) {
	const Model& m = fx.m;
	int32_t tok = -77;
	StatusIs(sslm_gpu_seq_rows_produced(nullptr, &tok), kArg, "null sequence");
	Check(tok == -77, "null sequence: out_token written");
	World w(m);
	if (!w.ok) { Check(false, "world"); return; }
	for (int pooled = 0; pooled <= 1; ++pooled) {
		const std::string base = pooled ? "pooled" : "dedicated";
		SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 200, kLongKey);
		if (!s) continue;
		StatusIs(sslm_gpu_seq_rows_produced(s, nullptr), kArg, base + ": null out_token");
		tok = -77;
		StatusIs(sslm_gpu_seq_rows_produced(s, &tok), S::SSLM_SEQUENCE_REJECTED, base + ": a fresh handle");
		Check(tok == -77, base + ": a fresh handle: out_token written");
		Hist h;
		Check(Replay(w, m, s, Prompt(12, 22), 1, &h), base + ": history");
		tok = -77;
		StatusIs(sslm_gpu_seq_rows_produced(s, &tok), S::SSLM_SEQUENCE_REJECTED, base + ": after the prompt and an Off token");
		// The twin gives the drafts: Off's next three tokens.
		SslmGpuSequenceHandle* t = w.Seq(pooled != 0, 200, kLongKey);
		Hist ht;
		if (!t || !Replay(w, m, t, Prompt(12, 22), 1, &ht)) { Check(false, base + ": twin"); continue; }
		TwinDecodes(w, m, t, &ht, 4);
		const std::vector<int32_t> next(ht.toks.begin() + 1, ht.toks.end());
		// A chunk open (EXACT_DRAFT): BUSY, out untouched.
		const std::vector<int32_t> rows = {h.pend, next[0], next[1], next[2]};
		int32_t adm = -9;
		StatusIs(EmbedRows(w, s, kExactDraft, rows, 0, &adm), S::SSLM_OK, base + ": embed_rows(EXACT_DRAFT)");
		tok = -77;
		StatusIs(sslm_gpu_seq_rows_produced(s, &tok), S::SSLM_BUSY, base + ": a chunk open");
		Check(tok == -77, base + ": a chunk open: out_token written");
		// The abandon: no close, so SEQUENCE_REJECTED.
		StatusIs(sslm_gpu_seq_rows_abandon(w.ctx, s), S::SSLM_OK, base + ": abandon");
		tok = -77;
		StatusIs(sslm_gpu_seq_rows_produced(s, &tok), S::SSLM_SEQUENCE_REJECTED, base + ": after an abandon");
		// The Exact close: Off's token after 4 rows (all accepted).
		const ExOut e = RunExact(w, m, s, rows);
		StatusIs(e.embed, S::SSLM_OK, base + ": embed_rows again");
		StatusIs(e.close, S::SSLM_OK, base + ": the close");
		StatusIs(e.prod, S::SSLM_OK, base + ": rows_produced after an Exact close");
		Check(e.tok == ht.toks[4], F("%s: rows_produced %d, Off's next %d", base.c_str(), e.tok, ht.toks[4]));
		// A restore has no close.
		const std::vector<uint8_t> b = Blob(w, s, base);
		SslmGpuSequenceHandle* r = nullptr;
		S rs = S::SSLM_OK;
		if (pooled) {
			SslmGpuKvPoolHandle* p = w.Pool(60);
			uint32_t shared = 0;
			rs = sslm_gpu_seq_restore_into_pool(w.ctx, w.model, p, b.data(), b.size(), nullptr, &r, &shared);
		} else {
			rs = sslm_gpu_seq_restore(w.ctx, w.model, b.data(), b.size(), &r);
		}
		StatusIs(rs, S::SSLM_OK, base + ": restore");
		if (r) {
			w.seqs.push_back(r);
			tok = -77;
			StatusIs(sslm_gpu_seq_rows_produced(r, &tok), S::SSLM_SEQUENCE_REJECTED, base + ": a restored handle");
		}
		// A reset clears it.
		StatusIs(sslm_gpu_seq_reset(w.ctx, s), S::SSLM_OK, base + ": reset");
		tok = -77;
		StatusIs(sslm_gpu_seq_rows_produced(s, &tok), S::SSLM_SEQUENCE_REJECTED, base + ": after a reset");
		// A PROMPT close: SEQUENCE_REJECTED.
		Hist h2;
		SslmGpuSequenceHandle* q = w.Seq(pooled != 0, 200, kLongKey);
		if (q && Replay(w, m, q, Prompt(12, 23), 0, &h2)) {
			const std::vector<int32_t> c = Chain(w, q);
			if (c.size() >= 3 && EmbedRows(w, q, SSLM_GPU_ROWS_PROMPT, {c[0], c[1], c[2]}, 0, &adm) == S::SSLM_OK) {
				StatusIs(CloseChunk(w, m, q), S::SSLM_OK, base + ": PROMPT close");
				tok = -77;
				StatusIs(sslm_gpu_seq_rows_produced(q, &tok), S::SSLM_SEQUENCE_REJECTED, base + ": after a PROMPT close");
				Check(tok == -77, base + ": after a PROMPT close: out_token written");
			} else {
				Check(false, base + ": the PROMPT chunk");
			}
		}
	}
}

// ------------------------------------------------------------------------------------------------ 2.3x
void Cell23x(Fsfd& fx) {
	const Model& m = fx.m;
	const std::vector<int32_t> prompt = Prompt(12, 23);
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int M : {2, 4, 8})
			for (int k = 1; k < M; ++k) {
				const std::string tag = F("%s M%d, a special id at row %d", pooled ? "pooled" : "dedicated", M, k);
				World w(m);
				if (!w.ok) { Check(false, tag); continue; }
				SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 200, kLongKey);
				Hist h;
				if (!s || !Replay(w, m, s, prompt, 1, &h)) { Check(false, tag + ": history"); continue; }
				std::vector<int32_t> c = Chain(w, s);  // the chain from the state after p: legal rows by construction
				if (static_cast<int>(c.size()) < M) { Check(false, F("%s: chain of %zu", tag.c_str(), c.size())); continue; }
				std::vector<int32_t> rows = {h.pend};
				rows.insert(rows.end(), c.begin(), c.begin() + (M - 1));
				rows[static_cast<size_t>(k)] = kSpecial;
				const Obs before = Observe(w, s);
				int32_t adm = -9;
				StatusIs(EmbedRows(w, s, kExactDraft, rows, 0, &adm), S::SSLM_OK, tag);
				Check(adm == k && RowsOpen(s) == k, F("%s: admitted %d rows_open %d, want %d", tag.c_str(), adm, RowsOpen(s), k));
				if (adm > 0) StatusIs(sslm_gpu_seq_rows_abandon(w.ctx, s), S::SSLM_OK, tag + ": abandon");
				Unchanged(w, s, before, tag + ": after the abandon");
			}
	// The checks before admission, each writing nothing.
	for (int pooled = 0; pooled <= 1; ++pooled) {
		const std::string base = pooled ? "pooled" : "dedicated";
		struct Z {
			std::string name;
			uint32_t flags;
			int32_t bad_at;   // a row set to `bad`
			int32_t bad;
			int room;         // headroom after the history (-1: plenty)
			S want;
			int32_t admitted; // when want is OK
		};
		const std::vector<Z> zs = {
		    {"flag 0x1 (the removed FIRST_PREWALKED)", 0x1u, -1, 0, -1, kArg, 0},
		    {"flag 0x2", 0x2u, -1, 0, -1, kArg, 0},
		    {"a draft id = vocab", 0, 2, m.vocab(), -1, S::SSLM_TOKEN_ID_OUT_OF_RANGE, 0},
		    {"a draft id -1", 0, 1, -1, -1, S::SSLM_TOKEN_ID_OUT_OF_RANGE, 0},
		    {"headroom 1", 0, -1, 0, 1, S::SSLM_OK, 1},
		    {"headroom 3", 0, -1, 0, 3, S::SSLM_OK, 3},
		    {"no headroom", 0, -1, 0, 0, pooled ? S::SSLM_GPU_KV_BUDGET_EXCEEDED : S::SSLM_GPU_CONTEXT_CAP_EXCEEDED, 0},
		};
		for (const Z& z : zs) {
			const std::string tag = base + ": " + z.name;
			World w(m);
			if (!w.ok) { Check(false, tag); continue; }
			const int32_t cap = z.room < 0 ? 200 : static_cast<int32_t>(prompt.size()) + 1 + z.room;
			SslmGpuSequenceHandle* s = w.Seq(pooled != 0, cap, kLongKey);
			Hist h;
			if (!s || !Replay(w, m, s, prompt, 1, &h)) { Check(false, tag + ": history"); continue; }
			std::vector<int32_t> c = Chain(w, s);
			if (c.size() < 5) { Check(false, tag + ": chain"); continue; }
			std::vector<int32_t> rows = {h.pend, c[0], c[1], c[2], c[3]};
			if (z.bad_at >= 0) rows[static_cast<size_t>(z.bad_at)] = z.bad;
			const Obs before = Observe(w, s);
			int32_t adm = -9;
			SslmGpuRowsSpec spec{sizeof(SslmGpuRowsSpec), kExactDraft, rows.data(), static_cast<int32_t>(rows.size()), z.flags};
			StatusIs(sslm_gpu_seq_embed_rows(w.ctx, s, &spec, &adm), z.want, tag);
			if (z.want == S::SSLM_OK) {
				Check(adm == z.admitted, F("%s: admitted %d, want %d", tag.c_str(), adm, z.admitted));
				(void)sslm_gpu_seq_rows_abandon(w.ctx, s);
			} else {
				Check(adm == -9, F("%s: out_admitted written (%d)", tag.c_str(), adm));
			}
			Unchanged(w, s, before, tag);
		}
	}
}

// ------------------------------------------------------------------------------------------------ 7.17
// One case: a history of `n` Off tokens (n >= 1; the last in hand), then `chunks` Exact chunks of M rows whose drafts
// are Off's own next tokens (all accepted), each graded against the twin. `wrong_at` > 0 makes the LAST chunk's row
// `wrong_at` a legal wrong token (choice cases only; see 7.18) -- not used here.
void ExactAgainstTwin(const Model& m, const char* schema, const std::vector<int32_t>& prompt, int n, int chunks, int M, bool pooled,
                      uint32_t budget, const std::string& tag) {
	World w(m);
	if (!w.ok) { Check(false, tag); return; }
	SslmGpuSequenceHandle* s = w.Seq(pooled, 200, schema);
	SslmGpuSequenceHandle* t = w.Seq(pooled, 200, schema);
	Hist hs, ht;
	if (!s || !t || !Replay(w, m, s, prompt, n, &hs) || !Replay(w, m, t, prompt, n, &ht)) { Check(false, tag + ": history"); return; }
	// Off's next tokens, from a third handle, so the twin stays at the chunk boundaries.
	SslmGpuSequenceHandle* o = w.Seq(pooled, 200, schema);
	Hist ho;
	if (!o || !Replay(w, m, o, prompt, n, &ho)) { Check(false, tag + ": oracle"); return; }
	TwinDecodes(w, m, o, &ho, chunks * M + 2);
	int32_t pend = hs.pend;
	size_t at = static_cast<size_t>(n);  // ho.toks[at] is Off's next token after pend
	for (int ch = 0; ch < chunks; ++ch) {
		const std::string tc = F("%s, chunk %d", tag.c_str(), ch);
		if (ho.toks.size() < at + static_cast<size_t>(M)) { Check(false, tc + ": the stream ended (pick another prompt)"); return; }
		std::vector<int32_t> rows = {pend};
		for (int r = 1; r < M; ++r) rows.push_back(ho.toks[at + static_cast<size_t>(r) - 1]);
		const ExOut e = RunExact(w, m, s, rows, budget);
		StatusIs(e.embed, S::SSLM_OK, tc + ": embed_rows(EXACT_DRAFT)");
		Check(e.adm == M, F("%s: admitted %d of %d (Off's own tokens are legal)", tc.c_str(), e.adm, M));
		StatusIs(e.close, S::SSLM_OK, tc + ": the close");
		StatusIs(e.prod, S::SSLM_OK, tc + ": rows_produced");
		const int32_t want = ho.toks[at + static_cast<size_t>(M) - 1];
		Check(e.tok == want, F("%s: produced %d, Off's token %d", tc.c_str(), e.tok, want));
		Check(LastCommitted(s) == M, F("%s: last_rows_committed %d, want %d", tc.c_str(), LastCommitted(s), M));
		TwinDecodes(w, m, t, &ht, M);
		Check(ReadyFlag(w, s) == 0 && ReadyFlag(w, t) == 0, F("%s: ready %d (twin %d), want 0 after a produced token", tc.c_str(), ReadyFlag(w, s), ReadyFlag(w, t)));
		Check(Walk(s) == Walk(t), F("%s: walk %u, Off's %u (the produced token walked once)", tc.c_str(), Walk(s), Walk(t)));
		BlobEq(Blob(w, s, tc), Blob(w, t, tc), tc + ": the whole blob");
		pend = e.tok >= 0 ? e.tok : ht.pend;
		at += static_cast<size_t>(M);
	}
	// And Off continues from the Exact handle as from the twin.
	int32_t a = -1, b = -1;
	Check(Dec(w, m, s, pend, &a) == S::SSLM_OK && Dec(w, m, t, ht.pend, &b) == S::SSLM_OK && a == b, F("%s: the next Off token %d, twin %d", tag.c_str(), a, b));
	BlobEq(Blob(w, s, tag), Blob(w, t, tag), tag + ": after one more Off token");
}

void Cell717(Fsfd& fx) {
	const Model& m = fx.m;
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int M : {2, 4, 8}) {
			const char* pk = pooled ? "pooled" : "dedicated";
			ExactAgainstTwin(m, kLongKey, Prompt(12, 717), 1, 1, M, pooled != 0, 0, F("%s M%d, the first chunk after the prompt's finish", pk, M));
			ExactAgainstTwin(m, kPotion, Prompt(12, 718), 3, 1, M, pooled != 0, 0, F("%s M%d, after an Off token", pk, M));
			ExactAgainstTwin(m, kLongKey, Prompt(12, 719), 1, 3, M, pooled != 0, m.full_budget(), F("%s M%d, after Exact closes (full slices)", pk, M));
		}
	// The dead end: a stream that ends with -2 (the accepting state's empty mask). The chunk ends at the row whose
	// finish has no transition: rows_produced -2, ready set, the walk kept; Off's blob after its -2.
	int found = 0;
	for (const char* schema : kSchemas)
		for (uint32_t stream = 1; stream < 30 && found < 6; ++stream) {
			const std::vector<int32_t> prompt = Prompt(12, 700 + stream);
			const OffRef off = RunOff(m, schema, prompt, 160);
			if (!off.ok || off.dead < 0) continue;
			for (int pooled = 0; pooled <= 1; ++pooled)
				for (int M : {1, 3, 8}) {
					const int e = off.dead;  // the decode that embeds o_(e-1) returns -2
					const int j = e - M;     // p = o_j in hand; rows o_j .. o_(e-1)
					if (j < 0) continue;
					const std::string tag = F("%s %s stream %u, dead end after %d tokens, M%d", pooled ? "pooled" : "dedicated", schema, stream, e, M);
					World w(m);
					if (!w.ok) { Check(false, tag); continue; }
					SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 240, schema);
					SslmGpuSequenceHandle* t = w.Seq(pooled != 0, 240, schema);
					Hist hs, ht;
					if (!s || !t || !Replay(w, m, s, prompt, j + 1, &hs) || !Replay(w, m, t, prompt, j + 1, &ht)) { Check(false, tag + ": history"); continue; }
					std::vector<int32_t> rows(off.toks.begin() + j, off.toks.begin() + e);
					const ExOut x = RunExact(w, m, s, rows);
					StatusIs(x.embed, S::SSLM_OK, tag + ": embed_rows");
					Check(x.adm == M, F("%s: admitted %d of %d", tag.c_str(), x.adm, M));
					StatusIs(x.close, S::SSLM_OK, tag + ": the close");
					StatusIs(x.prod, S::SSLM_OK, tag + ": rows_produced");
					Check(x.tok == -2, F("%s: rows_produced %d, want -2 (dead end)", tag.c_str(), x.tok));
					S tst = S::SSLM_OK;
					const int32_t last = TwinDecodes(w, m, t, &ht, M, &tst);
					Check(tst == S::SSLM_OK && last == -2, F("%s: the Off twin's dead end (%s, %d)", tag.c_str(), N(tst), last));
					Check(ReadyFlag(w, s) == 1, F("%s: ready %d, want 1 at a dead end (Off re-arms it; MX24)", tag.c_str(), ReadyFlag(w, s)));
					Check(Walk(s) == Walk(t), F("%s: walk %u, Off's %u", tag.c_str(), Walk(s), Walk(t)));
					BlobEq(Blob(w, s, tag), Blob(w, t, tag), tag + ": the whole blob, ready word included");
					// One more decode-bridge call on each: -2 at OK, no token consumed, equal again.
					const int64_t len = CtxLen(s);
					int32_t a = -1, b = -1;
					StatusIs(Dec(w, m, s, off.toks[static_cast<size_t>(e) - 1], &a), S::SSLM_OK, tag + ": the following decode");
					(void)Dec(w, m, t, off.toks[static_cast<size_t>(e) - 1], &b);
					Check(a == -2 && b == -2 && CtxLen(s) == len, F("%s: the following decode gave %d (Off %d), ctx %lld -> %lld", tag.c_str(), a, b,
					                                               (long long)len, (long long)CtxLen(s)));
					BlobEq(Blob(w, s, tag), Blob(w, t, tag), tag + ": after the following decode");
					++found;
				}
		}
	Check(found >= 4, F("only %d dead-end cases were found (want 4)", found));
}

// ------------------------------------------------------------------------------------------------ 7.18 / 7.10 (b)
// A chunk whose row `a` is a legal wrong token at a choice state: Choice ch, rows [o_j .. o_c, X, tail...] with
// j = c - a + 1. Accepted a; produced o_(c+1). `sat_row` >= 0 arms S-GSATROW at that chunk row (7.10 (b)).
void PartialCommit(const Model& m, const Choice& ch, int M, int a, bool pooled, int sat_row, const std::string& tag) {
	const int j = ch.c - a + 1;
	if (j < 0) { Check(false, tag + ": the choice is too early for this a"); return; }
	std::vector<int32_t> rows(ch.off.toks.begin() + j, ch.off.toks.begin() + ch.c + 1);  // a rows: o_j .. o_c
	if (a < M) {
		rows.push_back(ch.x);
		for (size_t i = 0; static_cast<int>(rows.size()) < M && i < ch.tail.size(); ++i) rows.push_back(ch.tail[i]);
	} else {
		for (int i = ch.c + 1; static_cast<int>(rows.size()) < M && i < static_cast<int>(ch.off.toks.size()); ++i) rows.push_back(ch.off.toks[static_cast<size_t>(i)]);
	}
	const int Mx = static_cast<int>(rows.size());
	World w(m);
	if (!w.ok) { Check(false, tag); return; }
	SslmGpuKvPoolHandle *ps = nullptr, *pt = nullptr;
	SslmGpuSequenceHandle* s = w.Seq(pooled, 200, ch.schema, &ps);
	SslmGpuSequenceHandle* t = w.Seq(pooled, 200, ch.schema, &pt);
	Hist hs, ht;
	if (!s || !t || !Replay(w, m, s, ch.prompt, j + 1, &hs) || !Replay(w, m, t, ch.prompt, j + 1, &ht)) { Check(false, tag + ": history"); return; }
	const mr2::Counters entry = mr2::ReadCounters(s);
	if (sat_row >= 0) ArmGpuSatRowForTest(sat_row);
	const ExOut e = RunExact(w, m, s, rows);
	if (sat_row >= 0) {
		Check(GpuSatRowArmedForTest() == 0, tag + F(": S-GSATROW at row %d never fired (the seam: mr5_entry.h)", sat_row));
		ArmGpuSatRowForTest(-1);
	}
	StatusIs(e.embed, S::SSLM_OK, tag + ": embed_rows");
	Check(e.adm == Mx, F("%s: admitted %d of %d (every row is legal: fsfd_ref.txt's paths)", tag.c_str(), e.adm, Mx));
	StatusIs(e.close, S::SSLM_OK, tag + ": the close");
	StatusIs(e.prod, S::SSLM_OK, tag + ": rows_produced");
	const int want_a = a < M ? a : Mx;
	Check(LastCommitted(s) == want_a, F("%s: last_rows_committed %d, want %d", tag.c_str(), LastCommitted(s), want_a));
	TwinDecodes(w, m, t, &ht, want_a);
	Check(e.tok == ht.pend, F("%s: produced %d, Off's %d", tag.c_str(), e.tok, ht.pend));
	Check(Walk(s) == Walk(t), F("%s: walk %u, Off's %u", tag.c_str(), Walk(s), Walk(t)));
	Check(CtxLen(s) == CtxLen(t), F("%s: ctx %lld, Off's %lld", tag.c_str(), (long long)CtxLen(s), (long long)CtxLen(t)));
	if (pooled) {
		Check(Mapped(s) == Mapped(t), F("%s: mapped pages %u, Off's %u", tag.c_str(), Mapped(s), Mapped(t)));
		Check(PoolFree(ps) == PoolFree(pt), F("%s: the pool's free pages %u, Off's %u", tag.c_str(), PoolFree(ps), PoolFree(pt)));
	}
	const mr2::Counters cs = mr2::ReadCounters(s), ct = mr2::ReadCounters(t);
	Check(cs == ct, F("%s: counters %s, Off's %s (entry %s)", tag.c_str(), cs.Str().c_str(), ct.Str().c_str(), entry.Str().c_str()));
	BlobEq(Blob(w, s, tag), Blob(w, t, tag), tag + ": the whole blob (mirror rows above the commit untouched)");
}

Choice g_choice;
bool g_choice_done = false;
const Choice& TheChoice(const Model& m) {
	if (!g_choice_done) {
		g_choice = FindChoice(m, 7, 7);
		g_choice_done = true;
		if (g_choice.c >= 0)
			std::printf("choice: %s, Off token %d at index %d reaches state %u; legal wrong token %d, tail %zu\n", g_choice.schema,
			            g_choice.off.toks[static_cast<size_t>(g_choice.c)], g_choice.c, g_choice.off.walk[static_cast<size_t>(g_choice.c)], g_choice.x,
			            g_choice.tail.size());
	}
	return g_choice;
}
// The reference's state ids are the engine's: every forced state the choice stream passes has the reference's chain.
void CheckRefNumbering(const Model& m, const Choice& ch) {
	World w(m);
	SslmGpuSequenceHandle* s = w.Seq(false, 240, ch.schema);
	Hist h;
	if (!s || !Replay(w, m, s, ch.prompt, 0, &h)) return;
	const RefSchema& rs = g_ref[ch.schema];
	int checked = 0;
	int32_t last = ch.prompt.back();
	for (size_t i = 0; i < ch.off.toks.size() && i < 24; ++i) {
		int32_t t = -1;
		if (Dec(w, m, s, last, &t) != S::SSLM_OK || t < 0) break;
		last = t;
		const uint32_t st = Walk(s);
		if (st < rs.states.size() && rs.states[st].forced) {
			Check(Chain(w, s) == rs.states[st].chain, F("fsfd_ref.txt state %u: the engine's chain %s, the reference's %s", st, Ids(Chain(w, s)).c_str(),
			                                             Ids(rs.states[st].chain).c_str()));
			++checked;
		}
	}
	Check(checked > 0, "no forced state checked against fsfd_ref.txt");
}

void Cell718(Fsfd& fx) {
	const Model& m = fx.m;
	const Choice& ch = TheChoice(m);
	Check(ch.c >= 0, "no choice point with a known legal wrong token was found (fsfd_ref.txt)");
	if (ch.c < 0) return;
	CheckRefNumbering(m, ch);
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int M : {2, 4, 8})
			for (int a = 1; a <= M; ++a)
				PartialCommit(m, ch, M, a, pooled != 0, -1, F("%s M%d a=%d", pooled ? "pooled" : "dedicated", M, a));
}

// ------------------------------------------------------------------------------------------------ 7.10
void Cell710(Fsfd& fx) {
	const Model& m = fx.m;
	// (a) Row 1 saturates and a = 3 (M = 3, all accepted): counters = entry + the deltas of rows 0..2 (MX19).
	for (int pooled = 0; pooled <= 1; ++pooled) {
		const std::string tag = F("%s: row 1 saturates, a = 3", pooled ? "pooled" : "dedicated");
		World w(m);
		if (!w.ok) { Check(false, tag); continue; }
		SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 200, kLongKey);
		SslmGpuSequenceHandle* t = w.Seq(pooled != 0, 200, kLongKey);
		SslmGpuSequenceHandle* o = w.Seq(pooled != 0, 200, kLongKey);
		Hist hs, ht, ho;
		const std::vector<int32_t> p = Prompt(12, 710);
		if (!s || !t || !o || !Replay(w, m, s, p, 2, &hs) || !Replay(w, m, t, p, 2, &ht) || !Replay(w, m, o, p, 2, &ho)) { Check(false, tag + ": history"); continue; }
		TwinDecodes(w, m, o, &ho, 3);
		const mr2::Counters entry = mr2::ReadCounters(s);
		ArmGpuSatRowForTest(1);
		const ExOut e = RunExact(w, m, s, {hs.pend, ho.toks[2], ho.toks[3]});
		Check(GpuSatRowArmedForTest() == 0, tag + ": S-GSATROW never fired in the chunk (the seam: mr5_entry.h)");
		ArmGpuSatRowForTest(-1);
		StatusIs(e.close, S::SSLM_OK, tag + ": the close");
		Check(e.adm == 3 && LastCommitted(s) == 3, F("%s: admitted %d, committed %d, want 3", tag.c_str(), e.adm, LastCommitted(s)));
		// The Off twin: the same row (its second token of the three) saturates.
		int32_t x = -1;
		S st = Dec(w, m, t, ht.pend, &x);
		ArmGpuSatRowForTest(0);
		int32_t y = -1;
		st = st == S::SSLM_OK ? Dec(w, m, t, x, &y) : st;
		Check(GpuSatRowArmedForTest() == 0, tag + ": S-GSATROW never fired on the Off twin");
		ArmGpuSatRowForTest(-1);
		int32_t z = -1;
		st = st == S::SSLM_OK ? Dec(w, m, t, y, &z) : st;
		StatusIs(st, S::SSLM_OK, tag + ": the Off twin");
		const mr2::Counters cs = mr2::ReadCounters(s), ct = mr2::ReadCounters(t);
		mr2::Counters want = entry;
		for (int i = 0; i < 5; ++i) want.v[i] += 1;  // row 1's delta: one per counter; rows 0 and 2 add none on F-SFD
		Check(cs == want, F("%s: counters %s, want entry + rows 0..2 = %s (MX19)", tag.c_str(), cs.Str().c_str(), want.Str().c_str()));
		Check(ct == want, F("%s: the Off twin's counters %s, want %s (the seam's own check)", tag.c_str(), ct.Str().c_str(), want.Str().c_str()));
		Check(e.tok == z, F("%s: produced %d, Off's %d", tag.c_str(), e.tok, z));
		BlobEq(Blob(w, s, tag), Blob(w, t, tag), tag + ": the SLM5/SLM6 blob");
	}
	// (b) A saturating REJECTED row r > a - 1 is excluded (MX23): the choice chunk with S-GSATROW at row a + 1.
	const Choice& ch = TheChoice(m);
	Check(ch.c >= 0, "no choice point for 7.10 (b)");
	if (ch.c < 0) return;
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int a : {2, 4}) PartialCommit(m, ch, 8, a, pooled != 0, a + 1, F("%s: a = %d, rejected row %d saturates", pooled ? "pooled" : "dedicated", a, a + 1));
	// (c) GX.1, GX.2, GX.4-GX.13 on this backend: the gx.* cells.
}

// ------------------------------------------------------------------------------------------------ 7.19
void Cell719(Fsfd& fx) {
	const Model& m = fx.m;
	const int M = 4;
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int r : {0, 1, M - 1}) {
			const std::string tag = F("%s M%d, NarrowRowChecked fails at row %d", pooled ? "pooled" : "dedicated", M, r);
			World w(m);
			if (!w.ok) { Check(false, tag); continue; }
			SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 200, kLongKey);
			SslmGpuSequenceHandle* t = w.Seq(pooled != 0, 200, kLongKey);
			SslmGpuSequenceHandle* o = w.Seq(pooled != 0, 200, kLongKey);
			Hist hs, ht, ho;
			const std::vector<int32_t> p = Prompt(12, 719);
			if (!s || !t || !o || !Replay(w, m, s, p, 2, &hs) || !Replay(w, m, t, p, 2, &ht) || !Replay(w, m, o, p, 2, &ho)) { Check(false, tag + ": history"); continue; }
			TwinDecodes(w, m, o, &ho, M);
			std::vector<int32_t> rows = {hs.pend};
			for (int i = 0; i < M - 1; ++i) rows.push_back(ho.toks[2 + static_cast<size_t>(i)]);
			// Off: r clean tokens, then the finish of row r fails.
			S tst = S::SSLM_OK;
			if (r > 0) TwinDecodes(w, m, t, &ht, r, &tst);
			ArmGpuFinishNarrowFaultForTest(0);
			int32_t tok = -1;
			const S off_status = Dec(w, m, t, ht.pend, &tok);
			ArmGpuFinishNarrowFaultForTest(-1);
			Check(off_status != S::SSLM_OK, F("%s: the Off twin's finish did not fail (%s): S-GNARROW is not built (mr5_entry.h)", tag.c_str(), N(off_status)));
			// Exact: the close's row r.
			int32_t adm = -9;
			StatusIs(EmbedRows(w, s, kExactDraft, rows, 0, &adm), S::SSLM_OK, tag + ": embed_rows");
			ArmGpuFinishNarrowFaultForTest(r);
			(void)CloseChunk(w, m, s);
			ArmGpuFinishNarrowFaultForTest(-1);
			int32_t out = -77;
			const S ps = sslm_gpu_seq_rows_produced(s, &out);
			Check(ps == off_status, F("%s: rows_produced returned %s, Off's finish %s", tag.c_str(), N(ps), N(off_status)));
			Check(out == -77, F("%s: out_token written (%d) with a failure status", tag.c_str(), out));
			Check(LastCommitted(s) == r + 1, F("%s: last_rows_committed %d, want %d (rows 0..%d, as Off committed them)", tag.c_str(), LastCommitted(s), r + 1, r));
			Check(ReadyFlag(w, s) == ReadyFlag(w, t), F("%s: ready %d, Off's %d", tag.c_str(), ReadyFlag(w, s), ReadyFlag(w, t)));
			BlobEq(Blob(w, s, tag), Blob(w, t, tag), tag + ": the whole blob at the finish failure");
		}
}

// ------------------------------------------------------------------------------------------------ 7.11e
void Cell711e(Fsfd& fx) {
	const Model& m = fx.m;
	for (int pooled = 0; pooled <= 1; ++pooled) {
		const std::string base = F("%s, no device logits head", pooled ? "pooled" : "dedicated");
		World w(m, /*head_on_device=*/false);
		if (!w.ok) { Check(false, base); continue; }
		SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 200, kLongKey);
		SslmGpuSequenceHandle* t = w.Seq(pooled != 0, 200, kLongKey);
		Hist hs, ht;
		const std::vector<int32_t> p = Prompt(12, 711);
		if (!s || !t || !Replay(w, m, s, p, 1, &hs) || !Replay(w, m, t, p, 1, &ht)) { Check(false, base + ": history"); continue; }
		const std::vector<int32_t> c = Chain(w, s);
		std::vector<int32_t> rows = {hs.pend};
		for (size_t i = 0; i < c.size() && rows.size() < 4; ++i) rows.push_back(c[i]);
		const Obs before = Observe(w, s);
		int32_t adm = -9;
		StatusIs(EmbedRows(w, s, kExactDraft, rows, 0, &adm), S::SSLM_SEQUENCE_REJECTED, base + ": embed_rows(EXACT_DRAFT)");
		Check(adm == -9, F("%s: out_admitted written (%d)", base.c_str(), adm));
		Unchanged(w, s, before, base + ": the refusal");
		// The jump verb still works there: Off's tokens and bytes.
		SslmGpuSequenceHandle* o = w.Seq(pooled != 0, 200, kLongKey);
		Hist ho;
		if (!o || !Replay(w, m, o, p, 1, &ho)) { Check(false, base + ": oracle"); continue; }
		TwinDecodes(w, m, o, &ho, 12);
		const std::vector<int32_t> next(ho.toks.begin() + 1, ho.toks.end());
		const Jump j = DoJump(w, s, hs.pend, 8, kNull);
		StatusIs(j.st, S::SSLM_OK, base + ": the jump (NULL options)");
		const int32_t k = Kept(c, 8, next);
		Check(j.count == k, F("%s: kept %d, want %d (the chain %s against Off's %s)", base.c_str(), j.count, k, Ids(c).c_str(), Ids(next).c_str()));
		int32_t a = -1;
		Check(Dec(w, m, s, j.count > 0 ? j.out[static_cast<size_t>(j.count) - 1] : hs.pend, &a) == S::SSLM_OK, base + ": the decode after");
		TwinDecodes(w, m, t, &ht, std::max(k, 0) + 1);
		Check(a == ht.pend, F("%s: the token after the jump %d, Off's %d", base.c_str(), a, ht.pend));
		BlobEq(Blob(w, s, base), Blob(w, t, base), base + ": after the jump and a decode");
	}
}

// ------------------------------------------------------------------------------------------------ 8.4
void Cell84(Fsfd& fx) {
	const Model& m = fx.m;
	const int T = 40;
	for (const char* schema : kSchemas)
		for (int pooled = 0; pooled <= 1; ++pooled)
			for (int M : {2, 4, 8})
				for (int full = 0; full <= 1; ++full) {
					const std::string tag = F("%s %s M%d %s slices", pooled ? "pooled" : "dedicated", schema, M, full ? "full" : "one-layer");
					const std::vector<int32_t> p = Prompt(12, 84);
					World w(m);
					if (!w.ok) { Check(false, tag); continue; }
					SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 240, schema);
					SslmGpuSequenceHandle* t = w.Seq(pooled != 0, 240, schema);
					Hist hs, ht;
					if (!s || !t || !Replay(w, m, s, p, 1, &hs) || !Replay(w, m, t, p, 1, &ht)) { Check(false, tag + ": history"); continue; }
					std::vector<int32_t> stream = hs.toks;
					int32_t pend = hs.pend;
					int closes = 0;
					bool dead = false;
					while (static_cast<int>(stream.size()) < T && !dead) {
						// The plugin's draft: the engine's forced chain from the state after p.
						const std::vector<int32_t> c = Chain(w, s);
						std::vector<int32_t> rows = {pend};
						for (size_t i = 0; i < c.size() && static_cast<int>(rows.size()) < M; ++i) rows.push_back(c[i]);
						const ExOut e = RunExact(w, m, s, rows, full ? m.full_budget() : 0);
						if (e.embed != S::SSLM_OK || e.close != S::SSLM_OK || e.prod != S::SSLM_OK) {
							Check(false, F("%s: chunk %d: embed %s close %s rows_produced %s", tag.c_str(), closes, N(e.embed), N(e.close), N(e.prod)));
							break;
						}
						const int32_t a = LastCommitted(s);
						// Off over the same rows: a decodes.
						const int32_t want = TwinDecodes(w, m, t, &ht, a);
						for (int r = 1; r < a; ++r) stream.push_back(rows[static_cast<size_t>(r)]);
						Check(a >= 1 && a <= static_cast<int32_t>(rows.size()) && e.tok == want,
						      F("%s: chunk %d: committed %d, produced %d, Off's %d", tag.c_str(), closes, a, e.tok, want));
						BlobEq(Blob(w, s, tag), Blob(w, t, tag), F("%s: after chunk %d (%zu tokens)", tag.c_str(), closes, stream.size()));
						++closes;
						if (e.tok == -2) dead = true;
						else { stream.push_back(e.tok); pend = e.tok; }
						if (e.tok != want) break;
					}
					const OffRef off = RunOff(m, schema, p, static_cast<int>(stream.size()));
					const std::vector<int32_t> want(off.toks.begin(), off.toks.begin() + static_cast<long>(std::min(off.toks.size(), stream.size())));
					Check(stream == want, F("%s: the Exact stream %s, Off's %s", tag.c_str(), Ids(stream).c_str(), Ids(want).c_str()));
				}
}

// ------------------------------------------------------------------------------------------------ gx.1 / gx.2
void CellGx1(Fsfd& fx) {
	const Model& m = fx.m;
	const int T = 48;
	for (const char* schema : kSchemas) {
		const std::vector<int32_t> p = Prompt(12, 901);
		const OffRef off = RunOff(m, schema, p, T);
		const CpuOff cpu = RunCpuOff(m, schema, p, T);
		Check(off.ok && cpu.ok, F("%s: the Off references (GPU %d, CPU %d)", schema, off.ok, cpu.ok));
		const size_t n = std::min(off.toks.size(), cpu.toks.size());
		Check(n > 0 && std::equal(off.toks.begin(), off.toks.begin() + static_cast<long>(n), cpu.toks.begin()),
		      F("%s: GPU Off %s differs from CPU Off %s (the base's own parity)", schema, Ids(off.toks).c_str(), Ids(cpu.toks).c_str()));
		for (int pooled = 0; pooled <= 1; ++pooled)
			for (int32_t budget : {1, 2, 4, 8, 17, 64}) {
				const std::string tag = F("%s %s budget %d", pooled ? "pooled" : "dedicated", schema, budget);
				World w(m);
				if (!w.ok) { Check(false, tag); continue; }
				SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 240, schema);
				Hist h;
				if (!s || !Replay(w, m, s, p, 1, &h)) { Check(false, tag + ": history"); continue; }
				const LoopOut o = ExactLoop(w, m, s, &h, budget, T, nullptr, nullptr, tag);
				const size_t k = std::min(o.toks.size(), off.toks.size());
				Check(o.ok && o.toks.size() == std::min<size_t>(T, off.toks.size()) &&
				          std::equal(o.toks.begin(), o.toks.begin() + static_cast<long>(k), off.toks.begin()),
				      F("%s: the Exact jump loop %s, Off's %s", tag.c_str(), Ids(o.toks).c_str(), Ids(off.toks).c_str()));
				Check(o.dead == (off.dead >= 0 && off.dead < T), F("%s: dead end %d, Off's %d", tag.c_str(), o.dead, off.dead));
			}
	}
}

int CountPoisonPages(SslmGpuKvPoolHandle* pool, uint32_t pages) {
	int n = 0;
	for (uint32_t pg = 0; pg < pages; ++pg) {
		uint8_t b[64];
		if (sslm_gpu_test_only_peek_pool_mirror_bytes(pool, pg, 0, sizeof b, b) != S::SSLM_OK) continue;
		bool all = true;
		for (uint8_t x : b) all = all && x == 0xCD;
		n += all;
	}
	return n;
}

void CellGx2(Fsfd& fx) {
	const Model& m = fx.m;
	const int T = 40;
	for (const char* schema : {kPotion, kLongKey})
		for (int pooled = 0; pooled <= 1; ++pooled)
			for (int32_t budget : {4, 17}) {
				const std::string tag = F("%s %s budget %d", pooled ? "pooled" : "dedicated", schema, budget);
				const std::vector<int32_t> p = Prompt(12, 902);
				World w(m);
				if (!w.ok) { Check(false, tag); continue; }
				SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 240, schema);
				SslmGpuSequenceHandle* t = w.Seq(pooled != 0, 240, schema);
				Hist hs, ht;
				if (!s || !t || !Replay(w, m, s, p, 1, &hs) || !Replay(w, m, t, p, 1, &ht)) { Check(false, tag + ": history"); continue; }
				int compared = 0;
				const LoopOut o = ExactLoop(w, m, s, &hs, budget, T, nullptr,
				                            [&](size_t count) {
					                            const int k = static_cast<int>(count) - static_cast<int>(ht.toks.size());
					                            if (k > 0) TwinDecodes(w, m, t, &ht, k);
					                            BlobEq(Blob(w, s, tag), Blob(w, t, tag), F("%s: after %zu tokens", tag.c_str(), count));
					                            if (pooled) Check(Mapped(s) == Mapped(t), F("%s: after %zu tokens: mapped %u, Off's %u", tag.c_str(), count, Mapped(s), Mapped(t)));
					                            ++compared;
				                            },
				                            tag);
				Check(o.ok && compared >= 4, F("%s: %d decode steps compared", tag.c_str(), compared));
			}
	// The rejected page crossing (XG17, XG18): a pooled sequence at the last position of a page, a draft whose first
	// row is legal and wrong (so only p is kept), the rejected rows on the next page. Mapped pages equal Off's, and
	// a release poisons the dirty page the rejected rows wrote (one more 0xCD page than Off's pool).
	const Choice& ch = TheChoice(m);
	if (ch.c < 0) { Check(false, "no choice point for the page crossing"); return; }
	{
		const std::string tag = "pooled, a rejected page crossing";
		World w(m);
		SslmGpuKvPoolHandle *ps = nullptr, *pt = nullptr;
		SslmGpuSequenceHandle* s = w.Seq(true, 200, ch.schema, &ps);
		SslmGpuSequenceHandle* t = w.Seq(true, 200, ch.schema, &pt);
		Hist hs, ht;
		if (!s || !t || !Replay(w, m, s, ch.prompt, ch.c + 1, &hs) || !Replay(w, m, t, ch.prompt, ch.c + 1, &ht)) { Check(false, tag + ": history"); return; }
		std::vector<int32_t> d = {ch.x};
		d.insert(d.end(), ch.tail.begin(), ch.tail.end());
		const Jump j = DoJump(w, s, hs.pend, 1 + static_cast<int32_t>(d.size()), SSLM_JUMP_EXACT, d.data(), static_cast<int32_t>(d.size()));
		StatusIs(j.st, S::SSLM_OK, tag + ": the jump");
		Check(j.count == 0, F("%s: kept %d drafts, want 0 (row 1 is legal and wrong)", tag.c_str(), j.count));
		int32_t a = -1, b = -1;
		(void)Dec(w, m, s, hs.pend, &a);
		(void)Dec(w, m, t, ht.pend, &b);
		Check(a == b && Mapped(s) == Mapped(t), F("%s: token %d (Off %d), mapped %u (Off %u) (XG17)", tag.c_str(), a, b, Mapped(s), Mapped(t)));
		BlobEq(Blob(w, s, tag), Blob(w, t, tag), tag);
		SslmGpuKvPoolStats st{};
		st.struct_size = sizeof st;
		sslm_gpu_kv_pool_stats(ps, &st);
		const uint32_t pages = st.page_count;
		sslm_gpu_seq_release(w.ctx, s);
		w.Forget(s);
		sslm_gpu_seq_release(w.ctx, t);
		w.Forget(t);
		const int cs = CountPoisonPages(ps, pages), ct = CountPoisonPages(pt, pages);
		Check(cs > ct, F("%s: 0xCD pages after release %d, Off's pool %d: the rejected rows' page was not kept dirty (XG18)", tag.c_str(), cs, ct));
	}
	// A saturating KEPT row (XG6): S-GSATROW at pass row 1 of a jump that keeps it; counters and blob equal Off's.
	for (int pooled = 0; pooled <= 1; ++pooled) {
		const std::string tag = F("%s: a kept row saturates", pooled ? "pooled" : "dedicated");
		World w(m);
		SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 200, kLongKey);
		SslmGpuSequenceHandle* t = w.Seq(pooled != 0, 200, kLongKey);
		SslmGpuSequenceHandle* o = w.Seq(pooled != 0, 200, kLongKey);
		Hist hs, ht, ho;
		const std::vector<int32_t> p = Prompt(12, 903);
		if (!s || !t || !o || !Replay(w, m, s, p, 1, &hs) || !Replay(w, m, t, p, 1, &ht) || !Replay(w, m, o, p, 1, &ho)) { Check(false, tag); continue; }
		TwinDecodes(w, m, o, &ho, 4);
		const std::vector<int32_t> d(ho.toks.begin() + 1, ho.toks.begin() + 4);
		ArmGpuSatRowForTest(1);
		const Jump j = DoJump(w, s, hs.pend, 4, SSLM_JUMP_EXACT, d.data(), 3);
		Check(GpuSatRowArmedForTest() == 0, tag + ": S-GSATROW never fired in the pass");
		ArmGpuSatRowForTest(-1);
		Check(j.st == S::SSLM_OK && j.count == 3, F("%s: %s, kept %d of 3", tag.c_str(), N(j.st), j.count));
		int32_t x = -1;
		S st = Dec(w, m, t, ht.pend, &x);
		ArmGpuSatRowForTest(0);
		int32_t y = -1, z = -1;
		if (st == S::SSLM_OK) st = Dec(w, m, t, x, &y);
		ArmGpuSatRowForTest(-1);
		if (st == S::SSLM_OK) st = Dec(w, m, t, y, &z);
		int32_t z2 = -1;
		if (st == S::SSLM_OK) st = Dec(w, m, t, z, &z2);
		int32_t a = -1;
		(void)Dec(w, m, s, j.count > 0 ? j.out[2] : hs.pend, &a);
		Check(a == z2, F("%s: the token after %d, Off's %d", tag.c_str(), a, z2));
		Check(mr2::ReadCounters(s) == mr2::ReadCounters(t), F("%s: counters %s, Off's %s (XG6)", tag.c_str(), mr2::ReadCounters(s).Str().c_str(),
		                                                         mr2::ReadCounters(t).Str().c_str()));
		BlobEq(Blob(w, s, tag), Blob(w, t, tag), tag);
	}
}

// ------------------------------------------------------------------------------------------------ gx.4
void CellGx4(Fsfd& fx) {
	const Model& m = fx.m;
	const int T = 32;
	for (int pooled = 0; pooled <= 1; ++pooled) {
		const std::string tag = pooled ? "pooled" : "dedicated";
		const std::vector<int32_t> p1 = Prompt(12, 941), p2 = Prompt(12, 942);
		const OffRef o1 = RunOff(m, kPotion, p1, T), o2 = RunOff(m, kLongKey, p2, T);
		World w(m);
		SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 240, kPotion);
		Hist h;
		if (!s || !Replay(w, m, s, p1, 1, &h)) { Check(false, tag); continue; }
		// The draft lives in a buffer the loop overwrites with a hostile id right after each call: the verb must keep
		// no reference to it.
		std::unique_ptr<std::vector<int32_t>> buf;
		const LoopOut a = ExactLoop(w, m, s, &h, 8, T,
		                            [&](SslmGpuSequenceHandle* q) {
			                            if (buf) std::fill(buf->begin(), buf->end(), kSpecial);
			                            buf = std::make_unique<std::vector<int32_t>>(Chain(w, q));
			                            return *buf;
		                            },
		                            nullptr, tag + ": first run");
		Check(a.toks == std::vector<int32_t>(o1.toks.begin(), o1.toks.begin() + static_cast<long>(std::min<size_t>(a.toks.size(), o1.toks.size()))),
		      F("%s: the first run %s, Off's %s", tag.c_str(), Ids(a.toks).c_str(), Ids(o1.toks).c_str()));
		StatusIs(sslm_gpu_seq_reset(w.ctx, s), S::SSLM_OK, tag + ": reset");
		StatusIs(SslmGpuSeqSetSchemaForG5Bridge(w.ctx, s, SslmGpuSchemaLookupForG5Bridge(w.model, kLongKey)), S::SSLM_OK, tag + ": rebind");
		Hist h2;
		if (!Replay(w, m, s, p2, 1, &h2)) { Check(false, tag + ": second history"); continue; }
		const LoopOut b = ExactLoop(w, m, s, &h2, 8, T, nullptr, nullptr, tag + ": second run");
		Check(b.toks == std::vector<int32_t>(o2.toks.begin(), o2.toks.begin() + static_cast<long>(std::min<size_t>(b.toks.size(), o2.toks.size()))),
		      F("%s: after reset and rebind %s, a fresh Off run %s", tag.c_str(), Ids(b.toks).c_str(), Ids(o2.toks).c_str()));
	}
}

// ------------------------------------------------------------------------------------------------ gx.5
void CellGx5(Fsfd& fx) {
	const Model& m = fx.m;
	const std::vector<int32_t> p = Prompt(12, 950);
	// Malformed options (2.X1's refusals on the GPU's argument status), each changing nothing.
	{
		World w(m);
		SslmGpuSequenceHandle* s = w.Seq(false, 200, kLongKey);
		Hist h;
		if (s && Replay(w, m, s, p, 1, &h)) {
			static const int32_t d[2] = {300, 301};
			struct C {
				const char* name;
				int32_t mode;
				const int32_t* draft;
				int32_t count;
				uint32_t size;
			};
			const C cs[] = {{"struct_size - 1", SSLM_JUMP_EXACT, d, 2, sizeof(SslmGpuJumpOptions) - 1},
			                {"struct_size + 4", SSLM_JUMP_EXACT, d, 2, sizeof(SslmGpuJumpOptions) + 4},
			                {"draft_count -1", SSLM_JUMP_EXACT, d, -1, sizeof(SslmGpuJumpOptions)},
			                {"NULL draft, draft_count 2", SSLM_JUMP_EXACT, nullptr, 2, sizeof(SslmGpuJumpOptions)},
			                {"mode 2", 2, nullptr, 0, sizeof(SslmGpuJumpOptions)}};
			for (const C& c : cs) {
				const Obs before = Observe(w, s);
				const Jump j = DoJump(w, s, h.pend, 8, c.mode, c.draft, c.count, c.size);
				StatusIs(j.st, kArg, std::string("malformed: ") + c.name);
				Check(j.count == -7, std::string("malformed: ") + c.name + ": the count was written");
				Unchanged(w, s, before, std::string("malformed: ") + c.name);
			}
		}
	}
	// Hostile drafts: the draft is cut at its first out-of-range, special or illegal id; nothing past it reaches the
	// device, and the run equals Off.
	struct H {
		const char* name;
		int32_t bad;
	};
	const H hs[] = {{"id = vocab", m.vocab()}, {"id -1", -1}, {"id INT32_MAX", 0x7FFFFFFF}, {"a special id", kSpecial}};
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (const H& hx : hs)
			for (int k : {0, 1, 3}) {
				const std::string tag = F("%s %s at draft row %d", pooled ? "pooled" : "dedicated", hx.name, k);
				World w(m);
				SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 200, kLongKey);
				SslmGpuSequenceHandle* t = w.Seq(pooled != 0, 200, kLongKey);
				SslmGpuSequenceHandle* o = w.Seq(pooled != 0, 200, kLongKey);
				Hist hs1, ht, ho;
				if (!s || !t || !o || !Replay(w, m, s, p, 1, &hs1) || !Replay(w, m, t, p, 1, &ht) || !Replay(w, m, o, p, 1, &ho)) { Check(false, tag); continue; }
				TwinDecodes(w, m, o, &ho, 7);
				std::vector<int32_t> d(ho.toks.begin() + 1, ho.toks.begin() + 7);  // Off's own six
				d[static_cast<size_t>(k)] = hx.bad;
				const Jump j = DoJump(w, s, hs1.pend, 8, SSLM_JUMP_EXACT, d.data(), static_cast<int32_t>(d.size()));
				StatusIs(j.st, S::SSLM_OK, tag + ": the jump");
				Check(j.count == k, F("%s: kept %d, want %d (cut at the hostile id)", tag.c_str(), j.count, k));
				bool none = true;
				for (int32_t i = 0; i < std::max(j.count, 0); ++i) none = none && j.out[static_cast<size_t>(i)] != hx.bad;
				Check(none, tag + ": the hostile id was written to out_forced");
				int32_t a = -1;
				(void)Dec(w, m, s, j.count > 0 ? j.out[static_cast<size_t>(j.count) - 1] : hs1.pend, &a);
				TwinDecodes(w, m, t, &ht, std::max(j.count, 0) + 1);
				Check(a == ht.pend, F("%s: the token after %d, Off's %d", tag.c_str(), a, ht.pend));
				BlobEq(Blob(w, s, tag), Blob(w, t, tag), tag);
			}
}

// ------------------------------------------------------------------------------------------------ gx.6
struct HookCtx {
	World* w = nullptr;
	SslmGpuSequenceHandle* target = nullptr;
	int32_t pend = -1;
	std::atomic<int> busy{0}, fired{0};
	bool armed = false;
};
HookCtx* g_hook = nullptr;
void ClaimHook(SslmGpuSequenceHandle* seq, void* user) {
	HookCtx* h = static_cast<HookCtx*>(user);
	if (!h->armed || seq != h->target) return;
	h->armed = false;  // one interleaving per arm
	h->fired++;
	// Another thread asks for the same handle while this one holds its claim: SSLM_BUSY, changing nothing.
	std::thread other([h] {
		std::vector<int32_t> out(16, -7);
		int32_t n = -7;
		const S st = sslm_gpu_seq_jump_forward(h->w->ctx, h->target, h->pend, 4, nullptr, out.data(), 16, &n);
		if (st == S::SSLM_BUSY && n == -7) h->busy++;
	});
	other.join();
}
void CellGx6(Fsfd& fx) {
	const Model& m = fx.m;
	const int T = 32;
	const std::vector<int32_t> pa = Prompt(12, 961), pb = Prompt(12, 962);
	const OffRef oa = RunOff(m, kPotion, pa, T), ob = RunOff(m, kLongKey, pb, T);
	World w(m);
	SslmGpuSequenceHandle* a = w.Seq(false, 240, kPotion);
	SslmGpuSequenceHandle* b = w.Seq(true, 240, kLongKey);
	Hist ha, hb;
	if (!a || !b || !Replay(w, m, a, pa, 1, &ha) || !Replay(w, m, b, pb, 1, &hb)) { Check(false, "history"); return; }
	HookCtx hc;
	hc.w = &w;
	SslmGpuSeqClaimHookForTest(ClaimHook, &hc);
	std::vector<int32_t> sa = ha.toks, sb = hb.toks;
	int32_t pa_ = ha.pend, pb_ = hb.pend;
	bool da = false, db = false;
	for (int step = 0; step < 200 && (!da || !db); ++step) {
		// a: NULL options at budget 3; b: its own forced chain as an explicit draft at budget 6. Alternating, with
		// the claim hook armed on whichever runs.
		SslmGpuSequenceHandle* s = (step % 2 == 0) ? a : b;
		bool& done = (s == a) ? da : db;
		std::vector<int32_t>& stream = (s == a) ? sa : sb;
		int32_t& pend = (s == a) ? pa_ : pb_;
		if (done) continue;
		if (static_cast<int>(stream.size()) >= T) { done = true; continue; }
		hc.target = s;
		hc.pend = pend;
		hc.armed = true;
		Jump j;
		if (s == a) {
			j = DoJump(w, s, pend, 3, kNull);
		} else {
			const std::vector<int32_t> d = Chain(w, s);
			j = DoJump(w, s, pend, 6, SSLM_JUMP_EXACT, d.empty() ? nullptr : d.data(), static_cast<int32_t>(d.size()));
		}
		hc.armed = false;
		if (j.st != S::SSLM_OK) { Check(false, F("step %d: the jump %s", step, N(j.st))); done = true; continue; }
		for (int32_t i = 0; i < j.count; ++i) stream.push_back(j.out[static_cast<size_t>(i)]);
		if (j.count > 0) pend = j.out[static_cast<size_t>(j.count) - 1];
		int32_t t = -1;
		if (Dec(w, m, s, pend, &t) != S::SSLM_OK || t < 0) { done = true; continue; }
		stream.push_back(t);
		pend = t;
	}
	SslmGpuSeqClaimHookForTest(nullptr, nullptr);
	Check(hc.fired > 0 && hc.busy == hc.fired, F("the claim hook fired %d times; the second caller saw BUSY %d times", hc.fired.load(), hc.busy.load()));
	auto prefix = [](const std::vector<int32_t>& v, size_t n) { return std::vector<int32_t>(v.begin(), v.begin() + static_cast<long>(std::min(n, v.size()))); };
	Check(prefix(sa, T) == prefix(oa.toks, T), F("a: %s, its Off run %s", Ids(sa).c_str(), Ids(oa.toks).c_str()));
	Check(prefix(sb, T) == prefix(ob.toks, T), F("b: %s, its Off run %s", Ids(sb).c_str(), Ids(ob.toks).c_str()));
}

// ------------------------------------------------------------------------------------------------ gx.7
void CellGx7(Fsfd& fx) {
	const Model& m = fx.m;
	const std::vector<int32_t> p = Prompt(12, 970);
	for (int pooled = 0; pooled <= 1; ++pooled) {
		const std::string base = pooled ? "pooled" : "dedicated";
		World w(m);
		// Fresh (bound, no prompt) and ready (after the prompt), no pending token: no-ops.
		SslmGpuSequenceHandle* f = w.Seq(pooled != 0, 200, kLongKey);
		if (f) {
			const Obs before = Observe(w, f);
			const Jump j = DoJump(w, f, -1, 8, kNull);
			Check(j.st == S::SSLM_OK && j.count == 0, F("%s fresh: %s count %d, want a no-op", base.c_str(), N(j.st), j.count));
			Unchanged(w, f, before, base + " fresh");
		}
		SslmGpuSequenceHandle* r = w.Seq(pooled != 0, 200, kLongKey);
		Hist hr;
		if (r && Replay(w, m, r, p, 0, &hr)) {
			const Obs before = Observe(w, r);
			const Jump j = DoJump(w, r, -1, 8, kNull);
			Check(j.st == S::SSLM_OK && j.count == 0, F("%s ready: %s count %d, want a no-op", base.c_str(), N(j.st), j.count));
			Unchanged(w, r, before, base + " ready");
			// A pending token at the ready shape: 1.17's entry-shape refusal.
			const Jump j2 = DoJump(w, r, hr.prompt.back(), 8, kNull);
			StatusIs(j2.st, S::SSLM_SEQUENCE_REJECTED, base + " ready with a pending token");
			Unchanged(w, r, before, base + " ready with a pending token");
		}
		// Mid-token: an embedded row with one layer run. 1.17's entry-shape refusal.
		SslmGpuSequenceHandle* mt = w.Seq(pooled != 0, 200, kLongKey);
		Hist hm;
		if (mt && Replay(w, m, mt, p, 1, &hm) && sslm_gpu_seq_embed_token(w.ctx, mt, hm.pend) == S::SSLM_OK) {
			S st = sslm_decode_step_gpu(w.ctx, mt, nullptr, m.dpl());
			if (st == S::SSLM_OK) st = Ready(w, mt);
			if (st != S::SSLM_OK) {
				Check(false, F("%s mid-token: the one-layer slice %s", base.c_str(), N(st)));
			} else {
				const Obs before = Observe(w, mt);
				const Jump j = DoJump(w, mt, hm.pend, 8, kNull);
				StatusIs(j.st, S::SSLM_SEQUENCE_REJECTED, base + " mid-token");
				Unchanged(w, mt, before, base + " mid-token");
			}
		} else {
			Check(false, base + " mid-token: setup");
		}
		// In hand: k = 0 (an empty draft: p stays pending), 1 and budget - 1.
		for (int k : {0, 1, 7}) {
			const std::string tag = F("%s in hand, k = %d", base.c_str(), k);
			SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 200, kLongKey);
			SslmGpuSequenceHandle* t = w.Seq(pooled != 0, 200, kLongKey);
			Hist hs, ht;
			if (!s || !t || !Replay(w, m, s, p, 1, &hs) || !Replay(w, m, t, p, 1, &ht)) { Check(false, tag); continue; }
			const std::vector<int32_t> c = Chain(w, s);
			SslmGpuSequenceHandle* o = w.Seq(pooled != 0, 200, kLongKey);
			Hist ho;
			if (!o || !Replay(w, m, o, p, 1, &ho)) { Check(false, tag); continue; }
			TwinDecodes(w, m, o, &ho, 9);
			std::vector<int32_t> d(ho.toks.begin() + 1, ho.toks.begin() + 1 + k);  // Off's own k tokens
			static const int32_t empty_marker = 0;
			const Obs before = Observe(w, s);
			const Jump j = DoJump(w, s, hs.pend, 8, SSLM_JUMP_EXACT, k ? d.data() : &empty_marker, k);
			StatusIs(j.st, S::SSLM_OK, tag + ": the jump");
			Check(j.count == k, F("%s: kept %d, want %d", tag.c_str(), j.count, k));
			if (k == 0) Unchanged(w, s, before, tag + ": an empty draft keeps p pending, nothing changed");
			int32_t a = -1;
			(void)Dec(w, m, s, k ? d.back() : hs.pend, &a);
			TwinDecodes(w, m, t, &ht, k + 1);
			Check(a == ht.pend, F("%s: the next token %d, Off's %d", tag.c_str(), a, ht.pend));
			BlobEq(Blob(w, s, tag), Blob(w, t, tag), tag);
			(void)c;
		}
	}
}

// ------------------------------------------------------------------------------------------------ gx.8
void CellGx8(Fsfd& fx) {
	const Model& m = fx.m;
	const std::vector<int32_t> p = Prompt(12, 980);
	for (int pooled = 0; pooled <= 1; ++pooled) {
		const std::string base = pooled ? "pooled" : "dedicated";
		// (a) A fault in the first pass, with a draft wrong at row 3: its status, nothing committed.
		for (S fault : {S::SSLM_DEVICE_LOST, S::SSLM_GPU_ALLOCATION_FAILED}) {
			const std::string tag = F("%s (a) %s in the first pass", base.c_str(), N(fault));
			World w(m);
			SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 240, kLongKey);
			SslmGpuSequenceHandle* o = w.Seq(pooled != 0, 240, kLongKey);
			Hist hs, ho;
			if (!s || !o || !Replay(w, m, s, p, 1, &hs) || !Replay(w, m, o, p, 1, &ho)) { Check(false, tag); continue; }
			TwinDecodes(w, m, o, &ho, 8);
			std::vector<int32_t> d(ho.toks.begin() + 1, ho.toks.begin() + 8);
			d[3] = kSpecial;
			const Obs before = Observe(w, s);
			StatusIs(sslm_gpu_test_only_arm_chunk_fault(0, fault), S::SSLM_OK, tag + ": arm");
			const Jump j = DoJump(w, s, hs.pend, 8, SSLM_JUMP_EXACT, d.data(), static_cast<int32_t>(d.size()));
			(void)sslm_gpu_test_only_arm_chunk_fault(-1, S::SSLM_DEVICE_LOST);
			StatusIs(j.st, fault, tag);
			Unchanged(w, s, before, tag + ": nothing committed");
		}
		// (b) A fault in a later pass (S-GCHUNKFAULT at pass 1) and a read-back fault in it (S-GREADBACK): OK, the
		// earlier passes' kept rows only, and the continuation equals Off.
		for (int kind = 0; kind <= 1; ++kind) {
			const std::string tag = F("%s (b) %s in the second pass", base.c_str(), kind ? "a read-back fault" : "a submission fault");
			World w(m);
			SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 240, kLongKey);
			SslmGpuSequenceHandle* t = w.Seq(pooled != 0, 240, kLongKey);
			SslmGpuSequenceHandle* o = w.Seq(pooled != 0, 240, kLongKey);
			Hist hs, ht, ho;
			if (!s || !t || !o || !Replay(w, m, s, p, 1, &hs) || !Replay(w, m, t, p, 1, &ht) || !Replay(w, m, o, p, 1, &ho)) { Check(false, tag); continue; }
			TwinDecodes(w, m, o, &ho, 24);
			std::vector<int32_t> d(ho.toks.begin() + 1, ho.toks.begin() + 24);  // all accepted: every pass runs
			if (kind) ArmGpuExactReadbackFaultForTest(1, 0);
			else (void)sslm_gpu_test_only_arm_chunk_fault(1, S::SSLM_DEVICE_LOST);
			const Jump j = DoJump(w, s, hs.pend, 24, SSLM_JUMP_EXACT, d.data(), static_cast<int32_t>(d.size()));
			ArmGpuExactReadbackFaultForTest(-1, 0);
			(void)sslm_gpu_test_only_arm_chunk_fault(-1, S::SSLM_DEVICE_LOST);
			StatusIs(j.st, S::SSLM_OK, tag);
			Check(j.count >= 1 && j.count < 23, F("%s: kept %d of 23 (the first pass's rows only)", tag.c_str(), j.count));
			bool prefix = j.count >= 0;
			for (int32_t i = 0; prefix && i < j.count; ++i) prefix = j.out[static_cast<size_t>(i)] == d[static_cast<size_t>(i)];
			Check(prefix, tag + ": out_forced is not the draft's prefix");
			int32_t a = -1;
			(void)Dec(w, m, s, j.count > 0 ? j.out[static_cast<size_t>(j.count) - 1] : hs.pend, &a);
			TwinDecodes(w, m, t, &ht, std::max(j.count, 0) + 1);
			Check(a == ht.pend, F("%s: the token after %d, Off's %d", tag.c_str(), a, ht.pend));
			BlobEq(Blob(w, s, tag), Blob(w, t, tag), tag + ": nothing past the kept rows remains");
		}
	}
}

// ------------------------------------------------------------------------------------------------ gx.9
void CellGx9(Fsfd& fx) {
	const Model& m = fx.m;
	const int T = 24;
	const std::vector<int32_t> pe = Prompt(12, 991), po = Prompt(12, 992), pu = Prompt(12, 993);
	const OffRef se = RunOff(m, kLongKey, pe, T), so = RunOff(m, kPotion, po, T), su = RunOff(m, nullptr, pu, T);
	World w(m);
	SslmGpuSequenceHandle* e = w.Seq(false, 240, kLongKey);
	SslmGpuSequenceHandle* o = w.Seq(true, 240, kPotion);
	SslmGpuSequenceHandle* u = w.Seq(false, 240, nullptr);
	Hist he, ho, hu;
	if (!e || !o || !u || !Replay(w, m, e, pe, 1, &he) || !Replay(w, m, o, po, 1, &ho) || !Replay(w, m, u, pu, 1, &hu)) { Check(false, "history"); return; }
	std::vector<int32_t> te = he.toks;
	int32_t pend = he.pend;
	for (int step = 0; step < 100 && static_cast<int>(te.size()) < T; ++step) {
		const Jump j = DoJump(w, e, pend, 5, kNull);
		if (j.st != S::SSLM_OK) { Check(false, F("the jump %s", N(j.st))); break; }
		for (int32_t i = 0; i < j.count; ++i) te.push_back(j.out[static_cast<size_t>(i)]);
		if (j.count > 0) pend = j.out[static_cast<size_t>(j.count) - 1];
		TwinDecodes(w, m, o, &ho, 1);
		TwinDecodes(w, m, u, &hu, 1);
		int32_t t = -1;
		if (Dec(w, m, e, pend, &t) != S::SSLM_OK || t < 0) break;
		te.push_back(t);
		pend = t;
	}
	auto pre = [](const std::vector<int32_t>& v, size_t n) { return std::vector<int32_t>(v.begin(), v.begin() + static_cast<long>(std::min(n, v.size()))); };
	Check(pre(te, T) == pre(se.toks, T), F("the Exact member %s, solo %s", Ids(te).c_str(), Ids(se.toks).c_str()));
	Check(pre(ho.toks, ho.toks.size()) == pre(so.toks, ho.toks.size()), F("the Off member %s, solo %s", Ids(ho.toks).c_str(), Ids(so.toks).c_str()));
	Check(pre(hu.toks, hu.toks.size()) == pre(su.toks, hu.toks.size()), F("the unbound member %s, solo %s", Ids(hu.toks).c_str(), Ids(su.toks).c_str()));
}

// ------------------------------------------------------------------------------------------------ gx.10
void CellGx10(Fsfd& fx) {
	const Model& m = fx.m;
	const int T = 36;
	for (const char* schema : {kPotion, kLongKey})
		for (int pooled = 0; pooled <= 1; ++pooled) {
			const std::string tag = F("%s %s (%s)", pooled ? "pooled" : "dedicated", schema, pooled ? "SLM6" : "SLM5");
			const std::vector<int32_t> p = Prompt(12, 1010);
			const OffRef off = RunOff(m, schema, p, T);
			World w(m);
			SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 240, schema);
			Hist h;
			if (!s || !Replay(w, m, s, p, 1, &h)) { Check(false, tag); continue; }
			std::vector<int32_t> stream = h.toks;
			int32_t pend = h.pend;
			int restores = 0;
			for (int step = 0; step < 100 && static_cast<int>(stream.size()) < T; ++step) {
				const Jump j = DoJump(w, s, pend, 6, kNull);
				if (j.st != S::SSLM_OK) { Check(false, F("%s: the jump %s", tag.c_str(), N(j.st))); break; }
				for (int32_t i = 0; i < j.count; ++i) stream.push_back(j.out[static_cast<size_t>(i)]);
				if (j.count > 0) pend = j.out[static_cast<size_t>(j.count) - 1];
				// Save and restore between the Exact call and the next decode, then drop the original.
				const std::vector<uint8_t> b = Blob(w, s, tag);
				SslmGpuSequenceHandle* r = nullptr;
				S rs;
				if (pooled) {
					SslmGpuKvPoolHandle* pl = w.Pool(70);
					uint32_t shared = 0;
					rs = sslm_gpu_seq_restore_into_pool(w.ctx, w.model, pl, b.data(), b.size(), nullptr, &r, &shared);
				} else {
					rs = sslm_gpu_seq_restore(w.ctx, w.model, b.data(), b.size(), &r);
				}
				if (rs != S::SSLM_OK || !r) { Check(false, F("%s: restore %s", tag.c_str(), N(rs))); break; }
				w.seqs.push_back(r);
				BlobEq(Blob(w, r, tag), b, F("%s: the restored handle's save (step %d)", tag.c_str(), step));
				sslm_gpu_seq_release(w.ctx, s);
				w.Forget(s);
				s = r;
				++restores;
				int32_t t = -1;
				if (Dec(w, m, s, pend, &t) != S::SSLM_OK || t < 0) break;
				stream.push_back(t);
				pend = t;
			}
			const size_t n = std::min(stream.size(), off.toks.size());
			Check(restores >= 3 && std::equal(stream.begin(), stream.begin() + static_cast<long>(n), off.toks.begin()) && n >= std::min<size_t>(T, off.toks.size()),
			      F("%s: %d restores; the stream %s, Off's %s", tag.c_str(), restores, Ids(stream).c_str(), Ids(off.toks).c_str()));
		}
}

// ------------------------------------------------------------------------------------------------ gx.11
void CellGx11(Fsfd& fx) {
	const Model& m = fx.m;
	const Choice& ch = TheChoice(m);
	if (ch.c < 0) { Check(false, "no choice point"); return; }
	// A legal, always-wrong draft at the choice: Exact keeps nothing of it. Under S-GVERIFY "always equal" the check
	// passes it, so the stream must differ from Off's: the comparison can fail.
	World w(m);
	SslmGpuSequenceHandle* s = w.Seq(false, 240, ch.schema);
	Hist h;
	if (!s || !Replay(w, m, s, ch.prompt, ch.c + 1, &h)) { Check(false, "history"); return; }
	std::vector<int32_t> d = {ch.x};
	d.insert(d.end(), ch.tail.begin(), ch.tail.end());
	SetGpuExactVerifyAlwaysEqualForTest(1);
	const Jump j = DoJump(w, s, h.pend, 1 + static_cast<int32_t>(d.size()), SSLM_JUMP_EXACT, d.data(), static_cast<int32_t>(d.size()));
	SetGpuExactVerifyAlwaysEqualForTest(0);
	std::vector<int32_t> got = h.toks;
	for (int32_t i = 0; i < std::max(j.count, 0); ++i) got.push_back(j.out[static_cast<size_t>(i)]);
	const std::vector<int32_t> want(ch.off.toks.begin(), ch.off.toks.begin() + static_cast<long>(std::min(got.size(), ch.off.toks.size())));
	Check(j.st == S::SSLM_OK && got != want,
	      F("under S-GVERIFY the jump (%s, kept %d) still matched Off: the seam is not built, or the comparison cannot fail", N(j.st), j.count));
}

// ------------------------------------------------------------------------------------------------ gx.12
struct Snap {
	S st = S::SSLM_OK;
	std::vector<int8_t> codes;
	int64_t sm = 0, se = 0;
	bool operator==(const Snap& o) const { return st == o.st && codes == o.codes && sm == o.sm && se == o.se; }
};
Snap ReadSnap(World& w, SslmGpuSequenceHandle* s) {
	Snap n;
	size_t need = 0;
	n.st = sslm_gpu_seq_read_prefill_final_hidden(w.ctx, s, nullptr, 0, &need, &n.sm, &n.se);
	if (need > 0) {
		n.codes.assign(need, 0);
		n.st = sslm_gpu_seq_read_prefill_final_hidden(w.ctx, s, n.codes.data(), need, &need, &n.sm, &n.se);
	}
	return n;
}
void CellGx12(Fsfd& fx) {
	const Model& m = fx.m;
	for (int pooled = 0; pooled <= 1; ++pooled) {
		const std::string tag = pooled ? "pooled" : "dedicated";
		World w(m);
		SslmGpuSequenceHandle* s = w.Seq(pooled != 0, 240, kLongKey);
		Hist h;
		if (!s || !Replay(w, m, s, Prompt(12, 1012), 1, &h)) { Check(false, tag); continue; }
		const Snap first = ReadSnap(w, s);
		int calls = 0;
		const LoopOut o = ExactLoop(w, m, s, &h, 8, 30, nullptr,
		                            [&](size_t n) {
			                            ++calls;
			                            Check(ReadSnap(w, s) == first, F("%s: the prefill snapshot changed after %zu tokens (XG12)", tag.c_str(), n));
		                            },
		                            tag);
		Check(o.ok && calls >= 4, F("%s: %d steps", tag.c_str(), calls));
	}
}

// ------------------------------------------------------------------------------------------------ gx.13
void CellGx13(Fsfd& fx) {
	const Model& m = fx.m;
	const std::vector<int32_t> p = Prompt(12, 1013);
	for (int wrong = 0; wrong <= 1; ++wrong) {
		const std::string tag = wrong ? "a mismatch at draft row 2" : "every draft row accepted";
		World w(m);
		SslmGpuSequenceHandle* s = w.Seq(false, 240, kLongKey);
		SslmGpuSequenceHandle* o = w.Seq(false, 240, kLongKey);
		Hist hs, ho;
		if (!s || !o || !Replay(w, m, s, p, 1, &hs) || !Replay(w, m, o, p, 1, &ho)) { Check(false, tag); continue; }
		TwinDecodes(w, m, o, &ho, 24);
		std::vector<int32_t> d(ho.toks.begin() + 1, ho.toks.begin() + 24);
		if (wrong) d[2] = kSpecial;  // cut there: the clipped draft is 2 rows, one pass
		GpuExactPassLogResetForTest();
		const Jump j = DoJump(w, s, hs.pend, 24, SSLM_JUMP_EXACT, d.data(), static_cast<int32_t>(d.size()));
		const int32_t passes = GpuExactPassCountForTest();
		StatusIs(j.st, S::SSLM_OK, tag);
		Check(j.count == (wrong ? 2 : 23), F("%s: kept %d", tag.c_str(), j.count));
		if (wrong) Check(passes == 1, F("%s: %d Exact passes ran, want 1 (no rows past the mismatch; XG13)", tag.c_str(), passes));
		else Check(passes >= 3, F("%s: %d Exact passes ran for 24 rows, want 3 or more (passes of at most %d rows)", tag.c_str(), passes, SSLM_GPU_MAX_ROWS));
	}
}

}  // namespace

int main(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--fixtures=", 0) == 0) g_opt.fixtures = a.substr(11);
		else if (a.rfind("--show=", 0) == 0) g_opt.show = std::atoi(a.c_str() + 7);
		else if (a.rfind("--cells=", 0) == 0) {
			std::string l = a.substr(8);
			for (size_t p = 0; p <= l.size();) {
				const size_t q = l.find(',', p);
				g_opt.cells.insert(l.substr(p, q == std::string::npos ? std::string::npos : q - p));
				if (q == std::string::npos) break;
				p = q + 1;
			}
		}
	}
	Fsfd fx;
	if (!fx.Load()) return 1;
	if (!LoadRef()) { std::printf("FAIL fsfd_ref.txt did not load\n"); return 1; }
	const std::vector<std::pair<const char*, std::function<void()>>> cells = {
	    {"2.2", [&] { Cell22(fx); }},     {"2.3x", [&] { Cell23x(fx); }},   {"7.10", [&] { Cell710(fx); }},   {"7.11e", [&] { Cell711e(fx); }},
	    {"7.17", [&] { Cell717(fx); }},   {"7.18", [&] { Cell718(fx); }},   {"7.19", [&] { Cell719(fx); }},   {"8.4", [&] { Cell84(fx); }},
	    {"gx.1", [&] { CellGx1(fx); }},   {"gx.2", [&] { CellGx2(fx); }},   {"gx.4", [&] { CellGx4(fx); }},   {"gx.5", [&] { CellGx5(fx); }},
	    {"gx.6", [&] { CellGx6(fx); }},   {"gx.7", [&] { CellGx7(fx); }},   {"gx.8", [&] { CellGx8(fx); }},   {"gx.9", [&] { CellGx9(fx); }},
	    {"gx.10", [&] { CellGx10(fx); }}, {"gx.11", [&] { CellGx11(fx); }}, {"gx.12", [&] { CellGx12(fx); }}, {"gx.13", [&] { CellGx13(fx); }},
	};
	bool all = true;
	for (auto& c : cells) {
		if (!g_opt.want(c.first)) continue;
		g_cell = c.first;
		c.second();
		const int f = g_fail[c.first], n = g_checks[c.first];
		if (n == 0) std::printf("    cell %s: n/a\n", c.first);
		else if (f) std::printf("RED cell %s: %d failures of %d checks\n", c.first, f, n);
		else std::printf("    cell %s: PASS (%d checks)\n", c.first, n);
		std::fflush(stdout);
		all = all && f == 0;
	}
	if (Mr5StubbedSeamsForTest()) std::printf("note: stubbed seams armed (mask 0x%x): mr5_entry.h names them\n", Mr5StubbedSeamsForTest());
	std::printf("cells: %s\n", all ? "PASS" : "RED");
	return all ? 0 : 1;
}
