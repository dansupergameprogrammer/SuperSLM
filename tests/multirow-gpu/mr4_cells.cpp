// Multi-row GPU plan (rev 1.3.2), slice MR4 (on 1.17): the cloud cells of §9's MR4 row. Wine + vkd3d-proton + lavapipe,
// on F-SFD (tools/gen_schema_fast_decode_fixture.py: fsfd.sslm, real Qwen vocabulary, 2 layers, four schemas with
// forced-next tables) and fsfd_refuse.sslm (mr2_fixtures.py's last-layer attn_residual patch applied to fsfd).
//
// Cells:
//   2.3   SCHEMA_FORCED diverging at row k admits k (k 1..M-1, M 2, 4, 8, dedicated and pooled), and the chunk then
//         closes equal to the one-row twin (1.17's walked embed, sslm_gpu_seq_embed_schema_token, row by row) fed
//         rows 0..k-1: whole blob, walk state. Zero admitted (rulings item 11), each refusal writing nothing:
//           bad spec -> the argument status, before everything below;
//           (a) no forced-next table -> SSLM_GPU_FORCED_TABLE_ABSENT, before (c) and (b);
//           (c) no headroom -> SSLM_GPU_CONTEXT_CAP_EXCEEDED (a pooled handle at its budget limit:
//               SSLM_GPU_KV_BUDGET_EXCEEDED, as MR3 and 1.17's verb), before (b);
//           (b) row 0 (row 1 under FIRST_PREWALKED) not the forced token, or no schema bound -> SSLM_SEQUENCE_REJECTED.
//         Headroom below the matching prefix admits the headroom.
//   7.9   sslm_gpu_seq_jump_forward on the recorder rests at entry on a guard refusal at every row j of its pass
//         (fsfd_refuse; each j's case is found at run time on the one-row path): 1.17's status for a fault in the
//         pass (SSLM_DEVICE_LOST: gpu_1p0.h, every fault but an allocation failure), and the whole
//         save blob (mirror included), the walk, the ready flag, the layer and the context equal entry's. Dedicated
//         and pooled (a pooled holder's mapped pages too), with and without a pending token.
//   (8.3 is mr4_83_canon.cpp, run by the runner against 1.17's own binary.)
//   8.10  A forced span after a produced token: with FIRST_PREWALKED, embed_rows(SCHEMA_FORCED, [t, chain...]) admits
//         the whole span, walks the chain once (the walk equals 1.17's jump_forward(pending = t)'s) and closes equal to
//         the one-row twin (t embedded unwalked, the chain walked row by row). Without the flag it admits nothing:
//         SSLM_SEQUENCE_REJECTED, nothing written (MX16).
// Verdicts: statuses, whole save blobs against the one-row twin in the same process, the walk state (the bench
// bridge's G5 accessor), and 1.17's own verbs.
//
// Usage: mr4_cells.exe --fixtures=DIR [--cells=2.3,7.9,8.10] [--show=N]
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cstdarg>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "mr2_common.h"

namespace {
using S = SslmGpuStatus;
using mr2::Model;
using mr2::World;

constexpr S kTableAbsent = S::SSLM_GPU_FORCED_TABLE_ABSENT;
constexpr S kCapStatus = S::SSLM_GPU_CONTEXT_CAP_EXCEEDED;
constexpr S kArgStatus = S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;

struct Opt {
	std::string fixtures;
	std::set<std::string> cells;
	int show = 16;  // failure lines printed per cell (--show=N)
	bool want(const char* c) const { return cells.empty() || cells.count(c); }
};
Opt g_opt;
std::map<std::string, int> g_fail, g_checks;
std::string g_cell;
void Check(bool ok, const std::string& what) {
	++g_checks[g_cell];
	if (!ok && ++g_fail[g_cell] <= g_opt.show) std::printf("FAIL %s: %s\n", g_cell.c_str(), what.c_str());
}
std::string F(const char* fmt, ...) {
	char b[1024];
	va_list ap;
	va_start(ap, fmt);
	std::vsnprintf(b, sizeof b, fmt, ap);
	va_end(ap);
	return b;
}
const char* N(S s) {
	static thread_local char buf[8][48];
	static thread_local int at = 0;
	char* b = buf[at++ % 8];
	if (s == kTableAbsent) return "FORCED_TABLE_ABSENT";
	if (s == kCapStatus) return "CONTEXT_CAP_EXCEEDED";
	std::snprintf(b, 48, "%u", static_cast<unsigned>(s));
	return b;
}
void StatusIs(S got, S want, const std::string& tag) { Check(got == want, F("%s: returned %s, want %s", tag.c_str(), N(got), N(want))); }
std::string Ids(const std::vector<int32_t>& v) {
	std::string s = "{";
	for (size_t i = 0; i < v.size() && i < 24; ++i) s += (i ? "," : "") + std::to_string(v[i]);
	return s + (v.size() > 24 ? ",...}" : "}");
}
void BlobEq(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, const std::string& tag) {
	long d = -1;
	for (size_t i = 0; i < std::min(got.size(), want.size()); ++i)
		if (got[i] != want[i]) { d = static_cast<long>(i); break; }
	if (d < 0 && got.size() != want.size()) d = static_cast<long>(std::min(got.size(), want.size()));
	size_t n = 0;
	for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) n += got[i] != want[i];
	Check(d < 0 && !got.empty(), F("%s: blob differs (sizes %zu/%zu, first byte %ld, %zu bytes differ)", tag.c_str(), got.size(), want.size(), d, n));
}

// ------------------------------------------------------------------------------------------------ the fixture
// The world on F-SFD: the model's schemas by name, the prompt the 1.17 suite uses for its pinned streams.
const char* kPotion = "potion_shop_order";
const char* kPromptResult = "prompt_result";
const char* kLongKey = "longkey";
struct Fsfd {
	Model m;
	bool ok = false;
	bool Load(const std::string& file) { return ok = m.Load(g_opt.fixtures + "/" + file); }
};
std::vector<int32_t> Prompt(int n, uint32_t stream) {
	std::vector<int32_t> v;
	for (int i = 0; i < n; ++i) v.push_back(mr2::StreamToken(i, 151936, stream) % 20000 + 200);  // ordinary text ids
	return v;
}
int32_t SchemaIndex(World& w, const char* name) { return SslmGpuSchemaLookupForG5Bridge(w.model, name); }
bool Bind(World& w, SslmGpuSequenceHandle* s, const char* schema) {
	const S st = SslmGpuSeqSetSchemaForG5Bridge(w.ctx, s, SchemaIndex(w, schema));
	Check(st == S::SSLM_OK, F("bind %s: %s", schema, N(st)));
	return st == S::SSLM_OK;
}
// A handle at `cap` (dedicated) or with budget `cap` on its own pool (pooled), bound to `schema` (null: unbound).
SslmGpuSequenceHandle* NewSeq(World& w, bool pooled, int32_t cap, const char* schema) {
	SslmGpuSequenceHandle* s = pooled ? w.Pooled(w.Pool(static_cast<uint32_t>(cap / 4 + 8)), cap) : w.Dedicated(cap);
	if (s && schema) Bind(w, s, schema);
	return s;
}
// The prompt through the prompt bridge (no walk): the handle rests ready, as the 1.17 suite's prompts do.
S PromptBridge(World& w, const Model& m, SslmGpuSequenceHandle* s, const std::vector<int32_t>& p) {
	return SslmGpuSeqPrefillPromptForG5Bridge(w.ctx, s, p.data(), static_cast<int32_t>(p.size()), m.full_budget());
}
uint32_t Walk(SslmGpuSequenceHandle* s) { return SslmGpuSeqWalkStateForG5Bridge(s); }
int32_t ReadyFlag(World& w, SslmGpuSequenceHandle* s) {
	int32_t r = -1;
	return sslm_gpu_seq_ready_for_logits(w.ctx, s, &r) == S::SSLM_OK ? r : -1;
}
uint32_t LayerIdx(SslmGpuSequenceHandle* s) { return *SslmGpuSeqHandleLayerIndexForBench(s); }
int64_t CtxLen(SslmGpuSequenceHandle* s) { return *SslmGpuSeqHandleContextLengthForBench(s); }
std::vector<int32_t> Chain(World& w, SslmGpuSequenceHandle* s) {
	std::vector<int32_t> ids(512, -7);
	int32_t n = 0, acc = 0;
	const S st = sslm_gpu_seq_forced_tokens(w.ctx, s, ids.data(), 512, &n, &acc);
	Check(st == S::SSLM_OK, F("forced_tokens: %s", N(st)));
	ids.resize(st == S::SSLM_OK && n >= 0 ? std::min(n, 512) : 0);
	return ids;
}
std::vector<uint8_t> Blob(World& w, SslmGpuSequenceHandle* s, const std::string& tag) {
	std::vector<uint8_t> b;
	S st = S::SSLM_OK;
	if (!mr2::Save(w, s, &b, &st)) Check(false, F("%s: save %s", tag.c_str(), N(st)));
	return b;
}
// Everything a refusal must leave as it was.
struct Obs {
	std::vector<uint8_t> blob;
	uint32_t walk = 0, layer = 0, mapped = 0;
	int32_t ready = -1, open = -1;
	int64_t ctx = -1;
	bool operator==(const Obs& o) const {
		return blob == o.blob && walk == o.walk && layer == o.layer && mapped == o.mapped && ready == o.ready && open == o.open && ctx == o.ctx;
	}
	std::string Str() const {
		return F("blob %zu bytes fnv %016llx walk %u layer %u ready %d rows_open %d ctx %lld mapped %u", blob.size(),
		         (unsigned long long)mr2::Fnv(blob), walk, layer, ready, open, (long long)ctx, mapped);
	}
};
Obs Observe(World& w, SslmGpuSequenceHandle* s) {
	Obs o;
	o.blob = Blob(w, s, "observe");
	o.walk = Walk(s);
	o.layer = LayerIdx(s);
	o.ready = ReadyFlag(w, s);
	o.ctx = CtxLen(s);
	sslm_gpu_seq_rows_open(s, &o.open);
	uint32_t page = 0;
	(void)sslm_gpu_test_only_seq_table_entry(s, 0, &page, &o.mapped);
	return o;
}
void Unchanged(World& w, SslmGpuSequenceHandle* s, const Obs& before, const std::string& tag) {
	const Obs after = Observe(w, s);
	Check(after == before, F("%s: changed the handle (before %s; after %s)", tag.c_str(), before.Str().c_str(), after.Str().c_str()));
}
S EmbedRows(World& w, SslmGpuSequenceHandle* s, int32_t kind, const std::vector<int32_t>& rows, uint32_t flags, int32_t* admitted) {
	SslmGpuRowsSpec spec{sizeof(SslmGpuRowsSpec), kind, rows.data(), static_cast<int32_t>(rows.size()), flags};
	*admitted = -9;
	return sslm_gpu_seq_embed_rows(w.ctx, s, &spec, admitted);
}
int32_t RowsOpen(SslmGpuSequenceHandle* s) {
	int32_t r = -1;
	return sslm_gpu_seq_rows_open(s, &r) == S::SSLM_OK ? r : -1;
}
int32_t LastCommitted(SslmGpuSequenceHandle* s) {
	int32_t r = -1;
	return sslm_gpu_seq_last_rows_committed(s, &r) == S::SSLM_OK ? r : -1;
}
// One-layer slices until the chunk closes; the closing ready's status.
S CloseChunk(World& w, const Model& m, SslmGpuSequenceHandle* s) {
	for (int i = 0; i < 64; ++i) {
		S st = sslm_decode_step_gpu(w.ctx, s, nullptr, m.dpl());
		if (st == S::SSLM_OK) st = mr2::Ready(w, s);
		if (st != S::SSLM_OK) return st;
		if (RowsOpen(s) == 0) return S::SSLM_OK;
	}
	return S::SSLM_DEVICE_LOST;
}
// The one-row twin of a forced row (1.17's walked embed, then the layers, no finish), and of a row embedded unwalked.
S TwinForced(World& w, const Model& m, SslmGpuSequenceHandle* s, int32_t tok) {
	S st = sslm_gpu_seq_embed_schema_token(w.ctx, s, tok);
	for (uint32_t g = 0; st == S::SSLM_OK && g <= m.layers(); ++g) {
		st = mr2::Slice(w, s, m.dpl());
		if (st == S::SSLM_OK && LayerIdx(s) >= m.layers()) return S::SSLM_OK;
	}
	return st == S::SSLM_OK ? S::SSLM_DEVICE_LOST : st;
}
S TwinPlain(World& w, const Model& m, SslmGpuSequenceHandle* s, int32_t tok) { return mr2::FeedRow(w, s, tok, m.dpl(), m.layers()); }
// The decode bridge's next token from a ready handle (the finish walks it): the produced token in hand.
int32_t Produce(World& w, const Model& m, SslmGpuSequenceHandle* s, int32_t last, S* st) {
	int32_t t = -1;
	*st = SslmGpuSeqDecodeStepForG5Bridge(w.ctx, s, last, m.full_budget(), &t);
	return t;
}
int32_t NotForced(int32_t forced) { return forced == 220 ? 221 : 220; }  // an ordinary id, never the forced one

// ------------------------------------------------------------------------------------------------ 2.3
void Cell23(Fsfd& fx) {
	g_cell = "2.3";
	const Model& m = fx.m;
	const std::vector<int32_t> prompt = Prompt(12, 23);
	// Divergence at row k: LONGKEY's start chain is at least 16 tokens (the fixture's self-check).
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int M : {2, 4, 8})
			for (int k = 1; k < M; ++k) {
				const std::string tag = F("%s M%d diverging at row %d", pooled ? "pooled" : "dedicated", M, k);
				World w(m);
				if (!w.ok) { Check(false, tag); continue; }
				SslmGpuSequenceHandle* a = NewSeq(w, pooled != 0, 128, kLongKey);
				SslmGpuSequenceHandle* t = NewSeq(w, pooled != 0, 128, kLongKey);
				Check(PromptBridge(w, m, a, prompt) == S::SSLM_OK && PromptBridge(w, m, t, prompt) == S::SSLM_OK, tag + ": prompts");
				std::vector<int32_t> chain = Chain(w, a);
				if (static_cast<int>(chain.size()) < M) { Check(false, F("%s: chain of %zu", tag.c_str(), chain.size())); continue; }
				std::vector<int32_t> rows(chain.begin(), chain.begin() + M);
				rows[k] = NotForced(chain[k]);
				int32_t adm = -9;
				StatusIs(EmbedRows(w, a, SSLM_GPU_ROWS_SCHEMA_FORCED, rows, 0, &adm), S::SSLM_OK, tag + ": embed_rows");
				Check(adm == k && RowsOpen(a) == k, F("%s: admitted %d rows_open %d, want %d", tag.c_str(), adm, RowsOpen(a), k));
				StatusIs(CloseChunk(w, m, a), S::SSLM_OK, tag + ": close");
				Check(LastCommitted(a) == k, F("%s: last_rows_committed %d, want %d", tag.c_str(), LastCommitted(a), k));
				for (int r = 0; r < k; ++r) Check(TwinForced(w, m, t, chain[r]) == S::SSLM_OK, tag + ": twin");
				Check(Walk(a) == Walk(t), F("%s: walk %u, twin %u", tag.c_str(), Walk(a), Walk(t)));
				Check(ReadyFlag(w, a) == ReadyFlag(w, t), F("%s: ready %d, twin %d", tag.c_str(), ReadyFlag(w, a), ReadyFlag(w, t)));
				BlobEq(Blob(w, a, tag), Blob(w, t, tag), tag + " vs the one-row twin");
			}
	// Headroom below the matching prefix admits the headroom (a dedicated cap, and a pooled budget limit).
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (int room : {1, 3}) {
			const std::string tag = F("%s headroom %d under a matching 6", pooled ? "pooled" : "dedicated", room);
			World w(m);
			if (!w.ok) { Check(false, tag); continue; }
			const int32_t cap = static_cast<int32_t>(prompt.size()) + room;
			SslmGpuSequenceHandle* a = NewSeq(w, pooled != 0, cap, kLongKey);
			SslmGpuSequenceHandle* t = NewSeq(w, pooled != 0, cap, kLongKey);
			Check(PromptBridge(w, m, a, prompt) == S::SSLM_OK && PromptBridge(w, m, t, prompt) == S::SSLM_OK, tag + ": prompts");
			const std::vector<int32_t> chain = Chain(w, a);
			if (chain.size() < 6) { Check(false, tag + ": chain"); continue; }
			int32_t adm = -9;
			StatusIs(EmbedRows(w, a, SSLM_GPU_ROWS_SCHEMA_FORCED, {chain.begin(), chain.begin() + 6}, 0, &adm), S::SSLM_OK, tag);
			Check(adm == room, F("%s: admitted %d, want %d", tag.c_str(), adm, room));
			StatusIs(CloseChunk(w, m, a), S::SSLM_OK, tag + ": close");
			for (int r = 0; r < room; ++r) Check(TwinForced(w, m, t, chain[r]) == S::SSLM_OK, tag + ": twin");
			Check(Walk(a) == Walk(t), F("%s: walk %u, twin %u", tag.c_str(), Walk(a), Walk(t)));
			BlobEq(Blob(w, a, tag), Blob(w, t, tag), tag + " vs the one-row twin");
		}

	// Zero admitted: the precedence of rulings item 11. Each case: a status, and nothing written.
	struct Z {
		const char* name;
		const char* file;     // the fixture: fsfd (a table) or pkv_def (schemas, no tokenizer: no table)
		bool bound;           // a schema bound
		int room;             // context rows left (-1: plenty)
		bool row0_forced;     // row 0 (row 1 under the flag) the forced token
		uint32_t flags;
		int32_t count;        // rows in the spec (0: a bad spec)
		S want;
	};
	const std::vector<Z> zs = {
	    {"bad spec (count 0) beats no table", "pkv_def", true, 0, false, 0, 0, kArgStatus},
	    {"bad spec (unknown flag) beats a wrong row 0", "fsfd", true, -1, false, 0x2u, 3, kArgStatus},
	    {"no table beats no headroom", "pkv_def", true, 0, false, 0, 3, kTableAbsent},
	    {"no table beats unbound", "pkv_def", false, -1, false, 0, 3, kTableAbsent},
	    {"no table, bound, room", "pkv_def", true, -1, false, 0, 3, kTableAbsent},
	    {"no headroom beats a wrong row 0", "fsfd", true, 0, false, 0, 3, kCapStatus},
	    {"no headroom beats unbound", "fsfd", false, 0, false, 0, 3, kCapStatus},
	    {"no headroom, matching rows", "fsfd", true, 0, true, 0, 3, kCapStatus},
	    {"row 0 not the forced token", "fsfd", true, -1, false, 0, 3, S::SSLM_SEQUENCE_REJECTED},
	    {"no schema bound", "fsfd", false, -1, true, 0, 3, S::SSLM_SEQUENCE_REJECTED},
	    {"FIRST_PREWALKED: row 1 not the forced token", "fsfd", true, -1, false, SSLM_GPU_ROWS_FIRST_PREWALKED, 3, S::SSLM_SEQUENCE_REJECTED},
	};
	Fsfd pkv;
	if (!pkv.Load("pkv_def.sslm")) { Check(false, "pkv_def.sslm (the no-table fixture) did not load"); return; }
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (const Z& z : zs) {
			const std::string tag = F("%s: %s", pooled ? "pooled" : "dedicated", z.name);
			const Model& zm = std::string(z.file) == "fsfd" ? m : pkv.m;
			World w(zm);
			if (!w.ok) { Check(false, tag); continue; }
			const std::vector<int32_t> p = std::string(z.file) == "fsfd" ? prompt : std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8};
			const int32_t cap = z.room < 0 ? 128 : static_cast<int32_t>(p.size()) + z.room;
			// The no-table fixture's schema (pkv_def carries one, S8; its name is the first the lookup finds).
			const char* schema = std::string(z.file) == "fsfd" ? kLongKey : nullptr;
			SslmGpuSequenceHandle* s = NewSeq(w, pooled != 0, cap, nullptr);
			if (!s) { Check(false, tag); continue; }
			if (z.bound) {
				const int32_t idx = schema ? SchemaIndex(w, schema) : 0;
				StatusIs(SslmGpuSeqSetSchemaForG5Bridge(w.ctx, s, idx), S::SSLM_OK, tag + ": bind");
			}
			Check(PromptBridge(w, zm, s, p) == S::SSLM_OK, tag + ": prompt");
			// The rows: the chain where there is one (row 0 forced or not), else ordinary ids.
			std::vector<int32_t> rows;
			if (std::string(z.file) == "fsfd" && z.bound) {
				std::vector<int32_t> chain = Chain(w, s);
				if (chain.size() < 3) { Check(false, tag + ": chain"); continue; }
				rows.assign(chain.begin(), chain.begin() + 3);
				if (z.flags & SSLM_GPU_ROWS_FIRST_PREWALKED) {
					rows = {300, chain[0], chain[1]};  // row 0 is the prewalked row; row 1 must be chain[0]
					if (!z.row0_forced) rows[1] = NotForced(chain[0]);
				} else if (!z.row0_forced) {
					rows[0] = NotForced(chain[0]);
				}
			} else if (std::string(z.file) == "pkv_def") {
				rows = {30, 31, 32};  // in pkv_def's vocabulary (256), so the table's status is reached (ruling 16)
			} else {
				rows = {300, 301, 302};
			}
			rows.resize(static_cast<size_t>(std::max(z.count, 0)));
			const Obs before = Observe(w, s);
			int32_t adm = -9;
			SslmGpuRowsSpec spec{sizeof(SslmGpuRowsSpec), SSLM_GPU_ROWS_SCHEMA_FORCED, rows.empty() ? nullptr : rows.data(), z.count, z.flags};
			if (z.count == 0) {
				static const int32_t one = 300;
				spec.tokens = &one;
			}
			// No headroom on a pooled handle is its budget limit (below the model's cap): the pooled limit status, as
			// MR3's 5.2 and 1.17's jump verb give it (the test author's reading of rulings item 11 for pooled handles).
			const S want = (pooled && z.want == kCapStatus) ? S::SSLM_GPU_KV_BUDGET_EXCEEDED : z.want;
			StatusIs(sslm_gpu_seq_embed_rows(w.ctx, s, &spec, &adm), want, tag);
			Check(adm == -9 || adm == 0, F("%s: out_admitted written %d (nothing is written on a refusal)", tag.c_str(), adm));
			Unchanged(w, s, before, tag);
		}
}

// ------------------------------------------------------------------------------------------------ 8.10
void Cell810(Fsfd& fx) {
	g_cell = "8.10";
	const Model& m = fx.m;
	const std::vector<int32_t> prompt = Prompt(10, 810);
	int cases = 0;
	for (int pooled = 0; pooled <= 1; ++pooled)
		for (const char* schema : {kPotion, kPromptResult, kLongKey}) {
			// Produce tokens until one leaves a forced chain of at least one row; then the span is [t, chain...] (<= 8).
			const std::string base = F("%s %s", pooled ? "pooled" : "dedicated", schema);
			for (int step = 0; step < 24; ++step) {
				World w(m);
				if (!w.ok) { Check(false, base); break; }
				SslmGpuSequenceHandle* a = NewSeq(w, pooled != 0, 200, schema);
				SslmGpuSequenceHandle* t = NewSeq(w, pooled != 0, 200, schema);
				SslmGpuSequenceHandle* j = NewSeq(w, pooled != 0, 200, schema);
				for (auto* h : {a, t, j}) Check(PromptBridge(w, m, h, prompt) == S::SSLM_OK, base + ": prompt");
				// The same history on all three: `step` decode-bridge tokens (forced spans walked by the verb), then one more.
				int32_t last = prompt.back(), tok = -1;
				bool ok = true;
				for (int d = 0; d <= step && ok; ++d) {
					for (auto* h : {a, t, j}) {
						if (d > 0) {
							// Drain the chain left by the previous token with 1.17's verb, as the canonical loop does.
							std::vector<int32_t> out(64);
							int32_t n = 0;
							const S js = sslm_gpu_seq_jump_forward(w.ctx, h, tok, 64, nullptr, out.data(), 64, &n);
							ok = ok && js == S::SSLM_OK;
						}
					}
					if (!ok) break;
					S st = S::SSLM_OK;
					const int32_t ta = Produce(w, m, a, d == 0 ? last : tok, &st);
					S st2 = S::SSLM_OK, st3 = S::SSLM_OK;
					const int32_t tt = Produce(w, m, t, d == 0 ? last : tok, &st2);
					const int32_t tj = Produce(w, m, j, d == 0 ? last : tok, &st3);
					ok = st == S::SSLM_OK && st2 == S::SSLM_OK && st3 == S::SSLM_OK && ta >= 0 && ta == tt && ta == tj;
					tok = ta;
				}
				if (!ok) break;  // a dead end or an accepting end: this schema has no more cases
				const std::vector<int32_t> chain = Chain(w, a);
				if (chain.empty()) continue;
				const std::string tag = F("%s after %d tokens (produced %d, chain %zu)", base.c_str(), step + 1, tok, chain.size());
				++cases;
				std::vector<int32_t> rows{tok};
				for (size_t i = 0; i < chain.size() && rows.size() < 8; ++i) rows.push_back(chain[i]);
				const int32_t span = static_cast<int32_t>(rows.size());
				// Without the flag: nothing admitted, nothing written (row 0 is matched against the chain, and t is not it).
				if (rows[0] != chain[0]) {
					const Obs before = Observe(w, a);
					int32_t adm = -9;
					StatusIs(EmbedRows(w, a, SSLM_GPU_ROWS_SCHEMA_FORCED, rows, 0, &adm), S::SSLM_SEQUENCE_REJECTED, tag + ": without FIRST_PREWALKED");
					Unchanged(w, a, before, tag + ": without FIRST_PREWALKED");
				}
				// With the flag: the whole span, the chain walked once.
				int32_t adm = -9;
				StatusIs(EmbedRows(w, a, SSLM_GPU_ROWS_SCHEMA_FORCED, rows, SSLM_GPU_ROWS_FIRST_PREWALKED, &adm), S::SSLM_OK, tag + ": with FIRST_PREWALKED");
				Check(adm == span, F("%s: admitted %d of %d", tag.c_str(), adm, span));
				StatusIs(CloseChunk(w, m, a), S::SSLM_OK, tag + ": close");
				Check(TwinPlain(w, m, t, tok) == S::SSLM_OK, tag + ": twin's produced row");
				for (size_t r = 1; r < rows.size(); ++r) Check(TwinForced(w, m, t, rows[r]) == S::SSLM_OK, tag + ": twin's forced row");
				std::vector<int32_t> out(64);
				int32_t n = 0;
				StatusIs(sslm_gpu_seq_jump_forward(w.ctx, j, tok, span, nullptr, out.data(), 64, &n), S::SSLM_OK, tag + ": 1.17's verb");
				Check(n == span - 1, F("%s: 1.17's verb admitted %d forced rows, want %d", tag.c_str(), n, span - 1));
				Check(Walk(a) == Walk(j) && Walk(a) == Walk(t), F("%s: walk %u, 1.17's verb %u, twin %u", tag.c_str(), Walk(a), Walk(j), Walk(t)));
				Check(CtxLen(a) == CtxLen(j), F("%s: context %lld, 1.17's verb %lld", tag.c_str(), (long long)CtxLen(a), (long long)CtxLen(j)));
				BlobEq(Blob(w, a, tag), Blob(w, t, tag), tag + " vs the one-row twin");
				break;
			}
		}
	Check(cases >= 4, F("only %d forced spans after a produced token were found", cases));
}

// ------------------------------------------------------------------------------------------------ 7.9
// fsfd_refuse (the last layer's attn_residual patched, mr4_fixtures.py) refuses some rows and not others. A prompt
// whose one-row run of longkey's 26-token start chain first refuses at row r >= 7 gives every j in 0..7: the pass
// starts at row o = r - j (rows 0..o-1 admitted first by the verb, which they pass), so its row j is the first refusal.
// The produced-token shape: the pending row is the pass's row 0; its cases are those the search meets (j = 0 when
// the produced token itself refuses, j > 0 when a chain row does).
std::vector<int32_t> ChainOf(World& w, SslmGpuSequenceHandle* s) { return Chain(w, s); }
// The first row of `rows` (after the prompt; the first row unwalked when `first_plain`) whose layers refuse, or -1.
int FirstRefusalRun(Fsfd& fx, uint32_t stream, bool produced, std::vector<int32_t>* rows_out, int32_t* tok_out) {
	World w(fx.m);
	if (!w.ok) return -2;
	SslmGpuSequenceHandle* s = NewSeq(w, false, 200, produced ? kPotion : kLongKey);
	const std::vector<int32_t> p = Prompt(6, stream);
	if (PromptBridge(w, fx.m, s, p) != S::SSLM_OK) return -2;
	std::vector<int32_t> rows;
	if (produced) {
		S st = S::SSLM_OK;
		const int32_t t = Produce(w, fx.m, s, p.back(), &st);
		if (st != S::SSLM_OK || t < 0) return -2;
		*tok_out = t;
		rows.push_back(t);
	}
	for (int32_t c : Chain(w, s)) rows.push_back(c);
	*rows_out = rows;
	// The one-row run on a fresh handle with the same history.
	World w2(fx.m);
	SslmGpuSequenceHandle* t = NewSeq(w2, false, 200, produced ? kPotion : kLongKey);
	if (PromptBridge(w2, fx.m, t, p) != S::SSLM_OK) return -2;
	if (produced) {
		S st = S::SSLM_OK;
		if (Produce(w2, fx.m, t, p.back(), &st) != *tok_out) return -2;
	}
	for (size_t r = 0; r < rows.size(); ++r) {
		const S st = (produced && r == 0) ? TwinPlain(w2, fx.m, t, rows[r]) : TwinForced(w2, fx.m, t, rows[r]);
		if (st == S::SSLM_SEQUENCE_REJECTED) return static_cast<int>(r);
		if (st != S::SSLM_OK) return -2;
	}
	return -1;
}
void JumpRestsAtEntry(Fsfd& rf, uint32_t stream, bool produced, int32_t tok, int o, int len, int j, const std::string& tag0) {
	const Model& m = rf.m;
	for (int pooled = 0; pooled <= 1; ++pooled) {
		const std::string tag = F("%s %s", tag0.c_str(), pooled ? "pooled" : "dedicated");
		World w(m);
		if (!w.ok) { Check(false, tag); continue; }
		SslmGpuSequenceHandle* s = NewSeq(w, pooled != 0, 200, produced ? kPotion : kLongKey);
		const std::vector<int32_t> p = Prompt(6, stream);
		Check(PromptBridge(w, m, s, p) == S::SSLM_OK, tag + ": prompt");
		int32_t pend = -1;
		if (produced) {
			S st = S::SSLM_OK;
			pend = Produce(w, m, s, p.back(), &st);
			Check(st == S::SSLM_OK && pend == tok, tag + ": the produced token");
		}
		std::vector<int32_t> out(64);
		int32_t n = -9;
		if (o > 0) {
			StatusIs(sslm_gpu_seq_jump_forward(w.ctx, s, -1, o, nullptr, out.data(), 64, &n), S::SSLM_OK, tag + F(": the verb over rows 0..%d", o - 1));
			Check(n == o, F("%s: %d rows before the window, want %d", tag.c_str(), n, o));
		}
		const Obs entry = Observe(w, s);
		n = -9;
		const S st = sslm_gpu_seq_jump_forward(w.ctx, s, pend, len, nullptr, out.data(), 64, &n);
		// 1.17's contract (gpu_1p0.h; the verb keeps it, plan §4.3, §4.7): a fault in the pass other than an allocation
		// failure, the guard's refusal included, restores the entry state and returns SSLM_DEVICE_LOST.
		StatusIs(st, S::SSLM_DEVICE_LOST, tag + F(": a pass of %d rows refusing at its row %d", len, j));
		Unchanged(w, s, entry, tag + ": rests at entry");
	}
}
void Cell79(Fsfd& rf) {
	g_cell = "7.9";
	// The ready shape: one prompt whose start chain first refuses at row r >= 7.
	int r = -1;
	uint32_t stream = 0;
	std::vector<int32_t> rows;
	int32_t tok = -1;
	for (uint32_t st = 1; st < 300 && r < 7; ++st) {
		const int got = FirstRefusalRun(rf, st, false, &rows, &tok);
		if (got >= 7) { r = got; stream = st; }
	}
	Check(r >= 7, "no prompt refuses first at row 7 or later of longkey's start chain (fsfd_refuse)");
	if (r >= 7) {
		std::printf("7.9: stream %u refuses first at chain row %d\n", stream, r);
		for (int j = 0; j < 8; ++j) {
			const int o = r - j, len = std::min<int>(8, static_cast<int>(rows.size()) - o);
			JumpRestsAtEntry(rf, stream, false, -1, o, len, j, F("ready, row %d of %d", j, len));
		}
	}
	// The produced-token shape: whatever first-refusal rows the search meets, at least one.
	std::set<int> seen;
	for (uint32_t st = 1; st < 120 && seen.size() < 4; ++st) {
		const int got = FirstRefusalRun(rf, st, true, &rows, &tok);
		if (got < 0 || got >= 8 || seen.count(got)) continue;
		seen.insert(got);
		const int len = std::min<int>(8, static_cast<int>(rows.size()));
		JumpRestsAtEntry(rf, st, true, tok, 0, len, got, F("after produced token %d, row %d of %d", tok, got, len));
	}
	Check(!seen.empty(), "no produced-token pass with a refusing row was found");
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
	Fsfd fx, rf;
	if (!fx.Load("fsfd.sslm")) return 1;
	const std::vector<std::pair<const char*, std::function<void()>>> cells = {
	    {"2.3", [&] { Cell23(fx); }},
	    {"8.10", [&] { Cell810(fx); }},
	    {"7.9", [&] { if (rf.Load("fsfd_refuse.sslm")) Cell79(rf); else { g_cell = "7.9"; Check(false, "fsfd_refuse.sslm did not load"); } }},
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
		all = all && f == 0;
	}
	std::printf("cells: %s\n", all ? "PASS" : "RED");
	return all ? 0 : 1;
}
