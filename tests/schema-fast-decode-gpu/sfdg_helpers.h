// Schema fast-decode plan rev 1.4.4 §11.2: the GPU cells' helpers (slices G1-F, G1-M, G1-J). One cell group per
// executable, each its own main(). Box only: the cells need a hardware D3D12 device (§11.0, *Placement*); the
// cloud compiles and links them (build_sfd_gpu_mingw.sh).
//
// Conventions reused, not invented: the CHECK/CHECK_MSG harness, FinishSuite and StatusName of
// tests/t2791-gpu-prefill-read-red-suite/fixture_common.h, the paged-KV GPU suite's environment and token
// driver (tests/paged-kv-gpu/g1e_helpers.h) and its CPU twin's model wrapper (g1e_cpu_twin.h). The fixture is
// F-SFD (tools/gen_schema_fast_decode_fixture.py): fsfd.sslm and its reference fsfd_ref.txt, in the directory
// --fixture-dir=DIR or $SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR names (the CPU suite's variable). A missing
// file is a failed check, never a skip.
//
// The reference is the CPU verb on the same fixture, or GPU off (§11.2). The CPU side is driven through the
// shipped C ABI here, restating the CPU suite's rig (tests/test_schema_fast_decode.cpp, Rig) over this
// harness.
//
// The new statuses are compared through test-side constants (ordinals 31 and 32, §7.3), as they were before the
// enumerators landed (G1-F's red round); the static_asserts below pin them to the enum (G0.2 checks the mirrors).
#ifndef SSLM_SFDG_HELPERS_H
#define SSLM_SFDG_HELPERS_H

#include "g1e_cpu_twin.h"  // tests/paged-kv-gpu: g1e_helpers.h, fixture_common.h, the CPU ABI
#include "d3d12_harness.h"  // the debug layer's info queue, read at exit (Finish, below)
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/sha256.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace sfdg {

using namespace g1e;

// ---- the statuses §7.3 appends (G0.2 pins the mirrors) -------------------------------------------------
constexpr SslmGpuStatus kForcedTableAbsent = static_cast<SslmGpuStatus>(31);
constexpr SslmGpuStatus kContextCapExceeded = static_cast<SslmGpuStatus>(32);
static_assert(kForcedTableAbsent == SslmGpuStatus::SSLM_GPU_FORCED_TABLE_ABSENT, "§7.3: ordinal 31");
static_assert(kContextCapExceeded == SslmGpuStatus::SSLM_GPU_CONTEXT_CAP_EXCEEDED, "§7.3: ordinal 32");
inline const char* Name(GS s) {
	if (s == kForcedTableAbsent) return "SSLM_GPU_FORCED_TABLE_ABSENT";
	if (s == kContextCapExceeded) return "SSLM_GPU_CONTEXT_CAP_EXCEEDED";
	return StatusName(s);
}

// ---- the fixture ------------------------------------------------------------------------------------------
inline std::string FixtureDir(int argc, char** argv) {
	std::string dir;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--fixture-dir=", 0) == 0) dir = a.substr(14);
	}
	if (dir.empty()) {
		const char* e = std::getenv("SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR");
		if (e) dir = e;
	}
	CHECK_MSG(!dir.empty(), "no fixture directory: pass --fixture-dir=DIR or set SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR "
	                        "(python tools/gen_schema_fast_decode_fixture.py DIR writes it)");
	return dir;
}

constexpr const char* kPotion = "potion_shop_order";
constexpr const char* kPromptResult = "prompt_result";
constexpr const char* kLongKey = "longkey";
constexpr const char* kUtf8Enum = "utf8enum";
constexpr const char* kSchemas[] = {kPotion, kPromptResult, kLongKey, kUtf8Enum};

// Cell 6.6's pinned table digests (tests/test_schema_fast_decode.cpp, kGoldenTables): SHA-256 over every state's
// LE32 next_forced then LE32 |F|. The GPU table's must equal them (G12.4).
struct GoldenTable {
	const char* schema;
	const char* digest;
};
constexpr GoldenTable kGoldenTables[] = {
    {kPotion, "7024d46353a94c1421ef9f6b57c162ea04671c58416c05bf52f3eae004bc337c"},
    {kPromptResult, "8fd3adfaf374de598b8bffd76fc74f1f19e746e573eb97006158afabdb69d8fd"},
    {kLongKey, "aa2c5c6b6e226720e06bf6d1c13c35dbc912cbd2e40e785e4acfe8c068530b3e"},
    {kUtf8Enum, "80cc3d912cbfad5daa498608bf488b245a91fef60f44b02db2b10db3dd7414d3"},
};
// Cell 6.6's pinned CPU jump-forward streams (greedy, chunk budget 64, the fixed prompts below): G7.1's digest.
struct GoldenStream {
	const char* schema;
	size_t length;
	const char* digest;
};
constexpr GoldenStream kGoldenStreams[] = {
    {kPotion, 24, "5498cef23900bfbfad790d7174b6badc04bddc58afdca629daf8e2d0b48488bd"},
    {kPromptResult, 42, "16072100850158b6b6314415875d694601b47c80de8335be6d276230c65b08fb"},
};
// MR5 RE-POINT (multi-row GPU plan, ruling 29 M1 as widened): 1.18 removes Canonical, so the streams above are no
// longer any loop's. Exact's stream is Off's: cell 6.X5's pinned Off streams (the CPU suite's kXOffGoldens: greedy,
// decode-only, the same prompts), which the GPU's Exact loop must equal.
constexpr GoldenStream kOffGoldens[] = {
    {kPotion, 33, "7953dd212aa523bc1800991ac79268947b1c51df48e20ee91c0075e817244bed"},
    {kPromptResult, 159, "674e41c93d082d38d189b1a55466bb03570fc760bdc76911c80e1d9584ec236c"},
};
constexpr const char* kPromptPotionText = "Customer: I would like to buy a health potion, please.\nOrder:";
constexpr const char* kPromptPrText = "Summarize the request in one line.\nNotes:";

inline std::string HexDigest(const std::vector<int32_t>& ids) {
	superslm::Sha256 h;
	for (int32_t x : ids) {
		const uint32_t u = static_cast<uint32_t>(x);
		const uint8_t b[4] = {static_cast<uint8_t>(u), static_cast<uint8_t>(u >> 8), static_cast<uint8_t>(u >> 16),
		                      static_cast<uint8_t>(u >> 24)};
		h.Update(b, 4);
	}
	uint8_t out[32];
	h.Final(out);
	static const char* hex = "0123456789abcdef";
	std::string s;
	for (uint8_t c : out) {
		s += hex[c >> 4];
		s += hex[c & 15];
	}
	return s;
}

inline std::string Str(const std::vector<int32_t>& v, size_t limit = 40) {
	std::string s = "{";
	for (size_t i = 0; i < v.size() && i < limit; ++i) s += (i ? "," : "") + std::to_string(v[i]);
	if (v.size() > limit) s += ",...(" + std::to_string(v.size()) + ")";
	return s + "}";
}

// fsfd_ref.txt (the generator's reference): per schema, per state: accepting, forced, |F|, the reference
// chain and a token path that reaches the state from the start.
struct RefState {
	bool accepting = false, forced = false;
	uint32_t f_len = 0;
	std::vector<int32_t> chain, path;
};
struct RefSchema {
	std::string name, digest;
	uint32_t n = 0, start = 0;
	std::vector<RefState> st;
};
inline std::map<std::string, RefSchema> ParseRef(const std::string& path) {
	std::map<std::string, RefSchema> out;
	std::ifstream f(path);
	CHECK_MSG(f.good(), "cannot read %s", path.c_str());
	std::string line;
	RefSchema* cur = nullptr;
	while (std::getline(f, line)) {
		std::istringstream is(line);
		std::string tag;
		is >> tag;
		if (tag == "schema") {
			RefSchema r;
			is >> r.name >> r.n >> r.start >> r.digest;
			out[r.name] = r;
			cur = &out[r.name];
		} else if (tag == "state" && cur) {
			RefState s;
			uint32_t idx = 0, acc = 0, forced = 0;
			std::string fhex;
			size_t cn = 0, pn = 0;
			is >> idx >> acc >> forced >> s.f_len >> fhex >> cn;
			s.accepting = acc != 0;
			s.forced = forced != 0;
			s.chain.resize(cn);
			for (auto& x : s.chain) is >> x;
			is >> pn;
			s.path.resize(pn);
			for (auto& x : s.path) is >> x;
			cur->st.push_back(s);
		}
	}
	return out;
}

// ---- the GPU side -----------------------------------------------------------------------------------------
// The context Finish reads beside the singleton (set by GRig::Open, cleared by ~GRig).
inline SslmGpuContext*& FinishContext() {
	static SslmGpuContext* ctx = nullptr;
	return ctx;
}
struct GRig {
	Env env;
	std::vector<uint8_t> raw;
	std::map<std::string, RefSchema> ref;
	std::vector<SslmGpuSequenceHandle*> seqs;
	bool ok = false;
	bool Open(int argc, char** argv, bool load_ref = true) {
		const std::string dir = FixtureDir(argc, argv);
		if (dir.empty()) return false;
		const bool opened = env.Open(dir + "/fsfd.sslm", false);
		FinishContext() = env.ctx;  // Finish reads this context's device too (engine code review N3)
		if (!opened) return false;
		raw = env.fx.bytes;
		if (load_ref) ref = ParseRef(dir + "/fsfd_ref.txt");
		ok = !load_ref || ref.size() == 4;
		CHECK_MSG(ok, "fsfd_ref.txt: %zu schemas, expected 4", ref.size());
		return ok;
	}
	~GRig() {
		ReleaseSeqs();
		if (FinishContext() == env.ctx) FinishContext() = nullptr;
	}
	void ReleaseSeqs() {
		for (SslmGpuSequenceHandle* s : seqs) sslm_gpu_seq_release(env.ctx, s);
		seqs.clear();
	}
	SslmGpuContext* ctx() const { return env.ctx; }
	SslmGpuModelHandle* model() const { return env.model(); }
	int32_t Index(const char* schema) const { return SslmGpuSchemaLookupForG5Bridge(env.model(), schema); }
	// A dedicated sequence at the model's cap, bound to `schema` (null: unbound).
	GSeq New(const char* schema, SslmGpuModelHandle* m = nullptr) {
		GSeq s;
		if (!m) m = env.model();
		const GS st = sslm_gpu_seq_create(env.ctx, m, env.fx.model_cap, &s.h);
		CHECK_MSG(st == GS::SSLM_OK && s.h, "sslm_gpu_seq_create: %s", Name(st));
		if (!s.h) return s;
		seqs.push_back(s.h);
		if (schema) {
			const int32_t i = SslmGpuSchemaLookupForG5Bridge(m, schema);
			const GS b = SslmGpuSeqSetSchemaForG5Bridge(env.ctx, s.h, i);
			CHECK_MSG(i >= 0 && b == GS::SSLM_OK, "bind %s: index %d, %s", schema, i, Name(b));
		}
		return s;
	}
	GS Prompt(GSeq& s, const std::vector<int32_t>& t) { return Prefill(env, s, t); }
	// SCHEMA_CONTENT prefill (the bridge), which leaves the ready flag set when it admits a token.
	GS Content(GSeq& s, const std::vector<int32_t>& t, int32_t* consumed = nullptr) {
		int32_t c = 0;
		if (t.empty()) {
			if (consumed) *consumed = 0;
			return GS::SSLM_OK;
		}
		const GS st = SslmGpuSeqPrefillSchemaContentForG5Bridge(env.ctx, s.h, t.data(), static_cast<int32_t>(t.size()),
		                                                        env.full_budget, &c);
		if (consumed) *consumed = c;
		if (c > 0) {
			s.last = t[static_cast<size_t>(c) - 1];
			s.ready = s.primed = true;
			s.mid = false;
		}
		return st;
	}
	std::vector<uint8_t> Blob(SslmGpuSequenceHandle* h) { return Save(env, h); }
	bool Bindable(SslmGpuSequenceHandle* h, const char* schema) {
		const int32_t i = SslmGpuSchemaLookupForG5Bridge(env.model(), schema);
		return SslmGpuSeqSetSchemaForG5Bridge(env.ctx, h, i) == GS::SSLM_OK;
	}
};

// A second mapping of a fixture file (another residency, or the --untied variant), owned by the cell.
struct Mapped {
	SslmGpuModelHandle* m = nullptr;
	superslm::SslmModelView view{};
	std::vector<uint8_t> bytes;
};

inline bool MapVariant(GRig& rig, const std::string& path, uint32_t flags, Mapped* out) {
	if (!ReadFileBytes(path, &out->bytes)) {
		CHECK_MSG(false, "[G6] cannot read %s", path.c_str());
		return false;
	}
	std::string err;
	if (superslm::SslmModel::Load(out->bytes.data(), out->bytes.size(), out->view, &err) != superslm::SslmModelStatus::Ok) {
		CHECK_MSG(false, "[G6] %s: %s", path.c_str(), err.c_str());
		return false;
	}
	GpuResidencyConfig cfg{};
	cfg.flags = flags;
	const GS st = sslm_gpu_model_map(rig.ctx(), &out->view, cfg, &out->m);
	CHECK_MSG(st == GS::SSLM_OK && out->m, "[G6] map %s (flags %u): %s", path.c_str(), flags, Name(st));
	return st == GS::SSLM_OK && out->m;
}

inline void Unmap(GRig& rig, Mapped* m) {
	if (m->m) sslm_gpu_model_unmap(rig.ctx(), m->m);
	m->m = nullptr;
}

struct GQuery {
	GS st = GS::SSLM_OK;
	std::vector<int32_t> ids;
	int32_t count = -9, accepting = -9;
};
inline GQuery QueryGpu(SslmGpuContext* ctx, SslmGpuSequenceHandle* h, int32_t cap = 512) {
	GQuery q;
	q.ids.assign(static_cast<size_t>(cap > 0 ? cap : 0), -7);
	q.st = sslm_gpu_seq_forced_tokens(ctx, h, cap > 0 ? q.ids.data() : nullptr, cap, &q.count, &q.accepting);
	if (q.st == GS::SSLM_OK && q.count >= 0 && q.count < cap) q.ids.resize(static_cast<size_t>(q.count));
	return q;
}

#if defined(SUPERSLM_GPU_TEST_SEAMS)
inline uint64_t Submissions(SslmGpuContext* ctx) {
	uint64_t n = 0;
	const GS st = sslm_gpu_test_only_submission_count(ctx, &n);
	CHECK_MSG(st == GS::SSLM_OK, "S-GSUB: %s", Name(st));
	return n;
}
#endif

// One finish at the state `path` reaches after `prompt`: reset, prompt, SCHEMA_CONTENT `path` (which leaves the
// ready flag set), then the decode bridge, which finishes the ready row. Returns the token (or -2) and the walk
// state after; -1 with the status on a failure.
struct FinishResult {
	GS st = GS::SSLM_OK;
	int32_t token = -1;
	uint32_t walk = 0;
};
inline FinishResult FinishAt(GRig& rig, GSeq& s, const std::vector<int32_t>& prompt, const std::vector<int32_t>& path) {
	FinishResult r;
	r.st = sslm_gpu_seq_reset(rig.ctx(), s.h);
	s = GSeq{s.h};
	if (r.st == GS::SSLM_OK) r.st = rig.Prompt(s, prompt);
	if (r.st == GS::SSLM_OK) r.st = rig.Content(s, path);
	if (r.st == GS::SSLM_OK) r.token = Token(rig.env, s, &r.st);
	r.walk = SslmGpuSeqWalkStateForG5Bridge(s.h);
	return r;
}

#if defined(SUPERSLM_GPU_TEST_SEAMS)
struct RowsEntry {
	uint64_t rows = 0;
	int32_t device = -1;
};
inline std::vector<RowsEntry> RowsLog() {
	std::vector<RowsEntry> v(sslm_gpu_test_only_rows_log_count());
	for (size_t i = 0; i < v.size(); ++i) sslm_gpu_test_only_rows_log_at(i, &v[i].rows, &v[i].device);
	return v;
}
#endif

// The number of legal ids at `state` of `schema` (its CSR row length), from the parsed SCM1 of the fixture.
inline uint32_t LegalCount(const superslm::SslmModelView& view, const char* schema, uint32_t state) {
	superslm::SchemaMasksTable t;
	const superslm::SslmSectionView* sec = view.Section(superslm::SslmSectionType::SchemaMasks);
	std::string err;
	if (!sec || !superslm::SchemaMasksTable::Parse(sec->data, sec->byte_size, view.config.vocab_size, t, &err)) return 0;
	for (size_t i = 0; i < t.Count(); ++i) {
		const superslm::SchemaEntry* e = t.ByIndex(i);
		if (!e || e->name != schema) continue;
		using superslm::schema_masks_detail::ReadLE32;
		return ReadLE32(e->state_offsets_le + size_t{state + 1} * 4) - ReadLE32(e->state_offsets_le + size_t{state} * 4);
	}
	return 0;
}

// Whether `id` is legal at `state` of `schema` (its mask page bit), from the parsed SCM1 of the fixture.
inline bool LegalAt(const superslm::SslmModelView& view, const char* schema, uint32_t state, int32_t id) {
	superslm::SchemaMasksTable t;
	const superslm::SslmSectionView* sec = view.Section(superslm::SslmSectionType::SchemaMasks);
	std::string err;
	if (!sec || !superslm::SchemaMasksTable::Parse(sec->data, sec->byte_size, view.config.vocab_size, t, &err)) return false;
	for (size_t i = 0; i < t.Count(); ++i) {
		const superslm::SchemaEntry* e = t.ByIndex(i);
		if (!e || e->name != schema) continue;
		const uint8_t* page = e->mask_pages + size_t{state} * t.MaskPageBytes();
		return ((page[id >> 3] >> (id & 7)) & 1u) != 0;
	}
	return false;
}

// ---- G1-J: the verb, the walked embed, the ready getter ----------------------------------------------------
struct JumpOut {
	GS st = GS::SSLM_OK;
	int32_t count = -9;
	std::vector<int32_t> forced;
};
// One verb call. `cap` 0 means max(budget, 64). On a call that admitted rows the GSeq is marked ready, so the
// decode bridge (Token) takes the shortcut.
inline JumpOut Jump(GRig& rig, GSeq& s, int32_t pending, int32_t budget, const SslmGpuJumpOptions* opts = nullptr,
                    int32_t cap = 0) {
	JumpOut j;
	if (cap == 0) cap = budget > 64 ? budget : 64;
	j.forced.assign(static_cast<size_t>(cap), -7);
	j.st = sslm_gpu_seq_jump_forward(rig.ctx(), s.h, pending, budget, opts, j.forced.data(), cap, &j.count);
	j.forced.resize(j.count >= 0 && j.count <= cap ? static_cast<size_t>(j.count) : 0);
	int32_t ready = 0;
	if (sslm_gpu_seq_ready_for_logits(rig.ctx(), s.h, &ready) == GS::SSLM_OK && ready) {
		s.ready = s.primed = true;
		s.mid = false;
	}
	return j;
}
inline int32_t ReadyFlag(SslmGpuContext* ctx, SslmGpuSequenceHandle* h) {
	int32_t r = -1;
	const GS st = sslm_gpu_seq_ready_for_logits(ctx, h, &r);
	return st == GS::SSLM_OK ? r : -100 - static_cast<int32_t>(st);
}
inline int64_t CtxLen(SslmGpuSequenceHandle* h) { return *SslmGpuSeqHandleContextLengthForBench(h); }
inline uint32_t LayerIdx(SslmGpuSequenceHandle* h) { return *SslmGpuSeqHandleLayerIndexForBench(h); }
inline SslmGpuJumpOptions Opts(int32_t mode, const int32_t* draft = nullptr, int32_t n = 0) {
	SslmGpuJumpOptions o{};
	o.struct_size = sizeof(SslmGpuJumpOptions);
	o.mode = mode;
	o.draft = draft;
	o.draft_count = n;
	return o;
}
// Everything a refusal must leave as it was: the blob (context, walk, counters, hidden state, layer, ready flag,
// mirror rows), the walk and ready flag read directly, the layer and context, and a pooled holder's mapped count.
struct SeqObs {
	std::vector<uint8_t> blob;
	uint32_t walk = 0;
	int32_t ready = -1;
	uint32_t layer = 0;
	int64_t ctx = -1;
	uint32_t mapped = 0;
	bool operator==(const SeqObs& o) const {
		return blob == o.blob && walk == o.walk && ready == o.ready && layer == o.layer && ctx == o.ctx &&
		       mapped == o.mapped;
	}
	bool operator!=(const SeqObs& o) const { return !(*this == o); }
};
inline SeqObs Observe(GRig& rig, SslmGpuSequenceHandle* h) {
	SeqObs o;
	o.blob = rig.Blob(h);
	o.walk = SslmGpuSeqWalkStateForG5Bridge(h);
	o.ready = ReadyFlag(rig.ctx(), h);
	o.layer = LayerIdx(h);
	o.ctx = CtxLen(h);
	uint32_t page = 0;
	(void)sslm_gpu_test_only_seq_table_entry(h, 0, &page, &o.mapped);
	return o;
}
// The prefill snapshot's read: its status and bytes.
struct SnapRead {
	GS st = GS::SSLM_OK;
	std::vector<int8_t> codes;
	int64_t m = 0, e = 0;
	bool operator==(const SnapRead& o) const { return st == o.st && codes == o.codes && m == o.m && e == o.e; }
};
inline SnapRead ReadSnapshot(SslmGpuContext* ctx, SslmGpuSequenceHandle* h) {
	SnapRead r;
	r.codes.assign(4096, 0);
	size_t req = 0;
	r.st = sslm_gpu_seq_read_prefill_final_hidden(ctx, h, r.codes.data(), r.codes.size(), &req, &r.m, &r.e);
	r.codes.resize(r.st == GS::SSLM_OK ? req : 0);
	return r;
}
// The probe's composition, the reference for the fused verb (G12.1): the prompt bridge for the pending token (which
// does not walk it), then the schema-content bridge for `chain`.
inline GS Compose(GRig& rig, GSeq& s, int32_t pending, const std::vector<int32_t>& chain) {
	GS st = GS::SSLM_OK;
	if (pending >= 0) st = rig.Prompt(s, {pending});
	if (st == GS::SSLM_OK && !chain.empty()) st = rig.Content(s, chain);
	return st;
}
// A run of the jump loop (the §5.6 host contract): the verb, then one decode; until a dead end. `pending` is the
// decoded token while it is the caller's.
// MR5 RE-POINT (ruling 29 M1 as widened): 1.17's loop called the verb until the query reported no chain, which
// Canonical's -1 call advanced. Under Exact a -1 call is a no-op (the next decode produces the pending token), so
// that loop would spin; the loop is now Exact's host loop: one verb call per decoded token, with that token pending.
struct GJfRun {
	std::vector<int32_t> stream;
	bool dead_end = false;
	GS end = GS::SSLM_OK;
	int verb_calls = 0, decodes = 0;
};
inline GJfRun RunGpuJf(GRig& rig, GSeq& s, int32_t budget, int max_decodes = 400,
                       const std::function<void(GSeq&, int)>& between = nullptr) {
	GJfRun r;
	int32_t pending = -1;
	for (int d = 0; d < max_decodes; ++d) {
		{
			const JumpOut j = Jump(rig, s, pending, budget);
			++r.verb_calls;
			r.stream.insert(r.stream.end(), j.forced.begin(), j.forced.end());
			if (j.st != GS::SSLM_OK) {
				r.end = j.st;
				return r;
			}
			if (ReadyFlag(rig.ctx(), s.h) == 1) pending = -1;  // admitted: the pending row is in
			if (between) between(s, r.verb_calls);
		}
		GS st = GS::SSLM_OK;
		if (pending >= 0) s.last = pending;
		const int32_t t = Token(rig.env, s, &st);
		++r.decodes;
		if (st != GS::SSLM_OK) {
			r.end = st;
			return r;
		}
		if (t == -2) {
			r.dead_end = true;
			return r;
		}
		r.stream.push_back(t);
		pending = t;
	}
	return r;
}
// A prompt of exactly `n` tokens: `base` repeated.
inline std::vector<int32_t> PromptOfLength(const std::vector<int32_t>& base, size_t n) {
	std::vector<int32_t> v;
	for (size_t i = 0; i < n && !base.empty(); ++i) v.push_back(base[i % base.size()]);
	return v;
}
// MR5 RE-POINT (ruling 29 M1 as widened): Exact's expectations, computed without the verb. Exact keeps the pending
// token and the draft's leading ids that equal Off's own next tokens (docs/api.md, "Exact mode"), and leaves the
// sequence at full depth on the last kept row, ready, so the next decode gives Off's next token.
// Off's own next `n` tokens from `s` by the decode bridge alone (a token in hand is embedded first): DecodeN's form,
// -2 appended at a dead end.
inline std::vector<int32_t> OffTokensGpu(GRig& rig, GSeq& s, int64_t n) { return DecodeN(rig.env, s, n); }
constexpr int32_t kJumpExact = 1;  // SSLM_JUMP_EXACT (sslm_abi.h's enum; the GPU verb takes the CPU's modes)
// One verb call with an explicit draft under SSLM_JUMP_EXACT.
inline JumpOut JumpDraft(GRig& rig, GSeq& s, int32_t pending, int32_t budget, const std::vector<int32_t>& draft) {
	SslmGpuJumpOptions o{};
	o.struct_size = sizeof(SslmGpuJumpOptions);
	o.mode = kJumpExact;
	o.draft = draft.empty() ? nullptr : draft.data();
	o.draft_count = static_cast<int32_t>(draft.size());
	return Jump(rig, s, pending, budget, &o);
}
// Exact's kept count: the draft's leading ids that equal `off`, at most `limit`.
inline int32_t ExactKeptPrefix(const std::vector<int32_t>& draft, const std::vector<int32_t>& off, int32_t limit) {
	int32_t k = 0;
	while (k < limit && static_cast<size_t>(k) < draft.size() && static_cast<size_t>(k) < off.size() &&
	       draft[static_cast<size_t>(k)] == off[static_cast<size_t>(k)])
		++k;
	return k;
}

// ---- the CPU side (the reference) ------------------------------------------------------------------------
struct CRig {
	std::unique_ptr<cpu::Model> m;
	std::unique_ptr<cpu::Aligned> pool_buf;
	sslm_kv_pool pool = nullptr;
	std::vector<sslm_seq> seqs;
	bool ok = false;
	explicit CRig(const std::vector<uint8_t>& raw, uint32_t blocks = 8) {
		m = std::make_unique<cpu::Model>(raw);
		if (!m->ok) return;
		const size_t size = blocks * sslm_kv_block_size(m->model) + sslm_kv_pool_overhead_size(m->model, blocks);
		pool_buf = std::make_unique<cpu::Aligned>(size);
		const sslm_status ps = sslm_kv_pool_create(m->model, pool_buf->p, size, blocks, &pool);
		CHECK_MSG(ps == SSLM_OK, "CPU sslm_kv_pool_create: %d", static_cast<int>(ps));
		ok = ps == SSLM_OK;
	}
	~CRig() {
		for (sslm_seq s : seqs) sslm_seq_release(s);
		if (pool) sslm_kv_pool_destroy(pool);
	}
	sslm_model M() const { return m->model; }
	sslm_seq New(const char* schema) {
		if (!ok) return nullptr;
		sslm_seq s = nullptr;
		const sslm_status st = sslm_seq_create(M(), &pool, &s);
		CHECK_MSG(st == SSLM_OK && s, "CPU sslm_seq_create: %d", static_cast<int>(st));
		if (!s) return nullptr;
		seqs.push_back(s);
		if (schema) {
			sslm_schema h = nullptr;
			const sslm_status ls = sslm_schema_lookup(M(), schema, &h);
			const sslm_status bs = ls == SSLM_OK ? sslm_seq_set_schema(s, h) : ls;
			CHECK_MSG(bs == SSLM_OK, "CPU bind %s: %d", schema, static_cast<int>(bs));
		}
		return s;
	}
	sslm_status Prefill(sslm_seq s, const std::vector<int32_t>& t, sslm_span_kind kind) {
		sslm_status st = SSLM_OK;
		for (size_t at = 0; at < t.size();) {
			const int32_t n = static_cast<int32_t>(std::min<size_t>(64, t.size() - at));
			int32_t c = 0;
			st = sslm_prefill(M(), s, t.data() + at, n, n, kind, nullptr, &c);
			if (st != SSLM_OK) break;
			at += static_cast<size_t>(n);
		}
		return st;
	}
	std::vector<int32_t> Tokenize(const char* text) {
		std::vector<int32_t> ids(512);
		int32_t n = static_cast<int32_t>(ids.size());
		const sslm_status st = sslm_tokenize(M(), text, ids.data(), &n);
		CHECK_MSG(st == SSLM_OK, "sslm_tokenize: %d", static_cast<int>(st));
		ids.resize(st == SSLM_OK ? static_cast<size_t>(n) : 0);
		return ids;
	}
	// One greedy decode: the token, -2 at a dead end, or -1000 - status.
	int32_t Decode(sslm_seq s) {
		sslm_decode_params p{};
		sslm_decode_params_init(M(), SSLM_DECODE_MODE_GREEDY, 64, &p);
		int32_t out = -100;
		const sslm_status st = sslm_decode_step_v2(M(), &s, 1, &p, nullptr, &out);
		return st == SSLM_OK ? out : -1000 - static_cast<int32_t>(st);
	}
};

struct CQuery {
	sslm_status st = SSLM_OK;
	std::vector<int32_t> ids;
	int32_t count = -9, accepting = -9;
};
inline CQuery QueryCpu(sslm_model m, sslm_seq s, int32_t cap = 512) {
	CQuery q;
	q.ids.assign(static_cast<size_t>(cap > 0 ? cap : 0), -7);
	q.st = sslm_seq_forced_tokens(m, s, cap > 0 ? q.ids.data() : nullptr, cap, &q.count, &q.accepting);
	if (q.st == SSLM_OK && q.count >= 0 && q.count < cap) q.ids.resize(static_cast<size_t>(q.count));
	return q;
}

// Plan §10 (rev 1.4.2, strike N3; rev 1.4.3, re-check note 4): every G binary reports the debug layer's queued
// messages at exit, each with its ID, so the box driver can compare them with GB0's baseline. One line per message,
// "# SFDG DEBUG LAYER MESSAGE id=<ID> severity=<0..4> category=<n>", then "# SFDG DEBUG LAYER: mode=<m> messages=<n>".
// The harness's own drain then prints the descriptions to stderr and empties the queue. A quiet drain is not proof
// that the recording was correct (d3d12_harness.h's caveat).
// Both devices are read (engine code review N3): the process singleton, which the layer passes submit on, and the
// open GRig's context device, which the map uploads and the device logits run on. Each message line names its
// device ("device=singleton" or "device=context"); the box driver reads the IDs from both.
inline size_t ReportDebugLayer(superslm_gpu::harness::Device& d, const char* which) {
	size_t n = 0;
	if (!d.debug_info_queue) return 0;
	const UINT64 stored = d.debug_info_queue->GetNumStoredMessages();
	for (UINT64 i = 0; i < stored; ++i) {
		SIZE_T len = 0;
		if (FAILED(d.debug_info_queue->GetMessage(i, nullptr, &len)) || len == 0) continue;
		std::vector<char> buf(len);
		D3D12_MESSAGE* m = reinterpret_cast<D3D12_MESSAGE*>(buf.data());
		if (FAILED(d.debug_info_queue->GetMessage(i, m, &len))) continue;
		std::printf("# SFDG DEBUG LAYER MESSAGE id=%d severity=%d category=%d device=%s\n", static_cast<int>(m->ID),
		            static_cast<int>(m->Severity), static_cast<int>(m->Category), which);
		++n;
	}
	(void)d.DrainDebugLayerMessages();
	return n;
}
inline int Finish(const char* name) {
	superslm_gpu::harness::Device& d = superslm_gpu::harness::GetDevice();
	size_t n = ReportDebugLayer(d, "singleton");
	size_t nc = 0;
	if (SslmGpuContext* ctx = FinishContext()) {
		if (void* cd = sslm_gpu_test_only_context_device(ctx)) {
			nc = ReportDebugLayer(*static_cast<superslm_gpu::harness::Device*>(cd), "context");
		}
	}
	std::printf("# SFDG DEBUG LAYER: mode=%d messages=%zu context_messages=%zu\n", d.debug_layer_mode, n + nc, nc);
	return FinishSuite(name);
}

}  // namespace sfdg

#endif  // SSLM_SFDG_HELPERS_H
