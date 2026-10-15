// Paged-KV plan (rev 16.2) step G1, group g1e: helpers for the GPU persistence cells 9.6, 9.8, 9.12,
// 9.13 (its G5 rows) and 8.17 (plan §3.7, §3.9 A10-A21, §4 item 10, §7 dimensions 8 and 9).
//
// Conventions reused, not invented: the CHECK/CHECK_MSG harness, FinishSuite, StatusName,
// GpuModelFixture and IndependentSchema are tests/t2791-gpu-prefill-read-red-suite/fixture_common.h's
// (one cell per executable, as that suite builds them); the token streams, the persona chain, the page
// arithmetic and the two-sided fill probe are the CPU slice's (tests/paged-kv/pkv_common.h,
// pkv_budget_c_helpers.h), transcribed for the GPU verbs so the GPU twins run the same constructions
// as their CPU cells. The fixture is the CPU slice's pkv_def (tools/gen_paged_kv_fixture.py; cap 4096,
// kv_block_size 16, so B = 16 and a whole-cap reservation is 256 pages), so every page count the plan
// writes for 9.9-9.11 (75, 33, 108, 183) holds here unchanged.
//
// HOW THESE CELLS ARE RED BEFORE THE BUILD. They call the G1-declared GPU verbs of
// include/superslm/gpu_1p0.h (pool create/destroy, budgeted create, the prefix verbs, adopt,
// restore-into-pool, the mirror peek), which have no definitions until G3 (pool, budgeted create,
// peek, wait seam), G4 (prefixes, adopt) and G5 (restore-into-pool, the SLM6 save). Each cell
// therefore compiles and fails to LINK (LNK2019 on those symbols), t2791's convention. Once a step
// defines its verbs, the cell links and is red at RUN for the reason its header states, until G5.
//
// Nothing here reads sslm_gpu_kv_pool_stats or *out_shared_pages for a verdict (§8, §9.12's grading
// is by admission): page counts are read by the GPU fill probe, bytes by the mirror peek, the blob,
// or an unsaved twin.
#ifndef SSLM_PKV_GPU_G1E_HELPERS_H
#define SSLM_PKV_GPU_G1E_HELPERS_H

// The CPU twin header (g1e_cpu_twin.h) includes the CPU C ABI, whose C enum `sslm_status` puts
// SSLM_OK and friends at global scope; t2791's `using enum SslmGpuStatus` would collide with it.
// Every GPU status here is therefore written GS::NAME.
#ifndef SUPERSLM_T2791_SUPPRESS_GPU_STATUS_USING_ENUM
#define SUPERSLM_T2791_SUPPRESS_GPU_STATUS_USING_ENUM 1
#endif
#include "fixture_common.h"  // tests/t2791-gpu-prefill-read-red-suite
#include "pkvg_slm6_layout.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace g1e {

using GS = SslmGpuStatus;

// ---- the fixture -------------------------------------------------------------------------------

// --fixture-dir=DIR, else $SUPERSLM_PAGED_KV_FIXTURE_DIR (the CPU slice's variable), holding the
// fixtures tools/gen_paged_kv_fixture.py writes. A missing directory or file is a failed check, never
// a skip: these cells have exactly one fixture.
inline std::string FixturePath(int argc, char** argv, const char* stem = "pkv_def") {
	std::string dir;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--fixture-dir=", 0) == 0) dir = a.substr(14);
	}
	if (dir.empty()) {
		const char* e = std::getenv("SUPERSLM_PAGED_KV_FIXTURE_DIR");
		if (e) dir = e;
	}
	CHECK_MSG(!dir.empty(), "no fixture directory: pass --fixture-dir=DIR or set SUPERSLM_PAGED_KV_FIXTURE_DIR "
	                        "(python tools/gen_paged_kv_fixture.py DIR writes it)");
	return dir.empty() ? std::string() : dir + "/" + stem + ".sslm";
}

// A token stream: tests/paged-kv/reference/pkv_scenarios.h's Stream, verbatim, so the GPU twins feed
// the very tokens the CPU cells feed.
inline std::vector<int32_t> Stream(uint32_t seed, int32_t n, int32_t vocab) {
	std::vector<int32_t> t(static_cast<size_t>(n > 0 ? n : 0));
	uint32_t x = seed * 2654435761u + 12345u;
	for (int32_t i = 0; i < n; ++i) {
		x = x * 1664525u + 1013904223u;
		t[static_cast<size_t>(i)] = static_cast<int32_t>((x >> 8) % static_cast<uint32_t>(vocab));
	}
	return t;
}

constexpr int32_t kVocab = 256;  // pkv_def's vocabulary; every stream stays inside it
// pkv_budget_c_helpers.h's prompts: a world of 1,000, personas of 200, plain prefixes.
inline std::vector<int32_t> WorldTokens() { return Stream(61, 1000, kVocab); }
inline std::vector<int32_t> PersonaTokens(int i) { return Stream(71 + static_cast<uint32_t>(i), 200, kVocab); }
inline std::vector<int32_t> PrefixTokens(int32_t len, uint32_t seed = 0) { return Stream(51 + seed * 7, len, kVocab); }

// ---- page geometry, written here from §3.1 and §3.4 ---------------------------------------------

struct Geo {
	uint32_t L = 0, Hkv = 0, D = 0;
	int64_t cap = 0, kvbs = 0, B = 0;
	size_t hidden = 0;
	// §3.1: B = kv_block_size when it is <= cap and divides it; otherwise B = cap.
	static int64_t PageB(int64_t kvbs, int64_t cap) { return (kvbs > 0 && kvbs <= cap && cap % kvbs == 0) ? kvbs : cap; }
	int64_t CapPages() const { return (cap + B - 1) / B; }
	// §3.4: R(budget) = min(ceil(budget / B) + 1, ceil(cap / B)).
	int64_t R(int64_t budget) const {
		const int64_t r = (budget + B - 1) / B + 1;
		return r < CapPages() ? r : CapPages();
	}
	size_t BytesPerToken() const { return size_t{L} * 2u * Hkv * D; }
	size_t PageBytes() const { return BytesPerToken() * static_cast<size_t>(B); }
	int64_t PrefixPages(int64_t len) const { return (len + B - 1) / B; }  // a frozen prefix: ceil(len/B)
	int64_t SharedPages(int64_t origin) const { return origin / B; }      // floor(origin/B)
	// §3.7: E = min(floor(origin/B), ceil(cap/B) - R(budget)).
	int64_t Materialized(int64_t origin, int64_t budget) const {
		const int64_t a = origin / B, b = CapPages() - R(budget);
		return a < b ? a : b;
	}
	int64_t Limit(int64_t origin, int64_t budget) const { return origin + budget < cap ? origin + budget : cap; }
	// pkv_budget_c_helpers.h's PersonaChainPages: the world's ceil(1000/B) (less its tail page once
	// released) plus the persona's own pages from the world's tail page on. 75 at B = 16, world released.
	int64_t PersonaChainPages(int64_t persona_len, bool world_released, int64_t world_len = 1000) const {
		const int64_t world = PrefixPages(world_len);
		const int64_t own = PrefixPages(world_len + persona_len) - SharedPages(world_len);
		return (world_released ? SharedPages(world_len) : world) + own;
	}
	// Byte offset of (layer, half, head, pos % B, 0) in a page: §3.1's layout [layer][K|V][head][pos][d].
	size_t InPage(uint32_t layer, uint32_t half, uint32_t head, int64_t pos) const {
		return ((size_t{layer} * 2 + half) * Hkv + head) * static_cast<size_t>(B) * D + static_cast<size_t>(pos % B) * D;
	}
};

inline Geo GeoOf(const superslm::SslmModelView& v) {
	Geo g;
	g.L = v.config.num_hidden_layers;
	g.Hkv = v.config.num_key_value_heads;
	g.D = v.config.head_dim;
	g.cap = static_cast<int64_t>(v.config.context_cap);
	g.kvbs = static_cast<int64_t>(v.config.kv_block_size);
	g.B = Geo::PageB(g.kvbs, g.cap);
	g.hidden = v.config.hidden_size;
	return g;
}

// ---- the environment: one context, the fixture mapped on it --------------------------------------

struct Env {
	SslmGpuContext* ctx = nullptr;
	GpuModelFixture fx;
	Geo geo;
	uint32_t full_budget = 0;  // one whole token per decode call
	bool ok = false;
	// `pkv_def_geometry`: the cells' page counts are written for pkv_def's (B, cap) = (16, 4096), so Open refuses any
	// other geometry by default. The G6 timing harness's 1.5B row (plan §10 R1, rev 16.16) runs the shipped 1.5B at
	// its own cap and passes false: it reads the geometry and counts no pages against a plan figure.
	bool Open(const std::string& path, bool pkv_def_geometry = true) {
		if (path.empty()) return false;
		const GS cs = sslm_gpu_context_create(GpuContextConfig{}, &ctx);
		CHECK_MSG(cs == GS::SSLM_OK && ctx, "sslm_gpu_context_create: %s", StatusName(cs));
		if (cs != GS::SSLM_OK || !ctx) return false;
		if (!fx.Open(path, ctx)) {
			CHECK_MSG(false, "fixture %s did not open on the GPU", path.c_str());
			return false;
		}
		geo = GeoOf(fx.view);
		full_budget = fx.one_layer_budget * fx.layers;
		if (!pkv_def_geometry) {
			CHECK_MSG(geo.B > 0 && geo.cap > 0, "model geometry: B = %lld, cap = %lld", static_cast<long long>(geo.B),
			          static_cast<long long>(geo.cap));
			ok = geo.B > 0 && geo.cap > 0;
			return ok;
		}
		CHECK_MSG(geo.B == 16 && geo.cap == 4096,
		          "fixture geometry: B = %lld, cap = %lld; the cells' page counts are written for pkv_def (16, 4096)",
		          static_cast<long long>(geo.B), static_cast<long long>(geo.cap));
		ok = geo.B == 16 && geo.cap == 4096;
		return ok;
	}
	SslmGpuModelHandle* model() const { return fx.model; }
	~Env() {
		fx.Close();
		if (ctx) sslm_gpu_context_destroy(ctx);
	}
};

// The value an out-handle is preset to before a verb whose refusal a cell checks by a null out:
// non-null, so a verb that never writes *out leaves it standing and the check fails. Never released.
template <class H>
inline H* OutSentinel() {
	return reinterpret_cast<H*>(uintptr_t{0x1});
}
template <class H>
inline bool Real(H* h) {
	return h != nullptr && h != OutSentinel<H>();
}

// ---- a GPU sequence and its test-side token state ------------------------------------------------
//
// The GPU decode bridge needs the token to embed, which the caller tracks (te441_gpu_cells.cpp's
// `last`). `ready`: a prefill or an adopt of a non-empty prefix left ready_for_logits set, so the next
// step finishes without an embed. `mid`: the sequence rests between layers (EnterMidToken), so the
// next step finishes the pending token instead of embedding. `primed`: there is something to decode
// from (CPU's "a holder with nothing to decode from" is !primed).
struct GSeq {
	SslmGpuSequenceHandle* h = nullptr;
	int32_t last = -1;
	bool ready = false;
	bool mid = false;
	bool primed = false;
};

inline SslmGpuStatus Drain(const Env& e, SslmGpuSequenceHandle* h) { return ::Drain(e.ctx, h); }

// One whole greedy token. Returns the token (or the -2 schema dead end), or -1 with the status in *st.
inline int32_t Token(const Env& e, GSeq& s, GS* st = nullptr) {
	int32_t tok = -1;
	GS r = GS::SSLM_OK;
	if (s.mid) {
		// The pending token: drive the remaining layers, drain, finish.
		r = sslm_decode_step_gpu(e.ctx, s.h, nullptr, e.full_budget);
		if (r == GS::SSLM_OK) r = Drain(e, s.h);
		if (r == GS::SSLM_OK) r = SslmGpuSeqFinishTokenForG5Bridge(e.ctx, s.h, &tok);
	} else {
		r = SslmGpuSeqDecodeStepForG5Bridge(e.ctx, s.h, s.last, e.full_budget, &tok);
	}
	if (st) *st = r;
	if (r != GS::SSLM_OK) return -1;
	if (tok >= 0) {
		s.last = tok;
		s.ready = s.mid = false;
		s.primed = true;
	}
	return tok;
}

// Up to `n` greedy tokens; stops at the first non-OK status (returned in *last) or a dead end.
inline std::vector<int32_t> DecodeN(const Env& e, GSeq& s, int64_t n, GS* last = nullptr) {
	std::vector<int32_t> out;
	GS st = GS::SSLM_OK;
	for (int64_t i = 0; i < n; ++i) {
		const int32_t t = Token(e, s, &st);
		if (st != GS::SSLM_OK || t < 0) {
			if (st == GS::SSLM_OK && t == -2) out.push_back(-2);
			break;
		}
		out.push_back(t);
	}
	if (last) *last = st;
	return out;
}
// Exactly `n` greedy steps, one outcome per step: the token, the schema dead end's -2, or -1000 - status
// for a refused step. Neither a dead end nor a refusal stops the run, so a retry-stable outcome shows as
// a repeat. The 8.16 outcome form (G4 cells fix; tests/paged-kv/c5_dim8_composition.cpp 8.3 CensusGreedy),
// for constructions whose state has no real continuation on this fixture.
inline std::vector<int32_t> StepOutcomes(const Env& e, GSeq& s, int64_t n) {
	std::vector<int32_t> out;
	for (int64_t i = 0; i < n; ++i) {
		GS st = GS::SSLM_OK;
		const int32_t t = Token(e, s, &st);
		out.push_back(st == GS::SSLM_OK ? t : -1000 - static_cast<int32_t>(st));
	}
	return out;
}
inline std::vector<int32_t> RunToRefusal(const Env& e, GSeq& s, GS* refusal) { return DecodeN(e, s, 1 << 20, refusal); }

inline GS Prefill(const Env& e, GSeq& s, const std::vector<int32_t>& t) {
	if (t.empty()) return GS::SSLM_OK;
	const GS st = SslmGpuSeqPrefillPromptForG5Bridge(e.ctx, s.h, t.data(), static_cast<int32_t>(t.size()), e.full_budget);
	if (st == GS::SSLM_OK) {
		s.last = t.back();
		s.ready = s.primed = true;
		s.mid = false;
	}
	return st;
}

// One layer of the next token: the sequence rests mid-token (0 < layer_index < L). A ready sequence
// first emits its pending token, as the CPU's legacy EnterMidToken does (pkv_legacy_b_helpers.h).
inline bool EnterMidToken(const Env& e, GSeq& s, std::vector<int32_t>* emitted = nullptr) {
	if (s.ready) {
		GS st = GS::SSLM_OK;
		const int32_t t = Token(e, s, &st);
		if (emitted && t >= 0) emitted->push_back(t);
		if (st != GS::SSLM_OK || t < 0) {
			CHECK_MSG(false, "entering mid-token: the pending token: %s, token %d", StatusName(st), t);
			return false;
		}
	}
	GS st = sslm_gpu_seq_embed_token(e.ctx, s.h, s.last);
	if (st == GS::SSLM_OK) st = sslm_decode_step_gpu(e.ctx, s.h, nullptr, e.fx.one_layer_budget);
	if (st == GS::SSLM_OK) st = Drain(e, s.h);
	const uint32_t li = LayerIndex(s.h);
	CHECK_MSG(st == GS::SSLM_OK && li > 0 && li < e.fx.layers, "entering mid-token: %s, layer_index %u of %u",
	          StatusName(st), li, e.fx.layers);
	s.mid = st == GS::SSLM_OK;
	return s.mid;
}

// The "next turn" of §3.9's keeps rows (pkv_budget_c_helpers.h's NextTurn): greedy to the first
// refusal; a holder with nothing to decode from first takes a one-token user turn.
struct Turn {
	std::vector<int32_t> tokens;
	GS prefill = GS::SSLM_OK;
	GS refusal = GS::SSLM_OK;
	bool operator==(const Turn& o) const { return tokens == o.tokens && prefill == o.prefill && refusal == o.refusal; }
};
inline Turn NextTurn(const Env& e, GSeq& s) {
	Turn t;
	if (!s.primed) t.prefill = Prefill(e, s, Stream(82, 1, kVocab));
	t.tokens = RunToRefusal(e, s, &t.refusal);
	return t;
}

inline std::vector<uint8_t> Save(const Env& e, SslmGpuSequenceHandle* h) {
	size_t need = 0;
	(void)sslm_gpu_seq_save(e.ctx, h, nullptr, &need);
	std::vector<uint8_t> blob(need);
	size_t wrote = blob.size();
	const GS st = need ? sslm_gpu_seq_save(e.ctx, h, blob.data(), &wrote) : GS::SSLM_DEVICE_LOST;
	CHECK_MSG(st == GS::SSLM_OK, "sslm_gpu_seq_save: %s (size %zu)", StatusName(st), need);
	if (st != GS::SSLM_OK) return {};
	blob.resize(wrote);
	return blob;
}

inline bool IsMagic(const std::vector<uint8_t>& b, const char* m) { return b.size() >= 4 && std::memcmp(b.data(), m, 4) == 0; }

// ---- the GPU blob formats --------------------------------------------------------------------------
//
// 'SLM5' and 'SLM4' are the shipped layouts (src/gpu/superslm_gpu.cpp, GpuSeqBlobHeader): a 120-byte
// header (magic, layer_index u32, scale m i64, scale e i64, kv_saturation u64, context_length i64,
// hidden_codes_size u64, workspace_size u64, model hash 32, the four per-site counts 4 x u64), then
// for 'SLM5' a 12-byte tail (bound_schema_index i32, dfa_walk_state u32, ready_for_logits u32), then
// the residual, then the whole dedicated workspace. 'SLM4' is the same without the tail.
constexpr size_t kSlmHeader = 120;
constexpr size_t kSlm5Tail = 12;

inline uint32_t Le32(const std::vector<uint8_t>& b, size_t o) {
	uint32_t v = 0;
	if (o + 4 <= b.size()) std::memcpy(&v, b.data() + o, 4);
	return v;
}
inline uint64_t Le64(const std::vector<uint8_t>& b, size_t o) {
	uint64_t v = 0;
	if (o + 8 <= b.size()) std::memcpy(&v, b.data() + o, 8);
	return v;
}
inline void Put32(std::vector<uint8_t>& b, size_t o, uint32_t v) {
	if (o + 4 <= b.size()) std::memcpy(b.data() + o, &v, 4);
}
inline void Put64(std::vector<uint8_t>& b, size_t o, uint64_t v) {
	if (o + 8 <= b.size()) std::memcpy(b.data() + o, &v, 8);
}

// An 'SLM4' blob from an 'SLM5' one: the magic, and the tail removed. That is SLM4's layout exactly
// (superslm_gpu.cpp: "a legacy 'SLM4' blob (which has no tail) is still exactly
// sizeof(GpuSeqBlobHeader)"), the construction t2899's cell_gpu_slm4_dump made by building the
// pre-SLM5 save path; here it is made from the bytes, so no second engine build is needed.
inline std::vector<uint8_t> Slm4FromSlm5(const std::vector<uint8_t>& slm5) {
	if (!IsMagic(slm5, "SLM5") || slm5.size() < kSlmHeader + kSlm5Tail) return {};
	std::vector<uint8_t> out(slm5.begin(), slm5.begin() + kSlmHeader);
	std::memcpy(out.data(), "SLM4", 4);
	out.insert(out.end(), slm5.begin() + kSlmHeader + kSlm5Tail, slm5.end());
	return out;
}

// 'SLM6' (§4 item 10): "SLM5 fields, plus kv_mode, budget, origin and the canonical [0, L') K/V
// gather of §3.7". The plan fixes the fields, not the byte offsets.
//
// PROVISIONAL LAYOUT (G5 fixes it). pkvg unification: the offsets are the suite's one SLM6 pin,
// pkvg_slm6_layout.h (shared with g1b): SLM5's 120-byte header with the magic 'SLM6' (and, this group's
// reading, workspace_size = L' * bytes_per_token); SLM5's 12-byte tail; kv_mode u32, budget i32, origin
// i64 (16 bytes, SSB6's order and widths, §3.7); the residual; then §3.7's K/V section: kv_positions u64
// = L', and the K/V rows [0, L') in canonical layout [layer][K|V][head][pos < L'][d] (SSB6's, "the
// one-page layout at B = L'"). Slm6Parse checks L' against context_length and layer_index (§3.7's
// validation). (g1e's draft had no kv_positions field; §3.7's K/V section and 2.6 [G5]'s hostile
// "kv_positions inconsistent" case both need it, so the unified pin follows g1b and the plan.)
//
// Every cell reads SLM6 fields only through Slm6Parse, and every cell that saves an SLM6 first runs
// Slm6SelfCheck against the budget and origin it declared, so a G5 layout that differs from this one
// turns the cells red with "SLM6 layout" in the message, never green. G5 updates the constants and
// Slm6Parse here, never an assertion.
constexpr size_t kSlm6KvModeOff = pkvg_slm6::kOffKvMode;  // 132
constexpr size_t kSlm6BudgetOff = pkvg_slm6::kOffBudget;  // 136
constexpr size_t kSlm6OriginOff = pkvg_slm6::kOffOrigin;  // 140
constexpr size_t kSlm6Header = pkvg_slm6::kFixedBytes;    // 148
static_assert(kSlmHeader == pkvg_slm6::kSlm5HeaderBytes && kSlm5Tail == pkvg_slm6::kSlm5TailBytes, "SLM5 pin");

struct Slm6View {
	bool ok = false;
	uint32_t layer_index = 0;
	int64_t context_length = 0;
	uint32_t kv_mode = 0;
	int32_t budget = 0;
	int64_t origin = 0;
	int32_t bound_schema_index = -1;
	uint32_t dfa_walk_state = 0;
	uint32_t ready = 0;
	int64_t Lp = 0;      // L'
	size_t kv_off = 0;   // the first K/V byte
};
inline Slm6View Slm6Parse(const std::vector<uint8_t>& b, const Geo& g) {
	Slm6View v;
	if (!IsMagic(b, "SLM6") || b.size() < kSlm6Header) return v;
	v.layer_index = Le32(b, 4);
	v.context_length = static_cast<int64_t>(Le64(b, 32));
	const uint64_t hidden = Le64(b, 40), kv = Le64(b, 48);
	v.bound_schema_index = static_cast<int32_t>(Le32(b, 120));
	v.dfa_walk_state = Le32(b, 124);
	v.ready = Le32(b, 128);
	v.kv_mode = Le32(b, kSlm6KvModeOff);
	v.budget = static_cast<int32_t>(Le32(b, kSlm6BudgetOff));
	v.origin = static_cast<int64_t>(Le64(b, kSlm6OriginOff));
	// L' (§3.7 as the CPU twin applies it, the G5 review ruling): row context_length only mid-token; the
	// GPU's rest at layer_index == L is a completed token.
	v.Lp = v.context_length + ((v.layer_index > 0 && v.layer_index < g.L) ? 1 : 0);
	const size_t kvpos_off = pkvg_slm6::OffKvPositions(hidden);
	if (b.size() < kvpos_off + pkvg_slm6::kKvPositionsBytes) return v;
	const uint64_t kv_positions = Le64(b, kvpos_off);
	v.kv_off = pkvg_slm6::OffRows(hidden);
	v.ok = hidden == g.hidden && kv_positions == static_cast<uint64_t>(v.Lp) &&
	       kv == static_cast<uint64_t>(v.Lp) * g.BytesPerToken() && b.size() == v.kv_off + kv;
	return v;
}
// The canonical row (layer, half, head, pos) of an SLM6 blob.
inline size_t Slm6RowOff(const Slm6View& v, const Geo& g, uint32_t layer, uint32_t half, uint32_t head, int64_t pos) {
	return v.kv_off + ((size_t{layer} * 2 + half) * g.Hkv + head) * static_cast<size_t>(v.Lp) * g.D +
	       static_cast<size_t>(pos) * g.D;
}
inline bool Slm6SelfCheck(const std::vector<uint8_t>& b, const Geo& g, int32_t budget, int64_t origin, int64_t ctxlen,
                          const char* what) {
	const Slm6View v = Slm6Parse(b, g);
	const bool good = v.ok && v.kv_mode == 1 && v.budget == budget && v.origin == origin && v.context_length == ctxlen;
	CHECK_MSG(good,
	          "%s: SLM6 layout: magic %.4s, parse %d, kv_mode %u, budget %d (want %d), origin %lld (want %lld), "
	          "context_length %lld (want %lld) -- a pooled save must be a budget SLM6 (§4 item 10); if the layout "
	          "moved, update g1e_helpers.h's SLM6 constants, never this assertion",
	          what, b.size() >= 4 ? reinterpret_cast<const char*>(b.data()) : "????", v.ok ? 1 : 0, v.kv_mode, v.budget,
	          budget, static_cast<long long>(v.origin), static_cast<long long>(origin),
	          static_cast<long long>(v.context_length), static_cast<long long>(ctxlen));
	return good;
}

// ---- pools, prefixes and the persona chain --------------------------------------------------------

struct GChain {
	SslmGpuPrefixHandle* world = nullptr;
	SslmGpuPrefixHandle* persona = nullptr;
	bool ok = false;
};

// A state the fill probe can rebuild: a pool and the handles made in it, released before the pool.
struct ProbeState {
	virtual ~ProbeState() = default;
	virtual SslmGpuKvPoolHandle* Pool() = 0;
	virtual SslmGpuContext* Ctx() = 0;
	virtual SslmGpuModelHandle* Model() = 0;
	virtual void DrainAll() = 0;
};

struct Rig : ProbeState {
	const Env& e;
	SslmGpuContext* ctx;
	SslmGpuModelHandle* model;
	SslmGpuKvPoolHandle* pool = nullptr;
	GS status = GS::SSLM_DEVICE_LOST;
	std::vector<SslmGpuSequenceHandle*> seqs;
	std::vector<SslmGpuPrefixHandle*> prefixes;
	GChain chain;
	Rig(const Env& env, uint32_t pages, SslmGpuContext* c = nullptr, SslmGpuModelHandle* m = nullptr)
	    : e(env), ctx(c ? c : env.ctx), model(m ? m : env.model()) {
		status = sslm_gpu_kv_pool_create(ctx, model, pages, &pool);
		CHECK_MSG(status == GS::SSLM_OK && pool, "sslm_gpu_kv_pool_create(%u pages): %s", pages, StatusName(status));
	}
	~Rig() override {
		for (auto it = seqs.rbegin(); it != seqs.rend(); ++it) sslm_gpu_seq_release(ctx, *it);
		ReleaseChain();
		for (auto it = prefixes.rbegin(); it != prefixes.rend(); ++it) sslm_gpu_prefix_release(ctx, *it);
		if (pool) sslm_gpu_kv_pool_destroy(ctx, pool);
	}
	SslmGpuKvPoolHandle* Pool() override { return pool; }
	SslmGpuContext* Ctx() override { return ctx; }
	SslmGpuModelHandle* Model() override { return model; }
	void DrainAll() override {
		for (SslmGpuSequenceHandle* s : seqs) (void)::Drain(ctx, s);
	}
	SslmGpuSequenceHandle* Seq(SslmGpuSequenceHandle* s) {
		if (Real(s)) seqs.push_back(s);
		return Real(s) ? s : nullptr;
	}
	SslmGpuPrefixHandle* Prefix(SslmGpuPrefixHandle* p) {
		if (Real(p)) prefixes.push_back(p);
		return Real(p) ? p : nullptr;
	}
	void ReleaseSeq(SslmGpuSequenceHandle* s) {
		for (auto it = seqs.begin(); it != seqs.end(); ++it)
			if (*it == s) {
				seqs.erase(it);
				break;
			}
		sslm_gpu_seq_release(ctx, s);
	}
	void ReleaseChain() {
		if (chain.persona) sslm_gpu_prefix_release(ctx, chain.persona);
		if (chain.world) sslm_gpu_prefix_release(ctx, chain.world);
		chain = GChain{};
	}
	// A budgeted pooled sequence; null (and a failed check) on refusal.
	GSeq Budgeted(int32_t budget) {
		GSeq s;
		SslmGpuSequenceHandle* h = nullptr;
		const GS st = sslm_gpu_seq_create_budgeted(ctx, model, pool, budget, &h);
		CHECK_MSG(st == GS::SSLM_OK && h, "sslm_gpu_seq_create_budgeted(%d): %s", budget, StatusName(st));
		s.h = Seq(h);
		return s;
	}
};

inline GS PrefixPrefillAll(SslmGpuContext* ctx, SslmGpuPrefixHandle* p, const std::vector<int32_t>& t, int32_t kind,
                           uint32_t dispatch_budget) {
	size_t at = 0;
	GS st = GS::SSLM_OK;
	while (at < t.size()) {
		int32_t consumed = 0;
		st = sslm_gpu_prefix_prefill(ctx, p, t.data() + at, static_cast<int32_t>(t.size() - at), kind, dispatch_budget,
		                             &consumed);
		at += static_cast<size_t>(consumed > 0 ? consumed : 0);
		if (st != GS::SSLM_OK || consumed <= 0) break;
	}
	return st;
}

// A budgeted prefix of `tokens` (budget = max(len, 1)), frozen unless `freeze` is false.
inline SslmGpuPrefixHandle* BudgetPrefix(Rig& rig, const std::vector<int32_t>& tokens, bool freeze = true) {
	SslmGpuPrefixHandle* p = nullptr;
	const int32_t budget = tokens.empty() ? 1 : static_cast<int32_t>(tokens.size());
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, budget, &p);
	if (st == GS::SSLM_OK && !tokens.empty()) st = PrefixPrefillAll(rig.ctx, p, tokens, 0, rig.e.full_budget);
	if (st == GS::SSLM_OK && freeze) st = sslm_gpu_prefix_freeze(rig.ctx, p);
	CHECK_MSG(st == GS::SSLM_OK, "GPU budget prefix of %zu tokens: %s", tokens.size(), StatusName(st));
	if (st != GS::SSLM_OK) {
		if (Real(p)) sslm_gpu_prefix_release(rig.ctx, p);
		return nullptr;
	}
	return rig.Prefix(p);
}

// The persona chain (pkv_budget_c_helpers.h's BuildPersonaChain, through the GPU verbs): a budget
// world of 1,000 frozen, then a persona by begin_from(world, 200) frozen; with the world released the
// chain holds 62 + 13 = 75 pages (§7 9.9).
inline GChain BuildPersonaChain(Rig& rig, int persona_index, bool release_world,
                                const std::vector<int32_t>* persona_tokens = nullptr, bool freeze_persona = true) {
	GChain c;
	const uint32_t db = rig.e.full_budget;
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, 1000, &c.world);
	if (st == GS::SSLM_OK) st = PrefixPrefillAll(rig.ctx, c.world, WorldTokens(), 0, db);
	if (st == GS::SSLM_OK) st = sslm_gpu_prefix_freeze(rig.ctx, c.world);
	const std::vector<int32_t> pt = persona_tokens ? *persona_tokens : PersonaTokens(persona_index);
	if (st == GS::SSLM_OK) st = sslm_gpu_prefix_begin_from(rig.ctx, c.world, static_cast<int32_t>(pt.size()), &c.persona);
	if (st == GS::SSLM_OK) st = PrefixPrefillAll(rig.ctx, c.persona, pt, 0, db);
	if (st == GS::SSLM_OK && freeze_persona) st = sslm_gpu_prefix_freeze(rig.ctx, c.persona);
	if (st == GS::SSLM_OK && release_world) {
		st = sslm_gpu_prefix_release(rig.ctx, c.world);
		c.world = nullptr;
	}
	CHECK_MSG(st == GS::SSLM_OK, "GPU persona chain %d: %s", persona_index, StatusName(st));
	if (!Real(c.world)) c.world = nullptr;
	if (!Real(c.persona)) c.persona = nullptr;
	c.ok = st == GS::SSLM_OK;
	rig.chain = c;
	return c;
}

// Adopts `p` onto the pooled `s`; the test-side state follows §4 item 7 (ready_for_logits copied from
// a prefix that holds content).
inline GS Adopt(Rig& rig, GSeq& s, SslmGpuPrefixHandle* p, int64_t prefix_len) {
	const GS st = sslm_gpu_seq_adopt_prefix(rig.ctx, s.h, p);
	if (st == GS::SSLM_OK) {
		s.ready = s.primed = prefix_len > 0;
		s.mid = false;
	}
	return st;
}

// restore-into-pool, with the out-handle preset to the sentinel. The restored sequence inherits the
// original's test-side token state (`like`), which the blob carries on the device side.
struct Restored {
	GS status = GS::SSLM_DEVICE_LOST;
	GSeq seq;
	SslmGpuSequenceHandle* raw_out = nullptr;  // what the verb left in *out (the sentinel if untouched)
};
inline Restored RestoreIntoPool(Rig& rig, const std::vector<uint8_t>& blob, SslmGpuPrefixHandle* prefix, const GSeq& like,
                                uint32_t* shared_diag = nullptr) {
	Restored r;
	SslmGpuSequenceHandle* out = OutSentinel<SslmGpuSequenceHandle>();
	r.status = sslm_gpu_seq_restore_into_pool(rig.ctx, rig.model, rig.pool, blob.data(), blob.size(), prefix, &out,
	                                          shared_diag);
	r.raw_out = out;
	r.seq = like;
	r.seq.h = rig.Seq(out);
	return r;
}

// ---- the GPU fill probe (§8; GPU form through sslm_gpu_seq_create_budgeted, read after sslm_gpu_ready) ----

inline std::vector<int64_t> ProbeSplit(const Geo& g, int64_t k) {
	std::vector<int64_t> parts;
	const int64_t M = g.CapPages();
	CHECK_MSG(M >= 2 && k >= 2, "fill probe: k = %lld on a %lld-page cap cannot be probed", static_cast<long long>(k),
	          static_cast<long long>(M));
	if (M < 2 || k < 2) return parts;
	while (k > 0) {
		int64_t r = k < M ? k : M;
		if (k - r == 1) r -= 1;  // never leave a 1-page remainder
		parts.push_back(r);
		k -= r;
	}
	return parts;
}

struct Refusal {
	int64_t pages = 0;
	GS status = GS::SSLM_OK;
	bool Any() const { return status != GS::SSLM_OK; }
	bool Capacity() const { return status == GS::SSLM_GPU_KV_POOL_EXHAUSTED; }
};

// Budgeted creates totalling `k` pages (a create of r pages declares (r - 1) * B, R = r). The admitted
// sequences are appended to `out`. A refusal for another reason than capacity is preferred in `refusal`.
inline bool AdmitPages(const Geo& g, ProbeState& s, int64_t k, std::vector<SslmGpuSequenceHandle*>* out,
                       Refusal* refusal = nullptr) {
	bool all = true;
	Refusal first;
	for (int64_t r : ProbeSplit(g, k)) {
		SslmGpuSequenceHandle* h = nullptr;
		const GS st = sslm_gpu_seq_create_budgeted(s.Ctx(), s.Model(), s.Pool(), static_cast<int32_t>((r - 1) * g.B), &h);
		if (st == GS::SSLM_OK && h) {
			out->push_back(h);
			continue;
		}
		all = false;
		if (!first.Any() || (first.Capacity() && st != GS::SSLM_GPU_KV_POOL_EXHAUSTED)) first = Refusal{r, st};
	}
	if (refusal) *refusal = first;
	return all;
}

// Exactly `k` pages free in the state `build` makes: creates totalling k all admitted on one build,
// creates totalling k + 1 refused for capacity on a fresh rebuild. Both legs drain every tracked
// sequence first (sslm_gpu_ready), as §8's GPU form reads the pool.
inline bool ProbeExactlyFree(const Env& e, const std::function<std::unique_ptr<ProbeState>()>& build, int64_t k,
                             const char* what) {
	bool admits_k = false, refuses_k1 = false, all_k1 = false;
	Refusal ra, rb;
	{
		std::unique_ptr<ProbeState> s = build();
		std::vector<SslmGpuSequenceHandle*> made;
		if (s) {
			s->DrainAll();
			admits_k = AdmitPages(e.geo, *s, k, &made, &ra);
			for (auto it = made.rbegin(); it != made.rend(); ++it) sslm_gpu_seq_release(s->Ctx(), *it);
		}
	}
	{
		std::unique_ptr<ProbeState> s = build();
		std::vector<SslmGpuSequenceHandle*> made;
		if (s) {
			s->DrainAll();
			all_k1 = AdmitPages(e.geo, *s, k + 1, &made, &rb);
			refuses_k1 = !all_k1 && rb.Capacity();
			for (auto it = made.rbegin(); it != made.rend(); ++it) sslm_gpu_seq_release(s->Ctx(), *it);
		}
	}
	CHECK_MSG(admits_k, "%s: fill probe: creates totalling %lld pages were not all admitted (a %lld-page create: %s)", what,
	          static_cast<long long>(k), static_cast<long long>(ra.pages), StatusName(ra.status));
	CHECK_MSG(refuses_k1,
	          "%s: fill probe: creates totalling %lld pages were %s (a %lld-page create: %s): at most %lld free is not shown",
	          what, static_cast<long long>(k + 1), all_k1 ? "all admitted" : "refused for a reason other than capacity",
	          static_cast<long long>(rb.pages), StatusName(rb.status), static_cast<long long>(k));
	return admits_k && refuses_k1;
}

// A 2-page create, admitted or not, released again. A refusal other than capacity is a failed check.
inline bool TwoPageCreateAdmitted(const Env& e, ProbeState& s) {
	SslmGpuSequenceHandle* h = nullptr;
	s.DrainAll();
	const GS st = sslm_gpu_seq_create_budgeted(s.Ctx(), s.Model(), s.Pool(), static_cast<int32_t>(e.geo.B), &h);
	if (h) sslm_gpu_seq_release(s.Ctx(), h);
	CHECK_MSG(st == GS::SSLM_OK || st == GS::SSLM_GPU_KV_POOL_EXHAUSTED,
	          "a 2-page create was refused with %s, not SSLM_GPU_KV_POOL_EXHAUSTED: it says nothing about free pages",
	          StatusName(st));
	return st == GS::SSLM_OK;
}
// Creates totalling exactly k pages admitted (and released again): the admission leg alone.
inline bool AdmitsPages(const Env& e, ProbeState& s, int64_t k) {
	std::vector<SslmGpuSequenceHandle*> made;
	s.DrainAll();
	const bool all = AdmitPages(e.geo, s, k, &made);
	for (auto it = made.rbegin(); it != made.rend(); ++it) sslm_gpu_seq_release(s.Ctx(), *it);
	return all;
}

// The GPU fill probe's one-state form (§8, "Precondition, and the one-state form"; the CPU's
// ProbeExactlyFreeOneState), for k >= 3, on the one state `s` holds: creates totalling k - 1 are admitted (at least
// k - 1 free); a 2-page create is refused for capacity (at most k free); one admitted create of r pages is released
// and creates totalling r + 1 are admitted (at least k were free). Every create is released again.
inline bool ProbeExactlyFreeOneState(const Env& e, ProbeState& s, int64_t k, const char* what) {
	CHECK_MSG(k >= 3, "%s: the one-state probe needs k >= 3 (k = %lld)", what, static_cast<long long>(k));
	if (k < 3) return false;
	s.DrainAll();
	std::vector<SslmGpuSequenceHandle*> made;
	Refusal first;
	const bool step1 = AdmitPages(e.geo, s, k - 1, &made, &first);
	CHECK_MSG(step1, "%s: one-state fill probe at k=%lld: creates totalling %lld pages were not all admitted (a %lld-page create: %s)",
	          what, static_cast<long long>(k), static_cast<long long>(k - 1), static_cast<long long>(first.pages),
	          StatusName(first.status));
	SslmGpuSequenceHandle* two = nullptr;
	const GS st2 = sslm_gpu_seq_create_budgeted(s.Ctx(), s.Model(), s.Pool(), static_cast<int32_t>(e.geo.B), &two);
	if (two) sslm_gpu_seq_release(s.Ctx(), two);
	const bool step2 = st2 == GS::SSLM_GPU_KV_POOL_EXHAUSTED;
	CHECK_MSG(step2, "%s: one-state fill probe at k=%lld: after %lld pages, a 2-page create returned %s, not exhausted", what,
	          static_cast<long long>(k), static_cast<long long>(k - 1), StatusName(st2));
	bool step3 = false;
	if (step1 && !made.empty()) {
		const int64_t r = ProbeSplit(e.geo, k - 1).back();
		sslm_gpu_seq_release(s.Ctx(), made.back());
		made.pop_back();
		step3 = AdmitPages(e.geo, s, r + 1, &made, &first);
		CHECK_MSG(step3, "%s: one-state fill probe at k=%lld: after releasing a %lld-page create, creates totalling %lld were not all "
		                 "admitted (%s)",
		          what, static_cast<long long>(k), static_cast<long long>(r), static_cast<long long>(r + 1), StatusName(first.status));
	}
	for (auto it = made.rbegin(); it != made.rend(); ++it) sslm_gpu_seq_release(s.Ctx(), *it);
	return step1 && step2 && step3;
}

// ---- the mirror peek, addressed by §3.1's formula ---------------------------------------------------

inline std::vector<uint8_t> PeekPage(const Env& e, SslmGpuKvPoolHandle* pool, uint32_t page) {
	std::vector<uint8_t> b(e.geo.PageBytes(), 0xEE);
	const GS st = sslm_gpu_test_only_peek_pool_mirror_bytes(pool, page, 0, b.size(), b.data());
	CHECK_MSG(st == GS::SSLM_OK, "mirror peek of page %u: %s", page, StatusName(st));
	return b;
}

// A prefix's pages as the blob says they must read: page j holds the rows [jB, min((j+1)B, P)) of the
// blob (every layer, K and V, every head) at §3.1's in-page offsets, and zero bytes past P (a pool is
// zero-filled at create, and a prefix built first in a fresh pool writes only its own rows).
inline std::vector<std::vector<uint8_t>> ExpectedPrefixPages(const Env& e, const std::vector<uint8_t>& slm6, int64_t P) {
	const Geo& g = e.geo;
	const Slm6View v = Slm6Parse(slm6, g);
	std::vector<std::vector<uint8_t>> pages;
	if (!v.ok || v.Lp < P) return pages;
	for (int64_t j = 0; j < g.PrefixPages(P); ++j) {
		std::vector<uint8_t> page(g.PageBytes(), 0);
		for (int64_t pos = j * g.B; pos < (j + 1) * g.B && pos < P; ++pos)
			for (uint32_t l = 0; l < g.L; ++l)
				for (uint32_t half = 0; half < 2; ++half)
					for (uint32_t h = 0; h < g.Hkv; ++h)
						std::memcpy(page.data() + g.InPage(l, half, h, pos), slm6.data() + Slm6RowOff(v, g, l, half, h, pos), g.D);
		pages.push_back(std::move(page));
	}
	return pages;
}

// The GPU twin of the CPU's SnapPrefix (the GPU has no table seam): the pool page holding each
// expected prefix page, found by scanning the mirror. A page not found is reported as -1.
struct PrefixSnap {
	std::vector<int64_t> index;
	std::vector<std::vector<uint8_t>> bytes;
	bool complete = false;
};
inline PrefixSnap LocatePrefixPages(const Env& e, SslmGpuKvPoolHandle* pool, uint32_t pool_pages,
                                    const std::vector<std::vector<uint8_t>>& expected) {
	PrefixSnap s;
	s.complete = true;
	std::vector<std::vector<uint8_t>> mirror;
	for (uint32_t p = 0; p < pool_pages; ++p) mirror.push_back(PeekPage(e, pool, p));
	for (const auto& want : expected) {
		int64_t at = -1;
		for (uint32_t p = 0; p < pool_pages && at < 0; ++p)
			if (mirror[p] == want) at = p;
		s.index.push_back(at);
		s.bytes.push_back(want);
		if (at < 0) s.complete = false;
	}
	return s;
}
// The pages found before still read as they did.
inline bool PrefixPagesUnchanged(const Env& e, SslmGpuKvPoolHandle* pool, const PrefixSnap& before) {
	for (size_t i = 0; i < before.index.size(); ++i) {
		if (before.index[i] < 0) return false;
		if (PeekPage(e, pool, static_cast<uint32_t>(before.index[i])) != before.bytes[i]) return false;
	}
	return true;
}

}  // namespace g1e

#endif  // SSLM_PKV_GPU_G1E_HELPERS_H
