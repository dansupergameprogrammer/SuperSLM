// Paged-KV plan (rev 16.2) step G1, group g1a: the shared harness for this group's GPU red cells
// (1.7, 1.8, 1.9, 4.8, 6.5a, 6.5b, 8.13-8.16, 11.7, 11.9 GPU twin), on the box (D3D12, both
// certified GPUs, one process per GPU through SSLM_GPU_ADAPTER_INDEX).
//
// What it carries:
//   - a cell registry (G1A_CELL) and the checks, modelled on tests/paged-kv/pkv_common.h; the
//     coordinator unifies the GPU suite's registries, so the names here are prefixed;
//   - fixtures: tools/gen_paged_kv_fixture.py's pkv_def and pkv_qk (SUPERSLM_PAGED_KV_FIXTURE_DIR,
//     the CPU suite's own cloud fixtures, B = 16 at cap 4096) and, for 4.8, the two real artifacts
//     (--model05= / --model15=); a missing fixture fails the cell, never skips it;
//   - the byte-equality oracle: tests/paged-kv/reference/v1.11.0_<fixture>.ref (step R0) through
//     reference/pkv_scenarios.h's Stream and RowsFromBlock, and an in-process CPU paged twin driven
//     through the 1.12.0 C ABI and read from its saved blob (SSB5 or SSB6; CpuRows says why not the seams);
//   - one scenario script per R0 scenario a cell needs, run on three backends (CPU paged, GPU pooled,
//     GPU dedicated), so the GPU half and its CPU twin are the same calls;
//   - the GPU K/V readers: a dedicated sequence's rows from its 'SLM5' blob (flat layout, B = cap),
//     a pooled sequence's rows from the pool's host mirror through the GPU mirror peek, addressed by
//     §3.1's formula written here;
//   - the GPU fill probe (§8): the two-sided form through sslm_gpu_seq_create_budgeted, read after
//     sslm_gpu_ready has drained every sequence, with no reading from sslm_gpu_kv_pool_stats.
//
// RED BEFORE THE BUILD. The 1.13.0 verbs are declared in include/superslm/gpu_1p0.h with no
// definition (g1-header, 63e7b0e), so every cell that calls one fails to LINK until its step; the
// helpers that call them are inline and emitted only where a cell calls them, so a per-step
// executable links exactly the symbols its own cells use (pkv_common.h's convention).
//
// SEAMS THE PLAN DOES NOT NAME (declared below, link-red, the G3 builder defines or renames them):
//   - sslm_gpu_test_only_seq_table_entry: the GPU twin of sslm_pkv_test_only_seq_table_entry. The
//     GPU mirror peek (§4 item 9) "is addressed with the §3.1 formula", whose operand is the
//     sequence's table; without the table a cell cannot find the page it peeks. CPU precedent: the
//     table-read seam landed with the peek at C4.
//   - sslm_gpu_test_only_arm_view_fault and sslm_gpu_test_only_last_ladder_status: cells 11.7 and
//     11.9's GPU twin need a paged GPU sequence whose view is one page short, one byte short per
//     page, null-based or with an overflowing per-page product. No C ABI call builds one (the host
//     maps every page before it writes, and the view's geometry is the pool's), so a seam arms the
//     fault on the next submission, and a second reads which ladder rung refused it (every rung maps
//     to SSLM_SEQUENCE_REJECTED at the C ABI, MapSubmitRejectionToGpuStatus).

#ifndef SUPERSLM_TESTS_G1A_HELPERS_H
#define SUPERSLM_TESTS_G1A_HELPERS_H

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "reference/pkv_scenarios.h"  // pkv::Stream, pkv::Geometry, pkv::Sha, pkv::RowsFromBlock

#include "superslm/checked_chain_funnel.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "superslm/sslm_abi.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <vector>

// ---- G3 seams this group needs and the plan does not name (see the header comment) ----------------
// Table entry i (< mapped) of a pooled GPU sequence; reading at or past `mapped`, or on a dedicated
// sequence, is SSLM_SEQUENCE_KV_BUFFER_MISMATCH, never a stale read. *out_mapped gets `mapped`.
SslmGpuStatus sslm_gpu_test_only_seq_table_entry(const SslmGpuSequenceHandle* seq, uint32_t i, uint32_t* out_page,
                                                 uint32_t* out_mapped) noexcept;
// Arms one fault on the view `seq`'s NEXT submission hands the GPU ladder (one-shot):
//   1  the table one page short (the view's mapped count is the real one minus 1);
//   2  page_bytes one byte short;
//   3  a null pool base;
//   4  an overflowing per-page product (page_positions = 2^61, the C2 cell's big_B).
// Changes nothing else. A pooled sequence only.
SslmGpuStatus sslm_gpu_test_only_arm_view_fault(SslmGpuSequenceHandle* seq, uint32_t fault) noexcept;
// The superslm::SslmForwardStatus the ladder returned for `seq`'s most recent refused submission,
// as its integer value; superslm::SslmForwardStatus::Ok when the last submission was not refused.
SslmGpuStatus sslm_gpu_test_only_last_ladder_status(const SslmGpuSequenceHandle* seq,
                                                    int32_t* out_forward_status) noexcept;

namespace g1a {

using GS = SslmGpuStatus;

constexpr uint32_t kViewFaultTableOneShort = 1;
constexpr uint32_t kViewFaultPageBytesShort = 2;
constexpr uint32_t kViewFaultNullPoolBase = 3;
constexpr uint32_t kViewFaultProductOverflow = 4;

// ---- the registry and the checks -------------------------------------------------------------

struct CellEntry {
	const char* id;    // the §7 cell id with its part ("1.7/G3")
	const char* step;  // the owning step of this part: G2, G3, G4 or G5
	void (*fn)();
};
inline std::vector<CellEntry>& Registry() {
	static std::vector<CellEntry> r;
	return r;
}
struct Registrar {
	Registrar(const char* id, const char* step, void (*fn)()) { Registry().push_back({id, step, fn}); }
};
#define G1A_CAT2(a, b) a##b
#define G1A_CAT(a, b) G1A_CAT2(a, b)
#define G1A_CELL(id, step, fn) static ::g1a::Registrar G1A_CAT(g1a_registrar_, __LINE__)(id, step, fn)

struct RunState {
	const char* cell = "(none)";
	int checks = 0;
	int failures = 0;
	int cell_failures = 0;
};
inline RunState& State() {
	static RunState s;
	return s;
}
inline void Fail(const char* file, int line, const char* fmt, ...) {
	RunState& s = State();
	++s.failures;
	++s.cell_failures;
	std::fprintf(stderr, "FAIL [%s] %s:%d: ", s.cell, file, line);
	va_list ap;
	va_start(ap, fmt);
	std::vfprintf(stderr, fmt, ap);
	va_end(ap);
	std::fputc('\n', stderr);
}
#define G1A_CHECK(cond)                                                \
	do {                                                               \
		++::g1a::State().checks;                                       \
		if (!(cond)) ::g1a::Fail(__FILE__, __LINE__, "%s", #cond);     \
	} while (0)
#define G1A_CHECK_MSG(cond, ...)                                       \
	do {                                                               \
		++::g1a::State().checks;                                       \
		if (!(cond)) ::g1a::Fail(__FILE__, __LINE__, __VA_ARGS__);     \
	} while (0)
#define G1A_CHECK_EQ(a, b)                                                                                \
	do {                                                                                                  \
		++::g1a::State().checks;                                                                          \
		const long long g1a_a_ = static_cast<long long>(a), g1a_b_ = static_cast<long long>(b);           \
		if (g1a_a_ != g1a_b_) ::g1a::Fail(__FILE__, __LINE__, "%s == %s (%lld vs %lld)", #a, #b, g1a_a_, g1a_b_); \
	} while (0)
#define G1A_CHECK_GPU(expr, want)                                                                         \
	do {                                                                                                  \
		++::g1a::State().checks;                                                                          \
		const ::SslmGpuStatus g1a_s_ = (expr);                                                            \
		if (g1a_s_ != (want))                                                                             \
			::g1a::Fail(__FILE__, __LINE__, "%s -> %s, want %s", #expr, ::g1a::GpuStatusName(g1a_s_),     \
			            ::g1a::GpuStatusName(want));                                                      \
	} while (0)

inline const char* GpuStatusName(SslmGpuStatus s) {
	static const char* const k[] = {"SSLM_OK", "SSLM_DISPATCH_BUDGET_TOO_SMALL", "SSLM_BUSY",
	                                "SSLM_CONTEXT_HAS_LIVE_HANDLES", "SSLM_MODEL_HAS_LIVE_SEQUENCES",
	                                "SSLM_ADAPTER_MODEL_MISMATCH", "SSLM_ADAPTER_BASE_HASH_MISMATCH",
	                                "SSLM_SEQUENCE_KV_BUFFER_MISMATCH", "SSLM_DEVICE_LOST", "SSLM_BATCH_BUDGET_EXHAUSTED",
	                                "SSLM_TOKEN_ID_OUT_OF_RANGE", "SSLM_SEQUENCE_REJECTED", "SSLM_RESTORE_MODEL_MISMATCH",
	                                "SSLM_MODEL_HAS_LIVE_ADAPTERS", "SSLM_ADAPTER_HAS_BOUND_SEQUENCES",
	                                "SSLM_GPU_SHADER_BINARY_STALE", "SSLM_GPU_ALLOCATION_FAILED",
	                                "SSLM_OUTPUT_BUFFER_TOO_SMALL", "SSLM_PREFILL_HIDDEN_UNAVAILABLE",
	                                "SSLM_GPU_SHADER_DIR_INVALID", "SSLM_GPU_SHADER_DIR_CONFLICT",
	                                "SSLM_GPU_PARALLEL_FOR_INVALID", "SSLM_GPU_PARALLEL_FOR_INCOMPLETE",
	                                "SSLM_GPU_RESIDENCY_FLAGS_INVALID", "SSLM_GPU_KV_BUDGET_EXCEEDED",
	                                "SSLM_GPU_KV_POOL_EXHAUSTED", "SSLM_GPU_KV_POOL_TOO_LARGE",
	                                "SSLM_GPU_HOLDER_MODE_MISMATCH", "SSLM_GPU_PREFIX_REJECTED",
	                                "SSLM_GPU_PREFIX_SCHEMA_MISMATCH"};
	const uint32_t i = static_cast<uint32_t>(s);
	return i < sizeof k / sizeof k[0] ? k[i] : "UNKNOWN_STATUS";
}
inline int AsInt(SslmGpuStatus s) { return static_cast<int>(s); }

// ---- run options (set by the runner from argv; environment as a fallback) -----------------------

struct Options {
	std::string fixture_dir;    // tools/gen_paged_kv_fixture.py's OUT_DIR
	std::string reference_dir;  // tests/paged-kv/reference
	std::string model05;        // the real Qwen2.5-0.5B artifact (cap 4096)
	std::string model15;        // the real Qwen2.5-1.5B artifact
	std::string gpu_ref_dir;    // where 6.5a/4.8a keep the pre-G2 GPU records
	bool record_gpu_ref = false;
};
inline Options& Opt() {
	static Options o;
	return o;
}
inline std::string EnvOr(const char* name, const std::string& fallback) {
	if (!fallback.empty()) return fallback;
	const char* v = std::getenv(name);
	return v ? std::string(v) : std::string();
}

// The certified GPU this process runs on. 4.8, 6.5a and 6.5b run once per GPU (the runner sets
// SSLM_GPU_ADAPTER_INDEX, which the harness device reads, d3d12_harness.h); a run that did not
// select one is not a vendor run and fails the cell.
inline std::string AdapterTag() {
	const char* v = std::getenv("SSLM_GPU_ADAPTER_INDEX");
	G1A_CHECK_MSG(v && *v, "SSLM_GPU_ADAPTER_INDEX is not set: this cell runs once per certified GPU");
	return v && *v ? std::string("adapter") + v : std::string("adapter-unset");
}

// ---- memory ---------------------------------------------------------------------------------

struct AlignedBuf {
	uint8_t* p = nullptr;
	size_t n = 0;
	AlignedBuf() = default;
	explicit AlignedBuf(size_t size, uint8_t fill = 0) { Reset(size, fill); }
	AlignedBuf(const AlignedBuf&) = delete;
	AlignedBuf& operator=(const AlignedBuf&) = delete;
	~AlignedBuf() { Free(); }
	void Reset(size_t size, uint8_t fill = 0) {
		Free();
		n = (size + 63) / 64 * 64;
		if (n == 0) n = 64;
		p = static_cast<uint8_t*>(::operator new(n, std::align_val_t(64)));
		std::memset(p, fill, n);
	}
	void Free() {
		if (p) ::operator delete(p, std::align_val_t(64));
		p = nullptr;
		n = 0;
	}
};

inline bool ReadAll(const std::string& path, std::vector<uint8_t>* out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	out->assign(std::istreambuf_iterator<char>(f), {});
	return true;
}

// ---- the GPU context ------------------------------------------------------------------------

inline SslmGpuContext* Ctx() {
	static SslmGpuContext* ctx = nullptr;
	static bool tried = false;
	if (!tried) {
		tried = true;
		GpuContextConfig cfg{};
		const SslmGpuStatus st = sslm_gpu_context_create(cfg, &ctx);
		G1A_CHECK_MSG(st == GS::SSLM_OK && ctx, "sslm_gpu_context_create -> %s", GpuStatusName(st));
	}
	return ctx;
}

// ---- fixtures -------------------------------------------------------------------------------

// One artifact mapped on both backends. Geometry, B and page_bytes are computed here from the
// artifact's own config by §3.1's rules, never read from the code under test.
struct Fx {
	std::string name;   // "pkv_def", "pkv_qk", "qwen05", "qwen15"
	std::string ref;    // the R0 .ref stem, empty for an artifact without scenario records
	bool ok = false;
	AlignedBuf bytes;
	size_t size = 0;
	std::string sha;
	superslm::SslmModelView view{};
	sslm_model cpu = nullptr;
	SslmGpuModelHandle* gpu = nullptr;
	pkv::Geometry geo{};
	int64_t B = 0;
	size_t elem = 1;
	size_t page_bytes = 0;
	size_t hidden = 0;
	int32_t vocab = 0;
	bool has_qk = false;
	uint32_t one_layer = 0;   // the dispatch budget of exactly one layer
	uint32_t full_depth = 0;  // the dispatch budget of every layer

	int64_t CapPages() const { return (geo.context_cap + B - 1) / B; }
	// §3.4's R(budget), written here: min(ceil(budget / B) + 1, ceil(cap / B)).
	int64_t R(int64_t budget) const {
		const int64_t r = (budget + B - 1) / B + 1;
		return r < CapPages() ? r : CapPages();
	}
	// §3.1's address within a page, written here: ((l*2 + half)*H_kv + h)*B*D + (pos % B)*D, in elements.
	size_t InPage(uint32_t l, uint32_t half, uint32_t h, int64_t pos) const {
		return ((((size_t{l} * 2u + half) * geo.kv_heads + h) * static_cast<size_t>(B)) +
		        static_cast<size_t>(pos % B)) * geo.head_dim * elem;
	}
	// The largest page count whose pool is at most 2^32 bytes (§4 item 3): 43,690 at 0.5B, 18,724 at 1.5B.
	uint32_t CeilingPages() const { return static_cast<uint32_t>((uint64_t{1} << 32) / page_bytes); }
	~Fx() {
		if (gpu && Ctx()) sslm_gpu_model_unmap(Ctx(), gpu);
		if (cpu) sslm_model_unmap(cpu);
	}
};

inline void FinishFx(Fx& f) {
	std::string err;
	if (superslm::SslmModel::Load(f.bytes.p, f.size, f.view, &err) != superslm::SslmModelStatus::Ok) {
		G1A_CHECK_MSG(false, "fixture %s: SslmModel::Load failed: %s", f.name.c_str(), err.c_str());
		return;
	}
	const superslm::SslmModelConfig& c = f.view.config;
	f.geo = {c.num_hidden_layers, c.num_key_value_heads, c.head_dim, static_cast<int64_t>(c.context_cap)};
	const int64_t cap = c.context_cap, kb = c.kv_block_size;
	f.B = (kb > 0 && kb <= cap && cap % kb == 0) ? kb : cap;  // §3.1, written here
	f.elem = c.kv_precision == superslm::SslmKvPrecision::Int8 ? 1u : 2u;
	f.page_bytes = size_t{f.geo.layers} * 2u * f.geo.kv_heads * static_cast<size_t>(f.B) * f.geo.head_dim * f.elem;
	f.hidden = c.hidden_size;
	f.vocab = static_cast<int32_t>(c.vocab_size);
	for (uint32_t l = 0; l < f.geo.layers; ++l)
		if (f.view.weights.Tensor("layer" + std::to_string(l) + ".q_norm.gain") != nullptr) f.has_qk = true;
	f.one_layer = superslm_gpu::DispatchesPerLayer(f.has_qk);
	f.full_depth = f.one_layer * f.geo.layers;
	const sslm_status cs = sslm_model_map(f.bytes.p, f.size, &f.cpu);
	G1A_CHECK_MSG(cs == SSLM_OK, "fixture %s: sslm_model_map -> %d", f.name.c_str(), static_cast<int>(cs));
	SslmGpuContext* ctx = Ctx();
	const SslmGpuStatus gs = ctx ? sslm_gpu_model_map(ctx, &f.view, GpuResidencyConfig{}, &f.gpu) : GS::SSLM_DEVICE_LOST;
	G1A_CHECK_MSG(gs == GS::SSLM_OK && f.gpu, "fixture %s: sslm_gpu_model_map -> %s", f.name.c_str(), GpuStatusName(gs));
	f.ok = cs == SSLM_OK && gs == GS::SSLM_OK && f.gpu;
	if (f.ok)
		std::printf("  fixture %s: L=%u H_kv=%u D=%u cap=%lld B=%lld page_bytes=%zu qk=%d\n", f.name.c_str(), f.geo.layers,
		            f.geo.kv_heads, f.geo.head_dim, static_cast<long long>(f.geo.context_cap),
		            static_cast<long long>(f.B), f.page_bytes, f.has_qk ? 1 : 0);
}

inline const Fx& LoadFrom(const std::string& name, const std::string& path, const std::string& ref) {
	static std::map<std::string, std::unique_ptr<Fx>> cache;
	auto it = cache.find(name);
	if (it != cache.end()) {
		G1A_CHECK_MSG(it->second->ok, "fixture %s did not load (see its first failure)", name.c_str());
		return *it->second;
	}
	auto f = std::make_unique<Fx>();
	f->name = name;
	f->ref = ref;
	std::vector<uint8_t> raw;
	if (!path.empty() && ReadAll(path, &raw) && !raw.empty()) {
		f->size = raw.size();
		f->bytes.Reset(raw.size());
		std::memcpy(f->bytes.p, raw.data(), raw.size());
		f->sha = pkv::Sha(raw.data(), raw.size());
		FinishFx(*f);
	} else {
		G1A_CHECK_MSG(false, "fixture %s: %s not found or empty", name.c_str(), path.empty() ? "(no path given)" : path.c_str());
	}
	const Fx& out = *f;
	cache[name] = std::move(f);
	return out;
}

// tools/gen_paged_kv_fixture.py's fixtures ("pkv_def", "pkv_qk"): 2 layers, 2 KV heads, vocabulary
// 256, kv_block_size 16, cap 4096, so B = 16 and a whole-cap reservation is 256 pages.
inline const Fx& PkvFixture(const char* stem) {
	const std::string dir = EnvOr("SUPERSLM_PAGED_KV_FIXTURE_DIR", Opt().fixture_dir);
	G1A_CHECK_MSG(!dir.empty(), "no fixture directory: pass --pkv-fixtures=DIR or set SUPERSLM_PAGED_KV_FIXTURE_DIR "
	                            "(python tools/gen_paged_kv_fixture.py DIR)");
	return LoadFrom(stem, dir.empty() ? std::string() : dir + "/" + stem + ".sslm", stem);
}
inline const Fx& Real05() { return LoadFrom("qwen05", EnvOr("SUPERSLM_G1A_MODEL_05B", Opt().model05), ""); }
inline const Fx& Real15() { return LoadFrom("qwen15", EnvOr("SUPERSLM_G1A_MODEL_15B", Opt().model15), ""); }

// ---- the R0 reference ---------------------------------------------------------------------------

struct Rec {
	std::string stage;
	std::vector<int32_t> tokens;  // decoded since the previous record
	int64_t L = -1;               // -1: no rows at this stage
	std::string rows_sha = "-";
};

struct RefFile {
	std::string fixture_sha;
	std::map<std::string, Rec> by_key;  // "<scenario> <stage>"
	bool ok = false;
};

inline std::vector<int32_t> ParseTokens(const std::string& s) {
	std::vector<int32_t> t;
	size_t at = 0;
	while (at < s.size()) {
		size_t comma = s.find(',', at);
		if (comma == std::string::npos) comma = s.size();
		if (comma > at) t.push_back(static_cast<int32_t>(std::atoi(s.substr(at, comma - at).c_str())));
		at = comma + 1;
	}
	return t;
}

inline RefFile ReadRef(const std::string& path) {
	RefFile r;
	std::ifstream f(path);
	if (!f) return r;
	std::string line;
	while (std::getline(f, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		if (line.empty()) continue;
		if (line[0] == '#') {
			const size_t at = line.find("sha256=");
			if (at != std::string::npos) r.fixture_sha = line.substr(at + 7, 64);
			continue;
		}
		// <scenario> <stage> L=<n> sat=<n> blob=<hex|-> rows=<hex|-> tokens=<a,b,...>
		const size_t s1 = line.find(' '), s2 = line.find(' ', s1 + 1);
		if (s1 == std::string::npos || s2 == std::string::npos) continue;
		Rec rec;
		rec.stage = line.substr(s1 + 1, s2 - s1 - 1);
		auto field = [&](const char* k) -> std::string {
			const size_t a = line.find(k);
			if (a == std::string::npos) return std::string();
			const size_t b = a + std::strlen(k), e = line.find(' ', b);
			return line.substr(b, e == std::string::npos ? std::string::npos : e - b);
		};
		rec.L = std::atoll(field(" L=").c_str());
		rec.rows_sha = field(" rows=");
		rec.tokens = ParseTokens(field(" tokens="));
		r.by_key[line.substr(0, s2)] = rec;
	}
	r.ok = true;
	return r;
}

inline const RefFile& Ref(const std::string& tag, const Fx& fx) {
	static std::map<std::string, RefFile> cache;
	const std::string key = tag + "_" + fx.ref;
	auto it = cache.find(key);
	if (it == cache.end()) {
		const std::string dir = EnvOr("SUPERSLM_PKV_REFERENCE_DIR", Opt().reference_dir);
		RefFile r = ReadRef(dir + "/" + key + ".ref");
		G1A_CHECK_MSG(r.ok, "reference %s/%s.ref not readable (pass --pkv-reference=tests/paged-kv/reference)",
		              dir.c_str(), key.c_str());
		G1A_CHECK_MSG(!r.ok || r.fixture_sha == fx.sha, "reference %s was made from fixture %s, this run maps %s",
		              key.c_str(), r.fixture_sha.c_str(), fx.sha.c_str());
		it = cache.emplace(key, std::move(r)).first;
	}
	return it->second;
}

enum class Cmp { kTokens, kTokensAndRows };

// Writes records in the .ref format (blob and sat not recorded), for 6.5a's pre-G2 GPU record.
inline bool WriteRecords(const std::string& path, const std::string& header, const char* scenario,
                         const std::vector<Rec>& recs, bool append) {
	std::ofstream f(path, append ? std::ios::app : std::ios::trunc);
	if (!f) return false;
	if (!append) f << "# " << header << "\n";
	for (const Rec& r : recs) {
		f << scenario << " " << r.stage << " L=" << r.L << " sat=-1 blob=- rows=" << r.rows_sha << " tokens=";
		for (size_t i = 0; i < r.tokens.size(); ++i) f << (i ? "," : "") << r.tokens[i];
		f << "\n";
	}
	return static_cast<bool>(f);
}

// Each record equals the reference's record for the same stage: tokens, and with kTokensAndRows the
// context length and the SHA-256 of rows [0, L) in pkv_scenarios.h's position-major order.
inline void ExpectRecords(const std::map<std::string, Rec>& want_by_key, const char* scenario,
                          const std::vector<Rec>& got, Cmp cmp, const std::string& what) {
	G1A_CHECK_MSG(!got.empty(), "%s %s: the run produced no record", what.c_str(), scenario);
	for (const Rec& g : got) {
		const auto it = want_by_key.find(std::string(scenario) + " " + g.stage);
		G1A_CHECK_MSG(it != want_by_key.end(), "%s: no reference record for %s %s", what.c_str(), scenario, g.stage.c_str());
		if (it == want_by_key.end()) continue;
		const Rec& w = it->second;
		G1A_CHECK_MSG(g.tokens == w.tokens, "%s %s %s: tokens differ from the reference", what.c_str(), scenario,
		              g.stage.c_str());
		if (cmp == Cmp::kTokensAndRows && w.L >= 0 && g.L >= 0) {
			G1A_CHECK_MSG(g.L == w.L, "%s %s %s: L %lld vs %lld", what.c_str(), scenario, g.stage.c_str(),
			              static_cast<long long>(g.L), static_cast<long long>(w.L));
			G1A_CHECK_MSG(g.rows_sha == w.rows_sha, "%s %s %s: K/V rows [0, L) differ from the reference", what.c_str(),
			              scenario, g.stage.c_str());
		}
	}
}
inline void ExpectReference(const char* tag, const Fx& fx, const char* scenario, const std::vector<Rec>& got, Cmp cmp,
                            const std::string& what) {
	ExpectRecords(Ref(tag, fx).by_key, scenario, got, cmp, what);
}
// Two runs of one scenario, record by record (a GPU run against its CPU twin, or a twin run).
inline void ExpectSameRecords(const std::vector<Rec>& got, const std::vector<Rec>& want, Cmp cmp, const std::string& what) {
	G1A_CHECK_MSG(got.size() == want.size() && !got.empty(), "%s: %zu records vs %zu", what.c_str(), got.size(), want.size());
	for (size_t i = 0; i < got.size() && i < want.size(); ++i) {
		G1A_CHECK_MSG(got[i].stage == want[i].stage, "%s: stage %s vs %s", what.c_str(), got[i].stage.c_str(),
		              want[i].stage.c_str());
		G1A_CHECK_MSG(got[i].tokens == want[i].tokens, "%s %s: tokens differ", what.c_str(), got[i].stage.c_str());
		if (cmp == Cmp::kTokensAndRows) {
			G1A_CHECK_MSG(got[i].L == want[i].L, "%s %s: L %lld vs %lld", what.c_str(), got[i].stage.c_str(),
			              static_cast<long long>(got[i].L), static_cast<long long>(want[i].L));
			G1A_CHECK_MSG(got[i].rows_sha == want[i].rows_sha, "%s %s: K/V rows differ", what.c_str(), got[i].stage.c_str());
		}
	}
}

// ---- K/V readers (position-major rows [0, L), pkv::RowsFromBlock's order) ------------------------

// Rows from a set of pages addressed by a table, by §3.1's formula written here.
template <typename PeekPage>
inline bool RowsFromTable(const Fx& fx, const std::vector<uint32_t>& table, int64_t L, PeekPage peek,
                          std::vector<uint8_t>* rows) {
	rows->clear();
	const int64_t need_pages = (L + fx.B - 1) / fx.B;
	if (static_cast<int64_t>(table.size()) < need_pages) return false;
	std::map<uint32_t, std::vector<uint8_t>> pages;
	for (int64_t j = 0; j < need_pages; ++j) {
		std::vector<uint8_t> pg(fx.page_bytes, 0);
		if (!peek(table[static_cast<size_t>(j)], pg.data())) return false;
		pages[table[static_cast<size_t>(j)]] = std::move(pg);
	}
	const size_t row = size_t{fx.geo.head_dim} * fx.elem;
	rows->reserve(static_cast<size_t>(L) * fx.geo.layers * 2u * fx.geo.kv_heads * row);
	for (int64_t p = 0; p < L; ++p)
		for (uint32_t l = 0; l < fx.geo.layers; ++l)
			for (uint32_t half = 0; half < 2; ++half)
				for (uint32_t h = 0; h < fx.geo.kv_heads; ++h) {
					const std::vector<uint8_t>& pg = pages[table[static_cast<size_t>(p / fx.B)]];
					const uint8_t* r = pg.data() + fx.InPage(l, half, h, p);
					rows->insert(rows->end(), r, r + row);
				}
	return true;
}

// A CPU holder's rows [0, L) from its saved blob, as 1.12.0's own CPU cells read them (pkv_common.h's
// BlobRows, pkv_scenarios.h's Run::Save), in pkv::RowsFromBlock's position-major order:
//   - a no-budget (whole_reserve) holder saves 1.9.0's 'SSB5', whose K/V section, the blob's last
//     sslm_kv_block_size bytes, is the whole block in the flat layout (context_cap positions);
//   - a budget holder saves 'SSB6': the 172-byte header, the residual (hidden_size), the history (4 bytes
//     each; history_count u64 at 112), kv_positions u64 = L', then rows [0, L') in the canonical layout
//     (the flat layout with context_cap replaced by L').
// Not through the CPU's page seams: since 1.12.0 the CPU's B is §3.1's rule (sslm_kv_page_positions, 128
// on pkv_def and 64 on pkv_qk) with rotated page slots, not the artifact's kv_block_size that Fx::B,
// page_bytes and InPage are written at, so a table read at this file's geometry finds too few pages
// ("rows unreadable") or reads the wrong bytes. The blob's layout does not depend on B.
inline bool CpuRows(const Fx& fx, sslm_seq s, int64_t L, std::vector<uint8_t>* rows) {
	rows->clear();
	size_t need = 0;
	if (sslm_seq_save(s, nullptr, &need) != SSLM_BUFFER_TOO_SMALL || need == 0) return false;
	std::vector<uint8_t> b(need, 0);
	size_t wrote = need;
	if (sslm_seq_save(s, b.data(), &wrote) != SSLM_OK) return false;
	b.resize(wrote);
	if (L < 0 || L > fx.geo.context_cap || b.size() < 4) return false;
	if (std::memcmp(b.data(), "SSB5", 4) == 0) {
		const size_t block = sslm_kv_block_size(fx.cpu);
		if (block == 0 || b.size() < block) return false;
		*rows = pkv::RowsFromBlock(b.data() + b.size() - block, fx.geo, L);
		return true;
	}
	if (std::memcmp(b.data(), "SSB6", 4) != 0 || b.size() < 172) return false;
	uint64_t history = 0, lprime = 0;
	std::memcpy(&history, b.data() + 112, 8);
	if (history > b.size()) return false;
	const size_t at = 172 + fx.hidden + 4 * static_cast<size_t>(history);
	if (at + 8 > b.size()) return false;
	std::memcpy(&lprime, b.data() + at, 8);
	const size_t per_pos = size_t{fx.geo.layers} * 2u * fx.geo.kv_heads * fx.geo.head_dim * fx.elem;
	if (static_cast<uint64_t>(L) > lprime || lprime > static_cast<uint64_t>(fx.geo.context_cap) ||
	    b.size() - (at + 8) < lprime * per_pos)
		return false;
	pkv::Geometry g = fx.geo;
	g.context_cap = static_cast<int64_t>(lprime);
	*rows = pkv::RowsFromBlock(b.data() + at + 8, g, L);
	return true;
}

// A pooled GPU sequence's table through the G3 table seam, and its rows through the GPU mirror peek.
inline std::vector<uint32_t> GpuTable(const SslmGpuSequenceHandle* s) {
	std::vector<uint32_t> t;
	uint32_t page = 0, mapped = 0;
	if (sslm_gpu_test_only_seq_table_entry(s, 0, &page, &mapped) != GS::SSLM_OK) return t;
	for (uint32_t i = 0; i < mapped; ++i)
		if (sslm_gpu_test_only_seq_table_entry(s, i, &page, nullptr) == GS::SSLM_OK) t.push_back(page);
	return t;
}
inline bool GpuPeekPage(const Fx& fx, const SslmGpuKvPoolHandle* pool, uint32_t page, uint8_t* out) {
	return sslm_gpu_test_only_peek_pool_mirror_bytes(pool, page, 0, fx.page_bytes, out) == GS::SSLM_OK;
}
inline bool GpuPooledRows(const Fx& fx, const SslmGpuKvPoolHandle* pool, const SslmGpuSequenceHandle* s, int64_t L,
                          std::vector<uint8_t>* rows) {
	return RowsFromTable(fx, GpuTable(s), L, [&](uint32_t page, uint8_t* out) { return GpuPeekPage(fx, pool, page, out); },
	                     rows);
}
// The whole pool mirror, page by page (the "whole GPU mirror unchanged" operand).
inline std::vector<uint8_t> WholeMirror(const Fx& fx, const SslmGpuKvPoolHandle* pool, uint32_t page_count) {
	std::vector<uint8_t> all(static_cast<size_t>(page_count) * fx.page_bytes, 0);
	for (uint32_t p = 0; p < page_count; ++p)
		if (!GpuPeekPage(fx, pool, p, all.data() + static_cast<size_t>(p) * fx.page_bytes)) {
			G1A_CHECK_MSG(false, "GPU mirror peek of page %u refused", p);
			break;
		}
	return all;
}

// A GPU sequence's saved blob (any magic), sized by the call itself.
inline std::vector<uint8_t> GpuSave(const SslmGpuSequenceHandle* s) {
	std::vector<uint8_t> blob;
	size_t need = 0;
	(void)sslm_gpu_seq_save(Ctx(), s, nullptr, &need);  // reports the size it needs
	if (need == 0) return blob;
	blob.assign(need, 0);
	size_t n = need;
	if (sslm_gpu_seq_save(Ctx(), s, blob.data(), &n) != GS::SSLM_OK) return {};
	blob.resize(n);
	return blob;
}
inline bool IsMagic(const std::vector<uint8_t>& b, const char* m) { return b.size() >= 4 && std::memcmp(b.data(), m, 4) == 0; }

// A dedicated GPU sequence's rows from its 'SLM5' blob: the 120-byte header, the 12-byte v5 tail,
// the residual (hidden_codes_size, LE64 at 40), then the K/V workspace (workspace_size, LE64 at
// 48) in the flat layout, which is the one-page view with B = cap (§3.1's identity).
inline bool GpuDedicatedRows(const Fx& fx, const SslmGpuSequenceHandle* s, int64_t L, std::vector<uint8_t>* rows) {
	const std::vector<uint8_t> b = GpuSave(s);
	if (!IsMagic(b, "SLM5") || b.size() < 132) return false;
	uint64_t hcs = 0, ws = 0;
	std::memcpy(&hcs, b.data() + 40, 8);
	std::memcpy(&ws, b.data() + 48, 8);
	const uint64_t want_ws = uint64_t{fx.geo.layers} * 2u * fx.geo.kv_heads * static_cast<uint64_t>(fx.geo.context_cap) *
	                         fx.geo.head_dim * fx.elem;
	if (ws != want_ws || b.size() < 132 + hcs + ws) return false;
	*rows = pkv::RowsFromBlock(b.data() + 132 + hcs, fx.geo, L);
	return true;
}

// ---- backends: one scenario script, three ways to run it ------------------------------------------

// Statuses are ints, 0 = OK (sslm_status or SslmGpuStatus).
struct Backend {
	const Fx& fx;
	explicit Backend(const Fx& f) : fx(f) {}
	virtual ~Backend() = default;
	virtual const char* Name() const = 0;
	virtual int Create(int32_t budget, void** out) = 0;
	virtual int Prefill(void* s, const std::vector<int32_t>& t) = 0;
	virtual int Decode(void* s, int32_t n, std::vector<int32_t>* out) = 0;
	virtual int Reset(void* s) = 0;
	virtual void Release(void* s) = 0;
	virtual int PrefixBegin(int32_t budget, void** out) = 0;
	virtual int PrefixBeginFrom(void* parent, int32_t budget, void** out) = 0;
	virtual int PrefixPrefill(void* p, const std::vector<int32_t>& t) = 0;
	virtual int Freeze(void* p) = 0;
	virtual void PrefixRelease(void* p) = 0;
	virtual int Adopt(void* s, void* p) = 0;
	virtual bool Rows(void* s, int64_t L, std::vector<uint8_t>* rows) = 0;
};

// The CPU paged twin: a page pool, budget holders and budget prefixes through the 1.12.0 C ABI.
struct CpuPagedBackend : Backend {
	AlignedBuf mem;
	sslm_kv_pool pool = nullptr;
	bool legacy = false;  // true: holders by sslm_seq_create (no budget; the legacy one-page view's CPU twin)
	CpuPagedBackend(const Fx& f, uint32_t pages, bool legacy_holders = false) : Backend(f), legacy(legacy_holders) {
		const size_t need = static_cast<size_t>(pages) * sslm_kv_page_size(f.cpu) + sslm_kv_page_pool_overhead_size(f.cpu, pages);
		mem.Reset(need);
		const sslm_status st = sslm_kv_page_pool_create(f.cpu, mem.p, mem.n, pages, &pool);
		G1A_CHECK_MSG(st == SSLM_OK, "CPU twin: sslm_kv_page_pool_create(%u) -> %d", pages, static_cast<int>(st));
	}
	~CpuPagedBackend() override {
		if (pool) sslm_kv_pool_destroy(pool);
	}
	const char* Name() const override { return legacy ? "cpu-legacy" : "cpu-paged"; }
	int Create(int32_t budget, void** out) override {
		sslm_seq s = nullptr;
		const sslm_status st = legacy ? sslm_seq_create(fx.cpu, &pool, &s) : sslm_seq_create_budgeted(fx.cpu, &pool, budget, &s);
		*out = s;
		return st;
	}
	int Prefill(void* s, const std::vector<int32_t>& t) override {
		size_t at = 0;
		while (at < t.size()) {
			int32_t consumed = 0;
			const sslm_status st = sslm_prefill(fx.cpu, static_cast<sslm_seq>(s), t.data() + at,
			                                    static_cast<int32_t>(t.size() - at), 64, SSLM_SPAN_PROMPT, nullptr, &consumed);
			if (st != SSLM_OK) return st;
			if (consumed <= 0) return SSLM_INVALID_ARGUMENT;
			at += static_cast<size_t>(consumed);
		}
		return SSLM_OK;
	}
	int Decode(void* s, int32_t n, std::vector<int32_t>* out) override {
		for (int32_t i = 0; i < n; ++i) {
			sslm_decode_params p{};
			p.layer_budget = 1;
			sslm_seq b[1] = {static_cast<sslm_seq>(s)};
			int32_t tok = -1;
			for (int guard = 0; tok < 0 && guard < 4096; ++guard) {
				const sslm_status st = sslm_decode_step(fx.cpu, b, 1, &p, nullptr, &tok);
				if (st != SSLM_OK) return st;
				if (tok == -2) break;
			}
			if (tok < 0) return SSLM_INVALID_ARGUMENT;
			out->push_back(tok);
		}
		return SSLM_OK;
	}
	int Reset(void* s) override { return sslm_seq_reset(static_cast<sslm_seq>(s)); }
	void Release(void* s) override { sslm_seq_release(static_cast<sslm_seq>(s)); }
	int PrefixBegin(int32_t budget, void** out) override {
		sslm_prefix p = nullptr;
		const sslm_status st = sslm_prefix_begin_budgeted(fx.cpu, &pool, budget, &p);
		*out = p;
		return st;
	}
	int PrefixBeginFrom(void* parent, int32_t budget, void** out) override {
		sslm_prefix p = nullptr;
		const sslm_status st = sslm_prefix_begin_from(static_cast<sslm_prefix>(parent), budget, &p);
		*out = p;
		return st;
	}
	int PrefixPrefill(void* p, const std::vector<int32_t>& t) override {
		size_t at = 0;
		while (at < t.size()) {
			int32_t consumed = 0;
			const sslm_status st = sslm_prefix_prefill(fx.cpu, static_cast<sslm_prefix>(p), t.data() + at,
			                                           static_cast<int32_t>(t.size() - at), 64, SSLM_SPAN_PROMPT, nullptr,
			                                           &consumed);
			if (st != SSLM_OK) return st;
			if (consumed <= 0) return SSLM_INVALID_ARGUMENT;
			at += static_cast<size_t>(consumed);
		}
		return SSLM_OK;
	}
	int Freeze(void* p) override { return sslm_prefix_freeze(static_cast<sslm_prefix>(p)); }
	void PrefixRelease(void* p) override { sslm_prefix_release(static_cast<sslm_prefix>(p)); }
	int Adopt(void* s, void* p) override {
		return sslm_seq_adopt_prefix(static_cast<sslm_seq>(s), static_cast<sslm_prefix>(p));
	}
	bool Rows(void* s, int64_t L, std::vector<uint8_t>* rows) override {
		return CpuRows(fx, static_cast<sslm_seq>(s), L, rows);
	}
};

// How a GPU backend decodes: the bridge decode (embed-if-needed, drive to depth with `budget`,
// finish), or the asynchronous verbs by hand (embed, sslm_decode_step_gpu with `budget` and
// sslm_gpu_ready until full depth, then the finish), which is how 8.13 splits a token's layers.
enum class GpuDecodeMode { kBridge, kAsyncByHand };

inline SslmGpuStatus DrainGpu(SslmGpuSequenceHandle* s) {
	int32_t ready = 0;
	SslmGpuStatus out_status = GS::SSLM_OK;
	for (int guard = 0; !ready && guard < 1 << 20; ++guard) {
		const SslmGpuStatus st = sslm_gpu_ready(Ctx(), s, /*block=*/1, &ready, &out_status);
		if (st != GS::SSLM_OK) return st;
	}
	return out_status;
}

// One greedy token on a GPU sequence. `last` is the previous emitted token (the one to embed when
// no ready token is pending), -1 if none.
inline SslmGpuStatus GpuNextToken(const Fx& fx, SslmGpuSequenceHandle* s, int32_t last, GpuDecodeMode mode,
                                  uint32_t budget, int32_t* tok) {
	*tok = -1;
	if (mode == GpuDecodeMode::kBridge) return SslmGpuSeqDecodeStepForG5Bridge(Ctx(), s, last < 0 ? 0 : last, budget, tok);
	// By hand. A pending ready token (after a prefill or an adopt) sits at full depth: finish it.
	if (*SslmGpuSeqHandleLayerIndexForBench(s) == fx.geo.layers) return SslmGpuSeqFinishTokenForG5Bridge(Ctx(), s, tok);
	SslmGpuStatus st = sslm_gpu_seq_embed_token(Ctx(), s, last < 0 ? 0 : last);
	if (st != GS::SSLM_OK) return st;
	for (int guard = 0; *SslmGpuSeqHandleLayerIndexForBench(s) < fx.geo.layers && guard < 4096; ++guard) {
		st = sslm_decode_step_gpu(Ctx(), s, nullptr, budget);
		if (st != GS::SSLM_OK) return st;
		st = DrainGpu(s);
		if (st != GS::SSLM_OK) return st;
	}
	return SslmGpuSeqFinishTokenForG5Bridge(Ctx(), s, tok);
}

struct GpuBackendBase : Backend {
	GpuDecodeMode mode = GpuDecodeMode::kBridge;
	uint32_t decode_budget = 0;  // 0: full depth
	std::map<void*, int32_t> last;
	explicit GpuBackendBase(const Fx& f) : Backend(f) {}
	int Prefill(void* s, const std::vector<int32_t>& t) override {
		if (t.empty()) return 0;
		const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(Ctx(), static_cast<SslmGpuSequenceHandle*>(s), t.data(),
		                                                            static_cast<int32_t>(t.size()), fx.one_layer);
		if (st == GS::SSLM_OK) last[s] = t.back();
		return AsInt(st);
	}
	int Decode(void* s, int32_t n, std::vector<int32_t>* out) override {
		for (int32_t i = 0; i < n; ++i) {
			int32_t tok = -1;
			const auto it = last.find(s);
			const SslmGpuStatus st = GpuNextToken(fx, static_cast<SslmGpuSequenceHandle*>(s), it == last.end() ? -1 : it->second,
			                                      mode, decode_budget ? decode_budget : fx.full_depth, &tok);
			if (st != GS::SSLM_OK) return AsInt(st);
			if (tok < 0) return AsInt(GS::SSLM_SEQUENCE_REJECTED);
			last[s] = tok;
			out->push_back(tok);
		}
		return 0;
	}
	int Reset(void* s) override {
		last.erase(s);
		return AsInt(sslm_gpu_seq_reset(Ctx(), static_cast<SslmGpuSequenceHandle*>(s)));
	}
	void Release(void* s) override {
		last.erase(s);
		sslm_gpu_seq_release(Ctx(), static_cast<SslmGpuSequenceHandle*>(s));
	}
};

// Today's GPU sequence: a dedicated buffer at the model's cap (the legacy one-page view from G2).
// It has no prefixes; a scenario that needs one cannot run on it.
struct GpuDedicatedBackend : GpuBackendBase {
	explicit GpuDedicatedBackend(const Fx& f) : GpuBackendBase(f) {}
	const char* Name() const override { return "gpu-dedicated"; }
	int Create(int32_t, void** out) override {
		SslmGpuSequenceHandle* s = nullptr;
		const SslmGpuStatus st = sslm_gpu_seq_create(Ctx(), fx.gpu, fx.geo.context_cap, &s);
		*out = s;
		return AsInt(st);
	}
	int PrefixBegin(int32_t, void** out) override { *out = nullptr; return AsInt(GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH); }
	int PrefixBeginFrom(void*, int32_t, void** out) override { *out = nullptr; return AsInt(GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH); }
	int PrefixPrefill(void*, const std::vector<int32_t>&) override { return AsInt(GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH); }
	int Freeze(void*) override { return AsInt(GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH); }
	void PrefixRelease(void*) override {}
	int Adopt(void*, void*) override { return AsInt(GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH); }
	bool Rows(void* s, int64_t L, std::vector<uint8_t>* rows) override {
		return GpuDedicatedRows(fx, static_cast<SslmGpuSequenceHandle*>(s), L, rows);
	}
};

// A GPU page pool with budgeted sequences (G3).
struct GpuPooledBackend : GpuBackendBase {
	SslmGpuKvPoolHandle* pool = nullptr;
	uint32_t pages = 0;
	GpuPooledBackend(const Fx& f, uint32_t page_count) : GpuBackendBase(f), pages(page_count) {
		const SslmGpuStatus st = sslm_gpu_kv_pool_create(Ctx(), f.gpu, page_count, &pool);
		G1A_CHECK_MSG(st == GS::SSLM_OK && pool, "sslm_gpu_kv_pool_create(%u) -> %s", page_count, GpuStatusName(st));
	}
	~GpuPooledBackend() override {
		if (pool) {
			const SslmGpuStatus st = sslm_gpu_kv_pool_destroy(Ctx(), pool);
			G1A_CHECK_MSG(st == GS::SSLM_OK, "sslm_gpu_kv_pool_destroy after the scenario's teardown -> %s", GpuStatusName(st));
		}
	}
	const char* Name() const override { return "gpu-pooled"; }
	int Create(int32_t budget, void** out) override {
		SslmGpuSequenceHandle* s = nullptr;
		const SslmGpuStatus st = sslm_gpu_seq_create_budgeted(Ctx(), fx.gpu, pool, budget, &s);
		*out = s;
		return AsInt(st);
	}
	// No GPU prefix exists before G4: a scenario that needs one fails on the first prefix call here.
	// GpuPrefixBackend (below) adds them, so the G3 executable links without any G4 verb.
	int PrefixBegin(int32_t, void** out) override { *out = nullptr; return AsInt(GS::SSLM_GPU_PREFIX_REJECTED); }
	int PrefixBeginFrom(void*, int32_t, void** out) override { *out = nullptr; return AsInt(GS::SSLM_GPU_PREFIX_REJECTED); }
	int PrefixPrefill(void*, const std::vector<int32_t>&) override { return AsInt(GS::SSLM_GPU_PREFIX_REJECTED); }
	int Freeze(void*) override { return AsInt(GS::SSLM_GPU_PREFIX_REJECTED); }
	void PrefixRelease(void*) override {}
	int Adopt(void*, void*) override { return AsInt(GS::SSLM_GPU_PREFIX_REJECTED); }
	bool Rows(void* s, int64_t L, std::vector<uint8_t>* rows) override {
		return GpuPooledRows(fx, pool, static_cast<SslmGpuSequenceHandle*>(s), L, rows);
	}
};

// The pooled backend with GPU prefixes (G4): sslm_gpu_prefix_begin/_begin_from/_prefill/_freeze/
// _release and sslm_gpu_seq_adopt_prefix. Only G4 and G5 cells construct it.
// Every token of `t` into GPU prefix `p` through sslm_gpu_prefix_prefill, looping on `*consumed`
// (§7 5.8: the verb's dispatch_budget may stop mid-span and the call is resumable). Each call passes
// a full-depth budget, i.e. one token's cost (dispatches per layer x layers), so every call admits at
// least one token. `*total` is the tokens that landed; returns the first non-OK status, or a call that
// admitted nothing as SSLM_SEQUENCE_REJECTED.
inline SslmGpuStatus GpuPrefixPrefillAll(const Fx& fx, SslmGpuPrefixHandle* p, const int32_t* t, int32_t n, int32_t kind,
                                         int32_t* total = nullptr) {
	int32_t done = 0;
	SslmGpuStatus st = GS::SSLM_OK;
	while (done < n) {
		int32_t consumed = 0;
		st = sslm_gpu_prefix_prefill(Ctx(), p, t + done, n - done, kind, fx.full_depth, &consumed);
		if (consumed > 0) done += consumed;
		if (st != GS::SSLM_OK) break;
		if (consumed <= 0) {
			st = GS::SSLM_SEQUENCE_REJECTED;
			break;
		}
	}
	if (total) *total = done;
	return st;
}

struct GpuPrefixBackend : GpuPooledBackend {
	using GpuPooledBackend::GpuPooledBackend;
	const char* Name() const override { return "gpu-pooled+prefix"; }
	int PrefixBegin(int32_t budget, void** out) override {
		SslmGpuPrefixHandle* p = nullptr;
		const SslmGpuStatus st = sslm_gpu_prefix_begin(Ctx(), fx.gpu, pool, budget, &p);
		*out = p;
		return AsInt(st);
	}
	int PrefixBeginFrom(void* parent, int32_t budget, void** out) override {
		SslmGpuPrefixHandle* p = nullptr;
		const SslmGpuStatus st = sslm_gpu_prefix_begin_from(Ctx(), static_cast<SslmGpuPrefixHandle*>(parent), budget, &p);
		*out = p;
		return AsInt(st);
	}
	int PrefixPrefill(void* p, const std::vector<int32_t>& t) override {
		if (t.empty()) return 0;
		int32_t consumed = 0;
		const SslmGpuStatus st = GpuPrefixPrefillAll(fx, static_cast<SslmGpuPrefixHandle*>(p), t.data(),
		                                             static_cast<int32_t>(t.size()), /*kind=*/0, &consumed);
		if (st != GS::SSLM_OK) return AsInt(st);
		return consumed == static_cast<int32_t>(t.size()) ? 0 : AsInt(GS::SSLM_SEQUENCE_REJECTED);
	}
	int Freeze(void* p) override { return AsInt(sslm_gpu_prefix_freeze(Ctx(), static_cast<SslmGpuPrefixHandle*>(p))); }
	void PrefixRelease(void* p) override { sslm_gpu_prefix_release(Ctx(), static_cast<SslmGpuPrefixHandle*>(p)); }
	int Adopt(void* s, void* p) override {
		last.erase(s);  // the adopt brings the prefix's ready token, never this sequence's last
		return AsInt(sslm_gpu_seq_adopt_prefix(Ctx(), static_cast<SslmGpuSequenceHandle*>(s), static_cast<SslmGpuPrefixHandle*>(p)));
	}
};

// ---- the scenario runner (BRun's shape: the first failing call stops the scenario) ---------------

struct SRun {
	Backend& b;
	bool with_rows;
	std::string fail;
	std::vector<Rec> records;
	std::vector<int32_t> pending;
	// The context length of each live holder, counted here from the calls the scenario made (a
	// prefill of n writes n rows; a decode writes one row unless a ready token is pending, which a
	// prefill or an adopt of a non-empty prefix leaves), never read from the code under test.
	struct Len {
		int64_t L = 0;
		bool ready = false;
	};
	std::map<void*, Len> len;
	std::map<void*, int64_t> prefix_len;
	// Optional per-mark hook (a cell's extra operand at each stage, e.g. 1.8's SLM6 blobs).
	std::function<void(const char* stage, void* s)> on_mark;
	SRun(Backend& backend, bool rows) : b(backend), with_rows(rows) {}
	bool Ok(int st, const char* what) {
		if (st == 0) return true;
		if (fail.empty()) fail = std::string(b.Name()) + ": " + what + " -> " + std::to_string(st);
		return false;
	}
	bool Create(int32_t budget, void** s) {
		if (!Ok(b.Create(budget, s), "create")) return false;
		len[*s] = Len{};
		return true;
	}
	bool Prefill(void* s, const std::vector<int32_t>& t) {
		if (!Ok(b.Prefill(s, t), "prefill")) return false;
		len[s].L += static_cast<int64_t>(t.size());
		if (!t.empty()) len[s].ready = true;
		return true;
	}
	bool Decode(void* s, int32_t n) {
		const size_t before = pending.size();
		const bool ok = Ok(b.Decode(s, n, &pending), "decode");
		for (size_t i = before; i < pending.size(); ++i) {
			if (len[s].ready) len[s].ready = false;
			else ++len[s].L;
		}
		return ok;
	}
	bool Reset(void* s) {
		if (!Ok(b.Reset(s), "reset")) return false;
		len[s] = Len{};
		return true;
	}
	void Release(void* s) {
		b.Release(s);
		len.erase(s);
	}
	bool PrefixBegin(int32_t budget, void** p) {
		if (!Ok(b.PrefixBegin(budget, p), "prefix_begin")) return false;
		prefix_len[*p] = 0;
		return true;
	}
	bool PrefixBeginFrom(void* parent, int32_t budget, void** p) {
		if (!Ok(b.PrefixBeginFrom(parent, budget, p), "prefix_begin_from")) return false;
		prefix_len[*p] = prefix_len[parent];
		return true;
	}
	bool PrefixPrefill(void* p, const std::vector<int32_t>& t) {
		if (!Ok(b.PrefixPrefill(p, t), "prefix_prefill")) return false;
		prefix_len[p] += static_cast<int64_t>(t.size());
		return true;
	}
	bool Freeze(void* p) { return Ok(b.Freeze(p), "prefix_freeze"); }
	void PrefixRelease(void* p) {
		b.PrefixRelease(p);
		prefix_len.erase(p);
	}
	bool Adopt(void* s, void* p) {
		if (!Ok(b.Adopt(s, p), "adopt")) return false;
		len[s].L = prefix_len[p];
		len[s].ready = prefix_len[p] > 0;
		return true;
	}
	bool Mark(const char* stage, void* s) {
		if (!fail.empty()) return false;
		Rec r;
		r.stage = stage;
		r.tokens.swap(pending);
		if (s) {
			r.L = len[s].L;
			if (with_rows) {
				std::vector<uint8_t> rows;
				if (!b.Rows(s, r.L, &rows)) return Ok(-1, "rows unreadable");
				r.rows_sha = pkv::Sha(rows.data(), rows.size());
			}
			if (on_mark) on_mark(stage, s);
		}
		records.push_back(std::move(r));
		return true;
	}
};

// ---- the R0 scenarios, written once over a backend ----------------------------------------------

// R0's "lifecycle" with a budget prefix and a budget holder (pkv_budget_a_helpers.h's
// BudgetLifecycle): fresh -> prefill 100 -> decode -> reset -> prefill 40 -> reset -> adopt a
// 300-token prefix -> decode -> prefill 37 -> decode, crossing 16-position pages at every stage.
//
// With `fresh`, every reset is replaced by a release and a new create, so each stage runs on a
// fresh handle driven the same way (1.1's and 1.8's "a fresh handle at each stage" operand).
inline void Lifecycle(SRun& d, int32_t seq_budget = 128, int32_t prefix_budget = 300, bool fresh = false) {
	const int32_t V = d.b.fx.vocab;
	void* px = nullptr;
	void* s = nullptr;
	auto restart = [&]() {
		if (!fresh) return d.Reset(s);
		d.Release(s);
		s = nullptr;
		return d.Create(seq_budget, &s);
	};
	if (!d.PrefixBegin(prefix_budget, &px)) return;
	if (d.PrefixPrefill(px, pkv::Stream(3, 300, V)) && d.Freeze(px) && d.Create(seq_budget, &s)) {
		(void)(d.Prefill(s, pkv::Stream(1, 100, V)) && d.Mark("prefill100", s) && d.Decode(s, 4) &&
		       d.Mark("prefill100+decode4", s) && d.Decode(s, 8) && d.Mark("continue8", nullptr) && restart() &&
		       d.Mark("reset1", s) && d.Prefill(s, pkv::Stream(2, 40, V)) && d.Decode(s, 4) &&
		       d.Mark("prefill40+decode4", s) && restart() && d.Adopt(s, px) && d.Mark("adopt300", s) && d.Decode(s, 8) &&
		       d.Mark("adopt300+decode8", s) && d.Prefill(s, pkv::Stream(4, 37, V)) && d.Decode(s, 4) &&
		       d.Mark("adopt+prefill37+decode4", s));
	}
	if (s) d.Release(s);
	d.PrefixRelease(px);
}

// R0's "long_prefill": prefill 1,500 (every chunk crosses pages), then decode 16.
inline void LongPrefill(SRun& d) {
	void* s = nullptr;
	if (!d.Create(static_cast<int32_t>(d.b.fx.geo.context_cap), &s)) return;
	if (d.Prefill(s, pkv::Stream(11, 1500, d.b.fx.vocab)) && d.Mark("prefill1500", s) && d.Decode(s, 16))
		d.Mark("prefill1500+decode16", s);
	d.Release(s);
}

// R0's "widths": one fresh holder per width W in {1, 15, 16, 17, 32, 1,064, cap - 1, cap}, with the
// budget its writes need (W + 1 below the cap).
inline void Widths(SRun& d) {
	const int64_t cap = d.b.fx.geo.context_cap;
	std::vector<int64_t> widths = {1, 15, 16, 17, 32, 1064};
	if (cap <= 4100) {
		widths.push_back(cap - 1);
		widths.push_back(cap);
	}
	for (int64_t w : widths) {
		void* s = nullptr;
		if (!d.Create(static_cast<int32_t>(std::min(w + 1, cap)), &s)) return;
		const std::string tag = "w" + std::to_string(w);
		bool ok = d.Prefill(s, pkv::Stream(41, static_cast<int32_t>(w), d.b.fx.vocab)) && d.Mark((tag + "_prefill").c_str(), s) &&
		          d.Decode(s, 1) && d.Mark((tag + "_ready").c_str(), nullptr);
		if (ok && w < cap) ok = d.Decode(s, 1) && d.Mark((tag + "_decode1").c_str(), s);
		d.Release(s);
		if (!ok) return;
	}
}

// R0's "prefix_lengths" with budget prefixes (budget max(len, 1)) and budget adopters.
inline void PrefixLengths(SRun& d, int32_t seq_budget = 64) {
	const int32_t V = d.b.fx.vocab;
	for (int32_t len : {0, 1, 15, 16, 17, 1000, 1008}) {
		void* px = nullptr;
		void* s = nullptr;
		if (!d.PrefixBegin(len > 0 ? len : 1, &px)) return;
		const std::string tag = "p" + std::to_string(len);
		const bool ok = (len == 0 || d.PrefixPrefill(px, pkv::Stream(51, len, V))) && d.Freeze(px) && d.Create(seq_budget, &s) &&
		                d.Adopt(s, px) && d.Mark((tag + "_adopted").c_str(), s) &&
		                (len == 0 || (d.Decode(s, 8) && d.Mark((tag + "_decode8").c_str(), s))) &&
		                d.Prefill(s, pkv::Stream(52, 20, V)) && d.Decode(s, 4) && d.Mark((tag + "_prefill20+decode4").c_str(), s);
		if (s) d.Release(s);
		d.PrefixRelease(px);
		if (!ok) return;
	}
}

// ---- the GPU fill probe (§8, the two-sided form; G3) ---------------------------------------------

// A scene a cell can rebuild: the pool under probe and every handle the construction left live.
struct GScene {
	SslmGpuKvPoolHandle* pool = nullptr;
	std::vector<SslmGpuSequenceHandle*> seqs;
	std::vector<SslmGpuPrefixHandle*> prefixes;
	GScene() = default;
	GScene(const GScene&) = delete;
	GScene& operator=(const GScene&) = delete;
	~GScene() {
		for (SslmGpuSequenceHandle* s : seqs)
			if (s) sslm_gpu_seq_release(Ctx(), s);
		for (SslmGpuPrefixHandle* p : prefixes)
			if (p) sslm_gpu_prefix_release(Ctx(), p);
		if (pool) sslm_gpu_kv_pool_destroy(Ctx(), pool);
	}
	void DrainAll() {
		for (SslmGpuSequenceHandle* s : seqs)
			if (s) (void)DrainGpu(s);
	}
};
using GBuild = std::function<std::unique_ptr<GScene>()>;

// k split into creates of 2 to ceil(cap/B) pages each (§8).
inline std::vector<int64_t> SplitPages(const Fx& fx, int64_t k) {
	std::vector<int64_t> parts;
	if (k < 2) return parts;
	const int64_t n = (k + fx.CapPages() - 1) / fx.CapPages();
	for (int64_t i = 0; i < n; ++i) parts.push_back(k / n + (i < k % n ? 1 : 0));
	return parts;
}
// Budget (r - 1) * B reserves exactly r pages for 2 <= r <= ceil(cap/B), by §3.4 written here.
inline int32_t BudgetForPages(const Fx& fx, int64_t r) { return static_cast<int32_t>((r - 1) * fx.B); }

// Creates totalling `k` pages on `scene`'s pool; true when every one was admitted.
inline bool AllAdmitted(const Fx& fx, GScene& scene, int64_t k) {
	bool all = true;
	for (int64_t r : SplitPages(fx, k)) {
		SslmGpuSequenceHandle* s = nullptr;
		const SslmGpuStatus st = sslm_gpu_seq_create_budgeted(Ctx(), fx.gpu, scene.pool, BudgetForPages(fx, r), &s);
		if (st == GS::SSLM_OK && s) scene.seqs.push_back(s);
		else {
			all = false;
			G1A_CHECK_MSG(st == GS::SSLM_GPU_KV_POOL_EXHAUSTED, "fill probe create of %lld pages -> %s (want OK or POOL_EXHAUSTED)",
			              static_cast<long long>(r), GpuStatusName(st));
		}
	}
	return all;
}

// Exactly `k` pages are free on the built state: the admission leg on one build, the refusal leg
// on a fresh rebuild. Each reading is taken after sslm_gpu_ready drained every sequence.
inline bool GpuFillProbeExact(const Fx& fx, const GBuild& build, int64_t k, const std::string& what) {
	G1A_CHECK_MSG(k >= 2, "%s: the fill probe resolves k >= 2 only (k = %lld)", what.c_str(), static_cast<long long>(k));
	bool admit = false, refuse = false;
	{
		std::unique_ptr<GScene> a = build();
		if (a && a->pool) {
			a->DrainAll();
			admit = AllAdmitted(fx, *a, k);
		}
	}
	{
		std::unique_ptr<GScene> b = build();
		if (b && b->pool) {
			b->DrainAll();
			refuse = !AllAdmitted(fx, *b, k + 1);
		}
	}
	G1A_CHECK_MSG(admit, "%s: creates totalling %lld pages were not all admitted (fewer than %lld free)", what.c_str(),
	              static_cast<long long>(k), static_cast<long long>(k));
	G1A_CHECK_MSG(refuse, "%s: creates totalling %lld pages were all admitted on a rebuild (more than %lld free)", what.c_str(),
	              static_cast<long long>(k + 1), static_cast<long long>(k));
	return admit && refuse;
}

// The GPU pool stats, printed as a diagnostic only (§7's grading rule: no verdict reads it).
inline void PrintPoolStatsDiagnostic(const SslmGpuKvPoolHandle* pool, const char* what) {
	SslmGpuKvPoolStats st{};
	st.struct_size = sizeof st;
	if (sslm_gpu_kv_pool_stats(pool, &st) == GS::SSLM_OK)
		std::printf("  diagnostic %s: page_count=%u free_pages=%u shared_pages=%u B=%lld\n", what, st.page_count, st.free_pages,
		            st.shared_pages, static_cast<long long>(st.page_positions));
}

// A deterministic generator for the mixed-cycle cells.
struct Lcg {
	uint32_t x;
	explicit Lcg(uint32_t seed) : x(seed * 2654435761u + 1u) {}
	uint32_t Next() {
		x = x * 1664525u + 1013904223u;
		return x >> 8;
	}
};

}  // namespace g1a

#endif  // SUPERSLM_TESTS_G1A_HELPERS_H
