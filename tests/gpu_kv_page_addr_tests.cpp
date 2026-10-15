// Paged-KV plan (rev 16.1) step G2, host-side cell for the GPU K/V page-table addressing (§3.1, §4
// item 1). CPU-only: it runs on every leg, with no device, against
// src/gpu/shaders/kv_page_addr_model.h, the C++ transliteration of site_common.hlsli's
// KvInPageRowOffsetGpu/KvPageRowAddrGpu/KvRowAddrGpu and of context_accumulate_site.hlsl's page-run
// loop bounds.
//
// What it checks:
//   T0  the transliteration is the shader text: every `SSLM-KV-ADDR:<tag>` line reads the same in the
//       model header and in the shader file that carries the tag (whitespace collapsed, the model's
//       template parameter `U` read as HLSL's `uint`).
//   T1  the one-page identity (§3.1, §4 item 2): with table {0} and B = context_cap, the address of
//       every byte (layer, half, kv_head, pos, d) equals the flat address the pre-G2 shaders computed
//       (KvHalfOffsetGpu + KvRowOffsetWithinHalfGpu, the inline V-half base), over the tiny fixture
//       geometries and caps 1, 4 and 16.
//   T2  paged views at B = 1, 4 and 16 under a scrambled table: the GPU model's address equals the
//       CPU engine's own KvPageView accessors (forward_sites.cpp, KeyRow/ValueRow) for every row, and
//       the layout is a bijection onto the mapped pages' bytes.
//   T3  the page-run decomposition covers [0, width) exactly once, ascending, each run inside one
//       page, for B = 1, 4 and 16 and every width up to three pages and a part.
//   T4  no `uint` intermediate reaches 2^32 below §4 item 3's ceiling (page_count * page_bytes <= 2^32),
//       for the two shipped geometries at B = 16: every corner of the index space through the checked
//       32-bit type, and the run loop over a whole context; one page past the ceiling is flagged
//       (the instrument can fail).
//   T5  the uint32 and size_t instantiations agree wherever T4 says uint32 cannot wrap (the host
//       offset builders use size_t, the shaders `uint`).
#include "gpu/shaders/kv_page_addr_model.h"
#include "superslm/forward_sites.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SSLM_SOURCE_DIR
#error "gpu_kv_page_addr_tests needs SSLM_SOURCE_DIR (CMakeLists.txt sets it)"
#endif

namespace m = superslm_gpu::kv_page_addr_model;

static int GChecks = 0;
static int GFailures = 0;

#define CHECK_MSG(cond, ...)                                                \
	do {                                                                    \
		++GChecks;                                                          \
		if (!(cond)) {                                                      \
			++GFailures;                                                    \
			if (GFailures <= 400) {                                         \
				std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
				std::printf(__VA_ARGS__);                                   \
				std::printf("\n");                                          \
			}                                                               \
		}                                                                   \
	} while (0)

// ---------------------------------------------------------------------------------------------------
// The pre-G2 flat addressing, copied from site_common.hlsli and context_accumulate_site.hlsl as they
// stood before this step (KvHalfOffsetGpu, KvRowOffsetWithinHalfGpu, and the inline V-half base
// `kv_half_off + g_context_cap * g_num_kv_heads * g_head_dim`), in HLSL's uint arithmetic.
// ---------------------------------------------------------------------------------------------------
static uint32_t FlatKvHalfOffsetGpu(uint32_t layer, uint32_t context_cap, uint32_t num_kv_heads, uint32_t head_dim) {
	return (layer * context_cap * num_kv_heads * head_dim) * 2u;
}
static uint32_t FlatKvRowOffsetWithinHalfGpu(uint32_t context_cap, uint32_t head_dim, uint32_t kv_head,
                                             uint32_t position) {
	return kv_head * context_cap * head_dim + position * head_dim;
}
static uint32_t FlatAddr(uint32_t layer, uint32_t half, uint32_t context_cap, uint32_t num_kv_heads,
                         uint32_t head_dim, uint32_t kv_head, uint32_t pos) {
	const uint32_t kv_half_off = FlatKvHalfOffsetGpu(layer, context_cap, num_kv_heads, head_dim);
	const uint32_t v_half_off = kv_half_off + context_cap * num_kv_heads * head_dim;
	return (half == 0u ? kv_half_off : v_half_off) +
	       FlatKvRowOffsetWithinHalfGpu(context_cap, head_dim, kv_head, pos);
}

struct Geometry {
	uint32_t layers, kv_heads, head_dim;
};

// The tiny hand-built fixture geometries (tests/test_main.cpp's RunLayerLoop fixtures and §3.1's
// executed set, up to 4 layers x 3 heads x 6 dims).
static const Geometry kTinyGeometries[] = {
    {1, 1, 1}, {1, 1, 2}, {2, 1, 2}, {4, 1, 2}, {2, 2, 4}, {3, 2, 6}, {4, 3, 6}, {2, 3, 2},
};

// ---------------------------------------------------------------------------------------------------
// T0: the transliteration is the shader text.
// ---------------------------------------------------------------------------------------------------
static std::string ReadFile(const std::string& rel) {
	const std::string path = std::string(SSLM_SOURCE_DIR) + "/" + rel;
	std::ifstream in(path, std::ios::binary);
	if (!in) return std::string();
	std::stringstream ss;
	ss << in.rdbuf();
	return ss.str();
}

static std::string Normalize(std::string line) {
	const size_t c = line.find("//");
	if (c != std::string::npos) line.erase(c);
	std::string out;
	bool space = false;
	for (char ch : line) {
		if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') {
			space = !out.empty();
			continue;
		}
		if (space) out.push_back(' ');
		space = false;
		out.push_back(ch);
	}
	// The model's template parameter is HLSL's `uint`.
	std::string res;
	for (size_t i = 0; i < out.size(); ++i) {
		const bool word_start = i == 0 || !(std::isalnum(static_cast<unsigned char>(out[i - 1])) || out[i - 1] == '_');
		const bool word_end = i + 1 == out.size() ||
		                      !(std::isalnum(static_cast<unsigned char>(out[i + 1])) || out[i + 1] == '_');
		if (out[i] == 'U' && word_start && word_end) {
			res += "uint";
		} else {
			res.push_back(out[i]);
		}
	}
	return res;
}

static std::map<std::string, std::vector<std::string>> TaggedLines(const std::string& text) {
	std::map<std::string, std::vector<std::string>> out;
	std::istringstream in(text);
	std::string line;
	const std::string tag = "SSLM-KV-ADDR:";
	while (std::getline(in, line)) {
		const size_t at = line.find(tag);
		if (at == std::string::npos) continue;
		if (line.find("//") == std::string::npos || line.find("//") > at) continue;
		size_t e = at + tag.size();
		while (e < line.size() && (std::isalnum(static_cast<unsigned char>(line[e])) || line[e] == '-')) ++e;
		const std::string name = line.substr(at + tag.size(), e - at - tag.size());
		const std::string code = Normalize(line);
		if (!code.empty()) out[name].push_back(code);
	}
	return out;
}

static void TestT0_TransliterationIsTheShaderText() {
	const std::string model = ReadFile("src/gpu/shaders/kv_page_addr_model.h");
	CHECK_MSG(!model.empty(), "kv_page_addr_model.h readable under SSLM_SOURCE_DIR");
	const auto model_tags = TaggedLines(model);
	struct Want {
		const char* tag;
		const char* shader;
	};
	const Want wants[] = {
	    {"in-page", "src/gpu/shaders/site_common.hlsli"},
	    {"row", "src/gpu/shaders/site_common.hlsli"},
	    {"page-run-loop", "src/gpu/shaders/context_accumulate_site.hlsl"},
	    {"page-run-rows", "src/gpu/shaders/context_accumulate_site.hlsl"},
	};
	for (const Want& w : wants) {
		const std::string shader = ReadFile(w.shader);
		CHECK_MSG(!shader.empty(), "%s readable", w.shader);
		const auto shader_tags = TaggedLines(shader);
		const auto mi = model_tags.find(w.tag);
		const auto si = shader_tags.find(w.tag);
		CHECK_MSG(mi != model_tags.end() && mi->second.size() == 1,
		          "the model carries exactly one SSLM-KV-ADDR:%s line", w.tag);
		CHECK_MSG(si != shader_tags.end() && !si->second.empty(), "%s carries an SSLM-KV-ADDR:%s line",
		          w.shader, w.tag);
		if (mi == model_tags.end() || si == shader_tags.end() || mi->second.empty()) continue;
		for (const std::string& s : si->second) {
			CHECK_MSG(s == mi->second[0], "SSLM-KV-ADDR:%s differs:\n  model:  %s\n  shader: %s", w.tag,
			          mi->second[0].c_str(), s.c_str());
		}
	}
}

// ---------------------------------------------------------------------------------------------------
// T1: the one-page identity against the pre-G2 flat addresses.
// ---------------------------------------------------------------------------------------------------
static void TestT1_OnePageViewIsTheFlatLayout() {
	static const uint32_t kOnePage[1] = {0u};
	uint64_t compared = 0;
	for (const Geometry& g : kTinyGeometries) {
		for (uint32_t cap : {1u, 4u, 16u}) {
			const uint32_t page_bytes = g.layers * 2u * g.kv_heads * cap * g.head_dim;  // the one page
			uint32_t mismatches = 0;
			for (uint32_t l = 0; l < g.layers; ++l)
				for (uint32_t half = 0; half < 2u; ++half)
					for (uint32_t h = 0; h < g.kv_heads; ++h)
						for (uint32_t pos = 0; pos < cap; ++pos)
							for (uint32_t d = 0; d < g.head_dim; ++d) {
								const uint32_t got =
								    m::KvRowAddrGpu<uint32_t>(kOnePage, page_bytes, /*B=*/cap, l, half, g.kv_heads,
								                              g.head_dim, h, pos) +
								    d;
								const uint32_t want = FlatAddr(l, half, cap, g.kv_heads, g.head_dim, h, pos) + d;
								++compared;
								if (got != want) {
									if (mismatches++ < 3) {
										CHECK_MSG(got == want,
										          "one-page view L=%u H=%u D=%u cap=%u (l=%u half=%u h=%u pos=%u d=%u): "
										          "paged %u, flat %u",
										          g.layers, g.kv_heads, g.head_dim, cap, l, half, h, pos, d, got, want);
									}
								}
							}
			CHECK_MSG(mismatches == 0, "one-page view L=%u H=%u D=%u cap=%u: %u of the bytes mismatch", g.layers,
			          g.kv_heads, g.head_dim, cap, mismatches);
		}
	}
	std::printf("T1 one-page identity: %llu bytes compared\n", static_cast<unsigned long long>(compared));
}

// ---------------------------------------------------------------------------------------------------
// T2: paged views against the CPU engine's own accessors, and the bijection.
// ---------------------------------------------------------------------------------------------------
static void TestT2_PagedViewMatchesTheCpuEngineAndIsABijection() {
	for (const Geometry& g : kTinyGeometries) {
		for (uint32_t B : {1u, 4u, 16u}) {
			const uint32_t cap = 48u;  // B divides it at every B here
			const uint32_t pages = cap / B;
			const uint32_t page_bytes = g.layers * 2u * g.kv_heads * B * g.head_dim;
			// A scrambled table: logical page i -> physical page (i * 7 + 3) mod pages, a permutation
			// whenever gcd(7, pages) == 1, and a fixed reversal otherwise.
			std::vector<uint32_t> table(pages);
			bool perm = true;
			for (uint32_t i = 0; i < pages; ++i) table[i] = (i * 7u + 3u) % pages;
			{
				std::set<uint32_t> seen(table.begin(), table.end());
				perm = seen.size() == pages;
			}
			if (!perm)
				for (uint32_t i = 0; i < pages; ++i) table[i] = pages - 1u - i;
			std::vector<uint8_t> pool(static_cast<size_t>(pages) * page_bytes);
			superslm::KvPageView view{pool.data(), table.data(), pages, static_cast<int64_t>(B), page_bytes};
			std::vector<uint8_t> hit(pool.size(), 0);
			uint32_t mismatches = 0, out_of_range = 0, double_hit = 0;
			for (uint32_t l = 0; l < g.layers; ++l)
				for (uint32_t half = 0; half < 2u; ++half)
					for (uint32_t h = 0; h < g.kv_heads; ++h)
						for (uint32_t pos = 0; pos < cap; ++pos) {
							const uint32_t gpu = m::KvRowAddrGpu<uint32_t>(table.data(), page_bytes, B, l, half,
							                                               g.kv_heads, g.head_dim, h, pos);
							const int8_t* cpu_row =
							    half == 0u ? superslm::KeyRow(view, l, g.kv_heads, g.head_dim, h, pos)
							               : superslm::ValueRow(view, l, g.kv_heads, g.head_dim, h, pos);
							const size_t cpu = static_cast<size_t>(reinterpret_cast<const uint8_t*>(cpu_row) -
							                                       pool.data());
							if (cpu != gpu) ++mismatches;
							for (uint32_t d = 0; d < g.head_dim; ++d) {
								const size_t a = static_cast<size_t>(gpu) + d;
								if (a >= hit.size()) {
									++out_of_range;
									continue;
								}
								if (hit[a]++ != 0) ++double_hit;
							}
						}
			const size_t covered = static_cast<size_t>(std::count(hit.begin(), hit.end(), 1));
			CHECK_MSG(mismatches == 0, "B=%u L=%u H=%u D=%u: %u rows where the GPU model and the CPU view differ", B,
			          g.layers, g.kv_heads, g.head_dim, mismatches);
			CHECK_MSG(out_of_range == 0 && double_hit == 0 && covered == pool.size(),
			          "B=%u L=%u H=%u D=%u: not a bijection onto the pool (%u out of range, %u bytes hit twice, %zu "
			          "of %zu covered)",
			          B, g.layers, g.kv_heads, g.head_dim, out_of_range, double_hit, covered, pool.size());
		}
	}
}

// ---------------------------------------------------------------------------------------------------
// T3: the page-run decomposition covers [0, width) exactly once.
// ---------------------------------------------------------------------------------------------------
static void TestT3_PageRunsCoverTheRowExactlyOnce() {
	for (uint32_t B : {1u, 4u, 16u}) {
		for (uint32_t width = 0; width <= 3u * B + B / 2u + 1u; ++width) {
			std::vector<uint32_t> count(width, 0);
			uint32_t next = 0, bad_page = 0, bad_order = 0, runs = 0;
			m::ForEachKvPageRunGpu<uint32_t>(width, B, [&](uint32_t p0, uint32_t rows) {
				++runs;
				if (p0 != next) ++bad_order;
				if (rows == 0u || rows > B || p0 % B != 0u || (p0 + rows - 1u) / B != p0 / B) ++bad_page;
				for (uint32_t r = 0; r < rows; ++r)
					if (p0 + r < width) ++count[p0 + r];
					else ++bad_page;
				next = p0 + rows;
			});
			uint32_t not_once = 0;
			for (uint32_t c : count)
				if (c != 1u) ++not_once;
			CHECK_MSG(not_once == 0 && bad_page == 0 && bad_order == 0 && next == width &&
			              runs == (width + B - 1u) / B,
			          "B=%u width=%u: %u positions not covered exactly once, %u runs off a page, %u out of order, "
			          "end %u, %u runs",
			          B, width, not_once, bad_page, bad_order, next, runs);
		}
	}
}

// ---------------------------------------------------------------------------------------------------
// T4: no uint intermediate reaches 2^32 below the §4 item 3 ceiling.
// ---------------------------------------------------------------------------------------------------
struct Checked32 {
	uint64_t v = 0;
	static bool overflowed;
	constexpr Checked32() = default;
	Checked32(uint64_t x) : v(x) { Note(); }  // NOLINT: implicit, as HLSL's literals convert
	void Note() const {
		if (v >= (UINT64_C(1) << 32)) overflowed = true;
	}
	friend Checked32 operator*(Checked32 a, Checked32 b) { return Checked32(a.v * b.v); }
	friend Checked32 operator+(Checked32 a, Checked32 b) { return Checked32(a.v + b.v); }
	friend Checked32 operator-(Checked32 a, Checked32 b) {
		if (b.v > a.v) overflowed = true;
		return Checked32(a.v - b.v);
	}
	friend Checked32 operator/(Checked32 a, Checked32 b) { return Checked32(a.v / b.v); }
	friend Checked32 operator%(Checked32 a, Checked32 b) { return Checked32(a.v % b.v); }
	friend bool operator<(Checked32 a, Checked32 b) { return a.v < b.v; }
	Checked32& operator+=(Checked32 b) { return *this = *this + b; }
	explicit operator uint64_t() const { return v; }
};
bool Checked32::overflowed = false;

struct ShippedGeometry {
	const char* name;
	uint32_t layers, kv_heads, head_dim, context_cap;
	uint32_t ceiling_pages;  // §4 item 3's executed ceiling at B = 16
};

static const ShippedGeometry kShipped[] = {
    {"0.5B", 24, 2, 64, 32768, 43690},
    {"1.5B", 28, 2, 128, 32768, 18724},
};

// Every corner of the index space (each index at 0 or its maximum) for physical page `phys`, plus the
// in-row byte offset at its maximum: the formula is a sum of products of non-negative factors that
// are all at least 1, so every intermediate is maximal at a corner.
static bool AnyCornerOverflows(const ShippedGeometry& g, uint32_t B, uint32_t page_bytes, uint32_t max_phys_page) {
	Checked32::overflowed = false;
	const uint32_t table[1] = {max_phys_page};
	for (int mask = 0; mask < 32; ++mask) {
		const uint32_t l = (mask & 1) ? g.layers - 1u : 0u;
		const uint32_t half = (mask & 2) ? 1u : 0u;
		const uint32_t h = (mask & 4) ? g.kv_heads - 1u : 0u;
		const uint32_t pos = (mask & 8) ? B - 1u : 0u;
		const uint32_t d = (mask & 16) ? g.head_dim - 1u : 0u;
		const Checked32 row = m::KvRowAddrGpu<Checked32>(table, page_bytes, B, l, half, g.kv_heads, g.head_dim, h, pos);
		const Checked32 byte = row + Checked32(d);
		(void)byte;
	}
	return Checked32::overflowed;
}

static void TestT4_NoUintOverflowBelowTheCeiling() {
	const uint32_t B = 16u;
	for (const ShippedGeometry& g : kShipped) {
		const uint64_t page_bytes64 = static_cast<uint64_t>(g.layers) * 2u * g.kv_heads * B * g.head_dim;
		const uint32_t page_bytes = static_cast<uint32_t>(page_bytes64);
		const uint64_t ceiling = (UINT64_C(1) << 32) / page_bytes64;  // page_count * page_bytes <= 2^32
		CHECK_MSG(ceiling == g.ceiling_pages, "%s: ceiling %llu pages, the plan's executed figure %u", g.name,
		          static_cast<unsigned long long>(ceiling), g.ceiling_pages);
		CHECK_MSG(!AnyCornerOverflows(g, B, page_bytes, static_cast<uint32_t>(ceiling - 1u)),
		          "%s: an intermediate reaches 2^32 inside a pool of %llu pages (the ceiling)", g.name,
		          static_cast<unsigned long long>(ceiling));
		// Vitality: one page past the ceiling, the last page's last byte is at or past 2^32.
		CHECK_MSG(AnyCornerOverflows(g, B, page_bytes, static_cast<uint32_t>(ceiling)),
		          "%s: one page past the ceiling is not flagged -- the instrument cannot fail", g.name);
		// The run loop over a whole context: p0 + B and width - p0 never wrap.
		Checked32::overflowed = false;
		uint64_t total_rows = 0;
		m::ForEachKvPageRunGpu<Checked32>(Checked32(g.context_cap), Checked32(B),
		                                  [&](Checked32, Checked32 rows) { total_rows += rows.v; });
		CHECK_MSG(!Checked32::overflowed && total_rows == g.context_cap,
		          "%s: the page-run loop over width = %u wraps or miscounts (%llu rows)", g.name, g.context_cap,
		          static_cast<unsigned long long>(total_rows));
	}
}

// ---------------------------------------------------------------------------------------------------
// T5: the host's size_t instantiation agrees with the shaders' uint where uint cannot wrap.
// ---------------------------------------------------------------------------------------------------
static void TestT5_HostAndShaderInstantiationsAgree() {
	uint32_t mismatches = 0;
	for (const ShippedGeometry& g : kShipped) {
		const uint32_t B = 16u;
		const uint32_t page_bytes = g.layers * 2u * g.kv_heads * B * g.head_dim;
		for (uint32_t phys : {0u, 1u, 77u, g.ceiling_pages - 1u}) {
			const uint32_t table32[1] = {phys};
			for (uint32_t l : {0u, g.layers / 2u, g.layers - 1u})
				for (uint32_t half = 0; half < 2u; ++half)
					for (uint32_t h = 0; h < g.kv_heads; ++h)
						for (uint32_t pos : {0u, 5u, B - 1u}) {
							const uint32_t a32 =
							    m::KvRowAddrGpu<uint32_t>(table32, page_bytes, B, l, half, g.kv_heads, g.head_dim, h, pos);
							const size_t a64 = m::KvRowAddrGpu<size_t>(table32, page_bytes, B, l, half, g.kv_heads,
							                                            g.head_dim, h, pos);
							if (a64 != a32) ++mismatches;
						}
		}
	}
	CHECK_MSG(mismatches == 0, "%u addresses where uint32 and size_t disagree below the ceiling", mismatches);
}

int main() {
	TestT0_TransliterationIsTheShaderText();
	TestT1_OnePageViewIsTheFlatLayout();
	TestT2_PagedViewMatchesTheCpuEngineAndIsABijection();
	TestT3_PageRunsCoverTheRowExactlyOnce();
	TestT4_NoUintOverflowBelowTheCeiling();
	TestT5_HostAndShaderInstantiationsAgree();
	std::printf("gpu_kv_page_addr_tests: %d checks, %d failures\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}
