// Paged-KV plan rev 16.21 §23.8, step T0: the R9 red suite's surface and shared harness.
//
// What R9 changes (plan §3.1, §21, §23.8): T1 derives the page size B from (head_dim, cap, element
// width) and rotates each run's slot by the logical page; T2 removes the one-page fallback (a
// partial last page at a cap B does not divide); T3 adds the contiguous pool option. At 2ebe5e3 B is
// CFG1's kv_block_size when it divides the cap and the cap otherwise, nothing rotates, and the
// option does not exist, so every cell here is red on 2ebe5e3 by construction of what it asserts.
//
// This file carries:
//   - the R9 surface, declared as the plan states it (§3.6's table, §3.1's pin). The test author's
//     readings where the plan leaves a field open are named at each declaration; the builder keeps
//     them or changes this file and the cells together, never the assertions;
//   - §3.1's rule and the address formula, written here from the plan's text, with the mutants the
//     cells name, so each cell's separation of each named mutant is executed at the cell's own
//     construction (the mutant's value substituted for the engine's observable);
//   - fixtures: the generated ones (tools/gen_paged_kv_fixture.py, tests/paged-kv/r9_gen_fixtures.py)
//     and variants made from them by patching CFG1 (cap, kv_precision, kv_block_size) and, for the
//     geometry-only cells, stripping the RoPE table (which sslm_model_map does not need: ROP1 is
//     optional at load and BuildEngineCache reads only WGT1 and the composition constants);
//   - an independent twin for fixtures that have no committed reference: the engine's flat layer
//     loops over the test's own one-page buffer (pkv_engine_helpers.h, B = cap, table {0}), which
//     reach neither the page module, the ABI holders nor the derived B.
//
// Every verdict reads only statuses, admission counts against pools the test sizes from its own
// formulas, tokens, blob bytes, and raw pool bytes through the peek seams addressed by the formula
// written here (plan §7's grading rule). sslm_kv_pool_stats and sslm_seq_kv_stats are recorded as
// diagnostics only, except where a cell's subject is the stats verb itself.

#ifndef SUPERSLM_TESTS_PKV_R9_H
#define SUPERSLM_TESTS_PKV_R9_H

#include "pkv_budget_c_helpers.h"
#include "pkv_common.h"
#include "pkv_engine_helpers.h"
#include "pkv_legacy_a_helpers.h"

#include "superslm/artifact.h"
#include "superslm/sha256.h"

#include "sslm_cfg1_hostile_fixtures.h"
#include "sslm_fixtures.h"
#include "sslm_sil1_hostile_fixtures.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

extern "C" {

// §3.6 / §21.2: the bytes a contiguous pool of `block_count` blocks needs: blocks of `cap` positions,
// `cap * L*2*H_kv*D*elem` bytes each, plus the page module's overhead for `block_count` pages
// (8 * block_count + 32). 0 on a null model, block_count 0, or when the product or the sum overflows.

// §3.6 / §21.2: a pool whose page is `cap` positions. Refusals in sslm_kv_pool_create's order: null
// out, null model or buffer, block_count 0 -> SSLM_INVALID_ARGUMENT; an overflowing size ->
// SSLM_INVALID_ARGUMENT; buf_size below the size -> SSLM_BUFFER_TOO_SMALL; a misaligned buffer ->
// SSLM_MISALIGNED_BUFFER; heap failure -> SSLM_ALLOCATION_FAILED.
size_t sslm_kv_pool_contiguous_size(sslm_model model, uint32_t block_count);
sslm_status sslm_kv_pool_create_contiguous(sslm_model model, void* buf, size_t buf_size, uint32_t block_count,
                                           sslm_kv_pool* out);

// §3.1: the test-only pin, compiled only with SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS. The plan names it
// `KvPagesTestPinPagePositions(model, int64_t B)` and says what it refuses; the test author's
// reading of the rest: C linkage (so cell 4.12's symbol scan names one symbol), and an sslm_status
// result, SSLM_OK when the pin took and SSLM_INVALID_ARGUMENT for every refusal (a B that neither
// divides the cap nor is the cap, B <= 0, B > cap, a pin after a pool exists on the model).
sslm_status KvPagesTestPinPagePositions(sslm_model model, int64_t page_positions);

// The geometry seam (rev 16.23, D-SLM8437; §3.1): C linkage, compiled only with
// SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS beside the pin. It loads `artifact` with SslmModel::Load, never maps it, and
// writes what the geometry and sizing verbs would return for it, computed through the verbs' one internal function:
// page_positions (sslm_kv_page_positions), page_size (sslm_kv_page_size), block_size (sslm_kv_block_size), overhead
// (sslm_kv_pool_overhead_size(block_count)), page_count_fits (block_count * ceil(cap/B) <= UINT32_MAX) and
// requirement_fits (block_count * block_size + overhead <= SIZE_MAX). SSLM_INVALID_ARGUMENT on a null `out`; on a
// refused artifact, the load's refusal. The test author's readings, where the plan names the fields but not their
// layout: the field order and types below, and SSLM_ARTIFACT_REJECTED for a refused load (the status sslm_model_map
// gives the same artifact).
typedef struct KvPagesTestPoolGeometryOut {
	int64_t page_positions;
	size_t page_size;
	size_t block_size;
	size_t overhead;
	int32_t page_count_fits;   // 1 or 0
	int32_t requirement_fits;  // 1 or 0
} KvPagesTestPoolGeometryOut;
sslm_status KvPagesTestPoolGeometry(const void* artifact, size_t size, uint32_t block_count, KvPagesTestPoolGeometryOut* out);

// G19's block-index peek (src/sslm_abi.cpp, every library build): n raw bytes of block `block_index`.
sslm_status sslm_g5_test_only_peek_kv_block_bytes(sslm_kv_pool pool, uint32_t block_index, uint8_t* out_buf, size_t n);

}  // extern "C"

namespace pkv {
namespace r9 {

// ---- §3.1's rule, written here -------------------------------------------------------------------

// The mutants the cells name (4.10, 4.4, 10.9, 2.9b, 2.9c). kShipped is 2ebe5e3's rule: CFG1's
// kv_block_size when it divides the cap, else the cap.
enum class RuleMutant {
	kNone,
	kNoCapTerm,     // the run target is 8 KiB everywhere (4.10)
	kNoFloor,       // no B >= 16 floor (4.10)
	kNoElem,        // B * D >= S, the element width dropped (4.10)
	kU32Product,    // the rule's loop product formed in 32 bits (4.10)
	kNoPow2,        // the smallest B >= 16 with B * D * elem >= S, not a power of two (4.10)
	kReadsCfg1,     // B = CFG1 kv_block_size where it is at most the cap (4.10, 4.11)
	kNoClamp,       // no B = cap below one rule page (4.10)
	kClampAt2B,     // the clamp fires at 2B > cap (4.10's boundary row)
	kFallback,      // one page at a cap B does not divide (4.4, 4.10, 10.9)
	kShipped,       // 2ebe5e3: kv_block_size if it divides the cap, else the cap
};

inline const char* RuleMutantName(RuleMutant m) {
	switch (m) {
		case RuleMutant::kNone: return "the rule";
		case RuleMutant::kNoCapTerm: return "no cap term";
		case RuleMutant::kNoFloor: return "no floor";
		case RuleMutant::kNoElem: return "elem dropped";
		case RuleMutant::kU32Product: return "32-bit product";
		case RuleMutant::kNoPow2: return "no power-of-two rounding";
		case RuleMutant::kReadsCfg1: return "reads CFG1 kv_block_size";
		case RuleMutant::kNoClamp: return "no clamp below one page";
		case RuleMutant::kClampAt2B: return "clamp at 2B > cap";
		case RuleMutant::kFallback: return "one-page fallback";
		case RuleMutant::kShipped: return "2ebe5e3's rule";
	}
	return "?";
}

// §3.1: S = 4 KiB when D <= 64 and cap <= 4,096, else 8 KiB; B = the smallest power of two with
// B >= 16 and B * D * elem >= S; B = cap when B > cap. `kvbs` is CFG1's kv_block_size, read only by
// the mutants that read it. Returns 0 where a mutant's arithmetic wraps to nothing.
inline int64_t RuleB(uint64_t D, uint64_t cap, uint64_t elem, RuleMutant m = RuleMutant::kNone, uint64_t kvbs = 16) {
	if (m == RuleMutant::kShipped)
		return static_cast<int64_t>((kvbs >= 1 && kvbs <= cap && cap % kvbs == 0) ? kvbs : cap);
	if (m == RuleMutant::kReadsCfg1) return static_cast<int64_t>(kvbs >= 1 && kvbs <= cap ? kvbs : cap);
	const uint64_t S = (m != RuleMutant::kNoCapTerm && D <= 64 && cap <= 4096) ? 4096 : 8192;
	const uint64_t e = m == RuleMutant::kNoElem ? 1 : elem;
	const uint64_t floor_b = m == RuleMutant::kNoFloor ? 1 : 16;
	uint64_t B = 0;
	if (m == RuleMutant::kNoPow2) {
		B = std::max<uint64_t>(floor_b, (S + D * e - 1) / (D * e));
	} else if (m == RuleMutant::kU32Product) {
		uint32_t b = static_cast<uint32_t>(floor_b);
		while (b != 0 && static_cast<uint32_t>(b * static_cast<uint32_t>(D) * static_cast<uint32_t>(e)) < S) b *= 2;
		B = b;
	} else {
		B = floor_b;
		while (B * D * e < S) B *= 2;
	}
	if (m == RuleMutant::kClampAt2B) {
		if (2 * B > cap) B = cap;
	} else if (m != RuleMutant::kNoClamp && B > cap) {
		B = cap;
	}
	if (m == RuleMutant::kFallback && B != 0 && cap % B != 0) B = cap;
	return static_cast<int64_t>(B);
}

inline int64_t CeilDiv(int64_t a, int64_t b) { return b > 0 ? (a + b - 1) / b : 0; }
// Pages of a whole-cap holder (§3.1): ceil(cap / B). The floor mutant (10.9, 4.10) counts floor.
inline int64_t CapPages(int64_t cap, int64_t B, bool floor_mutant = false) {
	return B <= 0 ? 0 : floor_mutant ? cap / B : CeilDiv(cap, B);
}
inline int64_t Slack(int64_t cap, int64_t B) { return CapPages(cap, B) * B - cap; }
// §3.4: R(budget) = min(ceil(budget / B) + 1, ceil(cap / B)).
inline int64_t Reservation(int64_t budget, int64_t cap, int64_t B) {
	return std::min<int64_t>(CeilDiv(budget, B) + 1, CapPages(cap, B));
}

// The page geometry the test derives for a model: element width from CFG1, B by the rule (or a pin).
// Rev 16.22 (D-SLM8427) separates two quantities. The flat block, `block_bytes`, is 1.11.0's
// L*cap*H_kv*D*2*elem at every cap: sslm_kv_block_size and the legacy save's K/V section. A legacy
// holder's ceil(cap/B) pages hold `slack` positions more, and sslm_kv_pool_overhead_size carries them:
// Overhead(N) = 8N + 32 + N * slack * q. So N blocks need N * pool_block_bytes + 8N + 32 bytes.
struct Lay {
	int64_t cap = 0, B = 0, pages = 0, slack = 0;
	uint32_t runs = 0;
	size_t elem = 1, bytes_per_position = 0, page_bytes = 0, block_bytes = 0, pool_block_bytes = 0,
	       contiguous_block_bytes = 0;
	size_t Overhead(uint64_t n) const { return static_cast<size_t>(8 * n + 32 + n * slack * bytes_per_position); }
	size_t PoolBytes(uint64_t n) const { return static_cast<size_t>(n * block_bytes) + Overhead(n); }
};

inline Lay LayoutOf(const Geometry& g, size_t elem, int64_t B) {
	Lay y;
	y.cap = g.context_cap;
	y.B = B;
	y.pages = CapPages(g.context_cap, B);
	y.slack = Slack(g.context_cap, B);
	y.runs = g.layers * 2u * g.kv_heads;
	y.elem = elem;
	y.bytes_per_position = size_t{g.layers} * 2u * g.kv_heads * g.head_dim * elem;
	y.page_bytes = y.bytes_per_position * static_cast<size_t>(B);
	y.block_bytes = y.bytes_per_position * static_cast<size_t>(g.context_cap);  // 1.11.0's block (D-SLM8427)
	y.pool_block_bytes = y.page_bytes * static_cast<size_t>(y.pages);          // ceil(cap/B) pages
	y.contiguous_block_bytes = y.block_bytes;
	return y;
}

// ---- the address formula, written here (§3.1) ----------------------------------------------------

// Slot of run r = (l*2 + half)*H_kv + h on logical page j: (r + j) mod runs, and the mutants 6.6a
// names: (r + (j & 63)) mod runs, (r + j) & (runs - 1), (r + j) mod pages, and a slot keyed on the
// physical page id; kNone is 2ebe5e3's unrotated slot r (7.9-MR-OPT's layout).
enum class Rot { kRotated, kNone, kJAnd63, kAndMask, kModPages, kPhysical };

inline const char* RotName(Rot m) {
	switch (m) {
		case Rot::kRotated: return "(r + j) mod runs";
		case Rot::kNone: return "unrotated (slot r)";
		case Rot::kJAnd63: return "(r + (j & 63)) mod runs";
		case Rot::kAndMask: return "(r + j) & (runs - 1)";
		case Rot::kModPages: return "(r + j) mod pages";
		case Rot::kPhysical: return "slot keyed on the physical page id";
	}
	return "?";
}

inline uint64_t Slot(uint64_t r, uint64_t j, uint64_t runs, Rot m, uint64_t physical = 0, uint64_t pages = 1) {
	switch (m) {
		case Rot::kRotated: return (r + j) % runs;
		case Rot::kNone: return r;
		case Rot::kJAnd63: return (r + (j & 63)) % runs;
		case Rot::kAndMask: return (r + j) & (runs - 1);
		case Rot::kModPages: return (r + j) % (pages ? pages : 1);
		case Rot::kPhysical: return (r + physical) % runs;
	}
	return r;
}

// Byte offset of (l, half, h, pos) inside its page, at page size B.
inline size_t InPage(const Geometry& g, size_t elem, int64_t B, uint32_t l, uint32_t half, uint32_t h, int64_t pos,
                     Rot m = Rot::kRotated, uint64_t physical = 0, uint64_t pages = 1) {
	// the formula itself is the suites' one helper (pkv_common.h, D-SLM8449); the R9 cells name the layout explicitly
	if (m == Rot::kRotated || m == Rot::kNone)
		return PageRunOffset(g.layers, g.kv_heads, size_t{g.head_dim} * elem, B, l, half, h, pos, m == Rot::kRotated);
	const uint64_t runs = uint64_t{g.layers} * 2u * g.kv_heads;
	const uint64_t r = (uint64_t{l} * 2u + half) * g.kv_heads + h;
	const uint64_t j = static_cast<uint64_t>(pos / B);
	const uint64_t slot = Slot(r, j, runs, m, physical, pages);
	return static_cast<size_t>((slot * static_cast<uint64_t>(B) + static_cast<uint64_t>(pos % B)) * g.head_dim * elem);
}

// The rows [0, L) of a holder whose mapped table is `table`, read from the pool's raw pages through
// the page-index peek at page size B under address formula `m`, in pkv_scenarios.h's position-major
// order (so a digest compares with a reference record's rows_sha or a twin's). False on a seam
// refusal or a table that does not cover L.
inline bool RowsFromTable(sslm_kv_pool pool, const std::vector<uint32_t>& table, const Geometry& g, size_t elem,
                          int64_t B, int64_t L, Rot m, std::vector<uint8_t>* rows) {
	rows->clear();
	if (B <= 0 || static_cast<int64_t>(table.size()) * B < L) return false;
	const Lay y = LayoutOf(g, elem, B);
	const size_t rb = size_t{g.head_dim} * elem;
	std::map<uint32_t, std::vector<uint8_t>> pages;
	for (int64_t p = 0; p < L; ++p) {
		const uint32_t page = table[static_cast<size_t>(p / B)];
		auto it = pages.find(page);
		if (it == pages.end()) {
			it = pages.emplace(page, legacy_a::PeekPage(pool, page, y.page_bytes)).first;
			if (it->second.empty()) return false;
		}
		for (uint32_t l = 0; l < g.layers; ++l)
			for (uint32_t half = 0; half < 2; ++half)
				for (uint32_t h = 0; h < g.kv_heads; ++h) {
					const size_t at = InPage(g, elem, B, l, half, h, p, m, page, static_cast<uint64_t>(y.pages));
					if (at + rb > it->second.size()) return false;
					rows->insert(rows->end(), it->second.begin() + static_cast<std::ptrdiff_t>(at),
					             it->second.begin() + static_cast<std::ptrdiff_t>(at + rb));
				}
	}
	return true;
}

inline bool SeqRows(sslm_kv_pool pool, sslm_seq s, const Geometry& g, size_t elem, int64_t B, int64_t L, Rot m,
                    std::vector<uint8_t>* rows) {
	std::vector<uint32_t> table;
	if (!legacy_a::SeqTable(s, &table)) return false;
	return RowsFromTable(pool, table, g, elem, B, L, m, rows);
}

// ---- fixtures --------------------------------------------------------------------------------

// CFG1 field offsets inside the Config section (model.cpp's ParseConfig).
constexpr size_t kCfgHeadDim = 24, kCfgCap = 36, kCfgKvPrecision = 44, kCfgKvBlockSize = 48;

inline bool MapInto(std::vector<uint8_t> b, const std::string& stem, const Geometry& geo, Fixture* out,
                    sslm_status* map_status = nullptr) {
	out->stem = stem;
	out->geo = geo;
	out->size = b.size();
	out->bytes.Reset(b.size());
	std::memcpy(out->bytes.p, b.data(), b.size());
	out->sha = Sha(b.data(), b.size());
	const sslm_status st = sslm_model_map(out->bytes.p, out->size, &out->model);
	if (map_status) *map_status = st;
	out->ok = st == SSLM_OK;
	return out->ok;
}

// One of r9_gen_fixtures.py's files (or a generated one), by stem, from SUPERSLM_PAGED_KV_FIXTURE_DIR,
// loaded once. A missing file fails the cell.
inline const Fixture& Get(const char* stem, Geometry geo) {
	static std::map<std::string, std::unique_ptr<Fixture>> cache;
	auto it = cache.find(stem);
	if (it != cache.end()) {
		PKV_CHECK_MSG(it->second->ok, "fixture %s did not load (see its first failure)", stem);
		return *it->second;
	}
	auto f = std::make_unique<Fixture>();
	const char* dir = std::getenv("SUPERSLM_PAGED_KV_FIXTURE_DIR");
	PKV_CHECK_MSG(dir && *dir, "SUPERSLM_PAGED_KV_FIXTURE_DIR is not set; run tests/paged-kv/r9_gen_fixtures.py DIR");
	std::vector<uint8_t> raw;
	if (dir && *dir && ReadAll(std::string(dir) + "/" + stem + ".sslm", &raw) && !raw.empty()) {
		sslm_status st = SSLM_OK;
		PKV_CHECK_MSG(MapInto(std::move(raw), stem, geo, f.get(), &st), "sslm_model_map(%s) == %s", stem,
		              AbiStatusName(st));
	} else {
		PKV_CHECK_MSG(false, "fixture %s/%s.sslm not found; run tests/paged-kv/r9_gen_fixtures.py",
		              dir ? dir : "(unset)", stem);
	}
	const Fixture& ref = *f;
	cache[stem] = std::move(f);
	return ref;
}

// The R9 fixtures (r9_gen_fixtures.py's ROWS; two layers and two KV heads unless named).
inline const Fixture& D2Cap8() { return Get("r9_d2_c8", Geometry{2, 2, 2, 8}); }
inline const Fixture& D64Cap(int cap) {
	const std::string stem = "r9_d64_c" + std::to_string(cap);
	return Get(stem.c_str(), Geometry{2, 2, 64, cap});
}
inline const Fixture& D512() { return Get("r9_d512_c64", Geometry{2, 2, 512, 64}); }
inline const Fixture& D1024() { return Get("r9_d1024_c64", Geometry{2, 2, 1024, 64}); }
inline const Fixture& L3() { return Get("r9_l3_c4096", Geometry{3, 2, 48, 4096}); }
inline const Fixture& Q8() { return Get("r9_q8_c64", Geometry{36, 8, 128, 64}); }

// A copy of `base` with CFG1 fields patched (any field < 0 is kept) and, when `strip_rope`, the RoPE
// table dropped and the artifact re-laid-out; the integrity hash is recomputed and the header flags
// kept. The weights and every other section are byte-for-byte the base's.
struct Variant {
	int64_t cap = -1;
	int64_t kv_block_size = -1;
	int64_t kv_precision = -1;  // 0 Int8, 1 Int16
	bool strip_rope = false;
};

inline bool VariantBytes(const Fixture& base, const Variant& v, std::vector<uint8_t>* out) {
	if (!base.ok) return false;
	const uint8_t* p = base.bytes.p;
	const size_t n = base.size;
	uint32_t count = 0, flags = 0;
	std::memcpy(&count, p + 12, 4);
	std::memcpy(&flags, p + 16, 4);
	std::vector<superslm_test::FixtureSection> sections;
	bool patched = false;
	for (uint32_t i = 0; i < count; ++i) {
		const uint8_t* row = p + superslm::kHeaderBytes + size_t{i} * superslm::kSectionDescBytes;
		uint32_t type = 0, dtype = 0, align = 0;
		uint64_t off = 0, size = 0;
		std::memcpy(&type, row, 4);
		std::memcpy(&dtype, row + 4, 4);
		std::memcpy(&off, row + 8, 8);
		std::memcpy(&size, row + 16, 8);
		std::memcpy(&align, row + 32, 4);
		if (off + size > n) return false;
		if (v.strip_rope && type == static_cast<uint32_t>(superslm::SslmSectionType::RopeTables)) continue;
		superslm_test::FixtureSection s;
		s.type = type;
		s.dtype = dtype;
		s.alignment = align;
		s.data.assign(p + off, p + off + size);
		if (type == static_cast<uint32_t>(superslm::SslmSectionType::Config) && s.data.size() >= kCfgKvBlockSize + 4) {
			auto put = [&](size_t at, int64_t value) {
				if (value < 0) return;
				const uint32_t u = static_cast<uint32_t>(value);
				std::memcpy(s.data.data() + at, &u, 4);
			};
			put(kCfgCap, v.cap);
			put(kCfgKvPrecision, v.kv_precision);
			put(kCfgKvBlockSize, v.kv_block_size);
			patched = true;
		}
		sections.push_back(std::move(s));
	}
	if (!patched) return false;
	*out = superslm_test::BuildArtifact(sections).bytes;
	superslm_test::PutU32(*out, 16, flags);
	superslm_test::RecomputeIntegrityHash(*out);
	return true;
}

// Maps the variant into `out` (geometry: the base's, with the patched cap). False, with the map
// status in *st, when the variant does not map.
inline bool MakeVariant(const Fixture& base, const Variant& v, Fixture* out, sslm_status* st = nullptr) {
	std::vector<uint8_t> b;
	if (!VariantBytes(base, v, &b)) {
		if (st) *st = SSLM_INVALID_ARGUMENT;
		return false;
	}
	Geometry g = base.geo;
	if (v.cap >= 0) g.context_cap = v.cap;
	std::string stem = base.stem;
	if (v.cap >= 0) stem += "_cap" + std::to_string(v.cap);
	if (v.kv_block_size >= 0) stem += "_kvbs" + std::to_string(v.kv_block_size);
	if (v.kv_precision >= 0) stem += "_prec" + std::to_string(v.kv_precision);
	if (v.strip_rope) stem += "_norope";
	return MapInto(std::move(b), stem, g, out, st);
}

// A CFG1-and-SigmoidLut-only artifact, the shape cells 2.9d, 4.10's hostile head_dim and 4.10's
// generality cell specify ("CFG1 and the sigmoid table only, the sections OpenFromMemory requires").
inline std::vector<uint8_t> Cfg1OnlyArtifact(const superslm_test::Cfg1Spec& spec) {
	return superslm_test::BuildArtifact({superslm_test::MakeSection(superslm::SslmSectionType::Config,
	                                                                superslm::SslmDtype::Raw,
	                                                                superslm_test::BuildCfg1(spec)),
	                                     superslm_test::MakeSigmoidLutSection()})
	    .bytes;
}

// The SslmModel::Load status of a byte image (the loader the cells cite), and its message.
inline superslm::SslmModelStatus LoadStatus(const std::vector<uint8_t>& b, std::string* err) {
	superslm::SslmModelView view;
	return superslm::SslmModel::Load(b.data(), b.size(), view, err);
}

// ---- pools and admission -----------------------------------------------------------------------

// A page pool of `pages` pages whose buffer is sized from the TEST's page bytes (plus the engine's
// overhead, which no verdict reads). A build whose page is larger than the rule's refuses the create
// with SSLM_BUFFER_TOO_SMALL, which the cell sees as a status.
struct PagePoolOf {
	AlignedBuf mem;
	sslm_kv_pool pool = nullptr;
	sslm_status status = SSLM_INVALID_ARGUMENT;
	PagePoolOf(sslm_model model, uint32_t pages, size_t page_bytes, uint8_t fill = 0) {
		mem.Reset(page_bytes * pages + sslm_kv_page_pool_overhead_size(model, pages), fill);
		status = sslm_kv_page_pool_create(model, mem.p, mem.n, pages, &pool);
	}
	~PagePoolOf() {
		if (pool) sslm_kv_pool_destroy(pool);
	}
};

// A legacy block pool of `blocks` blocks in exactly the bytes the test computes for it:
// N * block + Overhead(N) (rev 16.22: the pages ceil(cap/B) * page_bytes per block, plus 8N + 32).
struct BlockPoolOf {
	AlignedBuf mem;
	sslm_kv_pool pool = nullptr;
	sslm_status status = SSLM_INVALID_ARGUMENT;
	BlockPoolOf(sslm_model model, uint32_t blocks, const Lay& y, uint8_t fill = 0) {
		mem.Reset(y.PoolBytes(blocks), fill);
		status = sslm_kv_pool_create(model, mem.p, mem.n, blocks, &pool);
	}
	~BlockPoolOf() {
		if (pool) sslm_kv_pool_destroy(pool);
	}
};

// The test's contiguous-pool size (§21.2): block_count * (cap * bytes_per_position + 8) + 32, with
// exact overflow tests, so the cells can name an overflowing count. *fits is false past 2^64 - 1.
inline uint64_t ContiguousSize(const Lay& y, uint64_t block_count, bool* fits) {
	const uint64_t m = static_cast<uint64_t>(y.contiguous_block_bytes) + 8;
	if (m != 0 && block_count > UINT64_MAX / m) {
		*fits = false;  // the product overflows
		return 0;
	}
	const uint64_t p = block_count * m;
	*fits = p <= UINT64_MAX - 32;  // the sum with the fixed 32
	return p + 32;
}

struct ContigPoolOf {
	AlignedBuf mem;
	sslm_kv_pool pool = nullptr;
	sslm_status status = SSLM_INVALID_ARGUMENT;
	ContigPoolOf(sslm_model model, uint32_t blocks, size_t bytes, uint8_t fill = 0) {
		mem.Reset(bytes, fill);
		status = sslm_kv_pool_create_contiguous(model, mem.p, mem.n, blocks, &pool);
	}
	~ContigPoolOf() {
		if (pool) sslm_kv_pool_destroy(pool);
	}
};

// Creates until a refusal and counts the admissions: legacy creates when budget == 0, budget creates
// of `budget` otherwise. Every made handle is released. The refusal is reported in *refusal.
inline int CountCreates(sslm_model model, sslm_kv_pool* pool, int32_t budget, sslm_status* refusal) {
	std::vector<sslm_seq> made;
	sslm_status st = SSLM_OK;
	for (int guard = 0; guard < 1 << 20; ++guard) {
		sslm_seq s = nullptr;
		st = budget == 0 ? sslm_seq_create(model, pool, &s) : sslm_seq_create_budgeted(model, pool, budget, &s);
		if (st != SSLM_OK) break;
		made.push_back(s);
	}
	if (refusal) *refusal = st;
	for (auto it = made.rbegin(); it != made.rend(); ++it) sslm_seq_release(*it);
	return static_cast<int>(made.size());
}

// ---- tokens --------------------------------------------------------------------------------------

inline int32_t Next(sslm_model model, sslm_seq s, sslm_status* st = nullptr) { return NextToken(model, s, st); }

// Prefill `tokens` in chunks, then the ready token, then (when `decode1`) one more step: the widths
// scenario's shape (pkv_engine_helpers.h RunWidthUncached), through the ABI.
struct AbiWidth {
	sslm_status prefill = SSLM_INVALID_ARGUMENT;
	int32_t ready = -1, decode1 = -1;
	sslm_status decode_status = SSLM_OK;
};

inline AbiWidth RunAbiWidth(sslm_model model, sslm_seq s, int64_t W, int32_t vocab, bool decode1) {
	AbiWidth r;
	r.prefill = PrefillAll(model, s, Stream(pkv_engine::kWidthsSeed, static_cast<int32_t>(W), vocab), pkv_engine::kChunk);
	if (r.prefill != SSLM_OK) return r;
	r.ready = Next(model, s, &r.decode_status);
	if (decode1 && r.ready >= 0) r.decode1 = Next(model, s, &r.decode_status);
	return r;
}

// ---- the flat twin -------------------------------------------------------------------------------

// pkv_engine_helpers.h's EngineFixture, built for any fixture (GetEngineFixture serves the four
// generated stems only). The twin runs the flat one-page view: B = cap, table {0}.
inline std::unique_ptr<pkv_engine::EngineFixture> EngineOf(const Fixture& fx) {
	auto e = std::make_unique<pkv_engine::EngineFixture>();
	e->fx = &fx;
	if (!fx.ok) return e;
	std::string err;
	bool ok = superslm::SslmModel::Load(fx.bytes.p, fx.size, e->view, &err) == superslm::SslmModelStatus::Ok;
	PKV_CHECK_MSG(ok, "SslmModel::Load(%s): %s", fx.stem.c_str(), err.c_str());
	ok = ok && e->view.config.kv_precision == superslm::SslmKvPrecision::Int8;
	if (ok) {
		const uint32_t n = e->view.config.num_hidden_layers;
		e->backings.resize(n);
		e->layers.resize(n);
		for (uint32_t l = 0; l < n && ok; ++l) {
			std::string merr;
			ok = superslm_marshal::MarshalLayer(e->view, l, e->view.config.num_attention_heads,
			                                    e->view.config.num_key_value_heads, e->backings[l], e->layers[l], &merr);
			PKV_CHECK_MSG(ok, "MarshalLayer(%s, %u): %s", fx.stem.c_str(), l, merr.c_str());
		}
	}
	if (ok) {
		const superslm::SslmTensorView* embed_w = e->view.weights.Tensor("embed");
		const superslm::SslmTensorView* gain_w = e->view.weights.Tensor("final_norm.gain");
		const superslm::SslmTensorView* head_w =
		    e->view.config.tie_word_embeddings ? embed_w : e->view.weights.Tensor("lm_head");
		bool sc_ok = true;
		e->embed_sc = superslm_marshal::ReadCarriedScale(e->view.composition_constants, "embed", &sc_ok);
		e->final_sc = superslm_marshal::ReadCarriedScale(e->view.composition_constants, "final_norm", &sc_ok);
		ok = embed_w && gain_w && head_w && sc_ok;
		if (ok) {
			e->final_gain = superslm_marshal::WidenGainToInt32(*gain_w);
			e->embed = reinterpret_cast<const int8_t*>(embed_w->data);
			e->head = reinterpret_cast<const int8_t*>(head_w->data);
		}
	}
	e->ok = ok;
	return e;
}

// The flat twin's widths run at width W: tokens ready and decode1, and the canonical rows' digests.
inline pkv_engine::WidthRun FlatTwin(const pkv_engine::EngineFixture& e, int64_t W) {
	return pkv_engine::RunWidthUncached(e, e.Cap(), pkv_engine::Drive::kChunked, W);
}

// ---- 4.1 and 4.2 at the derived B (T-3196 F-2) -----------------------------------------------------

// The reference tool's `widths` and `prefix_lengths` scenarios are fixed at B = 16's boundaries ({1, 15, 16, 17,
// 32, 1,064, cap - 1, cap} and {0, 1, 15, 16, 17, 1,000, 1,008}), so at a derived B every one of them sits mid-page.
// Plan 4.1 and 4.2 define them from the run's B; these are those sets at the B the engine reports, graded against
// the flat twin (B = cap, table {0}), which has no page boundary to be off by one at.

// 4.1's widths at B: 1, B - 1, B, B + 1, 2B, 1,000 + max(64, 2B), and cap - 1, cap when the cap is at most 4,100 (as
// the scenario: the 32k fixture's cap-sized widths cost minutes; pkv_def, pkv_qk and pkv_odd carry the cap edge).
inline std::vector<int64_t> WidthsAt(int64_t B, int64_t cap) {
	std::vector<int64_t> w = {1, B - 1, B, B + 1, 2 * B, 1000 + std::max<int64_t>(64, 2 * B)};
	if (cap <= 4100) {
		w.push_back(cap - 1);
		w.push_back(cap);
	}
	std::vector<int64_t> out;
	for (int64_t x : w)
		if (x >= 1 && x <= cap && std::find(out.begin(), out.end(), x) == out.end()) out.push_back(x);
	return out;
}

// 4.2's prefix lengths at B, with the aligned length re-chosen as 1,024 (plan 4.13): 0, 1, B - 1, B, B + 1, 1,000,
// 1,024.
inline std::vector<int32_t> PrefixLengthsAt(int64_t B) {
	std::vector<int32_t> out;
	for (int64_t x : {int64_t{0}, int64_t{1}, B - 1, B, B + 1, int64_t{1000}, int64_t{1024}})
		if (x >= 0 && std::find(out.begin(), out.end(), static_cast<int32_t>(x)) == out.end())
			out.push_back(static_cast<int32_t>(x));
	return out;
}

// The flat twin's run at width W for a fixture, computed once per process.
inline const pkv_engine::WidthRun& TwinRun(const Fixture& fx, int64_t W) {
	static std::map<std::string, std::unique_ptr<pkv_engine::EngineFixture>> engines;
	static std::map<std::pair<std::string, int64_t>, pkv_engine::WidthRun> runs;
	const auto key = std::make_pair(fx.stem, W);
	auto it = runs.find(key);
	if (it != runs.end()) return it->second;
	auto& e = engines[fx.stem];
	if (!e) e = EngineOf(fx);
	pkv_engine::WidthRun r = e->ok ? FlatTwin(*e, W) : pkv_engine::WidthRun{};
	if (!e->ok) r.fail = "the flat twin did not build";
	return runs.emplace(key, std::move(r)).first->second;
}

enum BoundaryParts { kBoundaryWidths = 1, kBoundaryPrefixes = 2 };

// 4.1 and/or 4.2 at `B`, which the engine must report for `fx` (sslm_kv_page_positions): every width and prefix
// length below is placed from it. Widths: a legacy holder prefills W tokens of the widths stream in 64-token chunks;
// the ready token, the next token, and rows [0, W) and [0, W + 1) read through the peek at B rotated equal the flat
// twin's. Prefix lengths: a frozen legacy prefix of the stream's first P tokens, adopted (a legacy copy adopt,
// aligned at P = B and 1,024); rows [0, P) equal the twin's at P, then 20 more tokens prefilled and the ready and
// next tokens and rows [0, P + 21) equal the twin's at P + 20 (chunk boundaries do not change the result:
// c2_engine_widths.cpp's chunked and per-token drives grade against one record). Kills: a page-boundary map off by
// one at B, the tail copy of an aligned adopt, and a coverage guard off by one at a page edge.
inline void DerivedBoundaries(const Fixture& fx, int64_t B, int parts, const char* cell, Rot rot = Rot::kRotated) {
	const int64_t got = sslm_kv_page_positions(fx.model);
	PKV_CHECK_MSG(got == B, "%s %s: the engine reports B = %lld; the widths below sit on the boundaries of %lld", cell,
	              fx.stem.c_str(), static_cast<long long>(got), static_cast<long long>(B));
	const int64_t cap = fx.geo.context_cap;
	if (parts & kBoundaryWidths) {
		for (int64_t W : WidthsAt(B, cap)) {
			const pkv_engine::WidthRun& twin = TwinRun(fx, W);
			PKV_CHECK_MSG(twin.fail.empty(), "%s %s W = %lld: the flat twin: %s", cell, fx.stem.c_str(),
			              static_cast<long long>(W), twin.fail.c_str());
			if (!twin.fail.empty()) continue;
			LegacyPool pool(fx.model, 1);
			sslm_seq s = nullptr;
			PKV_CHECK_EQ(pool.status, SSLM_OK);
			if (!pool.pool || sslm_seq_create(fx.model, &pool.pool, &s) != SSLM_OK) {
				PKV_CHECK_MSG(false, "%s %s W = %lld: no legacy holder", cell, fx.stem.c_str(), static_cast<long long>(W));
				continue;
			}
			const AbiWidth w = RunAbiWidth(fx.model, s, W, fx.vocab, false);
			std::vector<uint8_t> rows;
			PKV_CHECK_MSG(w.prefill == SSLM_OK && w.ready == twin.ready, "%s %s W = %lld (W mod B = %lld): prefill %s, "
			              "ready %d, the twin's %d", cell, fx.stem.c_str(), static_cast<long long>(W),
			              static_cast<long long>(W % B), AbiStatusName(w.prefill), w.ready, twin.ready);
			PKV_CHECK_MSG(SeqRows(pool.pool, s, fx.geo, 1, B, W, rot, &rows) &&
			                  Sha(rows.data(), rows.size()) == twin.rows_prefill,
			              "%s %s W = %lld: rows [0, W) through the peek at B = %lld (%s) differ from the twin's", cell,
			              fx.stem.c_str(), static_cast<long long>(W), static_cast<long long>(B), RotName(rot));
			if (W < cap) {
				sslm_status st = SSLM_OK;
				const int32_t d1 = Next(fx.model, s, &st);
				PKV_CHECK_MSG(d1 == twin.decode1, "%s %s W = %lld: the next token %d, the twin's %d", cell, fx.stem.c_str(),
				              static_cast<long long>(W), d1, twin.decode1);
				PKV_CHECK_MSG(SeqRows(pool.pool, s, fx.geo, 1, B, W + 1, rot, &rows) &&
				                  Sha(rows.data(), rows.size()) == twin.rows_decode1,
				              "%s %s W = %lld: rows [0, W + 1) differ from the twin's", cell, fx.stem.c_str(),
				              static_cast<long long>(W));
			}
			sslm_seq_release(s);
		}
	}
	if (parts & kBoundaryPrefixes) {
		for (int32_t P : PrefixLengthsAt(B)) {
			const std::vector<int32_t> toks = Stream(pkv_engine::kWidthsSeed, P + 20, fx.vocab);
			LegacyPool pool(fx.model, 2);
			PKV_CHECK_EQ(pool.status, SSLM_OK);
			if (!pool.pool) continue;
			sslm_prefix p = bc::LegacyPrefix(fx, &pool.pool, std::vector<int32_t>(toks.begin(), toks.begin() + P));
			sslm_seq s = nullptr;
			PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
			PKV_CHECK_MSG(p && s, "%s %s P = %d: the prefix or the adopter was not made", cell, fx.stem.c_str(), P);
			if (p && s) {
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, p), SSLM_OK);
				std::vector<uint8_t> rows;
				if (P > 0) {
					const pkv_engine::WidthRun& at_p = TwinRun(fx, P);
					PKV_CHECK_MSG(at_p.fail.empty() && SeqRows(pool.pool, s, fx.geo, 1, B, P, rot, &rows) &&
					                  Sha(rows.data(), rows.size()) == at_p.rows_prefill,
					              "%s %s P = %d (P mod B = %lld): the adopted rows [0, P) differ from the twin's", cell,
					              fx.stem.c_str(), P, static_cast<long long>(P % B));
				}
				const pkv_engine::WidthRun& at = TwinRun(fx, P + 20);
				const sslm_status pf = PrefillAll(fx.model, s, std::vector<int32_t>(toks.begin() + P, toks.end()), pkv_engine::kChunk);
				sslm_status st = SSLM_OK;
				const int32_t ready = pf == SSLM_OK ? Next(fx.model, s, &st) : -1;
				const int32_t d1 = ready >= 0 ? Next(fx.model, s, &st) : -1;
				PKV_CHECK_MSG(at.fail.empty() && ready == at.ready && d1 == at.decode1,
				              "%s %s P = %d, then 20: tokens %d, %d; the twin's %d, %d (%s)", cell, fx.stem.c_str(), P,
				              ready, d1, at.ready, at.decode1, AbiStatusName(pf));
				PKV_CHECK_MSG(SeqRows(pool.pool, s, fx.geo, 1, B, P + 21, rot, &rows) &&
				                  Sha(rows.data(), rows.size()) == at.rows_decode1,
				              "%s %s P = %d, then 20: rows [0, P + 21) differ from the twin's", cell, fx.stem.c_str(), P);
			}
			if (s) sslm_seq_release(s);
			if (p) sslm_prefix_release(p);
		}
	}
}

// ---- batch decode across pool kinds and the partial last page (plan rev 16.24 §26.1, D-SLM8441) ------

// A legacy blob of the widths stream at width W, saved from a holder in a one-block page pool at the rule's B.
inline std::vector<uint8_t> WidthsBlob(const Fixture& fx, int64_t B, int64_t W) {
	const Lay y = LayoutOf(fx.geo, 1, B);
	BlockPoolOf pool(fx.model, 1, y);
	sslm_seq s = nullptr;
	std::vector<uint8_t> blob;
	if (pool.pool && sslm_seq_create(fx.model, &pool.pool, &s) == SSLM_OK) {
		if (PrefillAll(fx.model, s, Stream(pkv_engine::kWidthsSeed, static_cast<int32_t>(W), fx.vocab), pkv_engine::kChunk) ==
		    SSLM_OK)
			blob = bc::Save(s);
		sslm_seq_release(s);
	}
	PKV_CHECK_MSG(!blob.empty(), "%s: the width-%lld blob was not made", fx.stem.c_str(), static_cast<long long>(W));
	return blob;
}

struct BatchArm {
	sslm_kv_pool* pool;
	int64_t B;  // the page size of this holder's pool: the rule's B for a page pool, the cap for the option
	const char* kind;
};

constexpr int kBatchSteps = 13;  // the first step emits the restored ready token (no row); 12 more write rows W..W+11

// §26.1's member: `blob` (width W) restored into both arms' pools, which are 0x5A-filled; the two holders decode
// together, one sslm_decode_step per token, for kBatchSteps steps. The oracle is the single-sequence twin, a third
// restore into a fresh 0x5A-filled one-block page pool at `twin_B`, decoded alone with n = 1. Asserted: tokens equal
// across the batch and with the twin; each holder's rows [W, W + 12), read through the peek at its own pool's B
// rotated, equal the twin's; its rows [0, W) equal the blob's; precondition, no twin row of [W, W + 12) is D bytes
// of 0x5A. With `cap_after`, one more call returns SSLM_CONTEXT_CAP_EXCEEDED with out_tokens[0] = -1.
// Kills (§26.1, executed by the planner's probe): H1, the batch taking page_positions and page_bytes once from the
// first sequence's pool; H2, the batch taking them from the model.
inline void BatchAgainstTwin(const Fixture& fx, const std::vector<uint8_t>& blob, int64_t W, int64_t twin_B,
                             const BatchArm& a0, const BatchArm& a1, bool cap_after, const std::string& what,
                             Rot rot = Rot::kRotated) {
	if (blob.empty()) return;
	const size_t D = fx.geo.head_dim;
	const size_t per_pos = size_t{fx.geo.layers} * 2u * fx.geo.kv_heads * D;
	const int64_t L = W + kBatchSteps - 1;
	std::vector<int32_t> twin_tokens;
	std::vector<uint8_t> twin_rows;
	{
		BlockPoolOf tp(fx.model, 1, LayoutOf(fx.geo, 1, twin_B), 0x5A);
		sslm_seq t = nullptr;
		PKV_CHECK_EQ(tp.pool ? sslm_seq_restore(fx.model, &tp.pool, blob.data(), blob.size(), &t) : tp.status, SSLM_OK);
		if (!t) return;
		for (int i = 0; i < kBatchSteps; ++i) twin_tokens.push_back(Next(fx.model, t));
		std::vector<uint8_t> all;
		PKV_CHECK_MSG(SeqRows(tp.pool, t, fx.geo, 1, twin_B, L, rot, &all) && all.size() == size_t(L) * per_pos,
		              "%s: the twin's rows are unreadable", what.c_str());
		if (all.size() == size_t(L) * per_pos) twin_rows.assign(all.begin() + W * static_cast<int64_t>(per_pos), all.end());
		sslm_seq_release(t);
	}
	size_t fill_rows = 0;
	for (size_t at = 0; at + D <= twin_rows.size(); at += D)
		fill_rows += std::all_of(twin_rows.begin() + at, twin_rows.begin() + at + D, [](uint8_t b) { return b == 0x5A; });
	PKV_CHECK_MSG(!twin_rows.empty() && fill_rows == 0, "%s: construction: %zu twin rows are D bytes of 0x5A", what.c_str(),
	              fill_rows);
	sslm_seq s[2] = {nullptr, nullptr};
	const BatchArm* arm[2] = {&a0, &a1};
	for (int i = 0; i < 2; ++i) {
		const sslm_status st = sslm_seq_restore(fx.model, arm[i]->pool, blob.data(), blob.size(), &s[i]);
		PKV_CHECK_MSG(st == SSLM_OK, "%s: the restore into the %s pool: %s", what.c_str(), arm[i]->kind, AbiStatusName(st));
	}
	if (s[0] && s[1]) {
		sslm_decode_params p{};
		p.layer_budget = static_cast<int32_t>(fx.geo.layers);
		std::vector<int32_t> got[2];
		for (int i = 0; i < kBatchSteps; ++i) {
			int32_t out[2] = {-1, -1};
			const sslm_status st = sslm_decode_step(fx.model, s, 2, &p, nullptr, out);
			PKV_CHECK_MSG(st == SSLM_OK, "%s: step %d: %s", what.c_str(), i, AbiStatusName(st));
			got[0].push_back(out[0]);
			got[1].push_back(out[1]);
		}
		for (int i = 0; i < 2; ++i) {
			PKV_CHECK_MSG(got[i] == twin_tokens, "%s: the %s holder's batch tokens differ from the n = 1 twin's", what.c_str(),
			              arm[i]->kind);
			std::vector<uint8_t> rows, brows;
			const bool read = SeqRows(*arm[i]->pool, s[i], fx.geo, 1, arm[i]->B, L, rot, &rows) &&
			                  rows.size() == size_t(L) * per_pos;
			PKV_CHECK_MSG(read && BlobRows(blob, fx, HiddenSize(fx), W, &brows) &&
			                  std::equal(brows.begin(), brows.end(), rows.begin()),
			              "%s: the %s holder's restored rows [0, %lld) differ from the blob's (B = %lld)", what.c_str(),
			              arm[i]->kind, static_cast<long long>(W), static_cast<long long>(arm[i]->B));
			PKV_CHECK_MSG(read && std::equal(twin_rows.begin(), twin_rows.end(), rows.begin() + W * static_cast<int64_t>(per_pos)),
			              "%s: the %s holder's rows [%lld, %lld), read at its pool's B = %lld, differ from the twin's",
			              what.c_str(), arm[i]->kind, static_cast<long long>(W), static_cast<long long>(L),
			              static_cast<long long>(arm[i]->B));
		}
		if (cap_after) {
			int32_t out[2] = {7, 7};
			const sslm_status st = sslm_decode_step(fx.model, s, 2, &p, nullptr, out);
			PKV_CHECK_MSG(st == SSLM_CONTEXT_CAP_EXCEEDED && out[0] == -1,
			              "%s: the step past the cap: %s, out_tokens[0] = %d (SSLM_CONTEXT_CAP_EXCEEDED, -1 owed)", what.c_str(),
			              AbiStatusName(st), out[0]);
		}
	}
	for (sslm_seq x : s)
		if (x) sslm_seq_release(x);
}

// ---- engine-level rotation (6.6a) -----------------------------------------------------------------

// KvPageView gains `runs` at T1 (§3.2 item 1). The cells set it where the shipped struct has it, by
// detection, so the suite compiles on 2ebe5e3's struct (no field: the engine cannot rotate, and the
// cell's address assertions fail for exactly that reason) and on the R9 struct unchanged.
template <class V, class = void>
struct HasRuns : std::false_type {};
template <class V>
struct HasRuns<V, std::void_t<decltype(std::declval<V&>().runs)>> : std::true_type {};

template <class V>
inline void SetRuns(V& view, uint32_t runs) {
	if constexpr (HasRuns<V>::value) view.runs = runs;
	(void)view;
	(void)runs;
}

}  // namespace r9
}  // namespace pkv

#endif  // SUPERSLM_TESTS_PKV_R9_H
