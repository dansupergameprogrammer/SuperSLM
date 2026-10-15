// Paged-KV plan (rev 16.1) step C1: dimension 10's CPU achievement cells, owned by C6 (§7, §8). The figures are per
// page size from rev 16.25 (§27, D-SLM8450): kCohortFigures below, one engine-executed row per B.
//
//   10.1  the cohort on the 0.5B artifact at cap 4096: world 1,000 -> 10 personas (begin_from) -> 50
//         budget-512 sequences; each emits the §3.4 count, with tokens equal to the same prompts run
//         with no sharing on v1.11.0
//   10.2  the 1.5B artifact at cap 32768: one 8-sequence cohort on a nested prefix, tokens equal v1.11.0
//   10.4  the memory achievement: the row's pool (505 at B = 64, 287 at B = 128) admits the cohort, one page fewer
//         refuses exactly the 50th create; the construction peak (66, 38) admits all ten personas open, one fewer
//         refuses the tenth; a v1.11.0 block pool in the same bytes admits the row's contrast count (7, 8)
//   10.4/C6:table-mutant  bench-only vitality: the lookup forced to B = 16's row (1,842, 203, 7) at the running B;
//         10.4's bounds must fail, so a table not keyed by B is separated
//   10.6  the cohort survives a save and load: the row's pool admits the 50 shared restores, one page fewer refuses
//         the 50th, both passes (1,456 and the limit); tokens continue equal to the unsaved run's
//   10.8  the no-handle recovery: the row's private pool (1,405, 737) admits the 50 private restores, one fewer
//         refuses the 50th; after each re-adopts, exactly the row's further budget-512 creates (100, 90) are admitted
//
// Each cell is registered twice. "<id>/C6" is the box cell: it reads the real artifact from
// SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR (a missing variable or file is a failed check) and its reference
// from v1.11.0_<artifact stem>.ref (the box's R0 run); unpinned, the 0.5B runs at the rule's B = 64.
// "<id>/C6:pkv_def" (and ":pkv_32k" for 10.2) runs the same construction on the cloud fixture at the rule's B as
// well (D-SLM8447): pkv_def (D = 48, cap 4,096) runs at B = 128, not at 16 and not at the 0.5B's 64, so it takes the
// B = 128 row. The function looks its row up by the B the engine reports, asserted first to equal the rule's B written
// in the test; a B with no row fails, naming it. The test's §3.4 / §3.7 arithmetic (CohortPages, PrivateCohortPages,
// the peak, the further count) is kept as a cross-check asserted equal to the row: the row is executed on the engine
// (§27.2), independent of that arithmetic, so a formula mutant fails against the table and a table mutant against
// the engine's admission.
//
// The no-sharing reference is the scenario "cohort" (records persona<i>_1456 and persona<i>_1712): each
// persona's whole 1,200-token prompt prefilled into its own v1.11.0 block and 513 tokens decoded. The
// flat path through the new code is never the reference (§8). Page counts are graded by admission;
// *out_shared_pages and the stats verbs are never read for a verdict.

#include "pkv_budget_c_helpers.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {

using namespace pkv;
using namespace pkv::bc;

constexpr int kPersonas = 10;
constexpr int kSeqs = 50;  // five per persona
constexpr int32_t kBudget = 512;
constexpr int64_t kOrigin = 1200;
constexpr int64_t kMidAt = 1456;  // 10.6's mid-budget save

const char* k05B = "qwen2.5-0.5b-instruct-cap4096-aex.sslm";
const char* k15B = "qwen2.5-1.5b-instruct.sslm";
const Geometry kGeo05B{24, 2, 64, 4096};
const Geometry kGeo15B{28, 2, 128, 32768};

std::string ExecutableSha() {
	static const std::string digest = [] {
		char path[4096]{};
#if defined(_WIN32)
		const DWORD count = GetModuleFileNameA(nullptr, path, sizeof path);
		if (!count || count >= sizeof path) return std::string{};
#else
		const ssize_t count = readlink("/proc/self/exe", path, sizeof path - 1);
		if (count <= 0 || count >= static_cast<ssize_t>(sizeof path)) return std::string{};
		path[count] = 0;
#endif
		std::vector<uint8_t> bytes;
		if (!ReadAll(path, &bytes) || bytes.empty()) return std::string{};
		return Sha(bytes.data(), bytes.size());
	}();
	return digest;
}

int PersonaOf(int seq) { return seq / (kSeqs / kPersonas); }

// ---- the page arithmetic of §5, written from §3.4 / §3.7 ----------------------------------------

// World and ten personas, the world released: its 62 full pages (held by the personas) plus each
// persona's own 13 (its copied tail page onwards).
int64_t ChainPages(const Fixture& fx, int personas) {
	return SharedPages(fx, 1000) + personas * (PrefixPages(fx, kOrigin) - SharedPages(fx, 1000));
}
int64_t CohortPages(const Fixture& fx) { return ChainPages(fx, kPersonas) + kSeqs * fx.R(kBudget); }
int64_t PrivateCohortPages(const Fixture& fx) {
	return ChainPages(fx, kPersonas) + kSeqs * (fx.R(kBudget) + MaterializedPages(fx, kOrigin, kBudget));
}

// ---- the figures per B (plan rev 16.25 §27.2, D-SLM8450) ----------------------------------------

// Each figure is the smallest page pool at which the engine admits the stated verdict, bisected on 2ebe5e3's engine
// with the boundary's two neighbours run (§27.2, Claude/Plans/superslm-paged-kv/probes/cohort_engine_boundary.cpp),
// independent of the arithmetic above. Cap 4,096, 50 budget-512 sequences, 10 personas of 200 over a 1,000 world.
struct CohortFigures {
	int64_t B;
	int64_t pool;           // 10.4 and 10.6: admits the cohort; one page fewer refuses exactly the 50th
	int64_t peak;           // 10.4: construction peak, all ten personas open
	int64_t contrast;       // 10.4: a block pool in the same bytes admits this many
	int64_t private_pool;   // 10.8: admits the 50 restores without handles
	int64_t further;        // 10.8: further budget-512 creates after each re-adopts
};
const CohortFigures kCohortFigures[] = {
    {64, 505, 66, 7, 1405, 100},   // the box (0.5B, D = 64)
    {128, 287, 38, 8, 737, 90},    // the cloud twin (pkv_def, D = 48)
};
// B = 16's row, the method's control in §27.2. No C6 cell runs at 16; only the table mutant reads it.
const CohortFigures kCohortFiguresB16 = {16, 1842, 203, 7, 5592, 113};
bool g_table_mutant = false;  // 10.4/C6:table-mutant: the lookup ignores B and returns B = 16's row

// The row for the B the engine reports. Asserted first: that B equals the rule's B written in the test, and the table
// has a row for it; otherwise a failed check names B and the caller returns without running the cohort.
const CohortFigures* FiguresFor(const Fixture& fx, const char* cell) {
	const int64_t B = sslm_kv_page_positions(fx.model);
	const int64_t rule = PageRuleB(fx.geo.head_dim, static_cast<uint64_t>(fx.geo.context_cap), 1);
	PKV_CHECK_MSG(B == rule, "%s %s: the engine reports B = %lld; the rule's B is %lld", cell, fx.stem.c_str(),
	              static_cast<long long>(B), static_cast<long long>(rule));
	if (B != rule) return nullptr;
	if (g_table_mutant) return &kCohortFiguresB16;
	for (const CohortFigures& f : kCohortFigures)
		if (f.B == B) return &f;
	PKV_CHECK_MSG(false, "%s %s: no cohort figures row for B = %lld (rows: 64, 128)", cell, fx.stem.c_str(),
	              static_cast<long long>(B));
	return nullptr;
}

// What the table mutant must break: 10.4's two lower bounds (one page below the pool, one below the peak).
struct BoundsSeen {
	int pool_short_failed = 0;
	int peak_short_failed = 0;
};
BoundsSeen g_bounds;

// ---- the reference --------------------------------------------------------------------------------

// The no-sharing tokens of persona i: persona<i>_1456 then persona<i>_1712.
std::vector<int32_t> RefTokens(const Fixture& fx, int persona) {
	const RefFile& ref = Reference("v1.11.0", fx);
	const std::string tag = "persona" + std::to_string(persona);
	const RefRecord* a = RefLookup(ref, "cohort", tag + "_1456");
	const RefRecord* b = RefLookup(ref, "cohort", tag + "_1712");
	std::vector<int32_t> t;
	if (!a || !b) return t;
	t = a->tokens;
	t.insert(t.end(), b->tokens.begin(), b->tokens.end());
	PKV_CHECK_MSG(a->context_length == kMidAt && b->context_length == kOrigin + kBudget,
	              "cohort reference persona%d: positions %lld / %lld", persona, static_cast<long long>(a->context_length),
	              static_cast<long long>(b->context_length));
	return t;
}

// ---- the cohort run ---------------------------------------------------------------------------

// The world, then the ten personas by begin_from, then the world released (§5's order).
struct World {
	sslm_prefix world = nullptr;
	std::vector<sslm_prefix> personas;
	bool ok = false;
	void Release() {
		for (sslm_prefix p : personas)
			if (p) sslm_prefix_release(p);
		personas.clear();
		if (world) sslm_prefix_release(world);
		world = nullptr;
	}
};
World BuildWorld(const Fixture& fx, sslm_kv_pool* pool, int personas) {
	World w;
	sslm_status st = sslm_prefix_begin_budgeted(fx.model, pool, 1000, &w.world);
	if (st == SSLM_OK) st = PrefixPrefillAll(fx.model, w.world, WorldTokens(), kChunk);
	if (st == SSLM_OK) st = sslm_prefix_freeze(w.world);
	PKV_CHECK_MSG(st == SSLM_OK, "the world: status %d", static_cast<int>(st));
	for (int i = 0; i < personas && st == SSLM_OK; ++i) {
		sslm_prefix p = nullptr;
		st = sslm_prefix_begin_from(w.world, 200, &p);
		PKV_CHECK_MSG(st == SSLM_OK, "begin_from for persona %d: status %d", i, static_cast<int>(st));
		if (st != SSLM_OK) break;
		w.personas.push_back(p);
		st = PrefixPrefillAll(fx.model, p, PersonaTokens(i), kChunk);
		if (st == SSLM_OK) st = sslm_prefix_freeze(p);
		PKV_CHECK_MSG(st == SSLM_OK, "persona %d: status %d", i, static_cast<int>(st));
	}
	if (st == SSLM_OK) {
		st = sslm_prefix_release(w.world);
		w.world = nullptr;
	}
	w.ok = st == SSLM_OK;
	return w;
}

struct CohortRun {
	bool built = false;
	int first_refused = -1;            // the index of the first refused create, -1 if none
	sslm_status refusal = SSLM_OK;
	int creates_refused = 0;
	std::vector<std::vector<int32_t>> tokens;   // per admitted sequence: everything it emitted
	std::vector<sslm_status> end;               // the status that ended it
	std::vector<std::vector<uint8_t>> mid, lim;  // 10.6's saves at 1,456 and at the limit
	std::vector<std::string> mid_rows, lim_rows;  // direct page-peek digests at the reference record points
};

// Read the live page table and page bytes through the test seam, independently of
// token output and of the save path. The reference digests use position-major rows.
std::string CohortRows(const Fixture& fx, sslm_kv_pool pool, sslm_seq seq, int64_t length) {
	std::vector<uint8_t> rows;
	std::vector<uint8_t> page(fx.PageBytes());
	int64_t loaded = -1;
	for (int64_t pos = 0; pos < length; ++pos) {
		if (pos / fx.B() != loaded) {
			uint32_t index = 0, mapped = 0;
			if (sslm_pkv_test_only_seq_table_entry(seq, static_cast<uint32_t>(pos / fx.B()), &index, &mapped) != SSLM_OK ||
			    !mapped || sslm_pkv_test_only_peek_page_bytes(pool, index, page.data(), page.size()) != SSLM_OK)
				return "(unreadable)";
			loaded = pos / fx.B();
		}
		for (uint32_t layer = 0; layer < fx.geo.layers; ++layer)
			for (uint32_t half = 0; half < 2; ++half)
				for (uint32_t head = 0; head < fx.geo.kv_heads; ++head) {
					const size_t offset =
					    PageRunOffset(fx.geo.layers, fx.geo.kv_heads, fx.geo.head_dim, fx.B(), layer, half, head, pos);
					rows.insert(rows.end(), page.begin() + offset, page.begin() + offset + fx.geo.head_dim);
				}
	}
	const char* flip = std::getenv("SUPERSLM_PAGED_KV_ROW_DIGEST_FLIP");
	if (flip && !rows.empty() &&
	    (std::string(flip) == "both" ||
	     (std::string(flip) == "mid" && length == kMidAt) ||
	     (std::string(flip) == "limit" && length == kOrigin + kBudget))) rows[rows.size() / 2] ^= 1;
	return Sha(rows.data(), rows.size());
}

bool CheckCohortRows(const Fixture& fx, int persona, const std::string& mid, const std::string& lim) {
	const RefFile& ref = Reference("v1.11.0", fx);
	const std::string tag = "persona" + std::to_string(persona);
	const RefRecord* a = RefLookup(ref, "cohort", tag + "_1456");
	const RefRecord* b = RefLookup(ref, "cohort", tag + "_1712");
	return a && b && mid == a->rows_sha && lim == b->rows_sha;
}

// Runs the cohort in a page pool of `pages`: every sequence is created, adopts its persona, emits 257
// tokens (to 1,456: saved), and runs to its refusal (saved at the limit).
CohortRun RunCohort(const Fixture& fx, uint32_t pages, bool keep_blobs, bool drive_tokens = true) {
	CohortRun r;
	PagePool pool(fx.model, pages);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (pool.status != SSLM_OK) return r;
	World w = BuildWorld(fx, &pool.pool, kPersonas);
	r.built = w.ok;
	std::vector<sslm_seq> seqs;
	for (int i = 0; i < kSeqs && w.ok; ++i) {
		sslm_seq s = nullptr;
		const sslm_status st = sslm_seq_create_budgeted(fx.model, &pool.pool, kBudget, &s);
		if (st != SSLM_OK) {
			if (r.first_refused < 0) {
				r.first_refused = i;
				r.refusal = st;
			}
			++r.creates_refused;
			continue;
		}
		seqs.push_back(s);
		PKV_CHECK_MSG(sslm_seq_adopt_prefix(s, w.personas[static_cast<size_t>(PersonaOf(i))]) == SSLM_OK, "adopt of sequence %d", i);
		if (!drive_tokens) continue;  // 10.4 grades admission; 10.1 grades the full decode and its end state.
		sslm_status a = SSLM_OK, b = SSLM_OK;
		std::vector<int32_t> t = DecodeN(fx.model, s, kMidAt - kOrigin + 1, &a);
		if (drive_tokens) r.mid_rows.push_back(CohortRows(fx, pool.pool, s, kMidAt));
		if (keep_blobs) r.mid.push_back(Save(s));
		const std::vector<int32_t> rest = RunToRefusal(fx.model, s, &b);
		t.insert(t.end(), rest.begin(), rest.end());
		if (keep_blobs) r.lim.push_back(Save(s));
		if (drive_tokens) r.lim_rows.push_back(CohortRows(fx, pool.pool, s, kOrigin + kBudget));
		PKV_CHECK_MSG(a == SSLM_OK, "sequence %d: status %d before 1,456", i, static_cast<int>(a));
		r.tokens.push_back(t);
		r.end.push_back(b);
	}
	for (auto it = seqs.rbegin(); it != seqs.rend(); ++it) sslm_seq_release(*it);
	w.Release();
	return r;
}

// One bounded 10.1 process still proves admission of the complete 50-sequence cohort. It drives
// ten selected sequences while all 50 reservations and persona adoptions remain live. Five such
// processes cover every output; the other cells reassemble only green, source-bound shards.
CohortRun RunCohortShard(const Fixture& fx, int shard) {
	CohortRun r;
	PagePool pool(fx.model, static_cast<uint32_t>(CohortPages(fx)));
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (pool.status != SSLM_OK) return r;
	World w = BuildWorld(fx, &pool.pool, kPersonas);
	r.built = w.ok;
	std::vector<sslm_seq> seqs(kSeqs, nullptr);
	for (int i = 0; i < kSeqs && w.ok; ++i) {
		sslm_seq s = nullptr;
		const sslm_status st = sslm_seq_create_budgeted(fx.model, &pool.pool, kBudget, &s);
		if (st != SSLM_OK) {
			if (r.first_refused < 0) { r.first_refused = i; r.refusal = st; }
			++r.creates_refused;
			continue;
		}
		seqs[i] = s;
		PKV_CHECK_MSG(sslm_seq_adopt_prefix(s, w.personas[static_cast<size_t>(PersonaOf(i))]) == SSLM_OK,
		              "10.1 shard %d: adopt of sequence %d", shard, i);
	}
	r.tokens.resize(kSeqs);
	r.mid.resize(kSeqs);
	r.lim.resize(kSeqs);
	r.mid_rows.resize(kSeqs);
	r.lim_rows.resize(kSeqs);
	r.end.resize(kSeqs, SSLM_OK);
	for (int i = shard * 10; i < (shard + 1) * 10 && w.ok; ++i) {
		if (!seqs[i]) continue;
		sslm_status a = SSLM_OK, b = SSLM_OK;
		std::vector<int32_t> t = DecodeN(fx.model, seqs[i], kMidAt - kOrigin + 1, &a);
		r.mid[i] = Save(seqs[i]);
		r.mid_rows[i] = CohortRows(fx, pool.pool, seqs[i], kMidAt);
		const std::vector<int32_t> rest = RunToRefusal(fx.model, seqs[i], &b);
		t.insert(t.end(), rest.begin(), rest.end());
		r.lim[i] = Save(seqs[i]);
		r.lim_rows[i] = CohortRows(fx, pool.pool, seqs[i], kOrigin + kBudget);
		r.tokens[i] = std::move(t);
		r.end[i] = b;
		PKV_CHECK_MSG(a == SSLM_OK, "10.1 shard %d: sequence %d status %d before 1,456", shard, i, static_cast<int>(a));
	}
	for (auto it = seqs.rbegin(); it != seqs.rend(); ++it) if (*it) sslm_seq_release(*it);
	w.Release();
	return r;
}

// A successful 10.1 run may save its 50 source blobs for 10.6 and 10.8. This is a process
// checkpoint, never an independent reference: 10.6 still compares the continued tokens with both
// the unsaved run and v1.11.0, and 10.4 runs fresh admission states. The caller supplies a scratch
// directory and the exact source HEAD, while the file itself binds the loaded artifact's SHA-256.
bool CacheVector(std::fstream& f, std::vector<uint8_t>* bytes, bool reading) {
	uint32_t n = reading ? 0 : static_cast<uint32_t>(bytes->size());
	if (reading) f.read(reinterpret_cast<char*>(&n), sizeof(n));
	else f.write(reinterpret_cast<const char*>(&n), sizeof(n));
	if (!f || n > 100u * 1024u * 1024u) return false;
	if (reading) bytes->resize(n);
	if (n) {
		if (reading) f.read(reinterpret_cast<char*>(bytes->data()), n);
		else f.write(reinterpret_cast<const char*>(bytes->data()), n);
	}
	return static_cast<bool>(f);
}
bool CacheTokens(std::fstream& f, std::vector<int32_t>* tokens, bool reading) {
	uint32_t n = reading ? 0 : static_cast<uint32_t>(tokens->size());
	if (reading) f.read(reinterpret_cast<char*>(&n), sizeof(n));
	else f.write(reinterpret_cast<const char*>(&n), sizeof(n));
	if (!f || n > 1u + static_cast<uint32_t>(kBudget)) return false;
	if (reading) tokens->resize(n);
	if (n) {
		if (reading) f.read(reinterpret_cast<char*>(tokens->data()), n * sizeof(int32_t));
		else f.write(reinterpret_cast<const char*>(tokens->data()), n * sizeof(int32_t));
	}
	return static_cast<bool>(f);
}
bool CacheDigest(std::fstream& f, std::string* digest, bool reading) {
	char bytes[64]{};
	if (reading) {
		f.read(bytes, sizeof bytes);
		if (!f) return false;
		*digest = std::string(bytes, sizeof bytes);
		if (digest->find_first_not_of('\0') == std::string::npos) {
			digest->clear();
			return true;
		}
		return std::all_of(digest->begin(), digest->end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)); });
	}
	if (!digest->empty() && digest->size() != sizeof bytes) return false;
	f.write(digest->empty() ? bytes : digest->data(), sizeof bytes);
	return static_cast<bool>(f);
}
bool CacheStatus(std::fstream& f, sslm_status* status, bool reading) {
	int32_t value = reading ? 0 : static_cast<int32_t>(*status);
	if (reading) f.read(reinterpret_cast<char*>(&value), sizeof value);
	else f.write(reinterpret_cast<const char*>(&value), sizeof value);
	if (!f || (value != SSLM_OK && value != PKV_KV_BUDGET_EXCEEDED)) return false;
	if (reading) *status = static_cast<sslm_status>(value);
	return true;
}
bool CohortCache(const Fixture& fx, const std::string& path, const std::string& head, CohortRun* r, bool reading) {
	std::fstream f(path, std::ios::binary | (reading ? std::ios::in : std::ios::out | std::ios::trunc));
	if (!f) return false;
	if (reading) {
		std::string magic, sha, saved_head, executable;
		if (!GetTextLine(f, magic) || !GetTextLine(f, sha) || !GetTextLine(f, saved_head) || !GetTextLine(f, executable) ||
		    magic != "PKVC6-CACHE-4" || sha != fx.sha || saved_head != head || executable != ExecutableSha()) return false;
		r->built = true;
		r->tokens.resize(kSeqs);
		r->mid.resize(kSeqs);
		r->lim.resize(kSeqs);
		r->mid_rows.resize(kSeqs);
		r->lim_rows.resize(kSeqs);
		r->end.resize(kSeqs);
	} else {
		f << "PKVC6-CACHE-4\n" << fx.sha << "\n" << head << "\n" << ExecutableSha() << "\n";
	}
	for (int i = 0; i < kSeqs; ++i) {
		if (!CacheTokens(f, &r->tokens[i], reading) || !CacheVector(f, &r->mid[i], reading) ||
		    !CacheVector(f, &r->lim[i], reading) || !CacheDigest(f, &r->mid_rows[i], reading) ||
		    !CacheDigest(f, &r->lim_rows[i], reading) || !CacheStatus(f, &r->end[i], reading)) return false;
		if (reading && (r->tokens[i].size() != 1u + kBudget || r->mid[i].empty() || r->lim[i].empty() ||
		                r->end[i] != PKV_KV_BUDGET_EXCEEDED)) return false;
	}
	if (reading) return f.peek() == std::char_traits<char>::eof();
	f.flush();
	return static_cast<bool>(f);
}

std::string ShardPath(const std::string& dir, const Fixture& fx, const std::string& head, int shard) {
	return dir + "/cohort-" + fx.stem + "-" + fx.sha + "-" + head + "-" + ExecutableSha() + "-shard" + std::to_string(shard) + ".bin";
}
std::string CohortPath(const std::string& dir, const Fixture& fx, const std::string& head) {
	return dir + "/cohort-" + fx.stem + "-" + fx.sha + "-" + head + "-" + ExecutableSha() + ".bin";
}
bool CohortShardCache(const Fixture& fx, const std::string& path, const std::string& head, int shard,
                     CohortRun* r, bool reading) {
	std::fstream f(path, std::ios::binary | (reading ? std::ios::in : std::ios::out | std::ios::trunc));
	if (!f) return false;
	if (reading) {
		std::string magic, sha, saved_head, executable, saved_shard;
		if (!GetTextLine(f, magic) || !GetTextLine(f, sha) || !GetTextLine(f, saved_head) ||
		    !GetTextLine(f, executable) || !GetTextLine(f, saved_shard) || magic != "PKVC6-SHARD-4" || sha != fx.sha ||
		    saved_head != head || executable != ExecutableSha() || saved_shard != std::to_string(shard)) return false;
		r->built = true;
		r->tokens.resize(kSeqs);
		r->mid.resize(kSeqs);
		r->lim.resize(kSeqs);
		r->mid_rows.resize(kSeqs);
		r->lim_rows.resize(kSeqs);
		r->end.resize(kSeqs);
	} else {
		f << "PKVC6-SHARD-4\n" << fx.sha << "\n" << head << "\n" << ExecutableSha() << "\n" << shard << "\n";
	}
	for (int i = 0; i < kSeqs; ++i) {
		if (!CacheTokens(f, &r->tokens[i], reading) || !CacheVector(f, &r->mid[i], reading) ||
		    !CacheVector(f, &r->lim[i], reading) || !CacheDigest(f, &r->mid_rows[i], reading) ||
		    !CacheDigest(f, &r->lim_rows[i], reading) || !CacheStatus(f, &r->end[i], reading)) return false;
		if (reading) {
			const bool selected = i >= shard * 10 && i < (shard + 1) * 10;
			if (selected != (r->tokens[i].size() == 1u + kBudget && !r->mid[i].empty() && !r->lim[i].empty() &&
			                 !r->mid_rows[i].empty() && !r->lim_rows[i].empty() && r->end[i] == PKV_KV_BUDGET_EXCEEDED)) return false;
			if (!selected && (!r->tokens[i].empty() || !r->mid[i].empty() || !r->lim[i].empty() ||
			                  !r->mid_rows[i].empty() || !r->lim_rows[i].empty() || r->end[i] != SSLM_OK)) return false;
		}
	}
	if (reading) return f.peek() == std::char_traits<char>::eof();
	f.flush();
	return static_cast<bool>(f);
}

void CellCacheVersion() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const std::string exe = ExecutableSha();
	PKV_CHECK_MSG(exe.size() == 64, "cohort executable hash missing");
	std::printf("cohort executable sha256=%s\n", exe.c_str());
#if defined(_WIN32)
	const auto pid = GetCurrentProcessId();
#else
	const auto pid = getpid();
#endif
	const auto dir = std::filesystem::temp_directory_path() / ("pkv-cache-v4-" + std::to_string(pid));
	std::filesystem::create_directories(dir);
	const std::string path = CohortPath(dir.string(), fx, "test-head");
	const std::string shard = ShardPath(dir.string(), fx, "test-head", 0);
	PKV_CHECK(path.find(exe) != std::string::npos && shard.find(exe) != std::string::npos);
	CohortRun r;
	r.tokens.resize(kSeqs);r.mid.resize(kSeqs);r.lim.resize(kSeqs);
	r.mid_rows.resize(kSeqs);r.lim_rows.resize(kSeqs);r.end.resize(kSeqs,PKV_KV_BUDGET_EXCEEDED);
	for(int i=0;i<kSeqs;++i){r.tokens[i].assign(1+kBudget,0);r.mid[i]={1};r.lim[i]={1};
		r.mid_rows[i]=std::string(64,'a');r.lim_rows[i]=std::string(64,'b');}
	PKV_CHECK(CohortCache(fx,path,"test-head",&r,false));
	CohortRun loaded;
	PKV_CHECK(CohortCache(fx,path,"test-head",&loaded,true));
	{
		std::fstream f(path,std::ios::binary|std::ios::in|std::ios::out);
		std::string line;
		for(int i=0;i<3;++i) PKV_CHECK(GetTextLine(f,line));
		const auto at=f.tellg();
		f.seekp(at);f.put(exe[0]=='0'?'1':'0');f.flush();
		PKV_CHECK(!CohortCache(fx,path,"test-head",&loaded,true));
		f.seekp(at);f.put(exe[0]);
	}
	{
		std::fstream f(path,std::ios::binary|std::ios::in|std::ios::out);
		f.seekp(std::string("PKVC6-CACHE-").size());f.put('1');
	}
	PKV_CHECK(!CohortCache(fx,path,"test-head",&loaded,true));
	for(int i=10;i<kSeqs;++i){r.tokens[i].clear();r.mid[i].clear();r.lim[i].clear();
		r.mid_rows[i].clear();r.lim_rows[i].clear();r.end[i]=SSLM_OK;}
	PKV_CHECK(CohortShardCache(fx,shard,"test-head",0,&r,false));
	PKV_CHECK(CohortShardCache(fx,shard,"test-head",0,&loaded,true));
	{
		std::fstream f(shard,std::ios::binary|std::ios::in|std::ios::out);
		std::string line;
		for(int i=0;i<3;++i) PKV_CHECK(GetTextLine(f,line));
		const auto at=f.tellg();
		f.seekp(at);f.put(exe[0]=='0'?'1':'0');f.flush();
		PKV_CHECK(!CohortShardCache(fx,shard,"test-head",0,&loaded,true));
		f.seekp(at);f.put(exe[0]);
	}
	{
		std::fstream f(shard,std::ios::binary|std::ios::in|std::ios::out);
		f.seekp(std::string("PKVC6-SHARD-").size());f.put('1');
	}
	PKV_CHECK(!CohortShardCache(fx,shard,"test-head",0,&loaded,true));
	std::filesystem::remove(path);std::filesystem::remove(shard);std::filesystem::remove(dir);
}

// The cohort in exactly CohortPages (1,842), once per process. The optional on-disk checkpoint
// lets each release cell finish within a bounded process and is written only after all 50 complete.
const CohortRun& Cohort(const Fixture& fx) {
	static std::map<std::string, CohortRun> runs;
	auto it = runs.find(fx.stem);
	if (it != runs.end()) return it->second;
	const char* dir = std::getenv("SUPERSLM_PAGED_KV_COHORT_CACHE_DIR");
	const char* head = std::getenv("SUPERSLM_PAGED_KV_COHORT_HEAD");
	const bool cache = dir && *dir;
	if (cache) PKV_CHECK_MSG(head && *head, "C6 cohort cache requires the source HEAD");
	const std::string path = cache && head && *head ? CohortPath(dir, fx, head) : "";
	if (!path.empty()) {
		std::ifstream check(path, std::ios::binary);
		if (check) {
			CohortRun saved;
			const bool valid = CohortCache(fx, path, head, &saved, true);
			PKV_CHECK_MSG(valid, "C6 cohort cache invalid: %s", path.c_str());
			if (valid) {
				std::printf("C6 cohort checkpoint loaded: %s\n", path.c_str());
				return runs[fx.stem] = std::move(saved);
			}
			return runs[fx.stem] = CohortRun{};
		}
		// The five 10.1 shards are the other checkpoint form. Each covers disjoint sequence indices;
		// they bind the same source HEAD and model hash and are checked for exact coverage here.
		const std::string first = ShardPath(dir, fx, head, 0);
		if (std::ifstream(first, std::ios::binary)) {
			CohortRun joined;
			joined.built = true;
			joined.tokens.resize(kSeqs);
			joined.mid.resize(kSeqs);
			joined.lim.resize(kSeqs);
			joined.mid_rows.resize(kSeqs);
			joined.lim_rows.resize(kSeqs);
			joined.end.resize(kSeqs, SSLM_OK);
			for (int shard = 0; shard < 5; ++shard) {
				const std::string part = ShardPath(dir, fx, head, shard);
				CohortRun one;
				const bool valid = CohortShardCache(fx, part, head, shard, &one, true);
				PKV_CHECK_MSG(valid, "C6 cohort shard invalid: %s", part.c_str());
				if (!valid) return runs[fx.stem] = CohortRun{};
				for (int i = shard * 10; i < (shard + 1) * 10; ++i) {
					joined.tokens[i] = std::move(one.tokens[i]);
					joined.mid[i] = std::move(one.mid[i]);
			joined.lim[i] = std::move(one.lim[i]);
			joined.mid_rows[i] = std::move(one.mid_rows[i]);
			joined.lim_rows[i] = std::move(one.lim_rows[i]);
			joined.end[i] = one.end[i];
				}
				std::printf("C6 cohort shard loaded: %s\n", part.c_str());
			}
			return runs[fx.stem] = std::move(joined);
		}
	}
	if (cache && std::getenv("SUPERSLM_PAGED_KV_COHORT_REQUIRE_CACHE")) {
		PKV_CHECK_MSG(false, "C6 cohort checkpoint absent for %s at source HEAD %s", fx.stem.c_str(), head ? head : "(unset)");
		return runs[fx.stem] = CohortRun{};
	}
	CohortRun fresh = RunCohort(fx, static_cast<uint32_t>(CohortPages(fx)), true);
	if (!path.empty() && fresh.built && fresh.first_refused < 0 && fresh.tokens.size() == kSeqs &&
	    fresh.mid.size() == kSeqs && fresh.lim.size() == kSeqs && fresh.end.size() == kSeqs) {
		const std::string partial = path + ".partial";
		const bool written = CohortCache(fx, partial, head, &fresh, false) && std::rename(partial.c_str(), path.c_str()) == 0;
		PKV_CHECK_MSG(written, "C6 cohort checkpoint write failed: %s", path.c_str());
		if (written) std::printf("C6 cohort checkpoint written: %s\n", path.c_str());
	}
	return runs[fx.stem] = std::move(fresh);
}

// ================================================================================================
// 10.1: every sequence emits 1 + 512 = 513 tokens, then SSLM_KV_BUDGET_EXCEEDED, and its tokens equal
// the same prompt run with no sharing on v1.11.0.
// ================================================================================================
void Cell101On(const Fixture& fx) {
	if (!fx.ok) return;
	if (const char* selector = std::getenv("SUPERSLM_PAGED_KV_COHORT_SHARD")) {
		const int shard = std::atoi(selector);
		PKV_CHECK_MSG(std::string(selector) == std::to_string(shard) && shard >= 0 && shard < 5,
		              "10.1 invalid shard selector %s", selector);
		if (std::string(selector) != std::to_string(shard) || shard < 0 || shard >= 5) return;
		const char* dir = std::getenv("SUPERSLM_PAGED_KV_COHORT_CACHE_DIR");
		const char* head = std::getenv("SUPERSLM_PAGED_KV_COHORT_HEAD");
		PKV_CHECK_MSG(dir && *dir && head && *head, "10.1 shard requires cache directory and source HEAD");
		if (!dir || !*dir || !head || !*head) return;
		CohortRun r = RunCohortShard(fx, shard);
		bool valid = r.built && r.first_refused < 0 && r.creates_refused == 0;
		PKV_CHECK_MSG(valid, "10.1 shard %d: full cohort admission failed", shard);
		for (int i = shard * 10; i < (shard + 1) * 10; ++i) {
			const std::vector<int32_t> ref = RefTokens(fx, PersonaOf(i));
			const bool tokens = r.tokens[i].size() == 1u + kBudget && r.end[i] == PKV_KV_BUDGET_EXCEEDED &&
			                    r.tokens[i] == ref && !r.mid[i].empty() && !r.lim[i].empty();
			const bool rows = CheckCohortRows(fx, PersonaOf(i), r.mid_rows[i], r.lim_rows[i]);
			PKV_CHECK_MSG(tokens, "10.1 shard %d: sequence %d differs in tokens or did not reach its limit", shard, i);
			PKV_CHECK_MSG(rows, "10.1 shard %d: sequence %d K/V row digest differs from v1.11.0", shard, i);
			valid = valid && tokens && rows;
		}
		if (valid) {
			const std::string path = ShardPath(dir, fx, head, shard);
			const std::string partial = path + ".partial";
			const bool written = CohortShardCache(fx, partial, head, shard, &r, false) &&
			                     std::rename(partial.c_str(), path.c_str()) == 0;
			PKV_CHECK_MSG(written, "10.1 shard %d: checkpoint write failed: %s", shard, path.c_str());
			if (written) std::printf("C6 cohort shard written: %s\n", path.c_str());
		}
		return;
	}
	const CohortRun& r = Cohort(fx);
	PKV_CHECK_MSG(r.built && r.first_refused < 0 && r.tokens.size() == kSeqs && r.mid_rows.size() == kSeqs &&
	              r.lim_rows.size() == kSeqs, "10.1 %s: the cohort did not build (%zu sequences)",
	              fx.stem.c_str(), r.tokens.size());
	if (r.tokens.size() != kSeqs || r.mid_rows.size() != kSeqs || r.lim_rows.size() != kSeqs) return;
	std::vector<std::vector<int32_t>> ref;
	for (int p = 0; p < kPersonas; ++p) ref.push_back(RefTokens(fx, p));
	for (size_t i = 0; i < r.tokens.size(); ++i) {
		PKV_CHECK_EQ(r.tokens[i].size(), 1 + kBudget);  // §3.4
		PKV_CHECK_EQ(r.end[i], PKV_KV_BUDGET_EXCEEDED);
		// kills: anything sharing changes in what a user receives (a shared page read wrongly, a
		// tail copied short, a begin_from child missing its parent's state)
		PKV_CHECK_MSG(r.tokens[i] == ref[static_cast<size_t>(PersonaOf(static_cast<int>(i)))],
		              "10.1 %s: sequence %zu's tokens differ from the no-sharing run on v1.11.0", fx.stem.c_str(), i);
		PKV_CHECK_MSG(CheckCohortRows(fx, PersonaOf(static_cast<int>(i)), r.mid_rows[i], r.lim_rows[i]),
		              "10.1 %s: sequence %zu K/V rows differ from both reference record points", fx.stem.c_str(), i);
	}
}

// ================================================================================================
// 10.2: one 8-sequence cohort on a nested prefix (the world -> persona 0), tokens equal v1.11.0's
// no-sharing run of the same prompt.
// ================================================================================================
void Cell102On(const Fixture& fx) {
	if (!fx.ok) return;
	const int n = 8;
	PagePool pool(fx.model, static_cast<uint32_t>(fx.R(1000) + fx.R(200) + n * fx.R(kBudget)));
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (pool.status != SSLM_OK) return;
	World w = BuildWorld(fx, &pool.pool, 1);
	const std::vector<int32_t> ref = RefTokens(fx, 0);
	std::vector<sslm_seq> seqs;
	for (int i = 0; i < n && w.ok; ++i) {
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, kBudget, &s), SSLM_OK);
		if (!s) break;
		seqs.push_back(s);
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, w.personas[0]), SSLM_OK);
	}
	// decoded as one batch, as a host serves a cohort
	std::vector<std::vector<int32_t>> got(seqs.size());
	std::vector<std::string> mid_rows(seqs.size());
	sslm_decode_params p{};
	p.layer_budget = 1;
	std::vector<int32_t> out(seqs.size());
	sslm_status st = SSLM_OK;
	for (int guard = 0; guard < 1 << 16 && !seqs.empty(); ++guard) {
		st = sslm_decode_step(fx.model, seqs.data(), static_cast<int32_t>(seqs.size()), &p, nullptr, out.data());
		if (st != SSLM_OK) break;
		for (size_t i = 0; i < seqs.size(); ++i)
			if (out[i] >= 0) {
				got[i].push_back(out[i]);
				if (got[i].size() == static_cast<size_t>(kMidAt - kOrigin + 1))
					mid_rows[i] = CohortRows(fx, pool.pool, seqs[i], kMidAt);
			}
	}
	PKV_CHECK_EQ(st, PKV_KV_BUDGET_EXCEEDED);
	for (size_t i = 0; i < got.size(); ++i) {
		PKV_CHECK_EQ(got[i].size(), 1 + kBudget);
		PKV_CHECK_MSG(got[i] == ref, "10.2 %s: sequence %zu's tokens differ from v1.11.0's", fx.stem.c_str(), i);
		const std::string lim_rows = CohortRows(fx, pool.pool, seqs[i], kOrigin + kBudget);
		PKV_CHECK_MSG(CheckCohortRows(fx, 0, mid_rows[i], lim_rows),
		              "10.2 %s: sequence %zu K/V rows differ from both reference record points", fx.stem.c_str(), i);
	}
	for (auto it = seqs.rbegin(); it != seqs.rend(); ++it) sslm_seq_release(*it);
	w.Release();
}

// 10.4's construction peak. The plan's executed peak of 203 is the world's 63 frozen pages plus ten
// open personas at R(200) = 14 each: every begin_from runs before any persona is frozen (BuildWorld
// freezes each persona before the next begin_from, which peaks at 63 + 9 * 13 + 14 = 194). The
// personas are then filled and frozen and the world released, leaving the chain's 192 pages.
// Returns the index of the first refused begin_from (-1 if none) and its status.
int BuildWorldAllOpen(const Fixture& fx, sslm_kv_pool* pool, World* w, sslm_status* refusal) {
	*refusal = SSLM_OK;
	sslm_status st = sslm_prefix_begin_budgeted(fx.model, pool, 1000, &w->world);
	if (st == SSLM_OK) st = PrefixPrefillAll(fx.model, w->world, WorldTokens(), kChunk);
	if (st == SSLM_OK) st = sslm_prefix_freeze(w->world);
	PKV_CHECK_MSG(st == SSLM_OK, "10.4 peak: the world: status %d", static_cast<int>(st));
	if (st != SSLM_OK) return -2;
	for (int i = 0; i < kPersonas; ++i) {
		sslm_prefix p = nullptr;
		st = sslm_prefix_begin_from(w->world, 200, &p);
		if (st != SSLM_OK) {
			*refusal = st;
			return i;
		}
		w->personas.push_back(p);
	}
	for (int i = 0; i < kPersonas && st == SSLM_OK; ++i) {
		st = PrefixPrefillAll(fx.model, w->personas[static_cast<size_t>(i)], PersonaTokens(i), kChunk);
		if (st == SSLM_OK) st = sslm_prefix_freeze(w->personas[static_cast<size_t>(i)]);
		PKV_CHECK_MSG(st == SSLM_OK, "10.4 peak: persona %d: status %d", i, static_cast<int>(st));
	}
	if (st == SSLM_OK) {
		st = sslm_prefix_release(w->world);
		w->world = nullptr;
	}
	w->ok = st == SSLM_OK;
	return -1;
}

void Peak104(const Fixture& fx) {
	const CohortFigures* f = FiguresFor(fx, "10.4 peak");
	if (!f) return;
	const int64_t peak = f->peak;
	// the cross-check: §3.4's peak, written here, equals the engine-executed row
	PKV_CHECK_EQ(PrefixPages(fx, 1000) + kPersonas * fx.R(200), peak);
	{
		PagePool pool(fx.model, static_cast<uint32_t>(peak));
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		if (pool.status != SSLM_OK) return;
		World w;
		sslm_status refusal = SSLM_OK;
		const int refused = BuildWorldAllOpen(fx, &pool.pool, &w, &refusal);
		// kills: a begin_from that draws more than R(200) or copies the world's pages (refused at 203)
		PKV_CHECK_MSG(refused == -1 && w.ok, "10.4 peak %s: a pool of %lld refused begin_from %d (status %d)",
		              fx.stem.c_str(), static_cast<long long>(peak), refused, static_cast<int>(refusal));
		// The end of construction, graded by one-state admission: the chain holds ChainPages (62 + 10 * 13 = 192 at
		// B = 16, 15 + 10 * 4 = 55 at 64, 7 + 10 * 3 = 37 at 128), so the peak's pool has k = peak - ChainPages free.
		// kills: freeze not returning a persona's spare reserve page; the world's release keeping its
		// unshared tail page, or freeing a page the personas share (more free)
		// The fill probe resolves k >= 2 (its smallest create is 2 pages). At B = 128, k = 38 - 37 = 1: then a 2-page
		// create must be refused (at most 1 free), and with the last persona released (its own pages, PrefixPages(1,200)
		// - SharedPages(1,000), go back: the others still hold the shared ones) the probe grades k + own exactly.
		const int64_t k = peak - ChainPages(fx, kPersonas);
		if (w.ok && k >= 2) ProbeExactlyFreeOneState(fx, &pool.pool, k);
		if (w.ok && k < 2) {
			sslm_seq two = nullptr;
			PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 1, &two), SSLM_KV_POOL_EXHAUSTED);
			if (two) sslm_seq_release(two);
			const int64_t own = PrefixPages(fx, kOrigin) - SharedPages(fx, 1000);
			sslm_prefix_release(w.personas.back());
			w.personas.back() = nullptr;
			ProbeExactlyFreeOneState(fx, &pool.pool, k + own);
		}
		w.Release();
	}
	{
		PagePool pool(fx.model, static_cast<uint32_t>(peak - 1));
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		if (pool.status != SSLM_OK) return;
		World w;
		sslm_status refusal = SSLM_OK;
		const int refused = BuildWorldAllOpen(fx, &pool.pool, &w, &refusal);
		// kills: a begin_from that draws less than R(200) (the tenth admitted one page below the peak)
		g_bounds.peak_short_failed += !(refused == kPersonas - 1 && refusal == SSLM_KV_POOL_EXHAUSTED);
		PKV_CHECK_MSG(refused == kPersonas - 1 && refusal == SSLM_KV_POOL_EXHAUSTED,
		              "10.4 peak %s: a pool of %lld refused begin_from %d (status %d), want the last, exhausted",
		              fx.stem.c_str(), static_cast<long long>(peak - 1), refused, static_cast<int>(refusal));
		w.Release();
	}
}

// ================================================================================================
// 10.4: graded by host-visible end state. 1,842 pages: every create, begin_from and adopt succeeds; 1,841:
// exactly one create is refused, the 50th, with SSLM_KV_POOL_EXHAUSTED. Contrast: a block pool in the
// same bytes admits floor(bytes / block) = 7.
// ================================================================================================
void Cell104On(const Fixture& fx) {
	if (!fx.ok) return;
	const CohortFigures* f = FiguresFor(fx, "10.4");
	if (!f) return;
	const int64_t N = f->pool;
	PKV_CHECK_EQ(CohortPages(fx), N);  // the cross-check: §3.4 written here equals the engine-executed row
	{
		const CohortRun r = RunCohort(fx, static_cast<uint32_t>(N), false, false);
		// kills: any draw beyond §3.4's (an adopt that draws, a begin_from that copies the world)
		PKV_CHECK_MSG(r.built && r.first_refused < 0 && r.creates_refused == 0, "10.4 %s: a pool of %lld refused create %d (status %d)",
		              fx.stem.c_str(), static_cast<long long>(N), r.first_refused, static_cast<int>(r.refusal));
	}
	{
		const CohortRun r = RunCohort(fx, static_cast<uint32_t>(N - 1), false, false);
		// kills: a reservation one page short anywhere in the cohort (admitted one page below the pool)
		g_bounds.pool_short_failed +=
		    !(r.built && r.first_refused == kSeqs - 1 && r.creates_refused == 1 && r.refusal == SSLM_KV_POOL_EXHAUSTED);
		PKV_CHECK_MSG(r.built && r.first_refused == kSeqs - 1 && r.creates_refused == 1 && r.refusal == SSLM_KV_POOL_EXHAUSTED,
		              "10.4 %s: a pool of %lld refused %d creates, the first at index %d (status %d)", fx.stem.c_str(),
		              static_cast<long long>(N - 1), r.creates_refused, r.first_refused, static_cast<int>(r.refusal));
	}
	// The contrast: a v1.11.0-style block pool in the same bytes. The block count is the §3.1 formula
	// (block = ceil(cap/B) pages); the old verb admits exactly that many and refuses one more block.
	const size_t bytes = static_cast<size_t>(N) * fx.PageBytes();
	const size_t block = static_cast<size_t>(fx.CapPages()) * fx.PageBytes();
	const uint32_t blocks = static_cast<uint32_t>(bytes / block);
	PKV_CHECK_EQ(blocks, f->contrast);
	PKV_CHECK_EQ(block, sslm_kv_block_size(fx.model));
	AlignedBuf mem(bytes + sslm_kv_pool_overhead_size(fx.model, blocks + 1));
	sslm_kv_pool lp = nullptr;
	PKV_CHECK_EQ(sslm_kv_pool_create(fx.model, mem.p, bytes + sslm_kv_pool_overhead_size(fx.model, blocks + 1), blocks + 1, &lp),
	             SSLM_BUFFER_TOO_SMALL);
	if (lp) sslm_kv_pool_destroy(lp);
	lp = nullptr;
	PKV_CHECK_EQ(sslm_kv_pool_create(fx.model, mem.p, mem.n, blocks, &lp), SSLM_OK);
	if (lp) {
		sslm_status refusal = SSLM_OK;
		PKV_CHECK_EQ(CountLegacyCreates(fx.model, &lp, &refusal), blocks);
		PKV_CHECK_EQ(refusal, SSLM_KV_POOL_EXHAUSTED);
		sslm_kv_pool_destroy(lp);
	}
	// Box leg: the same block-pool count on the v1.11.0 binary (the timing harness's two-binary run).
	Peak104(fx);
}

// ================================================================================================
// 10.6 and 10.8: the cohort's blobs, saved at 1,456 and at the limit (the 1,842 run), restored into a
// fresh pool holding the rebuilt world and personas.
// ================================================================================================
struct RestoreRun {
	int first_refused = -1;
	int refused = 0;
	sslm_status refusal = SSLM_OK;
	uint64_t shared_diag = 0;
};

// Restores the 50 blobs in order into a pool of `pages` (with persona handles, or without), then calls
// `after` with the pool, the personas and the restored sequences before tearing down.
template <class After>
RestoreRun RestoreCohort(const Fixture& fx, uint32_t pages, const std::vector<std::vector<uint8_t>>& blobs, bool handles,
                         After after) {
	RestoreRun r;
	PagePool pool(fx.model, pages);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (pool.status != SSLM_OK) return r;
	World w = BuildWorld(fx, &pool.pool, kPersonas);
	std::vector<sslm_seq> seqs;
	for (size_t i = 0; i < blobs.size() && w.ok; ++i) {
		sslm_seq s = nullptr;
		uint32_t shared = 0;
		const sslm_prefix h = handles ? w.personas[static_cast<size_t>(PersonaOf(static_cast<int>(i)))] : nullptr;
		const sslm_status st = sslm_seq_restore_shared(fx.model, &pool.pool, blobs[i].data(), blobs[i].size(), h, &s, &shared);
		r.shared_diag += shared;
		if (st != SSLM_OK) {
			if (r.first_refused < 0) {
				r.first_refused = static_cast<int>(i);
				r.refusal = st;
			}
			++r.refused;
			seqs.push_back(nullptr);
			continue;
		}
		seqs.push_back(s);
	}
	after(&pool.pool, w, seqs);
	for (auto it = seqs.rbegin(); it != seqs.rend(); ++it)
		if (*it) sslm_seq_release(*it);
	w.Release();
	return r;
}

void Cell106On(const Fixture& fx) {
	if (!fx.ok) return;
	const CohortFigures* f = FiguresFor(fx, "10.6");  // first, so a wrong B is the red reason, not the run
	if (!f) return;
	const CohortRun& run = Cohort(fx);
	if (run.mid.size() != kSeqs || run.lim.size() != kSeqs) {
		PKV_CHECK_MSG(false, "10.6 %s: the cohort run did not save 50 sequences", fx.stem.c_str());
		return;
	}
	const int64_t N = f->pool;
	PKV_CHECK_EQ(CohortPages(fx), N);  // the cross-check
	for (int pass = 0; pass < 2; ++pass) {
		const char* name = pass == 0 ? "1,456" : "the limit";
		const std::vector<std::vector<uint8_t>>& blobs = pass == 0 ? run.mid : run.lim;
		for (size_t i = 0; i < blobs.size(); ++i)
			PKV_CHECK_EQ(ParseBlobWithHidden(blobs[i], HiddenSize(fx)).context_length, pass == 0 ? kMidAt : kOrigin + kBudget);
		const RestoreRun ok = RestoreCohort(fx, static_cast<uint32_t>(N), blobs, true,
		                                    [&](sslm_kv_pool*, World&, std::vector<sslm_seq>& seqs) {
			                                    for (size_t i = 0; i < seqs.size(); ++i) {
				                                    if (!seqs[i]) continue;
				                                    sslm_status end = SSLM_OK;
				                                    const std::vector<int32_t> t = RunToRefusal(fx.model, seqs[i], &end);
				                                    const std::vector<int32_t>& all = run.tokens[i];
				                                    const size_t from = pass == 0 ? static_cast<size_t>(kMidAt - kOrigin + 1) : all.size();
				                                    const std::vector<int32_t> want(all.begin() + static_cast<std::ptrdiff_t>(std::min(from, all.size())), all.end());
				                                    // kills: a restore that loses the budget or the origin (runs past or
				                                    // short of the limit), or shares a page it should not
				                                    PKV_CHECK_MSG(t == want && end == PKV_KV_BUDGET_EXCEEDED,
				                                                  "10.6 %s (%s): sequence %zu continues with %zu tokens (status %d), the unsaved run with %zu",
				                                                  fx.stem.c_str(), name, i, t.size(), static_cast<int>(end), want.size());
			                                    }
		                                    });
		// kills: a restore that does not re-share (needs the 108 private pages, refused early at 1,842)
		PKV_CHECK_MSG(ok.first_refused < 0, "10.6 %s (%s): a pool of %lld refused restore %d (status %d)", fx.stem.c_str(), name,
		              static_cast<long long>(N), ok.first_refused, static_cast<int>(ok.refusal));
		std::printf("10.6 %s (%s): *out_shared_pages summed over 50 restores = %llu (diagnostic)\n", fx.stem.c_str(), name,
		            static_cast<unsigned long long>(ok.shared_diag));
		const RestoreRun short1 = RestoreCohort(fx, static_cast<uint32_t>(N - 1), blobs, true,
		                                        [](sslm_kv_pool*, World&, std::vector<sslm_seq>&) {});
		// kills: a shared restore drawing less than R(512) (admitted at 1,841)
		PKV_CHECK_MSG(short1.first_refused == kSeqs - 1 && short1.refused == 1 && short1.refusal == SSLM_KV_POOL_EXHAUSTED,
		              "10.6 %s (%s): a pool of %lld refused %d restores, the first at %d (status %d)", fx.stem.c_str(), name,
		              static_cast<long long>(N - 1), short1.refused, short1.first_refused, static_cast<int>(short1.refusal));
	}
	// The unsaved run's tokens equal v1.11.0's too (10.1's check, restated for this run).
	for (size_t i = 0; i < run.tokens.size(); ++i)
		PKV_CHECK_MSG(run.tokens[i] == RefTokens(fx, PersonaOf(static_cast<int>(i))), "10.6 %s: the unsaved run's sequence %zu differs from v1.11.0",
		              fx.stem.c_str(), i);
}

void Cell108On(const Fixture& fx) {
	if (!fx.ok) return;
	const CohortFigures* f = FiguresFor(fx, "10.8");  // first, so a wrong B is the red reason, not the run
	if (!f) return;
	const CohortRun& run = Cohort(fx);
	if (run.mid.size() != kSeqs || run.lim.size() != kSeqs) {
		PKV_CHECK_MSG(false, "10.8 %s: the cohort run did not save 50 sequences", fx.stem.c_str());
		return;
	}
	const int64_t N = f->private_pool;
	const int64_t further = f->further;
	// the cross-checks: §3.4 / §3.7 written here equal the engine-executed row
	PKV_CHECK_EQ(PrivateCohortPages(fx), N);
	PKV_CHECK_EQ((PrivateCohortPages(fx) - CohortPages(fx)) / fx.R(kBudget), further);
	for (int pass = 0; pass < 2; ++pass) {
		const char* name = pass == 0 ? "1,456" : "the limit";
		const std::vector<std::vector<uint8_t>>& blobs = pass == 0 ? run.mid : run.lim;
		int admitted = -1;
		sslm_status refusal = SSLM_OK;
		const RestoreRun ok = RestoreCohort(fx, static_cast<uint32_t>(N), blobs, false,
		                                    [&](sslm_kv_pool* pool, World& w, std::vector<sslm_seq>& seqs) {
			                                    for (size_t i = 0; i < seqs.size(); ++i) {
				                                    if (!seqs[i]) continue;
				                                    PKV_CHECK_EQ(sslm_seq_reset(seqs[i]), SSLM_OK);
				                                    PKV_CHECK_EQ(sslm_seq_adopt_prefix(seqs[i], w.personas[static_cast<size_t>(PersonaOf(static_cast<int>(i)))]), SSLM_OK);
			                                    }
			                                    std::vector<sslm_seq> extra;
			                                    for (int guard = 0; guard < 100000; ++guard) {
				                                    sslm_seq s = nullptr;
				                                    refusal = sslm_seq_create_budgeted(fx.model, pool, kBudget, &s);
				                                    if (refusal != SSLM_OK) break;
				                                    extra.push_back(s);
			                                    }
			                                    admitted = static_cast<int>(extra.size());
			                                    for (auto it = extra.rbegin(); it != extra.rend(); ++it) sslm_seq_release(*it);
		                                    });
		// kills: a private restore drawing more than R + E (refused early at 5,592)
		PKV_CHECK_MSG(ok.first_refused < 0, "10.8 %s (%s): a pool of %lld refused restore %d (status %d)", fx.stem.c_str(), name,
		              static_cast<long long>(N), ok.first_refused, static_cast<int>(ok.refusal));
		// kills: materialized pages sent to the reserve at the reset (0 further creates), or kept (0)
		PKV_CHECK_MSG(admitted == further && refusal == SSLM_KV_POOL_EXHAUSTED,
		              "10.8 %s (%s): %d further budget-512 creates admitted after the re-adopts (refusal %d), expected %lld",
		              fx.stem.c_str(), name, admitted, static_cast<int>(refusal), static_cast<long long>(further));
		const RestoreRun short1 = RestoreCohort(fx, static_cast<uint32_t>(N - 1), blobs, false,
		                                        [](sslm_kv_pool*, World&, std::vector<sslm_seq>&) {});
		// kills: a private restore drawing less than R + E (admitted at 5,591)
		PKV_CHECK_MSG(short1.first_refused == kSeqs - 1 && short1.refused == 1 && short1.refusal == SSLM_KV_POOL_EXHAUSTED,
		              "10.8 %s (%s): a pool of %lld refused %d restores, the first at %d (status %d)", fx.stem.c_str(), name,
		              static_cast<long long>(N - 1), short1.refused, short1.first_refused, static_cast<int>(short1.refusal));
	}
}

// ---- registrations: the box cell on the real artifact, and its cloud twin on the fixture ----------

void Cell101Box() { Cell101On(GetArtifact(k05B, kGeo05B)); }
void Cell101Fixture() { Cell101On(GetFixture("pkv_def")); }
void Cell102Box() { Cell102On(GetArtifact(k15B, kGeo15B)); }
void Cell102Fixture() { Cell102On(GetFixture("pkv_32k")); }
void Cell104Box() { Cell104On(GetArtifact(k05B, kGeo05B)); }
void Cell104Fixture() { Cell104On(GetFixture("pkv_def")); }
void Cell106Box() { Cell106On(GetArtifact(k05B, kGeo05B)); }
void Cell106Fixture() { Cell106On(GetFixture("pkv_def")); }
void Cell108Box() { Cell108On(GetArtifact(k05B, kGeo05B)); }
void Cell108Fixture() { Cell108On(GetFixture("pkv_def")); }

// 10.4/C6:table-mutant (rev 16.25 §27.5 item 5), bench-only: 10.4 on the cloud twin with the lookup forced to
// B = 16's row at the running B. Its checks run muted; the member owes that both of 10.4's lower bounds fail (one page
// below the pool admits the 50th create, one page below the peak admits the tenth persona), which is what separates a
// table not keyed by B. Green on a correct build and on 2ebe5e3 alike: the mutant is the test's own.
void Cell104TableMutant() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	g_bounds = BoundsSeen{};
	State().muted = 1;
	State().muted_failures = 0;
	g_table_mutant = true;
	Cell104On(fx);
	g_table_mutant = false;
	State().muted = 0;
	const int muted = State().muted_failures;
	std::printf("10.4/C6:table-mutant: B = %lld, %d muted failures; pool bound failed %d, peak bound failed %d\n",
	            static_cast<long long>(sslm_kv_page_positions(fx.model)), muted, g_bounds.pool_short_failed,
	            g_bounds.peak_short_failed);
	PKV_CHECK_MSG(g_bounds.pool_short_failed > 0, "the table mutant passes 10.4's pool bound (B = 16's 1,841 refused the 50th)");
	PKV_CHECK_MSG(g_bounds.peak_short_failed > 0, "the table mutant passes 10.4's peak bound (B = 16's 202 refused the tenth)");
}

PKV_CELL("10.1/C6", "C6", Cell101Box);
PKV_CELL("10.1/C6:pkv_def", "C6", Cell101Fixture);
PKV_CELL("10.2/C6", "C6", Cell102Box);
PKV_CELL("10.2/C6:pkv_32k", "C6", Cell102Fixture);
PKV_CELL("10.4/C6", "C6", Cell104Box);
PKV_CELL("10.4/C6:pkv_def", "C6", Cell104Fixture);
PKV_CELL("10.4/C6:table-mutant", "C6", Cell104TableMutant);
PKV_CELL("10.6/C6", "C6", Cell106Box);
PKV_CELL("10.6/C6:pkv_def", "C6", Cell106Fixture);
PKV_CELL("10.8/C6", "C6", Cell108Box);
PKV_CELL("10.8/C6:pkv_def", "C6", Cell108Fixture);
PKV_CELL("cohort/C6:cache-version", "C6", CellCacheVersion);

}  // namespace
