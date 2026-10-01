// Paged-KV plan (rev 16.1) step C1: dimension 10's CPU achievement cells, owned by C6 (§7, §8).
//
//   10.1  the cohort on the 0.5B artifact at cap 4096: world 1,000 -> 10 personas (begin_from) -> 50
//         budget-512 sequences; each emits the §3.4 count, with tokens equal to the same prompts run
//         with no sharing on v1.11.0
//   10.2  the 1.5B artifact at cap 32768: one 8-sequence cohort on a nested prefix, tokens equal v1.11.0
//   10.4  the memory achievement: 1,842 pages admit the cohort, 1,841 refuse exactly the 50th create;
//         a v1.11.0 block pool in the same bytes admits 7
//   10.6  the cohort survives a save and load: 1,842 admits the 50 shared restores, 1,841 refuses the
//         50th, both passes (1,456 and the limit); tokens continue equal to the unsaved run's
//   10.8  the no-handle recovery: 5,592 admits the 50 private restores, 5,591 refuses the 50th; after
//         each re-adopts, exactly 113 further budget-512 creates are admitted
//
// Each cell is registered twice. "<id>/C6" is the box cell: it reads the real artifact from
// SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR (a missing variable or file is a failed check) and its reference
// from v1.11.0_<artifact stem>.ref (the box's R0 run). "<id>/C6:pkv_def" (and ":pkv_32k" for 10.2)
// runs the same construction on the cloud fixture, which has 0.5B's page arithmetic at cap 4096
// (B = 16, 256 pages per cap), so every page count below is the same number.
//
// The no-sharing reference is the scenario "cohort" (records persona<i>_1456 and persona<i>_1712): each
// persona's whole 1,200-token prompt prefilled into its own v1.11.0 block and 513 tokens decoded. The
// flat path through the new code is never the reference (§8). Page counts are computed here from §3.4 /
// §3.7 and graded by admission; *out_shared_pages and the stats verbs are never read for a verdict.

#include "pkv_budget_c_helpers.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

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
};

// Runs the cohort in a page pool of `pages`: every sequence is created, adopts its persona, emits 257
// tokens (to 1,456: saved), and runs to its refusal (saved at the limit).
CohortRun RunCohort(const Fixture& fx, uint32_t pages, bool keep_blobs) {
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
		sslm_status a = SSLM_OK, b = SSLM_OK;
		std::vector<int32_t> t = DecodeN(fx.model, s, kMidAt - kOrigin + 1, &a);
		if (keep_blobs) r.mid.push_back(Save(s));
		const std::vector<int32_t> rest = RunToRefusal(fx.model, s, &b);
		t.insert(t.end(), rest.begin(), rest.end());
		if (keep_blobs) r.lim.push_back(Save(s));
		PKV_CHECK_MSG(a == SSLM_OK, "sequence %d: status %d before 1,456", i, static_cast<int>(a));
		r.tokens.push_back(t);
		r.end.push_back(b);
	}
	for (auto it = seqs.rbegin(); it != seqs.rend(); ++it) sslm_seq_release(*it);
	w.Release();
	return r;
}

// The cohort in exactly CohortPages (1,842), once per artifact: 10.1, 10.4 and 10.6 read the same run.
const CohortRun& Cohort(const Fixture& fx) {
	static std::map<std::string, CohortRun> runs;
	auto it = runs.find(fx.stem);
	if (it != runs.end()) return it->second;
	return runs[fx.stem] = RunCohort(fx, static_cast<uint32_t>(CohortPages(fx)), true);
}

// ================================================================================================
// 10.1: every sequence emits 1 + 512 = 513 tokens, then SSLM_KV_BUDGET_EXCEEDED, and its tokens equal
// the same prompt run with no sharing on v1.11.0.
// ================================================================================================
void Cell101On(const Fixture& fx) {
	if (!fx.ok) return;
	const CohortRun& r = Cohort(fx);
	PKV_CHECK_MSG(r.built && r.first_refused < 0 && r.tokens.size() == kSeqs, "10.1 %s: the cohort did not build (%zu sequences)",
	              fx.stem.c_str(), r.tokens.size());
	std::vector<std::vector<int32_t>> ref;
	for (int p = 0; p < kPersonas; ++p) ref.push_back(RefTokens(fx, p));
	for (size_t i = 0; i < r.tokens.size(); ++i) {
		PKV_CHECK_EQ(r.tokens[i].size(), 1 + kBudget);  // §3.4
		PKV_CHECK_EQ(r.end[i], PKV_KV_BUDGET_EXCEEDED);
		// kills: anything sharing changes in what a user receives (a shared page read wrongly, a
		// tail copied short, a begin_from child missing its parent's state)
		PKV_CHECK_MSG(r.tokens[i] == ref[static_cast<size_t>(PersonaOf(static_cast<int>(i)))],
		              "10.1 %s: sequence %zu's tokens differ from the no-sharing run on v1.11.0", fx.stem.c_str(), i);
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
	sslm_decode_params p{};
	p.layer_budget = 1;
	std::vector<int32_t> out(seqs.size());
	sslm_status st = SSLM_OK;
	for (int guard = 0; guard < 1 << 16 && !seqs.empty(); ++guard) {
		st = sslm_decode_step(fx.model, seqs.data(), static_cast<int32_t>(seqs.size()), &p, nullptr, out.data());
		if (st != SSLM_OK) break;
		for (size_t i = 0; i < seqs.size(); ++i)
			if (out[i] >= 0) got[i].push_back(out[i]);
	}
	PKV_CHECK_EQ(st, PKV_KV_BUDGET_EXCEEDED);
	for (size_t i = 0; i < got.size(); ++i) {
		PKV_CHECK_EQ(got[i].size(), 1 + kBudget);
		PKV_CHECK_MSG(got[i] == ref, "10.2 %s: sequence %zu's tokens differ from v1.11.0's", fx.stem.c_str(), i);
	}
	for (auto it = seqs.rbegin(); it != seqs.rend(); ++it) sslm_seq_release(*it);
	w.Release();
}

// ================================================================================================
// 10.4: graded by host-visible end state. 1,842 pages: every create, begin_from and adopt succeeds; 1,841:
// exactly one create is refused, the 50th, with SSLM_KV_POOL_EXHAUSTED. Contrast: a block pool in the
// same bytes admits floor(bytes / block) = 7.
// ================================================================================================
void Cell104On(const Fixture& fx) {
	if (!fx.ok) return;
	const int64_t N = CohortPages(fx);
	PKV_CHECK_EQ(N, 1842);  // §5's figure, from §3.4 written here
	{
		const CohortRun& r = Cohort(fx);
		// kills: any draw beyond §3.4's (an adopt that draws, a begin_from that copies the world)
		PKV_CHECK_MSG(r.built && r.first_refused < 0 && r.creates_refused == 0, "10.4 %s: a pool of %lld refused create %d (status %d)",
		              fx.stem.c_str(), static_cast<long long>(N), r.first_refused, static_cast<int>(r.refusal));
		for (size_t i = 0; i < r.end.size(); ++i) PKV_CHECK_EQ(r.end[i], PKV_KV_BUDGET_EXCEEDED);
	}
	{
		const CohortRun r = RunCohort(fx, static_cast<uint32_t>(N - 1), false);
		// kills: a reservation one page short anywhere in the cohort (admitted at 1,841)
		PKV_CHECK_MSG(r.built && r.first_refused == kSeqs - 1 && r.creates_refused == 1 && r.refusal == SSLM_KV_POOL_EXHAUSTED,
		              "10.4 %s: a pool of %lld refused %d creates, the first at index %d (status %d)", fx.stem.c_str(),
		              static_cast<long long>(N - 1), r.creates_refused, r.first_refused, static_cast<int>(r.refusal));
	}
	// The contrast: a v1.11.0-style block pool in the same bytes. The block count is the §3.1 formula
	// (block = ceil(cap/B) pages); the old verb admits exactly that many and refuses one more block.
	const size_t bytes = static_cast<size_t>(N) * fx.PageBytes();
	const size_t block = static_cast<size_t>(fx.CapPages()) * fx.PageBytes();
	const uint32_t blocks = static_cast<uint32_t>(bytes / block);
	PKV_CHECK_EQ(blocks, 7);
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
	const CohortRun& run = Cohort(fx);
	if (run.mid.size() != kSeqs || run.lim.size() != kSeqs) {
		PKV_CHECK_MSG(false, "10.6 %s: the cohort run did not save 50 sequences", fx.stem.c_str());
		return;
	}
	const int64_t N = CohortPages(fx);
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
	const CohortRun& run = Cohort(fx);
	if (run.mid.size() != kSeqs || run.lim.size() != kSeqs) {
		PKV_CHECK_MSG(false, "10.8 %s: the cohort run did not save 50 sequences", fx.stem.c_str());
		return;
	}
	const int64_t N = PrivateCohortPages(fx);
	PKV_CHECK_EQ(N, 5592);
	const int64_t free_after = N - CohortPages(fx);
	const int64_t further = free_after / fx.R(kBudget);
	PKV_CHECK_EQ(further, 113);
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

PKV_CELL("10.1/C6", "C6", Cell101Box);
PKV_CELL("10.1/C6:pkv_def", "C6", Cell101Fixture);
PKV_CELL("10.2/C6", "C6", Cell102Box);
PKV_CELL("10.2/C6:pkv_32k", "C6", Cell102Fixture);
PKV_CELL("10.4/C6", "C6", Cell104Box);
PKV_CELL("10.4/C6:pkv_def", "C6", Cell104Fixture);
PKV_CELL("10.6/C6", "C6", Cell106Box);
PKV_CELL("10.6/C6:pkv_def", "C6", Cell106Fixture);
PKV_CELL("10.8/C6", "C6", Cell108Box);
PKV_CELL("10.8/C6:pkv_def", "C6", Cell108Fixture);

}  // namespace
