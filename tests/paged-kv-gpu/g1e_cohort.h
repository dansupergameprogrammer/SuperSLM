// Paged-KV plan (rev 16.2) §7 dimension 10, step G6 (cells 10.3, 10.5, 10.7; written at G1 per §9's G1 row):
// the GPU cohort, its CPU twin, the no-sharing reference and the page figures, shared by g1e_cell_10_3.cpp,
// g1e_cell_10_5.cpp and g1e_cell_10_7.cpp. The GPU twins of tests/paged-kv/c6_dim10_cohort.cpp's 10.1, 10.4,
// 10.6 and 10.8, transcribed for the GPU verbs over g1e's helpers (g1e_helpers.h: Env, Rig, GSeq, Token,
// DecodeN, RunToRefusal, Save, RestoreIntoPool, the fill probe) so they run the same constructions.
//
// The cohort (§5, §7 10.1/10.4): a budget world of 1,000 tokens frozen; 10 personas by begin_from(world, 200),
// each filled and frozen as it is built; the world released; then 50 budget-512 sequences, five per persona, each
// adopting its persona. Origin 1,200, limit 1,712, so each sequence emits 1 + 512 = 513 tokens (§3.4) and then
// SSLM_GPU_KV_BUDGET_EXCEEDED.
//
// LEGS. Each cell runs one leg per process (pkvg_suite jobs "<cell>/fixture" and "<cell>/box"):
//   --leg=fixture  pkv_def (--fixture-dir=DIR, else $SUPERSLM_PAGED_KV_FIXTURE_DIR): the cloud-runnable leg. The
//                  GPU page geometry is kv_block_size (§3.1 as rev 16.2 states it: B = 16) at cap 4,096, the very
//                  geometry the plan's figures were executed at, so every figure of 10.4-10.8 holds on it unchanged
//                  (the CPU twin's later per-B table does not apply to the GPU: the CPU slice moved its page rule
//                  to the head-dimension rule, the GPU did not; see FiguresFor).
//   --leg=box      the real Qwen2.5-0.5B artifact at cap 4,096 (--model05=PATH): the plan's own operand. Its
//                  kv_block_size is 16 (page_bytes 98,304, the 43,690-page ceiling of g1b_g3_pool_ceiling.cpp), so
//                  the same figures apply. Box-only: no real artifact is in the cloud.
// Both legs read --pkv-reference=DIR (default tests/paged-kv/reference, relative to the working directory) for
// v1.11.0_<artifact stem>.ref, R0's no-sharing records "cohort persona<i>_1456" and "..._1712" (the plan's
// "v1.9.0", re-anchored to v1.11.0 by the CPU slice).
//
// TOKENS EQUAL CPU (§7 10.3's verdict operand: "CPU run against the R0 reference"). The CPU twin runs in the same
// process through the shipped CPU C ABI (g1e_cpu_twin.h): the same world and personas in a CPU page pool, one
// budget-512 sequence per persona adopting it, run to its refusal. Its tokens are first checked against the R0
// record (the CPU's own 10.1 operand, re-read here so the chain GPU == CPU == v1.11.0 is closed in one run), and
// the GPU's tokens are then compared with the CPU twin's. Five GPU sequences share each persona's prompt and
// greedy decode, so one CPU sequence per persona is the CPU run of every one of them.
//
// Nothing here reads sslm_gpu_kv_pool_stats or *out_shared_pages for a verdict (§8: residency is graded by end
// state; the shared-page count is printed as a diagnostic only).
#ifndef SSLM_PKV_GPU_G1E_COHORT_H
#define SSLM_PKV_GPU_G1E_COHORT_H

#include "g1e_cpu_twin.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>

namespace g1e {
namespace cohort {

constexpr int kPersonas = 10;
constexpr int kSeqs = 50;  // five per persona
constexpr int32_t kBudget = 512;
constexpr int64_t kWorld = 1000, kPersona = 200;
constexpr int64_t kOrigin = kWorld + kPersona;  // 1,200
constexpr int64_t kMidAt = 1456;                 // 10.6's mid-budget save point
constexpr int64_t kLimit = kOrigin + kBudget;    // 1,712
constexpr size_t kTokens = 1 + static_cast<size_t>(kBudget);  // §3.4: 513

inline int PersonaOf(int seq) { return seq / (kSeqs / kPersonas); }

// ---- arguments ---------------------------------------------------------------------------------

struct Args {
	std::string leg;            // "fixture" or "box"
	std::string fixture_dir;    // --fixture-dir= (or $SUPERSLM_PAGED_KV_FIXTURE_DIR)
	std::string model05;        // --model05= (alias --m05=)
	std::string reference_dir = "tests/paged-kv/reference";
};
inline Args ParseArgs(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);  // --allow-skip
	Args a;
	for (int i = 1; i < argc; ++i) {
		const std::string s = argv[i];
		auto take = [&](const char* flag, std::string* out) {
			const size_t n = std::strlen(flag);
			if (s.compare(0, n, flag) != 0) return false;
			*out = s.substr(n);
			return true;
		};
		if (take("--leg=", &a.leg) || take("--fixture-dir=", &a.fixture_dir) || take("--model05=", &a.model05) ||
		    take("--m05=", &a.model05) || take("--pkv-reference=", &a.reference_dir))
			continue;
	}
	if (a.fixture_dir.empty())
		if (const char* d = std::getenv("SUPERSLM_PAGED_KV_FIXTURE_DIR")) a.fixture_dir = d;
	return a;
}
// The artifact the leg runs on; "" (and a failed check naming the flag) when it was not given. A missing
// artifact is a failed check, never a skip: each leg has exactly one operand.
inline std::string ModelPath(const Args& a, const char* cell) {
	if (a.leg == "fixture") {
		CHECK_MSG(!a.fixture_dir.empty(), "%s/fixture: no fixture directory (--fixture-dir=DIR or SUPERSLM_PAGED_KV_FIXTURE_DIR; "
		                                  "python tools/gen_paged_kv_fixture.py DIR writes it)", cell);
		return a.fixture_dir.empty() ? std::string() : a.fixture_dir + "/pkv_def.sslm";
	}
	if (a.leg == "box") {
		CHECK_MSG(!a.model05.empty(), "%s/box: no --model05=PATH (the real Qwen2.5-0.5B cap-4096 artifact)", cell);
		return a.model05;
	}
	CHECK_MSG(false, "%s: --leg=fixture or --leg=box is required (got '%s')", cell, a.leg.c_str());
	return std::string();
}
inline std::string Stem(const std::string& path) {
	const size_t cut = path.find_last_of("/\\");
	std::string name = cut == std::string::npos ? path : path.substr(cut + 1);
	const size_t dot = name.rfind(".sslm");
	return dot == std::string::npos ? name : name.substr(0, dot);
}

// ---- the figures (§5, §7 10.4-10.8), and the arithmetic that cross-checks them -------------------

// The page counts the plan states, executed at B = 16 and cap 4,096 (§5; §7 10.4, 10.6, 10.8; probes/plan_probes.py):
//   pool 1,842 admits the cohort and 1,841 refuses exactly the 50th create (10.4) or restore (10.6);
//   peak 203: the construction with all ten personas open (10.4's own leg);
//   contrast 7: a v1.9.0 block pool in 1,842 pages' bytes admits floor(bytes / block) = 7 sequences;
//   private 5,592 admits the 50 restores without handles, 5,591 refuses the 50th (10.8);
//   further 113: after every restored sequence resets and re-adopts, budget-512 creates admitted (10.8).
struct Figures {
	int64_t B, cap;
	int64_t pool, peak, contrast, private_pool, further;
};
constexpr Figures kPlanFigures = {16, 4096, 1842, 203, 7, 5592, 113};

// §3.4 / §3.7 written here (c6_dim10_cohort.cpp's ChainPages, CohortPages, PrivateCohortPages): the world's
// floor(1000/B) full pages, held by the personas once the world is released, plus each persona's own pages from
// the world's tail page on.
inline int64_t ChainPages(const Geo& g, int personas) {
	return g.SharedPages(kWorld) + personas * (g.PrefixPages(kOrigin) - g.SharedPages(kWorld));
}
inline int64_t CohortPages(const Geo& g) { return ChainPages(g, kPersonas) + kSeqs * g.R(kBudget); }
inline int64_t PrivateCohortPages(const Geo& g) {
	return ChainPages(g, kPersonas) + kSeqs * (g.R(kBudget) + g.Materialized(kOrigin, kBudget));
}
inline int64_t PeakPages(const Geo& g) { return g.PrefixPages(kWorld) + kPersonas * g.R(kPersona); }

// The plan's row for the GPU's page geometry, asserted equal to the arithmetic above (a formula mutant fails
// against the table, a table mutant against the engine's admission). The GPU's B is the artifact's
// kv_block_size (§3.1, rev 16.2; gpu_1p0.cpp's pool geometry), which is 16 on both legs' artifacts; another B has
// no plan figure and fails here, naming it, before any cohort runs.
inline const Figures* FiguresFor(const Geo& g, const char* cell) {
	CHECK_MSG(g.B == kPlanFigures.B && g.cap == kPlanFigures.cap,
	          "%s: the GPU page geometry is B = %lld at cap %lld; the plan's figures are executed at B = 16, cap 4,096",
	          cell, static_cast<long long>(g.B), static_cast<long long>(g.cap));
	if (g.B != kPlanFigures.B || g.cap != kPlanFigures.cap) return nullptr;
	const Figures& f = kPlanFigures;
	CHECK_MSG(CohortPages(g) == f.pool, "%s: §3.4's cohort pages %lld != the plan's %lld", cell,
	          static_cast<long long>(CohortPages(g)), static_cast<long long>(f.pool));
	CHECK_MSG(PeakPages(g) == f.peak, "%s: §3.4's peak %lld != the plan's %lld", cell, static_cast<long long>(PeakPages(g)),
	          static_cast<long long>(f.peak));
	CHECK_MSG(PrivateCohortPages(g) == f.private_pool, "%s: §3.7's private cohort pages %lld != the plan's %lld", cell,
	          static_cast<long long>(PrivateCohortPages(g)), static_cast<long long>(f.private_pool));
	CHECK_MSG((PrivateCohortPages(g) - CohortPages(g)) / g.R(kBudget) == f.further, "%s: §3.7's further creates != the plan's %lld",
	          cell, static_cast<long long>(f.further));
	return &f;
}

// ---- the no-sharing reference (R0: v1.11.0_<stem>.ref) ------------------------------------------

struct Reference {
	bool ok = false;
	std::vector<std::vector<int32_t>> persona;  // per persona: persona<i>_1456's tokens then persona<i>_1712's (513)
};
inline std::vector<int32_t> ParseTokens(const std::string& s) {
	std::vector<int32_t> t;
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, ','))
		if (!item.empty()) t.push_back(static_cast<int32_t>(std::stol(item)));
	return t;
}
// Reads the record file for `stem`, checks it was recorded against these very artifact bytes (its header's
// sha256), and returns each persona's 513 no-sharing tokens. A missing file, record or mismatch is a failed check.
inline Reference ReadReference(const Args& a, const std::string& stem, const std::vector<uint8_t>& artifact_bytes, const char* cell) {
	Reference r;
	const std::string path = a.reference_dir + "/v1.11.0_" + stem + ".ref";
	std::ifstream in(path);
	CHECK_MSG(bool(in), "%s: no R0 reference %s (--pkv-reference=DIR; tools/build_paged_kv_reference.* writes it)", cell, path.c_str());
	if (!in) return r;
	std::string line, header_sha;
	std::map<std::string, std::pair<int64_t, std::vector<int32_t>>> rec;
	while (std::getline(in, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		if (line.rfind("# writer=", 0) == 0) {
			const size_t at = line.find(" sha256=");
			if (at != std::string::npos) header_sha = line.substr(at + 8);
			continue;
		}
		if (line.rfind("cohort ", 0) != 0) continue;
		std::stringstream ss(line);
		std::string scenario, stage, field;
		ss >> scenario >> stage;
		int64_t L = -1;
		std::vector<int32_t> tokens;
		while (ss >> field) {
			if (field.rfind("L=", 0) == 0) L = std::stoll(field.substr(2));
			if (field.rfind("tokens=", 0) == 0) tokens = ParseTokens(field.substr(7));
		}
		rec[stage] = {L, tokens};
	}
	const std::string have = Sha256Hex(artifact_bytes.data(), artifact_bytes.size());
	CHECK_MSG(header_sha == have, "%s: %s was recorded against an artifact of sha256 %s; this one is %s", cell, path.c_str(),
	          header_sha.c_str(), have.c_str());
	bool all = header_sha == have;
	for (int p = 0; p < kPersonas; ++p) {
		const std::string tag = "persona" + std::to_string(p);
		const auto a1 = rec.find(tag + "_1456"), a2 = rec.find(tag + "_1712");
		const bool found = a1 != rec.end() && a2 != rec.end() && a1->second.first == kMidAt && a2->second.first == kLimit;
		CHECK_MSG(found, "%s: %s has no cohort %s_1456 (L=1456) and %s_1712 (L=1712) records", cell, path.c_str(), tag.c_str(),
		          tag.c_str());
		std::vector<int32_t> t;
		if (found) {
			t = a1->second.second;
			t.insert(t.end(), a2->second.second.begin(), a2->second.second.end());
		}
		CHECK_MSG(!found || t.size() == kTokens, "%s: reference %s holds %zu tokens, not 513", cell, tag.c_str(), t.size());
		all = all && found && t.size() == kTokens;
		r.persona.push_back(t);
	}
	r.ok = all;
	return r;
}

// ---- the CPU twin ---------------------------------------------------------------------------------

struct CpuTwin {
	bool ok = false;
	std::vector<std::vector<int32_t>> persona;  // per persona: the 513 tokens of a budget-512 sequence adopting it
};
// The world and the ten personas in a CPU page pool (the CPU's own B), the world released, then one budget-512
// sequence per persona adopting it, each run to its refusal (SSLM_KV_BUDGET_EXCEEDED). The pool is sized with
// slack: the CPU twin grades tokens only, never admission.
inline CpuTwin RunCpuTwin(const std::vector<uint8_t>& artifact_bytes, const char* cell) {
	CpuTwin t;
	cpu::Model cm(artifact_bytes);
	if (!cm.ok) return t;
	const int64_t B = sslm_kv_page_positions(cm.model);
	CHECK_MSG(B > 0, "%s: CPU sslm_kv_page_positions = %lld", cell, static_cast<long long>(B));
	if (B <= 0) return t;
	const int64_t cap = 4096, cap_pages = (cap + B - 1) / B;
	auto R = [&](int64_t budget) { return std::min((budget + B - 1) / B + 1, cap_pages); };
	const uint32_t pages = static_cast<uint32_t>((kWorld + B - 1) / B + kPersonas * R(kPersona) + kPersonas * R(kBudget) + 8);
	cpu::Pool pool(cm.model, pages);
	if (!pool.pool) return t;
	sslm_prefix world = nullptr;
	std::vector<sslm_prefix> personas;
	sslm_status st = sslm_prefix_begin_budgeted(cm.model, &pool.pool, static_cast<int32_t>(kWorld), &world);
	if (st == SSLM_OK) st = cpu::PrefixPrefillAll(cm.model, world, WorldTokens());
	if (st == SSLM_OK) st = sslm_prefix_freeze(world);
	for (int p = 0; p < kPersonas && st == SSLM_OK; ++p) {
		sslm_prefix child = nullptr;
		st = sslm_prefix_begin_from(world, static_cast<int32_t>(kPersona), &child);
		if (st == SSLM_OK) personas.push_back(pool.Prefix(child));
		if (st == SSLM_OK) st = cpu::PrefixPrefillAll(cm.model, child, PersonaTokens(p));
		if (st == SSLM_OK) st = sslm_prefix_freeze(child);
	}
	if (st == SSLM_OK) {
		st = sslm_prefix_release(world);
		world = nullptr;
	}
	CHECK_MSG(st == SSLM_OK, "%s: the CPU twin's world and personas: status %d", cell, static_cast<int>(st));
	if (world) sslm_prefix_release(world);
	if (st != SSLM_OK) return t;
	bool all = true;
	for (int p = 0; p < kPersonas; ++p) {
		sslm_seq s = nullptr;
		st = sslm_seq_create_budgeted(cm.model, &pool.pool, kBudget, &s);
		if (st == SSLM_OK) st = sslm_seq_adopt_prefix(pool.Seq(s), personas[static_cast<size_t>(p)]);
		CHECK_MSG(st == SSLM_OK, "%s: CPU twin sequence for persona %d: status %d", cell, p, static_cast<int>(st));
		std::vector<int32_t> tokens;
		sslm_status end = SSLM_OK;
		for (int guard = 0; st == SSLM_OK && guard < 4 * kBudget; ++guard) {
			const int32_t tok = cpu::Token(cm.model, s, &end);
			if (end != SSLM_OK || tok < 0) break;
			tokens.push_back(tok);
		}
		CHECK_MSG(tokens.size() == kTokens && end == SSLM_KV_BUDGET_EXCEEDED,
		          "%s: CPU twin persona %d emitted %zu tokens, ending with status %d (want 513, then SSLM_KV_BUDGET_EXCEEDED)", cell,
		          p, tokens.size(), static_cast<int>(end));
		all = all && st == SSLM_OK && tokens.size() == kTokens && end == SSLM_KV_BUDGET_EXCEEDED;
		t.persona.push_back(tokens);
	}
	t.ok = all;
	return t;
}

// The CPU twin, checked against R0 (the CPU's own 10.1 verdict, restated so this run closes GPU == CPU == v1.11.0).
inline CpuTwin CpuTwinPinned(const Args& a, const std::string& stem, const std::vector<uint8_t>& artifact_bytes, const char* cell) {
	const Reference ref = ReadReference(a, stem, artifact_bytes, cell);
	CpuTwin cpu = RunCpuTwin(artifact_bytes, cell);
	for (int p = 0; p < kPersonas && ref.ok && cpu.ok; ++p)
		CHECK_MSG(cpu.persona[static_cast<size_t>(p)] == ref.persona[static_cast<size_t>(p)],
		          "%s: the CPU twin's persona %d tokens differ from the v1.11.0 no-sharing record (the CPU's 10.1 operand)", cell, p);
	std::printf("%s: CPU twin on %s: %d personas x %zu tokens%s\n", cell, stem.c_str(), cpu.ok ? kPersonas : 0, kTokens,
	            ref.ok ? ", checked against the R0 record" : " (R0 record unreadable: see FAIL above)");
	return cpu;
}

// ---- the GPU world: a budget world, ten personas by begin_from, the world released ----------------

struct GWorld {
	SslmGpuPrefixHandle* world = nullptr;
	std::vector<SslmGpuPrefixHandle*> personas;  // registered with the rig, which releases them
	bool ok = false;
};

// §5's order: each persona filled and frozen as it is built (the construction peaks at 194 at B = 16). With
// `all_open`, every begin_from runs before any persona is filled (10.4's own peak leg, 203 at B = 16), and the
// returned `first_refused` is the index of the first refused begin_from (-1 if none). The world is released at
// the end in both orders.
inline GWorld BuildWorld(Rig& rig, bool all_open = false, int* first_refused = nullptr, GS* refusal = nullptr) {
	GWorld w;
	if (first_refused) *first_refused = -1;
	if (refusal) *refusal = GS::SSLM_OK;
	const uint32_t db = rig.e.full_budget;
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, static_cast<int32_t>(kWorld), &w.world);
	if (st == GS::SSLM_OK) st = PrefixPrefillAll(rig.ctx, w.world, WorldTokens(), 0, db);
	if (st == GS::SSLM_OK) st = sslm_gpu_prefix_freeze(rig.ctx, w.world);
	CHECK_MSG(st == GS::SSLM_OK, "the GPU world: %s", StatusName(st));
	if (!Real(w.world)) w.world = nullptr;
	if (st != GS::SSLM_OK) {
		if (w.world) sslm_gpu_prefix_release(rig.ctx, w.world);
		w.world = nullptr;
		return w;
	}
	for (int i = 0; i < kPersonas; ++i) {
		SslmGpuPrefixHandle* p = nullptr;
		st = sslm_gpu_prefix_begin_from(rig.ctx, w.world, static_cast<int32_t>(kPersona), &p);
		if (st != GS::SSLM_OK) {
			if (first_refused) *first_refused = i;
			if (refusal) *refusal = st;
			if (!all_open) CHECK_MSG(false, "begin_from for persona %d: %s", i, StatusName(st));
			break;
		}
		w.personas.push_back(rig.Prefix(p));
		if (!all_open) {
			st = PrefixPrefillAll(rig.ctx, p, PersonaTokens(i), 0, db);
			if (st == GS::SSLM_OK) st = sslm_gpu_prefix_freeze(rig.ctx, p);
			CHECK_MSG(st == GS::SSLM_OK, "GPU persona %d: %s", i, StatusName(st));
			if (st != GS::SSLM_OK) break;
		}
	}
	if (all_open && st == GS::SSLM_OK) {
		for (int i = 0; i < kPersonas && st == GS::SSLM_OK; ++i) {
			st = PrefixPrefillAll(rig.ctx, w.personas[static_cast<size_t>(i)], PersonaTokens(i), 0, db);
			if (st == GS::SSLM_OK) st = sslm_gpu_prefix_freeze(rig.ctx, w.personas[static_cast<size_t>(i)]);
			CHECK_MSG(st == GS::SSLM_OK, "GPU persona %d (all open): %s", i, StatusName(st));
		}
	}
	const GS rel = sslm_gpu_prefix_release(rig.ctx, w.world);
	CHECK_MSG(rel == GS::SSLM_OK, "releasing the GPU world: %s", StatusName(rel));
	w.world = nullptr;
	w.ok = st == GS::SSLM_OK && rel == GS::SSLM_OK && static_cast<int>(w.personas.size()) == kPersonas;
	return w;
}

// ---- the GPU cohort run -----------------------------------------------------------------------------

struct CohortRun {
	bool built = false;
	int first_refused = -1;  // the index of the first refused create, -1 if none
	GS refusal = GS::SSLM_OK;
	int creates_refused = 0;
	int adopts_refused = 0;
	std::vector<std::vector<int32_t>> tokens;  // per admitted sequence: everything it emitted
	std::vector<GS> end;                       // the status that ended it
	std::vector<std::vector<uint8_t>> mid, lim;  // 10.7's saves at 1,456 and at the limit
	std::vector<GSeq> mid_state, lim_state;      // the test-side token state at each save (RestoreIntoPool's `like`)
};

// The cohort in a pool of `pages`, in §5's order. Each admitted sequence adopts its persona; with `drive`, it
// then emits 257 tokens (to 1,456: saved when `keep_blobs`) and runs to its refusal (saved at the limit), before
// the next sequence is created (the plan's order: "50 budget-512 sequences, each running to its budget").
inline CohortRun RunCohort(const Env& e, uint32_t pages, bool drive, bool keep_blobs, const char* cell) {
	CohortRun r;
	Rig rig(e, pages);
	if (!rig.pool) return r;
	const GWorld w = BuildWorld(rig);
	r.built = w.ok;
	for (int i = 0; i < kSeqs && w.ok; ++i) {
		SslmGpuSequenceHandle* h = nullptr;
		const GS st = sslm_gpu_seq_create_budgeted(rig.ctx, rig.model, rig.pool, kBudget, &h);
		if (st != GS::SSLM_OK) {
			if (r.first_refused < 0) {
				r.first_refused = i;
				r.refusal = st;
			}
			++r.creates_refused;
			continue;
		}
		GSeq s;
		s.h = rig.Seq(h);
		const GS ad = Adopt(rig, s, w.personas[static_cast<size_t>(PersonaOf(i))], kOrigin);
		CHECK_MSG(ad == GS::SSLM_OK, "%s: adopt of sequence %d: %s", cell, i, StatusName(ad));
		if (ad != GS::SSLM_OK) ++r.adopts_refused;
		if (!drive || ad != GS::SSLM_OK) continue;
		GS a = GS::SSLM_OK, b = GS::SSLM_OK;
		std::vector<int32_t> t = DecodeN(e, s, kMidAt - kOrigin + 1, &a);
		CHECK_MSG(a == GS::SSLM_OK && t.size() == static_cast<size_t>(kMidAt - kOrigin + 1),
		          "%s: sequence %d: %zu tokens before 1,456 (status %s)", cell, i, t.size(), StatusName(a));
		if (keep_blobs) {
			r.mid.push_back(Save(e, s.h));
			r.mid_state.push_back(s);
		}
		const std::vector<int32_t> rest = RunToRefusal(e, s, &b);
		t.insert(t.end(), rest.begin(), rest.end());
		if (keep_blobs) {
			r.lim.push_back(Save(e, s.h));
			r.lim_state.push_back(s);
		}
		r.tokens.push_back(t);
		r.end.push_back(b);
	}
	return r;  // the rig releases the sequences, then the personas, then the pool
}

// The cohort's admission verdict at a pool of `pages`: `expect_refused` < 0 means every create and adopt is
// admitted; otherwise exactly one create is refused, the one at that index, with SSLM_GPU_KV_POOL_EXHAUSTED.
inline bool AdmissionAsExpected(const CohortRun& r, int expect_refused) {
	if (!r.built || r.adopts_refused) return false;
	if (expect_refused < 0) return r.first_refused < 0 && r.creates_refused == 0;
	return r.first_refused == expect_refused && r.creates_refused == 1 && r.refusal == GS::SSLM_GPU_KV_POOL_EXHAUSTED;
}

// ---- restore of the cohort's blobs into a fresh pool (10.6 / 10.8's GPU twins) -----------------------

struct RestoreRun {
	int first_refused = -1;
	int refused = 0;
	GS refusal = GS::SSLM_OK;
	uint64_t shared_diag = 0;
};

// A fresh pool of `pages` holding the rebuilt world and personas (the world released), then the 50 blobs
// restored in order, with their persona handles or without. `after(rig, world, restored)` runs before teardown;
// a refused restore leaves an empty GSeq in its slot.
template <class After>
inline RestoreRun RestoreCohort(const Env& e, uint32_t pages, const std::vector<std::vector<uint8_t>>& blobs,
                                const std::vector<GSeq>& like, bool handles, After after) {
	RestoreRun r;
	Rig rig(e, pages);
	if (!rig.pool) return r;
	const GWorld w = BuildWorld(rig);
	if (!w.ok) {
		// Nothing restored: reported as every restore refused, so no admission check can read it as "all admitted".
		CHECK_MSG(false, "the restore pool's world and personas did not rebuild in %u pages", pages);
		r.first_refused = 0;
		r.refused = static_cast<int>(blobs.size());
		return r;
	}
	std::vector<GSeq> seqs;
	for (size_t i = 0; i < blobs.size(); ++i) {
		uint32_t shared = 0;
		SslmGpuPrefixHandle* h = handles ? w.personas[static_cast<size_t>(PersonaOf(static_cast<int>(i)))] : nullptr;
		Restored got = RestoreIntoPool(rig, blobs[i], h, i < like.size() ? like[i] : GSeq{}, &shared);
		r.shared_diag += shared;
		if (got.status != GS::SSLM_OK || !got.seq.h) {
			if (r.first_refused < 0) {
				r.first_refused = static_cast<int>(i);
				r.refusal = got.status;
			}
			++r.refused;
			seqs.push_back(GSeq{});
			continue;
		}
		seqs.push_back(got.seq);
	}
	after(rig, w, seqs);
	return r;
}

inline std::string Ids(const std::vector<int32_t>& t, size_t max = 12) {
	std::string s;
	for (size_t i = 0; i < t.size() && i < max; ++i) s += (i ? "," : "") + std::to_string(t[i]);
	if (t.size() > max) s += ",...";
	return s;
}

}  // namespace cohort
}  // namespace g1e

#endif  // SSLM_PKV_GPU_G1E_COHORT_H
