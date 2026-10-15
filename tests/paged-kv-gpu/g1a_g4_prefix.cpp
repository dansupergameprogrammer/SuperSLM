// Paged-KV plan (rev 16.2) step G1, group g1a: the G4 cells and cell parts (GPU prefixes, adopt and
// begin_from).
//
//   1.8 [G4 part]     GPU twin of 1.1, adopt included: tokens equal a fresh handle and CPU per stage.
//   1.9 [G4]          GPU twin of 1.3: 1,000 mixed cycles, then the GPU fill probe reads the whole pool.
//   6.5b [G4 part]    GPU paged tokens and K/V byte-equal CPU paged, the adopt cases.
//   8.15 [G4]         GPU adopt x an adapter bound after adopt.
//   8.16 [G4]         GPU adopt x schema content in the prefix.
//
// RED NOW (LINK): each cell calls the G4 verbs (sslm_gpu_prefix_begin, _prefill, _freeze, _release,
// sslm_gpu_seq_adopt_prefix; 6.5b and 8.16 also sslm_gpu_prefix_begin_from; 8.16
// sslm_gpu_prefix_set_schema), declared and undefined until G4, besides the G3 verbs and seams.
// GREEN AT G4: each cell's checks pass on both GPUs with the debug layer on and clean (G4's gate:
// a clean debug layer for every cell that runs an adopt or begin_from).

#include "g1a_helpers.h"

#include "sslm_model_hostile_fixtures.h"  // superslm_test's artifact builders (the 8.15 adapter)
#include "superslm/artifact.h"

namespace {

using namespace g1a;

// ---- 1.8 [G4] -------------------------------------------------------------------------------------
// One pooled GPU handle driven through R0's lifecycle with a GPU prefix (fresh -> prefill 100 ->
// decode -> reset -> prefill 40 -> decode -> reset -> adopt 300 -> decode -> prefill 37 -> decode,
// crossing pages at every stage). At each stage its tokens equal a second GPU run in which every
// reset is a release and a fresh create (a fresh handle driven the same way), equal the CPU budget
// holder's same lifecycle, and equal the R0 reference's tokens. kills: a pooled reset that leaves the
// table or the reserve as it found them (the next prefill reads stale rows or draws pages it should
// not); an adopt onto a used holder that keeps a page of its old table.
void Cell18G4() {
	const Fx& fx = PkvFixture("pkv_def");
	if (!fx.ok) return;
	GpuPrefixBackend g_used(fx, 64), g_fresh(fx, 64);
	CpuPagedBackend cpu(fx, 64);
	SRun u(g_used, false), f(g_fresh, false), c(cpu, false);
	Lifecycle(u, 128, 300, /*fresh=*/false);
	Lifecycle(f, 128, 300, /*fresh=*/true);
	Lifecycle(c, 128, 300, false);
	G1A_CHECK_MSG(u.fail.empty() && f.fail.empty() && c.fail.empty(), "1.8: %s %s %s", u.fail.c_str(), f.fail.c_str(),
	              c.fail.c_str());
	ExpectSameRecords(u.records, f.records, Cmp::kTokens, "1.8 vs a fresh handle at each stage");
	ExpectSameRecords(u.records, c.records, Cmp::kTokens, "1.8 vs CPU");
	ExpectReference("v1.11.0", fx, "lifecycle", u.records, Cmp::kTokens, "1.8 vs R0");
}

// ---- 1.9 [G4] -------------------------------------------------------------------------------------
// 1,000 mixed cycles on one 400-page pool -- budgeted creates (budgets 1, 15, 16, 17, 64, 200),
// adopts of three GPU prefixes (37, 300 and 1,000 tokens, every one mid-page so every adopt copies a
// tail) onto used and fresh holders with a token or two decoded after, prefills into fresh holders,
// resets, releases, and prefixes released and rebuilt while their adopters live -- then a shuffled
// teardown. The GPU fill probe (two-sided, after sslm_gpu_ready drained every sequence) admits exactly
// 400 pages: no page leaked, none freed twice. The GPU pool stats are printed as a diagnostic only.
struct Mix19 {
	int creates = 0, adopts = 0, prefills = 0, resets = 0, releases = 0, prefix_rebuilds = 0;
};
constexpr uint32_t k19Pool = 400;

bool BenignStatus(SslmGpuStatus st) {
	return st == GS::SSLM_OK || st == GS::SSLM_GPU_KV_BUDGET_EXCEEDED || st == GS::SSLM_GPU_KV_POOL_EXHAUSTED;
}

SslmGpuPrefixHandle* FrozenPrefix(const Fx& fx, SslmGpuKvPoolHandle* pool, uint32_t seed, int32_t len) {
	SslmGpuPrefixHandle* p = nullptr;
	if (sslm_gpu_prefix_begin(Ctx(), fx.gpu, pool, len, &p) != GS::SSLM_OK || !p) return nullptr;
	const std::vector<int32_t> t = pkv::Stream(seed, len, fx.vocab);
	int32_t consumed = 0;
	const bool ok = GpuPrefixPrefillAll(fx, p, t.data(), len, 0, &consumed) == GS::SSLM_OK &&
	                consumed == len && sslm_gpu_prefix_freeze(Ctx(), p) == GS::SSLM_OK;
	G1A_CHECK_MSG(ok, "1.9: a GPU prefix of %d tokens did not build", len);
	if (!ok) {
		sslm_gpu_prefix_release(Ctx(), p);
		return nullptr;
	}
	return p;
}

std::unique_ptr<GScene> Build19(Mix19* counts) {
	auto sc = std::make_unique<GScene>();
	const Fx& fx = PkvFixture("pkv_def");
	if (!fx.ok) return sc;
	if (sslm_gpu_kv_pool_create(Ctx(), fx.gpu, k19Pool, &sc->pool) != GS::SSLM_OK) {
		G1A_CHECK_MSG(false, "1.9: pool create");
		return sc;
	}
	Mix19 n;
	Lcg g(19);
	const int32_t lens[3] = {37, 300, 1000};
	SslmGpuPrefixHandle* px[3] = {};
	for (int k = 0; k < 3; ++k) px[k] = FrozenPrefix(fx, sc->pool, 190 + static_cast<uint32_t>(k), lens[k]);
	struct Live {
		SslmGpuSequenceHandle* s;
		int32_t last;
		bool fresh;
	};
	std::vector<Live> live;
	auto pick = [&]() -> Live& { return live[g.Next() % live.size()]; };
	for (int c = 0; c < 1000; ++c) {
		switch (g.Next() % 8) {
			case 0:
			case 1: {
				if (live.size() >= 12) break;
				static const int32_t kBudgets[] = {1, 15, 16, 17, 64, 200};
				SslmGpuSequenceHandle* s = nullptr;
				const SslmGpuStatus st = sslm_gpu_seq_create_budgeted(Ctx(), fx.gpu, sc->pool, kBudgets[g.Next() % 6], &s);
				G1A_CHECK_MSG(BenignStatus(st), "1.9 cycle %d: create -> %s", c, GpuStatusName(st));
				if (st == GS::SSLM_OK && s) {
					live.push_back({s, -1, true});
					++n.creates;
				}
				break;
			}
			case 2:
			case 3: {
				if (live.empty()) break;
				Live& l = pick();
				const int k = static_cast<int>(g.Next() % 3);
				if (!px[k]) break;
				const SslmGpuStatus st = sslm_gpu_seq_adopt_prefix(Ctx(), l.s, px[k]);
				G1A_CHECK_MSG(st == GS::SSLM_OK, "1.9 cycle %d: adopt -> %s", c, GpuStatusName(st));
				if (st != GS::SSLM_OK) break;
				++n.adopts;
				l.fresh = false;
				l.last = -1;
				const int toks = 1 + static_cast<int>(g.Next() % 2);
				for (int i = 0; i < toks; ++i) {
					int32_t tok = -1;
					const SslmGpuStatus ds = GpuNextToken(fx, l.s, l.last, GpuDecodeMode::kBridge, fx.full_depth, &tok);
					G1A_CHECK_MSG(BenignStatus(ds), "1.9 cycle %d: decode -> %s", c, GpuStatusName(ds));
					if (ds != GS::SSLM_OK) break;  // a refusal at a tiny budget is fine
					l.last = tok;
				}
				break;
			}
			case 4: {
				if (live.empty()) break;
				Live& l = pick();
				G1A_CHECK_GPU(sslm_gpu_seq_reset(Ctx(), l.s), GS::SSLM_OK);
				l.fresh = true;
				l.last = -1;
				++n.resets;
				break;
			}
			case 5: {
				if (live.empty()) break;
				const size_t i = g.Next() % live.size();
				G1A_CHECK_GPU(sslm_gpu_seq_release(Ctx(), live[i].s), GS::SSLM_OK);
				live.erase(live.begin() + static_cast<std::ptrdiff_t>(i));
				++n.releases;
				break;
			}
			case 6: {  // release a prefix while its adopters live, and rebuild it
				const int k = static_cast<int>(g.Next() % 3);
				if (px[k]) G1A_CHECK_GPU(sslm_gpu_prefix_release(Ctx(), px[k]), GS::SSLM_OK);
				px[k] = FrozenPrefix(fx, sc->pool, 290 + static_cast<uint32_t>(c), lens[k]);
				++n.prefix_rebuilds;
				break;
			}
			default: {  // a prefill into a fresh or reset holder, across a page
				if (live.empty()) break;
				Live& l = pick();
				if (!l.fresh) break;
				const std::vector<int32_t> t = pkv::Stream(390 + static_cast<uint32_t>(c), 18, fx.vocab);
				const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(Ctx(), l.s, t.data(), 18, fx.one_layer);
				G1A_CHECK_MSG(BenignStatus(st), "1.9 cycle %d: prefill -> %s", c, GpuStatusName(st));
				l.fresh = false;
				l.last = t.back();
				++n.prefills;
				break;
			}
		}
	}
	// A shuffled full teardown: holders and prefixes interleaved.
	std::vector<std::pair<int, size_t>> order;
	for (size_t i = 0; i < live.size(); ++i) order.push_back({0, i});
	for (size_t k = 0; k < 3; ++k) order.push_back({1, k});
	for (size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[g.Next() % i]);
	for (const auto& o : order) {
		if (o.first == 0) G1A_CHECK_GPU(sslm_gpu_seq_release(Ctx(), live[o.second].s), GS::SSLM_OK);
		else if (px[o.second]) G1A_CHECK_GPU(sslm_gpu_prefix_release(Ctx(), px[o.second]), GS::SSLM_OK);
	}
	if (counts) *counts = n;
	return sc;
}

void Cell19() {
	const Fx& fx = PkvFixture("pkv_def");
	if (!fx.ok) return;
	Mix19 n;
	{
		std::unique_ptr<GScene> sc = Build19(&n);
		if (sc->pool) PrintPoolStatsDiagnostic(sc->pool, "1.9 after teardown");
	}
	std::printf("  1.9: creates=%d adopts=%d prefills=%d resets=%d releases=%d prefix_rebuilds=%d\n", n.creates, n.adopts,
	            n.prefills, n.resets, n.releases, n.prefix_rebuilds);
	G1A_CHECK_MSG(n.creates > 50 && n.adopts > 50 && n.resets > 20 && n.releases > 20 && n.prefix_rebuilds > 20 && n.prefills > 10,
	              "1.9: construction: the mix did not exercise every operation");
	// kills: a pooled reset, release or adopt that leaks a page (399) or frees one twice (401); a
	// prefix release that frees pages its adopters still map.
	GpuFillProbeExact(fx, [] { return Build19(nullptr); }, k19Pool, "1.9 after 1,000 mixed cycles");
}

// ---- 6.5b [G4: the adopt cases] -------------------------------------------------------------------
// R0's lifecycle (adopt of a 300-token prefix onto a used holder) and prefix_lengths (adopt of
// prefixes of 0, 1, 15, 16, 17, 1,000 mid-page and 1,008 aligned) on GPU prefixes and budget adopters:
// tokens, context length and rows [0, L) through the GPU mirror peek equal the CPU paged twin's and
// the R0 reference. Then a begin_from child (a 300-token world, a child continuing it by 37 tokens,
// so the parent's partial page is copied) adopted and decoded: tokens and rows equal the CPU twin's.
// On this GPU; run on both. kills: an adopt that maps the prefix's partial page shared instead of
// copying it; a tail copy of the wrong bytes or to the wrong page; a begin_from that shares the
// parent's partial page.
void BeginFromRun(SRun& d) {
	const int32_t V = d.b.fx.vocab;
	void* world = nullptr;
	void* child = nullptr;
	void* s = nullptr;
	(void)(d.PrefixBegin(300, &world) && d.PrefixPrefill(world, pkv::Stream(61, 300, V)) && d.Freeze(world) &&
	       d.PrefixBeginFrom(world, 64, &child) && d.PrefixPrefill(child, pkv::Stream(62, 37, V)) && d.Freeze(child) &&
	       d.Create(64, &s) && d.Adopt(s, child) && d.Mark("child_adopted", s) && d.Decode(s, 8) && d.Mark("child_decode8", s));
	if (s) d.Release(s);
	if (child) d.PrefixRelease(child);
	if (world) d.PrefixRelease(world);
}

void Cell65bG4() {
	const std::string adapter = AdapterTag();
	for (const char* stem : {"pkv_def", "pkv_qk"}) {
		const Fx& fx = PkvFixture(stem);
		if (!fx.ok) continue;
		GpuPrefixBackend gpu(fx, 96);
		CpuPagedBackend cpu(fx, 96);
		for (int scen = 0; scen < 3; ++scen) {
			const char* name = scen == 0 ? "lifecycle" : scen == 1 ? "prefix_lengths" : "begin_from";
			SRun g(gpu, true), c(cpu, true);
			if (scen == 0) {
				Lifecycle(g);
				Lifecycle(c);
			} else if (scen == 1) {
				PrefixLengths(g);
				PrefixLengths(c);
			} else {
				BeginFromRun(g);
				BeginFromRun(c);
			}
			G1A_CHECK_MSG(g.fail.empty() && c.fail.empty(), "6.5b %s %s: %s%s", stem, name, g.fail.c_str(), c.fail.c_str());
			const std::string what = std::string("6.5b ") + stem + " " + name + " " + adapter;
			ExpectSameRecords(g.records, c.records, Cmp::kTokensAndRows, what + " vs CPU paged");
			if (scen < 2) ExpectReference("v1.11.0", fx, name, g.records, Cmp::kTokensAndRows, what + " vs R0");
		}
	}
}

// ---- 8.15 [G4] GPU adopt x an adapter bound after adopt -----------------------------------------------
// A rank-1 LoRA on layer 0's q_proj and layer 1's v_proj (identity fold triples), built here against
// pkv_def's own Config and SigmoidLut and its integrity hash -- pkv_budget_b_helpers.h's adapter, so
// the CPU twin is the CPU suite's own 8.4 construction. A budget adopter of a frozen 1,000-token GPU
// prefix binds the adapter after the adopt, decodes 8, prefills 20 and decodes 4. Its tokens and rows
// equal (a) the CPU budget adopter with the same adapter set after its share adopt, and (b) a GPU
// pooled sequence that prefilled the same 1,000 tokens privately and bound the adapter after (no
// sharing). The adapter moves layer 1's V rows, so the cell first checks it changes the tokens.
// kills: an adapter's delta applied to (or skipped on) the shared prefix span; a shared page written
// under the adapter; a bind refused or ignored after a GPU adopt.
std::vector<uint8_t> BuildAdapterBytes(const Fx& fx) {
	using namespace superslm_test;
	superslm::SslmArtifact art;
	superslm::SslmError err;
	if (superslm::SslmArtifact::OpenFromMemory(fx.bytes.p, fx.size, art, &err) != superslm::SslmStatus::Ok) {
		G1A_CHECK_MSG(false, "8.15: the fixture does not open as an artifact");
		return {};
	}
	const superslm::SslmSectionView* cfg = art.Section(superslm::SslmSectionType::Config);
	const superslm::SslmSectionView* sil = art.Section(superslm::SslmSectionType::SigmoidLut);
	if (!cfg || !sil) {
		G1A_CHECK_MSG(false, "8.15: the fixture has no Config or SigmoidLut section");
		return {};
	}
	std::vector<uint8_t> adp = {'A', 'D', 'P', '1'};
	WriteU32LE(adp, 1);             // version
	WriteU32LE(adp, 1);             // rank
	WriteU32LE(adp, 0x1u | 0x40u);  // q_proj, v_proj
	const auto hash = art.RawIntegrityHash();
	adp.insert(adp.end(), hash.begin(), hash.end());
	const double alpha = 8.0;
	uint64_t bits = 0;
	std::memcpy(&bits, &alpha, 8);
	WriteU64LE(adp, bits);
	WriteU32LE(adp, 0);  // use_rslora
	WriteU32LE(adp, 0);  // reserved
	const std::string name = "pkv-budget-b";
	WriteU32LE(adp, static_cast<uint32_t>(name.size()));
	WriteBytes(adp, name);
	struct Proj {
		std::string name;
		uint32_t in, out;
	};
	const uint32_t hidden = static_cast<uint32_t>(fx.hidden);
	const std::vector<Proj> projs = {{"layer0.q_proj", hidden, fx.view.config.num_attention_heads * fx.geo.head_dim},
	                                 {"layer1.v_proj", hidden, fx.geo.kv_heads * fx.geo.head_dim}};
	std::vector<ManifestTensorSpec> wt, dt, ut;
	for (const Proj& p : projs) {
		wt.push_back({p.name + ".lora_A", {1, p.in}});
		wt.push_back({p.name + ".lora_B", {p.out, 1}});
		dt.push_back({p.name, {p.out * 3}});
		ut.push_back({p.name, {3}});
	}
	BuiltManifest w = BuildManifest(reinterpret_cast<const uint8_t*>("WGT1"), 1, wt);
	BuiltManifest d = BuildManifest(superslm::kDeltaFoldScalesMagic, 4, dt);
	BuiltManifest u = BuildManifest(superslm::kUFoldScalesMagic, 4, ut);
	for (BuiltManifest* m : {&d, &u})
		for (size_t t = 0; t < m->tensor_data_off.size(); ++t)
			for (uint64_t r = 0; r < m->tensor_elem_count[t] / 3; ++r) {
				const int32_t triple[3] = {1, 0, 0};
				std::memcpy(m->bytes.data() + m->tensor_data_off[t] + r * 12, triple, 12);
			}
	auto copy = [](superslm::SslmSectionType t, const superslm::SslmSectionView* v) {
		return MakeSection(t, v->dtype, std::vector<uint8_t>(v->data, v->data + v->byte_size));
	};
	BuiltArtifact a = BuildArtifact({copy(superslm::SslmSectionType::Config, cfg), copy(superslm::SslmSectionType::SigmoidLut, sil),
	                                 MakeSection(superslm::SslmSectionType::Provenance, superslm::SslmDtype::Raw, adp),
	                                 MakeSection(superslm::SslmSectionType::Weights, superslm::SslmDtype::Int8, w.bytes),
	                                 MakeSection(superslm::SslmSectionType::DeltaFoldScales, superslm::SslmDtype::Int32, d.bytes),
	                                 MakeSection(superslm::SslmSectionType::UFoldScales, superslm::SslmDtype::Int32, u.bytes)});
	return a.bytes;
}

struct Adapters {
	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	SslmGpuAdapterHandle* gpu = nullptr;
	sslm_adapter cpu = nullptr;
	explicit Adapters(const Fx& fx) : bytes(BuildAdapterBytes(fx)) {
		if (bytes.empty()) return;
		std::string err;
		G1A_CHECK_MSG(superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) == superslm::SslmModelStatus::Ok,
		              "8.15: the adapter does not load: %s", err.c_str());
		G1A_CHECK_GPU(sslm_gpu_adapter_map(Ctx(), fx.gpu, &view, &gpu), GS::SSLM_OK);
		G1A_CHECK_EQ(sslm_adapter_map(bytes.data(), bytes.size(), fx.cpu, &cpu), SSLM_OK);
	}
	~Adapters() {
		if (gpu) sslm_gpu_adapter_unmap(Ctx(), gpu);
		if (cpu) sslm_adapter_release(cpu);
	}
};

// One run: `adopt` shares the frozen world, else the world is prefilled privately; `bind` sets the
// adapter after that. Returns the record of the continuation (tokens and rows [0, L)).
template <typename BindFn>
Rec AdapterRun(Backend& be, bool adopt, BindFn bind) {
	SRun d(be, true);
	const int32_t V = be.fx.vocab;
	const std::vector<int32_t> world = pkv::Stream(151, 1000, V);
	void* px = nullptr;
	void* s = nullptr;
	bool ok;
	if (adopt) {
		ok = d.PrefixBegin(1000, &px) && d.PrefixPrefill(px, world) && d.Freeze(px) && d.Create(64, &s) && d.Adopt(s, px);
	} else {
		ok = d.Create(1064, &s) && d.Prefill(s, world);
	}
	ok = ok && d.Ok(bind(s), "bind adapter") && d.Decode(s, 8) && d.Prefill(s, pkv::Stream(152, 20, V)) && d.Decode(s, 4) &&
	     d.Mark("continuation", s);
	G1A_CHECK_MSG(ok, "8.15 %s (%s): %s", be.Name(), adopt ? "adopt" : "private", d.fail.c_str());
	if (s) d.Release(s);
	if (px) d.PrefixRelease(px);
	return d.records.empty() ? Rec{} : d.records.back();
}

void Cell815() {
	const Fx& fx = PkvFixture("pkv_def");
	if (!fx.ok) return;
	Adapters ad(fx);
	if (!ad.gpu || !ad.cpu) return;
	GpuPrefixBackend gpu(fx, 160);
	CpuPagedBackend cpu(fx, 160);
	auto gpu_bind = [&](void* s) {
		return AsInt(sslm_gpu_seq_bind_adapter(Ctx(), static_cast<SslmGpuSequenceHandle*>(s), ad.gpu));
	};
	auto no_bind = [](void*) { return 0; };
	auto cpu_bind = [&](void* s) { return static_cast<int>(sslm_seq_set_adapter(static_cast<sslm_seq>(s), ad.cpu)); };
	const Rec got = AdapterRun(gpu, true, gpu_bind);
	const Rec base = AdapterRun(gpu, true, no_bind);
	const Rec cpu_twin = AdapterRun(cpu, true, cpu_bind);
	const Rec unshared = AdapterRun(gpu, false, gpu_bind);
	G1A_CHECK_MSG(got.tokens != base.tokens, "8.15: construction: the adapter does not change the tokens, so the pair is vacuous");
	G1A_CHECK_MSG(got.tokens == cpu_twin.tokens && got.L == cpu_twin.L && got.rows_sha == cpu_twin.rows_sha,
	              "8.15: the GPU adopter with the adapter differs from the CPU twin");
	G1A_CHECK_MSG(got.tokens == unshared.tokens && got.L == unshared.L && got.rows_sha == unshared.rows_sha,
	              "8.15: the GPU adopter with the adapter differs from the unshared GPU twin");
}

// ---- 8.16 [G4] GPU adopt x schema content in the prefix ------------------------------------------------
// A GPU budget prefix bound to the fixture's schema with real progress: n0 prompt tokens then one
// schema-admitted token as SCHEMA_CONTENT (kind 1), so the forced row sits at position n0 -- the last
// row of a page for n0 = 15, the first of the next for n0 = 16. A begin_from child continues it by 20
// prompt tokens across the next page; a schema-bound budget sequence adopts the child (inheriting the
// walk) and decodes 6 tokens under the schema. Its tokens, schema_accepting and rows [0, L) equal the
// CPU twin's (the same calls on the CPU's budget prefix, child and adopter). kills: a GPU adopt or
// begin_from that drops the walk state or the binding; a forced row written to the wrong page at the
// boundary.
//
// CONSTRUCTION (G4 cells fix): pkv_def's schema accepts exactly one document, spelled by one token, so the
// admitted token completes it and no continuation exists under the schema after it -- on both twins the
// adopter's decode is the defined dead end (`-2`, SSLM_OK, retry-stable: the walk and the ready token are
// unchanged). The decode is therefore graded the CPU twin's way (c5_dim8_composition.cpp 8.3, CensusGreedy):
// one outcome per step, the token or the dead end's -2, six steps on each twin, byte-equal. That outcome is
// the walk's own observable: an adopt or begin_from that drops the walk restarts it and emits the admitted
// token, and one that drops the binding decodes unmasked -- either gives real tokens where the twin
// dead-ends. The rows graded are [0, L) with L the chain's n0 + 1 + 20 rows plus one per emitted token a
// later step embedded (Rows816).
constexpr int k816Steps = 6;

// The rows a decode of `toks` (one outcome per step) wrote past the adopted length: a real token's row is
// written by the next step's embed, so every real token but a final one counts.
int64_t Rows816(const std::vector<int32_t>& toks) {
	int64_t n = 0;
	for (size_t i = 0; i + 1 < toks.size(); ++i) n += toks[i] >= 0 ? 1 : 0;
	return n;
}

int32_t AdmittedToken(const Fx& fx, sslm_schema schema) {
	CpuPagedBackend legacy(fx, static_cast<uint32_t>(fx.CapPages()), true);
	sslm_seq d = nullptr;
	if (sslm_seq_create(fx.cpu, &legacy.pool, &d) != SSLM_OK || !d) return -1;
	int32_t tok = -1;
	if (sslm_seq_set_schema(d, schema) == SSLM_OK && legacy.Prefill(d, pkv::Stream(72, 20, fx.vocab)) == SSLM_OK) {
		std::vector<int32_t> t;
		if (legacy.Decode(d, 1, &t) == SSLM_OK && !t.empty()) tok = t[0];
	}
	sslm_seq_release(d);
	G1A_CHECK_MSG(tok >= 0, "8.16: no schema-admitted token from the start state");
	return tok;
}

struct SchemaOut {
	std::vector<int32_t> tokens;
	int32_t accepting = -1;
	int64_t L = -1;
	std::vector<uint8_t> rows;
};

SchemaOut SchemaRunGpu(const Fx& fx, int32_t schema, int32_t n0, int32_t admitted) {
	SchemaOut o;
	GpuPrefixBackend gpu(fx, 32);
	if (!gpu.pool) return o;
	const int32_t V = fx.vocab;
	SslmGpuPrefixHandle* root = nullptr;
	SslmGpuPrefixHandle* child = nullptr;
	SslmGpuSequenceHandle* s = nullptr;
	G1A_CHECK_GPU(sslm_gpu_prefix_begin(Ctx(), fx.gpu, gpu.pool, 32, &root), GS::SSLM_OK);
	if (!root) return o;
	G1A_CHECK_GPU(sslm_gpu_prefix_set_schema(Ctx(), root, schema), GS::SSLM_OK);
	const std::vector<int32_t> p0 = pkv::Stream(84, n0, V), p1 = pkv::Stream(85, 20, V);
	int32_t c = 0;
	G1A_CHECK_GPU(GpuPrefixPrefillAll(fx, root, p0.data(), n0, 0, &c), GS::SSLM_OK);
	G1A_CHECK_GPU(sslm_gpu_prefix_prefill(Ctx(), root, &admitted, 1, /*kind=*/1, fx.full_depth, &c), GS::SSLM_OK);
	G1A_CHECK_EQ(c, 1);
	G1A_CHECK_GPU(sslm_gpu_prefix_freeze(Ctx(), root), GS::SSLM_OK);
	G1A_CHECK_GPU(sslm_gpu_prefix_begin_from(Ctx(), root, 32, &child), GS::SSLM_OK);
	if (child) {
		G1A_CHECK_GPU(GpuPrefixPrefillAll(fx, child, p1.data(), 20, 0, &c), GS::SSLM_OK);
		G1A_CHECK_GPU(sslm_gpu_prefix_freeze(Ctx(), child), GS::SSLM_OK);
		G1A_CHECK_GPU(sslm_gpu_seq_create_budgeted(Ctx(), fx.gpu, gpu.pool, 32, &s), GS::SSLM_OK);
	}
	if (s) {
		G1A_CHECK_GPU(SslmGpuSeqSetSchemaForG5Bridge(Ctx(), s, schema), GS::SSLM_OK);
		G1A_CHECK_GPU(sslm_gpu_seq_adopt_prefix(Ctx(), s, child), GS::SSLM_OK);
		G1A_CHECK_GPU(SslmGpuSeqSchemaAcceptingForG5Bridge(Ctx(), s, &o.accepting), GS::SSLM_OK);
		int32_t last = -1;
		for (int i = 0; i < k816Steps; ++i) {
			int32_t tok = -1;
			const SslmGpuStatus st = GpuNextToken(fx, s, last, GpuDecodeMode::kBridge, fx.full_depth, &tok);
			G1A_CHECK_GPU(st, GS::SSLM_OK);
			if (st != GS::SSLM_OK) break;
			o.tokens.push_back(tok);
			if (tok >= 0) last = tok;
		}
		o.L = n0 + 1 + 20 + Rows816(o.tokens);  // the forced row, the child's 20, the decoded rows
		G1A_CHECK_MSG(gpu.Rows(s, o.L, &o.rows), "8.16: rows unreadable");
		gpu.Release(s);
	}
	if (child) sslm_gpu_prefix_release(Ctx(), child);
	sslm_gpu_prefix_release(Ctx(), root);
	return o;
}

SchemaOut SchemaRunCpu(const Fx& fx, sslm_schema schema, int32_t n0, int32_t admitted) {
	SchemaOut o;
	CpuPagedBackend cpu(fx, 32);
	const int32_t V = fx.vocab;
	sslm_prefix root = nullptr, child = nullptr;
	sslm_seq s = nullptr;
	G1A_CHECK_EQ(sslm_prefix_begin_budgeted(fx.cpu, &cpu.pool, 32, &root), SSLM_OK);
	if (!root) return o;
	G1A_CHECK_EQ(sslm_prefix_set_schema(root, schema), SSLM_OK);
	G1A_CHECK_EQ(cpu.PrefixPrefill(root, pkv::Stream(84, n0, V)), SSLM_OK);
	int32_t c = 0;
	G1A_CHECK_EQ(sslm_prefix_prefill(fx.cpu, root, &admitted, 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &c), SSLM_OK);
	G1A_CHECK_EQ(sslm_prefix_freeze(root), SSLM_OK);
	G1A_CHECK_EQ(sslm_prefix_begin_from(root, 32, &child), SSLM_OK);
	if (child) {
		G1A_CHECK_EQ(cpu.PrefixPrefill(child, pkv::Stream(85, 20, V)), SSLM_OK);
		G1A_CHECK_EQ(sslm_prefix_freeze(child), SSLM_OK);
		G1A_CHECK_EQ(sslm_seq_create_budgeted(fx.cpu, &cpu.pool, 32, &s), SSLM_OK);
	}
	if (s) {
		G1A_CHECK_EQ(sslm_seq_set_schema(s, schema), SSLM_OK);
		G1A_CHECK_EQ(sslm_seq_adopt_prefix(s, child), SSLM_OK);
		sslm_stats_out st{};
		G1A_CHECK_EQ(sslm_stats(fx.cpu, s, &st), SSLM_OK);
		o.accepting = st.schema_accepting;
		sslm_decode_params p{};
		p.layer_budget = static_cast<int32_t>(fx.geo.layers);
		sslm_seq b[1] = {s};
		for (int i = 0; i < k816Steps; ++i) {
			int32_t tok = -7;
			const sslm_status st = sslm_decode_step(fx.cpu, b, 1, &p, nullptr, &tok);
			G1A_CHECK_EQ(st, SSLM_OK);
			if (st != SSLM_OK) break;
			o.tokens.push_back(tok);
		}
		o.L = n0 + 1 + 20 + Rows816(o.tokens);
		G1A_CHECK_MSG(cpu.Rows(s, o.L, &o.rows), "8.16: CPU rows unreadable");
		sslm_seq_release(s);
	}
	if (child) sslm_prefix_release(child);
	sslm_prefix_release(root);
	return o;
}

std::string Join816(const std::vector<int32_t>& t) {
	std::string o;
	for (size_t i = 0; i < t.size(); ++i) o += (i ? "," : "") + std::to_string(t[i]);
	return o;
}

void Cell816() {
	const Fx& fx = PkvFixture("pkv_def");
	if (!fx.ok) return;
	sslm_schema cpu_schema = nullptr;
	G1A_CHECK_EQ(sslm_schema_lookup(fx.cpu, "g5_minimal_one_field", &cpu_schema), SSLM_OK);
	const int32_t gpu_schema = SslmGpuSchemaLookupForG5Bridge(fx.gpu, "g5_minimal_one_field");
	G1A_CHECK_MSG(gpu_schema >= 0 && cpu_schema, "8.16: the fixture's schema is not found");
	if (gpu_schema < 0 || !cpu_schema) return;
	const int32_t admitted = AdmittedToken(fx, cpu_schema);
	if (admitted < 0) return;
	for (int32_t n0 : {15, 16}) {
		const SchemaOut g = SchemaRunGpu(fx, gpu_schema, n0, admitted);
		const SchemaOut c = SchemaRunCpu(fx, cpu_schema, n0, admitted);
		G1A_CHECK_MSG(g.tokens == c.tokens && g.tokens.size() == static_cast<size_t>(k816Steps),
		              "8.16 forced token at %d: decode outcomes [%s] differ from the CPU twin's [%s]", n0,
		              Join816(g.tokens).c_str(), Join816(c.tokens).c_str());
		G1A_CHECK_MSG(g.accepting == c.accepting, "8.16 forced token at %d: schema_accepting %d vs %d", n0, g.accepting, c.accepting);
		G1A_CHECK_MSG(g.rows == c.rows && !g.rows.empty(), "8.16 forced token at %d: rows differ from the CPU twin", n0);
	}
}

}  // namespace

G1A_CELL("1.8/G4", "G4", Cell18G4);
G1A_CELL("1.9", "G4", Cell19);
G1A_CELL("6.5b/G4", "G4", Cell65bG4);
G1A_CELL("8.15", "G4", Cell815);
G1A_CELL("8.16", "G4", Cell816);
