// Precached prefixes, plan revs 1.13-1.17: the unpinned E2 cells.
//
//   4.7/E2:<model>      the rule B and a partial last page (its nested leg is E3's, not here);
//   8.13/E2             adopt across page sizes, restored against built;
//   9.8/E2:<model>      one blob through a rule-B page pool and a contiguous pool (nested leg: E3);
//   7.12/E2:pool-kind   7.12's pool-kind leg.
//
// §7 preamble (rev 1.14, 1.15): an unpinned cell maps its own model instance, never through GetFixture,
// SyntheticFixture or LoadSyntheticShape (each pins to 16 in superslm_precache_e1); it feeds the test-written
// rule B to every test-side helper; and before any other check it asserts sslm_kv_page_positions equals the rule B
// computed here, and fx.B() equal to it. This file is built only into superslm_precache_unpinned, which is compiled
// with PKV_FIXTURE_B_FROM_RULE and without PKV_PIN_GENERATED_FIXTURES, so Fixture::B() and with it CapPages(), R(),
// PageBytes(), ProbeSplit and PeekPage read the rule. It sits outside precache.cmake's e*.cpp glob, so the define
// never reaches superslm_precache_e1.
//
// PC_UNPINNED_RECEIPT_VIA_GETFIXTURE builds the same cells with the paged-KV fixtures taken from GetFixture under the
// e1 target's PKV_PIN_GENERATED_FIXTURES (superslm_precache_unpinned_getfixture_receipt). It exists for the plan's
// receipt: a 9.8 written through GetFixture("pkv_def") goes red on its B assertion (16, not 128). It is expected red.
//
// Written after the build: no cell here is red-first; each counts once its named mutant has turned it red
// (build/e2-ruleb-cells.md carries the receipts).

#include "precache_common.h"

#if !defined(PC_UNPINNED_RECEIPT_VIA_GETFIXTURE)
#if !defined(PKV_FIXTURE_B_FROM_RULE) || defined(PKV_PIN_GENERATED_FIXTURES)
#error "the unpinned cells build only with PKV_FIXTURE_B_FROM_RULE and without PKV_PIN_GENERATED_FIXTURES (§7)"
#endif
#endif

namespace {
using namespace pc;

// ---- the page-size rule, written in this test (S30; paged-KV §3.1) ------------------------------------------------
// S = 4 KiB when D <= 64 and cap <= 4,096, else 8 KiB; B is the smallest power of two >= 16 with B * D * elem >= S,
// clamped to the cap. Every model here has int8 K/V (elem 1); the page-byte check below corroborates it.
int64_t TestRuleB(uint64_t D, uint64_t cap, uint64_t elem) {
    const uint64_t S = (D <= 64 && cap <= 4096) ? 4096 : 8192;
    uint64_t B = 16;
    while (B * D * elem < S) B *= 2;
    return static_cast<int64_t>(B > cap ? cap : B);
}
constexpr uint64_t kElem = 1;

struct Shape {
    const char* name;
    const char* file;
    bool synthetic;  // SUPERSLM_PRECACHE_SYNTHETIC_DIR, else SUPERSLM_PAGED_KV_FIXTURE_DIR
    Geometry geo;
    int64_t plan_b;  // the plan's S30 figure, a third operand beside the rule and the engine
};

// 4.7's list (rev 1.13): the four generated fixtures and the two synthetic shapes.
const Shape kPkvDef{"pkv_def", "pkv_def.sslm", false, {2, 2, 48, 4096}, 128};
const Shape kPkvOdd{"pkv_odd", "pkv_odd.sslm", false, {2, 2, 48, 4100}, 256};
const Shape kPkv32k{"pkv_32k", "pkv_32k.sslm", false, {2, 2, 48, 32768}, 256};
const Shape kPkvQk{"pkv_qk", "pkv_qk.sslm", false, {2, 2, 128, 4096}, 64};
const Shape kSyn05{"synthetic0p5", "synthetic0p5.sslm", true, {24, 2, 64, 4096}, 64};
const Shape kSyn15{"synthetic1p5", "synthetic1p5.sslm", true, {28, 2, 128, 32768}, 64};
// 7.12's own L = 65 fixture model (rev 1.14, audit S7): the synthetic 0.5B shape, three schemas.
const Shape kSyn05Three{"synthetic0p5-three-schema", "synthetic0p5-three-schema.sslm", true, {24, 2, 64, 4096}, 64};

// One model instance of the cell's own.
struct Mapped {
    std::unique_ptr<Fixture> own;
    const Fixture* fx = nullptr;
};

Mapped MapOwn(const Shape& s) {
    Mapped m;
#if defined(PC_UNPINNED_RECEIPT_VIA_GETFIXTURE)
    if (!s.synthetic) {  // the receipt's wrong construction: the e1 target's pinned, cached mapping
        m.fx = &GetFixture(s.name);
        return m;
    }
#endif
    auto f = std::make_unique<Fixture>();
    const char* env = s.synthetic ? "SUPERSLM_PRECACHE_SYNTHETIC_DIR" : "SUPERSLM_PAGED_KV_FIXTURE_DIR";
    const char* dir = std::getenv(env);
    PKV_CHECK_MSG(dir && *dir, "%s is unset (model %s)", env, s.name);
    f->stem = s.name;
    f->geo = s.geo;
    f->vocab = 256;
    std::vector<uint8_t> raw;
    const std::string path = std::string(dir ? dir : "") + "/" + s.file;
    const bool read = dir && *dir && ReadAll(path, &raw) && !raw.empty();
    PKV_CHECK_MSG(read, "model fixture %s is missing or empty", path.c_str());
    if (read) {
        f->size = raw.size();
        f->bytes.Reset(raw.size());
        std::memcpy(f->bytes.p, raw.data(), raw.size());
        f->sha = Sha(raw.data(), raw.size());
        const sslm_status st = sslm_model_map(f->bytes.p, f->size, &f->model);
        PKV_CHECK_MSG(st == SSLM_OK, "sslm_model_map(%s) == %d", s.name, int(st));
        f->ok = st == SSLM_OK;
    }
    m.fx = f.get();
    m.own = std::move(f);
    return m;
}

// §7 preamble: the first checks of every unpinned cell. `below_cap`: 8.13 and 9.8 also require rule B < cap, so the
// two pool kinds have two page sizes.
bool RequireRuleB(const Fixture& fx, const Shape& s, bool below_cap) {
    if (!fx.ok) return false;
    const int64_t rule = TestRuleB(fx.geo.head_dim, uint64_t(fx.geo.context_cap), kElem);
    const int64_t engine = sslm_kv_page_positions(fx.model);
    PKV_CHECK_MSG(engine == rule, "%s: sslm_kv_page_positions reads %lld; the test-written rule B is %lld", s.name,
                  static_cast<long long>(engine), static_cast<long long>(rule));
    PKV_CHECK_MSG(fx.B() == rule, "%s: the test-side helpers read B = %lld, not the rule's %lld", s.name,
                  static_cast<long long>(fx.B()), static_cast<long long>(rule));
    if (engine != rule || fx.B() != rule) return false;
    PKV_CHECK_MSG(rule == s.plan_b, "%s: the rule gives B = %lld; the plan's S30 figure is %lld", s.name,
                  static_cast<long long>(rule), static_cast<long long>(s.plan_b));
    const size_t page = sslm_kv_page_size(fx.model);
    PKV_CHECK_MSG(page == size_t(rule) * fx.BytesPerToken() * kElem,
                  "%s: page bytes %zu, expected B * L * 2 * H_kv * D * elem = %zu", s.name, page,
                  size_t(rule) * fx.BytesPerToken() * kElem);
    bool ok = rule == s.plan_b && page == size_t(rule) * fx.BytesPerToken() * kElem;
    if (below_cap) {
        PKV_CHECK_MSG(rule < int64_t(fx.geo.context_cap), "%s: rule B %lld is not below the cap %lld", s.name,
                      static_cast<long long>(rule), static_cast<long long>(fx.geo.context_cap));
        ok = ok && rule < int64_t(fx.geo.context_cap);
    }
    std::printf("%s: rule B = %lld (engine %lld, helpers %lld), cap %lld, cap_pages %lld, last page holds %lld\n",
                s.name, static_cast<long long>(rule), static_cast<long long>(engine), static_cast<long long>(fx.B()),
                static_cast<long long>(fx.geo.context_cap), static_cast<long long>(fx.CapPages()),
                static_cast<long long>(fx.geo.context_cap - (fx.CapPages() - 1) * fx.B()));
    return ok;
}

int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// ---- pools and instruments -----------------------------------------------------------------------------------------

// The contiguous pool option (sslm_kv_pool_create_contiguous): B = cap, one page per block.
struct ContigPool {
    AlignedBuf mem;
    sslm_kv_pool pool = nullptr;
    sslm_status status = SSLM_INVALID_ARGUMENT;
    ContigPool(sslm_model model, uint32_t blocks) {
        mem.Reset(sslm_kv_pool_contiguous_size(model, blocks), 0);
        status = sslm_kv_pool_create_contiguous(model, mem.p, mem.n, blocks, &pool);
    }
    ~ContigPool() {
        if (pool) sslm_kv_pool_destroy(pool);
    }
};

// 9.8's single-page count (rev 1.13; S30 (d), S29): in a contiguous pool every budget-1 create reserves exactly one
// page, so the free pages are the creates admitted before the first refusal, which must be SSLM_KV_POOL_EXHAUSTED.
// Every admitted create is released again, so the state is left as found. -1 when the refusal had another status.
int64_t SinglePageCount(const Fixture& fx, sslm_kv_pool pool) {
    std::vector<sslm_seq> made;
    sslm_status st = SSLM_OK;
    for (int guard = 0; guard < 4096; ++guard) {
        sslm_seq s = nullptr;
        st = sslm_seq_create_budgeted(fx.model, &pool, 1, &s);
        if (st != SSLM_OK) break;
        made.push_back(s);
    }
    PKV_CHECK_MSG(st == SSLM_KV_POOL_EXHAUSTED,
                  "single-page count: create %zu refused with %s (%d), not SSLM_KV_POOL_EXHAUSTED", made.size() + 1,
                  AbiStatusName(st), int(st));
    for (auto it = made.rbegin(); it != made.rend(); ++it) sslm_seq_release(*it);
    return st == SSLM_KV_POOL_EXHAUSTED ? int64_t(made.size()) : -1;
}

// A rebuildable page-pool state for the two-sided fill probe: a fresh pool of `pages`, then the blob restored into it
// when `restore` is set. The handles are released before the pool is destroyed.
struct RestoreState : ProbeState {
    std::unique_ptr<PagePool> pool;
    Handles h;
    sslm_kv_pool* Pool() override { return &pool->pool; }
    ~RestoreState() override { h.ReleaseAll(); }
};

// The restore's draw, read by the two-sided fill probe before and after: a fresh pool of N pages reads exactly N
// free, and exactly N - drawn once the blob is restored into it. Graded by admission only (§8).
bool DrawReads(const Fixture& fx, const std::vector<uint8_t>& blob, uint32_t N, int64_t drawn, const char* what) {
    auto build = [&](bool restore) -> SceneBuilder {
        return [&fx, &blob, N, restore]() -> std::unique_ptr<ProbeState> {
            auto s = std::make_unique<RestoreState>();
            s->pool = std::make_unique<PagePool>(fx.model, N);
            Status(s->pool->status, SSLM_OK, "fill-probe page pool");
            if (restore && s->pool->status == SSLM_OK) {
                sslm_prefix r = nullptr;
                Restored(fx, &s->pool->pool, blob, nullptr, s->h, &r);
            }
            return s;
        };
    };
    const bool before = ProbeExactlyFree(fx, build(false), N);
    const bool after = ProbeExactlyFree(fx, build(true), int64_t(N) - drawn);
    PKV_CHECK_MSG(before && after, "%s: the restore's draw is not exactly %lld pages (fill probe: before %d, after %d)",
                  what, static_cast<long long>(drawn), before, after);
    return before && after;
}

// The restored rows against the blob's, by peek (§7 grading rule), addressed in the test with 1.13.0's run-slot
// rotation (pkv::PageRunOffset) at the pool's B; the blob's rows are canonical [run][pos < L][d] after the header and
// the residual (§3.2).
bool RowsEqualBlob(const Fixture& fx, sslm_kv_pool pool, sslm_prefix p, const std::vector<uint8_t>& blob, int64_t L,
                   const char* what) {
    const int64_t B = fx.B();
    const auto table = PrefixTable(p);
    PKV_CHECK_MSG(table.size() == size_t(CeilDiv(L, B)), "%s: the restored table maps %zu pages, expected %lld", what,
                  table.size(), static_cast<long long>(CeilDiv(L, B)));
    if (table.size() != size_t(CeilDiv(L, B))) return false;
    const size_t D = fx.geo.head_dim, H = fx.geo.kv_heads, base = kHeader + ArtifactHiddenSize(fx);
    if (blob.size() < base + size_t(L) * fx.BytesPerToken()) return false;
    size_t bad = 0, first_bad = SIZE_MAX;
    for (size_t j = 0; j < table.size(); ++j) {
        const auto page = PeekPage(fx, pool, table[j]);
        PKV_CHECK_MSG(page.size() == fx.PageBytes(), "%s: page-index peek of page %u was short", what, table[j]);
        if (page.size() != fx.PageBytes()) return false;
        for (uint32_t l = 0; l < fx.geo.layers; ++l)
            for (uint32_t half = 0; half < 2; ++half)
                for (uint32_t h = 0; h < fx.geo.kv_heads; ++h)
                    for (int64_t pos = int64_t(j) * B; pos < std::min<int64_t>(int64_t(j + 1) * B, L); ++pos) {
                        const size_t from = PageRunOffset(fx.geo.layers, fx.geo.kv_heads, D, B, l, half, h, pos);
                        const size_t run = (size_t(l) * 2 + half) * H + h;
                        const size_t to = base + (run * size_t(L) + size_t(pos)) * D;
                        if (std::memcmp(page.data() + from, blob.data() + to, D) != 0) {
                            ++bad;
                            if (first_bad == SIZE_MAX) first_bad = size_t(pos);
                        }
                    }
    }
    PKV_CHECK_MSG(bad == 0, "%s: %zu restored rows differ from the blob's by peek (first at position %zu)", what, bad,
                  first_bad);
    return bad == 0;
}

// ---- 4.7 [E2]: the rule B and a partial last page, unpinned ----------------------------------------------------------

void RuleB47(const Shape& s) {
    const Mapped m = MapOwn(s);
    if (!m.fx || !m.fx->ok) return;
    const Fixture& fx = *m.fx;
    if (!RequireRuleB(fx, s, false)) return;
    const int64_t B = fx.B(), cap = fx.geo.context_cap, cp = fx.CapPages();
    PKV_CHECK_MSG(cp >= 3, "4.7 %s: cap_pages %lld < 3 (S29)", s.name, static_cast<long long>(cp));
    if (cp < 3) return;
    std::vector<int64_t> lengths{0, 1, B - 1, B, B + 1, (cp - 1) * B, (cp - 1) * B + 1, cap - 1, cap};
    std::sort(lengths.begin(), lengths.end());
    lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
    for (const int64_t L : lengths) {
        char what[96];
        std::snprintf(what, sizeof what, "4.7 %s L=%lld", s.name, static_cast<long long>(L));
        Scene sc;
        // The twin and its restored copy (ceil(L/B) pages each), then one adopter at a time (a whole_reserve one
        // reserves cap_pages): every reading leaves at least 2 free.
        PagePool& pool = sc.AddPool(fx.model, uint32_t(3 * cp + 8));
        Status(pool.status, SSLM_OK, "4.7 page pool");
        if (pool.status != SSLM_OK) return;
        sslm_prefix twin = Root(fx, &pool.pool, sc.h, int(L));
        if (!twin) return;
        const auto blob = Save(twin);
        if (!RequireBlob(blob, what) || !Header(blob, L, fx)) return;
        // The draw: exactly ceil(L/B) pages, two-sided, on a fresh pool of cap_pages + 2.
        DrawReads(fx, blob, uint32_t(cp + 2), CeilDiv(L, B), what);
        sslm_prefix copy = nullptr;
        if (!Restored(fx, &pool.pool, blob, nullptr, sc.h, &copy)) continue;
        // save -> restore -> save (6.4's operand).
        PKV_CHECK_MSG(Save(copy) == blob, "%s: the restored prefix re-saves different bytes", what);
        RowsEqualBlob(fx, pool.pool, copy, blob, L, what);
        // min(64, cap - L) greedy tokens per adopter mode; none at L = 0 or L = cap (4.1's rule, rev 1.14 N3).
        const int continuation = int(std::min<int64_t>(64, cap - L));
        if (L > 0 && continuation > 0) TwinTokens(fx, &pool.pool, twin, copy, continuation);
    }
}
PKV_CELL("4.7/E2:pkv_def", "E2", [] { RuleB47(kPkvDef); });
PKV_CELL("4.7/E2:pkv_odd", "E2", [] { RuleB47(kPkvOdd); });
PKV_CELL("4.7/E2:pkv_32k", "E2", [] { RuleB47(kPkv32k); });
PKV_CELL("4.7/E2:pkv_qk", "E2", [] { RuleB47(kPkvQk); });
PKV_CELL("4.7/E2:synthetic0p5", "E2", [] { RuleB47(kSyn05); });
PKV_CELL("4.7/E2:synthetic1p5", "E2", [] { RuleB47(kSyn15); });

// ---- 9.8 [E2]: one blob, both pool kinds ---------------------------------------------------------------------------

void BothKinds98(const Shape& s) {
    const Mapped m = MapOwn(s);
    if (!m.fx || !m.fx->ok) return;
    const Fixture& fx = *m.fx;
    // Before anything else (§7 preamble; review S4, audit S1): the engine's B, the helpers' B, the rule, rule B < cap.
    if (!RequireRuleB(fx, s, true)) return;
    const int64_t B = fx.B(), cap = fx.geo.context_cap, cp = fx.CapPages();
    PKV_CHECK_MSG(cp >= 3, "9.8 %s: cap_pages %lld < 3", s.name, static_cast<long long>(cp));
    if (cp < 3) return;
    constexpr uint32_t kBlocks = 3;  // N >= 3 for the single-page count (S29, S30 (d))
    for (const int64_t L : {int64_t(0), int64_t(1), B, B + 1, cap - 1, cap}) {
        char what[96];
        std::snprintf(what, sizeof what, "9.8 %s L=%lld", s.name, static_cast<long long>(L));
        Scene sc;  // the page pool and its handles
        PagePool& page = sc.AddPool(fx.model, uint32_t(3 * cp + 8));
        ContigPool contig(fx.model, kBlocks);
        Handles ch;  // the contiguous pool's handles, released before it is destroyed
        Status(page.status, SSLM_OK, "9.8 page pool");
        Status(contig.status, SSLM_OK, "9.8 contiguous pool");
        if (page.status != SSLM_OK || contig.status != SSLM_OK) return;
        PKV_CHECK_MSG(sslm_kv_page_size(fx.model) * size_t(cap) / size_t(B) <= contig.mem.n,
                      "9.8 contiguous pool is not cap-sized blocks");
        sslm_prefix twin_page = Root(fx, &page.pool, sc.h, int(L));
        if (!twin_page) return;
        const auto blob = Save(twin_page);
        if (!RequireBlob(blob, what) || !Header(blob, L, fx)) return;
        // Into the contiguous pool: keeps, draws ceil(L/cap) (0 or 1) by the single-page count.
        const int64_t before = SinglePageCount(fx, contig.pool);
        PKV_CHECK_MSG(before == kBlocks, "%s: a fresh contiguous pool reads %lld free, not %u", what,
                      static_cast<long long>(before), kBlocks);
        sslm_prefix in_contig = nullptr;
        const bool kept = Restored(fx, &contig.pool, blob, nullptr, ch, &in_contig);
        const int64_t after = SinglePageCount(fx, contig.pool);
        PKV_CHECK_MSG(after == int64_t(kBlocks) - CeilDiv(L, cap),
                      "%s: the contiguous restore left %lld free, expected N - ceil(L/cap) = %lld", what,
                      static_cast<long long>(after), static_cast<long long>(int64_t(kBlocks) - CeilDiv(L, cap)));
        if (!kept) continue;
        const auto resaved = Save(in_contig);
        PKV_CHECK_MSG(resaved == blob, "%s: the contiguous restore re-saves different bytes", what);
        // Back into a page pool: draws ceil(L/B) (two-sided fill probe), keeps, re-saves byte-identical.
        DrawReads(fx, resaved, uint32_t(cp + 2), CeilDiv(L, B), what);
        sslm_prefix in_page = nullptr;
        if (!Restored(fx, &page.pool, resaved, nullptr, sc.h, &in_page)) continue;
        PKV_CHECK_MSG(Save(in_page) == blob, "%s: the page-pool restore of the re-saved blob re-saves different bytes",
                      what);
        // The prefill twin built in the contiguous pool, from the same tokens.
        sslm_prefix twin_contig = Root(fx, &contig.pool, ch, int(L));
        if (!twin_contig) continue;
        // Adopters of each mode in each pool kind against that kind's prefill twin; none at L = 0 or L = cap.
        const int continuation = int(std::min<int64_t>(64, cap - L));
        if (L > 0 && continuation > 0) {
            TwinTokens(fx, &page.pool, twin_page, in_page, continuation);
            TwinTokens(fx, &contig.pool, twin_contig, in_contig, continuation);
        }
        ch.ReleaseAll();
    }
}
PKV_CELL("9.8/E2:pkv_def", "E2", [] { BothKinds98(kPkvDef); });
PKV_CELL("9.8/E2:pkv_odd", "E2", [] { BothKinds98(kPkvOdd); });
PKV_CELL("9.8/E2:pkv_32k", "E2", [] { BothKinds98(kPkv32k); });
PKV_CELL("9.8/E2:pkv_qk", "E2", [] { BothKinds98(kPkvQk); });
PKV_CELL("9.8/E2:synthetic0p5", "E2", [] { BothKinds98(kSyn05); });
PKV_CELL("9.8/E2:synthetic1p5", "E2", [] { BothKinds98(kSyn15); });

// ---- 8.13 [E2]: adopt across page sizes, restored against built -----------------------------------------------------

constexpr uint32_t k813PagePages = 48;   // 4 prefix pages + a cap_pages (32) whole_reserve adopter + 12 spare
constexpr uint32_t k813ContigBlocks = 5; // 2 prefixes + 1 adopter + 2 spare
constexpr int32_t k813Budget = 68;       // the budget adopter's budget (Decode's count + 4)

enum class Which { kPageBuilt, kPageRestored, kContigBuilt, kContigRestored };
const char* WhichName(Which w) {
    switch (w) {
        case Which::kPageBuilt: return "page-pool built";
        case Which::kPageRestored: return "page-pool restored";
        case Which::kContigBuilt: return "contiguous built";
        case Which::kContigRestored: return "contiguous restored";
    }
    return "?";
}
bool InPage(Which w) { return w == Which::kPageBuilt || w == Which::kPageRestored; }

// The four prefixes at L = B + 1 (two pages in the page pool, one in the contiguous pool). Rebuildable, so the page
// pool can be read by the two-sided fill probe.
struct World813 : ProbeState {
    const Fixture* fx = nullptr;
    std::unique_ptr<PagePool> page;
    std::unique_ptr<ContigPool> contig;
    Handles h;
    sslm_prefix p[4] = {};
    std::vector<uint8_t> blob_page, blob_contig;
    bool ok = false;
    sslm_kv_pool* Pool() override { return &page->pool; }
    sslm_prefix Get(Which w) const { return p[int(w)]; }
    explicit World813(const Fixture& f) : fx(&f) {
        const int L = int(f.B()) + 1;
        page = std::make_unique<PagePool>(f.model, k813PagePages);
        contig = std::make_unique<ContigPool>(f.model, k813ContigBlocks);
        Status(page->status, SSLM_OK, "8.13 page pool");
        Status(contig->status, SSLM_OK, "8.13 contiguous pool");
        if (page->status != SSLM_OK || contig->status != SSLM_OK) return;
        p[0] = Root(f, &page->pool, h, L);
        if (!p[0]) return;
        blob_page = Save(p[0]);
        if (!RequireBlob(blob_page, "8.13 page-pool blob") || !Restored(f, &page->pool, blob_page, nullptr, h, &p[1]))
            return;
        p[2] = Root(f, &contig->pool, h, L);
        if (!p[2]) return;
        blob_contig = Save(p[2]);
        if (!RequireBlob(blob_contig, "8.13 contiguous blob") ||
            !Restored(f, &contig->pool, blob_contig, nullptr, h, &p[3]))
            return;
        ok = true;
    }
    // An adopter in the pool kind other than `w`'s; owned by h.
    sslm_seq Adopter(Which w, bool budget) {
        sslm_kv_pool* other = InPage(w) ? &contig->pool : &page->pool;
        sslm_seq s = nullptr;
        const sslm_status st = budget ? sslm_seq_create_budgeted(fx->model, other, k813Budget, &s)
                                      : sslm_seq_create(fx->model, other, &s);
        Status(st, SSLM_OK, "8.13 cross-kind adopter create");
        if (s) h.seqs.push_back(s);
        return s;
    }
    ~World813() override { h.ReleaseAll(); }
};

void AcrossPageSizes813() {
    const Mapped m = MapOwn(kPkvDef);
    if (!m.fx || !m.fx->ok) return;
    const Fixture& fx = *m.fx;
    if (!RequireRuleB(fx, kPkvDef, true)) return;
    const int64_t B = fx.B(), L = B + 1, cp = fx.CapPages();
    sslm_schema schema = nullptr;
    Status(sslm_schema_lookup(fx.model, "g5_minimal_one_field", &schema), SSLM_OK, "8.13 schema lookup");
    if (!schema) return;
    World813 w(fx);
    if (!w.ok) return;
    PKV_CHECK_MSG(PrefixTable(w.Get(Which::kPageBuilt)).size() == 2 && PrefixTable(w.Get(Which::kContigBuilt)).size() == 1,
                  "8.13: L = B + 1 is not two pages in the page pool and one in the contiguous pool");
    const int64_t prefix_pages = 2 * CeilDiv(L, B);  // the two page-pool prefixes, frozen (freeze returns the reserve)
    const int64_t prefix_blocks = 2;                 // the two contiguous prefixes, one block each
    sslm_status disposition[4][2];
    for (Which which : {Which::kPageBuilt, Which::kPageRestored, Which::kContigBuilt, Which::kContigRestored}) {
        for (bool budget : {false, true}) {
            char what[96];
            std::snprintf(what, sizeof what, "8.13 %s prefix, %s adopter in the %s pool", WhichName(which),
                          budget ? "budget" : "whole_reserve", InPage(which) ? "contiguous" : "page");
            sslm_seq a = w.Adopter(which, budget);
            if (!a) return;
            const int64_t adopter_contig = InPage(which) ? 1 : 0;
            const int64_t adopter_page = InPage(which) ? 0 : (budget ? fx.R(k813Budget) : cp);
            std::vector<uint8_t> saved_before, saved_after;
            PKV_CHECK_MSG(SaveBlob(a, &saved_before), "%s: adopter save before the call failed", what);
            const int64_t contig_before = SinglePageCount(fx, w.contig->pool);
            const sslm_status st = sslm_seq_adopt_prefix(a, w.Get(which));
            disposition[int(which)][budget] = st;
            Status(st, SSLM_INVALID_ARGUMENT, what);
            // The adopter is unchanged: its save is byte-equal, and the bind was not revoked.
            PKV_CHECK_MSG(SaveBlob(a, &saved_after) && saved_after == saved_before,
                          "%s: the refused adopter's save changed", what);
            Status(sslm_seq_set_schema(a, schema), SSLM_OK, (std::string(what) + ": schema bind after the call").c_str());
            // Nothing drawn in the contiguous pool: the single-page count, before and after the call.
            const int64_t contig_after = SinglePageCount(fx, w.contig->pool);
            const int64_t contig_expect = int64_t(k813ContigBlocks) - prefix_blocks - adopter_contig;
            PKV_CHECK_MSG(contig_before == contig_expect && contig_after == contig_expect,
                          "%s: the contiguous pool reads %lld then %lld free, expected %lld", what,
                          static_cast<long long>(contig_before), static_cast<long long>(contig_after),
                          static_cast<long long>(contig_expect));
            // Nothing drawn in the page pool: the two-sided fill probe on the rebuilt state, without and with the call.
            const int64_t page_expect = int64_t(k813PagePages) - prefix_pages - adopter_page;
            for (bool call : {false, true}) {
                const bool read = ProbeExactlyFree(
                    fx,
                    [&fx, which, budget, call]() -> std::unique_ptr<ProbeState> {
                        auto r = std::make_unique<World813>(fx);
                        if (r->ok) {
                            sslm_seq s = r->Adopter(which, budget);
                            if (s && call) sslm_seq_adopt_prefix(s, r->Get(which));
                        }
                        return r;
                    },
                    page_expect);
                PKV_CHECK_MSG(read, "%s: the page pool does not read %lld free %s the call", what,
                              static_cast<long long>(page_expect), call ? "after" : "before");
            }
            ReleaseSeq(w.h, a);
        }
    }
    // The subject: a restored prefix and the prefill-built one in the same pool give the same disposition.
    for (bool budget : {false, true}) {
        PKV_CHECK_MSG(disposition[int(Which::kPageRestored)][budget] == disposition[int(Which::kPageBuilt)][budget],
                      "8.13 page pool, %s adopter: restored %d, built %d", budget ? "budget" : "whole_reserve",
                      int(disposition[int(Which::kPageRestored)][budget]),
                      int(disposition[int(Which::kPageBuilt)][budget]));
        PKV_CHECK_MSG(
            disposition[int(Which::kContigRestored)][budget] == disposition[int(Which::kContigBuilt)][budget],
            "8.13 contiguous pool, %s adopter: restored %d, built %d", budget ? "budget" : "whole_reserve",
            int(disposition[int(Which::kContigRestored)][budget]), int(disposition[int(Which::kContigBuilt)][budget]));
    }
    // Control: an adopter in the prefix's own pool kind adopts, and its 64 tokens equal the built twin's adopter's.
    TwinTokens(fx, &w.page->pool, w.Get(Which::kPageBuilt), w.Get(Which::kPageRestored), 64);
    TwinTokens(fx, &w.contig->pool, w.Get(Which::kContigBuilt), w.Get(Which::kContigRestored), 64);
    std::printf("8.13: page-pool prefixes refused in the contiguous pool (whole_reserve %s, budget %s); contiguous "
                "prefixes refused in the page pool (whole_reserve %s, budget %s)\n",
                AbiStatusName(disposition[1][0]), AbiStatusName(disposition[1][1]), AbiStatusName(disposition[3][0]),
                AbiStatusName(disposition[3][1]));
}
PKV_CELL("8.13/E2", "E2", AcrossPageSizes813);

// ---- 7.12 [E2]: the pool-kind leg -----------------------------------------------------------------------------------

void PoolKind712() {
    const Mapped m = MapOwn(kSyn05Three);
    if (!m.fx || !m.fx->ok) return;
    const Fixture& fx = *m.fx;
    // Asserted first, against the test-written rule: sslm_kv_page_positions == 64 (rev 1.14, audit S7).
    if (!RequireRuleB(fx, kSyn05Three, true)) return;
    constexpr int L = 65;  // 7.12's own fixture length: L >= B + 1, two pages in the rule-B pool, one contiguous
    sslm_schema schema = nullptr;
    Status(sslm_schema_lookup(fx.model, "schema_two", &schema), SSLM_OK, "7.12 pool-kind schema lookup");
    if (!schema) return;
    PagePool page(fx.model, uint32_t(2 * fx.CapPages() + 8));
    ContigPool contig(fx.model, 4);
    Handles h;
    Status(page.status, SSLM_OK, "7.12 pool-kind page pool");
    Status(contig.status, SSLM_OK, "7.12 pool-kind contiguous pool");
    if (page.status != SSLM_OK || contig.status != SSLM_OK) return;
    // 7.12's fixture: schema_two bound, 65 schema-content tokens of 0 (the test-owned DFA reaches state 1), frozen.
    auto build = [&](sslm_kv_pool* pool) -> sslm_prefix {
        sslm_prefix p = nullptr;
        Status(sslm_prefix_begin_budgeted(fx.model, pool, L, &p), SSLM_OK, "7.12 pool-kind begin");
        if (!p) return nullptr;
        h.prefixes.push_back(p);
        Status(sslm_prefix_set_schema(p, schema), SSLM_OK, "7.12 pool-kind bind schema_two");
        Status(SchemaContentAll(fx.model, p, std::vector<int32_t>(L, 0), 64), SSLM_OK, "7.12 pool-kind prefill");
        Status(sslm_prefix_freeze(p), SSLM_OK, "7.12 pool-kind freeze");
        return p;
    };
    sslm_prefix in_page = build(&page.pool);
    sslm_prefix in_contig = build(&contig.pool);
    if (!in_page || !in_contig) return;
    PKV_CHECK_MSG(PrefixTable(in_page).size() == 2 && PrefixTable(in_contig).size() == 1,
                  "7.12 pool-kind: L = 65 is not two pages at B = 64 and one in the contiguous pool");
    const auto page_blob = Save(in_page);
    const auto contig_blob = Save(in_contig);
    if (!RequireBlob(page_blob, "7.12 pool-kind page") || !RequireBlob(contig_blob, "7.12 pool-kind contiguous"))
        return;
    Header(page_blob, L, fx);
    Header(contig_blob, L, fx);
    PKV_CHECK_MSG(Le64(page_blob, 96) != 0 && Le32(page_blob, 104) == 1,
                  "7.12 pool-kind fixture is not bound with an advanced walk");
    // Each blob into the other pool kind, saved again.
    sslm_prefix page_in_contig = nullptr, contig_in_page = nullptr;
    if (!Restored(fx, &contig.pool, page_blob, nullptr, h, &page_in_contig) ||
        !Restored(fx, &page.pool, contig_blob, nullptr, h, &contig_in_page))
        return;
    const auto page_blob_via_contig = Save(page_in_contig);
    const auto contig_blob_via_page = Save(contig_in_page);
    // One artifact: all four byte-identical, digest and model hash included.
    PKV_CHECK_MSG(page_blob == contig_blob, "7.12 pool-kind: the page-pool and contiguous saves differ");
    PKV_CHECK_MSG(page_blob_via_contig == page_blob, "7.12 pool-kind: the page-pool blob re-saved from the contiguous "
                                                     "pool differs");
    PKV_CHECK_MSG(contig_blob_via_page == page_blob, "7.12 pool-kind: the contiguous blob re-saved from the page pool "
                                                     "differs");
    // 7.12's five counts (nonzero, pairwise distinct, fixed before serialization) through both kinds as well.
    const std::array<uint64_t, 5> counts{11, 13, 17, 19, 23};
    auto seeded = page_blob;
    for (size_t i = 0; i < counts.size(); ++i) PutLe64(seeded, 144 + 8 * i, counts[i]);
    Digest(seeded);
    sslm_prefix seeded_page = nullptr, seeded_contig = nullptr;
    if (!Restored(fx, &page.pool, seeded, nullptr, h, &seeded_page)) return;
    const auto seeded_page_blob = Save(seeded_page);
    if (!Restored(fx, &contig.pool, seeded_page_blob, nullptr, h, &seeded_contig)) return;
    PKV_CHECK_MSG(seeded_page_blob == seeded && Save(seeded_contig) == seeded,
                  "7.12 pool-kind: the count-seeded blob changes through a pool kind");
    h.ReleaseAll();
}
PKV_CELL("7.12/E2:pool-kind", "E2", PoolKind712);

}  // namespace
