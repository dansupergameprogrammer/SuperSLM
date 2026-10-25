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

#include "unpinned_common.h"

namespace {
using namespace pc;
using namespace pc::unpinned;

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
