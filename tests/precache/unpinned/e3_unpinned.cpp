// Precached prefixes, plan revs 1.13-1.17: the unpinned E3 legs, the restore hint at the rule B.
//
//   4.7/E3:<model>   4.7's nested leg: a matching hint at origin B-1, B, B+1 (and, on a cap B does not divide, a
//                    parent ending at the last full page with the child running to the cap): s = floor(origin/B),
//                    the rows [s*B, L) come from the blob, and the restore draws ceil(L/B) - s (fill probe);
//   9.8/E3:<model>   9.8's nested leg: a nested blob from a rule-B page pool restored into a contiguous pool, where
//                    every operand of the hint uses the restore pool's B (the cap), never the model's rule B.
//
// Same harness rule as e2_unpinned.cpp (§7 preamble): each cell maps its own model instance, the test-written rule B
// feeds every helper, and the engine's B is asserted equal to it before any other check. Written after the build, so
// no cell here is red-first; each counts once its named mutant (apply_mutant.py) has turned it red.

#include "unpinned_common.h"

namespace {
using namespace pc;
using namespace pc::unpinned;

// A child of a frozen parent by begin_from, `extension` positions prefilled (none: begin_from then freeze with no
// prefill, so L = origin), frozen. Owned by `h`.
sslm_prefix ChildOf(const Fixture& fx, sslm_prefix parent, Handles& h, int extension, int seed = 62) {
    sslm_prefix child = nullptr;
    Status(sslm_prefix_begin_from(parent, std::max(extension, 1), &child), SSLM_OK, "begin_from fixture");
    if (!child) return nullptr;
    h.prefixes.push_back(child);
    if (extension > 0)
        Status(PrefixPrefillAll(fx.model, child, Stream(seed, extension, fx.vocab), 64), SSLM_OK,
               "child prefill fixture");
    Status(sslm_prefix_freeze(child), SSLM_OK, "child freeze fixture");
    return child;
}

// ---- 4.7 [E3]: the nested leg at the rule B ------------------------------------------------------------------------

// A rebuildable page-pool state for the two-sided fill probe: a fresh pool of `pages`, the parent prefilled to
// `origin` and frozen, then (when `restore`) the nested blob restored with that parent as its hint.
struct NestedState : ProbeState {
    std::unique_ptr<PagePool> pool;
    Handles h;
    sslm_kv_pool* Pool() override { return &pool->pool; }
    ~NestedState() override { h.ReleaseAll(); }
};

void Nested47(const Shape& s) {
    const Mapped m = MapOwn(s);
    if (!m.fx || !m.fx->ok) return;
    const Fixture& fx = *m.fx;
    if (!RequireRuleB(fx, s, false)) return;
    const int64_t B = fx.B(), cap = fx.geo.context_cap, cp = fx.CapPages();
    PKV_CHECK_MSG(cp >= 3, "4.7 %s: cap_pages %lld < 3 (S29)", s.name, static_cast<long long>(cp));
    if (cp < 3) return;
    std::vector<std::pair<int64_t, int64_t>> cases;  // (origin, L)
    for (const int64_t origin : {B - 1, B, B + 1}) cases.push_back({origin, 2 * B + 1});
    // A cap B does not divide (pkv_odd): the parent ends at the last full page, the child runs into the partial one.
    if (cap % B != 0) cases.push_back({(cp - 1) * B, cap});
    for (const auto& c : cases) {
        const int64_t origin = c.first, L = c.second, shared_expect = origin / B;
        const int64_t drawn = CeilDiv(L, B) - shared_expect;
        const int64_t parent_pages = CeilDiv(origin, B);
        char what[128];
        std::snprintf(what, sizeof what, "4.7 %s nested origin=%lld L=%lld", s.name, static_cast<long long>(origin),
                      static_cast<long long>(L));
        Scene sc;
        // Parent, child, the restored copy and one adopter at a time (a whole_reserve one reserves cap_pages).
        PagePool& pool = sc.AddPool(fx.model, uint32_t(3 * cp + 8));
        Status(pool.status, SSLM_OK, "4.7 nested page pool");
        if (pool.status != SSLM_OK) return;
        sslm_prefix parent = Root(fx, &pool.pool, sc.h, int(origin), 61);
        sslm_prefix child = parent ? ChildOf(fx, parent, sc.h, int(L - origin)) : nullptr;
        if (!child) return;
        const auto parent_blob = Save(parent);
        const auto blob = Save(child);
        if (!RequireBlob(parent_blob, what) || !RequireBlob(blob, what) || !Header(blob, L, fx)) return;
        PKV_CHECK_MSG(static_cast<int64_t>(Le64(blob, 120)) == origin, "%s: the blob's origin is %lld", what,
                      static_cast<long long>(Le64(blob, 120)));
        // The draw: exactly ceil(L/B) - s pages, two-sided, on a fresh pool holding only the parent.
        const uint32_t N = uint32_t(parent_pages + cp + 2);
        auto build = [&fx, &blob, N, origin](bool restore) -> SceneBuilder {
            return [&fx, &blob, N, origin, restore]() -> std::unique_ptr<ProbeState> {
                auto st = std::make_unique<NestedState>();
                st->pool = std::make_unique<PagePool>(fx.model, N);
                Status(st->pool->status, SSLM_OK, "4.7 nested fill-probe pool");
                if (st->pool->status != SSLM_OK) return st;
                sslm_prefix p = Root(fx, &st->pool->pool, st->h, int(origin), 61);
                if (restore && p) {
                    sslm_prefix r = nullptr;
                    Restored(fx, &st->pool->pool, blob, p, st->h, &r);
                }
                return st;
            };
        };
        const int64_t free_parent = int64_t(N) - parent_pages;
        const bool before = ProbeExactlyFree(fx, build(false), free_parent);
        const bool after = ProbeExactlyFree(fx, build(true), free_parent - drawn);
        PKV_CHECK_MSG(before && after,
                      "%s: the hinted restore's draw is not exactly ceil(L/B) - s = %lld pages (fill probe: before %d, "
                      "after %d)",
                      what, static_cast<long long>(drawn), before, after);
        // The restore against the live parent: s = floor(origin/B), the parent's leading pages physically shared.
        sslm_prefix copy = nullptr;
        uint32_t shared = UINT32_MAX;
        if (!Restored(fx, &pool.pool, blob, parent, sc.h, &copy, &shared)) continue;
        PKV_CHECK_MSG(int64_t(shared) == shared_expect, "%s: *out_shared_pages %u, expected floor(origin/B) = %lld",
                      what, shared, static_cast<long long>(shared_expect));
        const auto ptable = PrefixTable(parent), ctable = PrefixTable(copy);
        bool table_ok = ctable.size() == size_t(CeilDiv(L, B)) && ptable.size() >= size_t(shared_expect) &&
                        std::equal(ptable.begin(), ptable.begin() + shared_expect, ctable.begin());
        for (size_t j = size_t(shared_expect); table_ok && j < ctable.size(); ++j)
            table_ok = std::find(ptable.begin(), ptable.end(), ctable[j]) == ptable.end();
        PKV_CHECK_MSG(table_ok,
                      "%s: the restored table does not share the parent's first %lld pages and map %lld private ones",
                      what, static_cast<long long>(shared_expect), static_cast<long long>(drawn));
        // The rows: every one equal to the blob's by peek, the shared pages included; the rows [s*B, L) the blob's.
        PKV_CHECK_MSG(Save(copy) == blob, "%s: the restored child re-saves different bytes", what);
        RowsEqualBlob(fx, pool.pool, copy, blob, L, what);
        // Nothing was scattered into a shared page: the parent re-saves its own bytes.
        PKV_CHECK_MSG(Save(parent) == parent_blob, "%s: the parent's re-save changed after the hinted restore", what);
        // Adopters of each mode against the begin_from-built twin, min(64, cap - L) tokens; none at L = cap.
        const int continuation = int(std::min<int64_t>(64, cap - L));
        if (continuation > 0) TwinTokens(fx, &pool.pool, child, copy, continuation);
    }
}
PKV_CELL("4.7/E3:pkv_def", "E3", [] { Nested47(kPkvDef); });
PKV_CELL("4.7/E3:pkv_odd", "E3", [] { Nested47(kPkvOdd); });
PKV_CELL("4.7/E3:pkv_32k", "E3", [] { Nested47(kPkv32k); });
PKV_CELL("4.7/E3:pkv_qk", "E3", [] { Nested47(kPkvQk); });
PKV_CELL("4.7/E3:synthetic0p5", "E3", [] { Nested47(kSyn05); });
PKV_CELL("4.7/E3:synthetic1p5", "E3", [] { Nested47(kSyn15); });

// ---- 9.8 [E3]: the nested leg in the contiguous pool (rev 1.14; re-gate audit B1) ----------------------------------

constexpr uint32_t k98Blocks = 5;  // parent + twin + the restored child + one adopter, one spare (N >= 3, S29)

// One restore of `blob` into the contiguous pool with `hint`: keeps, *out_shared_pages == `shared_expect`, the
// single-page count reads `drawn` drawn, the re-save is the blob. Returns the restored prefix (owned by `h`).
sslm_prefix RestoreIntoContig(const Fixture& fx, ContigPool& contig, Handles& h, const std::vector<uint8_t>& blob,
                              sslm_prefix hint, uint32_t shared_expect, int64_t drawn, const char* what) {
    const int64_t before = SinglePageCount(fx, contig.pool);
    sslm_prefix copy = nullptr;
    uint32_t shared = UINT32_MAX;
    const bool kept = Restored(fx, &contig.pool, blob, hint, h, &copy, &shared);
    const int64_t after = SinglePageCount(fx, contig.pool);
    PKV_CHECK_MSG(before >= 0 && after == before - drawn,
                  "%s: the single-page count reads %lld then %lld free, expected %lld drawn", what,
                  static_cast<long long>(before), static_cast<long long>(after), static_cast<long long>(drawn));
    if (!kept) return nullptr;
    PKV_CHECK_MSG(shared == shared_expect, "%s: *out_shared_pages %u, expected %u", what, shared, shared_expect);
    PKV_CHECK_MSG(Save(copy) == blob, "%s: the contiguous restore re-saves different bytes", what);
    return copy;
}

void NestedBothKinds98(const Shape& s) {
    const Mapped m = MapOwn(s);
    if (!m.fx || !m.fx->ok) return;
    const Fixture& fx = *m.fx;
    // Before anything else (§7 preamble): the engine's B, the helpers' B, the rule, rule B < cap.
    if (!RequireRuleB(fx, s, true)) return;
    const int64_t B = fx.B(), cap = fx.geo.context_cap, cp = fx.CapPages();
    // (i) and (ii): origin in {B_r, B_r + 1}, L = 2 B_r + 1.
    for (const int64_t origin : {B, B + 1}) {
        const int64_t L = 2 * B + 1;
        char what[128];
        Scene sc;  // the page pool and its handles
        PagePool& page = sc.AddPool(fx.model, uint32_t(3 * cp + 8));
        ContigPool contig(fx.model, k98Blocks);
        Handles ch;  // the contiguous pool's handles, released before it is destroyed
        Status(page.status, SSLM_OK, "9.8 nested page pool");
        Status(contig.status, SSLM_OK, "9.8 nested contiguous pool");
        if (page.status != SSLM_OK || contig.status != SSLM_OK) return;
        // The nested child, built by begin_from in the rule-B page pool, and its blob.
        sslm_prefix parent_page = Root(fx, &page.pool, sc.h, int(origin), 61);
        sslm_prefix child_page = parent_page ? ChildOf(fx, parent_page, sc.h, int(L - origin)) : nullptr;
        if (!child_page) return;
        const auto blob = Save(child_page);
        std::snprintf(what, sizeof what, "9.8 %s nested origin=%lld", s.name, static_cast<long long>(origin));
        if (!RequireBlob(blob, what) || !Header(blob, L, fx)) return;
        PKV_CHECK_MSG(static_cast<int64_t>(Le64(blob, 120)) == origin, "%s: the blob's origin is %lld", what,
                      static_cast<long long>(Le64(blob, 120)));
        // The contiguous parent: the same tokens, prefill-built in the contiguous pool, matching in every respect
        // but the pool kind (its save is the page-pool parent's), and the contiguous twin built from it.
        sslm_prefix parent_contig = Root(fx, &contig.pool, ch, int(origin), 61);
        if (!parent_contig) return;
        const auto parent_blob = Save(parent_contig);
        PKV_CHECK_MSG(parent_blob == Save(parent_page), "%s: the contiguous parent's save is not the page parent's",
                      what);
        sslm_prefix twin_contig = ChildOf(fx, parent_contig, ch, int(L - origin));
        if (!twin_contig) return;
        PKV_CHECK_MSG(Save(twin_contig) == blob, "%s: the contiguous twin's save is not the page-pool child's", what);
        const int continuation = int(std::min<int64_t>(64, cap - L));
        // (i) A matching parent in that contiguous pool: s = floor(origin/cap) = 0, private, ceil(L/cap) = 1 drawn.
        std::snprintf(what, sizeof what, "9.8 %s nested origin=%lld (i) contiguous parent", s.name,
                      static_cast<long long>(origin));
        sslm_prefix copy = RestoreIntoContig(fx, contig, ch, blob, parent_contig, 0, 1, what);
        PKV_CHECK_MSG(Save(parent_contig) == parent_blob, "%s: the parent's re-save changed (a row scattered into it)",
                      what);
        if (copy) {
            if (continuation > 0) TwinTokens(fx, &contig.pool, twin_contig, copy, continuation);
            Status(ReleasePrefix(ch, copy), SSLM_OK, "9.8 nested release (i)");
        }
        // (ii) The parent in the page pool: (d) fails, s = 0, private, the same counts.
        std::snprintf(what, sizeof what, "9.8 %s nested origin=%lld (ii) page-pool parent", s.name,
                      static_cast<long long>(origin));
        copy = RestoreIntoContig(fx, contig, ch, blob, parent_page, 0, 1, what);
        PKV_CHECK_MSG(Save(parent_page) == parent_blob, "%s: the page-pool parent's re-save changed", what);
        if (copy) {
            if (continuation > 0) TwinTokens(fx, &contig.pool, twin_contig, copy, continuation);
            Status(ReleasePrefix(ch, copy), SSLM_OK, "9.8 nested release (ii)");
        }
        ch.ReleaseAll();
    }
    // (iii) origin = L = cap: begin_from then freeze with no prefill; a matching contiguous parent of length cap.
    {
        char what[128];
        std::snprintf(what, sizeof what, "9.8 %s nested origin=L=cap (iii)", s.name);
        Scene sc;
        PagePool& page = sc.AddPool(fx.model, uint32_t(2 * cp + 8));
        ContigPool contig(fx.model, k98Blocks);
        Handles ch;
        Status(page.status, SSLM_OK, "9.8 nested (iii) page pool");
        Status(contig.status, SSLM_OK, "9.8 nested (iii) contiguous pool");
        if (page.status != SSLM_OK || contig.status != SSLM_OK) return;
        sslm_prefix parent_page = Root(fx, &page.pool, sc.h, int(cap), 61);
        sslm_prefix child_page = parent_page ? ChildOf(fx, parent_page, sc.h, 0) : nullptr;
        if (!child_page) return;
        const auto blob = Save(child_page);
        if (!RequireBlob(blob, what) || !Header(blob, cap, fx)) return;
        PKV_CHECK_MSG(static_cast<int64_t>(Le64(blob, 120)) == cap, "%s: the blob's origin is %lld", what,
                      static_cast<long long>(Le64(blob, 120)));
        sslm_prefix parent_contig = Root(fx, &contig.pool, ch, int(cap), 61);
        if (!parent_contig) return;
        const auto parent_blob = Save(parent_contig);
        PKV_CHECK_MSG(parent_blob == Save(parent_page), "%s: the contiguous parent's save is not the page parent's",
                      what);
        // s = floor(cap/cap) = 1: the whole page shared, nothing drawn.
        sslm_prefix copy = RestoreIntoContig(fx, contig, ch, blob, parent_contig, 1, 0, what);
        if (copy) {
            const auto pt = PrefixTable(parent_contig), ct = PrefixTable(copy);
            PKV_CHECK_MSG(pt.size() == 1 && ct.size() == 1 && pt[0] == ct[0],
                          "%s: the restored child does not map the parent's one cap page", what);
        }
        PKV_CHECK_MSG(Save(parent_contig) == parent_blob, "%s: the parent's re-save changed", what);
        ch.ReleaseAll();
    }
}
PKV_CELL("9.8/E3:pkv_def", "E3", [] { NestedBothKinds98(kPkvDef); });
PKV_CELL("9.8/E3:pkv_odd", "E3", [] { NestedBothKinds98(kPkvOdd); });
PKV_CELL("9.8/E3:pkv_32k", "E3", [] { NestedBothKinds98(kPkv32k); });
PKV_CELL("9.8/E3:pkv_qk", "E3", [] { NestedBothKinds98(kPkvQk); });
PKV_CELL("9.8/E3:synthetic0p5", "E3", [] { NestedBothKinds98(kSyn05); });
PKV_CELL("9.8/E3:synthetic1p5", "E3", [] { NestedBothKinds98(kSyn15); });

}  // namespace
