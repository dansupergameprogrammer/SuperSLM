// Precached prefixes: the helpers the unpinned cells share (E2 cells in e2_unpinned.cpp, E3 nested legs in
// e3_unpinned.cpp). Moved out of e2_unpinned.cpp unchanged. Built only into the unpinned targets (§7 preamble).
#pragma once
#include "precache_common.h"


#if !defined(PC_UNPINNED_RECEIPT_VIA_GETFIXTURE)
#if !defined(PKV_FIXTURE_B_FROM_RULE) || defined(PKV_PIN_GENERATED_FIXTURES)
#error "the unpinned cells build only with PKV_FIXTURE_B_FROM_RULE and without PKV_PIN_GENERATED_FIXTURES (§7)"
#endif
#endif

namespace pc {
namespace unpinned {

// ---- the page-size rule, written in this test (S30; paged-KV §3.1) ------------------------------------------------
// S = 4 KiB when D <= 64 and cap <= 4,096, else 8 KiB; B is the smallest power of two >= 16 with B * D * elem >= S,
// clamped to the cap. Every model here has int8 K/V (elem 1); the page-byte check below corroborates it.
inline int64_t TestRuleB(uint64_t D, uint64_t cap, uint64_t elem) {
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

inline Mapped MapOwn(const Shape& s) {
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
inline bool RequireRuleB(const Fixture& fx, const Shape& s, bool below_cap) {
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

// CeilDiv is pkv::budget_a's (pkv_budget_a_helpers.h), the same formula.

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
inline int64_t SinglePageCount(const Fixture& fx, sslm_kv_pool pool) {
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
inline bool DrawReads(const Fixture& fx, const std::vector<uint8_t>& blob, uint32_t N, int64_t drawn, const char* what) {
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
inline bool RowsEqualBlob(const Fixture& fx, sslm_kv_pool pool, sslm_prefix p, const std::vector<uint8_t>& blob, int64_t L,
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

}  // namespace unpinned
}  // namespace pc
