// SuperSLM 1.18.0 delta prefix saves (plan: superslm-prefix-delta-1p18, §3-§7): the SPD1 blob, its
// restore onto a live parent, and the model-free assemble. Cells D<n>/E6, all on the generated paged-KV
// fixtures (pkv_def at the E1 target's pinned B = 16, pkv_qk as the other model).
//
// The expected SPD1 bytes are built here from the full SPX1 save of the same prefix (ExpectedDelta), and
// the expected assemble output is that full save, so neither reuses the engine's writer.
#include "precache_common.h"
#include <atomic>

#define PC_RESTORE_PARENT_MISMATCH static_cast<sslm_status>(31)
#ifdef SSLM_HAS_PREFIX_DELTA_ABI
static_assert(SSLM_RESTORE_PARENT_MISMATCH == PC_RESTORE_PARENT_MISMATCH,
              "parent mismatch must be appended at ordinal 31");
#else
// The red suite's stubs: the declarations the plan publishes (§4), returning the unbuilt status.
extern "C" {
sslm_status sslm_prefix_save_delta(sslm_prefix, void*, size_t*) { return static_cast<sslm_status>(pc::kUnbuilt); }
sslm_status sslm_prefix_assemble(const void*, size_t, const void*, size_t, void*, size_t*) {
    return static_cast<sslm_status>(pc::kUnbuilt);
}
}
#endif

namespace {
using namespace pc;

constexpr size_t kDeltaHeader = 232;

std::array<uint8_t, 32> Sha32(const uint8_t* p, size_t n) {
    std::array<uint8_t, 32> out{};
    superslm::Sha256Hash(p, n, out.data());
    return out;
}

uint32_t FreePages(sslm_kv_pool pool) {
    sslm_kv_pool_stats_out s{};
    s.struct_size = sizeof(s);
    Status(sslm_kv_pool_stats(pool, &s), SSLM_OK, "pool stats");
    return s.free_pages;
}
uint32_t SharedPages(sslm_kv_pool pool) {
    sslm_kv_pool_stats_out s{};
    s.struct_size = sizeof(s);
    Status(sslm_kv_pool_stats(pool, &s), SSLM_OK, "pool stats");
    return s.shared_pages;
}

size_t Runs(const Fixture& fx) { return size_t{fx.geo.layers} * 2u * fx.geo.kv_heads; }

// A frozen child of `parent`: begin_from, `extension` tokens of Stream(seed), freeze. extension 0 is an
// empty child (L == origin).
sslm_prefix Child(const Fixture& fx, sslm_prefix parent, Handles& h, int extension = 200, int seed = 62) {
    sslm_prefix child = nullptr;
    Status(sslm_prefix_begin_from(parent, std::max(extension, 1), &child), SSLM_OK, "begin_from fixture");
    if (!child) return nullptr;
    h.prefixes.push_back(child);
    if (extension)
        Status(PrefixPrefillAll(fx.model, child, Stream(seed, extension, fx.vocab), 64), SSLM_OK,
               "child prefill fixture");
    Status(sslm_prefix_freeze(child), SSLM_OK, "child freeze fixture");
    return child;
}

std::vector<uint8_t> SaveDelta(sslm_prefix p) {
    size_t n = 0;
    const sslm_status query = sslm_prefix_save_delta(p, nullptr, &n);
    Status(query, SSLM_BUFFER_TOO_SMALL, "sslm_prefix_save_delta size query");
    if (query != SSLM_BUFFER_TOO_SMALL || n < kDeltaHeader) return {};
    std::vector<uint8_t> b(n + 1, 0xa5);
    size_t written = n;
    const sslm_status st = sslm_prefix_save_delta(p, b.data(), &written);
    Status(st, SSLM_OK, "sslm_prefix_save_delta write");
    PKV_CHECK_EQ(written, n);
    PKV_CHECK_EQ(b[n], 0xa5);
    if (st != SSLM_OK || written != n) return {};
    b.resize(n);
    return b;
}

sslm_status Assemble(const std::vector<uint8_t>& parent, const std::vector<uint8_t>& delta,
                     std::vector<uint8_t>* out) {
    out->clear();
    size_t n = 0;
    const sslm_status query = sslm_prefix_assemble(parent.data(), parent.size(), delta.data(), delta.size(),
                                                   nullptr, &n);
    if (query != SSLM_BUFFER_TOO_SMALL) return query;
    std::vector<uint8_t> b(n + 1, 0x5a);
    size_t written = n;
    const sslm_status st = sslm_prefix_assemble(parent.data(), parent.size(), delta.data(), delta.size(),
                                                b.data(), &written);
    PKV_CHECK_EQ(b[n], 0x5a);
    if (st == SSLM_OK) {
        PKV_CHECK_EQ(written, n);
        b.resize(written);
        *out = std::move(b);
    }
    return st;
}

// The SPD1 of a nested prefix, built independently from its full SPX1 save (plan §3).
std::vector<uint8_t> ExpectedDelta(const Fixture& fx, const std::vector<uint8_t>& full) {
    const size_t hidden = ArtifactHiddenSize(fx), D = fx.geo.head_dim, runs = Runs(fx);
    if (full.size() < kHeader + hidden) return {};
    const int64_t L = static_cast<int64_t>(Le64(full, 112)), O = static_cast<int64_t>(Le64(full, 120));
    if (O < 1 || O > L || full.size() != kHeader + hidden + size_t(L) * runs * D) return {};
    std::vector<uint8_t> parent_rows;
    std::vector<uint8_t> own_rows;
    const uint8_t* rows = full.data() + kHeader + hidden;
    for (size_t r = 0; r < runs; ++r) {
        const uint8_t* run = rows + r * size_t(L) * D;
        parent_rows.insert(parent_rows.end(), run, run + size_t(O) * D);
        own_rows.insert(own_rows.end(), run + size_t(O) * D, run + size_t(L) * D);
    }
    std::vector<uint8_t> d(full.begin(), full.begin() + kHeader);
    std::memcpy(d.data(), "SPD1", 4);
    const auto digest = Sha32(parent_rows.data(), parent_rows.size());
    d.insert(d.end(), digest.begin(), digest.end());
    d.resize(kDeltaHeader, 0);
    PutLe32(d, 216, uint32_t(runs));
    PutLe32(d, 220, uint32_t(D));
    PutLe32(d, 224, uint32_t(hidden));
    PutLe32(d, 228, 0);
    d.insert(d.end(), full.begin() + kHeader, full.begin() + kHeader + hidden);
    d.insert(d.end(), own_rows.begin(), own_rows.end());
    Digest(d);
    return d;
}

std::vector<uint8_t> RowsSection(const Fixture& fx, const std::vector<uint8_t>& spx1) {
    const size_t start = kHeader + ArtifactHiddenSize(fx);
    return spx1.size() >= start ? std::vector<uint8_t>(spx1.begin() + start, spx1.end()) : std::vector<uint8_t>{};
}

// One scene: a root of `origin` tokens (seed 61) and its frozen child of `extension` tokens (seed 62), in a
// pool large enough for every restore a cell makes.
struct Family {
    Scene sc;
    PagePool* pool = nullptr;
    sslm_prefix parent = nullptr;
    sslm_prefix child = nullptr;
    std::vector<uint8_t> parent_full, child_full, delta;
    bool ok = false;
    Family(const Fixture& fx, int origin, int extension = 200, bool legacy_parent = false) {
        pool = &sc.AddPool(fx.model, uint32_t(6 * fx.CapPages()));
        if (pool->status != SSLM_OK) return;
        parent = Root(fx, &pool->pool, sc.h, origin, 61, legacy_parent);
        child = parent ? Child(fx, parent, sc.h, extension) : nullptr;
        if (!child) return;
        parent_full = Save(parent);
        child_full = Save(child);
        delta = SaveDelta(child);
        ok = !parent_full.empty() && !child_full.empty() && !delta.empty();
    }
};

constexpr int kOrigins[] = {992, 993, 1000, 1007};  // O mod 16 = 0, 1, 8, 15

// ---- D1: the bytes, and save_delta's refusals ---------------------------------------------------
void Format1() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    for (int origin : kOrigins) {
        Family f(fx, origin);
        if (!RequireBlob(f.delta, "D1")) return;
        const size_t hidden = ArtifactHiddenSize(fx);
        PKV_CHECK_EQ(f.delta.size(), kDeltaHeader + hidden + size_t(200) * fx.BytesPerToken());
        PKV_CHECK_MSG(IsMagic(f.delta, "SPD1"), "D1 O=%d: magic is not SPD1", origin);
        const auto expected = ExpectedDelta(fx, f.child_full);
        PKV_CHECK_MSG(!expected.empty() && f.delta == expected,
                      "D1 O=%d: save_delta differs from the delta built from the full save", origin);
        const auto parent_rows = RowsSection(fx, f.parent_full);
        const auto parent_digest = Sha32(parent_rows.data(), parent_rows.size());
        PKV_CHECK_MSG(f.delta.size() >= 216 &&
                      std::equal(parent_digest.begin(), parent_digest.end(), f.delta.begin() + 184),
                      "D1 O=%d: parent rows digest is not the SHA-256 of the parent's SPX1 rows", origin);
    }
    // Refusals: null, root, unfrozen, layer_index != 0. The buffer is untouched.
    Family f(fx, 1000);
    if (!f.ok) return;
    std::vector<uint8_t> buf(f.delta.size() + 8, 0xcc);
    auto untouched = [&](const char* what) {
        PKV_CHECK_MSG(std::all_of(buf.begin(), buf.end(), [](uint8_t b) { return b == 0xcc; }),
                      "D1 %s refusal wrote the buffer", what);
    };
    size_t n = buf.size();
    Status(sslm_prefix_save_delta(nullptr, buf.data(), &n), SSLM_INVALID_ARGUMENT, "D1 null prefix");
    Status(sslm_prefix_save_delta(f.child, buf.data(), nullptr), SSLM_INVALID_ARGUMENT, "D1 null n");
    n = buf.size();
    Status(sslm_prefix_save_delta(f.parent, buf.data(), &n), SSLM_INVALID_ARGUMENT, "D1 root prefix");
    untouched("root");
    sslm_prefix open = nullptr;
    Status(sslm_prefix_begin_from(f.parent, 8, &open), SSLM_OK, "D1 unfrozen child");
    if (open) {
        f.sc.h.prefixes.push_back(open);
        Status(PrefixPrefillAll(fx.model, open, Stream(7, 8, fx.vocab), 64), SSLM_OK, "D1 unfrozen prefill");
        n = buf.size();
        Status(sslm_prefix_save_delta(open, buf.data(), &n), SSLM_INVALID_ARGUMENT, "D1 unfrozen");
        untouched("unfrozen");
    }
    n = f.delta.size() - 1;
    Status(sslm_prefix_save_delta(f.child, buf.data(), &n), SSLM_BUFFER_TOO_SMALL, "D1 short buffer");
    PKV_CHECK_EQ(n, f.delta.size());
    untouched("short buffer");
}
PKV_CELL("D1/E6", "E6", Format1);

// ---- D2: restore onto the parent in its own pool (P1, P3) ----------------------------------------
void SamePool2() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    for (int origin : kOrigins) {
        Family f(fx, origin);
        if (!f.ok) return;
        sslm_prefix out = nullptr;
        uint32_t shared = UINT32_MAX;
        if (!Restored(fx, &f.pool->pool, f.delta, f.parent, f.sc.h, &out, &shared)) continue;
        const uint32_t s = uint32_t(origin / fx.B());
        PKV_CHECK_MSG(shared == s, "D2 O=%d: shared %u, expected %u", origin, shared, s);
        PKV_CHECK_MSG(Save(out) == f.child_full, "D2 O=%d: restored delta saves different SPX1 bytes", origin);
        const auto a = PrefixTable(f.parent), c = PrefixTable(out);
        PKV_CHECK_MSG(a.size() >= s && c.size() >= s && std::equal(a.begin(), a.begin() + s, c.begin()),
                      "D2 O=%d: the leading %u pages are not the parent's", origin, s);
        TwinTokens(fx, &f.pool->pool, f.child, out, 16);
    }
}
PKV_CELL("D2/E6", "E6", SamePool2);

// ---- D3: the parent in another pool of the same model is a row source, not a share ----------------
void OtherPool3() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    for (int origin : {993, 1007}) {
        Family f(fx, origin);
        if (!f.ok) return;
        PagePool other(fx.model, uint32_t(2 * fx.CapPages()));
        Handles oh;
        sslm_prefix twin = Root(fx, &other.pool, oh, origin, 61);
        if (!twin) return;
        const uint32_t free_other = FreePages(other.pool);
        sslm_prefix out = nullptr;
        uint32_t shared = UINT32_MAX;
        if (Restored(fx, &f.pool->pool, f.delta, twin, f.sc.h, &out, &shared)) {
            PKV_CHECK_EQ(shared, 0u);
            PKV_CHECK_MSG(Save(out) == f.child_full, "D3 O=%d: cross-pool restore saves different bytes", origin);
            TwinTokens(fx, &f.pool->pool, f.child, out, 16);
        }
        PKV_CHECK_EQ(FreePages(other.pool), free_other);
    }
}
PKV_CELL("D3/E6", "E6", OtherPool3);

// ---- D4: every kind of parent with the same rows ---------------------------------------------------
void ParentKinds4() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Family f(fx, 1000);
    if (!f.ok) return;
    auto check = [&](const char* kind, sslm_prefix parent, uint32_t want_shared) {
        sslm_prefix out = nullptr;
        uint32_t shared = UINT32_MAX;
        if (!Restored(fx, &f.pool->pool, f.delta, parent, f.sc.h, &out, &shared)) {
            PKV_CHECK_MSG(false, "D4 %s parent refused", kind);
            return;
        }
        PKV_CHECK_MSG(shared == want_shared, "D4 %s: shared %u, expected %u", kind, shared, want_shared);
        PKV_CHECK_MSG(Save(out) == f.child_full, "D4 %s: restored delta saves different bytes", kind);
    };
    sslm_prefix legacy = Root(fx, &f.pool->pool, f.sc.h, 1000, 61, true);
    if (legacy) check("whole_reserve", legacy, 62);
    sslm_prefix restored = nullptr;
    if (Restored(fx, &f.pool->pool, f.parent_full, nullptr, f.sc.h, &restored)) check("SPX1-restored", restored, 62);
    // A parent that is itself a restored delta: grandparent 500 -> parent 1000 -> child.
    Scene g;
    PagePool& gp = g.AddPool(fx.model, uint32_t(6 * fx.CapPages()));
    sslm_prefix grand = Root(fx, &gp.pool, g.h, 500, 61);
    sslm_prefix mid = grand ? Child(fx, grand, g.h, 500, 61) : nullptr;
    sslm_prefix leaf = mid ? Child(fx, mid, g.h, 200, 62) : nullptr;
    if (!leaf) return;
    const auto mid_delta = SaveDelta(mid), leaf_delta = SaveDelta(leaf), leaf_full = Save(leaf);
    sslm_prefix mid_restored = nullptr;
    if (!Restored(fx, &gp.pool, mid_delta, grand, g.h, &mid_restored)) return;
    sslm_prefix out = nullptr;
    uint32_t shared = UINT32_MAX;
    if (Restored(fx, &gp.pool, leaf_delta, mid_restored, g.h, &out, &shared)) {
        PKV_CHECK_EQ(shared, 62u);
        PKV_CHECK_MSG(Save(out) == leaf_full, "D4 delta-restored parent: leaf saves different bytes");
    }
}
PKV_CELL("D4/E6", "E6", ParentKinds4);

// ---- D5: parent refusals, and a refusal is clean (P4) ---------------------------------------------
void ParentRefusals5() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Family f(fx, 1000);
    if (!f.ok) return;
    auto refuse = [&](const char* what, sslm_prefix parent) {
        const uint32_t free_before = FreePages(f.pool->pool), shared_before = SharedPages(f.pool->pool);
        const auto parent_before = parent ? Save(parent) : std::vector<uint8_t>{};
        sslm_prefix out = nullptr;
        uint32_t shared = UINT32_MAX;
        Status(Restore(fx, &f.pool->pool, f.delta, parent, &out, &shared), PC_RESTORE_PARENT_MISMATCH, what);
        PKV_CHECK_MSG(out == nullptr, "D5 %s: *out not cleared", what);
        PKV_CHECK_MSG(shared == 0, "D5 %s: shared count %u", what, shared);
        PKV_CHECK_EQ(FreePages(f.pool->pool), free_before);
        PKV_CHECK_EQ(SharedPages(f.pool->pool), shared_before);
        if (parent) PKV_CHECK_MSG(Save(parent) == parent_before, "D5 %s: parent changed", what);
    };
    refuse("D5 null parent", nullptr);
    sslm_prefix shorter = Root(fx, &f.pool->pool, f.sc.h, 999, 61);
    if (shorter) refuse("D5 parent one token short", shorter);
    sslm_prefix longer = Root(fx, &f.pool->pool, f.sc.h, 1001, 61);
    if (longer) refuse("D5 parent one token long", longer);
    sslm_prefix other_rows = Root(fx, &f.pool->pool, f.sc.h, 1000, 99);
    if (other_rows) refuse("D5 parent with other tokens", other_rows);
    sslm_prefix open = nullptr;
    Status(sslm_prefix_begin_budgeted(fx.model, &f.pool->pool, 1000, &open), SSLM_OK, "D5 unfrozen parent");
    if (open) {
        f.sc.h.prefixes.push_back(open);
        Status(PrefixPrefillAll(fx.model, open, Stream(61, 1000, fx.vocab), 64), SSLM_OK, "D5 unfrozen prefill");
        sslm_prefix out = nullptr;
        Status(Restore(fx, &f.pool->pool, f.delta, open, &out), PC_RESTORE_PARENT_MISMATCH, "D5 unfrozen parent");
        PKV_CHECK(out == nullptr);
    }
    // One row byte changed in the parent: in a shared page, in the partial page, in the last layer's V.
    const size_t hidden = ArtifactHiddenSize(fx), D = fx.geo.head_dim, runs = Runs(fx);
    struct Edit { const char* what; size_t run; size_t pos; };
    const Edit edits[] = {{"shared-page row", 0, 5}, {"partial-page row", 3, 999}, {"last-layer V", runs - 1, 991}};
    for (const Edit& e : edits) {
        auto bytes = f.parent_full;
        const size_t off = kHeader + hidden + e.run * size_t(1000) * D + e.pos * D + (D - 1);
        if (off >= bytes.size()) { PKV_CHECK_MSG(false, "D5 %s offset out of range", e.what); continue; }
        bytes[off] ^= 1;
        Digest(bytes);
        sslm_prefix edited = nullptr;
        if (Restored(fx, &f.pool->pool, bytes, nullptr, f.sc.h, &edited)) refuse(e.what, edited);
    }
    // Another model (pkv_qk) of the same length: the parent belongs to another model.
    const Fixture& qk = GetFixture("pkv_qk");
    if (qk.ok) {
        PagePool qp(qk.model, uint32_t(2 * qk.CapPages()));
        Handles qh;
        sslm_prefix foreign = Root(qk, &qp.pool, qh, 1000, 61);
        if (foreign) refuse("D5 parent of another model", foreign);
    }
}
PKV_CELL("D5/E6", "E6", ParentRefusals5);

// ---- D6: the reader's order on SPD1 (§4.4) --------------------------------------------------------
void ReaderOrder6() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Family f(fx, 1000);
    if (!f.ok) return;
    struct Field { const char* what; size_t off; bool wide; uint64_t value; sslm_status want; };
    const Field fields[] = {
        {"backend", 4, false, 1, SSLM_INVALID_ARGUMENT},
        {"engine minor", 12, false, 999, SSLM_RESTORE_PRODUCER_MISMATCH},
        {"adapter vendor", 20, false, 0x10de, SSLM_RESTORE_PRODUCER_MISMATCH},
        {"kv_precision", 92, false, 77, SSLM_RESTORE_KV_MISMATCH},
        {"schema hash", 96, true, 0x1234567890abcdefull, SSLM_RESTORE_SCHEMA_MISMATCH},
        {"walk while unbound", 104, false, 3, SSLM_RESTORE_SCHEMA_MISMATCH},
        {"origin 0", 120, true, 0, SSLM_INVALID_ARGUMENT},
        {"origin past L", 120, true, 1201, SSLM_INVALID_ARGUMENT},
        {"length past cap", 112, true, 1u << 30, SSLM_INVALID_ARGUMENT},
        {"runs", 216, false, uint64_t(Runs(fx) + 1), SSLM_INVALID_ARGUMENT},
        {"head_dim", 220, false, uint64_t(fx.geo.head_dim + 1), SSLM_INVALID_ARGUMENT},
        {"hidden_size", 224, false, uint64_t(ArtifactHiddenSize(fx) + 1), SSLM_INVALID_ARGUMENT},
        {"reserved", 228, false, 1, SSLM_INVALID_ARGUMENT},
        {"token past vocab", 108, false, 1u << 30, SSLM_INVALID_ARGUMENT},
    };
    auto expect = [&](const char* what, const std::vector<uint8_t>& b, sslm_status want) {
        const uint32_t free_before = FreePages(f.pool->pool);
        sslm_prefix out = nullptr;
        Status(Restore(fx, &f.pool->pool, b, f.parent, &out), want, what);
        PKV_CHECK_MSG(out == nullptr, "D6 %s: *out not cleared", what);
        PKV_CHECK_EQ(FreePages(f.pool->pool), free_before);
    };
    for (const Field& x : fields) {
        auto b = f.delta;
        if (x.wide) PutLe64(b, x.off, x.value); else PutLe32(b, x.off, uint32_t(x.value));
        Digest(b);
        expect(x.what, b, x.want);
    }
    {
        auto b = f.delta;
        b[60] ^= 1;  // model hash
        Digest(b);
        expect("model hash", b, SSLM_RESTORE_MODEL_MISMATCH);
        PutLe32(b, 12, 999);  // producer is checked before the model
        Digest(b);
        expect("producer before model", b, SSLM_RESTORE_PRODUCER_MISMATCH);
    }
    {
        auto b = f.delta;
        b[200] ^= 1;  // the digest field itself is not touched: the stored digest no longer matches
        expect("digest", b, SSLM_INVALID_ARGUMENT);
        auto m = f.delta;
        std::memcpy(m.data(), "SPD2", 4);
        Digest(m);
        expect("magic", m, SSLM_INVALID_ARGUMENT);
        auto big = f.delta;
        big.push_back(0);
        Digest(big);
        expect("one byte long", big, SSLM_INVALID_ARGUMENT);
        auto small = f.delta;
        small.pop_back();
        Digest(small);
        expect("one byte short", small, SSLM_INVALID_ARGUMENT);
        std::vector<uint8_t> stub(f.delta.begin(), f.delta.begin() + kDeltaHeader - 1);
        expect("shorter than the header", stub, SSLM_INVALID_ARGUMENT);
    }
}
PKV_CELL("D6/E6", "E6", ReaderOrder6);

// ---- D7: assemble (P2) and its refusals -----------------------------------------------------------
void Assemble7() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    for (int origin : kOrigins) {
        Family f(fx, origin);
        if (!f.ok) return;
        std::vector<uint8_t> full;
        Status(Assemble(f.parent_full, f.delta, &full), SSLM_OK, "D7 assemble");
        PKV_CHECK_MSG(full == f.child_full, "D7 O=%d: assemble differs from the full save", origin);
    }
    Family f(fx, 1000);
    if (!f.ok) return;
    std::vector<uint8_t> out;
    size_t n = 0;
    Status(sslm_prefix_assemble(nullptr, 0, f.delta.data(), f.delta.size(), nullptr, &n), SSLM_INVALID_ARGUMENT,
           "D7 null parent");
    Status(sslm_prefix_assemble(f.parent_full.data(), f.parent_full.size(), nullptr, 0, nullptr, &n),
           SSLM_INVALID_ARGUMENT, "D7 null delta");
    Status(sslm_prefix_assemble(f.parent_full.data(), f.parent_full.size(), f.delta.data(), f.delta.size(), nullptr,
                                nullptr), SSLM_INVALID_ARGUMENT, "D7 null n");
    n = 0;
    Status(sslm_prefix_assemble(f.parent_full.data(), f.parent_full.size(), f.delta.data(), f.delta.size(), nullptr,
                                &n), SSLM_BUFFER_TOO_SMALL, "D7 size query");
    PKV_CHECK_EQ(n, f.child_full.size());
    auto bad_digest = f.parent_full;
    bad_digest[kHeader + 3] ^= 1;
    Status(Assemble(bad_digest, f.delta, &out), SSLM_INVALID_ARGUMENT, "D7 parent digest");
    auto bad_delta = f.delta;
    bad_delta[kDeltaHeader + 3] ^= 1;
    Status(Assemble(f.parent_full, bad_delta, &out), SSLM_INVALID_ARGUMENT, "D7 delta digest");
    Status(Assemble(f.delta, f.delta, &out), SSLM_INVALID_ARGUMENT, "D7 SPD1 parent");
    // Well formed, but not this delta's parent.
    const auto other_len = RootBlob(fx, 999);
    if (RequireBlob(other_len, "D7")) Status(Assemble(other_len, f.delta, &out), PC_RESTORE_PARENT_MISMATCH, "D7 length");
    struct Field { const char* what; size_t off; uint32_t value; };
    for (const Field& x : {Field{"backend", 4, 1}, Field{"engine patch", 16, 999}, Field{"adapter device", 24, 7},
                           Field{"kv_precision", 92, 77}}) {
        auto p = f.parent_full;
        PutLe32(p, x.off, x.value);
        Digest(p);
        Status(Assemble(p, f.delta, &out), PC_RESTORE_PARENT_MISMATCH, x.what);
    }
    auto model = f.parent_full;
    model[61] ^= 1;
    Digest(model);
    Status(Assemble(model, f.delta, &out), PC_RESTORE_PARENT_MISMATCH, "D7 model hash");
    auto rows = f.parent_full;
    rows[rows.size() - 1] ^= 1;
    Digest(rows);
    Status(Assemble(rows, f.delta, &out), PC_RESTORE_PARENT_MISMATCH, "D7 rows digest");
    // A short output buffer reports the size and writes nothing.
    std::vector<uint8_t> small(f.child_full.size() - 1, 0xcc);
    n = small.size();
    Status(sslm_prefix_assemble(f.parent_full.data(), f.parent_full.size(), f.delta.data(), f.delta.size(),
                                small.data(), &n), SSLM_BUFFER_TOO_SMALL, "D7 short buffer");
    PKV_CHECK_EQ(n, f.child_full.size());
    PKV_CHECK(std::all_of(small.begin(), small.end(), [](uint8_t b) { return b == 0xcc; }));
}
PKV_CELL("D7/E6", "E6", Assemble7);

// ---- D8: a chain of three --------------------------------------------------------------------------
void Chain8() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model, uint32_t(6 * fx.CapPages()));
    sslm_prefix root = Root(fx, &pool.pool, sc.h, 500, 61);
    sslm_prefix c1 = root ? Child(fx, root, sc.h, 333, 63) : nullptr;
    sslm_prefix c2 = c1 ? Child(fx, c1, sc.h, 171, 64) : nullptr;
    if (!c2) return;
    const auto r = Save(root), d1 = SaveDelta(c1), d2 = SaveDelta(c2), f2 = Save(c2);
    std::vector<uint8_t> a1, a2;
    Status(Assemble(r, d1, &a1), SSLM_OK, "D8 assemble root+d1");
    Status(Assemble(a1, d2, &a2), SSLM_OK, "D8 assemble (root+d1)+d2");
    PKV_CHECK_MSG(a2 == f2, "D8 assembled chain differs from the full save");
    Scene cold;
    PagePool& cp = cold.AddPool(fx.model, uint32_t(6 * fx.CapPages()));
    sslm_prefix rr = nullptr, r1 = nullptr, r2 = nullptr;
    if (!Restored(fx, &cp.pool, r, nullptr, cold.h, &rr)) return;
    if (!Restored(fx, &cp.pool, d1, rr, cold.h, &r1)) return;
    uint32_t shared = UINT32_MAX;
    if (!Restored(fx, &cp.pool, d2, r1, cold.h, &r2, &shared)) return;
    PKV_CHECK_EQ(shared, uint32_t(833 / fx.B()));
    PKV_CHECK_MSG(Save(r2) == f2, "D8 cold chain restore saves different bytes");
    // c2 lives in the build pool and r2 in the cold one: each decodes from its own pool.
    const auto built = Decode(fx, &pool.pool, c2, true, 16), cold_tokens = Decode(fx, &cp.pool, r2, true, 16);
    PKV_CHECK_MSG(built.size() == 16 && built == cold_tokens, "D8 cold chain decodes other tokens");
}
PKV_CELL("D8/E6", "E6", Chain8);

// ---- D9: exhaustion and the restore fault seams ---------------------------------------------------
void Exhaustion9() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Family f(fx, 1000);
    if (!f.ok) return;
    // Exactly ceil(1200/B) - floor(1000/B) = 75 - 62 = 13 pages are drawn: build a pool whose free count
    // after the parent is 12, then 13.
    for (uint32_t spare : {12u, 13u}) {
        Scene sc;
        const uint32_t parent_pages = uint32_t((1000 + fx.B() - 1) / fx.B());
        PagePool& pool = sc.AddPool(fx.model, parent_pages + spare);
        sslm_prefix parent = nullptr;
        if (!Restored(fx, &pool.pool, f.parent_full, nullptr, sc.h, &parent)) return;
        PKV_CHECK_EQ(FreePages(pool.pool), spare);
        sslm_prefix out = nullptr;
        const sslm_status st = Restore(fx, &pool.pool, f.delta, parent, &out);
        if (spare == 12) {
            Status(st, SSLM_KV_POOL_EXHAUSTED, "D9 one page short");
            PKV_CHECK(out == nullptr);
            PKV_CHECK_EQ(FreePages(pool.pool), 12u);
        } else {
            Status(st, SSLM_OK, "D9 exact pages");
            if (st == SSLM_OK && out) {
                sc.h.prefixes.push_back(out);
                PKV_CHECK_EQ(FreePages(pool.pool), 0u);
            }
        }
    }
    for (int position : {1, 2, 3}) {
        const uint32_t free_before = FreePages(f.pool->pool), shared_before = SharedPages(f.pool->pool);
        sslm_pc_test_only_restore_fault_arm(position, 0);
        sslm_prefix out = nullptr;
        const sslm_status st = Restore(fx, &f.pool->pool, f.delta, f.parent, &out);
        uint32_t free = 0, shared = 0, mapped = 0, reserve = 0;
        const int fired = sslm_pc_test_only_restore_fault_read(&free, &shared, &mapped, &reserve);
        sslm_pc_test_only_restore_fault_disarm();
        PKV_CHECK_MSG(fired == 1, "D9 seam %d was not reached on an SPD1 restore", position);
        PKV_CHECK_MSG(st != SSLM_OK && out == nullptr, "D9 seam %d: restore did not refuse cleanly (%d)", position,
                      int(st));
        PKV_CHECK_EQ(FreePages(f.pool->pool), free_before);
        PKV_CHECK_EQ(SharedPages(f.pool->pool), shared_before);
    }
}
PKV_CELL("D9/E6", "E6", Exhaustion9);

// ---- D10: other readers refuse SPD1 ---------------------------------------------------------------
void OtherReaders10() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Family f(fx, 1000);
    if (!f.ok) return;
    sslm_seq s = nullptr;
    Status(sslm_seq_restore(fx.model, &f.pool->pool, f.delta.data(), f.delta.size(), &s), SSLM_INVALID_ARGUMENT,
           "D10 seq_restore of SPD1");
    PKV_CHECK(s == nullptr);
    uint32_t shared = UINT32_MAX;
    Status(sslm_seq_restore_shared(fx.model, &f.pool->pool, f.delta.data(), f.delta.size(), f.parent, &s, &shared),
           SSLM_INVALID_ARGUMENT, "D10 seq_restore_shared of SPD1");
    PKV_CHECK(s == nullptr);
    // SPX1 is unchanged: the full save still restores with and without its hint (P5).
    sslm_prefix out = nullptr;
    if (Restored(fx, &f.pool->pool, f.child_full, nullptr, f.sc.h, &out))
        PKV_CHECK_MSG(Save(out) == f.child_full, "D10 SPX1 private restore changed");
}
PKV_CELL("D10/E6", "E6", OtherReaders10);

// ---- D11: concurrent restores onto one parent (the lazy parent digest) -----------------------------
void Concurrent11() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Family f(fx, 1000);
    if (!f.ok) return;
    sslm_prefix fresh = nullptr;  // a parent whose digest has never been computed
    if (!Restored(fx, &f.pool->pool, f.parent_full, nullptr, f.sc.h, &fresh)) return;
    constexpr int kThreads = 8;
    std::vector<sslm_prefix> outs(kThreads, nullptr);
    std::vector<sslm_status> st(kThreads, SSLM_INVALID_ARGUMENT);
    std::atomic<int> go{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i)
        threads.emplace_back([&, i] {
            while (go.load() == 0) {}
            st[i] = sslm_prefix_restore(fx.model, &f.pool->pool, f.delta.data(), f.delta.size(), fresh, &outs[i],
                                        nullptr);
        });
    go.store(1);
    for (auto& t : threads) t.join();
    for (int i = 0; i < kThreads; ++i) {
        Status(st[i], SSLM_OK, "D11 concurrent restore");
        if (outs[i]) {
            f.sc.h.prefixes.push_back(outs[i]);
            PKV_CHECK_MSG(Save(outs[i]) == f.child_full, "D11 thread %d restored different bytes", i);
        }
    }
}
PKV_CELL("D11/E6", "E6", Concurrent11);

// ---- D13: an empty child (L == origin) --------------------------------------------------------------
void Empty13() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Family f(fx, 1000, 0);
    if (!f.ok) return;
    PKV_CHECK_EQ(f.delta.size(), kDeltaHeader + ArtifactHiddenSize(fx));
    PKV_CHECK_MSG(f.delta == ExpectedDelta(fx, f.child_full), "D13 empty-child delta bytes");
    sslm_prefix out = nullptr;
    uint32_t shared = UINT32_MAX;
    if (Restored(fx, &f.pool->pool, f.delta, f.parent, f.sc.h, &out, &shared)) {
        PKV_CHECK_EQ(shared, uint32_t(1000 / fx.B()));
        PKV_CHECK_MSG(Save(out) == f.child_full, "D13 empty child restore");
    }
    std::vector<uint8_t> full;
    Status(Assemble(f.parent_full, f.delta, &full), SSLM_OK, "D13 assemble");
    PKV_CHECK_MSG(full == f.child_full, "D13 empty child assemble");
}
PKV_CELL("D13/E6", "E6", Empty13);

// ---- Rev 2 legs (coverage audit round 1) ------------------------------------------------------------

// A contiguous pool (B = cap), the only other page size a pool of one model can have.
struct ContiguousPool {
    AlignedBuf mem;
    sslm_kv_pool pool = nullptr;
    sslm_status status = SSLM_INVALID_ARGUMENT;
    ContiguousPool(sslm_model model, uint32_t blocks) {
        mem.Reset(sslm_kv_pool_contiguous_size(model, blocks));
        status = sslm_kv_pool_create_contiguous(model, mem.p, mem.n, blocks, &pool);
    }
    ~ContiguousPool() { if (pool) sslm_kv_pool_destroy(pool); }
};

// The digest of a prefix's rows [0, upto), from its SPX1 save, independently of the engine's digest code.
std::array<uint8_t, 32> PrefixRowsDigest(const Fixture& fx, sslm_prefix p, int64_t upto) {
    const auto full = Save(p);
    const size_t D = fx.geo.head_dim, hidden = ArtifactHiddenSize(fx);
    const int64_t L = full.size() >= kHeader ? static_cast<int64_t>(Le64(full, 112)) : 0;
    std::vector<uint8_t> rows;
    for (size_t r = 0; r < Runs(fx) && upto <= L; ++r) {
        const uint8_t* run = full.data() + kHeader + hidden + r * size_t(L) * D;
        rows.insert(rows.end(), run, run + size_t(upto) * D);
    }
    return Sha32(rows.data(), rows.size());
}

// D3b: across page sizes, both directions: paged parent -> contiguous child, contiguous parent -> paged child.
void ContiguousPools3b() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Family f(fx, 1007, 300);
    if (!f.ok) return;
    ContiguousPool cp(fx.model, 6);
    Status(cp.status, SSLM_OK, "D3b contiguous pool");
    if (cp.status != SSLM_OK) return;
    Handles ch;
    sslm_prefix out = nullptr;
    uint32_t shared = UINT32_MAX;
    if (Restored(fx, &cp.pool, f.delta, f.parent, ch, &out, &shared)) {  // paged parent, contiguous child
        PKV_CHECK_EQ(shared, 0u);
        PKV_CHECK_MSG(Save(out) == f.child_full, "D3b paged parent into a contiguous pool: bytes differ");
    }
    sslm_prefix cparent = nullptr;  // contiguous parent, paged child
    if (Restored(fx, &cp.pool, f.parent_full, nullptr, ch, &cparent)) {
        sslm_prefix back = nullptr;
        shared = UINT32_MAX;
        if (Restored(fx, &f.pool->pool, f.delta, cparent, f.sc.h, &back, &shared)) {
            PKV_CHECK_EQ(shared, 0u);
            PKV_CHECK_MSG(Save(back) == f.child_full, "D3b contiguous parent into a paged pool: bytes differ");
        }
        // And the same contiguous pool for both: one page each, nothing shared.
        sslm_prefix both = nullptr;
        shared = UINT32_MAX;
        if (Restored(fx, &cp.pool, f.delta, cparent, ch, &both, &shared)) {
            PKV_CHECK_EQ(shared, 0u);
            PKV_CHECK_MSG(Save(both) == f.child_full, "D3b contiguous parent and child: bytes differ");
        }
    }
}
PKV_CELL("D3b/E6", "E6", ContiguousPools3b);

// D5b: each parent check on its own (rows [0, O) still hash to the digest), and the two order legs.
void ParentChecksAlone5b() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Family f(fx, 1000);
    if (!f.ok) return;
    std::array<uint8_t, 32> want{};
    std::copy(f.delta.begin() + 184, f.delta.begin() + 216, want.begin());
    auto refused = [&](const char* what, sslm_prefix parent) {
        const uint32_t free_before = FreePages(f.pool->pool);
        sslm_prefix out = nullptr;
        Status(Restore(fx, &f.pool->pool, f.delta, parent, &out), PC_RESTORE_PARENT_MISMATCH, what);
        PKV_CHECK_MSG(out == nullptr, "%s: *out not cleared", what);
        PKV_CHECK_EQ(FreePages(f.pool->pool), free_before);
    };
    // O + 1: the same stream plus one token; its rows [0, O) are the parent's.
    sslm_prefix longer = Root(fx, &f.pool->pool, f.sc.h, 1001, 61);
    if (longer) {
        PKV_CHECK_MSG(PrefixRowsDigest(fx, longer, 1000) == want, "D5b precondition: the O+1 parent's rows differ");
        refused("D5b length O+1 alone", longer);
    }
    // Unfrozen: the same stream, not frozen (its rows are the parent's; it cannot be saved, so this is by stream).
    sslm_prefix open = nullptr;
    Status(sslm_prefix_begin_budgeted(fx.model, &f.pool->pool, 1000, &open), SSLM_OK, "D5b unfrozen begin");
    if (open) {
        f.sc.h.prefixes.push_back(open);
        Status(PrefixPrefillAll(fx.model, open, Stream(61, 1000, fx.vocab), 64), SSLM_OK, "D5b unfrozen prefill");
        refused("D5b unfrozen alone", open);
    }
    // Another model handle: a second mapping of the same pkv_def bytes, with the same rows.
    sslm_model second = nullptr;
    AlignedBuf copy(fx.size);
    std::memcpy(copy.p, fx.bytes.p, fx.size);
    Status(sslm_model_map(copy.p, fx.size, &second), SSLM_OK, "D5b second mapping");
    if (second) {
        if (fx.pinned > 0) Status(KvPagesTestPinPagePositions(second, fx.pinned), SSLM_OK, "D5b pin second mapping");
        Fixture twin;  // a non-owning view for the helpers; released below
        twin.model = second;
        twin.geo = fx.geo;
        twin.vocab = fx.vocab;
        twin.pinned = fx.pinned;
        twin.ok = true;
        {
            PagePool tp(second, uint32_t(2 * fx.CapPages()));
            Handles th;
            sslm_prefix other = Root(twin, &tp.pool, th, 1000, 61);
            if (other) {
                PKV_CHECK_MSG(Save(other).size() == f.parent_full.size() &&
                              PrefixRowsDigest(fx, other, 1000) == want,
                              "D5b precondition: the second mapping's rows differ");
                refused("D5b another model handle with the same rows", other);
            }
        }
        twin.model = nullptr;
        sslm_model_unmap(second);
    }
    // Order: a reader refusal outranks a parent refusal; a parent refusal outranks exhaustion.
    {
        auto bad = f.delta;
        bad[60] ^= 1;
        Digest(bad);
        sslm_prefix out = nullptr;
        Status(Restore(fx, &f.pool->pool, bad, nullptr, &out), SSLM_RESTORE_MODEL_MISMATCH,
               "D5b bad model hash with a null parent");
    }
    {
        Scene sc;
        const uint32_t parent_pages = uint32_t((1000 + fx.B() - 1) / fx.B());
        PagePool& pool = sc.AddPool(fx.model, parent_pages + 2 * parent_pages);
        sslm_prefix right = nullptr, wrong = nullptr;
        if (Restored(fx, &pool.pool, f.parent_full, nullptr, sc.h, &right) &&
            (wrong = Root(fx, &pool.pool, sc.h, 1000, 99)) != nullptr) {
            // Leave too few pages for the child: draw them away with a legacy-free budget prefix.
            sslm_prefix filler = nullptr;
            const uint32_t free_now = FreePages(pool.pool);
            if (free_now > 1) {
                Status(sslm_prefix_begin_budgeted(fx.model, &pool.pool, int32_t((free_now - 2) * fx.B()), &filler),
                       SSLM_OK, "D5b filler");
                if (filler) sc.h.prefixes.push_back(filler);
            }
            sslm_prefix out = nullptr;
            Status(Restore(fx, &pool.pool, f.delta, wrong, &out), PC_RESTORE_PARENT_MISMATCH,
                   "D5b wrong parent with too few pages");
            Status(Restore(fx, &pool.pool, f.delta, right, &out), SSLM_KV_POOL_EXHAUSTED,
                   "D5b right parent with too few pages");
        }
    }
}
PKV_CELL("D5b/E6", "E6", ParentChecksAlone5b);

// D7b: assemble keeps the delta's own header (forged version, backend 1, distinct counts) and refuses by number.
void AssembleFields7b() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Family f(fx, 1000);
    if (!f.ok) return;
    std::vector<uint8_t> out;
    {   // engine 1.17.0 on both, resealed
        auto p = f.parent_full, d = f.delta, c = f.child_full;
        for (auto* b : {&p, &d, &c}) { PutLe32(*b, 12, 17); PutLe32(*b, 16, 0); Digest(*b); }
        Status(Assemble(p, d, &out), SSLM_OK, "D7b forged 1.17.0");
        PKV_CHECK_MSG(out == c, "D7b assemble did not keep the delta's engine version");
    }
    {   // backend 1 on both
        auto p = f.parent_full, d = f.delta, c = f.child_full;
        for (auto* b : {&p, &d, &c}) { PutLe32(*b, 4, 1); Digest(*b); }
        Status(Assemble(p, d, &out), SSLM_OK, "D7b forged GPU backend");
        PKV_CHECK_MSG(out == c, "D7b assemble did not keep backend 1");
    }
    {   // distinct state fields in the delta: through assemble, and through restore -> save
        auto d = f.delta, c = f.child_full;
        const uint64_t counts[5] = {11, 13, 17, 19, 23};
        for (auto* b : {&d, &c}) {
            for (int i = 0; i < 5; ++i) PutLe64(*b, 144 + 8 * i, counts[i]);
            PutLe64(*b, 128, 12345);
            PutLe64(*b, 136, uint64_t(-7));
            PutLe32(*b, 108, 3);
            Digest(*b);
        }
        Status(Assemble(f.parent_full, d, &out), SSLM_OK, "D7b distinct fields");
        PKV_CHECK_MSG(out == c, "D7b assemble did not keep the delta's state fields");
        sslm_prefix r = nullptr;
        if (Restored(fx, &f.pool->pool, d, f.parent, f.sc.h, &r))
            PKV_CHECK_MSG(Save(r) == c, "D7b restore did not keep the delta's state fields");
    }
    struct Leg { const char* what; size_t off; bool wide; uint64_t value; sslm_status want; };
    for (const Leg& x : {Leg{"reserved", 228, false, 1, SSLM_INVALID_ARGUMENT},
                         Leg{"origin 0", 120, true, 0, SSLM_INVALID_ARGUMENT},
                         Leg{"origin past L", 120, true, 1201, SSLM_INVALID_ARGUMENT},
                         Leg{"runs 2^16", 216, false, 65536, SSLM_INVALID_ARGUMENT},
                         Leg{"length 2^40", 112, true, uint64_t(1) << 40, SSLM_INVALID_ARGUMENT}}) {
        auto d = f.delta;
        if (x.wide) PutLe64(d, x.off, x.value); else PutLe32(d, x.off, uint32_t(x.value));
        if (std::strcmp(x.what, "runs 2^16") == 0) PutLe32(d, 220, 65536);
        Digest(d);
        Status(Assemble(f.parent_full, d, &out), x.want, x.what);
    }
    {   // a parent whose size is not its length's under the delta's geometry
        auto p = f.parent_full;
        p.push_back(0);
        Digest(p);
        Status(Assemble(p, f.delta, &out), SSLM_INVALID_ARGUMENT, "D7b parent of another size");
        auto m = f.parent_full;
        std::memcpy(m.data(), "SPX9", 4);
        Digest(m);
        Status(Assemble(m, f.delta, &out), SSLM_INVALID_ARGUMENT, "D7b parent of another magic");
    }
    {   // a size query on a mismatched pair reports the mismatch
        const auto other = RootBlob(fx, 999);
        size_t n = 0;
        if (RequireBlob(other, "D7b"))
            Status(sslm_prefix_assemble(other.data(), other.size(), f.delta.data(), f.delta.size(), nullptr, &n),
                   PC_RESTORE_PARENT_MISMATCH, "D7b size query on a mismatched pair");
    }
}
PKV_CELL("D7b/E6", "E6", AssembleFields7b);

// D11b: the digest cache is only the parent use: save_delta then restore onto the same handle, the reverse, and
// both at once.
void CacheUses11b() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    for (bool save_first : {true, false}) {
        Scene sc;
        PagePool& pool = sc.AddPool(fx.model, uint32_t(6 * fx.CapPages()));
        sslm_prefix root = Root(fx, &pool.pool, sc.h, 500, 61);
        sslm_prefix c1 = root ? Child(fx, root, sc.h, 500, 63) : nullptr;
        sslm_prefix c2 = c1 ? Child(fx, c1, sc.h, 200, 64) : nullptr;
        if (!c2) return;
        const auto d2 = SaveDelta(c2), f2 = Save(c2);
        std::vector<uint8_t> d1;
        if (save_first) d1 = SaveDelta(c1);
        sslm_prefix out = nullptr;
        if (Restored(fx, &pool.pool, d2, c1, sc.h, &out))
            PKV_CHECK_MSG(Save(out) == f2, "D11b restore onto c1 (save %s)", save_first ? "first" : "after");
        if (!save_first) d1 = SaveDelta(c1);
        PKV_CHECK_MSG(d1 == ExpectedDelta(fx, Save(c1)), "D11b c1's delta changed by the cache (save %s)",
                      save_first ? "first" : "after");
    }
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model, uint32_t(8 * fx.CapPages()));
    sslm_prefix root = Root(fx, &pool.pool, sc.h, 500, 61);
    sslm_prefix c1 = root ? Child(fx, root, sc.h, 500, 63) : nullptr;
    sslm_prefix c2 = c1 ? Child(fx, c1, sc.h, 200, 64) : nullptr;
    if (!c2) return;
    const auto d2 = SaveDelta(c2), f2 = Save(c2), want1 = ExpectedDelta(fx, Save(c1));
    std::vector<std::vector<uint8_t>> saved(4);
    std::vector<sslm_prefix> outs(4, nullptr);
    std::vector<sslm_status> st(4, SSLM_INVALID_ARGUMENT);
    std::atomic<int> go{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&, i] {
            while (go.load() == 0) {}
            size_t n = want1.size();
            saved[i].assign(n, 0);
            if (sslm_prefix_save_delta(c1, saved[i].data(), &n) != SSLM_OK) saved[i].clear();
        });
        threads.emplace_back([&, i] {
            while (go.load() == 0) {}
            st[i] = sslm_prefix_restore(fx.model, &pool.pool, d2.data(), d2.size(), c1, &outs[i], nullptr);
        });
    }
    go.store(1);
    for (auto& t : threads) t.join();
    for (int i = 0; i < 4; ++i) {
        PKV_CHECK_MSG(saved[i] == want1, "D11b concurrent save_delta %d differs", i);
        Status(st[i], SSLM_OK, "D11b concurrent restore");
        if (outs[i]) {
            sc.h.prefixes.push_back(outs[i]);
            PKV_CHECK_MSG(Save(outs[i]) == f2, "D11b concurrent restore %d differs", i);
        }
    }
}
PKV_CELL("D11b/E6", "E6", CacheUses11b);

// D16: the parent may be released after the restore.
void ParentRelease16() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    PagePool pool(fx.model, uint32_t(3 * fx.CapPages()));
    const uint32_t start = FreePages(pool.pool);
    std::vector<uint8_t> parent_full, delta, child_full;
    {
        Handles h;
        sslm_prefix parent = Root(fx, &pool.pool, h, 1000, 61);
        sslm_prefix child = parent ? Child(fx, parent, h, 200) : nullptr;
        if (!child) return;
        parent_full = Save(parent);
        delta = SaveDelta(child);
        child_full = Save(child);
    }
    PKV_CHECK_EQ(FreePages(pool.pool), start);
    sslm_prefix parent = nullptr, out = nullptr;
    Handles h;
    if (!Restored(fx, &pool.pool, parent_full, nullptr, h, &parent)) return;
    if (Restore(fx, &pool.pool, delta, parent, &out) != SSLM_OK || !out) {
        PKV_CHECK_MSG(false, "D16 restore refused");
        return;
    }
    const auto before = Decode(fx, &pool.pool, out, true, 16);
    h.prefixes.clear();
    Status(sslm_prefix_release(parent), SSLM_OK, "D16 release parent");
    PKV_CHECK_MSG(Save(out) == child_full, "D16 child changed after its parent's release");
    PKV_CHECK_MSG(Decode(fx, &pool.pool, out, true, 16) == before, "D16 decode changed after the parent's release");
    Status(sslm_prefix_release(out), SSLM_OK, "D16 release child");
    PKV_CHECK_EQ(FreePages(pool.pool), start);
}
PKV_CELL("D16/E6", "E6", ParentRelease16);

}  // namespace
