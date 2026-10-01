#include "precache_common.h"

namespace {
using namespace pc;

void Shape41() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    for (int L : {0, 1, 15, 16, 17, 1000, 4095, 4096}) {
        Scene sc;
        PagePool& pool = sc.AddPool(fx.model, 600);
        sslm_prefix twin = Root(fx, &pool.pool, sc.h, L);
        if (!twin) continue;
        auto b = Save(twin);
        if (!RequireBlob(b, "4.1")) return;
        Header(b, L, fx);
        sslm_prefix copy = nullptr;
        if (Restored(fx, &pool.pool, b, nullptr, sc.h, &copy)) TwinTokens(fx, &pool.pool, twin, copy, 16);
    }
}
PKV_CELL("4.1/E2", "E1", Shape41);

void RoundTrip64() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    for (int L : {0, 17, 65, 1000}) {
        Scene sc;
        PagePool& pool = sc.AddPool(fx.model, 300);
        sslm_prefix p = Root(fx, &pool.pool, sc.h, L);
        if (!p) continue;
        const auto first = Save(p);
        if (!RequireBlob(first, "6.4")) return;
        sslm_prefix copy = nullptr;
        if (!Restored(fx, &pool.pool, first, nullptr, sc.h, &copy)) continue;
        const auto second = Save(copy);
        PKV_CHECK_MSG(first == second, "6.4 SPX1 round-trip blob differs");
        if (L) PKV_CHECK_EQ(Le32(first, 108), Le32(second, 108));
    }
}
PKV_CELL("6.4/E2", "E1", RoundTrip64);

void Tokens62() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model, 400);
    sslm_prefix p = Root(fx, &pool.pool, sc.h, 65);
    if (!p) return;
    const auto b = Save(p);
    if (!RequireBlob(b, "6.2")) return;
    sslm_prefix copy = nullptr;
    if (Restored(fx, &pool.pool, b, nullptr, sc.h, &copy)) TwinTokens(fx, &pool.pool, p, copy);
}
PKV_CELL("6.2/E2", "E1", Tokens62);
PKV_CELL("7.1/E2", "E1", Tokens62);
PKV_CELL("8.1/E2", "E1", Tokens62);

void Save24() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model, 64);
    sslm_prefix p = Root(fx, &pool.pool, sc.h, 17);
    if (!p) return;
    size_t n = 0;
    Status(sslm_prefix_save(nullptr, nullptr, &n), SSLM_INVALID_ARGUMENT, "2.4 null prefix");
    Status(sslm_prefix_save(p, nullptr, nullptr), SSLM_INVALID_ARGUMENT, "2.4 null size");
    n = 0;
    Status(sslm_prefix_save(p, nullptr, &n), SSLM_BUFFER_TOO_SMALL, "2.4 size query");
    if (!n) return;
    std::vector<uint8_t> out(n + 1, 0xa5);
    size_t short_n = n - 1;
    Status(sslm_prefix_save(p, out.data(), &short_n), SSLM_BUFFER_TOO_SMALL, "2.4 short buffer");
    PKV_CHECK_EQ(short_n, n);
    PKV_CHECK_MSG(std::all_of(out.begin(), out.end(), [](uint8_t v){return v == 0xa5;}),
        "2.4 short save wrote bytes");
    size_t exact = n;
    Status(sslm_prefix_save(p, out.data(), &exact), SSLM_OK, "2.4 exact buffer");
    PKV_CHECK_EQ(out[n], 0xa5);
}
PKV_CELL("2.4/E2", "E1", Save24);

void Restore25() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    const auto b = RootBlob(fx, 17);
    if (!RequireBlob(b, "2.5")) return;
    PagePool pool(fx.model, 32);
    sslm_prefix out = OutSentinel<sslm_prefix>();
    Status(sslm_prefix_restore(fx.model, &pool.pool, b.data(), b.size(), nullptr, nullptr, nullptr),
           SSLM_INVALID_ARGUMENT, "2.5 null out");
    for (int which = 0; which < 4; ++which) {
        out = OutSentinel<sslm_prefix>();
        sslm_kv_pool null_pool = nullptr;
        const sslm_status st = sslm_prefix_restore(which == 0 ? nullptr : fx.model,
            which == 1 ? nullptr : which == 2 ? &null_pool : &pool.pool,
            which == 3 ? nullptr : b.data(), b.size(), nullptr, &out, nullptr);
        Status(st, SSLM_INVALID_ARGUMENT, "2.5 null input");
        PKV_CHECK(out == nullptr);
    }
}
PKV_CELL("2.5/E2", "E1", Restore25);

void ByteCorruption22() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    const auto good = RootBlob(fx, 1);
    if (!RequireBlob(good, "2.2")) return;
    PagePool pool(fx.model, 32);
    for (size_t i = 0; i < good.size(); ++i) {
        auto b = good;
        b[i] ^= 1;
        sslm_prefix out = nullptr;
        const sslm_status st = Restore(fx, &pool.pool, b, nullptr, &out);
        PKV_CHECK_MSG(st == SSLM_INVALID_ARGUMENT && out == nullptr,
            "2.2 flipped byte %zu: expected integrity refusal, got %d", i, int(st));
        if (st == SSLM_OK && out && out != OutSentinel<sslm_prefix>()) sslm_prefix_release(out);
    }
}
PKV_CELL("2.2/E2", "E1", ByteCorruption22);
PKV_CELL("7.3/E2", "E1", ByteCorruption22);

void Truncation21() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    const auto good = RootBlob(fx, 1);
    if (!RequireBlob(good, "2.1")) return;
    PagePool pool(fx.model, 32);
    for (size_t n = 0; n < good.size(); ++n) {
        sslm_prefix out = OutSentinel<sslm_prefix>();
        const sslm_status st = sslm_prefix_restore(fx.model, &pool.pool, good.data(), n,
                                                   nullptr, &out, nullptr);
        PKV_CHECK_MSG(st == SSLM_INVALID_ARGUMENT && out == nullptr,
            "2.1 truncation at %zu: got %d", n, int(st));
        if (st == SSLM_OK && out && out != OutSentinel<sslm_prefix>()) sslm_prefix_release(out);
    }
    std::vector<uint8_t> extra = good;
    extra.push_back(0);
    sslm_prefix out = nullptr;
    Status(Restore(fx, &pool.pool, extra, nullptr, &out), SSLM_INVALID_ARGUMENT, "2.1 extra byte");
    PKV_CHECK(out == nullptr);
}
PKV_CELL("2.1/E2", "E1", Truncation21);

void Domains23() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    const auto good = RootBlob(fx, 1);
    if (!RequireBlob(good, "2.3")) return;
    PagePool pool(fx.model, 64);
    struct Edit { const char* name; size_t offset; uint64_t value; bool wide; sslm_status want; };
    const std::vector<Edit> edits = {
        {"backend",4,1,false,SSLM_INVALID_ARGUMENT},
        {"producer major",8,0xffffffffu,false,PC_RESTORE_PRODUCER_MISMATCH},
        {"producer minor",12,0xffffffffu,false,PC_RESTORE_PRODUCER_MISMATCH},
        {"producer patch",16,0xffffffffu,false,PC_RESTORE_PRODUCER_MISMATCH},
        {"adapter vendor",20,1,false,PC_RESTORE_PRODUCER_MISMATCH},
        {"adapter device",24,1,false,PC_RESTORE_PRODUCER_MISMATCH},
        {"model hash",60,uint64_t(good[60]^1),false,SSLM_RESTORE_MODEL_MISMATCH},
        {"precision",92,0xffffffffu,false,SSLM_RESTORE_KV_MISMATCH},
        {"origin > L",120,2,true,SSLM_INVALID_ARGUMENT},
        {"origin negative",120,uint64_t(-1),true,SSLM_INVALID_ARGUMENT},
        {"L negative",112,uint64_t(-1),true,SSLM_INVALID_ARGUMENT},
        {"L above cap",112,4097,true,SSLM_INVALID_ARGUMENT},
        {"token negative",108,0xfffffffeu,false,SSLM_INVALID_ARGUMENT},
        {"token INT32_MIN",108,0x80000000u,false,SSLM_INVALID_ARGUMENT},
        {"token above vocab",108,uint64_t(fx.vocab),false,SSLM_INVALID_ARGUMENT},
        {"mantissa above int32",128,uint64_t(INT32_MAX)+1,true,SSLM_INVALID_ARGUMENT},
        {"mantissa below int32",128,uint64_t(INT64_MAX),true,SSLM_INVALID_ARGUMENT},
    };
    for (const auto& e : edits) {
        auto b = good;
        if (e.wide) PutLe64(b,e.offset,e.value); else PutLe32(b,e.offset,uint32_t(e.value));
        Digest(b);
        sslm_prefix out = nullptr;
        const sslm_status st = Restore(fx,&pool.pool,b,nullptr,&out);
        PKV_CHECK_MSG(st == e.want && out == nullptr,
            "2.3 %s: expected %d, got %d; out=%p",e.name,int(e.want),int(st),out);
        if (st == SSLM_OK && out && out != OutSentinel<sslm_prefix>()) sslm_prefix_release(out);
    }
}
PKV_CELL("2.3/E2", "E1", Domains23);
PKV_CELL("7.6/E2", "E1", Domains23);
PKV_CELL("7.11/E2", "E1", Domains23);

void RootHint92() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model, 128);
    sslm_prefix original = Root(fx,&pool.pool,sc.h,65);
    sslm_prefix hint = Root(fx,&pool.pool,sc.h,32,82);
    if (!original || !hint) return;
    const auto b = Save(original);
    if (!RequireBlob(b,"9.2")) return;
    sslm_prefix out = nullptr;
    uint32_t shared = UINT32_MAX;
    if (Restored(fx,&pool.pool,b,hint,sc.h,&out,&shared)) {
        PKV_CHECK_EQ(shared,0u);
        TwinTokens(fx,&pool.pool,original,out,16);
    }
}
PKV_CELL("9.2/E2", "E1", RootHint92);

void NoModeBudget712() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model, uint32_t(fx.CapPages()+128));
    sslm_prefix legacy = Root(fx,&pool.pool,sc.h,65,41,true);
    sslm_prefix budget = Root(fx,&pool.pool,sc.h,65,41,false);
    if (!legacy || !budget) return;
    const auto a = Save(legacy), b = Save(budget);
    if (!RequireBlob(a,"7.12")) return;
    PKV_CHECK_MSG(a == b,"7.12 SPX1 carries mode or budget: identical state differs by creation mode");
}
PKV_CELL("7.12/E2:mode-budget", "E1", NoModeBudget712);
}
