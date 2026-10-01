#include "precache_common.h"

namespace {
using namespace pc;

sslm_prefix Child(const Fixture& fx, sslm_prefix parent, Handles& h,
                  int extension = 200, int seed = 62) {
    sslm_prefix child = nullptr;
    Status(sslm_prefix_begin_from(parent, std::max(extension, 1), &child), SSLM_OK,
           "begin_from fixture");
    if (!child) return nullptr;
    h.prefixes.push_back(child);
    if (extension) Status(PrefixPrefillAll(fx.model, child,
        Stream(seed, extension, fx.vocab), 64), SSLM_OK, "child prefill fixture");
    Status(sslm_prefix_freeze(child), SSLM_OK, "child freeze fixture");
    return child;
}

void Nested63() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model, 400);
    sslm_prefix world = Root(fx,&pool.pool,sc.h,1000,61);
    sslm_prefix built = world ? Child(fx,world,sc.h) : nullptr;
    if (!built) return;
    const auto b = Save(built);
    if (!RequireBlob(b,"6.3")) return;
    PKV_CHECK_EQ(static_cast<int64_t>(Le64(b,120)),1000);
    sslm_prefix shared = nullptr, priv = nullptr;
    uint32_t shared_count = UINT32_MAX;
    const bool a = Restored(fx,&pool.pool,b,world,sc.h,&shared,&shared_count);
    const bool c = Restored(fx,&pool.pool,b,nullptr,sc.h,&priv);
    if (a && c) {
        PKV_CHECK_EQ(shared_count,62u);
        TwinTokens(fx,&pool.pool,built,shared,16);
        TwinTokens(fx,&pool.pool,built,priv,16);
        PKV_CHECK_MSG(Save(shared)==Save(priv),"6.3 shared and private SPX1 bytes differ");
    }
}
PKV_CELL("6.3/E3", "E1", Nested63);
PKV_CELL("9.3/E3", "E1", Nested63);
PKV_CELL("7.5/E3", "E1", Nested63);

void Origins42() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    for (int origin : {0,1,15,16,31,32,65}) {
        Scene sc;
        PagePool& pool = sc.AddPool(fx.model,128);
        sslm_prefix parent = Root(fx,&pool.pool,sc.h,origin,41);
        sslm_prefix child = parent ? Child(fx,parent,sc.h,origin==65 ? 0 : 16) : nullptr;
        if (!child) continue;
        const auto b = Save(child);
        if (!RequireBlob(b,"4.2")) return;
        PKV_CHECK_EQ(static_cast<int64_t>(Le64(b,120)),origin);
        sslm_prefix restored = nullptr;
        uint32_t shared = UINT32_MAX;
        if (Restored(fx,&pool.pool,b,parent,sc.h,&restored,&shared)) {
            PKV_CHECK_EQ(shared,uint32_t(origin/fx.B()));
            TwinTokens(fx,&pool.pool,child,restored,16);
        }
    }
}
PKV_CELL("4.2/E3", "E1", Origins42);

void HintFailures26() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model,400);
    sslm_prefix world = Root(fx,&pool.pool,sc.h,1000,61);
    sslm_prefix child = world ? Child(fx,world,sc.h) : nullptr;
    if (!child) return;
    const auto b = Save(child);
    if (!RequireBlob(b,"2.6")) return;
    sslm_prefix wrong_length = Root(fx,&pool.pool,sc.h,992,61);
    sslm_prefix wrong_rows = Root(fx,&pool.pool,sc.h,1000,63);
    sslm_prefix unfrozen = nullptr;
    Status(sslm_prefix_begin_budgeted(fx.model,&pool.pool,1000,&unfrozen),SSLM_OK,
           "unfrozen hint fixture");
    if (unfrozen) sc.h.prefixes.push_back(unfrozen);
    PagePool foreign_pool(fx.model,80);
    Handles foreign_h;
    sslm_prefix foreign = Root(fx,&foreign_pool.pool,foreign_h,1000,61);
    for (sslm_prefix hint : {unfrozen,wrong_length,wrong_rows,foreign}) {
        if (!hint) continue;
        sslm_prefix out = nullptr;
        uint32_t shared = UINT32_MAX;
        if (Restored(fx,&pool.pool,b,hint,sc.h,&out,&shared)) {
            PKV_CHECK_EQ(shared,0u);
            TwinTokens(fx,&pool.pool,child,out,16);
        }
    }
}
PKV_CELL("2.6/E3", "E1", HintFailures26);

void Admission510() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model,77); // 63 parent pages + R(200)=14.
    sslm_prefix world = Root(fx,&pool.pool,sc.h,1000,61);
    sslm_prefix child = world ? Child(fx,world,sc.h) : nullptr;
    if (!child) return;
    const auto b = Save(child);
    if (!RequireBlob(b,"5.10")) return;
    Status(ReleasePrefix(sc.h,child),SSLM_OK,"release built child");
    for (sslm_prefix hint : {sslm_prefix(nullptr),world}) {
        sslm_prefix out = nullptr;
        const sslm_status st = Restore(fx,&pool.pool,b,hint,&out);
        Status(st,hint ? SSLM_OK : SSLM_KV_POOL_EXHAUSTED,
               hint ? "5.10 matching hint" : "5.10 private admission");
        if (!hint) PKV_CHECK(out == nullptr);
        if (st == SSLM_OK && out && out != OutSentinel<sslm_prefix>()) sc.h.prefixes.push_back(out);
    }
}
PKV_CELL("5.10/E3", "E1", Admission510);

void ParentRelease33() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model,200);
    sslm_prefix world = Root(fx,&pool.pool,sc.h,1000,61);
    sslm_prefix child = world ? Child(fx,world,sc.h) : nullptr;
    if (!child) return;
    const auto b = Save(child);
    if (!RequireBlob(b,"3.3")) return;
    sslm_prefix copy = nullptr;
    if (!Restored(fx,&pool.pool,b,world,sc.h,&copy)) return;
    std::thread release([&]{ Status(ReleasePrefix(sc.h,world),SSLM_OK,"release parent on other thread"); });
    release.join();
    TwinTokens(fx,&pool.pool,child,copy,16);
}
PKV_CELL("3.3/E3", "E1", ParentRelease33);

void Composition82() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model,400);
    sslm_prefix root = Root(fx,&pool.pool,sc.h,32);
    if (!root) return;
    const auto b = Save(root);
    if (!RequireBlob(b,"8.2")) return;
    sslm_prefix restored = nullptr;
    if (!Restored(fx,&pool.pool,b,nullptr,sc.h,&restored)) return;
    sslm_prefix child = Child(fx,restored,sc.h,32);
    if (!child) return;
    const auto cb = Save(child);
    sslm_prefix child_copy = nullptr;
    if (!Restored(fx,&pool.pool,cb,restored,sc.h,&child_copy)) return;
    sslm_prefix grandchild = Child(fx,child_copy,sc.h,32);
    const auto gb = grandchild ? Save(grandchild) : std::vector<uint8_t>{};
    sslm_prefix grandcopy = nullptr;
    if (Restored(fx,&pool.pool,gb,child_copy,sc.h,&grandcopy))
        TwinTokens(fx,&pool.pool,grandchild,grandcopy,16);
}
PKV_CELL("8.2/E3", "E1", Composition82);
}
