#include "precache_common.h"

namespace {
using namespace pc;

void NoCapacity51() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    const auto b = RootBlob(fx,65);
    if (!RequireBlob(b,"5.1")) return;
    auto make = [&]() -> std::unique_ptr<ProbeState> {
        auto sc = std::make_unique<Scene>();
        sc->AddPool(fx.model,4); // ceil(65/16)=5, one page short.
        return sc;
    };
    ProbeExactlyFree(fx,make,4);
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model,4);
    sslm_prefix out = nullptr;
    Status(Restore(fx,&pool.pool,b,nullptr,&out),SSLM_KV_POOL_EXHAUSTED,"5.1 root one short");
    PKV_CHECK(out == nullptr);
    ProbeExactlyFree(fx,make,4);
}
PKV_CELL("5.1/E2", "E1", NoCapacity51);
PKV_CELL("7.4/E2:capacity", "E1", NoCapacity51);

void Allocation52() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    const auto b = RootBlob(fx,65);
    if (!RequireBlob(b,"5.2")) return;
    PagePool pool(fx.model,32);
    sslm_pc_test_only_restore_fault_arm(3,0); // §8: allocation after draw/share/map.
    sslm_prefix out = nullptr;
    const sslm_status st = Restore(fx,&pool.pool,b,nullptr,&out);
    Status(st,SSLM_ALLOCATION_FAILED,"5.2 post-draw allocation fault");
    PKV_CHECK(out == nullptr);
    uint32_t free=0,shared=0,mapped=0,reserve=0;
    const int fired = sslm_pc_test_only_restore_fault_read(&free,&shared,&mapped,&reserve);
    PKV_CHECK_MSG(fired==1,"5.2 restore-site allocation fault was not reached");
    if (fired) {
        PKV_CHECK_MSG(free<32,"5.2 slot fired before any page was drawn");
        PKV_CHECK_EQ(shared,0u);
        PKV_CHECK_EQ(mapped,5u);
        PKV_CHECK_EQ(reserve,0u);
    }
    sslm_pc_test_only_restore_fault_disarm();
    PKV_CHECK_EQ(sslm_kv_pool_destroy(pool.pool),SSLM_OK);
    pool.pool=nullptr;
}
PKV_CELL("5.2/E2", "E1", Allocation52);

void Map511() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    const auto b = RootBlob(fx,65);
    if (!RequireBlob(b,"5.11")) return;
    PagePool pool(fx.model,32);
    sslm_pc_test_only_restore_fault_arm(2,0); // §8: drain reserve immediately before MapPagesFor.
    sslm_prefix out = nullptr;
    Status(Restore(fx,&pool.pool,b,nullptr,&out),SSLM_KV_POOL_EXHAUSTED,
           "5.11 map short-reserve refusal");
    PKV_CHECK(out == nullptr);
    uint32_t free=0,shared=0,mapped=0,reserve=0;
    PKV_CHECK_MSG(sslm_pc_test_only_restore_fault_read(&free,&shared,&mapped,&reserve)==1,
                  "5.11 map fault was not reached");
    sslm_pc_test_only_restore_fault_disarm();
    PKV_CHECK_EQ(sslm_kv_pool_destroy(pool.pool),SSLM_OK); // holder live_refs must fall.
    pool.pool=nullptr;
}
PKV_CELL("5.11/E2", "E1", Map511);

void EmptyPool56() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    const auto good = RootBlob(fx,1);
    if (!RequireBlob(good,"5.6")) return;
    PagePool pool(fx.model,8); // known N=8 pages.
    std::vector<sslm_seq> held;
    PKV_CHECK_MSG(AdmitPages(fx,&pool.pool,8,&held),"5.6 could not fill all 8 pages");
    sslm_seq extra=nullptr;
    Status(sslm_seq_create_budgeted(fx.model,&pool.pool,16,&extra),
           SSLM_KV_POOL_EXHAUSTED,"5.6 confirms zero free");
    auto corrupted=good;
    corrupted.back()^=1;
    sslm_pc_test_only_restore_fault_arm(3,0);
    sslm_prefix out=nullptr;
    Status(Restore(fx,&pool.pool,corrupted,nullptr,&out),SSLM_INVALID_ARGUMENT,
           "5.6 validation precedes draw");
    PKV_CHECK(out==nullptr);
    PKV_CHECK_MSG(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr)==0,
                  "5.6 allocation slot fired before validation");
    sslm_pc_test_only_restore_fault_disarm();
    for(auto s:held) sslm_seq_release(s);
}
PKV_CELL("5.6/E2", "E1", EmptyPool56);
PKV_CELL("7.2/E2", "E1", EmptyPool56);

void Precedence57() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    const auto good=RootBlob(fx,17);
    if (!RequireBlob(good,"5.7")) return;
    PagePool pool(fx.model,32);
    struct Pair { size_t early; size_t late; sslm_status want; };
    for (Pair p : {Pair{4,8,SSLM_INVALID_ARGUMENT},
                   Pair{8,60,PC_RESTORE_PRODUCER_MISMATCH},
                   Pair{60,92,SSLM_RESTORE_MODEL_MISMATCH},
                   Pair{92,96,SSLM_RESTORE_KV_MISMATCH},
                   Pair{96,120,SSLM_RESTORE_SCHEMA_MISMATCH}}) {
        auto b=good;
        b[p.early]^=1; b[p.late]^=1;
        Digest(b);
        sslm_prefix out=nullptr;
        const sslm_status st=Restore(fx,&pool.pool,b,nullptr,&out);
        Status(st,p.want,"5.7 first differing validation status");
        PKV_CHECK(out==nullptr);
        if (st==SSLM_OK && out && out!=OutSentinel<sslm_prefix>()) sslm_prefix_release(out);
    }
}
PKV_CELL("5.7/E2", "E1", Precedence57);

void Guard112() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,32);
    sslm_prefix p=Root(fx,&pool.pool,sc.h,17);
    if(!p) return;
    Status(sslm_pc_test_only_set_prefix_layer(p,1),SSLM_OK,
           "11.2 test-only layer mutation");
    size_t n=0;
    Status(sslm_prefix_save(p,nullptr,&n),SSLM_INVALID_ARGUMENT,
           "11.2 layer-index guard");
    Status(sslm_pc_test_only_set_prefix_layer(p,0),SSLM_OK,
           "11.2 restore fixture layer");
}
PKV_CELL("11.2/E2", "E1", Guard112);

void Composition83() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,128);
    sslm_prefix original=Root(fx,&pool.pool,sc.h,65);
    if(!original) return;
    const auto b=Save(original);
    if(!RequireBlob(b,"8.3")) return;
    sslm_prefix restored=nullptr;
    if(!Restored(fx,&pool.pool,b,nullptr,sc.h,&restored)) return;
    sslm_seq adopter=BudgetSeq(fx,&pool.pool,64,sc.h);
    if(!adopter) return;
    Status(sslm_seq_adopt_prefix(adopter,restored),SSLM_OK,"8.3 adopt");
    std::vector<uint8_t> ssb;
    PKV_CHECK_MSG(SaveBlob(adopter,&ssb) && IsMagic(ssb,"SSB6"),"8.3 adopter did not save SSB6");
    sslm_seq reloaded=nullptr;
    uint32_t shared=0;
    Status(sslm_seq_restore_shared(fx.model,&pool.pool,ssb.data(),ssb.size(),restored,
        &reloaded,&shared),SSLM_OK,"8.3 shared sequence restore");
    if(reloaded) {
        sc.h.seqs.push_back(reloaded);
        PKV_CHECK_EQ(shared,4u);
        std::vector<int32_t> a,c;
        PKV_CHECK_EQ(DecodeN(fx.model,adopter,16,&a),SSLM_OK);
        PKV_CHECK_EQ(DecodeN(fx.model,reloaded,16,&c),SSLM_OK);
        PKV_CHECK_MSG(a==c,"8.3 sequence restore tokens differ");
    }
}
PKV_CELL("8.3/E2", "E1", Composition83);

void Evol94() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    auto b=RootBlob(fx,17);
    if(!RequireBlob(b,"9.4")) return;
    PagePool pool(fx.model,32);
    b[3]='2';
    sslm_prefix out=nullptr;
    Status(Restore(fx,&pool.pool,b,nullptr,&out),SSLM_INVALID_ARGUMENT,"9.4 SPX2 magic");
    PKV_CHECK(out==nullptr);
}
PKV_CELL("9.4/E2", "E1", Evol94);

void FrozenDisposition811() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,64);
    sslm_prefix original=Root(fx,&pool.pool,sc.h,17);
    if(!original) return;
    const auto b=Save(original);
    if(!RequireBlob(b,"8.11")) return;
    sslm_prefix restored=nullptr;
    if(!Restored(fx,&pool.pool,b,nullptr,sc.h,&restored)) return;
    for (auto p : {original,restored}) {
        Status(sslm_prefix_freeze(p),SSLM_OK,"8.11 frozen freeze idempotent");
        Status(sslm_kv_pool_destroy(pool.pool),SSLM_POOL_HAS_LIVE_HANDLES,
               "8.11 destroy live pool");
        int32_t token=1,consumed=0;
        Status(sslm_prefix_prefill(fx.model,p,&token,1,1,SSLM_SPAN_PROMPT,nullptr,&consumed),
               SSLM_PREFIX_FROZEN_REJECTED,"8.11 frozen prefill");
    }
}
PKV_CELL("8.11/E2", "E1", FrozenDisposition811);
}
