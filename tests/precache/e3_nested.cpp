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
    sslm_status release_status=SSLM_INVALID_ARGUMENT;
    std::thread release([&]{ release_status=ReleasePrefix(sc.h,world); });
    release.join();
    Status(release_status,SSLM_OK,"release parent on other thread");
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

void NestedCapacity51() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene source;
    PagePool& source_pool=source.AddPool(fx.model,200);
    sslm_prefix source_parent=Root(fx,&source_pool.pool,source.h,1000,61);
    sslm_prefix source_child=source_parent ? Child(fx,source_parent,source.h) : nullptr;
    if(!source_child) return;
    const auto b=Save(source_child);
    if(!RequireBlob(b,"5.1 nested")) return;
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,75); // parent 63; 12 free, 13 needed.
    sslm_prefix parent=Root(fx,&pool.pool,sc.h,1000,61);
    if(!parent) return;
    sslm_prefix out=nullptr;
    Status(Restore(fx,&pool.pool,b,parent,&out),SSLM_KV_POOL_EXHAUSTED,
           "5.1 nested one page short");
    PKV_CHECK(out==nullptr);
}
PKV_CELL("5.1/E3", "E1", NestedCapacity51);

void NestedAllocation52() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,200);
    sslm_prefix parent=Root(fx,&pool.pool,sc.h,1000,61);
    sslm_prefix child=parent ? Child(fx,parent,sc.h) : nullptr;
    if(!child) return;
    const auto b=Save(child);
    if(!RequireBlob(b,"5.2 nested")) return;
    sslm_pc_test_only_restore_fault_arm(3,0);
    sslm_prefix out=nullptr;
    Status(Restore(fx,&pool.pool,b,parent,&out),SSLM_ALLOCATION_FAILED,
           "5.2 nested post-map allocation failure");
    PKV_CHECK(out==nullptr);
    uint32_t free=0,shared=0,mapped=0,reserve=0;
    PKV_CHECK_MSG(sslm_pc_test_only_restore_fault_read(&free,&shared,&mapped,&reserve)==1,
                  "5.2 nested allocation slot not reached");
    PKV_CHECK_EQ(shared,62u);
    PKV_CHECK_EQ(mapped,75u);
    PKV_CHECK_EQ(reserve,0u);
    sslm_pc_test_only_restore_fault_disarm();
}
PKV_CELL("5.2/E3", "E1", NestedAllocation52);

void NestedRefusals511() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene source;
    PagePool& source_pool=source.AddPool(fx.model,200);
    sslm_prefix parent=Root(fx,&source_pool.pool,source.h,1000,61);
    sslm_prefix child=parent ? Child(fx,parent,source.h) : nullptr;
    if(!child) return;
    const auto b=Save(child);
    if(!RequireBlob(b,"5.11 nested")) return;
    for(int status : {1,2}) { // test slot: 1=kInvalidArgument; 2=kRefcountUnderflow.
        sslm_pc_test_only_restore_fault_arm(1,status);
        sslm_prefix out=nullptr;
        Status(Restore(fx,&source_pool.pool,b,parent,&out),
               status==1 ? SSLM_INVALID_ARGUMENT : SSLM_ARTIFACT_REJECTED,
               "5.11 share refusal");
        PKV_CHECK(out==nullptr);
        uint32_t free=0,shared=0,mapped=0,reserve=0;
        PKV_CHECK_MSG(sslm_pc_test_only_restore_fault_read(&free,&shared,&mapped,&reserve)==1,
                      "5.11 share slot not reached");
        sslm_pc_test_only_restore_fault_disarm();
    }
    sslm_pc_test_only_restore_fault_arm(2,0);
    sslm_prefix out=nullptr;
    Status(Restore(fx,&source_pool.pool,b,parent,&out),SSLM_KV_POOL_EXHAUSTED,
           "5.11 nested map refusal");
    PKV_CHECK(out==nullptr);
    sslm_pc_test_only_restore_fault_disarm();
}
PKV_CELL("5.11/E3", "E1", NestedRefusals511);

void NestedAdmission74() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene source;
    PagePool& source_pool=source.AddPool(fx.model,200);
    sslm_prefix built_parent=Root(fx,&source_pool.pool,source.h,1000,61);
    sslm_prefix built_child=built_parent ? Child(fx,built_parent,source.h) : nullptr;
    if(!built_child) return;
    const auto b=Save(built_child);
    if(!RequireBlob(b,"7.4 nested")) return;
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,76); // 63 parent pages + exactly 13 free.
    sslm_prefix parent=Root(fx,&pool.pool,sc.h,1000,61);
    if(!parent) return;
    sslm_prefix out=nullptr;
    uint32_t shared=0;
    if(Restored(fx,&pool.pool,b,parent,sc.h,&out,&shared)) {
        PKV_CHECK_EQ(shared,62u);
        sslm_seq extra=nullptr;
        Status(sslm_seq_create_budgeted(fx.model,&pool.pool,16,&extra),
               SSLM_KV_POOL_EXHAUSTED,"7.4 no two pages after exact draw");
    }
}
PKV_CELL("7.4/E3", "E1", NestedAdmission74);

void Concurrent32() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,500);
    sslm_prefix parent=Root(fx,&pool.pool,sc.h,1000,61);
    sslm_prefix child=parent ? Child(fx,parent,sc.h) : nullptr;
    if(!child) return;
    const auto b=Save(child);
    if(!RequireBlob(b,"3.2")) return;
    std::atomic<int> bad{0};
    std::vector<std::thread> threads;
    for(int i=0;i<8;++i) threads.emplace_back([&]{
        sslm_prefix copy=nullptr;
        uint32_t shared=0;
        if(sslm_prefix_restore(fx.model,&pool.pool,b.data(),b.size(),parent,&copy,&shared)!=SSLM_OK)
            {++bad;return;}
        if(shared!=62) ++bad;
        sslm_prefix_release(copy);
    });
    for(int i=0;i<8;++i) threads.emplace_back([&]{
        sslm_seq seq=nullptr;
        if(sslm_seq_create_budgeted(fx.model,&pool.pool,16,&seq)!=SSLM_OK) {++bad;return;}
        if(sslm_seq_adopt_prefix(seq,parent)!=SSLM_OK) ++bad;
        sslm_seq_release(seq);
        sslm_prefix extension=nullptr;
        if(sslm_prefix_begin_from(parent,16,&extension)!=SSLM_OK) {++bad;return;}
        sslm_prefix_release(extension);
    });
    for(auto& thread:threads) thread.join();
    PKV_CHECK_MSG(bad==0,"3.2 concurrent re-share failures: %d",int(bad));
}
PKV_CELL("3.2/E3", "E1", Concurrent32);

void ReleaseOrders12() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene source;
    PagePool& source_pool=source.AddPool(fx.model,200);
    sslm_prefix source_parent=Root(fx,&source_pool.pool,source.h,1000,61);
    sslm_prefix source_child=source_parent ? Child(fx,source_parent,source.h) : nullptr;
    if(!source_child) return;
    const auto b=Save(source_child);
    if(!RequireBlob(b,"1.2")) return;
    std::array<int,3> order{0,1,2}; // parent, child, both adopters.
    do {
        PagePool pool(fx.model,200);
        Handles h;
        sslm_prefix parent=Root(fx,&pool.pool,h,1000,61);
        sslm_prefix child=nullptr;
        if(!Restored(fx,&pool.pool,b,parent,h,&child)) return;
        Handles refs;
        sslm_seq a=BudgetSeq(fx,&pool.pool,128,h);
        sslm_seq c=BudgetSeq(fx,&pool.pool,128,h);
        sslm_seq ra=BudgetSeq(fx,&source_pool.pool,128,refs);
        sslm_seq rc=BudgetSeq(fx,&source_pool.pool,128,refs);
        if(!a || !c || !ra || !rc) return;
        Status(sslm_seq_adopt_prefix(a,child),SSLM_OK,"1.2 budget adopter");
        Status(sslm_seq_adopt_prefix(c,child),SSLM_OK,"1.2 second adopter");
        Status(sslm_seq_adopt_prefix(ra,source_child),SSLM_OK,"1.2 twin adopter a");
        Status(sslm_seq_adopt_prefix(rc,source_child),SSLM_OK,"1.2 twin adopter b");
        bool adopters_live=true;
        for(int item:order) {
            if(item==0) Status(ReleasePrefix(h,parent),SSLM_OK,"1.2 release parent");
            if(item==1) Status(ReleasePrefix(h,child),SSLM_OK,"1.2 release child");
            if(item==2) {
                Status(ReleaseSeq(h,a),SSLM_OK,"1.2 release adopter a");
                Status(ReleaseSeq(h,c),SSLM_OK,"1.2 release adopter b");
                adopters_live=false;
            }
            if(adopters_live) {
                std::vector<int32_t> x,y,rx,ry;
                PKV_CHECK_EQ(DecodeN(fx.model,a,16,&x),SSLM_OK);
                PKV_CHECK_EQ(DecodeN(fx.model,c,16,&y),SSLM_OK);
                PKV_CHECK_EQ(DecodeN(fx.model,ra,16,&rx),SSLM_OK);
                PKV_CHECK_EQ(DecodeN(fx.model,rc,16,&ry),SSLM_OK);
                PKV_CHECK_MSG(x.size()==16 && x==rx && y==ry,
                              "1.2 adopters differ from prefill twins after release order");
            }
        }
        PKV_CHECK_EQ(sslm_kv_pool_destroy(pool.pool),SSLM_OK);
        pool.pool=nullptr;
    } while(std::next_permutation(order.begin(),order.end()));
}
PKV_CELL("1.2/E3", "E1", ReleaseOrders12);
}
