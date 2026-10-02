#include "precache_common.h"
#include "paged-kv/pkv_budget_b_helpers.h"

namespace {
using namespace pc;

void Adapter85() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,600);
    sslm_prefix twin=Root(fx,&pool.pool,sc.h,65);
    if(!twin) return;
    const auto blob=Save(twin);
    if(!RequireBlob(blob,"8.5")) return;
    sslm_prefix copy=nullptr;
    if(!Restored(fx,&pool.pool,blob,nullptr,sc.h,&copy)) return;
    pkv::budget_b::Adapter adapter(fx);
    if(!adapter.a) return;
    auto run=[&](sslm_prefix prefix,bool bind) {
        sslm_seq seq=nullptr;
        Status(sslm_seq_create_budgeted(fx.model,&pool.pool,80,&seq),SSLM_OK,"8.5 adopter create");
        if(!seq) return std::vector<int32_t>{};
        if(bind) Status(sslm_seq_set_adapter(seq,adapter.a),SSLM_OK,"8.5 adapter bind");
        Status(sslm_seq_adopt_prefix(seq,prefix),SSLM_OK,"8.5 adopt");
        std::vector<int32_t> out;
        Status(DecodeN(fx.model,seq,40,&out),SSLM_OK,"8.5 decode");
        sslm_seq_release(seq);
        return out;
    };
    const auto adapted=run(twin,true);
    PKV_CHECK_MSG(adapted!=run(twin,false),"8.5 adapter fixture did not affect tokens");
    PKV_CHECK_MSG(adapted.size()==40 && adapted==run(copy,true),
                  "8.5 restored prefix under adapter differs from prefill twin");
}
PKV_CELL("8.5/E2", "E1", Adapter85);

void LegacyPool86() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    const auto blob=RootBlob(fx,65);
    if(!RequireBlob(blob,"8.6")) return;
    LegacyPool baseline(fx.model,4), restored_pool(fx.model,4);
    Status(baseline.status,SSLM_OK,"8.6 baseline legacy pool");
    Status(restored_pool.status,SSLM_OK,"8.6 restored legacy pool");
    if(baseline.status!=SSLM_OK || restored_pool.status!=SSLM_OK) return;
    Handles base_handles, restored_handles;
    sslm_prefix twin=Root(fx,&baseline.pool,base_handles,65,41,true);
    sslm_prefix copy=nullptr;
    if(!twin || !Restored(fx,&restored_pool.pool,blob,nullptr,restored_handles,&copy)) return;
    const auto a=Decode(fx,&baseline.pool,twin,false,16);
    const auto b=Decode(fx,&restored_pool.pool,copy,false,16);
    PKV_CHECK_MSG(a.size()==16 && a==b,"8.6 legacy adopter differs from prefill twin");
    for(int step=0;step<4;++step) {
        const int base_count=CountLegacyCreates(fx.model,&baseline.pool);
        const int restored_count=CountLegacyCreates(fx.model,&restored_pool.pool);
        PKV_CHECK_MSG(base_count==restored_count,
                      "8.6 legacy admission differs at step %d: %d vs %d",
                      step,base_count,restored_count);
        sslm_seq x=nullptr,y=nullptr;
        const auto sx=sslm_seq_create(fx.model,&baseline.pool,&x);
        const auto sy=sslm_seq_create(fx.model,&restored_pool.pool,&y);
        PKV_CHECK_MSG(sx==sy && sx==(base_count ? SSLM_OK : SSLM_KV_POOL_EXHAUSTED),
                      "8.6 next legacy create/refusal differs at step %d",step);
        if(x) base_handles.seqs.push_back(x);
        if(y) restored_handles.seqs.push_back(y);
        if(!base_count) break;
    }
}
PKV_CELL("8.6/E2", "E1", LegacyPool86);

void CrossBuild91() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    const auto local=RootBlob(fx,65);
    if(!RequireBlob(local,"9.1")) return;
    PagePool pool(fx.model,100);
    Handles h;
    sslm_prefix same=nullptr;
    if(Restored(fx,&pool.pool,local,nullptr,h,&same))
        PKV_CHECK_MSG(Save(same)==local,"9.1 local blob changed on round trip");
    const char* path=std::getenv("SUPERSLM_PRECACHE_CLOUD_BLOB");
    PKV_CHECK_MSG(path && *path,"9.1 cloud blob path is unset");
    if(!path || !*path) return;
    std::vector<uint8_t> cloud;
    PKV_CHECK_MSG(ReadAll(path,&cloud) && !cloud.empty(),"9.1 cloud blob is missing");
    if(cloud.empty()) return;
    sslm_prefix remote=nullptr;
    if(Restored(fx,&pool.pool,cloud,nullptr,h,&remote))
        PKV_CHECK_MSG(Save(remote)==local,"9.1 cloud and box SPX1 bytes differ");
}
PKV_CELL("9.1/E2", "E1", CrossBuild91);

void Ordinals716() {
    PKV_CHECK_EQ(int(SSLM_STATUS_NEXT_FREE),31);
    PKV_CHECK_EQ(int(SSLM_KV_BUDGET_EXCEEDED),29);
#ifdef SSLM_HAS_PREFIX_SAVE_ABI
    PKV_CHECK_EQ(int(SSLM_RESTORE_PRODUCER_MISMATCH),30);
#else
    PKV_CHECK_MSG(false,"7.16 producer mismatch status and prefix ABI macro are absent");
#endif
}
PKV_CELL("7.16/E2", "E1", Ordinals716);

void ValidationVitality111() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    const auto good=RootBlob(fx,1);
    if(!RequireBlob(good,"11.1")) return;
    PagePool pool(fx.model,40);
    for(size_t offset : {size_t(0),size_t(4),size_t(8),size_t(28),size_t(60),size_t(92),size_t(108),size_t(112)}) {
        auto bad=good;
        bad[offset]^=1;
        if(offset!=28 && offset!=0) Digest(bad);
        sslm_prefix out=nullptr;
        const auto st=Restore(fx,&pool.pool,bad,nullptr,&out);
        PKV_CHECK_MSG(st!=SSLM_OK && out==nullptr,
                      "11.1 validation at blob offset %zu did not fire",offset);
        if(st==SSLM_OK && out) sslm_prefix_release(out);
    }
}
PKV_CELL("11.1/E2", "E1", ValidationVitality111);
}
