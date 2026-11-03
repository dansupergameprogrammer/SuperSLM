#include "precache_hostile.h"
#include "precache_provenance.h"
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
    const auto local=RootBlob(fx,1000);
    if(!RequireBlob(local,"9.1")) return;
    PagePool pool(fx.model,200);
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
    const std::string cloud_compiler=Env("SUPERSLM_PRECACHE_CLOUD_COMPILER");
      const std::string receipt_dir=Env("SUPERSLM_PRECACHE_CROSS_COMPILER_DIR");
      const auto directory=std::filesystem::path(receipt_dir);
      const auto receipt=ReadReceipt(directory/
                                     (cloud_compiler+".receipt"));
      const auto expected_cloud=directory/(cloud_compiler+".spx1");
      const auto local_receipt=LocalCompilerReceipt(fx,Stream(41,1000,fx.vocab),local);
      const bool provenance=
          (cloud_compiler=="GNU" || cloud_compiler=="Clang") &&
          std::filesystem::weakly_canonical(path)==std::filesystem::weakly_canonical(expected_cloud) &&
          ValidCompilerJob(directory,receipt,local_receipt,cloud_compiler.c_str()) &&
          receipt.at("blob_sha256")==Sha(cloud.data(),cloud.size());
      PKV_CHECK_MSG(provenance,
                    "9.1 cloud blob lacks its verified producer binary, compiler job, or exact captured path");
    if(!provenance) return;
    sslm_prefix remote=nullptr;
    if(Restored(fx,&pool.pool,cloud,nullptr,h,&remote))
        PKV_CHECK_MSG(Save(remote)==local,"9.1 cloud and box SPX1 bytes differ");
}
PKV_CELL("9.1/E2", "E1", CrossBuild91);

void Ordinals716() {
    // Schema skip-ahead on the GPU (1.17.0) appends two names after the producer status, which keeps 30.
    PKV_CHECK_EQ(int(SSLM_STATUS_NEXT_FREE),33);
    PKV_CHECK_EQ(int(SSLM_KV_BUDGET_EXCEEDED),29);
#ifdef SSLM_HAS_PREFIX_SAVE_ABI
    PKV_CHECK_EQ(int(SSLM_RESTORE_PRODUCER_MISMATCH),30);
    PKV_CHECK_EQ(int(SSLM_GPU_FORCED_TABLE_ABSENT),31);
    PKV_CHECK_EQ(int(SSLM_GPU_CONTEXT_CAP_EXCEEDED),32);
#else
    PKV_CHECK_MSG(false,"7.16 producer mismatch status and prefix ABI macro are absent");
#endif
}
PKV_CELL("7.16/E2", "E1", Ordinals716);

void ValidationVitality111() {
    const Fixture& fx=SyntheticFixture();
    if(!fx.ok) return;
    const auto good=RootBlob(fx,1);
    if(!RequireBlob(good,"11.1")) return;
    PagePool pool(fx.model,40);
    ProbeExactlyFreeOneState(fx,&pool.pool,40);
    size_t sites=0;
    auto refuse=[&](const HostileBlob& c) {
        ++sites;
        sslm_prefix out=nullptr;
        sslm_pc_test_only_restore_fault_arm(3,0);
        const auto st=Restore(fx,&pool.pool,c.bytes,nullptr,&out);
        PKV_CHECK_MSG(st==c.want && out==nullptr,
                      "11.1 Release validation site %s: expected %s, got %s",
                      c.name.c_str(),AbiStatusName(c.want),AbiStatusName(st));
        PKV_CHECK_MSG(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr)==0,
                      "11.1 validation site %s drew before refusing",c.name.c_str());
        sslm_pc_test_only_restore_fault_disarm();
        if(st==SSLM_OK && out) sslm_prefix_release(out);
    };
    auto short_blob=good; short_blob.resize(183);
    refuse({"step 2: short size",short_blob,SSLM_INVALID_ARGUMENT,true});
    auto magic=good; magic[0]^=1;
    refuse({"step 2: magic",magic,SSLM_INVALID_ARGUMENT,true});
    auto digest=good; digest.back()^=1;
    refuse({"step 3: digest",digest,SSLM_INVALID_ARGUMENT,true});
    DomainPopulation(fx,good,refuse); // Steps 4-8, one field changed and digest correct.
    const auto empty=RootBlob(fx,0);
    if(RequireBlob(empty,"11.1 empty token domain")) {
        auto token=empty;
        PutLe32(token,108,0);
        Digest(token);
        refuse({"step 8: L=0 token not sentinel",token,SSLM_INVALID_ARGUMENT,true});
    }
    const Fixture& bound_fx=SyntheticFixture(true);
    const auto bound=BoundWalkCase();
    if(RequireBlob(bound.bytes,"11.1 bound walk")) {
        PagePool bound_pool(bound_fx.model,40);
        sslm_prefix out=nullptr;
        const auto st=Restore(bound_fx,&bound_pool.pool,bound.bytes,nullptr,&out);
        PKV_CHECK_MSG(st==SSLM_RESTORE_SCHEMA_MISMATCH && out==nullptr,
                      "11.1 step 7 bound walk at state_count was not rejected");
        if(out && out!=OutSentinel<sslm_prefix>()) sslm_prefix_release(out);
        ++sites;
    }
    PKV_CHECK_MSG(sites>=27,"11.1 validation-site census is incomplete: %zu",sites);
    ProbeExactlyFreeOneState(fx,&pool.pool,40);
}
PKV_CELL("11.1/E2", "E1", ValidationVitality111);
}
