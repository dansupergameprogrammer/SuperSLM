#include "precache_hostile.h"
#include "precache_reference.h"

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
    ProbeExactlyFreeOneState(fx,&pool.pool,4);
    sslm_prefix out = nullptr;
    Status(Restore(fx,&pool.pool,b,nullptr,&out),SSLM_KV_POOL_EXHAUSTED,"5.1 root one short");
    PKV_CHECK(out == nullptr);
    ProbeExactlyFreeOneState(fx,&pool.pool,4);
    ProbeExactlyFree(fx,make,4);
    PKV_CHECK_EQ(sslm_kv_pool_destroy(pool.pool),SSLM_OK);
    pool.pool=nullptr;
}
PKV_CELL("5.1/E2", "E1", NoCapacity51);

void RootExact74() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    const auto b=RootBlob(fx,65); // ceil(65/16)=5.
    if(!RequireBlob(b,"7.4 exact root capacity")) return;
    PagePool pool(fx.model,5);
    ProbeExactlyFreeOneState(fx,&pool.pool,5);
    sslm_prefix out=nullptr;
    Status(Restore(fx,&pool.pool,b,nullptr,&out),SSLM_OK,
           "7.4 restore admitted with exactly five free pages");
    if(out && out!=OutSentinel<sslm_prefix>()) {
        sslm_seq two=nullptr;
        Status(sslm_seq_create_budgeted(fx.model,&pool.pool,16,&two),
               SSLM_KV_POOL_EXHAUSTED,"7.4 two-page create refused after exact draw");
        if(two) sslm_seq_release(two);
        Status(sslm_prefix_release(out),SSLM_OK,"7.4 release exact-capacity restore");
    }
    ProbeExactlyFreeOneState(fx,&pool.pool,5);
    PKV_CHECK_EQ(sslm_kv_pool_destroy(pool.pool),SSLM_OK);
    pool.pool=nullptr;
}
PKV_CELL("7.4/E2:capacity", "E1", RootExact74);

void Allocation52() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    const auto b = RootBlob(fx,65);
    if (!RequireBlob(b,"5.2")) return;
    PagePool pool(fx.model,32);
    ProbeExactlyFreeOneState(fx,&pool.pool,32);
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
    ProbeExactlyFreeOneState(fx,&pool.pool,32);
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
    ProbeExactlyFreeOneState(fx,&pool.pool,32);
    sslm_pc_test_only_restore_fault_arm(2,0); // §8: drain reserve immediately before MapPagesFor.
    sslm_prefix out = nullptr;
    Status(Restore(fx,&pool.pool,b,nullptr,&out),SSLM_KV_POOL_EXHAUSTED,
           "5.11 map short-reserve refusal");
    PKV_CHECK(out == nullptr);
    uint32_t free=0,shared=0,mapped=0,reserve=0;
    const int fired=sslm_pc_test_only_restore_fault_read(&free,&shared,&mapped,&reserve);
    PKV_CHECK_MSG(fired==1,
                  "5.11 map fault was not reached");
    if(fired) {
        PKV_CHECK_EQ(shared,0u);
        PKV_CHECK_EQ(mapped,0u);
        PKV_CHECK_EQ(reserve,0u);
        PKV_CHECK_EQ(free,32u); // reserve drained by the slot before mapping.
    }
    sslm_pc_test_only_restore_fault_disarm();
    ProbeExactlyFreeOneState(fx,&pool.pool,32);
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
    struct Fault {const char* name; size_t offset; uint64_t value; bool wide;
                  bool rehash; sslm_status want;};
    const std::vector<Fault> faults={
        {"magic",0,'Z',false,false,SSLM_INVALID_ARGUMENT},
        {"digest",28,1,false,false,SSLM_INVALID_ARGUMENT},
        {"backend",4,1,false,true,SSLM_INVALID_ARGUMENT},
        {"producer",8,0xffffffffu,false,true,PC_RESTORE_PRODUCER_MISMATCH},
        {"model hash",60,uint64_t(good[60]^1),false,true,SSLM_RESTORE_MODEL_MISMATCH},
        {"precision",92,0xffffffffu,false,true,SSLM_RESTORE_KV_MISMATCH},
        {"schema",96,1,true,true,SSLM_RESTORE_SCHEMA_MISMATCH},
        {"L",112,4097,true,true,SSLM_INVALID_ARGUMENT},
        {"origin",120,2,true,true,SSLM_INVALID_ARGUMENT},
        {"current token",108,0xffffffffu,false,true,SSLM_INVALID_ARGUMENT},
        {"mantissa",128,uint64_t(INT32_MAX)+1,true,true,SSLM_INVALID_ARGUMENT},
    };
    for(const auto& fault:faults) {
        auto corrupted=good;
        if(fault.offset==0) corrupted[0]=uint8_t(fault.value);
        else if(fault.offset==28) corrupted[28]^=1;
        else if(fault.wide) PutLe64(corrupted,fault.offset,fault.value);
        else PutLe32(corrupted,fault.offset,uint32_t(fault.value));
        if(fault.rehash) Digest(corrupted);
        sslm_pc_test_only_restore_fault_arm(3,0);
        sslm_prefix out=nullptr;
        Status(Restore(fx,&pool.pool,corrupted,nullptr,&out),fault.want,
               "5.6 validation precedes draw");
        PKV_CHECK_MSG(out==nullptr,"5.6 %s produced a handle",fault.name);
        PKV_CHECK_MSG(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr)==0,
                      "5.6 %s reached allocation slot before validation",fault.name);
        sslm_pc_test_only_restore_fault_disarm();
    }
    sslm_pc_test_only_restore_fault_arm(3,0);
    sslm_prefix short_out=nullptr;
    Status(sslm_prefix_restore(fx.model,&pool.pool,good.data(),183,nullptr,&short_out,nullptr),
           SSLM_INVALID_ARGUMENT,"5.6 short header before draw");
    PKV_CHECK(short_out==nullptr);
    PKV_CHECK_MSG(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr)==0,
                  "5.6 short header reached allocation slot");
    sslm_pc_test_only_restore_fault_disarm();
    for(auto s:held) sslm_seq_release(s);

    // Re-run the exact generated 2.1/2.2/2.3 population, rather than a
    // hand-picked corruption sample, at exactly zero free pages.
    const Fixture& synthetic=SyntheticFixture();
    if(!synthetic.ok) return;
    const auto one=RootBlob(synthetic,1);
    const auto thousand=RootBlob(synthetic,1000);
    if(!RequireBlob(one,"5.6 synthetic L=1") ||
       !RequireBlob(thousand,"5.6 synthetic L=1000")) return;
    PagePool zero(synthetic.model,8);
    std::vector<sslm_seq> zero_held;
    PKV_CHECK_MSG(AdmitPages(synthetic,&zero.pool,8,&zero_held),
                  "5.6 synthetic pool did not admit all eight pages");
    sslm_seq overflow=nullptr;
    Status(sslm_seq_create_budgeted(synthetic.model,&zero.pool,16,&overflow),
           SSLM_KV_POOL_EXHAUSTED,"5.6 synthetic zero-free confirmation");
    if(overflow) sslm_seq_release(overflow);
    size_t discriminating=0,nondiscriminating=0;
    auto check=[&](const Fixture& f,sslm_kv_pool* target,const HostileBlob& c) {
        if(c.draw_discriminating) ++discriminating; else ++nondiscriminating;
        sslm_pc_test_only_restore_fault_arm(3,0);
        sslm_prefix out=OutSentinel<sslm_prefix>();
        const auto st=Restore(f,target,c.bytes,nullptr,&out);
        PKV_CHECK_MSG(st==c.want && out==nullptr,
                      "5.6 %s: validation status %d became %d at zero free",
                      c.name.c_str(),int(c.want),int(st));
        PKV_CHECK_MSG(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr)==0,
                      "5.6 %s reached allocation slot before validation",c.name.c_str());
        sslm_pc_test_only_restore_fault_disarm();
        if(st==SSLM_OK && out && out!=OutSentinel<sslm_prefix>()) sslm_prefix_release(out);
    };
    TruncationPopulation(one,thousand,[&](const HostileBlob& c){check(synthetic,&zero.pool,c);});
    FlipPopulation(one,[&](const HostileBlob& c){check(synthetic,&zero.pool,c);});
    DomainPopulation(synthetic,one,[&](const HostileBlob& c){check(synthetic,&zero.pool,c);});
    const Fixture& bound_fx=SyntheticFixture(true);
    const auto bound=BoundWalkCase();
    if(RequireBlob(bound.bytes,"5.6 bound walk")) {
        PagePool bound_zero(bound_fx.model,8);
        std::vector<sslm_seq> bound_held;
        PKV_CHECK_MSG(AdmitPages(bound_fx,&bound_zero.pool,8,&bound_held),
                      "5.6 bound-model pool did not reach zero free");
        check(bound_fx,&bound_zero.pool,bound);
        for(auto seq:bound_held) sslm_seq_release(seq);
    }
    // The non-null-pool 2.5 cases run against an armed allocation slot too.
    sslm_pc_test_only_restore_fault_arm(3,0);
    sslm_prefix null_model=OutSentinel<sslm_prefix>();
    const auto null_model_st=sslm_prefix_restore(nullptr,&zero.pool,one.data(),one.size(),
                                                  nullptr,&null_model,nullptr);
    Status(null_model_st,SSLM_INVALID_ARGUMENT,"5.6 null model before draw");
    PKV_CHECK(null_model==nullptr);
    PKV_CHECK_EQ(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr),0);
    sslm_pc_test_only_restore_fault_disarm();
    sslm_pc_test_only_restore_fault_arm(3,0);
    Status(sslm_prefix_restore(synthetic.model,&zero.pool,one.data(),one.size(),
                               nullptr,nullptr,nullptr),SSLM_INVALID_ARGUMENT,
           "5.6 null out before draw");
    PKV_CHECK_EQ(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr),0);
    sslm_pc_test_only_restore_fault_disarm();
    sslm_pc_test_only_restore_fault_arm(3,0);
    sslm_prefix null_buffer=OutSentinel<sslm_prefix>();
    const auto null_buffer_st=sslm_prefix_restore(synthetic.model,&zero.pool,nullptr,one.size(),
                                                   nullptr,&null_buffer,nullptr);
    Status(null_buffer_st,SSLM_INVALID_ARGUMENT,"5.6 null buffer before draw");
    PKV_CHECK(null_buffer==nullptr);
    PKV_CHECK_EQ(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr),0);
    sslm_pc_test_only_restore_fault_disarm();
    // null buf, negative L, and truncations before byte 120 have no readable
    // positive L: their status is asserted but they do not discriminate draw-first.
    const Fixture& foreign=GetFixture("pkv_qk");
    if(foreign.ok) {
        PagePool foreign_zero(foreign.model,8);
        std::vector<sslm_seq> foreign_held;
        PKV_CHECK_MSG(AdmitPages(foreign,&foreign_zero.pool,8,&foreign_held),
                      "5.6 foreign-model pool did not reach zero free");
        sslm_seq foreign_extra=nullptr;
        Status(sslm_seq_create_budgeted(foreign.model,&foreign_zero.pool,16,&foreign_extra),
               SSLM_KV_POOL_EXHAUSTED,"5.6 foreign-model zero-free confirmation");
        if(foreign_extra) sslm_seq_release(foreign_extra);
        check(synthetic,&foreign_zero.pool,
              HostileBlob{"foreign-model pool",one,SSLM_INVALID_ARGUMENT,true});
        for(auto seq:foreign_held) sslm_seq_release(seq);
    }
    for(auto seq:zero_held) sslm_seq_release(seq);
    PKV_CHECK_MSG(discriminating>one.size() && nondiscriminating>=120,
                  "5.6 generated population or explicit nondiscriminating exceptions shrank");
}
PKV_CELL("5.6/E2", "E1", EmptyPool56);
PKV_CELL("7.2/E2", "E1", EmptyPool56);

void Precedence57() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    const auto good=RootBlob(fx,17);
    if (!RequireBlob(good,"5.7")) return;
    PagePool pool(fx.model,32);
    sslm_prefix unedited=nullptr;
    const auto accepted=Restore(fx,&pool.pool,good,nullptr,&unedited);
    PKV_CHECK_MSG(accepted==SSLM_OK && unedited &&
                  unedited!=OutSentinel<sslm_prefix>(),
                  "5.7 unedited CPU source must restore on the refusal pool");
    if(accepted!=SSLM_OK || !unedited || unedited==OutSentinel<sslm_prefix>()) {
        if(unedited && unedited!=OutSentinel<sslm_prefix>()) sslm_prefix_release(unedited);
        return;
    }
    Status(sslm_prefix_release(unedited),SSLM_OK,
           "5.7 release unedited CPU source on the refusal pool");
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
    auto producer=good;
    PutLe32(producer,8,UINT32_MAX);
    Digest(producer);
    sslm_prefix null_model=OutSentinel<sslm_prefix>();
    const auto null_st=sslm_prefix_restore(nullptr,&pool.pool,producer.data(),producer.size(),
                                           nullptr,&null_model,nullptr);
    PKV_CHECK_MSG(null_st==SSLM_INVALID_ARGUMENT && null_model==nullptr,
                  "5.7 step1 null model did not precede producer mismatch");
    auto short_blob=producer;
    short_blob.resize(kHeader-1);
    sslm_prefix short_out=nullptr;
    const auto short_st=Restore(fx,&pool.pool,short_blob,nullptr,&short_out);
    PKV_CHECK_MSG(short_st==SSLM_INVALID_ARGUMENT && short_out==nullptr,
                  "5.7 step2 size did not precede producer mismatch");
    auto magic=producer;
    magic[0]='Z';
    Digest(magic);
    sslm_prefix magic_out=nullptr;
    const auto magic_st=Restore(fx,&pool.pool,magic,nullptr,&magic_out);
    PKV_CHECK_MSG(magic_st==SSLM_INVALID_ARGUMENT && magic_out==nullptr,
                  "5.7 step2 magic did not precede producer mismatch");
    auto corrupted=producer;
    corrupted[28]^=1;
    sslm_prefix digest_out=nullptr;
    const auto digest_st=Restore(fx,&pool.pool,corrupted,nullptr,&digest_out);
    PKV_CHECK_MSG(digest_st==SSLM_INVALID_ARGUMENT && digest_out==nullptr,
                  "5.7 step3 digest did not precede producer mismatch");
    PKV_CHECK_MSG(Header(good,17,fx),"5.7 good CPU save is malformed");
    auto backend_only=good;
    PutLe32(backend_only,4,1);
    Digest(backend_only);
    auto gpu=backend_only;
    constexpr uint32_t vendor=0x1234,device=0x5678;
    PutLe32(gpu,20,vendor);
    PutLe32(gpu,24,device);
    Digest(gpu);
    const auto valid_digest=[](const std::vector<uint8_t>& b) {
        auto recomputed=b;
        Digest(recomputed);
        return b.size()>=kHeader &&
               std::equal(b.begin()+28,b.begin()+60,recomputed.begin()+28);
    };
    const bool later_fields=
        good.size()==kHeader+ArtifactHiddenSize(fx)+size_t(17)*fx.BytesPerToken() &&
        IsMagic(good,"SPX1") && Le32(good,4)==0 &&
        Le32(good,20)==0 && Le32(good,24)==0 &&
        backend_only.size()==good.size() && gpu.size()==good.size() &&
        Le32(backend_only,4)==1 && Le32(backend_only,20)==0 && Le32(backend_only,24)==0 &&
        Le32(gpu,4)==1 && Le32(gpu,20)==vendor && Le32(gpu,24)==device &&
        valid_digest(backend_only) && valid_digest(gpu) &&
        std::equal(good.begin()+60,good.end(),backend_only.begin()+60) &&
        std::equal(good.begin()+60,good.end(),gpu.begin()+60) &&
        Le32(gpu,92)==Le32(good,92) && Le64(gpu,96)==Le64(good,96) &&
        Le32(gpu,104)==Le32(good,104) && Le32(gpu,108)==Le32(good,108) &&
        Le64(gpu,112)==17 && Le64(gpu,120)==Le64(good,120);
    PKV_CHECK_MSG(later_fields,
                  "5.7 constructed GPU operands lost model, precision, schema, domain, size or payload");
    if(!later_fields) return;
    auto refuse_gpu=[&](const std::vector<uint8_t>& b,const char* name) {
        sslm_prefix out=nullptr;
        const auto st=Restore(fx,&pool.pool,b,nullptr,&out);
        PKV_CHECK_MSG(st==SSLM_INVALID_ARGUMENT && out==nullptr,
                      "5.7 %s backend was not first invalid status",name);
        if(st==SSLM_OK && out && out!=OutSentinel<sslm_prefix>()) sslm_prefix_release(out);
    };
    refuse_gpu(backend_only,"backend-only control");
    refuse_gpu(gpu,"GPU-shaped current-version control");
    PutLe32(gpu,8,UINT32_MAX);
    Digest(gpu);
    PKV_CHECK_MSG(valid_digest(gpu) &&
                  std::equal(backend_only.begin()+60,backend_only.end(),gpu.begin()+60),
                  "5.7 GPU producer edit changed later fields or invalidated digest");
    refuse_gpu(gpu,"GPU backend plus producer-version pair");
    PagePool zero(fx.model,8);
    std::vector<sslm_seq> held;
    PKV_CHECK_MSG(AdmitPages(fx,&zero.pool,8,&held),"5.7 zero-free pair setup failed");
    sslm_seq extra=nullptr;
    Status(sslm_seq_create_budgeted(fx.model,&zero.pool,16,&extra),
           SSLM_KV_POOL_EXHAUSTED,"5.7 zero-free pair confirmation");
    if(extra) sslm_seq_release(extra);
    auto domain=good;
    PutLe64(domain,120,uint64_t(-1));
    Digest(domain);
    sslm_prefix domain_out=nullptr;
    const auto domain_st=Restore(fx,&zero.pool,domain,nullptr,&domain_out);
    PKV_CHECK_MSG(domain_st==SSLM_INVALID_ARGUMENT && domain_out==nullptr,
                  "5.7 step8 domain did not precede zero-free capacity refusal");
    for(auto seq:held) sslm_seq_release(seq);
}
PKV_CELL("5.7/E2", "E1", Precedence57);

void RealGpuPrecedence57() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    const auto cpu=RootBlob(fx,17);
    if(!RequireBlob(cpu,"5.7 F2 real GPU comparison")) return;
    PagePool pool(fx.model,32);
    sslm_prefix unedited=nullptr;
    const auto accepted=Restore(fx,&pool.pool,cpu,nullptr,&unedited);
    PKV_CHECK_MSG(accepted==SSLM_OK && unedited &&
                  unedited!=OutSentinel<sslm_prefix>(),
                  "5.7 F2 unedited CPU source must restore on the refusal pool");
    if(accepted!=SSLM_OK || !unedited || unedited==OutSentinel<sslm_prefix>()) {
        if(unedited && unedited!=OutSentinel<sslm_prefix>()) sslm_prefix_release(unedited);
        return;
    }
    Status(sslm_prefix_release(unedited),SSLM_OK,
           "5.7 F2 release unedited CPU source on the refusal pool");
    const auto blob_path=Env("SUPERSLM_PRECACHE_GPU_BLOB");
    const auto receipt_path=Env("SUPERSLM_PRECACHE_GPU_CAPTURE_RECEIPT");
    const auto runner_path=Env("SUPERSLM_PRECACHE_GPU_RUNNER_EXE");
    if(blob_path.empty() || receipt_path.empty() || runner_path.empty()) return;
    std::vector<uint8_t> gpu;
    const auto receipt=LicenseFields(receipt_path);
    const auto field=[&](const char* key) {
        const auto it=receipt.find(key);
        return it==receipt.end() ? std::string{} : it->second;
    };
    const bool captured=ReadAll(blob_path,&gpu) && receipt.size()==8 &&
        field("format")=="PC-GPU-CAPTURE-1" && field("status")=="CAPTURED" &&
        field("backend")=="1" && field("model_sha256")==fx.sha &&
        field("blob_sha256")==ShaFile(blob_path) &&
        field("runner_sha256")==ShaFile(runner_path) &&
        !field("vendor").empty() && !field("device").empty();
    PKV_CHECK_MSG(captured,"5.7 F2 GPU blob lacks its captured runner/model/blob receipt");
    if(!captured || gpu.size()<kHeader || cpu.size()<kHeader) return;
    auto digest_copy=gpu;
    Digest(digest_copy);
    const bool later_fields=
        gpu.size()==kHeader+ArtifactHiddenSize(fx)+size_t(17)*fx.BytesPerToken() &&
        IsMagic(gpu,"SPX1") && Le32(gpu,4)==1 &&
        Le32(gpu,20)!=0 && Le32(gpu,24)!=0 &&
        field("vendor")==std::to_string(Le32(gpu,20)) &&
        field("device")==std::to_string(Le32(gpu,24)) &&
        std::equal(gpu.begin()+28,gpu.begin()+60,digest_copy.begin()+28) &&
        std::equal(gpu.begin()+60,gpu.begin()+92,cpu.begin()+60) &&
        Le32(gpu,92)==Le32(cpu,92) && Le64(gpu,96)==Le64(cpu,96) &&
        Le32(gpu,104)==Le32(cpu,104) && Le32(gpu,108)==Le32(cpu,108) &&
        Le64(gpu,112)==17 && Le64(gpu,120)==Le64(cpu,120);
    PKV_CHECK_MSG(later_fields,
                  "5.7 F2 GPU capture lacks valid backend, adapter, digest or later-field domain");
    if(!later_fields) return;
    auto reject=[&](const std::vector<uint8_t>& b,const char* what) {
        sslm_prefix out=nullptr;
        const auto st=Restore(fx,&pool.pool,b,nullptr,&out);
        PKV_CHECK_MSG(st==SSLM_INVALID_ARGUMENT && out==nullptr,
                      "5.7 F2 %s did not reject backend before producer",what);
        if(st==SSLM_OK && out && out!=OutSentinel<sslm_prefix>()) sslm_prefix_release(out);
    };
    reject(gpu,"real GPU current-version control");
    PutLe32(gpu,8,UINT32_MAX);
    Digest(gpu);
    reject(gpu,"real GPU backend/producer pair");
}
PKV_CELL("5.7/F2:real-gpu", "E1", RealGpuPrecedence57);

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
    const auto prefix_pages=PrefixTable(restored), adopter_pages=SeqTable(adopter);
    PKV_CHECK_MSG(prefix_pages.size()>=4 && adopter_pages.size()>=4 &&
                  std::equal(prefix_pages.begin(),prefix_pages.begin()+4,
                             adopter_pages.begin()),
                  "8.3 first adopter does not share four prefix page indices");
    std::vector<uint8_t> ssb;
    PKV_CHECK_MSG(SaveBlob(adopter,&ssb) && IsMagic(ssb,"SSB6"),"8.3 adopter did not save SSB6");
    sslm_seq reloaded=nullptr;
    uint32_t shared=0;
    Status(sslm_seq_restore_shared(fx.model,&pool.pool,ssb.data(),ssb.size(),restored,
        &reloaded,&shared),SSLM_OK,"8.3 shared sequence restore");
    if(reloaded) {
        sc.h.seqs.push_back(reloaded);
        PKV_CHECK_EQ(shared,4u);
        const auto reload_pages=SeqTable(reloaded);
        PKV_CHECK_MSG(prefix_pages.size()>=4 && reload_pages.size()>=4 &&
                      std::equal(prefix_pages.begin(),prefix_pages.begin()+4,
                                 reload_pages.begin()),
                      "8.3 SSB6 restored adopter does not share prefix page indices");
        Status(ReleasePrefix(sc.h,restored),SSLM_OK,"8.3 release prefix with adopters live");
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
    sslm_prefix empty=Root(fx,&pool.pool,sc.h,0);
    if(!empty) return;
    const auto empty_blob=Save(empty);
    if(!RequireBlob(empty_blob,"8.11 empty")) return;
    sslm_prefix empty_copy=nullptr;
    if(!Restored(fx,&pool.pool,empty_blob,nullptr,sc.h,&empty_copy)) return;
    sslm_schema schema=nullptr;
    Status(sslm_schema_lookup(fx.model,"g5_minimal_one_field",&schema),SSLM_OK,
           "8.11 schema lookup");
    if(schema) {
        const auto built_status=sslm_prefix_set_schema(empty,schema);
        const auto restored_status=sslm_prefix_set_schema(empty_copy,schema);
        Status(built_status,SSLM_INVALID_ARGUMENT,
               "8.11 prefill-built empty frozen schema refusal");
        Status(restored_status,SSLM_INVALID_ARGUMENT,
               "8.11 restored empty frozen schema refusal");
        PKV_CHECK_MSG(built_status==restored_status,
                      "8.11 empty frozen schema dispositions differ");
        PKV_CHECK_MSG(Save(empty)==empty_blob && Save(empty_copy)==empty_blob,
                      "8.11 refused frozen schema mutation changed SPX1 state");
    }
    for(auto p : {original,restored,empty,empty_copy})
        Status(ReleasePrefix(sc.h,p),SSLM_OK,"8.11 release frozen prefix");
    Status(sslm_kv_pool_destroy(pool.pool),SSLM_OK,"8.11 destroy drained pool");
    pool.pool=nullptr;
}
PKV_CELL("8.11/E2", "E1", FrozenDisposition811);
}
