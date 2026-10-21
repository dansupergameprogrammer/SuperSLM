#include "precache_hostile.h"
#include "precache_checkpoint.h"
#include "paged-kv/pkv_legacy_a_helpers.h"

namespace {
using namespace pc;

void Shape41() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    for (int L : {0, 1, 15, 16, 17, 1000, 4095, 4096}) {
        Scene sc;
        // At L=cap, original and restored prefixes each retain cap/B pages;
        // TwinTokens creates a third whole-cap legacy reservation.
        PagePool& pool = sc.AddPool(fx.model,
            uint32_t(3*fx.CapPages()+fx.R(20)));
        sslm_prefix twin = Root(fx, &pool.pool, sc.h, L);
        if (!twin) continue;
        auto b = Save(twin);
        if (!RequireBlob(b, "4.1")) return;
        Header(b, L, fx);
        sslm_prefix copy = nullptr;
        if (Restored(fx, &pool.pool, b, nullptr, sc.h, &copy)) {
            // An empty prefix has no last token to adopt, and cap-length
            // prefixes have no continuation budget. Their SPX1 checks stand.
            const int continuation=std::min(16,int(fx.geo.context_cap)-L);
            if(L>0 && continuation>0)
                TwinTokens(fx, &pool.pool, twin, copy, continuation);
        }
    }
}
PKV_CELL("4.1/E2", "E1", Shape41);

void SyntheticShapes41() {
    for(const auto& shape : {std::pair<const char*,Geometry>{"synthetic0p5.sslm",{24,2,64,4096}},
                             {"synthetic1p5.sslm",{28,2,128,32768}}}) {
        auto base=LoadSyntheticShape(shape.first,shape.second);
        if(!base->ok) return;
        Fixture four;
        PKV_CHECK_MSG(pkv::legacy_a::PatchedBlockSizeTwin(*base,4,&four),
                      "4.1 failed to construct B=4 synthetic twin");
        if(!four.ok || !PinFixtureB(&four,4)) return;
        for(const auto& geometry : {std::pair<const Fixture*,int>{base.get(),16},{&four,4}}) {
        const Fixture& fx=*geometry.first;
        const int B=geometry.second;
        PKV_CHECK_MSG(sslm_kv_page_size(fx.model)==size_t(B)*fx.BytesPerToken(),
                      "4.1 selected model is not B=%d",B);
        // Each pool has one cap-length prefix and then a full-cap legacy
        // adopter, including B=4 where cap/B is four times the B=16 count.
        const uint32_t pool_pages=uint32_t(2*fx.CapPages()+fx.R(20));
        PagePool source(fx.model,pool_pages);
        PagePool target(fx.model,pool_pages);
        Status(source.status,SSLM_OK,"4.1 source pool");
        Status(target.status,SSLM_OK,"4.1 target pool");
        if(source.status!=SSLM_OK || target.status!=SSLM_OK) return;
        // Each length is one checkpointed case; both pools carry across the
        // lengths of this geometry, so they form the case scope.
        CaseScope scope{{source.pool,target.pool}};
        for(int L : {0,1,B-1,B,B+1,1000,int(fx.geo.context_cap)-1,int(fx.geo.context_cap)}) {
            const CaseEnd end=RunCase(std::string(shape.first)+"/B="+std::to_string(B)+
                                      "/L="+std::to_string(L),{&fx},&scope,[&]() {
            Handles built,restored;
            sslm_prefix twin=Root(fx,&source.pool,built,L);
            if(!twin) return CaseEnd::StopCell;
            const auto b=Save(twin);
            if(!RequireBlob(b,"4.1 synthetic geometries")) return CaseEnd::StopCell;
            PKV_CHECK_MSG(b.size()==kHeader+ArtifactHiddenSize(fx)+size_t(L)*fx.BytesPerToken(),
                          "4.1 B=%d L=%d SPX1 length mismatch",B,L);
            sslm_prefix copy=nullptr;
            if(Restored(fx,&target.pool,b,nullptr,restored,&copy)) {
                PKV_CHECK_MSG(Save(copy)==b,"4.1 synthetic shape SPX1 bytes changed");
                const int continuation=std::min(16,int(fx.geo.context_cap)-L);
                if(L>0 && continuation>0)
                    for(bool budget : {false,true}) {
                        const auto a=Decode(fx,&source.pool,twin,budget,continuation);
                        const auto c=Decode(fx,&target.pool,copy,budget,continuation);
                        PKV_CHECK_MSG(a.size()==size_t(continuation) && a==c,
                                      "4.1 synthetic shape adopter differs from prefill twin");
                    }
            }
            return CaseEnd::Next;
            });
            if(end==CaseEnd::StopCell) return;
        }
        }
    }
}
PKV_CELL("4.1/E2:synthetic-shapes", "E1", SyntheticShapes41);

void RoundTrip64() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    auto round_trip=[](const Fixture& f,sslm_kv_pool* pool,Handles& handles,
                       sslm_prefix prefix,sslm_prefix parent,int32_t token,
                       const char* shape) {
        const auto first=Save(prefix);
        if(!RequireBlob(first,shape)) return;
        PKV_CHECK_MSG(int32_t(Le32(first,108))==token,
                      "6.4 %s last-prefill token field is wrong",shape);
        sslm_prefix copy=nullptr;
        if(!Restored(f,pool,first,parent,handles,&copy)) return;
        const char* mutant=std::getenv("SUPERSLM_PRECACHE_CURRENT_TOKEN_MUTANT");
        if(mutant && *mutant && token>=0) {
            const int32_t replacement=(token+1)%f.vocab;
            Status(sslm_pc_test_only_set_current_token(copy,replacement),SSLM_OK,
                   "6.4 isolated current-token mutant");
        }
        const auto second=Save(copy);
        PKV_CHECK_MSG(first==second,"6.4 %s SPX1 round-trip blob differs",shape);
        if(second.size()>=112)
            PKV_CHECK_MSG(int32_t(Le32(second,108))==token,
                          "6.4 %s restored token field differs from fixture input",shape);
    };
    for (int L : {0, 17, 65, 1000}) {
        Scene sc;
        PagePool& pool = sc.AddPool(fx.model, 300);
        sslm_prefix p = Root(fx, &pool.pool, sc.h, L);
        if (!p) continue;
        round_trip(fx,&pool.pool,sc.h,p,nullptr,
                   L ? Stream(41,L,fx.vocab).back() : -1,L ? "root" : "empty");
    }
    {
        Scene sc;
        PagePool& pool=sc.AddPool(fx.model,128);
        sslm_prefix parent=Root(fx,&pool.pool,sc.h,32);
        sslm_prefix child=nullptr;
        if(!parent) return;
        Status(sslm_prefix_begin_from(parent,17,&child),SSLM_OK,"6.4 nested begin_from");
        if(!child) return;
        sc.h.prefixes.push_back(child);
        const auto extension=Stream(62,17,fx.vocab);
        Status(PrefixPrefillAll(fx.model,child,extension,17),SSLM_OK,
               "6.4 nested extension prefill");
        Status(sslm_prefix_freeze(child),SSLM_OK,"6.4 nested freeze");
        round_trip(fx,&pool.pool,sc.h,child,parent,extension.back(),"nested");
    }
    const char* dir=std::getenv("SUPERSLM_PRECACHE_SYNTHETIC_DIR");
    PKV_CHECK_MSG(dir && *dir,"6.4 bound fixture directory is unset");
    if(!dir || !*dir) return;
    ModelFile model(std::string(dir)+"/synthetic0p5-three-schema.sslm");
    Status(model.status,SSLM_OK,"6.4 bound model map");
    if(model.status!=SSLM_OK) return;
    Fixture bound;
    bound.model=model.model; model.model=nullptr;
    bound.geo={24,2,64,4096}; bound.vocab=256;
    bound.stem="synthetic0p5-three-schema.sslm"; bound.ok=true;
    if(!PinFixtureB(&bound,16)) return;  // 1.13.0: the rule, not CFG1, selects B (precache_common.h)
    Scene sc;
    PagePool& pool=sc.AddPool(bound.model,64);
    sslm_schema schema=nullptr;
    Status(sslm_schema_lookup(bound.model,"schema_two",&schema),SSLM_OK,
           "6.4 bound schema lookup");
    sslm_prefix prefix=nullptr;
    Status(sslm_prefix_begin_budgeted(bound.model,&pool.pool,17,&prefix),SSLM_OK,
           "6.4 bound begin");
    if(!prefix) return;
    sc.h.prefixes.push_back(prefix);
    Status(sslm_prefix_set_schema(prefix,schema),SSLM_OK,"6.4 bind schema");
    std::vector<int32_t> ids(17,0);
    Status(SchemaContentAll(bound.model,prefix,ids,17),SSLM_OK,"6.4 bound prefill");
    Status(sslm_prefix_freeze(prefix),SSLM_OK,"6.4 bound freeze");
    round_trip(bound,&pool.pool,sc.h,prefix,nullptr,0,"schema-bound");
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
    if (!Restored(fx, &pool.pool, b, nullptr, sc.h, &copy)) return;
    const auto root_pages=PrefixTable(p), restored_pages=PrefixTable(copy);
    PKV_CHECK_MSG(root_pages.size()>=4 && restored_pages.size()>=4,
                  "8.1 fixture lacks four full prefix pages");
    sslm_seq shared_adopter=BudgetSeq(fx,&pool.pool,80,sc.h);
    if(shared_adopter) {
        Status(sslm_seq_adopt_prefix(shared_adopter,copy),SSLM_OK,"8.1 budget adoption");
        const auto adopter_pages=SeqTable(shared_adopter);
        PKV_CHECK_MSG(adopter_pages.size()>=4 && restored_pages.size()>=4 &&
                      std::equal(restored_pages.begin(),restored_pages.begin()+4,
                                 adopter_pages.begin()),
                      "8.1 budget adopter does not physically share floor(L/B) pages");
        Status(ReleasePrefix(sc.h,copy),SSLM_OK,"8.1 release restored prefix");
        copy=nullptr;
        std::vector<int32_t> keep;
        Status(DecodeN(fx.model,shared_adopter,16,&keep),SSLM_OK,
               "8.1 adopter retains shared pages after prefix release");
        PKV_CHECK_EQ(keep.size(),size_t(16));
    }
    if(!copy) {
        if(!Restored(fx,&pool.pool,b,nullptr,sc.h,&copy)) return;
    }
    TwinTokens(fx, &pool.pool, p, copy);
    auto mode_run=[&](sslm_prefix prefix,bool budget,int32_t mode) {
        sslm_seq seq=nullptr;
        Status(budget ? sslm_seq_create_budgeted(fx.model,&pool.pool,80,&seq)
                      : sslm_seq_create(fx.model,&pool.pool,&seq),
               SSLM_OK,"6.2 mode adopter create");
        if(!seq) return std::vector<int32_t>{};
        Status(sslm_seq_adopt_prefix(seq,prefix),SSLM_OK,"6.2 mode adopt");
        sslm_decode_params params{};
        Status(sslm_decode_params_init(fx.model,mode,int32_t(fx.geo.layers),&params),
               SSLM_OK,"6.2 mode init");
        sslm_seq batch[1]={seq};
        std::vector<int32_t> tokens;
        for(int i=0;i<64;++i) {
            int32_t token=-1;
            const auto st=sslm_decode_step_v2(fx.model,batch,1,&params,nullptr,&token);
            Status(st,SSLM_OK,"6.2 mode decode");
            if(st!=SSLM_OK || token<0) break;
            tokens.push_back(token);
        }
        sslm_seq_release(seq);
        return tokens;
    };
    for(bool budget : {false,true})
        for(auto mode : {SSLM_DECODE_MODE_GREEDY,SSLM_DECODE_MODE_DAMPED_GREEDY}) {
            const auto a=mode_run(p,budget,mode);
            const auto c=mode_run(copy,budget,mode);
            PKV_CHECK_MSG(a.size()==64 && a==c,"6.2 mode adopter differs from prefill twin");
        }
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
    sslm_prefix unfrozen=nullptr;
    Status(sslm_prefix_begin_budgeted(fx.model,&pool.pool,17,&unfrozen),SSLM_OK,
           "2.4 unfrozen fixture begin");
    if(unfrozen) {
        sc.h.prefixes.push_back(unfrozen);
        size_t unfrozen_n=0;
        Status(sslm_prefix_save(unfrozen,nullptr,&unfrozen_n),SSLM_INVALID_ARGUMENT,
               "2.4 unfrozen prefix save refusal");
    }
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
    ProbeExactlyFreeOneState(fx,&pool.pool,32);
    sslm_prefix out = OutSentinel<sslm_prefix>();
    sslm_pc_test_only_restore_fault_arm(3,0);
    Status(sslm_prefix_restore(fx.model, &pool.pool, b.data(), b.size(), nullptr, nullptr, nullptr),
           SSLM_INVALID_ARGUMENT, "2.5 null out");
    PKV_CHECK_EQ(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr),0);
    sslm_pc_test_only_restore_fault_disarm();
    for (int which = 0; which < 4; ++which) {
        out = OutSentinel<sslm_prefix>();
        sslm_kv_pool null_pool = nullptr;
        sslm_pc_test_only_restore_fault_arm(3,0);
        const sslm_status st = sslm_prefix_restore(which == 0 ? nullptr : fx.model,
            which == 1 ? nullptr : which == 2 ? &null_pool : &pool.pool,
            which == 3 ? nullptr : b.data(), b.size(), nullptr, &out, nullptr);
        Status(st, SSLM_INVALID_ARGUMENT, "2.5 null input");
        PKV_CHECK(out == nullptr);
        PKV_CHECK_EQ(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr),0);
        sslm_pc_test_only_restore_fault_disarm();
    }
    ProbeExactlyFreeOneState(fx,&pool.pool,32);
    const Fixture& foreign=GetFixture("pkv_qk");
    if(foreign.ok) {
        PagePool foreign_pool(foreign.model,32);
        ProbeExactlyFreeOneState(foreign,&foreign_pool.pool,32);
        out=OutSentinel<sslm_prefix>();
        sslm_pc_test_only_restore_fault_arm(3,0);
        Status(Restore(fx,&foreign_pool.pool,b,nullptr,&out),SSLM_INVALID_ARGUMENT,
               "2.5 pool belongs to another model");
        PKV_CHECK(out==nullptr);
        PKV_CHECK_EQ(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr),0);
        sslm_pc_test_only_restore_fault_disarm();
        ProbeExactlyFreeOneState(foreign,&foreign_pool.pool,32);
    }
}
PKV_CELL("2.5/E2", "E1", Restore25);

void ByteCorruption22() {
    const Fixture& fx = SyntheticFixture();
    if (!fx.ok) return;
    const auto good = RootBlob(fx, 1);
    if (!RequireBlob(good, "2.2")) return;
    PagePool pool(fx.model, 32);
    ProbeExactlyFreeOneState(fx,&pool.pool,32);
    FlipPopulation(good,[&](const HostileBlob& c) {
        sslm_prefix out=nullptr;
        sslm_pc_test_only_restore_fault_arm(3,0);
        const auto st=Restore(fx,&pool.pool,c.bytes,nullptr,&out);
        PKV_CHECK_MSG(st==c.want && out==nullptr,
                      "2.2 %s: expected integrity refusal, got %d",c.name.c_str(),int(st));
        PKV_CHECK_MSG(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr)==0,
                      "2.2 %s reached allocation before validation",c.name.c_str());
        sslm_pc_test_only_restore_fault_disarm();
        if(st==SSLM_OK && out && out!=OutSentinel<sslm_prefix>()) sslm_prefix_release(out);
    });
    ProbeExactlyFreeOneState(fx,&pool.pool,32);
    sslm_seq seq=nullptr;
    Status(sslm_seq_restore(fx.model,&pool.pool,good.data(),good.size(),&seq),
           SSLM_INVALID_ARGUMENT,"2.2 SPX1 rejected by sequence restore");
    PKV_CHECK(seq==nullptr);
    uint32_t shared=UINT32_MAX;
    Status(sslm_seq_restore_shared(fx.model,&pool.pool,good.data(),good.size(),
                                   nullptr,&seq,&shared),SSLM_INVALID_ARGUMENT,
           "2.2 SPX1 rejected by shared sequence restore");
    PKV_CHECK(seq==nullptr);
}
PKV_CELL("2.2/E2", "E1", ByteCorruption22);
PKV_CELL("7.3/E2", "E1", ByteCorruption22);

void Truncation21() {
    const Fixture& fx = SyntheticFixture();
    if (!fx.ok) return;
    const auto good = RootBlob(fx, 1);
    if (!RequireBlob(good, "2.1")) return;
    PagePool pool(fx.model, 32);
    const auto long_blob=RootBlob(fx,1000);
    if(!RequireBlob(long_blob,"2.1 L=1000")) return;
    PKV_CHECK_MSG(good.size()==kHeader+ArtifactHiddenSize(fx)+fx.BytesPerToken(),
                  "2.1 synthetic L=1 operand does not match CFG1 residual and K/V geometry");
    ProbeExactlyFreeOneState(fx,&pool.pool,32);
    TruncationPopulation(good,long_blob,[&](const HostileBlob& c) {
        sslm_prefix out=nullptr;
        sslm_pc_test_only_restore_fault_arm(3,0);
        const auto st=Restore(fx,&pool.pool,c.bytes,nullptr,&out);
        PKV_CHECK_MSG(st==c.want && out==nullptr,
                      "2.1 %s: expected size refusal, got %d",c.name.c_str(),int(st));
        PKV_CHECK_MSG(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr)==0,
                      "2.1 %s reached allocation before validation",c.name.c_str());
        sslm_pc_test_only_restore_fault_disarm();
        if(st==SSLM_OK && out && out!=OutSentinel<sslm_prefix>()) sslm_prefix_release(out);
    });
    ProbeExactlyFreeOneState(fx,&pool.pool,32);
}
PKV_CELL("2.1/E2", "E1", Truncation21);

void Domains23() {
    const Fixture& fx = SyntheticFixture();
    if (!fx.ok) return;
    const auto good = RootBlob(fx, 1);
    if (!RequireBlob(good, "2.3")) return;
    PagePool pool(fx.model, 64);
    ProbeExactlyFreeOneState(fx,&pool.pool,64);
    DomainPopulation(fx,good,[&](const HostileBlob& c) {
        sslm_prefix out=nullptr;
        sslm_pc_test_only_restore_fault_arm(3,0);
        const auto st=Restore(fx,&pool.pool,c.bytes,nullptr,&out);
        PKV_CHECK_MSG(st==c.want && out==nullptr,
                      "2.3 %s: expected %d, got %d",c.name.c_str(),int(c.want),int(st));
        PKV_CHECK_MSG(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr)==0,
                      "2.3 %s reached allocation before validation",c.name.c_str());
        sslm_pc_test_only_restore_fault_disarm();
        if(st==SSLM_OK && out && out!=OutSentinel<sslm_prefix>()) sslm_prefix_release(out);
    });
    ProbeExactlyFreeOneState(fx,&pool.pool,64);
    const Fixture& bound_fx=SyntheticFixture(true);
    const auto bound=BoundWalkCase();
    if(!RequireBlob(bound.bytes,"2.3 bound walk at state_count")) return;
    PagePool bound_pool(bound_fx.model,32);
    ProbeExactlyFreeOneState(bound_fx,&bound_pool.pool,32);
    sslm_prefix bound_out=nullptr;
    sslm_pc_test_only_restore_fault_arm(3,0);
    const auto bound_status=Restore(bound_fx,&bound_pool.pool,bound.bytes,nullptr,&bound_out);
    Status(bound_status,SSLM_RESTORE_SCHEMA_MISMATCH,"2.3 bound walk at state_count");
    PKV_CHECK(bound_out==nullptr);
    PKV_CHECK_EQ(sslm_pc_test_only_restore_fault_read(nullptr,nullptr,nullptr,nullptr),0);
    sslm_pc_test_only_restore_fault_disarm();
    if(bound_status==SSLM_OK && bound_out && bound_out!=OutSentinel<sslm_prefix>())
        sslm_prefix_release(bound_out);
    ProbeExactlyFreeOneState(bound_fx,&bound_pool.pool,32);
}
PKV_CELL("2.3/E2", "E1", Domains23);
PKV_CELL("7.6/E2", "E1", Domains23);
PKV_CELL("7.11/E2", "E1", Domains23);

void RootHint92() {
    const Fixture& fx = GetFixture("pkv_def");
    if (!fx.ok) return;
    Scene sc;
    // Three live prefix holders (65, 32, restored 65) precede TwinTokens'
    // cap/B-page legacy adopter.
    PagePool& pool = sc.AddPool(fx.model,
        uint32_t(fx.CapPages()+2*fx.R(65)+fx.R(32)+fx.R(20)));
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
