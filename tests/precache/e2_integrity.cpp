#include "precache_hostile.h"
#include <set>

namespace {
using namespace pc;

void PageRows(const Fixture& fx,sslm_kv_pool pool,sslm_prefix prefix,
              const std::vector<uint8_t>& blob,int64_t L) {
    const auto table=PrefixTable(prefix);
    PKV_CHECK_EQ(table.size(),size_t((L+fx.B()-1)/fx.B()));
    if(table.size()!=size_t((L+fx.B()-1)/fx.B())) return;
    std::map<uint32_t,std::vector<uint8_t>> pages;
    const size_t D=fx.geo.head_dim,H=fx.geo.kv_heads;
    size_t differing=0;
    for(uint32_t l=0;l<fx.geo.layers;++l)
      for(uint32_t half=0;half<2;++half)
       for(uint32_t h=0;h<fx.geo.kv_heads;++h)
        for(int64_t pos=0;pos<L;++pos) {
            const uint32_t physical=table[size_t(pos/fx.B())];
            auto it=pages.find(physical);
            if(it==pages.end()) it=pages.emplace(physical,PeekPage(fx,pool,physical)).first;
            // 1.13.0 rotates every page's run slots (paged-KV §3.1, T1): the in-page address is
            // pkv::PageRunOffset's, the one every test-side page read goes through.
            const size_t page_off=PageRunOffset(fx.geo.layers,fx.geo.kv_heads,D,fx.B(),l,half,h,pos);
            const size_t blob_off=kHeader+ArtifactHiddenSize(fx)+
                ((size_t(l)*2+half)*H+h)*size_t(L)*D+size_t(pos)*D;
            if(it->second.size()<page_off+D || blob.size()<blob_off+D ||
               std::memcmp(it->second.data()+page_off,blob.data()+blob_off,D)!=0) ++differing;
        }
    PKV_CHECK_MSG(differing==0,"6.1: %zu K/V rows differ from the SPX1 canonical rows",differing);
}

void Fields61() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    for(int L : {0,1,15,16,17,65,1000}) {
        Scene sc;
        PagePool& pool=sc.AddPool(fx.model,256);
        sslm_prefix p=Root(fx,&pool.pool,sc.h,L);
        if(!p) continue;
        const auto b=Save(p);
        if(!RequireBlob(b,"6.1")) return;
        Header(b,L,fx);
        sslm_prefix restored=nullptr;
        if(!Restored(fx,&pool.pool,b,nullptr,sc.h,&restored)) continue;
        PageRows(fx,pool.pool,restored,b,L);
        auto adopter_state=[&](sslm_prefix prefix) {
            std::vector<uint8_t> ssb;
            sslm_seq seq=BudgetSeq(fx,&pool.pool,32,sc.h);
            if(!seq) return ssb;
            Status(sslm_seq_adopt_prefix(seq,prefix),SSLM_OK,"6.1 SSB6 adopter");
            PKV_CHECK_MSG(SaveBlob(seq,&ssb) && IsMagic(ssb,"SSB6"),
                          "6.1 adopted state did not save as SSB6");
            return ssb;
        };
        const auto original_state=adopter_state(p);
        const auto restored_state=adopter_state(restored);
        if(original_state.size()>=156 && restored_state.size()>=156) {
            for(size_t off : {size_t(40),size_t(76),size_t(84),size_t(92),
                              size_t(124),size_t(132),size_t(140),size_t(148)})
                PKV_CHECK_MSG(Le64(original_state,off)==Le64(restored_state,off),
                              "6.1 independent SSB6 state differs at offset %zu",off);
            PKV_CHECK_MSG(Le32(original_state,48)==Le32(restored_state,48),
                          "6.1 independent SSB6 schema walk differs");
        }
        const auto after=Save(restored);
        PKV_CHECK_MSG(after==b,"6.1: residual, scale, counts, token or schema changed");
        if(L==65) {
            auto distinct=b;
            PutLe64(distinct,128,17);
            PutLe64(distinct,136,uint64_t(-3));
            const std::array<uint64_t,5> counts{11,13,17,19,23};
            PKV_CHECK_MSG(std::all_of(counts.begin(),counts.end(),
                          [](uint64_t n){return n!=0;} ) &&
                          std::set<uint64_t>(counts.begin(),counts.end()).size()==5,
                          "6.1 count operand must be nonzero and pairwise distinct");
            for(size_t i=0;i<counts.size();++i) PutLe64(distinct,144+i*8,counts[i]);
            Digest(distinct);
            sslm_prefix edited=nullptr;
            if(Restored(fx,&pool.pool,distinct,nullptr,sc.h,&edited)) {
                const auto state=adopter_state(edited);
                if(state.size()>=156) {
                    PKV_CHECK_EQ(int64_t(Le64(state,76)),17);
                    PKV_CHECK_EQ(int64_t(Le64(state,84)),-3);
                    for(size_t i=0;i<counts.size();++i)
                        PKV_CHECK_MSG(Le64(state,i==0 ? 92 : 116+i*8)==counts[i],
                                      "6.1 saturation count %zu lost or swapped in SSB6",i);
                }
            }
        }
    }
}
PKV_CELL("6.1/E2", "E1", Fields61);
PKV_CELL("7.12/E2:format", "E1", Fields61);

void InDomain23b() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    const auto good=RootBlob(fx,65);
    if(!RequireBlob(good,"2.3b")) return;
    // The unedited prefix plus nine edited m/e prefixes stay live; each may
    // retain R(65) pages. TwinTokens also creates a whole-cap legacy adopter.
    const uint32_t pool_pages=uint32_t(fx.CapPages()+10*fx.R(65)+fx.R(20));
    PagePool pool(fx.model,pool_pages);
    Handles h;
    sslm_prefix unedited=nullptr;
    if(!Restored(fx,&pool.pool,good,nullptr,h,&unedited)) return;
    for(int64_t m : {int64_t(INT32_MIN),int64_t(0),int64_t(INT32_MAX)})
     for(int64_t e : {INT64_MIN,int64_t(0),INT64_MAX}) {
        auto b=good;
        PutLe64(b,128,uint64_t(m));
        PutLe64(b,136,uint64_t(e));
        Digest(b);
        sslm_prefix edited=nullptr;
        if(Restored(fx,&pool.pool,b,nullptr,h,&edited))
            TwinTokens(fx,&pool.pool,unedited,edited,16); // T-3024 F6.
     }
    auto bounded=[&](std::vector<uint8_t> b,const char* field) {
        Digest(b);
        sslm_prefix edited=nullptr;
        const auto st=Restore(fx,&pool.pool,b,nullptr,&edited);
        PKV_CHECK_MSG(int(st)>=0 && int(st)<int(SSLM_STATUS_NEXT_FREE),
                      "2.3b %s returned an unnamed status %d",field,int(st));
        if(st!=SSLM_OK) {
            PKV_CHECK_MSG(edited==nullptr,"2.3b %s refusal left a handle",field);
            std::printf("2.3b boundary=%s outcome=refused status=%s (%d)\n",
                        field,AbiStatusName(st),int(st));
            return;
        }
        PKV_CHECK_MSG(edited && edited!=OutSentinel<sslm_prefix>(),
                      "2.3b %s accepted without a handle",field);
        if(!edited || edited==OutSentinel<sslm_prefix>()) return;
        sslm_seq seq=nullptr;
        Status(sslm_seq_create_budgeted(fx.model,&pool.pool,32,&seq),SSLM_OK,
               "2.3b adopter create");
        if(seq) {
            Status(sslm_seq_adopt_prefix(seq,edited),SSLM_OK,"2.3b adopter adopt");
            std::vector<int32_t> tokens;
            const auto forward=DecodeN(fx.model,seq,16,&tokens);
            PKV_CHECK_MSG(int(forward)>=0 && int(forward)<int(SSLM_STATUS_NEXT_FREE),
                          "2.3b %s forward status is unnamed",field);
            PKV_CHECK_MSG(forward!=SSLM_OK || tokens.size()==16,
                          "2.3b %s forward stopped without a status",field);
            std::printf("2.3b boundary=%s outcome=%s tokens=%zu forward=%s (%d)\n",
                        field,forward==SSLM_OK ? "decoded16" : "named-forward-status",
                        tokens.size(),AbiStatusName(forward),int(forward));
            sslm_seq_release(seq);
        }
        sslm_prefix_release(edited);
    };
    for(size_t i=0;i<5;++i)
        for(uint64_t count : {uint64_t(0),UINT64_MAX}) {
            auto b=good;
            PutLe64(b,144+i*8,count);
            bounded(std::move(b),"saturation count");
        }
    const size_t rows=kHeader+ArtifactHiddenSize(fx);
    const size_t half_rows=size_t(65)*fx.geo.kv_heads*fx.geo.head_dim;
    const std::array<std::pair<const char*,size_t>,7> byte_edges{{
        {"residual first",kHeader},{"residual last",rows-1},
        {"K first",rows},{"K last",rows+half_rows-1},
        {"V first",rows+half_rows},{"last-layer V first",rows+3*half_rows},
        {"last-layer V last",good.size()-1}}};
    for(const auto& edge:byte_edges)
        for(uint8_t value : {uint8_t(0x80),uint8_t(0x7f)}) {
            auto b=good;
            b[edge.second]=value;
            bounded(std::move(b),edge.first);
        }
    for(uint32_t token : {uint32_t(0),uint32_t(fx.vocab-1)}) {
        auto b=good;
        PutLe32(b,108,token);
        bounded(std::move(b),"current_token");
    }
    const Fixture& bound_fx=SyntheticFixture(true);
    auto last_walk=BoundWalkCase();
    if(RequireBlob(last_walk.bytes,"2.3b walk=state_count-1")) {
        PutLe32(last_walk.bytes,104,1); // two-state DFA: last valid state.
        Digest(last_walk.bytes);
        PagePool bound_pool(bound_fx.model,32);
        sslm_prefix candidate=nullptr;
        const auto st=Restore(bound_fx,&bound_pool.pool,last_walk.bytes,nullptr,&candidate);
        PKV_CHECK_MSG(int(st)>=0 && int(st)<int(SSLM_STATUS_NEXT_FREE),
                      "2.3b last-valid walk returned unnamed status %d",int(st));
        if(st==SSLM_OK && candidate) {
            sslm_schema schema=nullptr;
            Status(sslm_schema_lookup(bound_fx.model,"schema_two",&schema),
                   SSLM_OK,"2.3b last-valid-walk schema lookup");
            sslm_seq unbound=nullptr;
            Status(sslm_seq_create_budgeted(bound_fx.model,&bound_pool.pool,32,&unbound),
                   SSLM_OK,"2.3b last-valid-walk unbound adopter");
            if(unbound) {
                Status(sslm_seq_adopt_prefix(unbound,candidate),
                       SSLM_PREFIX_SCHEMA_MISMATCH,
                       "2.3b unbound adopter rejects schema progress");
                sslm_seq_release(unbound);
            }
            sslm_seq adopter=nullptr;
            Status(sslm_seq_create_budgeted(bound_fx.model,&bound_pool.pool,32,&adopter),
                   SSLM_OK,"2.3b last-valid-walk bound adopter");
            if(adopter) {
                if(schema)
                    Status(sslm_seq_set_schema(adopter,schema),SSLM_OK,
                           "2.3b bind last-valid-walk adopter");
                const auto adopt=sslm_seq_adopt_prefix(adopter,candidate);
                Status(adopt,SSLM_OK,"2.3b last-valid-walk adopt");
                std::vector<int32_t> tokens;
                if(adopt==SSLM_OK) {
                    const auto forward=DecodeN(bound_fx.model,adopter,16,&tokens);
                    PKV_CHECK_MSG(int(forward)>=0 && int(forward)<int(SSLM_STATUS_NEXT_FREE) &&
                                  (forward!=SSLM_OK || tokens.size()==16),
                                  "2.3b last-valid walk has unnamed/short forward outcome");
                    std::printf("2.3b boundary=walk-last outcome=%s tokens=%zu forward=%s (%d)\n",
                                forward==SSLM_OK ? "decoded16" : "named-forward-status",
                                tokens.size(),AbiStatusName(forward),int(forward));
                }
                sslm_seq_release(adopter);
            }
            sslm_prefix_release(candidate);
        } else {
            PKV_CHECK(candidate==nullptr);
            std::printf("2.3b boundary=walk-last outcome=refused status=%s (%d)\n",
                        AbiStatusName(st),int(st));
        }
    }
}
PKV_CELL("2.3b/E2", "E1", InDomain23b);

void Reuse13() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    const auto a=RootBlob(fx,65);
    if(!RequireBlob(a,"1.3")) return;
    Scene source;
    PagePool& source_pool=source.AddPool(fx.model,32);
    sslm_prefix different=Root(fx,&source_pool.pool,source.h,65,83);
    if(!different) return;
    const auto b=Save(different);
    PagePool pool(fx.model,5); // the same five pages must be reused.
    sslm_prefix first=nullptr;
    Status(Restore(fx,&pool.pool,a,nullptr,&first),SSLM_OK,"1.3 first restore");
    if(!first || first==OutSentinel<sslm_prefix>()) return;
    const auto table=PrefixTable(first);
    Status(sslm_prefix_release(first),SSLM_OK,"1.3 first release");
    ExpectPoisoned(fx,pool.pool,std::set<uint32_t>(table.begin(),table.end()),"1.3 released scattered pages");
    sslm_prefix second=nullptr;
    Status(Restore(fx,&pool.pool,b,nullptr,&second),SSLM_OK,"1.3 second restore");
    if(second && second!=OutSentinel<sslm_prefix>()) {
        PageRows(fx,pool.pool,second,b,65);
        sslm_prefix_release(second);
    }
}
PKV_CELL("1.3/E2", "E1", Reuse13);

void Lifetime11() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    std::vector<std::vector<uint8_t>> blobs;
    for(int L:{1,16,17,65}) blobs.push_back(RootBlob(fx,L));
    if(!RequireBlob(blobs.front(),"1.1")) return;
    // pkv_def: cap/B=4096/16=256 legacy pages; largest live restore is
    // R(65)=ceil(65/16)+1=6; Decode(2) asks for budget 2+4, R(6)=2.
    const uint32_t pool_pages=uint32_t(fx.CapPages()+fx.R(65)+fx.R(6));
    PagePool pool(fx.model,pool_pages);
    for(int i=0;i<1000;++i) {
        sslm_prefix p=nullptr;
        const auto& b=blobs[size_t(i)%blobs.size()];
        Status(Restore(fx,&pool.pool,b,nullptr,&p),SSLM_OK,"1.1 restore cycle");
        if(!p || p==OutSentinel<sslm_prefix>()) return;
        for(bool budget:{false,true}) {
            const auto tokens=Decode(fx,&pool.pool,p,budget,2);
            PKV_CHECK_EQ(tokens.size(),2u);
        }
        Status(sslm_prefix_release(p),SSLM_OK,"1.1 release cycle");
    }
    ProbeExactlyFreeOneState(fx,&pool.pool,pool_pages);
    PKV_CHECK_EQ(sslm_kv_pool_destroy(pool.pool),SSLM_OK);
    pool.pool=nullptr;
}
PKV_CELL("1.1/E2", "E1", Lifetime11);

void Concurrent31() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,300);
    sslm_prefix immutable=Root(fx,&pool.pool,sc.h,65);
    if(!immutable) return;
    const auto expected=Save(immutable);
    if(!RequireBlob(expected,"3.1")) return;
    sslm_prefix other_prefix=Root(fx,&pool.pool,sc.h,17);
    if(!other_prefix) return;
    const auto other=Save(other_prefix);
    if(!RequireBlob(other,"3.1 other blob")) return;
    const auto source_tokens=Decode(fx,&pool.pool,immutable,true,16);
    const auto other_tokens=Decode(fx,&pool.pool,other_prefix,true,16);
    PKV_CHECK_MSG(source_tokens.size()==16 && other_tokens.size()==16,
                  "3.1 prefill twin continuations are unavailable");
    std::atomic<int> bad{0};
    std::vector<std::thread> workers;
    for(int i=0;i<8;++i) workers.emplace_back([&]{
        for(int j=0;j<8;++j) {
            size_t n=0;
            if(sslm_prefix_save(immutable,nullptr,&n)!=SSLM_BUFFER_TOO_SMALL || n!=expected.size()) {++bad;continue;}
            std::vector<uint8_t> b(n);
            if(sslm_prefix_save(immutable,b.data(),&n)!=SSLM_OK || b!=expected) ++bad;
            sslm_seq seq=nullptr;
            if(sslm_seq_create_budgeted(fx.model,&pool.pool,16,&seq)!=SSLM_OK) {++bad;continue;}
            if(sslm_seq_adopt_prefix(seq,immutable)!=SSLM_OK) ++bad;
            else {
                std::vector<int32_t> got;
                if(DecodeN(fx.model,seq,16,&got)!=SSLM_OK || got!=source_tokens) ++bad;
            }
            sslm_seq_release(seq);
        }
    });
    for(int i=0;i<8;++i) workers.emplace_back([&]{
        for(int j=0;j<8;++j) {
            sslm_prefix p=nullptr;
            if(sslm_prefix_restore(fx.model,&pool.pool,other.data(),other.size(),nullptr,&p,nullptr)!=SSLM_OK)
                {++bad;continue;}
            sslm_seq seq=nullptr;
            if(sslm_seq_create_budgeted(fx.model,&pool.pool,32,&seq)!=SSLM_OK) ++bad;
            else {
                std::vector<int32_t> got;
                if(sslm_seq_adopt_prefix(seq,p)!=SSLM_OK ||
                   DecodeN(fx.model,seq,16,&got)!=SSLM_OK || got!=other_tokens) ++bad;
                sslm_seq_release(seq);
            }
            sslm_prefix_release(p);
        }
    });
    for(auto& t:workers) t.join();
    PKV_CHECK_MSG(bad==0,"3.1 concurrent save/restore had %d failures",int(bad));
    Status(ReleasePrefix(sc.h,immutable),SSLM_OK,"3.1 release shared source");
    Status(ReleasePrefix(sc.h,other_prefix),SSLM_OK,"3.1 release other prefill twin");
    ProbeExactlyFreeOneState(fx,&pool.pool,300);
}
PKV_CELL("3.1/E2", "E1", Concurrent31);
}
