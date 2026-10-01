#include "precache_common.h"
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
            const size_t page_off=((size_t(l)*2+half)*H+h)*size_t(fx.B())*D+
                                  size_t(pos%fx.B())*D;
            const size_t blob_off=kHeader+HiddenSize(fx)+
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
        const auto after=Save(restored);
        PKV_CHECK_MSG(after==b,"6.1: residual, scale, counts, token or schema changed");
    }
}
PKV_CELL("6.1/E2", "E1", Fields61);
PKV_CELL("7.12/E2:format", "E1", Fields61);

void InDomain23b() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    const auto good=RootBlob(fx,65);
    if(!RequireBlob(good,"2.3b")) return;
    PagePool pool(fx.model,200);
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
    PagePool pool(fx.model,32);
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
    std::atomic<int> bad{0};
    std::vector<std::thread> workers;
    for(int i=0;i<8;++i) workers.emplace_back([&]{
        for(int j=0;j<8;++j) {
            size_t n=0;
            if(sslm_prefix_save(immutable,nullptr,&n)!=SSLM_BUFFER_TOO_SMALL || n!=expected.size()) {++bad;continue;}
            std::vector<uint8_t> b(n);
            if(sslm_prefix_save(immutable,b.data(),&n)!=SSLM_OK || b!=expected) ++bad;
            sslm_prefix p=nullptr;
            if(sslm_prefix_restore(fx.model,&pool.pool,expected.data(),expected.size(),nullptr,&p,nullptr)!=SSLM_OK)
                {++bad;continue;}
            sslm_prefix_release(p);
        }
    });
    for(auto& t:workers) t.join();
    PKV_CHECK_MSG(bad==0,"3.1 concurrent save/restore had %d failures",int(bad));
}
PKV_CELL("3.1/E2", "E1", Concurrent31);
}
