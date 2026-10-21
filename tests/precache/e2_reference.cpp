#include "precache_reference.h"

namespace {
using namespace pc;
namespace fs=std::filesystem;

sslm_prefix BuildIds(const Fixture& fx,sslm_kv_pool* pool,Handles& h,
                     const std::vector<int32_t>& ids,sslm_prefix parent=nullptr) {
    sslm_prefix p=nullptr;
    const auto st=parent ? sslm_prefix_begin_from(parent,int32_t(ids.size()),&p)
                         : sslm_prefix_begin_budgeted(fx.model,pool,int32_t(ids.size()),&p);
    Status(st,SSLM_OK,"reference prefix begin");
    if(!p) return nullptr;
    h.prefixes.push_back(p);
    Status(PrefixPrefillAll(fx.model,p,ids,64),SSLM_OK,"reference prefix prefill");
    Status(sslm_prefix_freeze(p),SSLM_OK,"reference prefix freeze");
    return p;
}

void FlatReferenceE2() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    const auto prompts=Env("SUPERSLM_PRECACHE_SYNTHETIC_PROMPTS");
    if(prompts.empty()) return;
    const auto input=fs::path(prompts)/"flat00.ids";
    const auto ids=ReadIds(input);
    if(ids.empty()) return;
    Scene sc;
    // Twin and restored copies are both live when the legacy adopter draws
    // cap/B pages; the budget adopter asks for 64+4 decode tokens.
    PagePool& pool=sc.AddPool(fx.model,
        uint32_t(fx.CapPages()+2*fx.R(int64_t(ids.size()))+fx.R(68)));
    sslm_prefix twin=BuildIds(fx,&pool.pool,sc.h,ids);
    if(!twin) return;
    const auto blob=Save(twin);
    if(!RequireBlob(blob,"6.2 pinned synthetic flat")) return;
    sslm_prefix copy=nullptr;
    if(!Restored(fx,&pool.pool,blob,nullptr,sc.h,&copy)) return;
    int graded=0;
    for(int a=0;a<2;++a) {
        const bool budget=a==0;
        const auto got=Decode(fx,&pool.pool,copy,budget,64);
        const auto twin_tokens=Decode(fx,&pool.pool,twin,budget,64);
        const auto want=Reference("P6-pkv_def-flat","flat00",input,fx.sha,a,"greedy");
        PKV_CHECK_MSG(got.size()==64 && got==twin_tokens && got==want,
                      "6.2 pinned synthetic flat adopter %d differs",a);
        if(got.size()==64 && got==twin_tokens && got==want) ++graded;
    }
    PKV_CHECK_EQ(graded,2);
}
PKV_CELL("6.2/E2:reference", "E1", FlatReferenceE2);

void NestedReferenceE3() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    const auto prompts=Env("SUPERSLM_PRECACHE_SYNTHETIC_PROMPTS");
    if(prompts.empty()) return;
    const auto dir=fs::path(prompts);
    const auto world_ids=ReadIds(dir/"world.ids");
    const auto persona_ids=ReadIds(dir/"persona00.ids");
    const auto input=dir/"nested00.ids";
    const auto nested_ids=ReadIds(input);
    if(world_ids.empty() || persona_ids.empty() || nested_ids.empty()) return;
    std::vector<int32_t> joined=world_ids;
    joined.insert(joined.end(),persona_ids.begin(),persona_ids.end());
    PKV_CHECK_MSG(joined==nested_ids,"6.3 nested reference prompt is not world + persona");
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,500);
    sslm_prefix world=BuildIds(fx,&pool.pool,sc.h,world_ids);
    sslm_prefix twin=world ? BuildIds(fx,&pool.pool,sc.h,persona_ids,world) : nullptr;
    if(!twin) return;
    const auto blob=Save(twin);
    if(!RequireBlob(blob,"6.3 pinned synthetic nested")) return;
    sslm_prefix copy=nullptr;
    uint32_t shared=0;
    if(!Restored(fx,&pool.pool,blob,world,sc.h,&copy,&shared)) return;
    PKV_CHECK_EQ(shared,uint32_t(world_ids.size()/fx.B()));
    int graded=0;
    for(int a=0;a<2;++a) {
        const bool budget=a==0;
        const auto got=Decode(fx,&pool.pool,copy,budget,64);
        const auto twin_tokens=Decode(fx,&pool.pool,twin,budget,64);
        const auto want=Reference("P6-pkv_def-nested","nested00",input,fx.sha,a,"greedy");
        PKV_CHECK_MSG(got.size()==64 && got==twin_tokens && got==want,
                      "6.3 pinned synthetic nested adopter %d differs",a);
        if(got.size()==64 && got==twin_tokens && got==want) ++graded;
    }
    PKV_CHECK_EQ(graded,2);
}
PKV_CELL("6.3/E3:reference", "E1", NestedReferenceE3);
}
