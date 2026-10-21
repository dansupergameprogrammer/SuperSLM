#pragma once
#include "precache_common.h"

namespace pc {
struct HostileBlob {
    std::string name;
    std::vector<uint8_t> bytes;
    sslm_status want = SSLM_INVALID_ARGUMENT;
    bool draw_discriminating = true;
};

template<class Visit>
void TruncationPopulation(const std::vector<uint8_t>& one,
                          const std::vector<uint8_t>& thousand,Visit visit) {
    for(size_t n=0;n<one.size();++n)
        visit(HostileBlob{"L=1 truncation at " + std::to_string(n),
                          {one.begin(),one.begin()+n},SSLM_INVALID_ARGUMENT,n>=120});
    auto extra=one;
    extra.push_back(0);
    visit(HostileBlob{"L=1 one byte over",std::move(extra),SSLM_INVALID_ARGUMENT,true});
    for(size_t n=0;n<std::min(kHeader,thousand.size());++n)
        visit(HostileBlob{"L=1000 header truncation at " + std::to_string(n),
                          {thousand.begin(),thousand.begin()+n},SSLM_INVALID_ARGUMENT,n>=120});
    for(size_t n=kHeader;n<thousand.size();n+=4096)
        visit(HostileBlob{"L=1000 strided truncation at " + std::to_string(n),
                          {thousand.begin(),thousand.begin()+n},SSLM_INVALID_ARGUMENT,true});
    extra=thousand;
    extra.push_back(0);
    visit(HostileBlob{"L=1000 one byte over",std::move(extra),SSLM_INVALID_ARGUMENT,true});
}

template<class Visit>
void FlipPopulation(const std::vector<uint8_t>& one,Visit visit) {
    for(size_t offset=0;offset<one.size();++offset) {
        auto b=one;
        b[offset]^=1;
        const bool readable=b.size()>=120 && int64_t(Le64(b,112))>=1;
        visit(HostileBlob{"L=1 flipped byte " + std::to_string(offset),
                          std::move(b),SSLM_INVALID_ARGUMENT,readable});
    }
    for(const char* magic : {"SSB6","SSB5","SLM6"}) {
        auto b=one;
        std::memcpy(b.data(),magic,4);
        visit(HostileBlob{std::string("foreign magic ")+magic,std::move(b),
                          SSLM_INVALID_ARGUMENT,true});
    }
}

template<class Visit>
void DomainPopulation(const Fixture& fx,const std::vector<uint8_t>& one,Visit visit) {
    struct Edit {const char* name;size_t offset;uint64_t value;bool wide;sslm_status want;};
    const Edit edits[] = {
        {"backend",4,1,false,SSLM_INVALID_ARGUMENT},
        {"producer major",8,UINT32_MAX,false,PC_RESTORE_PRODUCER_MISMATCH},
        {"producer minor",12,UINT32_MAX,false,PC_RESTORE_PRODUCER_MISMATCH},
        {"producer patch",16,UINT32_MAX,false,PC_RESTORE_PRODUCER_MISMATCH},
        {"adapter vendor",20,1,false,PC_RESTORE_PRODUCER_MISMATCH},
        {"adapter device",24,1,false,PC_RESTORE_PRODUCER_MISMATCH},
        {"precision",92,UINT32_MAX,false,SSLM_RESTORE_KV_MISMATCH},
        {"unknown schema",96,1,true,SSLM_RESTORE_SCHEMA_MISMATCH},
        {"unbound walk",104,1,false,SSLM_RESTORE_SCHEMA_MISMATCH},
        {"origin above L",120,2,true,SSLM_INVALID_ARGUMENT},
        {"origin negative",120,uint64_t(-1),true,SSLM_INVALID_ARGUMENT},
        {"L negative",112,uint64_t(-1),true,SSLM_INVALID_ARGUMENT},
        {"L above cap",112,uint64_t(fx.geo.context_cap)+1,true,SSLM_INVALID_ARGUMENT},
        {"L wrap onto original n",112,(uint64_t{1}<<53)+1,true,SSLM_INVALID_ARGUMENT},
        {"size disagrees with L",112,2,true,SSLM_INVALID_ARGUMENT},
        {"token -1",108,UINT32_MAX,false,SSLM_INVALID_ARGUMENT},
        {"token -2",108,UINT32_MAX-1,false,SSLM_INVALID_ARGUMENT},
        {"token INT32_MIN",108,uint32_t(INT32_MIN),false,SSLM_INVALID_ARGUMENT},
        {"token at vocab",108,uint64_t(fx.vocab),false,SSLM_INVALID_ARGUMENT},
        {"mantissa above int32",128,uint64_t(INT32_MAX)+1,true,SSLM_INVALID_ARGUMENT},
        {"mantissa below int32",128,uint64_t(int64_t(INT32_MIN)-1),true,SSLM_INVALID_ARGUMENT},
    };
    for(const auto& e:edits) {
        auto b=one;
        if(e.wide) PutLe64(b,e.offset,e.value);
        else PutLe32(b,e.offset,uint32_t(e.value));
        Digest(b);
        visit(HostileBlob{e.name,std::move(b),e.want,
                          int64_t(Le64(one,112))>=1 && std::strcmp(e.name,"L negative")!=0});
    }
    auto model=one;
    model[60]^=1;
    Digest(model);
    visit(HostileBlob{"model hash",std::move(model),SSLM_RESTORE_MODEL_MISMATCH,true});
    PKV_CHECK_MSG((uint64_t{1}<<53)+1 <= uint64_t(INT64_MAX) &&
                  uint64_t(kHeader+ArtifactHiddenSize(fx))+
                  (((uint64_t{1}<<53)+1)*uint64_t(fx.BytesPerToken()))==one.size(),
                  "2.3 L-wrap operand does not wrap the size formula onto n");
}

inline HostileBlob BoundWalkCase() {
    const Fixture& fx=SyntheticFixture(true);
    if(!fx.ok) return {"bound walk at state_count",{},SSLM_RESTORE_SCHEMA_MISMATCH,true};
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,32);
    sslm_schema schema=nullptr;
    Status(sslm_schema_lookup(fx.model,"schema_two",&schema),SSLM_OK,
           "2.3 bound-walk schema lookup");
    sslm_prefix prefix=nullptr;
    Status(sslm_prefix_begin_budgeted(fx.model,&pool.pool,1,&prefix),SSLM_OK,
           "2.3 bound-walk prefix begin");
    if(!prefix || !schema) return {"bound walk at state_count",{},SSLM_RESTORE_SCHEMA_MISMATCH,true};
    sc.h.prefixes.push_back(prefix);
    Status(sslm_prefix_set_schema(prefix,schema),SSLM_OK,"2.3 bound-walk bind");
    std::vector<int32_t> zero{0};
    Status(SchemaContentAll(fx.model,prefix,zero,1),SSLM_OK,"2.3 bound-walk prefill");
    Status(sslm_prefix_freeze(prefix),SSLM_OK,"2.3 bound-walk freeze");
    auto bytes=Save(prefix);
    if(bytes.size()>=108) {
        PKV_CHECK_MSG(Le32(bytes,104)==1,
                      "2.3 bound operand did not advance DFA walk to 1");
        PutLe32(bytes,104,2); // the fixture has exactly two DFA states.
        Digest(bytes);
    }
    return {"bound walk at state_count",std::move(bytes),SSLM_RESTORE_SCHEMA_MISMATCH,true};
}
}
