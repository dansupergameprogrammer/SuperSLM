// T-3022 E0: prefill-only reference streams at engine anchor 5d5ca58.
// This executable calls no save/restore prefix verb and contains no precache implementation.
#include "superslm/sslm_abi.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Check(sslm_status s, const char* verb) {
    if (s != SSLM_OK) throw std::runtime_error(std::string(verb) + " status=" + std::to_string(int(s)));
}
uint32_t U32(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
uint64_t U64(const uint8_t* p) { uint64_t v=0; for (int i=0;i<8;++i) v|=uint64_t(p[i])<<(8*i); return v; }
struct Aligned {
    uint8_t* p=nullptr;
    explicit Aligned(size_t n) : p(static_cast<uint8_t*>(::operator new(n, std::align_val_t(64)))) {}
    ~Aligned() { if(p) ::operator delete(p, std::align_val_t(64)); }
};
std::vector<int32_t> Stream(uint32_t seed, int n) {
    std::vector<int32_t> out;
    uint32_t x=seed*2654435761u+12345u;
    for(int i=0;i<n;++i) { x=x*1664525u+1013904223u; out.push_back(int32_t((x>>8)%256)); }
    return out;
}
struct Input {
    std::unique_ptr<Aligned> bytes;
    size_t size=0;
    sslm_model model=nullptr;
    uint32_t cap=0, vocab=0;
    explicit Input(const char* path) {
        std::ifstream f(path, std::ios::binary|std::ios::ate);
        if(!f) throw std::runtime_error("open model failed");
        size=static_cast<size_t>(f.tellg());
        bytes=std::make_unique<Aligned>(size);
        f.seekg(0);
        if(!f.read(reinterpret_cast<char*>(bytes->p), static_cast<std::streamsize>(size))) throw std::runtime_error("read model failed");
        const uint8_t* h=bytes->p;
        if(size<64 || std::memcmp(h,"SSLM",4)!=0 || U64(h+24)!=size) throw std::runtime_error("bad SSLM header");
        const uint32_t count=U32(h+12);
        for(uint32_t i=0;i<count;++i) {
            const uint8_t* row=h+64+size_t(i)*40;
            if(U32(row)==0) {
                const uint8_t* cfg=h+U64(row+8);
                if(std::memcmp(cfg,"CFG1",4)!=0) throw std::runtime_error("bad CFG1");
                vocab=U32(cfg+32); cap=U32(cfg+36); break;
            }
        }
        if(!cap || !vocab) throw std::runtime_error("missing config");
        Check(sslm_model_map(bytes->p,size,&model),"model_map");
    }
    ~Input() { if(model) sslm_model_unmap(model); }
};
struct Pool {
    std::unique_ptr<Aligned> bytes;
    sslm_kv_pool pool=nullptr;
    Pool(sslm_model model, uint32_t cap) {
        const int64_t raw_b=sslm_kv_page_positions(model);
        if(raw_b<=0 || raw_b>UINT32_MAX) throw std::runtime_error("invalid page positions");
        const uint32_t b=static_cast<uint32_t>(raw_b);
        const uint32_t pages=(cap+b-1)/b+300;
        const size_t n=sslm_kv_page_size(model)*pages+sslm_kv_page_pool_overhead_size(model,pages);
        bytes=std::make_unique<Aligned>(n);
        Check(sslm_kv_page_pool_create(model,bytes->p,n,pages,&pool),"pool_create");
    }
    ~Pool() { if(pool) sslm_kv_pool_destroy(pool); }
};
void Prefill(sslm_model model, sslm_prefix p, const std::vector<int32_t>& tokens, sslm_span_kind kind) {
    size_t at=0;
    while(at<tokens.size()) {
        int32_t consumed=0;
        const int32_t n=std::min<int32_t>(64,int32_t(tokens.size()-at));
        Check(sslm_prefix_prefill(model,p,tokens.data()+at,n,64,kind,nullptr,&consumed),"prefix_prefill");
        if(consumed<=0) throw std::runtime_error("prefix_prefill consumed no token");
        at+=size_t(consumed);
    }
}
struct Prefix {
    sslm_prefix p=nullptr;
    ~Prefix() { if(p) sslm_prefix_release(p); }
};
std::string SchemaName(sslm_model model, int index) {
    size_t n=0;
    if(sslm_schema_name(model,index,nullptr,&n)!=SSLM_BUFFER_TOO_SMALL || !n) throw std::runtime_error("schema_name query failed");
    std::string s(n,'\0');
    Check(sslm_schema_name(model,index,s.data(),&n),"schema_name");
    return s;
}
std::string Capture(sslm_model model, Pool& pool, sslm_prefix prefix, bool budget, bool damped,
                    sslm_schema schema, sslm_adapter adapter) {
    sslm_seq s=nullptr;
    if(budget) Check(sslm_seq_create_budgeted(model,&pool.pool,128,&s),"seq_create_budgeted");
    else Check(sslm_seq_create(model,&pool.pool,&s),"seq_create");
    try {
        if(schema) Check(sslm_seq_set_schema(s,schema),"seq_set_schema");
        Check(sslm_seq_adopt_prefix(s,prefix),"seq_adopt_prefix");
        if(adapter) Check(sslm_seq_set_adapter(s,adapter),"seq_set_adapter");
        sslm_decode_params params{};
        if(damped) Check(sslm_decode_params_init(model,SSLM_DECODE_MODE_DAMPED_GREEDY,1,&params),"params_init_damped");
        else { params.layer_budget=1; }
        std::string out;
        sslm_seq batch[1]={s};
        for(int i=0;i<64;++i) {
            int32_t token=-1;
            sslm_status st=SSLM_OK;
            for(int guard=0;token==-1 && guard<4096;++guard) {
                st=damped ? sslm_decode_step_v2(model,batch,1,&params,nullptr,&token)
                          : sslm_decode_step(model,batch,1,&params,nullptr,&token);
                if(st!=SSLM_OK) break;
            }
            if(st!=SSLM_OK || token<0) {
                out+=" status="+std::to_string(int(st))+" terminal="+std::to_string(token);
                break;
            }
            if(i) out+=',';
            out+=std::to_string(token);
        }
        Check(sslm_seq_release(s),"seq_release");
        return out;
    } catch(...) { sslm_seq_release(s); throw; }
}
void Row(std::ofstream& out, const std::string& scenario, int index, int adopter,
         sslm_model model, Pool& pool, sslm_prefix p, bool damped_available, sslm_schema schema, sslm_adapter adapter) {
    const bool budget=(adopter%2)==0;
    const bool damped=damped_available && (adopter==2 || adopter==3);
    out<<scenario<<' '<<index<<' '<<adopter<<" mode="<<(budget?"budget":"legacy")
       <<" decode="<<(damped?"damped":"greedy")<<" schema="<<(schema?"bound":"none")
       <<" adapter="<<(adapter?"bound":"none")<<" tokens="
       <<Capture(model,pool,p,budget,damped,schema,adapter)<<'\n';
    if(!out) throw std::runtime_error("write reference failed");
}
void Flat(std::ofstream& out, Input& in, Pool& pool, bool damped) {
    const int lengths[]={128,256,384,512,768,1000,1200,1500,1800,2000};
    for(int i=0;i<10;++i) {
        Prefix prefix;
        Check(sslm_prefix_begin_budgeted(in.model,&pool.pool,lengths[i],&prefix.p),"prefix_begin_budgeted");
        Prefill(in.model,prefix.p,Stream(51+uint32_t(i)*7,lengths[i]),SSLM_SPAN_PROMPT);
        Check(sslm_prefix_freeze(prefix.p),"prefix_freeze");
        for(int a=0;a<5;++a) Row(out,"flat",i,a,in.model,pool,prefix.p,damped,nullptr,nullptr);
        std::cerr<<"flat "<<i<<"\n";
    }
}
void Nested(std::ofstream& out, Input& in, Pool& pool, bool damped) {
    Prefix world;
    Check(sslm_prefix_begin_budgeted(in.model,&pool.pool,1000,&world.p),"world_begin");
    Prefill(in.model,world.p,Stream(61,1000),SSLM_SPAN_PROMPT);
    Check(sslm_prefix_freeze(world.p),"world_freeze");
    for(int i=0;i<10;++i) {
        Prefix child;
        Check(sslm_prefix_begin_from(world.p,200,&child.p),"prefix_begin_from");
        Prefill(in.model,child.p,Stream(71+uint32_t(i),200),SSLM_SPAN_PROMPT);
        Check(sslm_prefix_freeze(child.p),"child_freeze");
        for(int a=0;a<5;++a) Row(out,"nested",i,a,in.model,pool,child.p,damped,nullptr,nullptr);
        std::cerr<<"nested "<<i<<"\n";
    }
}
void Composition(std::ofstream& out, Input& in, Pool& pool, bool damped, const char* adapter_path) {
    if(sslm_schema_count(in.model)) {
        const std::string name=SchemaName(in.model,0);
        sslm_schema schema=nullptr;
        Check(sslm_schema_lookup(in.model,name.c_str(),&schema),"schema_lookup");
        Prefix p;
        Check(sslm_prefix_begin_budgeted(in.model,&pool.pool,128,&p.p),"schema_prefix_begin");
        Check(sslm_prefix_set_schema(p.p,schema),"prefix_set_schema");
        Prefill(in.model,p.p,Stream(84,128),SSLM_SPAN_PROMPT);
        Check(sslm_prefix_freeze(p.p),"schema_prefix_freeze");
        for(int a=0;a<4;++a) Row(out,"schema_"+name,0,a,in.model,pool,p.p,damped,schema,nullptr);
    }
    if(adapter_path && *adapter_path) {
        std::ifstream f(adapter_path,std::ios::binary|std::ios::ate);
        if(!f) throw std::runtime_error("adapter open failed");
        const size_t n=static_cast<size_t>(f.tellg());
        Aligned bytes(n); f.seekg(0);
        if(!f.read(reinterpret_cast<char*>(bytes.p),static_cast<std::streamsize>(n))) throw std::runtime_error("adapter read failed");
        sslm_adapter adapter=nullptr;
        Check(sslm_adapter_map(bytes.p,n,in.model,&adapter),"adapter_map");
        Prefix p;
        Check(sslm_prefix_begin_budgeted(in.model,&pool.pool,128,&p.p),"adapter_prefix_begin");
        Prefill(in.model,p.p,Stream(91,128),SSLM_SPAN_PROMPT);
        Check(sslm_prefix_freeze(p.p),"adapter_prefix_freeze");
        for(int a=0;a<2;++a) Row(out,"adapter",0,a,in.model,pool,p.p,false,nullptr,adapter);
        Check(sslm_adapter_release(adapter),"adapter_release");
    }
}
}
#ifndef T3022_NO_MAIN
int main(int argc,char** argv) {
    try {
        if(argc<3 || argc>4) throw std::runtime_error("usage: t3022_precache_reference MODEL OUT [ADAPTER]");
        Input in(argv[1]); Pool pool(in.model,in.cap);
        std::ofstream out(argv[2],std::ios::binary|std::ios::trunc);
        if(!out) throw std::runtime_error("open output failed");
        sslm_decode_params probe{};
        const bool damped=sslm_decode_params_init(in.model,SSLM_DECODE_MODE_DAMPED_GREEDY,1,&probe)==SSLM_OK;
        out<<"# anchor=5d5ca58 scenarios=flat10x5,nested10x5,schema,adapter damped_available="<<int(damped)<<'\n';
        Flat(out,in,pool,damped);
        Nested(out,in,pool,damped);
        Composition(out,in,pool,damped,argc==4?argv[3]:nullptr);
        out.flush(); if(!out) throw std::runtime_error("flush reference failed");
        std::cerr<<"reference complete\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"T-3022 reference error: "<<e.what()<<'\n'; return 1; }
}
#endif
