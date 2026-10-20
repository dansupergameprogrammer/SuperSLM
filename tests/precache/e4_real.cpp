#include "precache_provenance.h"
#include <chrono>
#include <cmath>
#include <numeric>
#include <filesystem>
#include <sstream>

namespace {
using namespace pc;
namespace fs=std::filesystem;

std::vector<int32_t> Ids(const fs::path& path) {
    std::ifstream f(path);
    PKV_CHECK_MSG(bool(f),"real prompt IDs missing: %s",path.string().c_str());
    std::vector<int32_t> ids;
    int32_t id=0;
    while(f>>id) ids.push_back(id);
    PKV_CHECK_MSG(ids.size()>=128,"real prompt %s has %zu IDs, needs >=128",
                  path.string().c_str(),ids.size());
    return ids;
}

std::vector<int32_t> DecodeReal(const Fixture& fx,sslm_kv_pool* pool,
    sslm_prefix prefix,bool budget,const char* mode) {
    sslm_seq seq=nullptr;
    const sslm_status create=budget ? sslm_seq_create_budgeted(fx.model,pool,80,&seq)
                                    : sslm_seq_create(fx.model,pool,&seq);
    Status(create,SSLM_OK,"10.1 real adopter create");
    if(!seq) return {};
    Status(sslm_seq_adopt_prefix(seq,prefix),SSLM_OK,"10.1 real adopter adopt");
    sslm_decode_params params{};
    Status(sslm_decode_params_init(fx.model,
        std::strcmp(mode,"damped")==0 ? SSLM_DECODE_MODE_DAMPED_GREEDY : SSLM_DECODE_MODE_GREEDY,
        int32_t(fx.geo.layers),&params),SSLM_OK,"10.1 decode mode init");
    sslm_seq batch[1]={seq};
    std::vector<int32_t> out;
    for(int i=0;i<64;++i) {
        int32_t token=-1;
        const sslm_status st=sslm_decode_step_v2(fx.model,batch,1,&params,nullptr,&token);
        Status(st,SSLM_OK,"10.1 real decode");
        if(st!=SSLM_OK || token<0) break;
        out.push_back(token);
    }
    sslm_seq_release(seq);
    return out;
}

void Real101() {
    const auto model_path=Env("SUPERSLM_PRECACHE_REAL_MODEL");
    const auto prompts=Env("SUPERSLM_PRECACHE_PROMPT_DIR");
    const auto blob_dir=Env("SUPERSLM_PRECACHE_BLOB_DIR");
    if(model_path.empty() || prompts.empty() || blob_dir.empty()) return;
    ModelFile model(model_path);
    Status(model.status,SSLM_OK,"10.1 real model map");
    if(model.status!=SSLM_OK) return;
    const auto model_sha=Sha(model.bytes.p,model.size);
    Fixture fx;
    fx.model=model.model; model.model=nullptr;
    fx.geo={24,2,64,32768}; fx.vocab=151936;
    PKV_CHECK_MSG(sslm_kv_block_size(fx.model)==size_t(fx.geo.context_cap)*fx.BytesPerToken() &&
                  sslm_kv_page_size(fx.model)==size_t(16)*fx.BytesPerToken(),
                  "10.1 real 0.5B artifact geometry differs from receipt");
    // One flat prefix is live at a time. The legacy adopter reserves all
    // cap/B pages before adoption; add the largest prompt's R(L) and R(80).
    int64_t largest_prefix=0;
    for(int i=0;i<10;++i) {
        char name[16]; std::snprintf(name,sizeof(name),"flat%02d",i);
        const auto ids=Ids(fs::path(prompts)/(std::string(name)+".ids"));
        if(ids.empty()) return;
        largest_prefix=std::max(largest_prefix,fx.R(int64_t(ids.size())));
    }
    PagePool pool(fx.model,uint32_t(fx.CapPages()+largest_prefix+fx.R(80)));
    Handles h;
    const std::string phase=Env("SUPERSLM_PRECACHE_PHASE");
    PKV_CHECK_MSG(phase=="capture" || phase=="restore",
                  "10.1 phase must name capture or restore");
    if(phase!="capture" && phase!="restore") return;
    int graded=0;
    for(int i=0;i<10;++i) {
        char name[16]; std::snprintf(name,sizeof(name),"flat%02d",i);
        const fs::path input=fs::path(prompts)/(std::string(name)+".ids");
        const auto ids=Ids(input);
        if(ids.empty()) return;
        sslm_prefix p=nullptr;
        if(phase=="capture") {
            Status(sslm_prefix_begin_budgeted(fx.model,&pool.pool,int32_t(ids.size()),&p),
                   SSLM_OK,"10.1 real prefix begin");
            if(!p) return;
            h.prefixes.push_back(p);
            Status(PrefixPrefillAll(fx.model,p,ids,64),SSLM_OK,"10.1 real prefill");
            Status(sslm_prefix_freeze(p),SSLM_OK,"10.1 real freeze");
            auto b=Save(p);
            if(!RequireBlob(b,"10.1")) return;
            fs::create_directories(blob_dir);
            std::ofstream out(fs::path(blob_dir)/(std::string(name)+".spx1"),std::ios::binary);
            out.write(reinterpret_cast<const char*>(b.data()),std::streamsize(b.size()));
        } else {
            std::vector<uint8_t> b;
            PKV_CHECK_MSG(ReadAll((fs::path(blob_dir)/(std::string(name)+".spx1")).string(),&b),
                          "10.1 missing captured blob %s",name);
            if(b.empty()) return;
            if(!Restored(fx,&pool.pool,b,nullptr,h,&p)) return;
            for(int a=0;a<5;++a) {
                const bool budget=(a%2)==0;
                const char* mode=(a<2 || a==4) ? "greedy" : "damped";
                const auto got=DecodeReal(fx,&pool.pool,p,budget,mode);
                const auto want=Reference("P1-real0p5-flat",name,input,model_sha,a,mode);
                PKV_CHECK_MSG(got==want && got.size()==64,"10.1 %s adopter %d differs from pinned prefill",name,a);
                if(got==want && got.size()==64) ++graded;
            }
        }
        Status(ReleasePrefix(h,p),SSLM_OK,"10.1 release prefix before next cohort member");
    }
    if(phase=="restore") {
        PKV_CHECK_EQ(graded,50);
        PKV_CHECK_EQ(sslm_pc_test_only_prefix_prefill_calls(),0u);
    }
}
PKV_CELL("10.1/E4", "E1", Real101);

void CrossCompiler43() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    const auto b=RootBlob(fx,1000);
    if(!RequireBlob(b,"4.3")) return;
    const auto directory=Env("SUPERSLM_PRECACHE_CROSS_COMPILER_DIR");
    if(directory.empty()) return;
    const auto local=LocalCompilerReceipt(fx,Stream(41,1000,fx.vocab),b);
    if(local.at("runner_sha256").empty()) return;
    const fs::path dir(directory);
    const std::string phase=Env("SUPERSLM_PRECACHE_COMPILER_PHASE");
    if(phase=="capture") {
        fs::create_directories(dir);
        const std::string compiler=local.at("compiler");
        PKV_CHECK_MSG(compiler=="MSVC" || compiler=="GNU" || compiler=="Clang",
                      "4.3 capture compiler identity is unsupported: %s",compiler.c_str());
        PKV_CHECK_MSG(WriteReceipt(dir/(compiler+".receipt"),local),
                      "4.3 compiler receipt write failed");
        std::ofstream output(dir/(compiler+".spx1"),std::ios::binary);
        output.write(reinterpret_cast<const char*>(b.data()),std::streamsize(b.size()));
        PKV_CHECK_MSG(bool(output),"4.3 compiler blob write failed");
        return;
    }
    PKV_CHECK_MSG(phase=="compare","4.3 compiler phase must be capture or compare");
    if(phase!="compare") return;
      std::set<std::string> runners;
      std::set<std::string> source_revisions;
    for(const char* compiler:{"MSVC","GNU","Clang"}) {
        const auto r=ReadReceipt(dir/(std::string(compiler)+".receipt"));
          PKV_CHECK_MSG(ValidCompilerJob(dir,r,local,compiler),
                        "4.3 %s receipt lacks its hashed producer binary, compiler job, or generated blob",
                        compiler);
          if(!ValidCompilerJob(dir,r,local,compiler)) continue;
          runners.insert(r.at("runner_sha256"));
          source_revisions.insert(ReadCompilerJob(dir/(std::string(compiler)+".job")).at("source_revision"));
        std::vector<uint8_t> cloud;
        PKV_CHECK_MSG(ReadAll((dir/(std::string(compiler)+".spx1")).string(),&cloud) &&
                      cloud==b && Sha(cloud.data(),cloud.size())==r.at("blob_sha256"),
                      "4.3 %s blob bytes or digest differ",compiler);
        if(r.at("compiler")==local.at("compiler"))
            PKV_CHECK_MSG(r==local,
                          "4.3 local build receipt differs from its compiler job");
    }
      PKV_CHECK_MSG(runners.size()==3,
                    "4.3 renamed or copied compiler receipt lacks three distinct built runners");
      PKV_CHECK_MSG(source_revisions.size()==1,
                    "4.3 compiler jobs were not built from one source revision");
}
PKV_CELL("4.3/E4", "E1", CrossCompiler43);

void Timing103() {
    const auto model_path=Env("SUPERSLM_PRECACHE_REAL_MODEL");
    const auto prompts=Env("SUPERSLM_PRECACHE_PROMPT_DIR");
    if(model_path.empty() || prompts.empty()) return;
    ModelFile model(model_path);
    Status(model.status,SSLM_OK,"10.3 real model map");
    if(model.status!=SSLM_OK) return;
    Fixture fx;
    fx.model=model.model; model.model=nullptr;
    fx.geo={24,2,64,32768}; fx.vocab=151936;
    PKV_CHECK_MSG(sslm_kv_block_size(fx.model)==size_t(fx.geo.context_cap)*fx.BytesPerToken() &&
                  sslm_kv_page_size(fx.model)==size_t(16)*fx.BytesPerToken(),
                  "10.3 real 0.5B artifact geometry differs from receipt");
    PagePool pool(fx.model,300);
    const auto receipt_path=Env("SUPERSLM_PRECACHE_TIMING_RECEIPT");
    const auto instrument_path=Env("SUPERSLM_PRECACHE_TIMING_INSTRUMENT");
    const auto prompt_manifest=Env("SUPERSLM_PRECACHE_TIMING_PROMPT_MANIFEST");
    const auto box=Env("SUPERSLM_PRECACHE_BOX_ID");
    const auto threads=Env("SUPERSLM_PRECACHE_THREADS");
    const auto receipt=LicenseFields(receipt_path);
    const auto field=[&](const char* name) {
        const auto it=receipt.find(name);
        return it==receipt.end() ? std::string{} : it->second;
    };
    const bool commissioned=receipt.size()==13 &&
        field("format")=="PC-TIMING-1" && field("status")=="COMMISSIONED" &&
        field("commissioning_verdict")=="PASS" && field("must_accept")=="PASS" &&
        field("must_reject")=="PASS" && field("threshold")=="1000" &&
        field("box_id")==box && field("threads")==threads &&
        field("threads")=="1" &&
        field("geometry")=="0p5:24x2x64x32768;1p5:28x2x128x32768;B=16" &&
        field("model_sha256")==Sha(model.bytes.p,model.size) &&
        field("prompt_manifest_sha256")==ShaFile(prompt_manifest) &&
        field("instrument_sha256")==ShaFile(instrument_path) &&
        field("anchor")==Env("SUPERSLM_PRECACHE_TIMING_ANCHOR");
    PKV_CHECK_MSG(commissioned,"10.3 timing receipt identity or independent commission is invalid");
    if(!commissioned) return;
    const auto receipt_sha=ShaFile(receipt_path);
    constexpr int kPairs=8;
    constexpr double kStudent95=2.365; // two-sided t_0.975,7.
    const double tick_ms=1000.0*double(std::chrono::steady_clock::period::num)/
                         double(std::chrono::steady_clock::period::den);
    for(int i=0;i<10;++i) {
        char name[16]; std::snprintf(name,sizeof(name),"flat%02d",i);
        const auto ids=Ids(fs::path(prompts)/(std::string(name)+".ids"));
        if(ids.empty()) return;
        std::vector<double> logs;
        double prefill_sum=0,restore_sum=0;
        size_t bytes=0;
        for(int j=0;j<kPairs;++j) {
            sslm_prefix built=nullptr;
            const auto start=std::chrono::steady_clock::now();
            Status(sslm_prefix_begin_budgeted(fx.model,&pool.pool,int32_t(ids.size()),&built),
                   SSLM_OK,"10.3 paired prefill begin");
            if(!built) return;
            Status(PrefixPrefillAll(fx.model,built,ids,64),SSLM_OK,"10.3 paired prefill");
            Status(sslm_prefix_freeze(built),SSLM_OK,"10.3 paired freeze");
            const auto prefill_end=std::chrono::steady_clock::now();
            const auto b=Save(built);
            if(!RequireBlob(b,"10.3 paired")) {sslm_prefix_release(built);return;}
            bytes=b.size();
            sslm_prefix_release(built);
            sslm_prefix restored=nullptr;
            const auto r0=std::chrono::steady_clock::now();
            const sslm_status st=Restore(fx,&pool.pool,b,nullptr,&restored);
            const auto r1=std::chrono::steady_clock::now();
            Status(st,SSLM_OK,"10.3 paired restore including digest");
            if(st!=SSLM_OK) return;
            sslm_prefix_release(restored);
            const double prefill_ms=std::chrono::duration<double,std::milli>(prefill_end-start).count();
            const double restore_ms=std::chrono::duration<double,std::milli>(r1-r0).count();
            PKV_CHECK_MSG(prefill_ms>100*tick_ms && restore_ms>100*tick_ms,
                          "10.3 %s pair %d falls below timer resolving floor",name,j);
            if(prefill_ms<=0 || restore_ms<=0) return;
            prefill_sum+=prefill_ms;
            restore_sum+=restore_ms;
            logs.push_back(std::log(prefill_ms/restore_ms));
        }
        const double mean=std::accumulate(logs.begin(),logs.end(),0.0)/kPairs;
        double squared=0;
        for(double value:logs) squared+=(value-mean)*(value-mean);
        const double half=kStudent95*std::sqrt(squared/(kPairs-1)/kPairs);
        const double lower=std::exp(mean-half),upper=std::exp(mean+half);
        std::printf("10.3 receipt=%s box=%s threads=%s geometry=0p5 B=16 %s pairs=%d bytes_including_digest=%zu prefill_ms=%.3f restore_ms=%.3f ratio95=[%.1f,%.1f]\n",
                    receipt_sha.c_str(),box.c_str(),threads.c_str(),name,kPairs,bytes,
                    prefill_sum/kPairs,restore_sum/kPairs,lower,upper);
        PKV_CHECK_MSG(lower>=1000,
                      "10.3 %s threshold unresolved or missed: 95%% ratio interval [%.1f,%.1f]",
                      name,lower,upper);
    }
    const auto worst_path=Env("SUPERSLM_PRECACHE_REAL_1P5_MODEL");
    if(worst_path.empty()) return;
    ModelFile worst(worst_path);
    Status(worst.status,SSLM_OK,"10.3 1.5B worst-setting map");
    if(worst.status!=SSLM_OK) return;
    Fixture wf;
    wf.model=worst.model; worst.model=nullptr;
    wf.geo={28,2,128,32768}; wf.vocab=151936;
    PKV_CHECK_MSG(sslm_kv_block_size(wf.model)==size_t(wf.geo.context_cap)*wf.BytesPerToken() &&
                  sslm_kv_page_size(wf.model)==size_t(16)*wf.BytesPerToken(),
                  "10.3 real 1.5B artifact geometry differs from receipt");
    PagePool worst_pool(wf.model,uint32_t(wf.CapPages()+4));
    Status(worst_pool.status,SSLM_OK,"10.3 cap-length pool");
    if(worst_pool.status!=SSLM_OK) return;
    sslm_prefix cap_prefix=nullptr;
    const auto cap_start=std::chrono::steady_clock::now();
    Status(sslm_prefix_begin_budgeted(wf.model,&worst_pool.pool,32768,&cap_prefix),
           SSLM_OK,"10.3 cap-length begin");
    if(!cap_prefix) return;
    Status(PrefixPrefillAll(wf.model,cap_prefix,Stream(41,32768,wf.vocab),64),
           SSLM_OK,"10.3 cap-length prefill");
    Status(sslm_prefix_freeze(cap_prefix),SSLM_OK,"10.3 cap-length freeze");
    const auto cap_prefill_end=std::chrono::steady_clock::now();
    const auto cap_blob=Save(cap_prefix);
    if(!RequireBlob(cap_blob,"10.3 cap-length")) {sslm_prefix_release(cap_prefix);return;}
    PKV_CHECK_MSG(cap_blob.size()==kHeader+ArtifactHiddenSize(wf)+
                  size_t(32768)*wf.BytesPerToken(),
                  "10.3 cap-length 1.5B byte count omits header, digest or canonical rows");
    sslm_prefix_release(cap_prefix);
    double cap_restore_ms=0;
    for(int i=0;i<3;++i) {
        sslm_prefix restored=nullptr;
        const auto begin=std::chrono::steady_clock::now();
        const auto st=Restore(wf,&worst_pool.pool,cap_blob,nullptr,&restored);
        const auto end=std::chrono::steady_clock::now();
        Status(st,SSLM_OK,"10.3 cap-length restore");
        if(st!=SSLM_OK) return;
        cap_restore_ms+=std::chrono::duration<double,std::milli>(end-begin).count();
        sslm_prefix_release(restored);
    }
    cap_restore_ms/=3.0;
    const double cap_prefill_ms=std::chrono::duration<double,std::milli>(cap_prefill_end-cap_start).count();
    const double mib=double(cap_blob.size())/(1024.0*1024.0);
    std::printf("10.3 receipt=%s box=%s threads=%s worst cap=32768 geometry=1p5 28x2x128 B=16 bytes_including_digest=%zu prefill_ms=%.3f restore_ms=%.3f MiB=%.3f MiB/s=%.1f\n",
                receipt_sha.c_str(),box.c_str(),threads.c_str(),cap_blob.size(),
                cap_prefill_ms,cap_restore_ms,mib,1000.0*mib/cap_restore_ms);
}
PKV_CELL("10.3/E4", "E1", Timing103);

void Real102() {
    const auto model_path=Env("SUPERSLM_PRECACHE_REAL_MODEL");
    const auto prompts=Env("SUPERSLM_PRECACHE_PROMPT_DIR");
    const auto blob_dir=Env("SUPERSLM_PRECACHE_BLOB_DIR");
    if(model_path.empty() || prompts.empty() || blob_dir.empty()) return;
    ModelFile model(model_path);
    Status(model.status,SSLM_OK,"10.2 real model map");
    if(model.status!=SSLM_OK) return;
    const auto model_sha=Sha(model.bytes.p,model.size);
    Fixture fx;
    fx.model=model.model; model.model=nullptr;
    fx.geo={24,2,64,32768}; fx.vocab=151936;
    PKV_CHECK_MSG(sslm_kv_block_size(fx.model)==size_t(fx.geo.context_cap)*fx.BytesPerToken() &&
                  sslm_kv_page_size(fx.model)==size_t(16)*fx.BytesPerToken(),
                  "10.2 real 0.5B artifact geometry differs from receipt");
    const auto world_ids=Ids(fs::path(prompts)/"world.ids");
    if(world_ids.empty()) return;
    // World and all ten persona children remain live through the loop. A
    // hinted restore can fall back to a private full joined prefix before
    // the sharing oracle rejects it, and the legacy adopter reserves cap/B.
    int64_t retained_prefixes=fx.R(int64_t(world_ids.size()));
    for(int i=0;i<10;++i) {
        char persona[16]; std::snprintf(persona,sizeof(persona),"persona%02d",i);
        const auto suffix=Ids(fs::path(prompts)/(std::string(persona)+".ids"));
        if(suffix.empty()) return;
        retained_prefixes+=fx.R(int64_t(world_ids.size()+suffix.size()));
    }
    PagePool pool(fx.model,uint32_t(fx.CapPages()+retained_prefixes+fx.R(80)));
    Handles h;
    const auto phase=Env("SUPERSLM_PRECACHE_PHASE");
    PKV_CHECK_MSG(phase=="capture" || phase=="restore",
                  "10.2 phase must name capture or restore");
    if(phase!="capture" && phase!="restore") return;
    int graded=0;
    sslm_prefix world=nullptr;
    if(phase=="capture") {
        Status(sslm_prefix_begin_budgeted(fx.model,&pool.pool,int32_t(world_ids.size()),&world),
               SSLM_OK,"10.2 world begin");
        if(!world) return;
        h.prefixes.push_back(world);
        Status(PrefixPrefillAll(fx.model,world,world_ids,64),SSLM_OK,"10.2 world prefill");
        Status(sslm_prefix_freeze(world),SSLM_OK,"10.2 world freeze");
        auto b=Save(world);
        if(!RequireBlob(b,"10.2")) return;
        fs::create_directories(blob_dir);
        std::ofstream file(fs::path(blob_dir)/"world.spx1",std::ios::binary);
        file.write(reinterpret_cast<const char*>(b.data()),std::streamsize(b.size()));
    } else {
        std::vector<uint8_t> b;
        PKV_CHECK_MSG(ReadAll((fs::path(blob_dir)/"world.spx1").string(),&b),
                      "10.2 world SPX1 missing after process restart");
        if(b.empty() || !Restored(fx,&pool.pool,b,nullptr,h,&world)) return;
    }
    for(int i=0;i<10;++i) {
        char name[16],persona[16];
        std::snprintf(name,sizeof(name),"nested%02d",i);
        std::snprintf(persona,sizeof(persona),"persona%02d",i);
        const auto suffix=Ids(fs::path(prompts)/(std::string(persona)+".ids"));
        if(suffix.empty()) return;
        sslm_prefix child=nullptr;
        if(phase=="capture") {
            Status(sslm_prefix_begin_from(world,int32_t(suffix.size()),&child),
                   SSLM_OK,"10.2 persona begin_from");
            if(!child) return;
            h.prefixes.push_back(child);
            Status(PrefixPrefillAll(fx.model,child,suffix,64),SSLM_OK,"10.2 persona prefill");
            Status(sslm_prefix_freeze(child),SSLM_OK,"10.2 persona freeze");
            const auto b=Save(child);
            if(!RequireBlob(b,"10.2 nested")) return;
            std::ofstream out(fs::path(blob_dir)/(std::string(name)+".spx1"),std::ios::binary);
            out.write(reinterpret_cast<const char*>(b.data()),std::streamsize(b.size()));
        } else {
            std::vector<uint8_t> b;
            PKV_CHECK_MSG(ReadAll((fs::path(blob_dir)/(std::string(name)+".spx1")).string(),&b),
                          "10.2 persona blob %s missing",name);
            if(b.empty()) return;
            uint32_t shared=0;
            if(!Restored(fx,&pool.pool,b,world,h,&child,&shared)) return;
            PKV_CHECK_EQ(shared,uint32_t(world_ids.size()/16));
            const auto input=fs::path(prompts)/(std::string(name)+".ids");
            for(int a=0;a<5;++a) {
                const bool budget=(a%2)==0;
                const char* mode=(a<2 || a==4) ? "greedy" : "damped";
                const auto got=DecodeReal(fx,&pool.pool,child,budget,mode);
                const auto want=Reference("P2-real0p5-nested",name,input,model_sha,a,mode);
                PKV_CHECK_MSG(got==want && got.size()==64,
                              "10.2 %s adopter %d differs from pinned prefill",name,a);
                if(got==want && got.size()==64) ++graded;
            }
        }
    }
    if(phase=="restore") PKV_CHECK_EQ(graded,50);
}
PKV_CELL("10.2/E4", "E1", Real102);

struct HookCounts { std::atomic<int> calls{0},tasks{0}; };
void Hook(void* user,int32_t n,sslm_task_fn task,void* ctx) {
    auto& counts=*static_cast<HookCounts*>(user);
    ++counts.calls;
    std::thread worker([&]{for(int32_t i=1;i<n;i+=2) {++counts.tasks;task(ctx,i);}});
    for(int32_t i=0;i<n;i+=2) {++counts.tasks;task(ctx,i);}
    worker.join();
}

void Parallel66() {
    const Fixture& fx=GetFixture("pkv_def");
    if(!fx.ok) return;
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,128);
    sslm_prefix original=Root(fx,&pool.pool,sc.h,65);
    if(!original) return;
    const auto b=Save(original);
    if(!RequireBlob(b,"6.6")) return;
    sslm_prefix restored=nullptr;
    if(!Restored(fx,&pool.pool,b,nullptr,sc.h,&restored)) return;
    sslm_seq serial=BudgetSeq(fx,&pool.pool,80,sc.h);
    sslm_seq parallel=BudgetSeq(fx,&pool.pool,80,sc.h);
    if(!serial || !parallel) return;
    Status(sslm_seq_adopt_prefix(serial,restored),SSLM_OK,"6.6 serial adopt");
    Status(sslm_seq_adopt_prefix(parallel,restored),SSLM_OK,"6.6 parallel adopt");
    const sslm_config cfg{1,64,int32_t(fx.geo.layers),0};
    AlignedBuf memory(sslm_workspace_size(fx.model,&cfg));
    sslm_workspace ws=nullptr;
    Status(sslm_workspace_create(fx.model,&cfg,memory.p,memory.n,&ws),
           SSLM_OK,"6.6 workspace create");
    if(!ws) return;
    HookCounts counts;
    const sslm_parallel_for hook{Hook,&counts,4,SSLM_PARALLEL_FOR_MATVEC};
    Status(sslm_workspace_set_parallel_for(ws,&hook),SSLM_OK,"6.6 hook install");
    sslm_decode_params p{};
    p.layer_budget=int32_t(fx.geo.layers);
    for(int i=0;i<32;++i) {
        sslm_seq a[1]={serial},c[1]={parallel};
        int32_t ta=-1,tc=-1;
        Status(sslm_decode_step(fx.model,a,1,&p,nullptr,&ta),SSLM_OK,"6.6 serial decode");
        Status(sslm_decode_step(fx.model,c,1,&p,ws,&tc),SSLM_OK,"6.6 parallel decode");
        PKV_CHECK_MSG(ta>=0 && ta==tc,"6.6 restored parallel token differs at %d",i);
    }
    PKV_CHECK_MSG(counts.calls>0 && counts.tasks>0,
                  "6.6 installed parallel hook never ran (calls=%d, tasks=%d)",
                  int(counts.calls),int(counts.tasks));
    Status(sslm_workspace_destroy(ws),SSLM_OK,"6.6 workspace destroy");
}
PKV_CELL("6.6/E4", "E1", Parallel66);
}
