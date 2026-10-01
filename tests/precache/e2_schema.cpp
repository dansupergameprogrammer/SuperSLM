#include "precache_common.h"

namespace {
using namespace pc;

std::string ArtifactDir() {
    const char* p=std::getenv("SUPERSLM_PRECACHE_E0_ARTIFACT_DIR");
    PKV_CHECK_MSG(p && *p,"SUPERSLM_PRECACHE_E0_ARTIFACT_DIR is unset");
    return p ? p : "";
}

void Parser213() {
    const auto dir=ArtifactDir();
    if(dir.empty()) return;
    for(const char* file : {"must-reject-two-name.sslm","must-reject-zero-hash.sslm"}) {
        ModelFile model(dir+"/"+file);
        PKV_CHECK_MSG(model.status==SSLM_ARTIFACT_REJECTED && model.model==nullptr,
            "2.13 PC-12 parser accepted %s or returned %d instead of SSLM_ARTIFACT_REJECTED",
            file,int(model.status));
    }
    ModelFile healthy(dir+"/must-accept-renamed-pair.sslm");
    Status(healthy.status,SSLM_OK,"2.13 renamed pair must map");
}
PKV_CELL("2.13/E2", "E1", Parser213);
PKV_CELL("7.15/E2", "E1", Parser213);
PKV_CELL("11.6/E2", "E1", Parser213);

std::string SyntheticDir() {
    const char* p=std::getenv("SUPERSLM_PRECACHE_SYNTHETIC_DIR");
    PKV_CHECK_MSG(p && *p,"SUPERSLM_PRECACHE_SYNTHETIC_DIR is unset");
    return p ? p : "";
}

void Schema67() {
    const auto dir=SyntheticDir();
    if(dir.empty()) return;
    ModelFile model(dir+"/synthetic0p5-three-schema.sslm");
    Status(model.status,SSLM_OK,"6.7 three-schema model map");
    if(model.status!=SSLM_OK) return;
    Fixture fx;
    fx.model=model.model;
    model.model=nullptr; // Fixture owns the mapped handle from here.
    fx.geo={24,2,64,4096};
    fx.vocab=256;
    PagePool pool(fx.model,128);
    Handles h;
    for(const char* name : {"schema_zero","schema_one","schema_two"}) {
        sslm_schema schema=nullptr;
        Status(sslm_schema_lookup(fx.model,name,&schema),SSLM_OK,"6.7 schema lookup");
        if(!schema) continue;
        sslm_prefix p=nullptr;
        Status(sslm_prefix_begin_budgeted(fx.model,&pool.pool,65,&p),SSLM_OK,"6.7 prefix begin");
        if(!p) continue;
        h.prefixes.push_back(p);
        Status(sslm_prefix_set_schema(p,schema),SSLM_OK,"6.7 schema bind");
        std::vector<int32_t> input(65,0); // first token advances the fixture DFA.
        Status(PrefixPrefillAll(fx.model,p,input,64),SSLM_OK,"6.7 schema prefill");
        Status(sslm_prefix_freeze(p),SSLM_OK,"6.7 freeze");
        const auto b=Save(p);
        if(!RequireBlob(b,"6.7")) return;
        sslm_prefix copy=nullptr;
        if(Restored(fx,&pool.pool,b,nullptr,h,&copy)) {
            PKV_CHECK_MSG(Save(copy)==b,"6.7 schema binding or walk changed on restore");
            TwinTokens(fx,&pool.pool,p,copy,16);
        }
    }
}
PKV_CELL("6.7/E2", "E1", Schema67);
PKV_CELL("8.4/E2", "E1", Schema67);

void ExistingSsb5_214() {
    Parser213(); // The hostile model must fail before an SSB5 can be saved against it.
    const auto dir=SyntheticDir();
    if(dir.empty()) return;
    ModelFile model(dir+"/synthetic0p5-three-schema.sslm");
    Status(model.status,SSLM_OK,"2.14 distinct-name model map");
    if(model.status!=SSLM_OK) return;
    Fixture fx;
    fx.model=model.model; model.model=nullptr;
    fx.geo={24,2,64,4096}; fx.vocab=256;
    PagePool pool(fx.model,600); // Two legacy whole-cap reservations coexist.
    sslm_schema schema=nullptr;
    Status(sslm_schema_lookup(fx.model,"schema_two",&schema),SSLM_OK,"2.14 index-2 lookup");
    sslm_seq original=nullptr;
    Status(sslm_seq_create(fx.model,&pool.pool,&original),SSLM_OK,"2.14 legacy sequence create");
    if(!original) return;
    Status(sslm_seq_set_schema(original,schema),SSLM_OK,"2.14 bind schema_two");
    std::vector<int32_t> input(17,0);
    Status(PrefillAll(fx.model,original,input,16),SSLM_OK,"2.14 advance index-2 walk");
    std::vector<uint8_t> blob;
    PKV_CHECK_MSG(SaveBlob(original,&blob) && IsMagic(blob,"SSB5"),"2.14 legacy save not SSB5");
    sslm_seq restored=nullptr;
    Status(sslm_seq_restore(fx.model,&pool.pool,blob.data(),blob.size(),&restored),
           SSLM_OK,"2.14 SSB5 restore");
    if(restored) {
        std::vector<int32_t> a,c;
        PKV_CHECK_EQ(DecodeN(fx.model,original,16,&a),SSLM_OK);
        PKV_CHECK_EQ(DecodeN(fx.model,restored,16,&c),SSLM_OK);
        PKV_CHECK_MSG(a==c,"2.14 index-2 SSB5 restore changed tokens");
        sslm_seq_release(restored);
    }
    sslm_seq_release(original);
}
PKV_CELL("2.14/E2", "E1", ExistingSsb5_214);
}
