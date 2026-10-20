#include "precache_common.h"
#include "superslm/artifact.h"
#include "superslm/schema_masks.h"

namespace {
using namespace pc;

uint64_t ExpectedNameHash(const char* name) {
    uint64_t h=14695981039346656037ull;
    for(const unsigned char* p=reinterpret_cast<const unsigned char*>(name);*p;++p)
        h=(h^*p)*1099511628211ull;
    return h;
}

std::string ArtifactDir() {
    const char* p=std::getenv("SUPERSLM_PRECACHE_E0_ARTIFACT_DIR");
    PKV_CHECK_MSG(p && *p,"SUPERSLM_PRECACHE_E0_ARTIFACT_DIR is unset");
    return p ? p : "";
}

void Parser213() {
    const auto dir=ArtifactDir();
    if(dir.empty()) return;
    for(const auto& case_file : {std::pair{"must-reject-two-name.sslm","DuplicateSchemaNameHash"},
                                 std::pair{"must-reject-zero-hash.sslm","ZeroSchemaNameHash"}}) {
        const char* file=case_file.first;
        ModelFile model(dir+"/"+file);
        PKV_CHECK_MSG(model.status==SSLM_ARTIFACT_REJECTED && model.model==nullptr,
            "2.13 PC-12 parser accepted %s or returned %d instead of SSLM_ARTIFACT_REJECTED",
            file,int(model.status));
        if(!model.size) continue;
        superslm::SslmArtifact artifact;
        superslm::SslmError artifact_error;
        const auto artifact_status=superslm::SslmArtifact::OpenFromMemory(
            model.bytes.p,model.size,artifact,&artifact_error);
        PKV_CHECK_MSG(artifact_status==superslm::SslmStatus::Ok,
                      "2.13 independent artifact parse failed for %s",file);
        if(artifact_status!=superslm::SslmStatus::Ok) continue;
        const auto* section=artifact.Section(superslm::SslmSectionType::SchemaMasks);
        PKV_CHECK_MSG(section!=nullptr,"2.13 %s lacks its SCM1 section",file);
        if(!section || section->byte_size<24) continue;
        const uint8_t* data=section->data;
        const uint32_t vocab=uint32_t(data[12]) | (uint32_t(data[13])<<8) |
                             (uint32_t(data[14])<<16) | (uint32_t(data[15])<<24);
        superslm::SchemaMasksTable table;
        std::string reason;
        const bool parsed=superslm::SchemaMasksTable::Parse(
            data,section->byte_size,vocab,table,&reason);
        PKV_CHECK_MSG(!parsed && reason==case_file.second,
                      "2.13 %s: expected parser reason %s, got %s (accepted=%d)",
                      file,case_file.second,reason.c_str(),int(parsed));
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
    fx.stem="synthetic0p5-three-schema.sslm"; fx.ok=true;
    if(!PinFixtureB(&fx,16)) return;  // 1.13.0: the rule, not CFG1, selects B (precache_common.h)
    PagePool pool(fx.model,800); // two simultaneous legacy whole-cap adopters.
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
        Status(SchemaContentAll(fx.model,p,input,64),SSLM_OK,"6.7 schema prefill");
        Status(sslm_prefix_freeze(p),SSLM_OK,"6.7 freeze");
        const auto b=Save(p);
        if(!RequireBlob(b,"6.7")) return;
        auto hostile_walk=b;
        PutLe32(hostile_walk,104,UINT32_MAX);
        Digest(hostile_walk);
        sslm_prefix refused=nullptr;
        Status(Restore(fx,&pool.pool,hostile_walk,nullptr,&refused),
               SSLM_RESTORE_SCHEMA_MISMATCH,"2.3 bound walk outside DFA");
        PKV_CHECK(refused==nullptr);
        sslm_prefix copy=nullptr;
        if(Restored(fx,&pool.pool,b,nullptr,h,&copy)) {
            PKV_CHECK_MSG(Save(copy)==b,"6.7 schema binding or walk changed on restore");
            PKV_CHECK_MSG(Le64(b,96)==ExpectedNameHash(name) && Le32(b,104)==1,
                          "6.7 source prefix lacks its named hash and advanced walk");
            const char* adopter_names[]={"schema_zero","schema_one","schema_two",nullptr};
            for(const char* adopter_name : adopter_names) {
                for(bool budget : {false,true}) {
                    sslm_seq original=nullptr,adopted=nullptr;
                    const auto create=[&](sslm_seq* out) {
                        return budget ? sslm_seq_create_budgeted(fx.model,&pool.pool,80,out)
                                      : sslm_seq_create(fx.model,&pool.pool,out);
                    };
                    Status(create(&original),SSLM_OK,"6.7 prefill-twin adopter create");
                    Status(create(&adopted),SSLM_OK,"6.7 restored adopter create");
                    if(!original || !adopted) {
                        if(original) sslm_seq_release(original);
                        if(adopted) sslm_seq_release(adopted);
                        continue;
                    }
                    if(adopter_name) {
                        sslm_schema chosen=nullptr;
                        Status(sslm_schema_lookup(fx.model,adopter_name,&chosen),SSLM_OK,
                               "6.7 adopter schema lookup");
                        Status(sslm_seq_set_schema(original,chosen),SSLM_OK,
                               "6.7 prefill-twin adopter bind");
                        Status(sslm_seq_set_schema(adopted,chosen),SSLM_OK,
                               "6.7 restored adopter bind");
                    }
                    const auto want=adopter_name && std::strcmp(adopter_name,name)==0
                                  ? SSLM_OK : SSLM_PREFIX_SCHEMA_MISMATCH;
                    const auto a=sslm_seq_adopt_prefix(original,p);
                    const auto c=sslm_seq_adopt_prefix(adopted,copy);
                    PKV_CHECK_MSG(a==want && c==want,
                                  "6.7 prefix=%s adopter=%s mode=%s: expected disposition %d, got %d/%d",
                                  name,adopter_name ? adopter_name : "unbound",
                                  budget ? "budget" : "legacy",int(want),int(a),int(c));
                    if(a==SSLM_OK && c==SSLM_OK) {
                        if(budget) {
                            std::vector<uint8_t> ssb;
                            PKV_CHECK_MSG(SaveBlob(adopted,&ssb) && IsMagic(ssb,"SSB6"),
                                          "6.7 restored adopter did not save SSB6");
                            if(ssb.size()>=52) {
                                PKV_CHECK_EQ(Le64(ssb,40),ExpectedNameHash(name));
                                PKV_CHECK_EQ(Le32(ssb,48),1u);
                            }
                        }
                        std::vector<int32_t> x,y;
                        const auto sx=DecodeN(fx.model,original,64,&x);
                        const auto sy=DecodeN(fx.model,adopted,64,&y);
                        PKV_CHECK_MSG(sx==SSLM_OK && sy==SSLM_OK && x.size()==64 && x==y,
                                      "6.7 prefix=%s adopter=%s mode=%s 64-token continuation differs",
                                      name,adopter_name ? adopter_name : "unbound",
                                      budget ? "budget" : "legacy");
                    }
                    sslm_seq_release(adopted);
                    sslm_seq_release(original);
                }
            }
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
    fx.stem="synthetic0p5-three-schema.sslm"; fx.ok=true;
    if(!PinFixtureB(&fx,16)) return;  // 1.13.0: the rule, not CFG1, selects B (precache_common.h)
    PagePool pool(fx.model,600); // Two legacy whole-cap reservations coexist.
    sslm_schema schema=nullptr;
    Status(sslm_schema_lookup(fx.model,"schema_two",&schema),SSLM_OK,"2.14 index-2 lookup");
    sslm_seq original=nullptr;
    Status(sslm_seq_create(fx.model,&pool.pool,&original),SSLM_OK,"2.14 legacy sequence create");
    if(!original) return;
    Status(sslm_seq_set_schema(original,schema),SSLM_OK,"2.14 bind schema_two");
    std::vector<int32_t> input(17,0);
    Status(SeqSchemaContentAll(fx.model,original,input,16),SSLM_OK,
           "2.14 advance index-2 walk");
    std::vector<uint8_t> blob;
    PKV_CHECK_MSG(SaveBlob(original,&blob) && IsMagic(blob,"SSB5"),"2.14 legacy save not SSB5");
    sslm_seq restored=nullptr;
    Status(sslm_seq_restore(fx.model,&pool.pool,blob.data(),blob.size(),&restored),
           SSLM_OK,"2.14 SSB5 restore");
    if(restored) {
        std::vector<uint8_t> restored_blob;
        PKV_CHECK_MSG(SaveBlob(restored,&restored_blob) && IsMagic(restored_blob,"SSB5"),
                      "2.14 restored index-2 sequence did not save SSB5");
        if(restored_blob.size()>=52) {
            PKV_CHECK_EQ(Le64(restored_blob,40),ExpectedNameHash("schema_two"));
            PKV_CHECK_EQ(Le32(restored_blob,48),1u);
        }
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
