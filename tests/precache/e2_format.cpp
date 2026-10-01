#include "precache_common.h"
#include <filesystem>
#include <sstream>

namespace {
using namespace pc;
namespace fs=std::filesystem;

void Independent712() {
    const char* fixture_dir=std::getenv("SUPERSLM_PRECACHE_SYNTHETIC_DIR");
    PKV_CHECK_MSG(fixture_dir && *fixture_dir,"7.12 synthetic fixture directory is unset");
    if(!fixture_dir || !*fixture_dir) return;
    ModelFile model(std::string(fixture_dir)+"/synthetic0p5-three-schema.sslm");
    Status(model.status,SSLM_OK,"7.12 three-schema fixture map");
    if(model.status!=SSLM_OK) return;
    Fixture fx;
    fx.model=model.model; model.model=nullptr;
    fx.geo={24,2,64,4096};
    fx.vocab=256;
    fx.sha=Sha(model.bytes.p,model.size);
    Scene sc;
    PagePool& pool=sc.AddPool(fx.model,128);
    constexpr int L=65;
    sslm_schema schema=nullptr;
    Status(sslm_schema_lookup(fx.model,"schema_two",&schema),SSLM_OK,
           "7.12 bound schema lookup");
    if(!schema) return;
    sslm_prefix prefix=nullptr;
    Status(sslm_prefix_begin_budgeted(fx.model,&pool.pool,L,&prefix),SSLM_OK,
           "7.12 prefix begin");
    if(!prefix) return;
    sc.h.prefixes.push_back(prefix);
    Status(sslm_prefix_set_schema(prefix,schema),SSLM_OK,"7.12 bind schema_two");
    const std::vector<int32_t> input(L,0); // the test-owned DFA reaches state 1 on token 0.
    Status(PrefixPrefillAll(fx.model,prefix,input,64),SSLM_OK,"7.12 prefill");
    Status(sslm_prefix_freeze(prefix),SSLM_OK,"7.12 freeze");
    const auto blob=Save(prefix);
    if(!RequireBlob(blob,"7.12 independent parser")) return;
    sslm_seq adopter=BudgetSeq(fx,&pool.pool,80,sc.h);
    if(!adopter) return;
    Status(sslm_seq_adopt_prefix(adopter,prefix),SSLM_OK,"7.12 SSB6 twin adopt");
    std::vector<uint8_t> ssb;
    PKV_CHECK_MSG(SaveBlob(adopter,&ssb) && IsMagic(ssb,"SSB6"),
                  "7.12 adopter did not provide an SSB6 field oracle");
    if(!IsMagic(ssb,"SSB6")) return;
    const auto table=PrefixTable(prefix);
    PKV_CHECK_EQ(table.size(),size_t((L+fx.B()-1)/fx.B()));
    if(table.size()!=size_t((L+fx.B()-1)/fx.B())) return;
    constexpr size_t hidden=896; // synthetic0p5 fixture's embedded hidden size.
    std::vector<uint8_t> rows(size_t(L)*fx.BytesPerToken());
    const size_t H=fx.geo.kv_heads,D=fx.geo.head_dim;
    std::map<uint32_t,std::vector<uint8_t>> pages;
    for(uint32_t l=0;l<fx.geo.layers;++l)
      for(uint32_t half=0;half<2;++half)
       for(uint32_t h=0;h<fx.geo.kv_heads;++h)
        for(int pos=0;pos<L;++pos) {
            const uint32_t page=table[size_t(pos/fx.B())];
            auto it=pages.find(page);
            if(it==pages.end()) it=pages.emplace(page,PeekPage(fx,pool.pool,page)).first;
            const size_t from=((size_t(l)*2+half)*H+h)*size_t(fx.B())*D+
                              size_t(pos%fx.B())*D;
            const size_t to=((size_t(l)*2+half)*H+h)*size_t(L)*D+size_t(pos)*D;
            PKV_CHECK_MSG(it->second.size()>=from+D,"7.12 page-index peek was short");
            if(it->second.size()<from+D) return;
            std::memcpy(rows.data()+to,it->second.data()+from,D);
        }
    const fs::path dir=fs::temp_directory_path()/"superslm-precache-e1-format";
    fs::create_directories(dir);
    auto write=[&](const char* name,const std::vector<uint8_t>& bytes) {
        const fs::path path=dir/name;
        std::ofstream file(path,std::ios::binary|std::ios::trunc);
        file.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));
        PKV_CHECK_MSG(bool(file),"7.12 failed to write %s",path.string().c_str());
        return path;
    };
    const fs::path blob_path=write("prefix.spx1",blob);
    const fs::path ssb_path=write("adopter.ssb6",ssb);
    const fs::path rows_path=write("peek-rows.bin",rows);
    std::ostringstream command;
    command << '"' << PC_PYTHON_PATH << '"' << " \"" << PC_FORMAT_SCRIPT_PATH
            << "\" --blob \"" << blob_path.string() << "\" --ssb \""
            << ssb_path.string() << "\" --rows \"" << rows_path.string()
            << "\" --cmake \"" << PC_CMAKE_PATH << "\" --model-sha " << fx.sha
            << " --length " << L << " --origin 0 --layers " << fx.geo.layers
            << " --heads " << fx.geo.kv_heads << " --dim " << fx.geo.head_dim
            << " --hidden " << hidden
            << " --current-token " << input.back()
            << " --schema-name schema_two --walk 1";
    PKV_CHECK_MSG(std::system(command.str().c_str())==0,
                  "7.12 independent struct/hashlib SPX1 parser rejected a field or row");
}
PKV_CELL("7.12/E2:independent-parser", "E1", Independent712);
}
