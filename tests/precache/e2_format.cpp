#include "precache_common.h"
#include "paged-kv/pkv_legacy_a_helpers.h"
#include <filesystem>
#include <sstream>

namespace {
using namespace pc;
namespace fs=std::filesystem;

// Engine 5d5ca58: artifact.h kIntegrityHashOffset=32, artifact.cpp hashes
// the file with [32,64) zeroed, and RawIntegrityHash() returns those bytes.
bool VerifiedArtifactIntegrity(const uint8_t* data,size_t size,
                               std::array<uint8_t,32>* stored) {
    if(!data || size<64) return false;
    std::copy(data+32,data+64,stored->begin());
    std::vector<uint8_t> zeroed(data,data+size);
    std::fill(zeroed.begin()+32,zeroed.begin()+64,uint8_t{0});
    static constexpr char hex[]="0123456789abcdef";
    std::string expected;
    expected.reserve(64);
    for(uint8_t byte:*stored) {
        expected.push_back(hex[byte>>4]);
        expected.push_back(hex[byte&15]);
    }
    return expected==Sha(zeroed.data(),zeroed.size());
}

void Independent712() {
    const std::array<uint64_t,5> fixture_counts{11,13,17,19,23};
    PKV_CHECK_MSG(std::all_of(fixture_counts.begin(),fixture_counts.end(),
                  [](uint64_t n){return n>0;}) &&
                  std::set<uint64_t>(fixture_counts.begin(),fixture_counts.end()).size()==5,
                  "7.12 independent count fixture is not pairwise distinct and nonzero");
    const char* fixture_dir=std::getenv("SUPERSLM_PRECACHE_SYNTHETIC_DIR");
    PKV_CHECK_MSG(fixture_dir && *fixture_dir,"7.12 synthetic fixture directory is unset");
    if(!fixture_dir || !*fixture_dir) return;
    ModelFile model(std::string(fixture_dir)+"/synthetic0p5-three-schema.sslm");
    Status(model.status,SSLM_OK,"7.12 three-schema fixture map");
    if(model.status!=SSLM_OK) return;
    std::array<uint8_t,32> model_integrity{};
    const bool integrity=VerifiedArtifactIntegrity(model.bytes.p,model.size,&model_integrity);
    PKV_CHECK_MSG(integrity,"7.12 model stored integrity hash fails zeroed-field digest");
    if(!integrity) return;
    Fixture fx;
    fx.model=model.model; model.model=nullptr;
    fx.geo={24,2,64,4096};
    fx.vocab=256;
    fx.sha=Sha(model.bytes.p,model.size);
    fx.stem="synthetic0p5-three-schema.sslm"; fx.ok=true;
    if(!PinFixtureB(&fx,16)) return;  // 1.13.0: the rule, not CFG1, selects B (precache_common.h)
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
    Status(SchemaContentAll(fx.model,prefix,input,64),SSLM_OK,"7.12 schema-content prefill");
    Status(sslm_prefix_freeze(prefix),SSLM_OK,"7.12 freeze");
    auto seeded=Save(prefix);
    if(!RequireBlob(seeded,"7.12 count seed")) return;
    for(size_t i=0;i<fixture_counts.size();++i)
        PutLe64(seeded,144+8*i,fixture_counts[i]);
    Digest(seeded);
    sslm_prefix counted=nullptr;
    if(!Restored(fx,&pool.pool,seeded,nullptr,sc.h,&counted)) return;
    prefix=counted;
    const auto blob=Save(prefix);
    if(!RequireBlob(blob,"7.12 independent parser")) return;
    sslm_seq unbound=BudgetSeq(fx,&pool.pool,80,sc.h);
    if(!unbound) return;
    Status(sslm_seq_adopt_prefix(unbound,prefix),SSLM_PREFIX_SCHEMA_MISMATCH,
           "7.12 unbound adopter rejects schema progress");
    sslm_seq adopter=BudgetSeq(fx,&pool.pool,80,sc.h);
    if(!adopter) return;
    Status(sslm_seq_set_schema(adopter,schema),SSLM_OK,
           "7.12 bind SSB6 adopter to prefix schema");
    Status(sslm_seq_adopt_prefix(adopter,prefix),SSLM_OK,"7.12 SSB6 twin adopt");
    std::vector<uint8_t> ssb;
    PKV_CHECK_MSG(SaveBlob(adopter,&ssb) && IsMagic(ssb,"SSB6"),
                  "7.12 adopter did not provide an SSB6 field oracle");
    if(!IsMagic(ssb,"SSB6")) return;
    const auto table=PrefixTable(prefix);
    PKV_CHECK_EQ(table.size(),size_t((L+fx.B()-1)/fx.B()));
    if(table.size()!=size_t((L+fx.B()-1)/fx.B())) return;
    const size_t hidden=ArtifactHiddenSize(model.bytes.p,model.size);
    if(!hidden) return;
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
            // 1.13.0's rotated in-page address (paged-KV §3.1, T1), through pkv::PageRunOffset.
            const size_t from=PageRunOffset(fx.geo.layers,fx.geo.kv_heads,D,fx.B(),l,half,h,pos);
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
            << "\" --cmake \"" << PC_CMAKE_PATH << "\" --model-artifact \""
            << (fs::path(fixture_dir)/"synthetic0p5-three-schema.sslm").string() << '"'
            << " --length " << L << " --origin 0 --layers " << fx.geo.layers
            << " --heads " << fx.geo.kv_heads << " --dim " << fx.geo.head_dim
            << " --hidden " << hidden
            << " --current-token " << input.back()
            << " --schema-name schema_two --walk 1"
            << " --counts 11,13,17,19,23";
    const auto run_parser=[](const std::string& args) {
#ifdef _WIN32
        // The Windows CRT passes this to cmd /c; an executable path quoted
        // alongside quoted arguments needs one outer quote pair as well.
        return std::system(("\""+args+"\"").c_str());
#else
        return std::system(args.c_str());
#endif
    };
    const bool accepted=run_parser(command.str())==0;
    PKV_CHECK_MSG(accepted,
                  "7.12 independent struct/hashlib SPX1 parser rejected a field or row");
    if(!accepted) return;
    auto swapped_blob=blob,swapped_ssb=ssb;
    PutLe64(swapped_blob,144,fixture_counts[1]);
    PutLe64(swapped_blob,152,fixture_counts[0]);
    Digest(swapped_blob);
    PutLe64(swapped_ssb,92,fixture_counts[1]);
    PutLe64(swapped_ssb,124,fixture_counts[0]);
    const auto swap_blob_path=write("swapped.spx1",swapped_blob);
    const auto swap_ssb_path=write("swapped.ssb6",swapped_ssb);
    std::string swap_command=command.str();
    const auto replace_once=[&](const std::string& from,const std::string& to) {
        const size_t at=swap_command.find(from);
        if(at!=std::string::npos) swap_command.replace(at,from.size(),to);
        return at!=std::string::npos;
    };
    const bool swapped_paths=replace_once(blob_path.string(),swap_blob_path.string()) &&
                             replace_once(ssb_path.string(),swap_ssb_path.string());
    swap_command += " --expect-error \"known saturation counts\"";
    PKV_CHECK_MSG(swapped_paths && run_parser(swap_command)==0,
                  "7.12 symmetric SPX1/SSB6 count swap did not fail for known saturation counts");
}
PKV_CELL("7.12/E2:independent-parser", "E1", Independent712);

void CrossView712() {
    const std::array<uint64_t,5> counts{11,13,17,19,23};
    const Fixture& sixteen=SyntheticFixture();
    if(!sixteen.ok) return;
    Fixture four;
    PKV_CHECK_MSG(pkv::legacy_a::PatchedBlockSizeTwin(sixteen,4,&four),
                  "7.12 B=4 model view could not be constructed");
    if(!four.ok || !PinFixtureB(&four,4)) return;
    std::array<uint8_t,32> sixteen_integrity{},four_integrity{};
    const bool verified_sixteen=VerifiedArtifactIntegrity(sixteen.bytes.p,sixteen.size,
                                                          &sixteen_integrity);
    const bool verified_four=VerifiedArtifactIntegrity(four.bytes.p,four.size,&four_integrity);
    PKV_CHECK_MSG(verified_sixteen && verified_four,
                  "7.12 B-view artifact stored integrity fails zeroed-field digest");
    if(!verified_sixteen || !verified_four) return;
    PKV_CHECK_MSG(sixteen_integrity!=four_integrity && sixteen.B()==16 && four.B()==4,
                  "7.12 model views need distinct hashes and B=16/B=4");
    constexpr int L=65;
    struct View {std::vector<uint8_t> ssb,rows,blob;};
    auto build=[&](const Fixture& fx) {
        View v;
        Scene sc;
        PagePool& pool=sc.AddPool(fx.model,128);
        sslm_prefix original=Root(fx,&pool.pool,sc.h,L,41);
        if(!original) return v;
        auto seed=Save(original);
        if(!RequireBlob(seed,"7.12 B-view count seed")) return v;
        for(size_t i=0;i<counts.size();++i) PutLe64(seed,144+8*i,counts[i]);
        Digest(seed);
        sslm_prefix prefix=nullptr;
        if(!Restored(fx,&pool.pool,seed,nullptr,sc.h,&prefix)) return v;
        sslm_seq adopter=BudgetSeq(fx,&pool.pool,80,sc.h);
        if(!adopter) return v;
        Status(sslm_seq_adopt_prefix(adopter,prefix),SSLM_OK,
               "7.12 B-view semantic witness adopt");
        PKV_CHECK_MSG(SaveBlob(adopter,&v.ssb) && IsMagic(v.ssb,"SSB6"),
                      "7.12 B-view semantic witness is missing");
        const auto table=PrefixTable(prefix);
        if(table.size()!=size_t((L+fx.B()-1)/fx.B())) return v;
        v.rows.resize(size_t(L)*fx.BytesPerToken());
        std::map<uint32_t,std::vector<uint8_t>> pages;
        const size_t H=fx.geo.kv_heads,D=fx.geo.head_dim;
        for(uint32_t l=0;l<fx.geo.layers;++l)
          for(uint32_t half=0;half<2;++half)
           for(uint32_t h=0;h<fx.geo.kv_heads;++h)
            for(int pos=0;pos<L;++pos) {
                const uint32_t page=table[size_t(pos/fx.B())];
                auto it=pages.find(page);
                if(it==pages.end()) it=pages.emplace(page,PeekPage(fx,pool.pool,page)).first;
                // 1.13.0's rotated in-page address (paged-KV §3.1, T1), through pkv::PageRunOffset.
                const size_t from=PageRunOffset(fx.geo.layers,fx.geo.kv_heads,D,fx.B(),l,half,h,pos);
                const size_t to=((size_t(l)*2+half)*H+h)*size_t(L)*D+size_t(pos)*D;
                if(it->second.size()<from+D) return View{};
                std::memcpy(v.rows.data()+to,it->second.data()+from,D);
            }
        // This view's semantic witnesses exist before its final save.
        v.blob=Save(prefix);
        return v;
    };
    const View a=build(sixteen),b=build(four);
    if(!RequireBlob(a.blob,"7.12 B=16") || !RequireBlob(b.blob,"7.12 B=4")) return;
    PKV_CHECK_MSG(a.ssb.size()>=156 && b.ssb.size()>=156 &&
                  a.rows.size()==size_t(L)*sixteen.BytesPerToken() && a.rows==b.rows,
                  "7.12 B=4/B=16 pre-save peeked canonical K/V rows differ");
    PKV_CHECK_MSG(a.ssb.size()>=36 && b.ssb.size()>=36 &&
                  std::equal(sixteen_integrity.begin(),sixteen_integrity.end(),a.ssb.begin()+4) &&
                  std::equal(four_integrity.begin(),four_integrity.end(),b.ssb.begin()+4),
                  "7.12 B-view SSB6 hashes differ from verified artifact integrity fields");
    for(size_t offset:{size_t(36),size_t(40),size_t(48),size_t(60),
                       size_t(76),size_t(84),size_t(92),size_t(124),
                       size_t(132),size_t(140),size_t(148)})
        PKV_CHECK_MSG(a.ssb.size()>=offset+8 && b.ssb.size()>=offset+8 &&
                      std::equal(a.ssb.begin()+offset,a.ssb.begin()+offset+8,b.ssb.begin()+offset),
                      "7.12 B views differ in pre-save semantic SSB6 field %zu",offset);
    auto inspect=[&](const View& v,const Fixture& fx,
                     const std::array<uint8_t,32>& expected_integrity) {
        PKV_CHECK_MSG(Header(v.blob,L,fx),"7.12 B-view size, shape or digest failed");
        if(v.blob.size()<kHeader+ArtifactHiddenSize(fx)+v.rows.size()) return;
        const bool model_hash=std::equal(expected_integrity.begin(),
                                         expected_integrity.end(),v.blob.begin()+60);
        PKV_CHECK_MSG(model_hash,"7.12 B-view SPX1 model hash differs from artifact");
        for(size_t i=0;i<counts.size();++i)
            PKV_CHECK_MSG(Le64(v.blob,144+8*i)==counts[i],
                          "7.12 B-view count %zu differs from fixed fixture",i);
        PKV_CHECK_MSG(v.blob.size()>=kHeader+ArtifactHiddenSize(fx) &&
                      std::equal(v.rows.begin(),v.rows.end(),
                                 v.blob.begin()+kHeader+ArtifactHiddenSize(fx)),
                      "7.12 B-view writer rows differ from independent peeks");
    };
    inspect(a,sixteen,sixteen_integrity);
    inspect(b,four,four_integrity);
    PKV_CHECK_MSG(a.blob.size()>=kHeader && a.blob.size()==b.blob.size() &&
                  !std::equal(a.blob.begin()+60,a.blob.begin()+92,b.blob.begin()+60) &&
                  std::equal(a.blob.begin(),a.blob.begin()+28,b.blob.begin()) &&
                  std::equal(a.blob.begin()+92,a.blob.end(),b.blob.begin()+92),
                  "7.12 B=4/B=16 canonical SPX1 bytes differ outside digest/model identity");
}
PKV_CELL("7.12/E2:cross-view", "E1", CrossView712);
}
