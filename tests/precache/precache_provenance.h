#pragma once
#include "precache_reference.h"
#include <filesystem>
#include <map>

#ifndef PC_COMPILER_ID
#error PC_COMPILER_ID must be supplied by CMake
#endif
#ifndef PC_COMPILER_VERSION
#error PC_COMPILER_VERSION must be supplied by CMake
#endif
#ifndef PC_PROJECT_VERSION
#error PC_PROJECT_VERSION must be supplied by CMake
#endif

namespace pc {
constexpr const char* kFixtureRevision="precache-fixtures-2026-10-01-r1";
using Receipt=std::map<std::string,std::string>;

inline std::string IdsSha(const std::vector<int32_t>& ids) {
    std::string canonical;
    for(int32_t id:ids) canonical+=std::to_string(id)+"\n";
    return Sha(reinterpret_cast<const uint8_t*>(canonical.data()),canonical.size());
}

inline Receipt LocalCompilerReceipt(const Fixture& fx,const std::vector<int32_t>& ids,
                                    const std::vector<uint8_t>& blob) {
    const std::string runner=Env("SUPERSLM_PRECACHE_RUNNER_EXE");
    Receipt r{{"format","PC-COMPILER-1"},{"compiler",PC_COMPILER_ID},
              {"compiler_version",PC_COMPILER_VERSION},{"project_version",PC_PROJECT_VERSION},
              {"fixture_revision",kFixtureRevision},{"model_sha256",fx.sha},
              {"input_ids_sha256",IdsSha(ids)},
              {"blob_sha256",Sha(blob.data(),blob.size())},
              {"runner_sha256",ShaFile(runner)}};
    PKV_CHECK_MSG(r["runner_sha256"].size()==64,
                  "4.3 compiled runner executable is missing for provenance");
    return r;
}

inline bool WriteReceipt(const std::filesystem::path& path,const Receipt& r) {
    std::ofstream out(path,std::ios::binary);
    for(const auto& [key,value]:r) out<<key<<"="<<value<<"\n";
    return bool(out);
}

inline Receipt ReadReceipt(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary);
    Receipt r;
    std::string line;
    while(std::getline(in,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        const size_t eq=line.find('=');
        if(eq==std::string::npos || eq==0 || !r.emplace(line.substr(0,eq),line.substr(eq+1)).second)
            return {};
    }
    return in.eof() && r.size()==9 ? r : Receipt{};
}

inline bool ValidCompilerReceipt(const Receipt& r,const Receipt& local,
                                 const char* compiler) {
    const auto has=[&](const char* field){return r.find(field)!=r.end();};
    if(r.size()!=9 || !has("format") || !has("compiler") ||
       r.at("format")!="PC-COMPILER-1" || r.at("compiler")!=compiler)
        return false;
    for(const char* field:{"project_version","fixture_revision","model_sha256",
                           "input_ids_sha256","blob_sha256"})
        if(!has(field) || r.at(field)!=local.at(field)) return false;
    return has("compiler_version") && has("runner_sha256") &&
           !r.at("compiler_version").empty() && r.at("runner_sha256").size()==64;
}
}
