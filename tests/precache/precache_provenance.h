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

inline Receipt ReadCompilerJob(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary);
    Receipt r;
    std::string line;
    while(std::getline(in,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        const size_t eq=line.find('=');
        if(eq==std::string::npos || eq==0 || eq+1==line.size() ||
           !r.emplace(line.substr(0,eq),line.substr(eq+1)).second) return {};
    }
    static const std::set<std::string> keys={
        "format","compiler","compiler_version","runner_sha256",
        "compiler_tool_sha256","cmake_record_sha256","build_log_sha256",
        "capture_log_sha256","blob_sha256","source_revision"};
    if(!in.eof() || r.size()!=keys.size()) return {};
    for(const auto& [key,value]:r) if(!keys.count(key) || value.empty()) return {};
    return r;
}

inline std::string CmakeCompilerField(const std::string& text,const char* key) {
    const std::string prefix=std::string("set(")+key+" \"";
    const size_t first=text.find(prefix);
    if(first==std::string::npos) return {};
    const size_t begin=first+prefix.size(),end=text.find('"',begin);
    return end==std::string::npos ? std::string{} : text.substr(begin,end-begin);
}

// The bundle is produced by capture_compiler_job.py after a successful configure,
// build and selected 4.3 capture run. All listed artifacts are rehashed here.
inline bool ValidCompilerJob(const std::filesystem::path& dir,const Receipt& r,
                             const Receipt& local,const char* compiler) {
    const auto has=[&](const char* field){return r.find(field)!=r.end();};
    if(r.size()!=9 || !has("format") || !has("compiler") ||
       r.at("format")!="PC-COMPILER-1" || r.at("compiler")!=compiler)
        return false;
    for(const char* field:{"project_version","fixture_revision","model_sha256",
                           "input_ids_sha256","blob_sha256"})
        if(!has(field) || r.at(field)!=local.at(field)) return false;
    if(!has("compiler_version") || !has("runner_sha256") ||
       r.at("compiler_version").empty()) return false;
    const auto job=ReadCompilerJob(dir/(std::string(compiler)+".job"));
    if(job.size()!=10 || job.at("format")!="PC-COMPILER-JOB-1" ||
       job.at("compiler")!=compiler ||
       job.at("compiler_version")!=r.at("compiler_version") ||
       job.at("runner_sha256")!=r.at("runner_sha256") ||
       job.at("blob_sha256")!=r.at("blob_sha256")) return false;
    const std::string stem=compiler;
    for(const auto& [suffix,key]:{
            std::pair<const char*,const char*>{"runner","runner_sha256"},
            {"compiler","compiler_tool_sha256"},
            {"cmake","cmake_record_sha256"},
            {"build.log","build_log_sha256"},
            {"capture.log","capture_log_sha256"},
            {"spx1","blob_sha256"}})
        if(ShaFile(dir/(stem+"."+suffix))!=job.at(key)) return false;
    std::vector<uint8_t> cmake_bytes,capture_bytes;
    if(!ReadAll((dir/(stem+".cmake")).string(),&cmake_bytes) ||
       !ReadAll((dir/(stem+".capture.log")).string(),&capture_bytes)) return false;
    const std::string cmake(cmake_bytes.begin(),cmake_bytes.end());
    const std::string capture(capture_bytes.begin(),capture_bytes.end());
    return CmakeCompilerField(cmake,"CMAKE_CXX_COMPILER_ID")==compiler &&
           CmakeCompilerField(cmake,"CMAKE_CXX_COMPILER_VERSION")==r.at("compiler_version") &&
           !CmakeCompilerField(cmake,"CMAKE_CXX_COMPILER").empty() &&
           capture.find("4.3/E4 green")!=std::string::npos &&
           capture.find("1 cells, 0 red")!=std::string::npos &&
           job.at("source_revision").size()==40;
}
}
