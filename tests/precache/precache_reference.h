#pragma once
#include "precache_common.h"
#include <filesystem>
#include <sstream>

namespace pc {
namespace fs=std::filesystem;

inline std::string Env(const char* name) {
    const char* p=std::getenv(name);
    PKV_CHECK_MSG(p && *p,"%s is unset",name);
    return p ? p : "";
}

inline std::vector<int32_t> ReadIds(const fs::path& path) {
    std::ifstream file(path);
    PKV_CHECK_MSG(bool(file),"reference prompt IDs missing: %s",path.string().c_str());
    std::vector<int32_t> ids;
    int32_t id=0;
    while(file>>id) ids.push_back(id);
    PKV_CHECK_MSG(ids.size()>=128,"reference prompt %s has %zu IDs, needs >=128",
                  path.string().c_str(),ids.size());
    return ids;
}

inline std::string ShaFile(const fs::path& path) {
    std::vector<uint8_t> b;
    if(!ReadAll(path.string(),&b)) return {};
    return Sha(b.data(),b.size());
}

inline std::string Field(const std::string& s,const std::string& key) {
    const std::string pattern=key+"=";
    const size_t begin=s.find(pattern);
    if(begin==std::string::npos) return {};
    const size_t value=begin+pattern.size();
    return s.substr(value,s.find_first_of(" \t\r\n",value)-value);
}

// One licensed reference path for E2-E4. T-3035 supplies the population.
inline std::vector<int32_t> Reference(const std::string& group,const std::string& stem,
                                      const fs::path& ids,const std::string& model_sha,
                                      int adopter,const char* mode) {
    const auto dir=Env("SUPERSLM_PRECACHE_REFERENCE_DIR");
    const auto merge_file=Env("SUPERSLM_PRECACHE_MERGE_BASE_FILE");
    const auto license_file=Env("SUPERSLM_PRECACHE_DISCRIMINATION_RECEIPT");
    if(dir.empty() || merge_file.empty() || license_file.empty()) return {};
    std::ifstream merge(merge_file),license(license_file);
    std::string anchor,receipt;
    std::getline(merge,anchor);
    std::getline(license,receipt);
    PKV_CHECK_MSG(anchor.size()==40,"reference merge-base fixture missing a full SHA");
    PKV_CHECK_MSG(receipt.find("COMMISSIONED")!=std::string::npos,
                  "reference population has no commissioned discrimination receipt");
    if(anchor.size()!=40 || receipt.find("COMMISSIONED")==std::string::npos) return {};
    const fs::path base=fs::path(dir)/group/(stem+"-run1.ref");
    const fs::path twin=fs::path(dir)/group/(stem+"-run2.ref");
    const auto first_sha=ShaFile(base),second_sha=ShaFile(twin);
    PKV_CHECK_MSG(!first_sha.empty() && first_sha==second_sha,
                  "reference runs do not reproduce byte-for-byte: %s",stem.c_str());
    std::ifstream ref(base);
    std::string header,line;
    std::getline(ref,header);
    PKV_CHECK_MSG(Field(header,"anchor")==anchor,"stale reference %s anchor %s; current %s",
                  stem.c_str(),Field(header,"anchor").c_str(),anchor.c_str());
    PKV_CHECK_MSG(Field(header,"prompt_ids_sha256")==ShaFile(ids),
                  "reference %s IDs SHA differs from the prompt",stem.c_str());
    PKV_CHECK_MSG(Field(header,"model_sha256")==model_sha,
                  "reference %s model SHA differs from the mapped artifact",stem.c_str());
    const auto discrimination=Field(header,"discrimination_record_sha256");
    PKV_CHECK_MSG(discrimination.size()==64 && receipt.find(discrimination)!=std::string::npos,
                  "reference %s lacks its commissioned discrimination record identity",stem.c_str());
    if(Field(header,"anchor")!=anchor || Field(header,"prompt_ids_sha256")!=ShaFile(ids) ||
       Field(header,"model_sha256")!=model_sha ||
       discrimination.size()!=64 || receipt.find(discrimination)==std::string::npos) return {};
    while(std::getline(ref,line)) {
        if(Field(line,"adopter")!=std::to_string(adopter) || Field(line,"decode_mode")!=mode) continue;
        const auto expected_mode=(adopter%2)==0 ? "budget" : "legacy";
        PKV_CHECK_MSG(Field(line,"adopter_mode")==expected_mode,
                      "reference %s adopter %d mode mismatch",stem.c_str(),adopter);
        const auto encoded=Field(line,"tokens");
        std::istringstream in(encoded);
        std::vector<int32_t> tokens;
        std::string one;
        while(std::getline(in,one,',')) tokens.push_back(std::stoi(one));
        PKV_CHECK_EQ(tokens.size(),64u);
        return tokens;
    }
    PKV_CHECK_MSG(false,"reference %s lacks adopter %d %s",stem.c_str(),adopter,mode);
    return {};
}
}
