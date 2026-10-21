#pragma once
#include "precache_common.h"
#include <filesystem>
#include <set>
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

// A reference row's value for `key`: the key only as a whole whitespace-separated key=value token, never as the
// suffix of another key; empty when it is absent or given twice (reference_header_identity.cpp's must-rejects).
inline std::string Field(const std::string& s,const std::string& key) {
    std::istringstream in(s);
    std::string token,found;
    bool seen=false;
    while(in>>token) {
        const size_t at=token.find('=');
        if(at==std::string::npos || token.compare(0,at,key)!=0 || at!=key.size()) continue;
        if(seen) return {};
        seen=true;
        found=token.substr(at+1);
    }
    return found;
}

using ReferenceHeader=std::map<std::string,std::string>;

inline bool ParseReferenceHeader(const std::string& line,ReferenceHeader* fields) {
    fields->clear();
    if(line.rfind("# ",0)!=0) return false;
    static const std::set<std::string> allowed={
        "B","adapter_sha256","anchor","cap","decode_modes_available",
        "discrimination_record_sha256","forced_content_id","forced_content_ids_sha256",
        "harness","model_sha256","persona_ids_sha256","persona_source_sha256",
        "prompt_id","prompt_ids_sha256","prompt_source_sha256","tokenizer",
        "world_ids_sha256","world_source_sha256"};
    std::istringstream in(line.substr(2));
    std::string token;
    while(in>>token) {
        const size_t at=token.find('=');
        if(at==std::string::npos || at==0 || at+1==token.size() ||
           token.find('=',at+1)!=std::string::npos ||
           !allowed.count(token.substr(0,at)) ||
           !fields->emplace(token.substr(0,at),token.substr(at+1)).second) {
            fields->clear();
            return false;
        }
    }
    for(const char* required:{"anchor","model_sha256","prompt_ids_sha256",
                              "discrimination_record_sha256"})
        if(!fields->count(required)) {fields->clear();return false;}
    return true;
}

inline std::map<std::string,std::string> LicenseFields(const fs::path& path) {
    std::ifstream in(path);
    std::map<std::string,std::string> fields;
    std::string line;
    while(std::getline(in,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        const size_t at=line.find('=');
        if(at==std::string::npos || at==0 ||
           !fields.emplace(line.substr(0,at),line.substr(at+1)).second) return {};
    }
    return in.eof() ? fields : std::map<std::string,std::string>{};
}

inline bool HasLicensedRecord(const std::string& csv,const std::string& hash) {
    std::istringstream in(csv);
    std::string field;
    while(std::getline(in,field,',')) if(field==hash) return true;
    return false;
}

// One licensed reference path for E2-E4. T-3035 supplies the population.
inline std::vector<int32_t> Reference(const std::string& group,const std::string& stem,
                                      const fs::path& ids,const std::string& model_sha,
                                      int adopter,const char* mode) {
    const auto dir=Env("SUPERSLM_PRECACHE_REFERENCE_DIR");
    const auto merge_file=Env("SUPERSLM_PRECACHE_MERGE_BASE_FILE");
    const auto license_file=Env("SUPERSLM_PRECACHE_DISCRIMINATION_RECEIPT");
    const auto manifest=Env("SUPERSLM_PRECACHE_REFERENCE_MANIFEST");
    const auto instrument=Env("SUPERSLM_PRECACHE_DISCRIMINATION_INSTRUMENT");
    if(dir.empty() || merge_file.empty() || license_file.empty() ||
       manifest.empty() || instrument.empty()) return {};
    std::ifstream merge(merge_file);
    std::string anchor;
    std::getline(merge,anchor);
    if(!anchor.empty() && anchor.back()=='\r') anchor.pop_back();
    const auto license=LicenseFields(license_file);
    const auto field=[&](const char* name) {
        const auto found=license.find(name);
        return found==license.end() ? std::string{} : found->second;
    };
    const bool licensed=anchor.size()==40 && license.size()==9 &&
        field("format")=="PC-DISCRIMINATION-1" && field("status")=="COMMISSIONED" &&
        field("commissioning_verdict")=="PASS" && field("must_accept")=="PASS" &&
        field("must_reject")=="PASS" && field("anchor")==anchor &&
        field("population_manifest_sha256")==ShaFile(manifest) &&
        field("instrument_source_sha256")==ShaFile(instrument) &&
        !field("record_hashes").empty();
    PKV_CHECK_MSG(licensed,
                  "reference license is missing, stale, uncommissioned or differs from the exact population/instrument");
    if(!licensed) return {};
    const fs::path base=fs::path(dir)/group/(stem+"-run1.ref");
    const fs::path twin=fs::path(dir)/group/(stem+"-run2.ref");
    const auto first_sha=ShaFile(base),second_sha=ShaFile(twin);
    PKV_CHECK_MSG(!first_sha.empty() && first_sha==second_sha,
                  "reference runs do not reproduce byte-for-byte: %s",stem.c_str());
    std::ifstream ref(base);
    std::string header,line;
    std::getline(ref,header);
    ReferenceHeader header_fields;
    const bool parsed=ParseReferenceHeader(header,&header_fields);
    PKV_CHECK_MSG(parsed,
                  "reference %s header has a missing, duplicate, misnamed or unknown identity field",
                  stem.c_str());
    if(!parsed) return {};
    PKV_CHECK_MSG(header_fields.at("anchor")==anchor,"stale reference %s anchor %s; current %s",
                  stem.c_str(),header_fields.at("anchor").c_str(),anchor.c_str());
    PKV_CHECK_MSG(header_fields.at("prompt_ids_sha256")==ShaFile(ids),
                  "reference %s IDs SHA differs from the prompt",stem.c_str());
    PKV_CHECK_MSG(header_fields.at("model_sha256")==model_sha,
                  "reference %s model SHA differs from the mapped artifact",stem.c_str());
    const auto discrimination=header_fields.at("discrimination_record_sha256");
    PKV_CHECK_MSG(discrimination.size()==64 &&
                  HasLicensedRecord(field("record_hashes"),discrimination),
                  "reference %s lacks its commissioned discrimination record identity",stem.c_str());
    if(header_fields.at("anchor")!=anchor ||
       header_fields.at("prompt_ids_sha256")!=ShaFile(ids) ||
       header_fields.at("model_sha256")!=model_sha ||
       discrimination.size()!=64 ||
       !HasLicensedRecord(field("record_hashes"),discrimination)) return {};
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
