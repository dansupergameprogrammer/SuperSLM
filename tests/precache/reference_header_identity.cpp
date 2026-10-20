#include "precache_reference.h"
#include <cstdio>

int main() {
    using pc::ParseReferenceHeader;
    using pc::ReferenceHeader;
    const std::string good=
        "# anchor=0123456789abcdef0123456789abcdef01234567 "
        "harness=abc model_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
        "prompt_ids_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb "
        "discrimination_record_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc "
        "B=16 cap=4096 tokenizer=synthetic-Stream(e8b3fdd)";
    ReferenceHeader fields;
    if(!ParseReferenceHeader(good,&fields) || fields.size()!=8 ||
       fields.at("anchor")!="0123456789abcdef0123456789abcdef01234567") {
        std::fputs("reference header must-accept failed\n",stderr);
        return 1;
    }
    for(const char* key:{"anchor","model_sha256","prompt_ids_sha256",
                         "discrimination_record_sha256"}) {
        std::string misnamed=good;
        const std::string original=std::string(key)+"=";
        const size_t at=misnamed.find(original);
        if(at==std::string::npos) return 1;
        misnamed.insert(at,"not_");
        if(ParseReferenceHeader(misnamed,&fields)) {
            std::fprintf(stderr,"reference header accepted not_%s as %s\n",key,key);
            return 1;
        }
    }
    if(ParseReferenceHeader(good+" anchor=0123456789abcdef0123456789abcdef01234567",&fields) ||
       ParseReferenceHeader(good+" fake_anchor=0123456789abcdef0123456789abcdef01234567",&fields) ||
       ParseReferenceHeader("# not_anchor=abc",&fields)) {
        std::fputs("reference header accepted duplicate or unknown identity field\n",stderr);
        return 1;
    }
    std::puts("reference header: exact must-accept; four misnamed, duplicate and unknown must-reject");
    return 0;
}
