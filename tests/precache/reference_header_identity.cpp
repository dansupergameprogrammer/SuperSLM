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
    // A reference row's selectors (adopter, decode_mode, adopter_mode) and its tokens are read through pc::Field,
    // which must take a key only as a whole whitespace-separated key=value token (plan §7, the carried F10-F13
    // portions: every identity field through the exact parse). Must-accept: the exact row. Must-reject: a key
    // reached as the suffix of another key, and a key given twice.
    const std::string row="adopter=1 decode_mode=greedy adopter_mode=legacy tokens=1,2,3";
    if(pc::Field(row,"adopter")!="1" || pc::Field(row,"decode_mode")!="greedy" ||
       pc::Field(row,"adopter_mode")!="legacy" || pc::Field(row,"tokens")!="1,2,3") {
        std::fputs("reference row must-accept failed\n",stderr);
        return 1;
    }
    const std::string suffix="not_adopter=0 adopter=1 x_decode_mode=damped decode_mode=greedy "
                             "adopter_mode=legacy prefix_tokens=9 tokens=1,2,3";
    if(pc::Field(suffix,"adopter")!="1" || pc::Field(suffix,"decode_mode")!="greedy" ||
       pc::Field(suffix,"tokens")!="1,2,3" || !pc::Field("not_adopter=0","adopter").empty()) {
        std::fprintf(stderr,"reference row read a key as the suffix of another key (adopter '%s', decode_mode '%s', "
                     "tokens '%s')\n",pc::Field(suffix,"adopter").c_str(),pc::Field(suffix,"decode_mode").c_str(),
                     pc::Field(suffix,"tokens").c_str());
        return 1;
    }
    if(!pc::Field("adopter=0 adopter=1 decode_mode=greedy","adopter").empty()) {
        std::fputs("reference row accepted a duplicate key\n",stderr);
        return 1;
    }
    std::puts("reference header: exact must-accept; four misnamed, duplicate and unknown must-reject; "
              "reference row: exact must-accept; suffix-key and duplicate-key must-reject");
    return 0;
}
