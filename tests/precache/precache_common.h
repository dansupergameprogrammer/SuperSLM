#pragma once
#include "precache_abi.h"
#include "paged-kv/pkv_budget_a_helpers.h"
#include "superslm/artifact.h"
#include "superslm/sha256.h"
#include <algorithm>
#include <array>
#include <climits>
#include <fstream>
#include <thread>

// The paged-KV test-only pin (pkv_r9.h; superslm_test_injection only), for PinFixtureB below.
extern "C" sslm_status KvPagesTestPinPagePositions(sslm_model model, int64_t page_positions);

namespace pc {
using namespace pkv;
using namespace pkv::budget_a;
constexpr size_t kHeader = 184;
constexpr int kUnbuilt = 0x7fff;

// SuperSLM 1.13.0 (paged-KV §3.1, T1): CFG1's kv_block_size no longer selects B; the page-size rule
// does (B = 64 at the synthetic 0.5B and 1.5B shapes, 128 at pkv_def). These cells were written at
// B = 16 and the B = 4 twin, so each fixture they map is pinned through the test-only pin, as the
// paged-KV C4/C5 loaders pin theirs (D-SLM8439; GetFixture under PKV_PIN_GENERATED_FIXTURES). The
// pin's SSLM_OK and the engine's B are both required; `pinned` makes Fixture::B() report it.
inline bool PinFixtureB(Fixture* fx, int64_t B) {
    if(!fx || !fx->ok) return false;
    const sslm_status pin=KvPagesTestPinPagePositions(fx->model,B);
    PKV_CHECK_MSG(pin==SSLM_OK,"the pin of %s to B = %lld is refused (%d)",fx->stem.c_str(),
                  static_cast<long long>(B),int(pin));
    PKV_CHECK_MSG(sslm_kv_page_positions(fx->model)==B,"%s runs at B = %lld after its pin, not %lld",
                  fx->stem.c_str(),static_cast<long long>(sslm_kv_page_positions(fx->model)),
                  static_cast<long long>(B));
    if(pin!=SSLM_OK) return false;
    fx->pinned=B;
    return true;
}

// Read the residual width from this fixture's CFG1, independently of the
// mapped model and SPX1 writer. The paged-KV harness's HiddenSize() is
// intentionally fixed to its 192-byte fixture family.
inline size_t ArtifactHiddenSize(const uint8_t* bytes, size_t size) {
    auto u32=[](const uint8_t* p) -> uint32_t {
        return uint32_t(p[0]) | (uint32_t(p[1])<<8) |
               (uint32_t(p[2])<<16) | (uint32_t(p[3])<<24);
    };
    auto u64=[&](const uint8_t* p) -> uint64_t {
        return uint64_t(u32(p)) | (uint64_t(u32(p+4))<<32);
    };
    const bool header=bytes && size>=superslm::kHeaderBytes &&
                      std::memcmp(bytes,superslm::kMagic,4)==0;
    PKV_CHECK_MSG(header,"precache artifact lacks an SSLM section header");
    if(!header) return 0;
    const uint32_t count=u32(bytes+12);
    const bool table=count<=superslm::kMaxSections &&
        uint64_t(count)*superslm::kSectionDescBytes<=
            size-superslm::kHeaderBytes;
    PKV_CHECK_MSG(table,"precache artifact section table is out of bounds");
    if(!table) return 0;
    size_t hidden=0;
    int config_count=0;
    for(uint32_t i=0;i<count;++i) {
        const uint8_t* row=bytes+superslm::kHeaderBytes+
                           size_t(i)*superslm::kSectionDescBytes;
        if(u32(row)!=uint32_t(superslm::SslmSectionType::Config)) continue;
        ++config_count;
        const uint64_t offset=u64(row+8), length=u64(row+16);
        const bool cfg=length>=12 && offset<=size &&
                       length<=size-offset &&
                       std::memcmp(bytes+offset,"CFG1",4)==0;
        PKV_CHECK_MSG(cfg,"precache CFG1 section is invalid");
        if(!cfg) return 0;
        hidden=u32(bytes+offset+8);
    }
    PKV_CHECK_MSG(config_count==1 && hidden>0,
                  "precache artifact must have one nonzero CFG1 hidden_size");
    return config_count==1 ? hidden : 0;
}

inline size_t ArtifactHiddenSize(const Fixture& fx) {
    return ArtifactHiddenSize(fx.bytes.p,fx.size);
}

inline void Status(sslm_status got, sslm_status want, const char* operation) {
    PKV_CHECK_MSG(got == want, "%s: expected %s (%d), got %s (%d)%s", operation,
        AbiStatusName(want), int(want), AbiStatusName(got), int(got),
        int(got) == kUnbuilt ? "; prefix ABI stub is unimplemented" : "");
}

inline std::vector<uint8_t> Save(sslm_prefix p) {
    size_t n = 0;
    const sslm_status query = sslm_prefix_save(p, nullptr, &n);
    Status(query, SSLM_BUFFER_TOO_SMALL, "sslm_prefix_save size query");
    if (query != SSLM_BUFFER_TOO_SMALL || n < kHeader) return {};
    std::vector<uint8_t> b(n + 1, 0xa5);
    size_t written = n;
    const sslm_status st = sslm_prefix_save(p, b.data(), &written);
    Status(st, SSLM_OK, "sslm_prefix_save write");
    PKV_CHECK_EQ(written, n);
    PKV_CHECK_EQ(b[n], 0xa5);
    if (st != SSLM_OK || written != n) return {};
    b.resize(n);
    return b;
}

inline void Digest(std::vector<uint8_t>& b) {
    if (b.size() < kHeader) return;
    std::fill(b.begin() + 28, b.begin() + 60, uint8_t{0});
    uint8_t hash[32];
    superslm::Sha256Hash(b.data(), b.size(), hash);
    std::copy(hash, hash + 32, b.begin() + 28);
}

inline sslm_status Restore(const Fixture& fx, sslm_kv_pool* pool,
    const std::vector<uint8_t>& b, sslm_prefix parent, sslm_prefix* out,
    uint32_t* shared = nullptr) {
    *out = OutSentinel<sslm_prefix>();
    return sslm_prefix_restore(fx.model, pool, b.data(), b.size(), parent, out, shared);
}

inline bool Restored(const Fixture& fx, sslm_kv_pool* pool,
    const std::vector<uint8_t>& b, sslm_prefix parent, Handles& h,
    sslm_prefix* out, uint32_t* shared = nullptr) {
    const sslm_status st = Restore(fx, pool, b, parent, out, shared);
    Status(st, SSLM_OK, "sslm_prefix_restore");
    if (st == SSLM_OK && *out && *out != OutSentinel<sslm_prefix>()) h.prefixes.push_back(*out);
    return st == SSLM_OK && *out && *out != OutSentinel<sslm_prefix>();
}

inline sslm_prefix Root(const Fixture& fx, sslm_kv_pool* pool, Handles& h,
                        int len = 65, int seed = 41, bool legacy = false) {
    sslm_prefix p = nullptr;
    sslm_status st = legacy ? sslm_prefix_begin(fx.model, pool, &p)
                            : sslm_prefix_begin_budgeted(fx.model, pool, std::max(len, 1), &p);
    Status(st, SSLM_OK, "prefix begin fixture");
    if (st != SSLM_OK || !p) return nullptr;
    h.prefixes.push_back(p);
    if (len) Status(PrefixPrefillAll(fx.model, p, Stream(seed, len, fx.vocab), 64),
                    SSLM_OK, "prefix prefill fixture");
    Status(sslm_prefix_freeze(p), SSLM_OK, "prefix freeze fixture");
    return p;
}

inline std::vector<int32_t> Decode(const Fixture& fx, sslm_kv_pool* pool,
                                   sslm_prefix p, bool budget, int count) {
    sslm_seq s = nullptr;
    sslm_status st = budget ? sslm_seq_create_budgeted(fx.model, pool, count + 4, &s)
                            : sslm_seq_create(fx.model, pool, &s);
    Status(st, SSLM_OK, "adopter create");
    if (st != SSLM_OK || !s) return {};
    Status(sslm_seq_adopt_prefix(s, p), SSLM_OK, "adopt prefix");
    std::vector<int32_t> tokens;
    for (int i = 0; i < count; ++i) {
        sslm_status ds = SSLM_OK;
        const int32_t token = NextToken(fx.model, s, &ds);
        Status(ds, SSLM_OK, "decode adopter");
        if (ds != SSLM_OK || token < 0) break;
        tokens.push_back(token);
    }
    Status(sslm_seq_release(s), SSLM_OK, "release adopter");
    return tokens;
}

inline bool TwinTokens(const Fixture& fx, sslm_kv_pool* pool, sslm_prefix original,
                       sslm_prefix restored, int count = 64) {
    for (bool budget : {false, true}) {
        const auto a = Decode(fx, pool, original, budget, count);
        const auto b = Decode(fx, pool, restored, budget, count);
        PKV_CHECK_MSG(a.size() == size_t(count) && a == b,
            "restored %s adopter tokens differ from prefill twin", budget ? "budget" : "legacy");
        if (a.size() != size_t(count) || a != b) return false;
    }
    return true;
}

inline bool Header(const std::vector<uint8_t>& b, int64_t L, const Fixture& fx) {
    const size_t expected = kHeader + ArtifactHiddenSize(fx) + size_t(L) * fx.BytesPerToken();
    PKV_CHECK_MSG(b.size() == expected, "SPX1 size %zu, expected %zu", b.size(), expected);
    PKV_CHECK_MSG(IsMagic(b, "SPX1"), "prefix writer did not write SPX1");
    if (b.size() < kHeader) return false;
    PKV_CHECK_EQ(Le32(b, 4), 0u);
    PKV_CHECK_EQ(static_cast<int64_t>(Le64(b, 112)), L);
    std::vector<uint8_t> copy = b;
    std::array<uint8_t, 32> got{};
    std::copy(b.begin() + 28, b.begin() + 60, got.begin());
    Digest(copy);
    PKV_CHECK_MSG(std::equal(got.begin(), got.end(), copy.begin() + 28), "SPX1 digest mismatch");
    return b.size() == expected && IsMagic(b, "SPX1");
}

inline std::vector<uint8_t> RootBlob(const Fixture& fx, int len = 65, bool legacy = false) {
    Scene sc;
    PagePool& pool = sc.AddPool(fx.model, std::max<uint32_t>(512, uint32_t(fx.CapPages() + 32)));
    if (pool.status != SSLM_OK) return {};
    sslm_prefix p = Root(fx, &pool.pool, sc.h, len, 41, legacy);
    return p ? Save(p) : std::vector<uint8_t>{};
}

inline bool RequireBlob(const std::vector<uint8_t>& b, const char* cell) {
    PKV_CHECK_MSG(!b.empty(), "%s: no SPX1 blob; prefix-save stub is unimplemented", cell);
    return !b.empty();
}

struct ModelFile {
    AlignedBuf bytes;
    size_t size = 0;
    sslm_model model = nullptr;
    sslm_status status = SSLM_INVALID_ARGUMENT;
    explicit ModelFile(const std::string& path) {
        std::vector<uint8_t> raw;
        PKV_CHECK_MSG(ReadAll(path,&raw),"model fixture missing: %s",path.c_str());
        if(raw.empty()) return;
        size=raw.size();
        bytes.Reset(size);
        std::memcpy(bytes.p,raw.data(),size);
        status=sslm_model_map(bytes.p,size,&model);
    }
    ~ModelFile() { if(model) sslm_model_unmap(model); }
};

// The hostile-input population uses a synthetic L=1 operand whose CFG1
// residual width and canonical K/V width are checked at each call site.
inline const Fixture& SyntheticFixture(bool three_schemas = false) {
    static std::map<bool, std::unique_ptr<Fixture>> cache;
    auto found = cache.find(three_schemas);
    if (found != cache.end()) return *found->second;
    auto fx = std::make_unique<Fixture>();
    const char* dir = std::getenv("SUPERSLM_PRECACHE_SYNTHETIC_DIR");
    PKV_CHECK_MSG(dir && *dir, "synthetic precache fixture directory is unset");
    if (dir && *dir) {
        const std::string file = three_schemas ? "synthetic0p5-three-schema.sslm"
                                               : "synthetic0p5.sslm";
        std::vector<uint8_t> raw;
        PKV_CHECK_MSG(ReadAll(std::string(dir) + "/" + file, &raw) && !raw.empty(),
                      "synthetic precache fixture %s is missing", file.c_str());
        if (!raw.empty()) {
            fx->size = raw.size();
            fx->bytes.Reset(raw.size());
            std::memcpy(fx->bytes.p, raw.data(), raw.size());
            fx->sha = Sha(raw.data(), raw.size());
            fx->geo = {24, 2, 64, 4096};
            fx->vocab = 256;
            const auto st = sslm_model_map(fx->bytes.p, fx->size, &fx->model);
            PKV_CHECK_MSG(st == SSLM_OK, "synthetic precache model map returned %d", int(st));
            fx->ok = st == SSLM_OK;
            fx->stem = file;
            PinFixtureB(fx.get(), 16);
        }
    }
    const Fixture& result = *fx;
    cache.emplace(three_schemas, std::move(fx));
    return result;
}

inline std::unique_ptr<Fixture> LoadSyntheticShape(const char* filename,
                                                   Geometry geometry) {
    auto fx=std::make_unique<Fixture>();
    const char* dir=std::getenv("SUPERSLM_PRECACHE_SYNTHETIC_DIR");
    PKV_CHECK_MSG(dir && *dir,"synthetic precache fixture directory is unset");
    std::vector<uint8_t> raw;
    if(!dir || !*dir || !ReadAll(std::string(dir)+"/"+filename,&raw) || raw.empty()) {
        PKV_CHECK_MSG(false,"synthetic shape %s is missing",filename);
        return fx;
    }
    fx->stem=filename;
    fx->size=raw.size();
    fx->bytes.Reset(raw.size());
    std::memcpy(fx->bytes.p,raw.data(),raw.size());
    fx->sha=Sha(raw.data(),raw.size());
    fx->geo=geometry;
    fx->vocab=256;
    fx->ok=sslm_model_map(fx->bytes.p,fx->size,&fx->model)==SSLM_OK;
    PKV_CHECK_MSG(fx->ok,"synthetic shape %s did not map",filename);
    PinFixtureB(fx.get(),16);
    return fx;
}

inline sslm_status SchemaContentAll(sslm_model model,sslm_prefix prefix,
                                     const std::vector<int32_t>& ids,int32_t chunk) {
    size_t at=0;
    while(at<ids.size()) {
        int32_t consumed=0;
        const auto st=sslm_prefix_prefill(model,prefix,ids.data()+at,
            int32_t(ids.size()-at),chunk,SSLM_SPAN_SCHEMA_CONTENT,nullptr,&consumed);
        if(st!=SSLM_OK || consumed<=0) return st==SSLM_OK ? SSLM_INVALID_ARGUMENT : st;
        at+=size_t(consumed);
    }
    return SSLM_OK;
}

inline sslm_status SeqSchemaContentAll(sslm_model model,sslm_seq seq,
                                       const std::vector<int32_t>& ids,int32_t chunk) {
    size_t at=0;
    while(at<ids.size()) {
        int32_t consumed=0;
        const auto st=sslm_prefill(model,seq,ids.data()+at,
            int32_t(ids.size()-at),chunk,SSLM_SPAN_SCHEMA_CONTENT,nullptr,&consumed);
        if(st!=SSLM_OK || consumed<=0) return st==SSLM_OK ? SSLM_INVALID_ARGUMENT : st;
        at+=size_t(consumed);
    }
    return SSLM_OK;
}
}
