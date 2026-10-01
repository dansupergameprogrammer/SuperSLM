#pragma once
#include "precache_abi.h"
#include "paged-kv/pkv_budget_a_helpers.h"
#include "superslm/sha256.h"
#include <algorithm>
#include <array>
#include <climits>
#include <fstream>
#include <thread>

namespace pc {
using namespace pkv;
using namespace pkv::budget_a;
constexpr size_t kHeader = 184;
constexpr int kUnbuilt = 0x7fff;

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
    const size_t expected = kHeader + HiddenSize(fx) + size_t(L) * fx.BytesPerToken();
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
}
