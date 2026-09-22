// Full A/B state-membership and persistence oracle. This cell is built only after
// the three independent declaration cells compile, so API absence never masks a sibling.
#include "superslm/sslm_abi.h"
#define SUPERSLM_T2791_SUPPRESS_GPU_STATUS_USING_ENUM
#include "../t2791-gpu-prefill-read-red-suite/fixture_common.h"
#if defined(SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT)
#include "support/gpu_chunk_dispatch_instrument.h"
namespace superslm_test {
// §3.10.8 commissions these two test-only counters. They intentionally link-red
// until the builder places increments at the real fence-wait/readiness-poll sites.
extern std::atomic<int64_t> g_gpu_fence_wait_count_probe;
extern std::atomic<int64_t> g_gpu_ready_poll_count_probe;
}
#endif
#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

namespace {
bool ReadFile(const char* path, std::vector<uint8_t>* bytes) {
    FILE* f = nullptr; if (fopen_s(&f, path, "rb") || !f) return false;
    std::fseek(f, 0, SEEK_END); const long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    bytes->resize(n > 0 ? size_t(n) : 0);
    const bool ok = n >= 0 && (n == 0 || std::fread(bytes->data(), 1, bytes->size(), f) == bytes->size());
    std::fclose(f); return ok;
}
#undef CHECK
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; std::printf("FAIL line=%d %s\n", __LINE__, #x); } } while (0)
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    int checks = 0, failures = 0;
    std::vector<uint8_t> bytes; if (!ReadFile(argv[1], &bytes)) return 2;

    sslm_model cpu_model = nullptr; CHECK(sslm_model_map(bytes.data(), bytes.size(), &cpu_model) == SSLM_OK);
    const size_t block = sslm_kv_block_size(cpu_model), overhead = sslm_kv_pool_overhead_size(cpu_model, 1);
    std::vector<uint8_t> storage(block + overhead + 63); void* aligned = storage.data(); size_t space = storage.size();
    CHECK(std::align(64, block + overhead, aligned, space) != nullptr);
    sslm_kv_pool pool = nullptr; CHECK(sslm_kv_pool_create(cpu_model, aligned, block + overhead, 1, &pool) == SSLM_OK);
    sslm_seq cpu_seq = nullptr; CHECK(sslm_seq_create(cpu_model, &pool, &cpu_seq) == SSLM_OK);
    int32_t out = -77; CHECK(sslm_seq_schema_bound(cpu_seq, &out) == SSLM_OK && out == 0);
    sslm_schema cpu_a = nullptr; CHECK(sslm_schema_lookup(cpu_model, "t2922_accepts_q", &cpu_a) == SSLM_OK);
    CHECK(sslm_seq_set_schema(cpu_seq, cpu_a) == SSLM_OK);
    out = -77; CHECK(sslm_seq_schema_bound(cpu_seq, &out) == SSLM_OK && out == 1);
    CHECK(sslm_seq_reset(cpu_seq) == SSLM_OK);
    out = -77; CHECK(sslm_seq_schema_bound(cpu_seq, &out) == SSLM_OK && out == 1);
    out = 0x10203040; CHECK(sslm_seq_schema_bound(nullptr, &out) == SSLM_INVALID_ARGUMENT && out == 0x10203040);

    SslmGpuContext* ctx = nullptr;
    CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SslmGpuStatus::SSLM_OK);
    GpuModelFixture fx; CHECK(fx.Open(argv[1], ctx));
    SslmGpuSequenceHandle* seq = nullptr; CHECK(sslm_gpu_seq_create(ctx, fx.model, fx.model_cap, &seq) == SslmGpuStatus::SSLM_OK);
    int32_t accepting = -1, bound = -1;
#if defined(SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT)
    const int64_t submit_before = superslm_test::g_gpu_chunk_submit_count_probe.load();
    const int64_t dispatch_before = superslm_test::g_gpu_chunk_dispatch_count_probe.load();
    const int64_t wait_before = superslm_test::g_gpu_fence_wait_count_probe.load();
    const int64_t ready_before = superslm_test::g_gpu_ready_poll_count_probe.load();
#endif
    CHECK(SslmGpuSeqSchemaAcceptingForG5Bridge(ctx, seq, &accepting) == SslmGpuStatus::SSLM_OK && accepting == 0);
    CHECK(SslmGpuSeqSchemaBoundForG5Bridge(ctx, seq, &bound) == SslmGpuStatus::SSLM_OK && bound == 0);
#if defined(SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT)
    CHECK(superslm_test::g_gpu_chunk_submit_count_probe.load() == submit_before);
    CHECK(superslm_test::g_gpu_chunk_dispatch_count_probe.load() == dispatch_before);
    CHECK(superslm_test::g_gpu_fence_wait_count_probe.load() == wait_before);
    CHECK(superslm_test::g_gpu_ready_poll_count_probe.load() == ready_before);
#endif
    const int32_t a = SslmGpuSchemaLookupForG5Bridge(fx.model, "t2922_accepts_q");
    const int32_t b = SslmGpuSchemaLookupForG5Bridge(fx.model, "t2922_rejects_q");
    CHECK(a >= 0 && b >= 0 && a != b);
    CHECK(SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, a) == SslmGpuStatus::SSLM_OK);
    accepting = bound = -1;
    CHECK(SslmGpuSeqSchemaAcceptingForG5Bridge(ctx, seq, &accepting) == SslmGpuStatus::SSLM_OK && accepting == 0);
    CHECK(SslmGpuSeqSchemaBoundForG5Bridge(ctx, seq, &bound) == SslmGpuStatus::SSLM_OK && bound == 1);
    const int32_t token_q = 0; int32_t consumed = 0;
    CHECK(SslmGpuSeqPrefillSchemaContentForG5Bridge(ctx, seq, &token_q, 1, fx.one_layer_budget, &consumed) == SslmGpuStatus::SSLM_OK && consumed == 1);
    CHECK(SslmGpuSeqWalkStateForG5Bridge(seq) == 1);
    accepting = -1; CHECK(SslmGpuSeqSchemaAcceptingForG5Bridge(ctx, seq, &accepting) == SslmGpuStatus::SSLM_OK && accepting == 1);

    size_t need = 0; CHECK(sslm_gpu_seq_save(ctx, seq, nullptr, &need) == SslmGpuStatus::SSLM_DEVICE_LOST && need > 0);
    std::vector<uint8_t> blob(need); size_t written = need;
    CHECK(sslm_gpu_seq_save(ctx, seq, blob.data(), &written) == SslmGpuStatus::SSLM_OK);
    SslmGpuSequenceHandle* restored = nullptr;
    CHECK(sslm_gpu_seq_restore(ctx, fx.model, blob.data(), written, &restored) == SslmGpuStatus::SSLM_OK);
    accepting = bound = -1;
    CHECK(SslmGpuSeqSchemaAcceptingForG5Bridge(ctx, restored, &accepting) == SslmGpuStatus::SSLM_OK && accepting == 1);
    CHECK(SslmGpuSeqSchemaBoundForG5Bridge(ctx, restored, &bound) == SslmGpuStatus::SSLM_OK && bound == 1);

    CHECK(sslm_gpu_seq_reset(ctx, seq) == SslmGpuStatus::SSLM_OK);
    CHECK(SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, b) == SslmGpuStatus::SSLM_OK);
    consumed = 0; CHECK(SslmGpuSeqPrefillSchemaContentForG5Bridge(ctx, seq, &token_q, 1, fx.one_layer_budget, &consumed) == SslmGpuStatus::SSLM_OK && consumed == 1);
    CHECK(SslmGpuSeqWalkStateForG5Bridge(seq) == 1);
    accepting = bound = -1;
    CHECK(SslmGpuSeqSchemaAcceptingForG5Bridge(ctx, seq, &accepting) == SslmGpuStatus::SSLM_OK && accepting == 0);
    CHECK(SslmGpuSeqSchemaBoundForG5Bridge(ctx, seq, &bound) == SslmGpuStatus::SSLM_OK && bound == 1);

    accepting = 0x13572468;
    CHECK(SslmGpuSeqSchemaAcceptingForG5Bridge(nullptr, seq, &accepting) == SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && accepting == 0x13572468);
    bound = 0x24681357;
    CHECK(SslmGpuSeqSchemaBoundForG5Bridge(nullptr, seq, &bound) == SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && bound == 0x24681357);

#if defined(SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT)
    SslmGpuSequenceHandle* commissioned = nullptr;
    CHECK(sslm_gpu_seq_create(ctx, fx.model, fx.model_cap, &commissioned) == SslmGpuStatus::SSLM_OK);
    const int64_t submit_control = superslm_test::g_gpu_chunk_submit_count_probe.load();
    const int64_t dispatch_control = superslm_test::g_gpu_chunk_dispatch_count_probe.load();
    const int64_t wait_control = superslm_test::g_gpu_fence_wait_count_probe.load();
    const int64_t ready_control = superslm_test::g_gpu_ready_poll_count_probe.load();
    int32_t produced = -99;
    CHECK(SslmGpuSeqDecodeStepForG5Bridge(ctx, commissioned, 7, fx.one_layer_budget, &produced) == SslmGpuStatus::SSLM_OK);
    CHECK(superslm_test::g_gpu_chunk_submit_count_probe.load() > submit_control);
    CHECK(superslm_test::g_gpu_chunk_dispatch_count_probe.load() > dispatch_control);
    CHECK(superslm_test::g_gpu_fence_wait_count_probe.load() > wait_control);
    CHECK(superslm_test::g_gpu_ready_poll_count_probe.load() > ready_control);
    sslm_gpu_seq_release(ctx, commissioned);
#endif

    if (restored) sslm_gpu_seq_release(ctx, restored); sslm_gpu_seq_release(ctx, seq);
    fx.Close(); sslm_gpu_context_destroy(ctx); sslm_seq_release(cpu_seq);
    sslm_kv_pool_destroy(pool); sslm_model_unmap(cpu_model);
    std::printf("SUMMARY checks=%d failures=%d skips=0\n", checks, failures);
    return failures ? 1 : 0;
}
