#include "../t2791-gpu-prefill-read-red-suite/fixture_common.h"
#include "superslm/sslm_abi.h"
#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>

namespace {
std::vector<int32_t> ReadIds(const char* path) {
    std::ifstream in(path); std::vector<int32_t> ids; int32_t id;
    while (in >> id) ids.push_back(id); return ids;
}
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; std::printf("FAIL line=%d %s\n", __LINE__, #x); } } while (0)
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    int checks = 0, failures = 0;
    const std::vector<int32_t> prompt = ReadIds(argv[2]); if (prompt.empty()) return 2;
    std::vector<uint8_t> bytes; if (!ReadFileBytes(argv[1], &bytes)) return 2;
    sslm_model model = nullptr; CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
    sslm_schema schema = nullptr; CHECK(sslm_schema_lookup(model, "potion_shop_order", &schema) == SSLM_OK);
    const size_t block = sslm_kv_block_size(model), overhead = sslm_kv_pool_overhead_size(model, 1);
    std::vector<uint8_t> storage(block + overhead + 63); void* aligned = storage.data(); size_t space = storage.size();
    CHECK(std::align(64, block + overhead, aligned, space) != nullptr);
    sslm_kv_pool pool = nullptr; CHECK(sslm_kv_pool_create(model, aligned, block + overhead, 1, &pool) == SSLM_OK);
    sslm_seq cpu = nullptr; CHECK(sslm_seq_create(model, &pool, &cpu) == SSLM_OK);
    CHECK(sslm_seq_set_schema(cpu, schema) == SSLM_OK);
    int32_t offset = 0;
    while (offset < static_cast<int32_t>(prompt.size())) {
        int32_t consumed = 0;
        CHECK(sslm_prefill(model, cpu, prompt.data() + offset,
              static_cast<int32_t>(prompt.size()) - offset, 64, SSLM_SPAN_PROMPT,
              nullptr, &consumed) == SSLM_OK && consumed > 0);
        if (consumed <= 0) return 2; offset += consumed;
    }

    SslmGpuContext* ctx = nullptr; CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
    GpuModelFixture fx; CHECK(fx.Open(argv[1], ctx));
    const int32_t gpu_schema = SslmGpuSchemaLookupForG5Bridge(fx.model, "potion_shop_order");
    CHECK(gpu_schema >= 0);
    SslmGpuSequenceHandle* gpu = nullptr; CHECK(sslm_gpu_seq_create(ctx, fx.model, fx.model_cap, &gpu) == SSLM_OK);
    CHECK(SslmGpuSeqSetSchemaForG5Bridge(ctx, gpu, gpu_schema) == SSLM_OK);
    CHECK(SslmGpuSeqPrefillPromptForG5Bridge(ctx, gpu, prompt.data(),
          static_cast<int32_t>(prompt.size()), fx.one_layer_budget) == SSLM_OK);

    sslm_decode_params params{}; params.layer_budget = static_cast<int32_t>(fx.layers);
    int32_t gpu_input = prompt.back(); int produced = 0;
    for (; produced < 300; ++produced) {
        int32_t cpu_token = -99, gpu_token = -99;
        CHECK(sslm_decode_step(model, &cpu, 1, &params, nullptr, &cpu_token) == SSLM_OK);
        CHECK(SslmGpuSeqDecodeStepForG5Bridge(ctx, gpu, gpu_input, fx.one_layer_budget,
                                              &gpu_token) == SSLM_OK);
        CHECK(cpu_token >= 0 && gpu_token >= 0 && cpu_token == gpu_token);
        if (cpu_token < 0 || gpu_token < 0) break;
        gpu_input = gpu_token;
    }
    sslm_stats_out stats{}; CHECK(sslm_stats(model, cpu, &stats) == SSLM_OK);
    int32_t cpu_bound = -1, gpu_accepting = -1, gpu_bound = -1;
    CHECK(sslm_seq_schema_bound(cpu, &cpu_bound) == SSLM_OK && cpu_bound == 1);
    CHECK(SslmGpuSeqSchemaAcceptingForG5Bridge(ctx, gpu, &gpu_accepting) == SSLM_OK);
    CHECK(SslmGpuSeqSchemaBoundForG5Bridge(ctx, gpu, &gpu_bound) == SSLM_OK);
    CHECK(produced == 300 && stats.schema_accepting == 0 && gpu_accepting == 0 && gpu_bound == 1);
    std::printf("CELL real_model stop=%s produced=%d cpu_accepting=%d gpu_accepting=%d gpu_bound=%d\n",
                produced == 300 ? "budget" : "early", produced, stats.schema_accepting,
                gpu_accepting, gpu_bound);
    sslm_gpu_seq_release(ctx, gpu); fx.Close(); sslm_gpu_context_destroy(ctx);
    sslm_seq_release(cpu); sslm_kv_pool_destroy(pool); sslm_model_unmap(model);
    std::printf("SUMMARY checks=%d failures=%d skips=0\n", checks, failures);
    return failures ? 1 : 0;
}
