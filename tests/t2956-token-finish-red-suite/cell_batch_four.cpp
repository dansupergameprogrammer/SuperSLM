// T-2851 row 8: one four-sequence hooked decode equals four single decodes.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <malloc.h>
#include "superslm/parallel_for.h"
#include "superslm/sslm_abi.h"

extern "C" sslm_status sslm_workspace_set_parallel_for(sslm_workspace,
                                                       const sslm_parallel_for*);

namespace {
constexpr int32_t prompt[] = {40, 1035, 1075, 311, 3695, 264, 2820, 60108, 11, 4486, 13};
void Reverse(void* host, int32_t count, sslm_task_fn task, void* ctx) {
    ++*static_cast<int*>(host);
    for (int32_t i = count - 1; i >= 0; --i) task(ctx, i);
}
std::vector<uint8_t> Read(const char* path) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") || !f) return {};
    _fseeki64(f, 0, SEEK_END);
    const auto size = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    const bool okay = fread(bytes.data(), 1, bytes.size(), f) == bytes.size();
    fclose(f);
    return okay ? bytes : std::vector<uint8_t>{};
}
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    auto bytes = Read(argv[1]);
    if (bytes.empty()) return 3;
    void* aligned = _aligned_malloc(bytes.size(), 64);
    if (!aligned) return 4;
    std::memcpy(aligned, bytes.data(), bytes.size());
    sslm_model model = nullptr;
    if (sslm_model_map(aligned, bytes.size(), &model) != SSLM_OK) return 5;
    sslm_decode_params params{};
    int layers = 0;
    for (int count = 64; count >= 1; --count)
        if (sslm_decode_params_init(model, SSLM_DECODE_MODE_GREEDY, count, &params) == SSLM_OK) {
            layers = count;
            break;
        }
    if (!layers) return 6;
    sslm_config config{4, 64, layers, 0};
    const auto pool_bytes = 4 * sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, 4);
    void* pool_storage = _aligned_malloc(pool_bytes, 64);
    sslm_kv_pool pool = nullptr;
    if (sslm_kv_pool_create(model, pool_storage, pool_bytes, 4, &pool) != SSLM_OK) return 7;
    const auto ws_bytes = sslm_workspace_size(model, &config);
    void* ws_storage = _aligned_malloc(ws_bytes, 64);
    sslm_workspace ws = nullptr;
    if (sslm_workspace_create(model, &config, ws_storage, ws_bytes, &ws) != SSLM_OK) return 8;
    sslm_seq source = nullptr;
    if (sslm_seq_create(model, &pool, &source) != SSLM_OK) return 9;
    int done = 0;
    while (done < static_cast<int>(sizeof prompt / sizeof prompt[0])) {
        int32_t consumed = 0;
        if (sslm_prefill(model, source, prompt + done,
                static_cast<int32_t>(sizeof prompt / sizeof prompt[0]) - done, 64,
                SSLM_SPAN_PROMPT, ws, &consumed) != SSLM_OK || !consumed) return 10;
        done += consumed;
    }
    std::vector<uint8_t> blob(sslm_seq_state_size(model));
    size_t written = blob.size();
    if (sslm_seq_save(source, blob.data(), &written) != SSLM_OK) return 11;
    blob.resize(written);
    sslm_seq_release(source);
    int calls = 0;
    sslm_parallel_for hook{};
    hook.run = &Reverse;
    hook.host_ctx = &calls;
    hook.max_tasks = 4;
    if (sslm_workspace_set_parallel_for(ws, &hook) != SSLM_OK) return 12;
    int32_t singles[4]{};
    for (int i = 0; i < 4; ++i) {
        sslm_seq seq = nullptr;
        if (sslm_seq_restore(model, &pool, blob.data(), blob.size(), &seq) != SSLM_OK) return 13;
        if (sslm_decode_step_v2(model, &seq, 1, &params, ws, &singles[i]) != SSLM_OK) return 14;
        sslm_seq_release(seq);
    }
    const int calls_after_singles = calls;
    sslm_seq seqs[4]{};
    for (auto& seq : seqs)
        if (sslm_seq_restore(model, &pool, blob.data(), blob.size(), &seq) != SSLM_OK) return 15;
    int32_t batch[4]{};
    const auto status = sslm_decode_step_v2(model, seqs, 4, &params, ws, batch);
    for (auto seq : seqs) sslm_seq_release(seq);
    if (status != SSLM_OK || calls_after_singles < 4 || calls - calls_after_singles < 4) {
        std::fprintf(stderr, "FAIL batch status=%d single_calls=%d batch_calls=%d\n",
                     static_cast<int>(status), calls_after_singles, calls - calls_after_singles);
        return 16;
    }
    for (int i = 0; i < 4; ++i) if (batch[i] != singles[i]) {
        std::fprintf(stderr, "FAIL batch index=%d got=%d single=%d\n", i, batch[i], singles[i]);
        return 17;
    }
    if (sslm_workspace_set_parallel_for(ws, nullptr) != SSLM_OK) return 18;
    sslm_workspace_destroy(ws);
    sslm_kv_pool_destroy(pool);
    sslm_model_unmap(model);
    _aligned_free(ws_storage);
    _aligned_free(pool_storage);
    _aligned_free(aligned);
    std::printf("PASS batch four tokens=%d,%d,%d,%d hook_calls=%d\n",
                batch[0], batch[1], batch[2], batch[3], calls);
    return 0;
}
