// T-2851 rows 1, 3 and 9: workspace isolation, concurrent decodes,
// clear/reinstall, save/restore under hook changes, and host_ctx lifetime.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <new>
#include <thread>
#include <vector>
#include <malloc.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "superslm/parallel_for.h"
#include "superslm/sslm_abi.h"

extern "C" sslm_status sslm_workspace_set_parallel_for(sslm_workspace, const sslm_parallel_for*);

namespace {
constexpr int32_t kPrompt[] = {40, 1035, 1075, 311, 3695, 264, 2820, 60108, 11, 4486, 13};
struct HookState {
    std::atomic<int> calls{0};
    std::atomic<int> wrong_thread{0};
    std::atomic<DWORD> caller{0};
};
void Inline(void* host, int32_t count, sslm_task_fn task, void* ctx) {
    auto& h = *static_cast<HookState*>(host);
    h.calls.fetch_add(1);
    if (GetCurrentThreadId() != h.caller.load()) h.wrong_thread.fetch_add(1);
    for (int32_t i = count - 1; i >= 0; --i) task(ctx, i);
}
sslm_parallel_for Hook(HookState* state) {
    sslm_parallel_for pf{};
    pf.run = &Inline;
    pf.host_ctx = state;
    pf.max_tasks = 4;
    return pf;
}
std::vector<uint8_t> Read(const char* path) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") || !f) return {};
    _fseeki64(f, 0, SEEK_END);
    auto n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    std::vector<uint8_t> b(static_cast<size_t>(n));
    bool ok = fread(b.data(), 1, b.size(), f) == b.size();
    fclose(f);
    return ok ? b : std::vector<uint8_t>{};
}
struct Workspace {
    void* storage = nullptr;
    sslm_workspace handle = nullptr;
    bool Create(sslm_model model, int layers) {
        sslm_config cfg{1, 64, layers, 0};
        auto size = sslm_workspace_size(model, &cfg);
        storage = _aligned_malloc(size, 64);
        return sslm_workspace_create(model, &cfg, storage, size, &handle) == SSLM_OK;
    }
    void Destroy() {
        if (handle) sslm_workspace_destroy(handle);
        _aligned_free(storage);
    }
};
int Decode(sslm_model model, sslm_kv_pool* pool, const std::vector<uint8_t>& blob,
           sslm_workspace ws, const sslm_decode_params& params) {
    sslm_seq seq = nullptr;
    if (sslm_seq_restore(model, pool, blob.data(), blob.size(), &seq) != SSLM_OK) return -100;
    int32_t token = -99;
    const auto status = sslm_decode_step_v2(model, &seq, 1, &params, ws, &token);
    sslm_seq_release(seq);
    return status == SSLM_OK ? token : -101;
}
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    auto file = Read(argv[1]);
    if (file.empty()) return 3;
    void* bytes = _aligned_malloc(file.size(), 64);
    std::memcpy(bytes, file.data(), file.size());
    sslm_model model = nullptr;
    if (sslm_model_map(bytes, file.size(), &model) != SSLM_OK) return 4;
    sslm_decode_params params{};
    int layers = 0;
    for (int L = 64; L >= 1; --L)
        if (sslm_decode_params_init(model, SSLM_DECODE_MODE_GREEDY, L, &params) == SSLM_OK) {
            layers = L; break;
        }
    if (!layers) return 5;
    const auto pool_size = 4 * sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, 4);
    void* pool_bytes = _aligned_malloc(pool_size, 64);
    sslm_kv_pool pool = nullptr;
    if (sslm_kv_pool_create(model, pool_bytes, pool_size, 4, &pool) != SSLM_OK) return 6;
    Workspace a, b;
    if (!a.Create(model, layers) || !b.Create(model, layers)) return 7;
    sslm_seq source = nullptr;
    if (sslm_seq_create(model, &pool, &source) != SSLM_OK) return 8;
    int done = 0;
    while (done < static_cast<int>(std::size(kPrompt))) {
        int32_t consumed = 0;
        if (sslm_prefill(model, source, kPrompt + done,
                static_cast<int32_t>(std::size(kPrompt)) - done, 64,
                SSLM_SPAN_PROMPT, a.handle, &consumed) != SSLM_OK || !consumed) return 9;
        done += consumed;
    }
    std::vector<uint8_t> blob(sslm_seq_state_size(model));
    size_t written = blob.size();
    if (sslm_seq_save(source, blob.data(), &written) != SSLM_OK) return 10;
    blob.resize(written);
    sslm_seq_release(source);
    const int serial = Decode(model, &pool, blob, a.handle, params);
    if (serial < 0) return 11;

    sslm_schema schema = SSLM_SCHEMA_NONE;
    if (sslm_schema_lookup(model, "potion_shop_order", &schema) != SSLM_OK) return 21;
    sslm_seq schema_source = nullptr;
    if (sslm_seq_create(model, &pool, &schema_source) != SSLM_OK ||
        sslm_seq_set_schema(schema_source, schema) != SSLM_OK) return 22;
    done = 0;
    while (done < static_cast<int>(std::size(kPrompt))) {
        int32_t consumed = 0;
        if (sslm_prefill(model, schema_source, kPrompt + done,
                static_cast<int32_t>(std::size(kPrompt)) - done, 64,
                SSLM_SPAN_PROMPT, a.handle, &consumed) != SSLM_OK || !consumed) return 23;
        done += consumed;
    }
    int32_t content_token = -1;
    if (sslm_decode_step_v2(model, &schema_source, 1, &params, a.handle,
                            &content_token) != SSLM_OK || content_token < 0) return 24;
    sslm_seq_release(schema_source);

    HookState state_a, state_b;
    auto hook_a = Hook(&state_a), hook_b = Hook(&state_b);
    if (sslm_workspace_set_parallel_for(a.handle, &hook_a) != SSLM_OK ||
        sslm_workspace_set_parallel_for(b.handle, &hook_b) != SSLM_OK) return 12;
    sslm_seq saved_under_hook = nullptr;
    if (sslm_seq_restore(model, &pool, blob.data(), blob.size(), &saved_under_hook) != SSLM_OK)
        return 30;
    std::vector<uint8_t> hook_blob(blob.size());
    size_t hook_written = hook_blob.size();
    const auto save_status = sslm_seq_save(saved_under_hook, hook_blob.data(), &hook_written);
    sslm_seq_release(saved_under_hook);
    if (save_status != SSLM_OK || hook_written != blob.size() || hook_blob != blob) {
        std::fprintf(stderr, "FAIL CPU hook leaked into save blob\n");
        return 31;
    }
    sslm_prefix prefix = nullptr;
    if (sslm_prefix_begin(model, &pool, &prefix) != SSLM_OK) return 25;
    int32_t prefix_consumed = 0;
    if (sslm_prefix_prefill(model, prefix, kPrompt, 1, 64, SSLM_SPAN_PROMPT,
                            a.handle, &prefix_consumed) != SSLM_OK || prefix_consumed != 1 ||
        state_a.calls.load() || state_b.calls.load()) return 26;
    sslm_prefix_release(prefix);
    sslm_seq schema_target = nullptr;
    if (sslm_seq_create(model, &pool, &schema_target) != SSLM_OK ||
        sslm_seq_set_schema(schema_target, schema) != SSLM_OK) return 27;
    done = 0;
    while (done < static_cast<int>(std::size(kPrompt))) {
        int32_t consumed = 0;
        if (sslm_prefill(model, schema_target, kPrompt + done,
                static_cast<int32_t>(std::size(kPrompt)) - done, 64,
                SSLM_SPAN_PROMPT, a.handle, &consumed) != SSLM_OK || !consumed) return 28;
        done += consumed;
    }
    int32_t content_consumed = 0;
    if (sslm_prefill(model, schema_target, &content_token, 1, 64,
                     SSLM_SPAN_SCHEMA_CONTENT, a.handle, &content_consumed) != SSLM_OK ||
        content_consumed != 1 || state_a.calls.load() || state_b.calls.load()) {
        std::fprintf(stderr, "FAIL schema-content prefill hook calls a=%d b=%d consumed=%d\n",
                     state_a.calls.load(), state_b.calls.load(), content_consumed);
        return 29;
    }
    sslm_seq_release(schema_target);
    int token_a = -1, token_b = -1;
    std::thread first([&] {
        state_a.caller.store(GetCurrentThreadId());
        token_a = Decode(model, &pool, blob, a.handle, params);
    });
    std::thread second([&] {
        state_b.caller.store(GetCurrentThreadId());
        token_b = Decode(model, &pool, blob, b.handle, params);
    });
    first.join(); second.join();
    if (token_a != serial || token_b != serial || !state_a.calls.load() ||
        !state_b.calls.load() || state_a.wrong_thread.load() || state_b.wrong_thread.load())
        return 13;
    const int calls_a = state_a.calls.load();
    const int calls_b = state_b.calls.load();
    if (sslm_workspace_set_parallel_for(b.handle, nullptr) != SSLM_OK) return 14;
    if (Decode(model, &pool, blob, b.handle, params) != serial ||
        state_a.calls.load() != calls_a || state_b.calls.load() != calls_b) return 15;
    if (sslm_workspace_set_parallel_for(a.handle, nullptr) != SSLM_OK) return 16;
    state_a.caller.store(GetCurrentThreadId());
    if (sslm_workspace_set_parallel_for(a.handle, &hook_a) != SSLM_OK ||
        Decode(model, &pool, blob, a.handle, params) != serial ||
        state_a.calls.load() <= calls_a) return 17;

    void* page = VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!page) return 18;
    auto* owned = new (page) HookState;
    owned->caller.store(GetCurrentThreadId());
    auto transient = Hook(owned);
    if (sslm_workspace_set_parallel_for(a.handle, &transient) != SSLM_OK ||
        sslm_workspace_set_parallel_for(a.handle, nullptr) != SSLM_OK) return 19;
    owned->~HookState();
    DWORD prior = 0;
    if (!VirtualProtect(page, 4096, PAGE_NOACCESS, &prior)) return 20;
    a.Destroy();
    VirtualFree(page, 0, MEM_RELEASE);
    b.Destroy();
    sslm_kv_pool_destroy(pool);
    sslm_model_unmap(model);
    _aligned_free(pool_bytes);
    _aligned_free(bytes);
    std::printf("PASS hook lifecycle serial=%d concurrent=2 isolated=1 clear=1 reinstall=1\n", serial);
    return 0;
}
