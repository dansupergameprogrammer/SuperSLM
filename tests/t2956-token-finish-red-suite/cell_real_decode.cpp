// T-2851 row 10: real-artifact decode, token identity, and thread ownership.
// Build twice: T2956_BASELINE links the v1.6.0 libraries; T2956_CANDIDATE
// links the checkout under test. The runner compares complete token lists.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <malloc.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#include "superslm/sslm_abi.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_g5_bridge.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#if defined(T2956_ALL_MASKED)
extern "C" void ArmCpuFinishDegenerateLogitRowInjection();
extern "C" void ArmGpuFinishDegenerateLogitRowInjection();
#endif
#if defined(T2956_CANDIDATE)
#include "superslm/parallel_for.h"
extern "C" sslm_status sslm_workspace_set_parallel_for(sslm_workspace, const sslm_parallel_for*);
SslmGpuStatus sslm_gpu_context_set_host_parallel_for(SslmGpuContext*, const sslm_parallel_for*) noexcept;
void ArmGpuDeviceLogitsReadbackOverride(int32_t row, int64_t value) noexcept;
void SslmGpuAllocCounterResetForTest() noexcept;
uint32_t SslmGpuAllocCountForTest() noexcept;
#endif

namespace {
constexpr int32_t kPinnedPrompt[] = {40, 1035, 1075, 311, 3695, 264, 2820, 60108, 11, 4486, 13};
constexpr const char* kPrompt = "I would like to buy a health potion, please.";

std::vector<uint8_t> Read(const char* path) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") || !f) return {};
    _fseeki64(f, 0, SEEK_END);
    auto n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes(static_cast<size_t>(n));
    bool ok = fread(bytes.data(), 1, bytes.size(), f) == bytes.size();
    fclose(f);
    return ok ? bytes : std::vector<uint8_t>{};
}

struct ThreadOrigin {
    uintptr_t start = 0;
    std::string module = "<unresolved>";
};
using ThreadSnapshot = std::map<DWORD, ThreadOrigin>;

ThreadOrigin Origin(DWORD id) {
    ThreadOrigin origin;
    HANDLE thread = OpenThread(THREAD_QUERY_INFORMATION, FALSE, id);
    if (!thread) return origin;
    using QueryThread = LONG (NTAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);
    const auto query = reinterpret_cast<QueryThread>(GetProcAddress(
        GetModuleHandleA("ntdll.dll"), "NtQueryInformationThread"));
    void* start = nullptr;
    if (query && query(thread, 9 /* ThreadQuerySetWin32StartAddress */,
                       &start, sizeof start, nullptr) >= 0) {
        origin.start = reinterpret_cast<uintptr_t>(start);
        HMODULE module = nullptr;
        if (start && GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCSTR>(start), &module)) {
            char path[MAX_PATH]{};
            if (GetModuleFileNameA(module, path, MAX_PATH)) origin.module = path;
        }
    }
    CloseHandle(thread);
    return origin;
}

ThreadSnapshot Threads() {
    ThreadSnapshot result;
    HANDLE h = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (h == INVALID_HANDLE_VALUE) return result;
    THREADENTRY32 e{};
    e.dwSize = sizeof e;
    if (Thread32First(h, &e)) do {
        if (e.th32OwnerProcessID == GetCurrentProcessId())
            result.emplace(e.th32ThreadID, Origin(e.th32ThreadID));
    } while (Thread32Next(h, &e));
    CloseHandle(h);
    return result;
}

struct Hook {
    DWORD caller = 0;
    ThreadSnapshot before;
    ThreadSnapshot observed_new;
    std::set<DWORD> new_threads;
    std::set<DWORD> disappeared;
    int calls = 0;
    int tasks = 0;
    int wrong_thread = 0;
    int excess_tasks = 0;
    int changed_census = 0;
    int max_tasks = 0;
    bool spawn_probe = false;
    bool probe_fired = false;
    std::string malformed;
};

void CheckThreads(Hook& hook, const ThreadSnapshot& now, bool in_hook) {
    bool new_at_snapshot = false;
    for (const auto& [id, _] : now) {
        if (!hook.before.contains(id)) {
            hook.new_threads.insert(id);
            hook.observed_new.emplace(id, now.at(id));
            new_at_snapshot = true;
        }
    }
    for (const auto& [id, _] : hook.before)
        if (!now.contains(id)) hook.disappeared.insert(id);
    if (in_hook && new_at_snapshot) ++hook.changed_census;
}

void ReportThreads(Hook& hook, const ThreadSnapshot& after) {
    CheckThreads(hook, after, false);
    for (DWORD id : hook.disappeared) {
        const auto& origin = hook.before.at(id);
        std::printf("THREAD_EXIT id=%lu start=%p module=%s\n", id,
                    reinterpret_cast<void*>(origin.start), origin.module.c_str());
    }
    for (DWORD id : hook.new_threads) {
        const auto& origin = hook.observed_new.at(id);
        std::printf("THREAD_NEW id=%lu start=%p module=%s\n", id,
                    reinterpret_cast<void*>(origin.start), origin.module.c_str());
    }
    const bool same_ids = hook.before.size() == after.size() &&
        std::all_of(hook.before.begin(), hook.before.end(),
                    [&](const auto& entry) { return after.contains(entry.first); });
    std::printf("THREADS before=%zu after=%zu same=%d new=%zu disappeared=%zu "
                "hook_calls=%d task_calls=%d wrong_caller=%d bad_count=%d changed_at_hook=%d\n",
                hook.before.size(), after.size(), same_ids,
                hook.new_threads.size(), hook.disappeared.size(), hook.calls, hook.tasks,
                hook.wrong_thread, hook.excess_tasks, hook.changed_census);
}

#if defined(T2956_CANDIDATE)
void RunInline(void* host, int32_t count, sslm_task_fn task, void* ctx) {
    auto& h = *static_cast<Hook*>(host);
    ++h.calls;
    h.tasks += count;
    if (GetCurrentThreadId() != h.caller) ++h.wrong_thread;
    if (count < 1 || count > h.max_tasks) ++h.excess_tasks;
    CheckThreads(h, Threads(), true);
    if (h.spawn_probe && !h.probe_fired) {
        h.probe_fired = true;
        std::atomic<bool> ready{false}, done{false};
        std::thread extra([&] {
            ready.store(true, std::memory_order_release);
            while (!done.load(std::memory_order_acquire)) std::this_thread::yield();
        });
        while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
        CheckThreads(h, Threads(), true); // must-reject: the new thread is alive here
        done.store(true, std::memory_order_release);
        extra.join();
    }
    if (h.malformed == "omit0") {
        for (int32_t i = 1; i < count; ++i) task(ctx, i);
    } else if (h.malformed == "omitlast") {
        for (int32_t i = 0; i < count - 1; ++i) task(ctx, i);
    } else if (h.malformed == "dupinplace") {
        task(ctx, 0); task(ctx, 0);
        for (int32_t i = 2; i < count; ++i) task(ctx, i);
    } else if (h.malformed == "dupextra") {
        for (int32_t i = 0; i < count; ++i) task(ctx, i);
        task(ctx, 1);
    } else if (h.malformed == "outofrange") {
        for (int32_t i = 0; i < count; ++i) task(ctx, i);
        task(ctx, count);
    } else if (h.malformed == "concurrentdup") {
        std::atomic<int> arrived{0};
        std::atomic<bool> go{false};
        auto twin = [&] {
            arrived.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            task(ctx, 0);
        };
        std::thread a(twin), b(twin);
        while (arrived.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        go.store(true, std::memory_order_release);
        a.join(); b.join();
        for (int32_t i = 1; i < count; ++i) task(ctx, i);
    } else {
        for (int32_t i = count - 1; i >= 0; --i) task(ctx, i);
    }
}

sslm_parallel_for MakeHook(Hook& h) {
    sslm_parallel_for pf{};
    pf.run = &RunInline;
    pf.host_ctx = &h;
    pf.max_tasks = h.max_tasks;
    return pf;
}
#endif

std::vector<int32_t> Prompt(sslm_model model) {
    int32_t ids[512]{};
    int32_t n = 512;
    if (sslm_tokenize(model, kPrompt, ids, &n) == SSLM_OK)
        return {ids, ids + n};
    return {std::begin(kPinnedPrompt), std::end(kPinnedPrompt)};
}

std::vector<uint8_t> CpuBlob(sslm_model model, sslm_seq seq) {
    std::vector<uint8_t> blob(sslm_seq_state_size(model));
    size_t n = blob.size();
    if (sslm_seq_save(seq, blob.data(), &n) != SSLM_OK) return {};
    blob.resize(n);
    return blob;
}

std::vector<uint8_t> GpuBlob(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
    size_t n = 0;
    (void)sslm_gpu_seq_save(ctx, seq, nullptr, &n);
    if (!n) return {};
    std::vector<uint8_t> blob(n);
    if (sslm_gpu_seq_save(ctx, seq, blob.data(), &n) != SslmGpuStatus::SSLM_OK) return {};
    blob.resize(n);
    return blob;
}

int Cpu(sslm_model model, const std::vector<int32_t>& prompt, const char* schema_name,
        int count, int max_tasks, const char* malformed) {
    sslm_decode_params p{};
    int layers = 0;
    char damped_value[2]{};
    const bool damped = GetEnvironmentVariableA("T2956_DAMPED", damped_value,
                                                sizeof damped_value) != 0;
    for (int L = 64; L >= 1; --L) {
        if (sslm_decode_params_init(model, SSLM_DECODE_MODE_GREEDY, L, &p) == SSLM_OK) {
            layers = L;
            break;
        }
    }
    if (!layers) return 20;
    if (damped && sslm_decode_params_init(model, SSLM_DECODE_MODE_DAMPED_GREEDY,
                                          layers, &p) != SSLM_OK) return 19;
    sslm_config cfg{1, 64, layers, 0};
    const size_t pb = sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, 1);
    void* pool_bytes = _aligned_malloc(pb, 64);
    sslm_kv_pool pool = nullptr;
    if (sslm_kv_pool_create(model, pool_bytes, pb, 1, &pool) != SSLM_OK) return 21;
    const size_t wb = sslm_workspace_size(model, &cfg);
    void* workspace_bytes = _aligned_malloc(wb, 64);
    sslm_workspace ws = nullptr;
    if (sslm_workspace_create(model, &cfg, workspace_bytes, wb, &ws) != SSLM_OK) return 22;
    Hook hook{};
    hook.caller = GetCurrentThreadId();
    hook.before = Threads();
    hook.max_tasks = max_tasks;
    hook.malformed = malformed;
    hook.spawn_probe = GetEnvironmentVariableA("T2956_THREAD_SPAWN_PROBE", nullptr, 0) != 0;
#if defined(T2956_CANDIDATE)
    sslm_parallel_for pf = MakeHook(hook);
    if (sslm_workspace_set_parallel_for(ws, max_tasks ? &pf : nullptr) != SSLM_OK) return 26;
#else
    if (max_tasks) return 26;
#endif
    sslm_seq seq = nullptr;
    if (sslm_seq_create(model, &pool, &seq) != SSLM_OK) return 23;
    if (std::strcmp(schema_name, "-") != 0) {
        sslm_schema schema = SSLM_SCHEMA_NONE;
        if (sslm_schema_lookup(model, schema_name, &schema) != SSLM_OK ||
            sslm_seq_set_schema(seq, schema) != SSLM_OK) return 24;
    }
    int done = 0;
    while (done < static_cast<int>(prompt.size())) {
        int32_t consumed = 0;
        const sslm_status ps = sslm_prefill(model, seq, prompt.data() + done,
                         static_cast<int32_t>(prompt.size()) - done, 64,
                         SSLM_SPAN_PROMPT, ws, &consumed);
        if (ps != SSLM_OK || consumed <= 0) {
            std::fprintf(stderr, "FAIL prefill status=%d consumed=%d prompt=%zu done=%d\n",
                         static_cast<int>(ps), consumed, prompt.size(), done);
            return 25;
        }
        done += consumed;
    }
    if (hook.calls || hook.tasks) return 53; // prefill may not read the hook
#if defined(T2956_ALL_MASKED)
    if (GetEnvironmentVariableA("T2956_ALL_MASKED", nullptr, 0)) {
        const auto before = CpuBlob(model, seq);
        if (before.size() < 52) return 61;
        ArmCpuFinishDegenerateLogitRowInjection();
        int32_t out = -9;
        const auto status = sslm_decode_step_v2(model, &seq, 1, &p, ws, &out);
        const auto after = CpuBlob(model, seq);
        const bool walk_same = after.size() >= 52 &&
            std::memcmp(before.data() + 48, after.data() + 48, 4) == 0;
        std::printf("ALL_MASKED backend=cpu status=%d token=%d walk_same=%d\n",
                    static_cast<int>(status), out, walk_same);
        if (status != SSLM_OK || out != -2 || !walk_same) {
            std::printf("FAIL all-masked CPU status=%d token=%d walk_same=%d\n",
                        static_cast<int>(status), out, walk_same);
            return 62;
        }
        return 0;
    }
#endif
    if (malformed[0]) {
#if defined(T2956_CANDIDATE)
        const auto before = CpuBlob(model, seq);
        if (before.empty()) return 45;
        const int repetitions = hook.malformed == "concurrentdup" ? 1000 : 1;
        for (int r = 0; r < repetitions; ++r) {
            int32_t rejected_token = -99;
            const auto status = sslm_decode_step_v2(model, &seq, 1, &p, ws, &rejected_token);
            if (status != SSLM_INVALID_ARGUMENT || CpuBlob(model, seq) != before) {
                std::fprintf(stderr, "FAIL malformed CPU %s rep=%d status=%d token=%d\n",
                             malformed, r, static_cast<int>(status), rejected_token);
                return 46;
            }
        }
        hook.malformed.clear();
        int32_t retry = -99;
        if (sslm_decode_step_v2(model, &seq, 1, &p, ws, &retry) != SSLM_OK || retry < 0)
            return 47;
        std::printf("MALFORMED backend=cpu case=%s repetitions=%d status=SSLM_INVALID_ARGUMENT "
                    "unchanged=1 retry=%d\nTOKENS %d\n", malformed, repetitions, retry, retry);
        return 0;
#else
        return 48;
#endif
    }
    std::vector<int32_t> tokens;
    for (int i = 0; i < count; ++i) {
        int32_t out = -9;
        if (sslm_decode_step_v2(model, &seq, 1, &p, ws, &out) != SSLM_OK) return 27;
        if (out == -2) break;
        if (out < 0) return 28;
        tokens.push_back(out);
    }
    const auto after = Threads();
    ReportThreads(hook, after);
    if (!hook.new_threads.empty() || hook.wrong_thread || hook.excess_tasks ||
        (max_tasks > 1 && hook.calls == 0)) return 29;
    std::printf("TOKENS");
    for (int32_t t : tokens) std::printf(" %d", t);
    std::printf("\n");
    sslm_seq_release(seq);
    sslm_workspace_destroy(ws);
    sslm_kv_pool_destroy(pool);
    _aligned_free(workspace_bytes);
    _aligned_free(pool_bytes);
    return 0;
}

int Gpu(const uint8_t* bytes, size_t size, const std::vector<int32_t>& prompt,
        const char* schema_name, int count, int max_tasks, bool device_head, const char* malformed) {
    superslm::SslmModelView view;
    std::string err;
    if (superslm::SslmModel::Load(bytes, size, view, &err) != superslm::SslmModelStatus::Ok) return 30;
    GpuContextConfig cc{};
    char shader_override[MAX_PATH]{};
    if (GetEnvironmentVariableA("T2956_SHADER_DIR", shader_override, sizeof shader_override))
        cc.shader_dir = shader_override;
    GpuResidencyConfig rc{};
#if defined(T2956_CANDIDATE)
#if defined(T2956_STUB)
    if (device_head) rc.reserved = 1; // compile/link check against v1.6.0 only
#else
    if (device_head) rc.flags = SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE;
#endif
#else
    if (device_head) return 31;
#endif
    SslmGpuContext* ctx = nullptr;
    if (sslm_gpu_context_create(cc, &ctx) != SslmGpuStatus::SSLM_OK) return 32;
    SslmGpuModelHandle* gm = nullptr;
    if (sslm_gpu_model_map(ctx, &view, rc, &gm) != SslmGpuStatus::SSLM_OK) return 33;
    Hook hook{};
    hook.caller = GetCurrentThreadId();
    hook.before = Threads();
    hook.max_tasks = max_tasks;
    hook.malformed = malformed;
    hook.spawn_probe = GetEnvironmentVariableA("T2956_THREAD_SPAWN_PROBE", nullptr, 0) != 0;
#if defined(T2956_CANDIDATE)
    sslm_parallel_for pf = MakeHook(hook);
    if (sslm_gpu_context_set_host_parallel_for(ctx, max_tasks ? &pf : nullptr) !=
        SslmGpuStatus::SSLM_OK) return 34;
#else
    if (max_tasks) return 34;
#endif
    const int32_t schema = std::strcmp(schema_name, "-") == 0 ? -1 :
        SslmGpuSchemaLookupForG5Bridge(gm, schema_name);
    if (std::strcmp(schema_name, "-") != 0 && schema < 0) return 35;
    SslmGpuSequenceHandle* seq = nullptr;
    if (sslm_gpu_seq_create(ctx, gm, static_cast<int64_t>(view.config.context_cap), &seq) !=
        SslmGpuStatus::SSLM_OK) return 36;
    if (schema >= 0 && SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, schema) !=
        SslmGpuStatus::SSLM_OK) return 37;
    const uint32_t budget = superslm_gpu::kDispatchesPerLayer *
        static_cast<uint32_t>(view.config.num_hidden_layers);
    if (SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prompt.data(),
            static_cast<int32_t>(prompt.size()), budget) != SslmGpuStatus::SSLM_OK) return 38;
    if (hook.calls || hook.tasks) return 54; // GPU prefill may not read the hook
#if defined(T2956_ALL_MASKED)
    if (GetEnvironmentVariableA("T2956_ALL_MASKED", nullptr, 0)) {
        const auto before = GpuBlob(ctx, seq);
        if (before.size() < 52) return 63;
        ArmGpuFinishDegenerateLogitRowInjection();
        int32_t out = -9;
        const auto status = SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, &out);
        const auto after = GpuBlob(ctx, seq);
        const bool walk_same = after.size() >= 52 &&
            std::memcmp(before.data() + 48, after.data() + 48, 4) == 0;
        std::printf("ALL_MASKED backend=gpu status=%u token=%d walk_same=%d\n",
                    static_cast<unsigned>(status), out, walk_same);
        if (status != SslmGpuStatus::SSLM_OK || out != -2 || !walk_same) {
            std::printf("FAIL all-masked GPU status=%u token=%d walk_same=%d\n",
                        static_cast<unsigned>(status), out, walk_same);
            return 64;
        }
        return 0;
    }
#endif
    char overflow_value[2]{};
    const bool overflow = GetEnvironmentVariableA("T2956_OVERFLOW", overflow_value,
                                                  sizeof overflow_value) != 0;
    if (overflow) {
#if defined(T2956_CANDIDATE)
        if (!device_head) return 55;
        const auto before = GpuBlob(ctx, seq);
        if (before.empty()) return 56;
        ArmGpuDeviceLogitsReadbackOverride(17, 2147483648LL);
        int32_t refused_token = -99;
        const auto refused = SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, &refused_token);
        if (refused != SslmGpuStatus::SSLM_SEQUENCE_REJECTED || GpuBlob(ctx, seq) != before) {
            std::fprintf(stderr, "FAIL overflow status=%u token=%d\n",
                         static_cast<unsigned>(refused), refused_token);
            return 57;
        }
        int32_t retry = -99;
        if (SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, &retry) != SslmGpuStatus::SSLM_OK || retry < 0)
            return 58;
        std::printf("OVERFLOW status=SSLM_SEQUENCE_REJECTED unchanged=1 retry=%d\nTOKENS %d\n",
                    retry, retry);
        return 0;
#else
        return 59;
#endif
    }
    if (malformed[0]) {
#if defined(T2956_CANDIDATE)
        const auto before = GpuBlob(ctx, seq);
        if (before.empty()) return 49;
        const int repetitions = hook.malformed == "concurrentdup" ? 1000 : 1;
        for (int r = 0; r < repetitions; ++r) {
            int32_t rejected_token = -99;
            const auto status = SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, &rejected_token);
            if (status != static_cast<SslmGpuStatus>(22) || GpuBlob(ctx, seq) != before) {
                std::fprintf(stderr, "FAIL malformed GPU %s rep=%d status=%u token=%d\n",
                             malformed, r, static_cast<unsigned>(status), rejected_token);
                return 50;
            }
        }
        hook.malformed.clear();
        int32_t retry = -99;
        if (SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, &retry) != SslmGpuStatus::SSLM_OK || retry < 0)
            return 51;
        std::printf("MALFORMED backend=gpu case=%s repetitions=%d status=22 "
                    "unchanged=1 retry=%d\nTOKENS %d\n", malformed, repetitions, retry, retry);
        return 0;
#else
        return 52;
#endif
    }
    std::vector<int32_t> tokens;
    int32_t out = -9;
    for (int i = 0; i < count; ++i) {
        if (i) {
            if (sslm_gpu_seq_embed_token(ctx, seq, out) != SslmGpuStatus::SSLM_OK) return 39;
            if (sslm_decode_step_gpu(ctx, seq, nullptr, budget) != SslmGpuStatus::SSLM_OK) return 40;
            int32_t ready = 0;
            SslmGpuStatus result = SslmGpuStatus::SSLM_OK;
            if (sslm_gpu_ready(ctx, seq, 1, &ready, &result) != SslmGpuStatus::SSLM_OK ||
                !ready || result != SslmGpuStatus::SSLM_OK) return 41;
        }
#if defined(T2956_CANDIDATE)
        if (device_head) SslmGpuAllocCounterResetForTest();
#endif
        if (SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, &out) != SslmGpuStatus::SSLM_OK) return 42;
#if defined(T2956_CANDIDATE)
        if (device_head && SslmGpuAllocCountForTest() != 0) {
            std::fprintf(stderr, "FAIL finish device allocations=%u\n",
                         SslmGpuAllocCountForTest());
            return 60;
        }
#endif
        if (out == -2) break;
        if (out < 0) return 43;
        tokens.push_back(out);
    }
    const auto after = Threads();
    ReportThreads(hook, after);
    if (!hook.new_threads.empty() || hook.wrong_thread || hook.excess_tasks ||
        (max_tasks > 1 && !device_head && hook.calls == 0)) return 44;
    std::printf("TOKENS");
    for (int32_t t : tokens) std::printf(" %d", t);
    std::printf("\n");
    sslm_gpu_seq_release(ctx, seq);
    sslm_gpu_model_unmap(ctx, gm);
    sslm_gpu_context_destroy(ctx);
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 7 || argc > 9) {
        std::fprintf(stderr, "usage: cell_real_decode cpu|gpu MODEL SCHEMA|- COUNT MAX_TASKS DEVICE_HEAD [PROMPT_CSV] [MALFORMED]\n");
        return 2;
    }
    const auto file = Read(argv[2]);
    if (file.empty()) return 3;
    void* aligned = _aligned_malloc(file.size(), 64);
    if (!aligned) return 4;
    std::memcpy(aligned, file.data(), file.size());
    sslm_model model = nullptr;
    if (sslm_model_map(aligned, file.size(), &model) != SSLM_OK) return 5;
    auto prompt = Prompt(model);
    if (argc >= 8 && std::strcmp(argv[7], "-") != 0) {
        prompt.clear();
        const char* p = argv[7];
        while (*p) {
            char* end = nullptr;
            long value = std::strtol(p, &end, 10);
            if (end == p || value < 0 || value > INT32_MAX) return 7;
            prompt.push_back(static_cast<int32_t>(value));
            p = *end == ',' ? end + 1 : end;
            if (*end && *end != ',') return 7;
        }
        if (prompt.empty()) return 7;
    }
    const int count = std::atoi(argv[4]);
    const int tasks = std::atoi(argv[5]);
    const bool device = std::atoi(argv[6]) != 0;
    const char* malformed = argc == 9 ? argv[8] : "";
    int result = 6;
    if (std::strcmp(argv[1], "cpu") == 0 && !device)
        result = Cpu(model, prompt, argv[3], count, tasks, malformed);
    else if (std::strcmp(argv[1], "gpu") == 0)
        result = Gpu(static_cast<const uint8_t*>(aligned), file.size(), prompt,
                     argv[3], count, tasks, device, malformed);
    sslm_model_unmap(model);
    _aligned_free(aligned);
    if (result) std::fprintf(stderr, "FAIL cell_real_decode stage=%d\n", result);
    return result;
}
