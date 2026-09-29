// Route P hook state, RUNTIME witness at the engine (plan rev 6, cell 10.0 H; C11 witness kind (a)).
// Links the plugin's own SuperSLMFinishHook.cpp (compiled verbatim against UE shims) and the engine
// under test. Installs exactly the value Make(k) produces -- reserved, max_tasks, host_ctx unchanged --
// on a real CPU workspace, with `run` wrapped only to count the engine's calls to it, the way both
// plugin CPU install sites do (SuperSLMSubsystem.cpp:1608-1609, SuperSLMCalibrateCostsCommand.cpp:96-97).
// Then: a 64-token prefill in ONE call (M = 64 >= 8, the most favourable shape), counting `run` calls; rev 10: a
// third argument sets the prefill batch, so merge acceptance reads H at the reopening need's B (M = B);
// then one decode step (the positive control: the finish is the hook's documented reader).
// Rule (plan rev 7, M6-5c: the witness's rule, and the plan's text aligned to it): H(route P) = "hooked" iff
// the engine accepted the install AND the decode control called run (the counter is live) AND the prefill
// count is > 0. The `reserved` bits are printed as a report and decide nothing. An install the engine refuses
// reads "not hooked" (fail closed). A decode count of 0 at an installed k voids the reading.
// -DWITNESS_DEAD_COUNTER builds the dead-counter fixture: the hook is installed unwrapped, so the counter
// never moves; the witness must then read VOID, never "not hooked" (coverage-mutants-rev6 M6-5e).
#include "SuperSLMFinishHook.h"
#include "superslm/sslm_abi.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <vector>

static long g_runs = 0;
static sslm_run_tasks_fn g_inner = nullptr;
static void CountingRun(void* host, int32_t n, sslm_task_fn task, void* ctx) { ++g_runs; g_inner(host, n, task, ctx); }
static void Die(const char* w, int st) { std::printf("FATAL %s status=%d\n", w, st); std::exit(3); }

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <artifact.sslm> <FinishParallelTasks> [prefill batch B, default 64]\n", argv[0]); return 2; }
    const int k = std::atoi(argv[2]);
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});
    sslm_model model = nullptr; sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
    if (st != SSLM_OK) Die("model_map", st);
    const int32_t count = argc > 3 ? std::atoi(argv[3]) : 64;   // rev 10 (adversary round 5 F2): merge acceptance reads H at the reopening B
    const size_t kvb = sslm_kv_block_size(model), kvo = sslm_kv_pool_overhead_size(model, 1);
    std::vector<uint8_t> pool_raw(kvb + kvo + SSLM_ABI_ALIGNMENT_BYTES); void* pa = pool_raw.data(); size_t ps = pool_raw.size();
    std::align(SSLM_ABI_ALIGNMENT_BYTES, kvb + kvo, pa, ps);
    sslm_kv_pool pool = nullptr; if ((st = sslm_kv_pool_create(model, pa, kvb + kvo, 1, &pool)) != SSLM_OK) Die("pool", st);
    sslm_config cfg{}; cfg.max_batch = 1; cfg.max_chunk_budget = count; cfg.max_layer_budget = 1;
    const size_t wsb = sslm_workspace_size(model, &cfg);
    std::vector<uint8_t> ws_raw(wsb + SSLM_ABI_ALIGNMENT_BYTES); void* wa = ws_raw.data(); size_t wsz = ws_raw.size();
    std::align(SSLM_ABI_ALIGNMENT_BYTES, wsb, wa, wsz);
    sslm_workspace ws = nullptr; if ((st = sslm_workspace_create(model, &cfg, wa, wsb, &ws)) != SSLM_OK) Die("ws", st);

    sslm_parallel_for hook = SuperSLMFinishHook::Make(k);
    const bool install = SuperSLMFinishHook::ShouldInstall(k);
    sslm_status ist = SSLM_OK;
#ifndef WITNESS_DEAD_COUNTER
    if (install) { g_inner = hook.run; hook.run = &CountingRun; ist = sslm_workspace_set_parallel_for(ws, &hook); }
#else
    if (install) { ist = sslm_workspace_set_parallel_for(ws, &hook); }
#endif
    sslm_seq seq = nullptr; if ((st = sslm_seq_create(model, &pool, &seq)) != SSLM_OK) Die("seq", st);
    std::vector<int32_t> toks(count); for (int32_t i = 0; i < count; ++i) toks[i] = 1 + (i % 200);
    int32_t consumed = 0;
    g_runs = 0;
    if ((st = sslm_prefill(model, seq, toks.data(), count, count, SSLM_SPAN_PROMPT, ws, &consumed)) != SSLM_OK) Die("prefill", st);
    const long prefill_runs = g_runs;
    g_runs = 0;
    sslm_decode_params dp{}; dp.layer_budget = 1; int32_t out = -1;
    if ((st = sslm_decode_step(model, &seq, 1, &dp, ws, &out)) != SSLM_OK) Die("decode", st);
    const long decode_runs = g_runs;
    const bool installed = install && ist == SSLM_OK;
    std::printf("route P: Make(%d) reserved=0x%x max_tasks=%d install=%s engine_install_status=%d "
                "run calls: prefill(M=%d)=%ld decode=%ld\n", k, hook.reserved, hook.max_tasks,
                install ? "yes" : "no", (int)ist, count, prefill_runs, decode_runs);
    const char* h = !install ? "not hooked (plugin installs no hook at this k)"
                  : !installed ? "not hooked (engine refused the install; fail closed)"
                  : decode_runs == 0 ? "VOID (the finish never called run: the counter is not live)"
                  : prefill_runs > 0 ? "hooked" : "not hooked";
    // rev 10 (F2): at an explicit B the reading also requires that the ONE prefill call consumed all B tokens,
    // so the GEMMs it ran were at M = B; otherwise the reading is not at B and is VOID.
    if (argc > 3 && consumed != count) h = "VOID (the prefill call did not consume B tokens: not a reading at M = B)";
    if (argc > 3) std::printf("route P: H(k=%d, B=%d) = %s  [prefill consumed %d in one call]\n", k, (int)count, h, (int)consumed);
    else std::printf("route P: H(k=%d) = %s\n", k, h);
    sslm_seq_release(seq); sslm_workspace_destroy(ws); sslm_kv_pool_destroy(pool); sslm_model_unmap(model);
    return 0;
}
