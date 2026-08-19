// T-2178 test-only instrumentation seam for T-2169's GPU-side batched-prefill chunk primitive
// (Claude/Vitruvius/t2169-gpu-batched-prefill-design-2026-08-18.md Sec5/Sec8 rung 2/Sec9;
// Claude/Curie/t2178-t2169-gpu-batched-prefill-red-suite-2026-08-18.md).
//
// WHY THIS SEAM IS THE GATING SYMBOL, NOT A CMAKE OPTION. T-2158's own precedent
// (tests/support/matmul_dispatch_instrument.h) gates its cells behind a CMake option because
// its call target (DetectBestDotRowTier) is itself an UNDECLARED symbol -- the cells cannot even
// compile without it, so the option keeps them out of the translation unit entirely. T-2169's
// two G5 bridge entry points (SslmGpuSeqPrefillPromptForG5Bridge,
// SslmGpuSeqPrefillSchemaContentForG5Bridge) are, by design (Sec5/Sec7: "no signature change"),
// ALREADY declared and defined today -- a black-box cell that only calls them compiles, links,
// and runs successfully whether or not batching exists, and would PASS vacuously before the
// feature is built (both a single N-token call and N separate 1-token calls reduce to the
// identical per-token loop today) -- exactly the "test that passes before the feature is built
// is not a red test" failure StandardsDocument.md Sec5.4/Curie.md Phase 4 name. The genuinely
// new, load-bearing claim this design makes -- Sec4's own "the batching lever is submission
// granularity" -- is invisible to a black-box content comparison and is exactly what Sec9's
// Guard-vitality row already commissions an instrument for ("instruments the actual GPU
// dispatch count issued"). This header IS that instrument, declared `extern` (not `inline`,
// unlike bad_alloc_injection.h/matmul_dispatch_instrument.h's own header-only counters) so that
// referencing it before the builder defines it fails at LINK time (LNK2019/LNK1120) --
// mirroring this exact subsystem's own established red-suite convention
// (tests/t2112-gpu-1p0-red-suite/build_link_red.bat's "RED BY LINK" disposition against the
// declared-but-undefined 1.0 API surface) rather than T-2158's compile-gate, because here the
// public surface is frozen and the compile-gate mechanism has nothing undeclared to gate on.
//
// THE COUNTERS ARE WIRED INTO PRODUCTION (T-2180 rungs 2-4; T-2184 remedy M2, Brunel fix round 1,
// D-SLM3662; T-2185 remedy N3, Brunel fix round 2, D-SLM3676 -- this paragraph corrected a
// SECOND time: the M2 rewrite named only one of the two sites for each counter and stated the
// chunk path's own per-invocation semantics as if they held everywhere). BOTH counters have TWO
// increment sites -- the single-token path (`RunLayerLoopGpuSubmit`, superslm_gpu.cpp) and the
// chunk path (`SubmitOneSubChunkToFullDepthForG5Bridge`, same file) -- and the two sites advance
// `g_gpu_chunk_dispatch_count_probe` by DIFFERENT amounts per call, because they dispatch
// different amounts of work per call:
//   1. `g_gpu_chunk_submit_count_probe` increments exactly once per command list opened and
//      submitted -- once per `RunLayerLoopGpuSubmit` call (superslm_gpu.cpp:2126, the
//      single-token/decode-step path -- a degenerate one-token "chunk" under this counter's own
//      general definition, its own site comment states this) AND once per
//      `SubmitOneSubChunkToFullDepthForG5Bridge` call (superslm_gpu.cpp:2433, the batched-chunk
//      path's own per-sub-chunk primitive) -- both guarded by
//      `SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT`.
//   2. `g_gpu_chunk_dispatch_count_probe` advances by DIFFERENT amounts at its two sites, both
//      invoking the same `RecordOneTokenFullDepthDispatchBody` (design Sec5's own D-SLM3595
//      ruling): at the CHUNK site (superslm_gpu.cpp:2514, inside
//      `SubmitOneSubChunkToFullDepthForG5Bridge`'s own per-admitted-token loop) that call always
//      passes `layers_to_record = N` (num_hidden_layers) -- full depth -- so the counter advances
//      by `N` there, once per admitted token, matching this header's own documented
//      "delta == admit_count * num_hidden_layers" contract below. At the SINGLE-TOKEN site
//      (superslm_gpu.cpp:2170, inside `RunLayerLoopGpuSubmit`) that same call passes
//      `layers_to_record = min(layer_budget, num_hidden_layers - seq.layer_index)` -- a
//      BUDGET-LIMITED PARTIAL SLICE, never the full layer count in general -- and the counter
//      advances by exactly ONE there, once per call, regardless of how many layers that call's
//      own partial slice actually dispatched: the single-token site counts CALLS, the chunk site
//      counts LAYERS-DISPATCHED, and `admit_count * num_hidden_layers` is a property of the
//      chunk path only.
// Both are declared here, at namespace scope (not inside test_main.cpp's translation unit, and
// not inside an anonymous namespace -- see tests/t2112-gpu-1p0-red-suite/fixture_common.h's own
// header note on why a bench-bridge extern must bind to the real global symbol) so a definition
// landing in src/gpu/superslm_gpu.cpp under the same guard resolves this suite's own reference.
#ifndef SUPERSLM_TESTS_SUPPORT_GPU_CHUNK_DISPATCH_INSTRUMENT_H
#define SUPERSLM_TESTS_SUPPORT_GPU_CHUNK_DISPATCH_INSTRUMENT_H

#include <atomic>
#include <cstdint>

namespace superslm_test {

// One increment per command list opened and submitted -- the single-token path
// (`RunLayerLoopGpuSubmit`, once per call) AND the chunk path
// (`SubmitOneSubChunkToFullDepthForG5Bridge`, once per (sub-)chunk call) both increment this,
// which is the observable that distinguishes "N tokens batched into few command lists" from
// "N tokens, N submissions" a black-box content comparison cannot see. Declared `extern`, defined
// only once the builder wires Rung 2 (see header comment above) -- referencing it before then is
// a genuine unresolved external, not a header-only no-op.
extern std::atomic<int64_t> g_gpu_chunk_submit_count_probe;

// Advances by `num_hidden_layers` per per-admitted-token invocation of
// `RecordOneTokenFullDepthDispatchBody` INSIDE THE CHUNK PATH's own open-list loop
// (`SubmitOneSubChunkToFullDepthForG5Bridge`, which always requests full depth per token) --
// the "actual dispatch count issued" instrument design Sec9's Guard-vitality row names for all
// three admission clamps (DFA, position-cap, embed_admit_count): a cell drives a chunk through
// the batched primitive, reads this counter's delta, and asserts it equals exactly
// `admit_count * num_hidden_layers` (mutation-provable: a clamp removed or off-by-one'd changes
// how many tokens are admitted, which changes this delta, which the cell's own assertion then
// catches). The single-token path (`RunLayerLoopGpuSubmit`) invokes the SAME body but with a
// budget-limited partial layer slice, not full depth, and advances this counter by exactly one
// per call regardless of that slice's own size -- the `admit_count * num_hidden_layers` contract
// above holds on the chunk path only; see the header comment above for both sites' own semantics.
extern std::atomic<int64_t> g_gpu_chunk_dispatch_count_probe;

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SUPPORT_GPU_CHUNK_DISPATCH_INSTRUMENT_H
