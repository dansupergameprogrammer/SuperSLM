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
// THE COUNTERS ARE NOT YET WIRED INTO PRODUCTION. Rung 2 (design Sec8) is expected to:
//   1. Increment `g_gpu_chunk_submit_count_probe` exactly once per
//      `SubmitChunkToFullDepthForG5Bridge` call (i.e. once per command list opened and
//      submitted for a chunk/sub-chunk -- design Sec5 steps 1/3/4), from inside that function's
//      own body, guarded by `SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT`.
//   2. Advance `g_gpu_chunk_dispatch_count_probe` by `num_hidden_layers` per invocation of the
//      Rung-2-extracted per-token, per-layer dispatch body (design Sec5's own D-SLM3595 ruling --
//      the function `RunLayerLoopGpuSubmit` is reduced to open/call/close around, and one
//      invocation of that body issues one token's own FULL-DEPTH dispatch chain, all layers in a
//      single call), same guard -- matching this header's own documented "delta ==
//      admit_count * num_hidden_layers" contract below, not a plain per-invocation `++`.
// Both are declared here, at namespace scope (not inside test_main.cpp's translation unit, and
// not inside an anonymous namespace -- see tests/t2112-gpu-1p0-red-suite/fixture_common.h's own
// header note on why a bench-bridge extern must bind to the real global symbol) so a definition
// landing in src/gpu/superslm_gpu.cpp under the same guard resolves this suite's own reference.
#ifndef SUPERSLM_TESTS_SUPPORT_GPU_CHUNK_DISPATCH_INSTRUMENT_H
#define SUPERSLM_TESTS_SUPPORT_GPU_CHUNK_DISPATCH_INSTRUMENT_H

#include <atomic>
#include <cstdint>

namespace superslm_test {

// One increment per chunk (or TDR-safe sub-chunk) submission -- the observable that
// distinguishes "N tokens batched into few command lists" from "N tokens, N submissions" a
// black-box content comparison cannot see. Declared `extern`, defined only once the builder
// wires Rung 2 (see header comment above) -- referencing it before then is a genuine unresolved
// external, not a header-only no-op.
extern std::atomic<int64_t> g_gpu_chunk_submit_count_probe;

// Advances by `num_hidden_layers` per per-token dispatch body invocation inside an open chunk
// list (one such invocation issues that token's own full-depth dispatch chain, all layers in one
// call) -- the "actual dispatch count issued" instrument design Sec9's Guard-vitality row names for all three
// admission clamps (DFA, position-cap, embed_admit_count): a cell drives a chunk through the
// batched primitive, reads this counter's delta, and asserts it equals exactly
// `admit_count * num_hidden_layers` (mutation-provable: a clamp removed or off-by-one'd changes
// how many tokens are admitted, which changes this delta, which the cell's own assertion then
// catches).
extern std::atomic<int64_t> g_gpu_chunk_dispatch_count_probe;

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SUPPORT_GPU_CHUNK_DISPATCH_INSTRUMENT_H
