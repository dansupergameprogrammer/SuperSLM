# Changelog

All notable changes to SuperSLM (Layer 1) are recorded here.

## [Unreleased]

### Damped greedy decoding

- Added deterministic damped-greedy decoding as an explicit opt-in on the CPU
  generation path. Greedy remains the default and the legacy
  `sslm_decode_step` ABI remains greedy-only.
- The ruled defaults are `alpha=2` (`alpha_q15=65536`), anti-LM order `n=2`,
  and `top_k=6` (clamped only for a model whose vocabulary is smaller).
  `sslm_decode_params_init` fills those values and derives the fixed-point scale
  constants from the mapped model artifact; callers do not need to reproduce
  converter arithmetic.
- `convert_model.py --enable-damped-greedy` emits the required DGC1 constants
  section and feature bit. The default conversion path remains unflagged and
  compatible with pre-1.2 runtimes.
- End-to-end confirmation covered 192 paired generations across Qwen2.5 0.5B
  and 1.5B, 100- and 300-token ceilings. The three observed 0.5B greedy loop
  locks fell to zero; damped greedy substantially reduced repeated trigrams in
  all four cells. It is a quality tradeoff, not a dominance claim: legitimate
  repeated structure can also be penalized, and the `list_primed_00` case
  visibly changed list formatting. See
  [the confirmation packet](docs/calibration/t2199-phase-e-confirmation.md).

### Public surface and verification

- `sslm_decode_step_v2` selects greedy or damped greedy through the extended
  parameter struct. Damped state participates in reset, save/restore, prefix
  adoption, schema masking, adapter attachment, digesting, and concurrent
  teardown contracts.
- The production converter, CLI, C ABI initializer, independent greedy oracle,
  Phase D suite, and the previously link-only T-2138 ABI suite are now wired
  into release verification.

### Fixed

- **`sslm_convert_adapter`'s B3 per-pair review diagnostic no longer
  over-flags `composed_mean`/`effect_mean`.** These two margins graded a
  VALIDATION-partition `upper_ci` (already `mean + 1.645*se`) against a
  threshold instead of the partition's own `mean`, adding a spurious extra
  `1.645` standard errors on top of the already-conservative threshold to
  every mean-conjunct margin — the two tail conjuncts already used the raw
  point estimate correctly. A pair's `composed_mean`/`effect_mean` review
  flag now reflects the same statistic the tail conjuncts always used.
- **`sslm_convert_adapter`'s pooled B3 accept/reject gate is retired.** It
  never discriminated a healthy converted adapter from a corrupted one on
  its own merits — its accept boundary was one frozen reference adapter's
  own idiosyncratic scale, and no in-band corruption ever elevated the
  statistic once that scale was accounted for. Converting an adapter can no
  longer be refused on B3 pooled quality grounds; only a domain trip (an
  unrepresentable ratio) still refuses to write an artifact. Two things
  ship in its place: the per-pair review diagnostics above are now this
  tool's primary B3 signal, and a new wide-tolerance magnitude sanity check
  compares a candidate's pooled composed LoRA delta norm against an
  optional reference (`--reference-delta-norm`) — a candidate far outside
  tolerance prints a named WARNING for review, never a rejection.
- **The per-pair review diagnostics' own reported margins shifted once, at
  the retirement above, and are stable after.** The diagnostics reused a
  random-number stream that two now-deleted pooled-statistic calls used to
  draw from first; deleting those calls moved every pair's own bootstrap
  draws to a different point in the same stream, with no change to the
  diagnostic's own arithmetic. The stream is now seeded independently for
  this loop alone, so an unrelated future change elsewhere in the pooled
  report cannot shift these numbers again.
- **`build_runtime_additive_sections`'s `checkpoint_path` resume path no
  longer crashes, and no longer silently reports a zero magnitude for a
  resumed pair.** The magnitude sanity check above added a required
  per-pair field that a checkpoint file written before this fix lacks;
  resuming from such a file now recomputes the field from the pair's own
  current adapter weights instead of crashing (`.tolist()` on a plain
  `float`) or silently defaulting to zero. `checkpoint_path` is a Python
  keyword argument, not a CLI flag, so this affects only direct callers of
  `build_runtime_additive_sections`, not `sslm_convert_adapter.py`'s CLI.

## [1.1.0] - 2026-08-19

A performance release: both halves of 1.0's own "compute-bound, not
memory-bound" finding get a real lever, one on the CPU path and one on the
GPU path. No public API signatures changed, but one public entry-point
failure behavior did — see Fixed below — and existing consumers should read
that section before upgrading. See [README.md](README.md) for the full
capability descriptions and [docs/platform-support.md](docs/platform-support.md)
for every measured number and where it was measured.

### CPU: a wider-vector prefill kernel

- The scalar-and-SSE2-only integer matmul kernel 1.0 shipped is now a
  runtime-dispatched, three-tier kernel: SSE2 (the unconditional
  architectural floor), AVX2, and AVX-512, selected once per process by a
  CPUID+XGETBV probe, with SSE2 as the automatic fallback on hardware
  lacking the wider tiers. All three tiers are proven bit-identical to the
  scalar reference on real hardware — SSE2 and AVX2 on this project's
  reference machines, AVX-512 with the full forced suite on AVX-512
  silicon, its cross-tier digest matching every other tier exactly — the
  same determinism guarantee 1.0 established, carried across every new
  dispatch path.
- Measured on a real 1.5B-parameter model artifact, batched prefill,
  SSE2 to AVX2: **about 1.68x-1.72x faster**, two independent runs. This is
  the CPU-side answer to the lever 1.0's own changelog named but did not
  yet build.

### GPU: batched prompt prefill

- Prompt prefill on the GPU path now runs a whole prefill span through the
  device in one submission rather than one token's worth of dispatches per
  round trip, internally splitting only when needed for driver stability
  (see [docs/platform-support.md](docs/platform-support.md) for how that
  internal bound was found). Proven bit-identical to the pre-1.1,
  one-round-trip-per-token path at every span size and every internal
  split boundary tested.
- Measured on both certified GPUs, forced prefill spans against a real
  1.5B-parameter model: **about 6.91x-7.19x faster** than the pre-1.1
  per-token path on the NVIDIA RTX 2080 SUPER, and **about 13.8x faster**
  on the AMD RX 7900 XTX, where the per-call round-trips cost more.
  Certified bit-identical on both, including spans at and above the
  internal sub-chunk split bound. The two public GPU prefill entry points
  are documented as bulk-throughput calls as of this release: their
  per-call dispatch-budget parameter is still validated but no longer
  slices submission per token on the prefill path — see
  [docs/api.md](docs/api.md). Per-frame budget slicing for interactive
  decode is unchanged.

### Fixed

- **GPU decode/prefill entry points no longer terminate the process on a
  device or allocation fault; they return `SSLM_DEVICE_LOST`.** Before this
  release, a `Close()`/`Signal()` failure or an allocation failure while
  submitting or finishing a GPU layer-loop chunk raised a raw C++ exception
  that crossed the documented `SslmGpuStatus` C ABI boundary uncaught,
  terminating the calling process. `sslm_decode_step_gpu` /
  `SslmGpuSeqDecodeStepForG5Bridge`, `SslmGpuSeqPrefillPromptForG5Bridge`,
  and `SslmGpuSeqPrefillSchemaContentForG5Bridge` now catch the fault at
  its own source and return `SSLM_DEVICE_LOST`; the context stays usable
  after a caught fault (a second call on the same context is proven bit-
  identical to a never-faulted reference on both certified GPUs, NVIDIA
  and AMD) except when the device is
  genuinely, confirmably removed, which remains terminal for that context.
  `SSLM_DEVICE_LOST` carries two dispositions at these entry points — see
  [include/superslm/gpu_1p0.h](include/superslm/gpu_1p0.h) for which is
  which and the documented recovery bounds. A consumer that previously
  relied on process termination as its own crash-recovery signal for this
  fault class should add an explicit `SSLM_DEVICE_LOST` check instead.
- **`DetectBestDotRowTier()` no longer reads CPUID leaf 7 without first
  checking leaf 0's own max supported basic leaf.** Leaf 7 is
  architecturally undefined below basic leaf 7; on an older or limited x64
  target this could false-positive an AVX2 or AVX-512 tier the hardware
  does not support, defeating the documented SSE2 architectural floor. The
  dispatch decision now gates every leaf-7-derived bit on
  `max_basic_leaf >= 7`.
- **`sslm_convert_adapter` no longer crashes converting a bf16-trained LoRA
  adapter.** The prior reader called a numpy cast with no bfloat16
  representation before ever widening the tensor, raising `TypeError: data
  type bfloat16 not understood` on every bf16-trained adapter — the
  prevailing PEFT/LoRA training default. Adapter tensors are now read
  through the same manual safetensors parser and exact bit-shift widening
  the base checkpoint converter already used, lossless for bf16.

### Continuous integration

- The CI matrix gained three independently-forced kernel-tier legs (SSE2,
  AVX2, AVX-512), each its own full test-suite run plus its own
  cross-toolchain digest check, alongside the existing dispatch-live
  default — a runner that lacks a wider tier's hardware skips that tier's
  leg loudly rather than silently passing a build it cannot actually
  exercise. The scalar tier's own cross-toolchain digest leg shipped in
  1.0 and is unchanged by this release; scalar has no forced full-suite
  leg — digest comparison only.

### Known gaps, tracked

- The AVX-512 tier is proven bit-identical AND measured on real AVX-512
  silicon (full forced suite, cross-tier digest match; about 1.18x over
  AVX2 on Zen 4's double-pumped units — see
  [docs/platform-support.md](docs/platform-support.md)). Its CI leg probes
  for hardware and reports SKIPPED honestly on runners without AVX-512.

## [1.0.0] - 2026-08-18

This is the first public release line. Rather than a chronology of internal
build steps, this entry summarizes what 1.0 delivers as a whole; see
[README.md](README.md) for the full capability descriptions and
[docs/platform-support.md](docs/platform-support.md) for every measured
number and where it was measured.

### Core runtime

- A deterministic, integer-only inference engine: no floating point on the
  inference path, so the same model, prompt, and decoding configuration
  produce identical output tokens on every certified platform.
- A versioned, integrity-checked `.sslm` artifact format
  ([docs/sslm_format.md](docs/sslm_format.md)) — a trust-boundary loader
  that treats every file as hostile input and rejects any deviation with a
  versioned diagnostic rather than a silent partial load.
- An integer-only, dependency-free tokenizer for the Qwen2.5 lineage,
  verified bit-for-bit against the upstream Hugging Face tokenizer across
  thousands of adversarial and multilingual lines.
- Bit-exact integer kernels for the full forward pass (rotary embeddings,
  a fixed-point SiLU activation, and quantized matrix multiplication),
  each proven identical between its scalar and SIMD implementations and
  across every measured compiler/toolchain axis.
- Chunk-batched prompt prefill: a whole prefill chunk runs through the
  matrix kernels in one call rather than one token at a time, proven
  bit-identical to the per-token path at every chunk size and every
  chunk-boundary split. Measured on a real artifact: a modest, real gain
  (about 1.1x), because this workload turned out to be bound by the integer
  kernel's own compute throughput rather than memory bandwidth — see
  [docs/platform-support.md](docs/platform-support.md) and the README's
  roadmap for the wider-vector kernel that targets the actual bottleneck.

### GPU acceleration

- A D3D12-backed GPU inference path (Windows only), certified — bit-
  identical to the CPU reference at every layer — on NVIDIA Turing and AMD
  RDNA3 hardware. See [docs/platform-support.md](docs/platform-support.md)
  for the certified devices and every measured throughput number.
- Sliceable inference: a caller-chosen per-decode-call layer budget lets a
  consumer spread one token's worth of GPU work across multiple calls, with
  bit-identical output at every granularity down to one layer per call —
  the mechanism a real-time consumer uses to keep inference off its
  frame-time spike path.
- Batched multi-sequence decoding, with per-sequence rejections that don't
  abort the rest of the batch.
- Schema-constrained generation: a compiled schema forbids the model from
  emitting a token that would break your output format, including on spans
  the schema forces deterministically without a real decode step
  ("jump-forward"), under the same determinism guarantee as unconstrained
  decoding. Shipped on both the CPU consumer ABI and the GPU handle API —
  see [docs/api.md](docs/api.md). The GPU path is proven bit-identical to
  the CPU reference on both certified GPUs, NVIDIA and AMD.

### Runtime-switchable LoRA adapters

- Attach or detach a LoRA specialization on an already-resident model
  without unloading or duplicating the base weights, and without breaking
  the determinism guarantee. Measured on certified hardware: about 58x
  faster than a full model reload with a different adapter baked in.

### Conversion pipeline

- An offline converter (build-time Python) that calibrates a raw Hugging
  Face checkpoint, converts its tokenizer and weights into `.sslm`
  artifacts, and independently re-verifies every artifact it writes
  through the same C++ loader a consumer uses — see
  [docs/quickstart.md](docs/quickstart.md) for the full walkthrough,
  including a worked LoRA adapter conversion.
- A clean-checkout release-verification procedure
  (`tools/verify_clean_checkout.ps1`) that proves the entire quickstart
  path — build, calibrate, convert, decode, convert an adapter — from a
  bare extraction of the repository with no development-tree dependency.

### Known gaps, tracked

- Linux and macOS are exercised at continuous-integration extent only —
  CPU inference, no GPU backend (GPU is D3D12, Windows-only), and no
  dedicated throughput measurement published for either.
- An AMD integrated GPU on RDNA3 diverges from the certified determinism
  guarantee on the asynchronous decode path; it is explicitly not a
  certified target while that is under investigation. See
  [docs/platform-support.md](docs/platform-support.md).
