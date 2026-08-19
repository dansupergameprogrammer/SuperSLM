# Changelog

All notable changes to SuperSLM (Layer 1) are recorded here.

## [1.1.0] - 2026-08-19

A performance release: both halves of 1.0's own "compute-bound, not
memory-bound" finding get a real lever, one on the CPU path and one on the
GPU path. No API surface changes; see
[README.md](README.md) for the full capability descriptions and
[docs/platform-support.md](docs/platform-support.md) for every measured
number and where it was measured.

### CPU: a wider-vector prefill kernel

- The scalar-and-SSE2-only integer matmul kernel 1.0 shipped is now a
  runtime-dispatched, three-tier kernel: SSE2 (the unconditional
  architectural floor), AVX2, and AVX-512, selected once per process by a
  CPUID+XGETBV probe, with SSE2 as the automatic fallback on hardware
  lacking the wider tiers. Every tier is proven bit-identical to the
  scalar reference — the same determinism guarantee 1.0 established, now
  carried across three additional dispatch paths rather than weakened by
  them.
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
- Measured on a certified NVIDIA GPU, forced prefill spans of 128 and 256
  tokens against a real 1.5B-parameter model: **about 6.91x-7.19x faster**
  than the pre-1.1 per-token path. The two public GPU prefill entry points
  are documented as bulk-throughput calls as of this release: their
  per-call dispatch-budget parameter is still validated but no longer
  slices submission per token on the prefill path — see
  [docs/api.md](docs/api.md). Per-frame budget slicing for interactive
  decode is unchanged.

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

- The AVX-512 kernel tier's CI leg is defined — probes for hardware support,
  compiles, and bit-identity-checks where a runner has it — but has not yet
  executed on any hosted runner (GitHub Actions hosted runs are capped since
  2026-07-23) or on any AVX-512-capable machine this project has access to,
  so it carries no bit-identity result and no dedicated real-hardware
  throughput measurement yet.
- GPU batched prefill's determinism proof and throughput measurement are
  on the certified NVIDIA GPU only in this release; certification on the
  certified AMD GPU is not yet complete for the batched path specifically
  (the pre-1.1 per-token GPU path stays certified on both, unaffected by
  this release).

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
