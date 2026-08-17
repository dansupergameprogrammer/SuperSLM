# Changelog

All notable changes to SuperSLM (Layer 1) are recorded here.

## [1.0.0-unreleased]

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

### In development

- **Schema-constrained generation** — grammar-guaranteed structured
  output, integrated into the same determinism boundary as the rest of
  decoding. This is the one capability remaining before the 1.0 tag; see
  the README for its shipped form once it lands.

### Known gaps, tracked

- The CMake `superslm_tests` target does not yet link the GPU library;
  GPU-path testing runs through a separate, locally-run suite rather than
  through `ctest`.
- Linux and macOS are exercised at continuous-integration extent only —
  CPU inference, no GPU backend (GPU is D3D12, Windows-only), and no
  dedicated throughput measurement published for either.
- An AMD integrated GPU on RDNA3 diverges from the certified determinism
  guarantee on the asynchronous decode path; it is explicitly not a
  certified target while that is under investigation. See
  [docs/platform-support.md](docs/platform-support.md).
