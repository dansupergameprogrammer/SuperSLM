# SuperSLM

SuperSLM is a deterministic small-language-model inference runtime designed
specifically for real-time games. It loads a converted, quantized `.sslm`
artifact and runs it with integer-only arithmetic, no floating point on the
inference path: the same model, prompt, and decoding configuration produce
identical tokens every time, on every certified platform.

Being deterministic is what makes SuperSLM usable inside a frame budget. The
engine can slice a decode step across a caller-chosen number of transformer
layers per call — spreading one token's worth of work across several frames
instead of spending it in one frame-time spike — and the exact same layer
slicing produces the exact same output tokens as running the whole step at
once. A game can therefore throttle inference to fit whatever GPU headroom a
frame has left without changing what the model says.

**Status: pre-1.0.** Every capability below — including schema-constrained
generation — is built and measured on the platforms named, on both certified
GPUs: the schema-constrained-decoding GPU parity check passed bit-identical
on the certified NVIDIA GPU, and passed bit-identical on the certified AMD
GPU as well (measured 2026-08-17 on the Radeon RX 7900 XTX; see
[Certified platforms](#certified-platforms)).

## Capabilities

### Cross-platform determinism

The same model, prompt, and decoding configuration produce identical output
tokens on every **certified** platform — see [Certified platforms](#certified-platforms)
below for exactly which ones and what "certified" means there. Determinism is
checked at the level of individual forward-pass outputs, not just final
tokens: on a certified GPU, every intermediate layer's key/value state and
every decoded token match a CPU reference run bit-for-bit.

### Runtime-switchable LoRA adapters

A LoRA specialization can be attached to and detached from a running model
without unloading or duplicating the base weights. Measured on an NVIDIA RTX
2080 SUPER (Windows x64, a real 1.5B-parameter base model and a real LoRA
adapter): switching adapters takes 0.128 s, against 7.52 s to fully reload
the model with a different adapter merged in — about 58x faster. The base
model's resident weights are untouched by the switch, and decode output
remains bit-identical to the pre-switch determinism guarantee.

### Sliceable inference

A caller sets a per-frame GPU budget, measured in transformer layers per
decode call, and the engine spreads one token's worth of work across
however many calls that budget requires — designed to keep inference off
the frame-time spike path. This is proven **slice-invariant**: decoding
the same prompt at any layer-budget granularity, down to one layer per
call, produces the identical output tokens as decoding it in one call, on
every certified platform. An in-game frame-time measurement (part of the
SuperSLM-Unreal integration, which follows this engine's own 1.0 release)
will turn "designed to avoid spikes" into a measured frame-time result;
today the guarantee is the mechanism the design relies on, proven at the
engine level.

Prompt prefill (processing the tokens you send in before decoding begins)
batches an entire prefill chunk through the engine's matrix kernels in one
pass on the CPU path, proven bit-identical to processing the same tokens one
at a time at every chunk size and every chunk-boundary split. On the
reference hardware this workload turned out to be bound by the integer
kernel's own compute throughput rather than by memory bandwidth, so the
batching gives a modest, real gain rather than a large one — see
[the roadmap](#roadmap-beyond-10) for the wider-vector kernel that targets
the actual bottleneck.

### Schema-constrained generation

Schema-constrained generation means your output format is always correct —
the model cannot emit a token that breaks your schema, so the output always
parses. A compiled schema is a table of valid-token masks, one per parser
state; the engine indexes into that table by the sequence's own parse state
before every decode step and masks out any token that would break the
schema, so an invalid token is never a candidate in the first place — not a
post-hoc filter on the model's raw output.

The determinism guarantee above carries through even when the engine skips
ahead through the parts of the output your schema already dictates (for
example, a fixed key name or punctuation your schema forces regardless of
what the model would otherwise produce) without running a real decode step
for those tokens — a schema-constrained decode is exactly as reproducible,
on the same certified platform, as an unconstrained one. Format — grammar
and parse validity — is what this guarantees by construction; cross-field
semantic validity (e.g. "this value must be less than that one") is
reported, not enforced, since that is a property of your schema's meaning
rather than its shape.

Schema-constrained decoding is proven bit-identical between the CPU and GPU
paths on both certified GPUs — NVIDIA and AMD (see
[Certified platforms](#certified-platforms)). The mechanism, the artifact
format it relies on, and the full CPU consumer ABI it ships through are
documented in [docs/api.md](docs/api.md) and
[docs/sslm_format.md](docs/sslm_format.md).

## Certified platforms

"Certified" means: built, run, and measured on that exact hardware, with
every determinism check passing bit-for-bit against the CPU reference on
that same run.

| Platform | CPU inference | GPU inference |
|---|---|---|
| Windows x64 | Certified | Certified — NVIDIA Turing (measured on RTX 2080 SUPER) and AMD RDNA3 (measured on Radeon RX 7900 XTX) |
| Linux x64 | CI extent (see note below) — GCC and Clang | Not built (the GPU backend is D3D12, Windows-only) |
| macOS (Apple Silicon) | CI extent (see note below) | Not built |

"CI extent" means the platform is exercised by this project's own continuous
integration matrix, and no further hardware-specific measurement has been
made beyond what that gives. The workflows exist and define the full
matrix ([.github/workflows/tests.yml](.github/workflows/tests.yml):
windows-x64, linux-x64, linux-x64-asan, macos-arm64) — **hosted runs on
GitHub Actions are currently capped by the account's spending limit** and
have not executed since 2026-07-23; today, the verification path a
consumer can actually run is the local one: the [Building](#building)
section's `cmake`+`ctest` matrix on any platform, and `build.bat` on
Windows. See [docs/platform-support.md](docs/platform-support.md) for the
full table, every measured number, and where each one was measured.

GPU determinism is scoped to the certified adapters above: an AMD
integrated GPU on the same RDNA3 test machine passes the direct-dispatch
determinism check but diverges on the asynchronous decode path under
investigation, and is explicitly not a certified target today.

## What ships in an artifact

A `.sslm` file is a versioned, integrity-checked container: a header, a
section table, and aligned sections (weights, scales, RoPE tables, biases,
tokenizer, schema masks, golden reference hashes). The offline converter
(build-time Python) reads a Hugging Face checkpoint and emits the artifact;
the runtime maps and validates it. The format is specified in
[`docs/sslm_format.md`](docs/sslm_format.md).

The loader is a **trust boundary**: it treats the whole file as hostile
input and validates every field against declared bounds before reading a
section byte. Any deviation — bad magic, unsupported version, an
out-of-bounds or overlapping section, a hash mismatch — is a rejection
with a versioned diagnostic, never a silent partial load.

## Quickstart

[docs/quickstart.md](docs/quickstart.md) walks through converting your own
Hugging Face checkpoint into a `.sslm` model and decoding from it, including
a worked LoRA adapter conversion.

## Building

```
cmake -B build && cmake --build build && ctest --test-dir build
```

This builds the CPU-only product library (`superslm`), the public headers,
and the test suite. On Linux and macOS this is standard-library only, no
third-party runtime dependency, and works from CMake 3.16. On Windows, the
test suite's own T-2019 GPU-serial-port section calls the D3D12 GPU path
directly and unconditionally, so `superslm_tests` also compiles and links
`src/gpu/superslm_gpu.cpp` and its compiled shaders on every Windows
configure — this needs CMake 3.20+ and the DirectX Shader Compiler
(`dxc.exe`, from the Windows SDK), matching `build.bat`'s own long-standing
requirement below; a Windows configure without `dxc.exe` fails at
`cmake -B build` with a one-line diagnostic naming what is missing, rather
than at link time. The public, installable GPU library is a separate,
opt-in CMake target on top of that (Windows/MSVC only):

```
cmake -B build -DSUPERSLM_BUILD_GPU=ON && cmake --build build --target superslm_gpu
```

This is for a consumer who wants the GPU acceleration library itself
installed and exported via `find_package(superslm)`; it reuses the same
shader compilation the default Windows configure already performs above,
and needs nothing further.

On Windows with Visual Studio installed, `build.bat` is a one-shot MSVC
build that compiles the full local development suite, GPU included (it
requires the Windows SDK for `dxc.exe`, same as above). The test suite is
standard-library only and prints a `checks / failures` summary; a nonzero
exit means a failure. `build.bat` also runs a small number of optional
local checks (an ABI verb-count re-derivation, a couple of Python-based
CI-source checks) when `bash`/`python` happen to be on `PATH`; each is a
loud, named, non-fatal skip when its tool is absent — none is required to
build or to pass the suite.

## Layout

| Path                     | What                                                    |
|--------------------------|-----------------------------------------------------------|
| `include/superslm/`      | public headers — the embeddable API                     |
| `src/`                   | implementation                                          |
| `tests/`                 | the test suite (one harness, no third-party framework)  |
| `tools/`                 | build-time tooling — converters, calibration, verification |
| `docs/`                  | format, API, and platform-support documentation         |

## API surfaces

Two public C APIs are documented in [docs/api.md](docs/api.md), both
shipped: the GPU handle-based API (`SslmGpu*`) and the CPU-side, from-scratch
consumer ABI (`sslm_*`) for embedding SuperSLM directly in another process
without the GPU handle types.

## License

SuperSLM is licensed under Apache License 2.0. The permissive license and
express patent grant are deliberate: they make adoption safe for consumers,
and closed forks remain permitted.

## Roadmap beyond 1.0

Work already committed for the release after 1.0, alongside its own full
design and review loop when scheduled:

- **True shared-prefix KV memory.** 1.0 ships a straightforward per-sequence
  KV layout; a block-table indirection layer is the next step, giving a
  cohort of sequences that share a prompt prefix N-fold memory savings
  instead of each holding its own copy.
- **Async-wrapper overhead reduction.** The 1.0 API's convenience wrapper
  around the dispatch path measurably costs throughput versus the raw
  dispatch path — see the Direct dispatch vs. Async wrapper API rows in
  [docs/platform-support.md](docs/platform-support.md#certified-gpu-measurements);
  closing that gap is a named follow-on campaign.
- **Launch-floor reduction.** The fixed per-call GPU launch overhead has an
  identified, not-yet-built optimization path.
- **A wider-vector CPU prefill kernel.** Prefill throughput on the CPU path
  is bound by the scalar/SSE2 integer kernel's own instruction throughput,
  not by memory bandwidth — a wider-vector kernel (AVX2/AVX-512) is the
  identified lever for a further speedup, not yet built.
- **Flat-batch dispatch fusion and the remaining single-group tail sites.**
  Two named, scoped opportunities to close the gap between batched and
  single-sequence throughput further.
- **The integrated-GPU divergence.** Root-causing why the async decode path
  diverges on an AMD integrated GPU where the direct-dispatch path does not
  (see [Certified platforms](#certified-platforms)), toward certifying
  integrated GPUs as a target.
- **A second GPU generation.** Extending cross-vendor certification to a
  newer NVIDIA generation (Blackwell) beyond the currently-certified Turing
  and RDNA3.
- **macOS beyond CI extent.** Measuring on real Apple Silicon hardware,
  beyond what GitHub's own macOS runners can exercise today.
