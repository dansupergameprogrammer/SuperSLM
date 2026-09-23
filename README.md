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

Current release: **1.7.1**. The engine library writes nothing to stdout: the `# adapter:
<name>` line `SSLM_GPU_ADAPTER_INDEX` prints now goes to stderr, like every other harness
diagnostic, so a host that pipes model output through stdout is no longer corrupted by it. No
token, ABI surface, or measured figure changes from 1.7.0. See the
[1.7.1 release note](docs/releases/1.7.1.md).

1.7.0: the token finish can take its logits off the calling thread, through a caller-supplied host
parallel-for hook on both backends or an opt-in device-resident head on the GPU, with tokens
identical to 1.6.0. On an RTX 2080 SUPER the device head took Qwen2.5-0.5B from 42.67 to
73.21 tok/s and Qwen2.5-1.5B from 27.85 to 55.49 tok/s, for 137,433,088 B and 234,627,072 B of
VRAM per mapped model. A host can also choose the directory GPU shaders load from, and a GPU
context stays usable after an out-of-memory failure during an upload. With no hook and no device
head, tokens are identical to 1.6.0 and the path is measurably slower in some cells (−1.98% tok/s
on the CPU at 0.5B). `GpuContextConfig` grew, so code compiled against an earlier `gpu_1p0.h` must
be recompiled. See the [1.7.0 release note](docs/releases/1.7.0.md) for every figure with its
hardware and settings, and [CHANGELOG.md](CHANGELOG.md).

## Capabilities

### Cross-platform determinism

The same model, prompt, and decoding configuration produce identical output
tokens on every **certified** platform — see [Certified platforms](#certified-platforms)
below for exactly which ones and what "certified" means there. Determinism is
checked at the level of individual forward-pass outputs, not just final
tokens: on a certified GPU, every intermediate layer's key/value state and
every decoded token match a CPU reference run bit-for-bit.

### Opt-in damped greedy decoding

Damped greedy is a deterministic anti-repetition decoder for calls where plain
greedy falls into a loop. It scores the model's top candidates against a small
anti-language-model built from the generated history. It is opt-in at both
conversion and decode time; existing artifacts and callers continue to use
greedy unchanged.

The ruled operating point is `alpha=2`, anti-LM order `n=2`, and `top_k=6`.
Across 192 paired real generations on Qwen2.5 0.5B and 1.5B, the three observed
0.5B greedy loop locks fell to zero and repeated-trigram rates dropped in all
four model/length cells. Many outputs read more coherently, but that is an
observation over this corpus rather than a general quality guarantee. The
decoder can penalize legitimate repetition too: one list-continuation case
changed list formatting. Prefer greedy when preserving the model's learned
format/style continuity matters, or use a compiled schema when validity is the
actual requirement. The full population, side-by-side text, and timing scope are in
[the Phase E confirmation](docs/calibration/t2199-phase-e-confirmation.md).

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

1.1 makes prompt prefill — processing the tokens you send in before
decoding begins — substantially faster on both paths, without changing a
single output bit.

On the CPU, the bottleneck was the kernel itself: the integer matmul is
now runtime-dispatched across SSE2, AVX2, and AVX-512 tiers, and batched
prefill measures **1.68x-1.72x faster** than 1.0's SSE2-only kernel on a
real 1.5B model. Every tier produces bit-identical output to the scalar
reference; the wider vectors change speed, not results.

On the GPU, the bottleneck was per-token submission: the 1.0 path paid a
host round-trip fence wait per token per layer batch. Prefill now submits
a whole chunk of tokens per command list, measuring **6.91x-7.19x faster**
on the certified NVIDIA GPU and **13.8x faster** on the certified AMD GPU,
where the round-trips cost more. Output is bit-identical to the per-token
path at every chunk size and every chunk-boundary split, on both certified
GPUs.

Both measurements, their exact hardware and span-length cells, and what
remains open are in
[docs/platform-support.md](docs/platform-support.md) and
[the roadmap](#roadmap-beyond-14).

### Schema-constrained generation

Schema-constrained generation means the model cannot emit a token that
breaks your schema — so the output parses whenever the decode completes
within its token budget. A compiled schema is a table of valid-token masks,
one per parser state; the engine indexes into that table by the sequence's
own parse state before every decode step and masks out any token that would
break the schema, so an invalid token is never a candidate in the first
place — not a post-hoc filter on the model's raw output. Every token the
engine does emit is legal at the moment it is emitted; format validity
holds for whatever was emitted, not only for a decode that ran to
completion.

An unbounded field (the free-text `"type": "string"` leaf, for example) has
no length bound the engine enforces, so its own close is up to the model —
if a caller's decode budget is reached first, the returned sequence is
legal so far but not closed, and will not parse as JSON on its own. A
caller detects this without guessing from the token count alone:
`sslm_stats`'s `schema_accepting` field is 1 iff a schema is bound and the
sequence's current parse state is one where stopping is valid, and 0
otherwise. Because 0 also means no schema is bound, callers that need to
distinguish those cases use `sslm_seq_schema_bound`. A schema-bound decode
that stops (budget reached, or any other reason) while
`schema_accepting == 0` returned an incomplete value; a caller that needs a
guaranteed-parseable result checks both facts before treating the output as
final. GPU callers use `SslmGpuSeqSchemaBoundForG5Bridge` and
`SslmGpuSeqSchemaAcceptingForG5Bridge` after draining and finishing the step.

Schema bind, rebind, and unbind are valid only on a newly created or reset
sequence. A generation call that passes argument validation makes
the sequence ineligible until reset, including a valid no-op call or CPU
empty-prefix adoption. A call rejected for invalid arguments leaves the sequence
untouched and still bindable; a GPU Submitted/busy refusal likewise preserves
eligibility. A restored CPU or GPU sequence must be reset before binding.

The determinism guarantee above carries through even when the engine skips
ahead through the parts of the output your schema already dictates (for
example, a fixed key name or punctuation your schema forces regardless of
what the model would otherwise produce) without running a real decode step
for those tokens — a schema-constrained decode is exactly as reproducible,
on the same certified platform, as an unconstrained one. Format — grammar
validity for every token emitted, and parse validity for a decode that
completes within its budget — is what this guarantees by construction;
cross-field semantic validity (e.g. "this value must be less than that
one") is reported, not enforced, since that is a property of your schema's
meaning rather than its shape.

Schema-constrained decoding is proven bit-identical between the CPU and GPU
paths on both certified GPUs — NVIDIA and AMD (see
[Certified platforms](#certified-platforms)). The mechanism, the artifact
format it relies on, and the full CPU consumer ABI it ships through are
documented in [docs/api.md](docs/api.md) and
[docs/sslm_format.md](docs/sslm_format.md).

## Status

Every capability above is built and measured on the platforms named.
Determinism, sliceable inference, and schema-constrained generation are
all measured on both certified GPUs — the schema-constrained-decoding GPU
parity check, for example, passed bit-identical on the certified NVIDIA
GPU, and passed bit-identical on the certified AMD GPU as well (measured
2026-08-17 on the Radeon RX 7900 XTX; see
[Certified platforms](#certified-platforms)). Two capabilities are
narrower: adapter switching is measured on the certified NVIDIA GPU only
(see [Runtime-switchable LoRA adapters](#runtime-switchable-lora-adapters)
above), and CPU-side prefill batching is proven on every certified
platform on the CPU path, with the GPU path — new in 1.1 — measured and
certified on both certified GPUs, NVIDIA and AMD.
[docs/platform-support.md](docs/platform-support.md) carries every
measured number with its device and cell.

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
made beyond what that gives. What that matrix has actually executed, and as
of when, is stated in exactly one place —
[docs/platform-support.md § CI execution status](docs/platform-support.md#ci-execution-status)
— rather than restated here. A consumer can reproduce the same verification
locally regardless of hosted-run status: the [Building](#building) section's
`cmake`+`ctest` matrix on any platform, and `build.bat` on Windows. See
[docs/platform-support.md](docs/platform-support.md) for the full table,
every measured number, and where each one was measured.

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
and the test suite. To build only the CPU product on Windows, with no DXC or
D3D12 test dependency, use `cmake -B build -DBUILD_TESTING=OFF` followed by
the ordinary default `cmake --build build --config Release`; no test target
is created. On Linux and macOS this is standard-library only, no
third-party runtime dependency, and works from CMake 3.16. On Windows, the
enabled test suite's own GPU-serial-port section calls the D3D12 GPU path
directly, so `superslm_tests` compiles and links `src/gpu/superslm_gpu.cpp`
and its compiled shaders when `BUILD_TESTING=ON` — this needs CMake 3.20+
and the DirectX Shader Compiler
(`dxc.exe`, from the Windows SDK), matching `build.bat`'s own long-standing
requirement below; a Windows configure without `dxc.exe` fails at
`cmake -B build` with a one-line diagnostic naming what is missing, rather
than at link time. The public, installable GPU library is a separate,
opt-in CMake target on top of that (Windows/MSVC only):

```
cmake -B build -DSUPERSLM_BUILD_GPU=ON && cmake --build build --target superslm_gpu
```

This is for a consumer who wants the GPU acceleration library itself
installed and exported via `find_package(superslm)`; it enables the same
shader compilation used by the Windows test build and installs the compiled
shaders with the package. By default the GPU library loads them from a
`shaders` directory beside the host executable, so a consuming CMake target
deploys them there:

```cmake
find_package(superslm REQUIRED)
target_link_libraries(my_app PRIVATE superslm::superslm_gpu)
superslm_deploy_gpu_shaders(my_app)
```

A host that does not own its executable's directory (an editor plugin, for
example) can skip the copy and point `GpuContextConfig::shader_dir` at the
installed `superslm_GPU_SHADER_DIR` instead. The shader directory is
process-wide; [docs/api.md](docs/api.md) describes the rule.

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

Two public APIs are documented in [docs/api.md](docs/api.md), both shipped:
the C++-linkage GPU handle API (`SslmGpu*`) and the CPU-side C ABI (`sslm_*`)
for embedding SuperSLM directly in another process without the GPU handle types.

## License

SuperSLM is licensed under Apache License 2.0. The permissive license and
express patent grant are deliberate: they make adoption safe for consumers,
and closed forks remain permitted.

## Roadmap beyond 1.5

Named follow-on work after 1.5:

- **True shared-prefix KV memory.** 1.0 ships a straightforward per-sequence
  KV layout; a block-table indirection layer is the next step, giving a
  cohort of sequences that share a prompt prefix N-fold memory savings
  instead of each holding its own copy.
- **Async-wrapper overhead reduction.** The API's convenience wrapper
  around the dispatch path measurably costs throughput versus the raw
  dispatch path — see the Direct dispatch vs. Async wrapper API rows in
  [docs/platform-support.md](docs/platform-support.md#certified-gpu-measurements);
  closing that gap is a named follow-on campaign.
- **Launch-floor reduction.** The fixed per-call GPU launch overhead has an
  identified, not-yet-built optimization path.
- **AVX-512 on full-width silicon.** The tier is proven and measured on
  Zen 4 (about 1.18x over AVX2, bounded by that microarchitecture's
  double-pumped 512-bit execution); measuring on full-width datapaths
  (Zen 5, server Intel) is open — see
  [Known gaps](docs/platform-support.md#known-gaps).
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
