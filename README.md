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

**Status: pre-1.0.** Every capability below through cross-platform determinism
and runtime adapter switching is built, measured, and certified on the
platforms named. Schema-constrained generation (below) is still in
development and is the one gate remaining before the 1.0 tag.

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

## In development: schema-constrained generation

Schema-constrained generation means your output format is always correct —
the model cannot emit a token that breaks your schema, so the output always
parses. The determinism guarantee above carries through even when the
engine skips ahead through parts of the output your schema already
dictates, so a schema-constrained decode is exactly as reproducible as an
unconstrained one. Format — grammar and parse validity — is what this
guarantees by construction; cross-field semantic validity (e.g. "this value
must be less than that one") is reported, not enforced, since that is a
property of your schema's meaning rather than its shape.

This is under active development and is the one capability standing between
the current state of the repository and the 1.0 tag.

## Certified platforms

"Certified" means: built, run, and measured on that exact hardware, with
every determinism check passing bit-for-bit against the CPU reference on
that same run.

| Platform | CPU inference | GPU inference |
|---|---|---|
| Windows x64 | Certified | Certified — NVIDIA Turing (measured on RTX 2080 SUPER) and AMD RDNA3 (measured on Radeon RX 7900 XTX) |
| Linux x64 | CI extent — built and the full test suite passes on every push, GCC and Clang | Not built (the GPU backend is D3D12, Windows-only) |
| macOS (Apple Silicon) | CI extent — built and the full test suite passes on every push | Not built |

"CI extent" means the platform is exercised by the project's own continuous
integration on every change, and no further hardware-specific measurement
has been made beyond what that gives. See
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
and the test suite — no third-party runtime dependency, standard library
only. GPU support is a separate, opt-in CMake target for Windows/MSVC:

```
cmake -B build -DSUPERSLM_BUILD_GPU=ON && cmake --build build --target superslm_gpu
```

This requires CMake 3.20+ (the CPU-only build above works from CMake 3.16)
and the DirectX Shader Compiler (`dxc.exe`, from the Windows SDK) to compile
the GPU kernels. The `superslm_tests` target does not yet link
`superslm_gpu` — GPU-path correctness is exercised by a separate, locally-run
test suite rather than through `ctest` today; that wiring is a known,
tracked gap.

On Windows with Visual Studio installed, `build.bat` is a one-shot MSVC
build that compiles the full local development suite, GPU included (it
requires the Windows SDK for `dxc.exe`, same as above). The test suite is
standard-library only and prints a `checks / failures` summary; a nonzero
exit means a failure.

## Layout

| Path                     | What                                                    |
|--------------------------|-----------------------------------------------------------|
| `include/superslm/`      | public headers — the embeddable API                     |
| `src/`                   | implementation                                          |
| `tests/`                 | the test suite (one harness, no third-party framework)  |
| `tools/`                 | build-time tooling — converters, calibration, verification |
| `docs/`                  | format, API, and platform-support documentation         |

## API surfaces

Two public C APIs are documented in [docs/api.md](docs/api.md): the GPU
handle-based API (`SslmGpu*`, shipped) and the CPU-side consumer API
(`sslm_*`, under active development).

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
  around the dispatch path costs roughly 5.3 ms/token versus the raw
  dispatch path (measured on an NVIDIA RTX 2080 SUPER); closing that gap is
  a named follow-on campaign.
- **Launch-floor reduction.** The fixed per-call GPU launch overhead
  (roughly 2.7 ms/token on the same hardware) has an identified, not-yet-
  built optimization path.
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
