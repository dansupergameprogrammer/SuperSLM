# Platform support

"Certified" in this document means: built and run on that exact hardware,
with every determinism check (bit-identical output against the CPU
reference, at every layer-budget granularity the engine supports) passing
on that run. "CI extent" means the platform is built and tested on every
change by the project's continuous integration, and no measurement beyond
that has been made. Every number below states the device, the artifact,
and the surface it was measured through — a number without that context is
not included here.

## CPU inference

| Platform | Status | How it's exercised |
|---|---|---|
| Windows x64 | Certified | Local build + full test suite; continuous integration |
| Linux x64 | CI extent | GitHub Actions (`ubuntu-latest`), GCC and Clang toolchains — build + full test suite |
| macOS (Apple Silicon, arm64) | CI extent | GitHub Actions (`macos-latest`) — build + full test suite |

Cross-toolchain determinism at the primitive level (the SiLU
lookup-table and integer matmul kernels) is checked by comparing a digest
of each kernel's output across every measured toolchain axis: Linux/GCC,
Linux/Clang, Linux/Clang with SIMD forced off, Windows/MSVC,
Windows/Clang-cl, and macOS/Clang (arm64) — six axes, byte-identical
digests on all six as of this writing.

## GPU inference

The GPU backend is D3D12/HLSL and Windows-only; there is no GPU backend on
Linux or macOS today.

| GPU | Vendor / architecture | Status |
|---|---|---|
| NVIDIA RTX 2080 SUPER | NVIDIA, Turing | Certified |
| AMD Radeon RX 7900 XTX | AMD, RDNA3 | Certified |
| AMD Radeon Graphics (integrated) | AMD, same RDNA3-generation machine as the 7900 XTX | Not certified — see below |

### Certified GPU measurements

All figures below are measured against a real 1.5B-parameter model
artifact, decoding through the `SslmGpu*` API described in
[api.md](api.md). Throughput is tokens/second; higher is better.

| Metric | RTX 2080 SUPER (Turing) | Radeon RX 7900 XTX (RDNA3) |
|---|---|---|
| Direct dispatch | 62.79–63.11 | 51.70 |
| Async wrapper API | 52.81–55.34 | 52.75–60.41 |
| With a LoRA adapter attached | 35.55–35.70 | 32.18–32.44 |
| Batched, 4 concurrent sequences | not separately measured | 59.68–61.78 |
| Host CPU, same machine | 4.5–6.3 | 8.91 |

**Reproducing these numbers:** `build.bat` compiles `tools/t2100_gpu_throughput.cpp`
into `out\t2100_gpu_throughput.exe` on every Windows run of that
script, unconditionally — `SUPERSLM_BUILD_GPU` is the separate CMake-side option
and does not govern this tool (the harness needs a real `.sslm` model
artifact on disk, so it is built but not auto-run). Invoke it directly:
`out\t2100_gpu_throughput.exe <model.sslm> [steps] [token_id]` — it runs N
successive decode steps through the same `SslmGpu*` entry points a real
generation loop calls, one warmup step discarded, and reports the mean
tokens/second over the timed steps.

The 2080 SUPER's context-length curve declines roughly 3–4% across a 16x
growth in context length. Determinism: on both GPUs, every decoded token
and every layer's key/value state is bit-identical to the CPU reference,
checked at every layer-budget granularity the engine supports, down to one
layer per decode call. On the RDNA3 run, the full 45,845-token argmax
sequence checked matched the reference exactly, token for token.

Adapter switching (attaching a different LoRA specialization to an
already-resident model) was measured on the RTX 2080 SUPER against a real
1.5B base model and a real LoRA adapter: 0.128 s, versus 7.52 s to reload
the model with a different adapter merged in at load time — about 58x
faster, with the base model's resident weights untouched by the switch.

### Schema-constrained decoding on the GPU

The GPU twin of schema-constrained decoding (see [api.md](api.md)) is
proven bit-identical to the CPU reference on the certified NVIDIA GPU: 80
real decode steps against a real schema and a real model, matching SHA-256
digest between the two paths. The identical check against the certified
AMD GPU has not been run yet — it is the one item outstanding before the
1.0 tag. This is a narrower, additional check on top of the base
(unconstrained) determinism guarantee both certified GPUs already carry
above; it does not affect that guarantee.

### The integrated GPU: known, scoped divergence

The same RDNA3-generation machine's integrated GPU ("AMD Radeon Graphics")
passes the direct-dispatch determinism check bit-for-bit, exactly like the
two certified discrete GPUs above. Its asynchronous decode path, however,
diverges from the CPU reference starting at the second decode step, at
every layer-budget granularity tested — a deterministic divergence, not an
intermittent one, and distinct from the direct-dispatch path that shares
the same underlying kernels. This is under active investigation and the
integrated GPU is explicitly not a certified target today: any
determinism claim in this project's documentation is scoped to the
certified GPUs above, never to "GPUs" unqualified.

## What's next

Running the schema-constrained-decoding parity check (above) on the
certified AMD GPU is the one item remaining before the 1.0 tag itself, not
post-1.0 work. Extending GPU certification to a newer NVIDIA generation
(Blackwell) and to integrated GPUs (pending the divergence investigation
above), and measuring macOS beyond what GitHub's own CI runners can
exercise, are committed post-1.0 work — see the README's roadmap section.
