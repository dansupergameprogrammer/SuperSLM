# Platform support

"Certified" in this document means: built and run on that exact hardware,
with every determinism check (bit-identical output against the CPU
reference, at every layer-budget granularity the engine supports) passing
on that run.

## CI execution status

This is the ONE place this project states what its continuous integration
has actually executed, and every other public document (README, CHANGELOG,
API docs) points here rather than restating it — a present-tense
CI-execution claim living in more than one place is how the same claim goes
stale in one copy while staying correct in another (T-2192/T-2195, three
consecutive review rounds).

"CI extent" means the platform is exercised by this project's continuous
integration matrix, which exists and defines the full build + test job for
each platform below (`.github/workflows/tests.yml`: windows-x64, linux-x64,
linux-x64-asan, macos-arm64). **GitHub Actions hosted runs are currently
capped by the account's spending limit and have not executed since
2026-07-23.** The matrix **was fully green on the 1.0 release commit** —
every platform leg, sanitizer leg, cross-toolchain digest leg, and CI
checker — a dated, past-tense fact that stays true regardless of hosted-run
status today. Since the cap, the same matrix each job runs is reproducible
locally (this project's own `cmake`+`ctest`, per platform, and `build.bat`
on Windows) and that local run is the verification path CI extent below
stands on until hosted runs resume. Every number below states the device,
the artifact, and the surface it was measured through — a number without
that context is not included here.

## CPU inference

| Platform | Status | How it's exercised |
|---|---|---|
| Windows x64 | Certified | Local build + full test suite; CI job defined, hosted runs currently capped |
| Linux x64 | CI extent | `cmake`+`ctest` (GCC and Clang toolchains) — locally reproducible; GitHub Actions (`ubuntu-latest`) job defined, hosted runs currently capped |
| macOS (Apple Silicon, arm64) | CI extent | `cmake`+`ctest` — locally reproducible; GitHub Actions (`macos-latest`) job defined, hosted runs currently capped |

Cross-toolchain determinism at the primitive level (the SiLU
lookup-table and integer matmul kernels) is checked by comparing a digest
of each kernel's output across nine toolchain axes: six pre-1.1 axes —
Linux/GCC, Linux/Clang, Linux/Clang with SIMD forced off (the scalar
tier), Windows/MSVC, Windows/Clang-cl, and macOS/Clang (arm64) — plus
three axes 1.1 adds, each forcing one of the remaining matmul kernel
tiers on Linux/Clang: SSE2, AVX2, and AVX-512. A forced axis whose runner
lacks the tier's required hardware is a loud, named skip, not a silent
pass — this is how the AVX-512 axis behaves on every runner available to
this project today.

**What has actually run, as of this writing:** the six pre-1.1 axes
matched byte-for-byte as of the 1.0 release. Of the three new axes, SSE2
and AVX2 have each been run locally — one machine, one toolchain — and
match the scalar/SSE2/AVX2 golden digest exactly. The AVX-512 axis has
not produced a digest anywhere: this project's own build and test
hardware has no AVX-512 support (a forced AVX-512 binary compiles clean
and exits via SIGILL when run), and the CI leg's designed outcome on a
runner without the hardware is the same loud SKIP, not a pass — a real
AVX-512 evidence run on capable hardware is owed. The nine-way
comparison across GitHub Actions' own hosted runners has not run at
all yet — hosted runs are capped, per this document's own opening note
— and lands at the first real Actions matrix run after that cap lifts.

### CPU matmul kernel tiers

The integer matmul kernel runtime-dispatches among three SIMD tiers —
SSE2 (the unconditional x86-64 architectural floor), AVX2, and AVX-512 —
selected once per process by a CPUID+XGETBV probe, with SSE2 as the
automatic fallback on hardware lacking the wider tiers. All three tiers,
plus the scalar reference, are proven bit-identical against each other; a
consumer never trades correctness for the faster tier its hardware
happens to support.

| Tier | Status | How it's exercised |
|---|---|---|
| Scalar | Certified | Every platform's default build; also independently force-selectable |
| SSE2 | Certified | The x86-64 architectural floor; force-selectable; CI leg defined (1.1), not yet executed on a hosted runner (capped since 2026-07-23) — verified locally, matches the golden digest exactly |
| AVX2 | Certified, measured | Force-selectable; CI leg defined (1.1), not yet executed on a hosted runner — verified locally, matches the golden digest exactly; throughput measured on real hardware (below) |
| AVX-512 | CI extent | Force-selectable; CI leg defined (1.1), not yet executed on a hosted runner or on any AVX-512-capable machine this project has access to — bit-identity unverified; no dedicated throughput measurement published yet — the same disposition this document gives macOS below |

**Measured, real 1.5B-parameter model artifact, batched prefill, SSE2 to
AVX2, this project's own reference AMD hardware (Zen 2, no AVX-512):
about 1.68x-1.72x faster**, two independent runs, each its own paired
SSE2/AVX2 baseline: run 1 at 6.93 → 11.68 tok/s (1.68x), run 2 at
6.60 → 11.33 tok/s (1.72x) — within the 2-5% run-to-run noise this
project measured on this machine. AVX-512's own throughput is not yet
measured on real hardware — tracked as a known gap below, not assumed
from the AVX2 figure.

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

### GPU-side batched prompt prefill

Prompt prefill on the GPU path records and submits a whole prefill span in
one device round trip rather than one round trip per token. Measured on the
certified NVIDIA RTX 2080 SUPER, forced prefill spans against a real
1.5B-parameter model, comparing the pre-batching one-round-trip-per-token
path against the batched path through the same public entry point:

| Span length | Pre-batching (per-token) | Batched (one call) | Speedup |
|---|---|---|---|
| 128 tokens | 8.62 tok/s | 61.96 tok/s | 7.19x |
| 256 tokens | 8.96 tok/s | 61.93 tok/s | 6.91x |

Both paths proven bit-identical at every span size and every internal
split boundary tested. **The internal split bound stated honestly:** a
prefill span larger than a certain size is submitted as multiple smaller
sub-chunks rather than one, each finished before the next opens. That size
is not a theoretical or configured limit — it was found empirically, by
running progressively larger spans against this GPU's real driver until an
unrelated, third-party driver defect reproduced deterministically at one
exact size, and set to half that size as a safety margin. It is measured
on this one device/driver pairing only; a second vendor's own data point
is a named gap below, not assumed to match.

This measurement is on the certified NVIDIA GPU only — see
[Known gaps](#known-gaps) below for AMD certification status on this
specific capability.

### Schema-constrained decoding on the GPU

The GPU twin of schema-constrained decoding (see [api.md](api.md)) is
proven bit-identical to the CPU reference on both certified GPUs: real
decode steps against a real schema and a real model, matching SHA-256
digest between the two paths at 18, 24, and 80 steps. Measured on the
NVIDIA RTX 2080 SUPER, and again on the AMD Radeon RX 7900 XTX
(2026-08-17), with the identical digest at every step count between the
two vendors as well as between each vendor's own CPU/GPU paths. This is a
narrower, additional check on top of the base (unconstrained) determinism
guarantee both certified GPUs already carry above; it does not affect that
guarantee.

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

## Known gaps

- **AVX-512 has no dedicated real-hardware throughput measurement yet, and no
  bit-identity run at all.** The CI leg is defined (probes, compiles, and
  would run wherever a runner has the hardware) but has not yet executed on
  any hosted runner or any AVX-512-capable machine this project has access
  to — see [CPU matmul kernel tiers](#cpu-matmul-kernel-tiers) above — so
  neither its bit-identity nor its tokens/second figure, alongside SSE2's and
  AVX2's above, is yet published.
- **GPU-side batched prompt prefill is measured and certified on the
  certified NVIDIA GPU only.** The pre-batching, per-token GPU path stays
  certified on both certified GPUs, unaffected; certifying the batched
  path on the certified AMD GPU as well — including spans at and above the
  internal sub-chunk split bound — is the next step for this capability
  specifically. The split bound itself (see
  [GPU-side batched prompt prefill](#gpu-side-batched-prompt-prefill)
  above) is measured on the NVIDIA device/driver pairing only; a second
  vendor's own value is unmeasured.

## What's next

Extending GPU certification to a newer NVIDIA generation (Blackwell) and
to integrated GPUs (pending the divergence investigation above), closing
the two gaps above, and measuring macOS beyond what GitHub's own CI
runners can exercise, are committed post-1.1 work — see the README's
roadmap section.
