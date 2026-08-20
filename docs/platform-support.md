# Platform support

"Certified" in this document means: built and run on that exact hardware,
with every determinism check (bit-identical output against the CPU
reference, at every layer-budget granularity the engine supports) passing
on that run.

## CPU inference

| Platform | Status | How it's exercised |
|---|---|---|
| Windows x64 | Certified | Local build + full test suite; hosted CI leg executed green 2026-08-20 (run 32336576519) |
| Linux x64 | CI extent | `cmake`+`ctest` (GCC and Clang toolchains); hosted CI legs (incl. ASan/TSan) executed green 2026-08-20 |
| macOS (Apple Silicon, arm64) | CI extent | `cmake`+`ctest`; hosted CI leg executed green 2026-08-20 |

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

**What has actually run, as of 2026-08-20:** eight of the nine axes
executed on GitHub Actions' own hosted runners in one matrix run
(32336576519) and produced byte-identical global digests. The ninth —
the forced-AVX-512 axis — reported its designed loud SKIP on that run
(the hosted runner lacked AVX-512F/BW), and its digest is instead proven
by a manual evidence run of the full forced-AVX-512 suite on real
AVX-512 silicon (Zen 4): 34,174 checks, 0 failures, global digest
matching the other eight axes exactly. Between them, all nine axes have
produced matching digests on executed hardware.

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
| SSE2 | Certified | The x86-64 architectural floor; force-selectable; forced full-suite and digest CI legs executed hosted 2026-08-20 (run 32336576519), green, digest matching |
| AVX2 | Certified, measured | Force-selectable; forced full-suite and digest CI legs executed hosted 2026-08-20, green, digest matching; throughput measured on real hardware (below) |
| AVX-512 | Certified (bit-identity) | Force-selectable; full forced suite executed on real AVX-512 silicon (Zen 4): 34,174 checks, 0 failures, cross-tier digest matching every other tier. Dedicated throughput measurement still open (known gap below). CI leg probes and reports SKIPPED honestly on runners without the hardware |

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

| GPU | Span length | Pre-batching (per-token) | Batched (one call) | Speedup |
|---|---|---|---|---|
| NVIDIA RTX 2080 SUPER | 128 tokens | 8.62 tok/s | 61.96 tok/s | 7.19x |
| NVIDIA RTX 2080 SUPER | 256 tokens | 8.96 tok/s | 61.93 tok/s | 6.91x |
| AMD Radeon RX 7900 XTX | 256 tokens | 5.33 tok/s | 73.44 tok/s | 13.8x |

The AMD measurement is larger because that driver's per-call round-trip
cost is higher, so removing the round-trips buys more. Same binaries, same
artifacts, same public entry point, run from this release's own evidence
package.

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

Batched prefill is certified on both certified GPUs: the full release
evidence suite — bit-identity at every span size and split boundary, the
exit-path census, the fault-recovery cells, and the sub-chunk-bound spans —
ran clean on the AMD RX 7900 XTX with counts matching the NVIDIA and
in-repo certifications exactly. The AMD driver also handles spans at and
above the empirically-set sub-chunk bound without incident, giving the
bound its second-vendor data point.

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

## CI execution status

This is the ONE place this project states what its continuous integration
has actually executed, and every other public document (README, CHANGELOG,
API docs) points here rather than restating it — a present-tense
CI-execution claim living in more than one place is how the same claim goes
stale in one copy while staying correct in another (T-2192/T-2195, three
consecutive review rounds).

"CI extent" means the platform is exercised by this project's continuous
integration matrix (`.github/workflows/tests.yml`: windows-x64, linux-x64,
linux-x64-asan, macos-arm64, plus the forced-tier and digest legs 1.1
adds). **Hosted runs resumed on 2026-08-20**: the full 28-job matrix
executed on the 1.1 candidate (run 32336576519), every leg green except
the branch-coverage job's designed first-run red — that job records its
own floor measurement and fails until the recorded number is committed,
which the immediately following commit did. The matrix was also fully
green on the 1.0 release commit. Between 2026-07-23 and 2026-08-20 hosted
runs were capped by the account's spending limit and the same matrix was
reproduced locally (`cmake`+`ctest` per platform, `build.bat` on Windows).
Every number below states the device, the artifact, and the surface it was
measured through — a number without that context is not included here.


## Known gaps

- **AVX-512 bit-identity is proven on real hardware; its dedicated
  throughput figure is not yet published.** The full forced-AVX-512 suite
  has executed on AVX-512 silicon (Zen 4): 34,174 checks, 0 failures, with
  the cross-tier digest matching every other tier exactly. What remains
  open is a tokens/second measurement for the AVX-512 tier alongside the
  SSE2 and AVX2 figures above, and a hosted-runner execution of the CI leg
  (which probes and reports SKIPPED honestly until the scheduler provides
  capable hardware).

## What's next

Extending GPU certification to a newer NVIDIA generation (Blackwell) and
to integrated GPUs (pending the divergence investigation above), closing
the two gaps above, and measuring macOS beyond what GitHub's own CI
runners can exercise, are committed post-1.1 work — see the README's
roadmap section.
