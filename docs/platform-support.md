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
| AVX-512 | Certified (bit-identity) | Force-selectable; full forced suite executed on real AVX-512 silicon (Zen 4): 34,174 checks, 0 failures, cross-tier digest matching every other tier; throughput measured (below). CI leg probes and reports SKIPPED honestly on runners without the hardware |

**Measured, real 1.5B-parameter model artifact, batched prefill, SSE2 to
AVX2, this project's own reference AMD hardware (Zen 2, no AVX-512):
about 1.68x-1.72x faster**, two independent runs, each its own paired
SSE2/AVX2 baseline: run 1 at 6.93 → 11.68 tok/s (1.68x), run 2 at
6.60 → 11.33 tok/s (1.72x) — within the 2-5% run-to-run noise this
project measured on this machine.

**Measured, same artifact, AVX2 to AVX-512, on AVX-512-capable hardware
(Zen 4): about 1.18x faster** — AVX2-forced 20.15–20.33 tok/s, AVX-512-forced
23.83–23.95 tok/s, batched prefill, two reps each. The gain is modest by
design of the silicon: Zen 4 executes 512-bit operations double-pumped
through 256-bit units, so the wider tier buys a denser instruction stream
rather than a wider datapath. Every rep also re-proved chunk-boundary
split bit-identity on the AVX-512 tier — the first such execution on
AVX-512 silicon.

### Tiled prefill GEMM (unreleased)

On the AVX2 and AVX-512 tiers, a prefill GEMM of 8 or more tokens runs a
register-tiled kernel over weights packed per call (4 tokens x 16 outputs on
AVX2, 8 x 32 on AVX-512BW) instead of one dot product per (token, output)
cell. Every output is bit-identical to the scalar reference: the kernel adds
the same exact products and flushes its 32-bit lanes to 64 bits within the
bound the shipped tiers already use. Below 8 tokens, on the scalar and SSE2
tiers, and for decode, nothing changes.

| Build | Tiled path | Status |
|---|---|---|
| GCC / Clang, AVX2 tier | on at M >= 8 | Bit-identity: full suite forced AVX2, tiled golden, cross-tier digest, save-blob equality against 1.9.0 on the in-tree fixture and on synthetic real-width artifacts |
| GCC / Clang, AVX-512 tier | on at M >= 8 | Same evidence, forced AVX-512 and auto dispatch |
| MSVC / clang-cl, AVX2 tier | on at M >= 8 | Full suite, auto and forced AVX2, green on the Windows CI legs and on a Zen 2 desktop (MSVC 19.33, clang-cl 15); digests equal to the GCC/Clang builds |
| MSVC / clang-cl, AVX-512 tier | **off** (`SUPERSLM_TILED_AVX512_MSVC=0`) | Held on the shipped per-row kernel until an MSVC AVX-512 build has executed the tiled kernel |

**Measured, engine level, one GEMM, on a 4-vCPU cloud Xeon (AVX2 and
AVX-512BW, shared host, best of N):** at the 1.5B `gate`/`up` shape
(K 1536, N 8960) and 32 tokens, the tiled AVX2 kernel is 2.05x the shipped
AVX2 kernel (median of 15 interleaved pairs, quartiles 1.99-2.06), and the
tiled AVX-512 kernel 2.14x the shipped AVX-512 kernel. Across the 0.5B,
0.6B and 1.5B projection shapes at 8 to 512 tokens the range is about
1.7x-3.0x on AVX2 and 1.6x-4.3x on AVX-512. The top of each range is the
1.5B `down` shape at 512 tokens, where the shipped kernel itself slows down;
the tiled kernel is not faster there than elsewhere.

**Measured, whole batched prefill, one or two layers of a synthetic artifact
at Qwen2.5-0.5B width (hidden 896, MLP 4864) and Qwen3-0.6B width (hidden
1024, MLP 3072), same host:** at 128 tokens the prefill is about 1.5x-1.6x
faster than 1.9.0, with auto dispatch (AVX-512) and forced AVX2 alike
(1.54x-1.64x, best of 7 runs; 1.48x-1.67x by median). At 32 tokens it is
about 1.35x-1.6x faster, and at 8 tokens about 1.2x-1.4x. An earlier
reading of 1.96x (forced AVX2, 0.5B width, 1 layer) did not reproduce: its
1.9.0 time was 161 ms against 129 ms on a rerun. These
are per-layer engine figures on synthetic weights. They are not a
statement about any consumer's end-to-end speed, and they are not a
measurement on this project's reference hardware.

### Model load integrity hash (unreleased)

Loading a model hashes the whole file with SHA-256. On x86-64 CPUs with the
SHA extensions (CPUID leaf 7 EBX bit 29, with SSSE3 and SSE4.1; most AMD CPUs
since Zen, Intel since Ice Lake and Goldmont) the hash uses those
instructions; everywhere else it uses the portable implementation. The digest
is the same either way.

**Measured, Zen 2 desktop (Ryzen 9 3950X), MSVC, 510 MB Qwen2.5-0.5B model,
best of 3:** 0.31 s with the SHA extensions against 3.25 s for 1.9.0's
portable hash. The portable path itself is also about 1.7x faster than in
1.9.0 (1.95 s).

### Attention prob·V on int16 multiply-add (unreleased)

On the AVX2 and AVX-512 tiers, `GemmProbQ15Accumulate` (the attention
probability row times the value rows, per head) multiplies pairs of keys
with `vpmaddwd` into 32-bit lanes and widens each lane to 64 bits. The
probability pairs are formed in registers. The fast path is taken only when
the head dimension is a multiple of 16 and the row passes the int16
condition: every p in [0, 32767], and sum at most 2^15. Under that condition
no lane can exceed 128 x 2^15 = 2^22, so every output is the exact sum
v1.9.0's loop computes. Every other row takes the v1.9.0 loop, as do the
scalar and SSE2 tiers. On real prompts that fallback is the one-hot rows:
every width-1 row, and rarely a wider one.

| Build | Fast path | Status |
|---|---|---|
| GCC / Clang, AVX2 tier | on | Bit-identity: full suite forced AVX2, the prob·V golden pinned from 1.9.0, cross-tier digest, save-blob equality against 1.9.0 |
| GCC / Clang, AVX-512 tier | on | Same evidence, forced AVX-512 and auto dispatch |
| MSVC / clang-cl, AVX2 tier | on | Built by the forced Windows legs; not yet executed on Windows |
| MSVC / clang-cl, AVX-512 tier | **off** (`SUPERSLM_SITES_AVX512_MSVC=0`) | Held on the 1.9.0 loop until an MSVC AVX-512 build has executed the fast path; the forced AVX-512 Windows legs build with it on |

**Measured, engine level, same host as above (best of 50, 9 interleaved
rounds):** at head dimension 64 one call is 17-21x faster than 1.9.0's
loop at 128 to 1,024 keys on AVX-512, and 16-19x on AVX2. For example, at
601 keys it takes 24.8 µs on 1.9.0, 1.18 µs on AVX-512 and 1.32 µs on AVX2.
Scaled to Qwen2.5-0.5B depth (24 layers x 14 heads), the step saves about
0.82 / 3.4 / 6.7 ms per prompt token at 128 / 512 / 1,024 tokens, and
3.9 / 7.9 ms per decode token at context 300 / 600. A one-layer forward
agrees at 512 tokens and in decode. These are engine figures on synthetic
weights, not a consumer's end-to-end speed
(`docs/attention-rowsites/s2/bench.md`).

### Requantization in 64-bit lanes (unreleased)

Every checked-chain funnel call (`RequantChainChecked`: the projections,
norms, activation and residuals) ends by converting a row of 64-bit
accumulators to int8 codes. On the AVX2 and AVX-512 tiers that conversion
now runs in 4 or 8 unsigned 64-bit lanes (`RequantRowWide`). It computes
the element code's exact identity: the product |x| x r splits into 32-bit
halves, is rounded and shifted, then clamped at 127 and given back its sign.
Every intermediate stays exact up to the funnel's largest input, so every
code equals v1.9.0's per-element `RequantTokenCodeWide`. There is no
runtime guard; the funnel's own preflight is the contract. The last
n mod 4 (or 8) elements, and every element on the scalar and SSE2 tiers,
run the v1.9.0 code.

| Build | Lanes | Status |
|---|---|---|
| GCC / Clang, AVX2 tier | on | Bit-identity: full suite forced AVX2, the requant golden pinned from 1.9.0, cross-tier digest, save-blob equality against 1.9.0 |
| GCC / Clang, AVX-512 tier | on | Same evidence, forced AVX-512 and auto dispatch |
| MSVC / clang-cl, AVX2 tier | on | Built by the forced Windows legs; not yet executed on Windows |
| MSVC / clang-cl, AVX-512 tier | **off** (`SUPERSLM_SITES_AVX512_MSVC=0`) | The same switch as prob·V above |

**Measured, engine level, same host as above (best of 50, 9 interleaved
rounds):** one funnel call at width 4,864 takes 31.5 µs on 1.9.0, 5.6 µs
on AVX-512 and 6.7 µs on AVX2; at 896, 5.2, 1.0 and 1.2 µs. At
Qwen2.5-0.5B depth (24 layers x 11 funnel calls, plus the embed) that saves
about 2.7 ms per token on AVX-512 and 2.5 on AVX2, prefill and decode alike.
These are engine figures on synthetic weights, not a consumer's end-to-end
speed (`docs/attention-rowsites/s3/bench.md`).

### Guarded softmax rows (unreleased)

Attention's softmax row (`SoftmaxRowQ15`) turns a row of scores into Q15
probabilities. On the AVX2 and AVX-512 tiers it first checks a row guard:
width at most 2^14, q_ln2 >= 1, q_c >= 0, M = q_b^2 + q_c in [1, 2^47]
(formed in 128 bits, as the 1.9.0 body forms it), q_ln2 <= 2 q_b + 1, and
every score within 2^61. Inside the guard it computes the row 4 or 8
elements at a time. Each element's quotient by q_ln2 and each
probability's divide by the row total is an integer reciprocal estimate,
corrected exactly by one integer comparison each way, so every probability
and the returned bool equal v1.9.0's. Outside the guard the 1.9.0 body
runs unchanged. The estimates are integer arithmetic, so the library
stays floating-point-free.

| Build | Fast path | Status |
|---|---|---|
| GCC / Clang, AVX2 tier | on | Bit-identity: full suite forced AVX2, the softmax golden pinned from 1.9.0, cross-tier digest, save-blob equality against 1.9.0 |
| GCC / Clang, AVX-512 tier | on | Same evidence, forced AVX-512 and auto dispatch |
| MSVC / clang-cl, AVX2 tier | on | Built by the forced Windows legs; not yet executed on Windows |
| MSVC / clang-cl, AVX-512 tier | **off** (`SUPERSLM_SITES_AVX512_MSVC=0`) | The same switch as prob·V above |

**Measured, engine level, same host as above (best of 30, 9 interleaved
rounds):** a row of 512 keys takes 3.8 µs on 1.9.0, 1.06 µs on AVX2 and
0.90 µs on AVX-512. At Qwen2.5-0.5B depth (24 layers x 14 heads) that saves
about 0.10 / 0.46 / 0.91 ms per prompt token at 128 / 512 / 1,024 tokens
on AVX2, and 0.53 ms per decode token at context 300. A one-key row is
about 0.03 µs slower (the guard and two reciprocal divides per row).
These are engine figures on synthetic weights, not a consumer's end-to-end
speed (`docs/attention-rowsites/s4/bench.md`).

### Q31 attention score rows (unreleased)

QK-norm models (the Qwen3 path) score each key with a Q31 product,
`RoundingDivideByPOT(sum_d q_d * k_d * ratio_d, 31)`. `QkQ31ScoreRow`
computes every key's score for one query head in one call, and both layer
loops (prefill and decode) call it. On the AVX2 and AVX-512 tiers it first
checks a guard: head_dim at most 512 and every ratio in [0, 2^32). Inside the
guard, each channel's w = q * ratio is split exactly into three pieces,
w = a2 * 2^30 + a1 * 2^15 + a0, with a0 and a1 in [0, 32767] and a2 inside
int16. Each piece's sum over the channels is a 16-bit multiply-add into
int32 lanes. At head_dim 512 that sum stays inside int32 by 65,535, so the
guard is load-bearing. The three sums recombine exactly in int64, and the
rounding is vectorised with ties away from zero. Every score equals v1.9.0's
per-key `QkQ31Score`. Outside the guard, the per-key loop runs unchanged. It
is integer arithmetic only.

| Build | Fast path | Status |
|---|---|---|
| GCC / Clang, AVX2 tier | on | Bit-identity: full suite forced AVX2, the Q31 golden and the QK-norm fixture pinned from 1.9.0, cross-tier digest, save-blob equality against 1.9.0 |
| GCC / Clang, AVX-512 tier | on | Same evidence, forced AVX-512 and auto dispatch |
| MSVC / clang-cl, AVX2 tier | on | Built by the forced Windows legs; not yet executed on Windows |
| MSVC / clang-cl, AVX-512 tier | **off** (`SUPERSLM_SITES_AVX512_MSVC=0`) | The same switch as prob·V above |

No real Qwen3 artifact has run the new kernel yet: a QK-norm artifact is
still refused at map time on this host. The evidence is a QK-norm fixture
that drives both layer loops, pinned to v1.9.0, plus a one-layer forward
at Qwen3-0.6B width whose outputs match the base.

**Measured, engine level, same host as above (best of 30, 9 interleaved
rounds, head_dim 128):** one score costs about 410 ns per head and key on
1.9.0's AVX2 path and 13.5 ns on the AVX2 row (330 and 12.8 ns on AVX-512).
At Qwen3-0.6B depth (28 layers x 16 heads) that saves about 11 / 46 / 94 ms
per prompt token at 128 / 512 / 1,024 tokens on AVX2, and 55 ms per decode
token at context 300. A one-layer forward at the same width saves 0.42 /
1.66 / 3.66 ms per layer and token. A one-key row on AVX-512 is about
0.06 µs slower (it packs a whole 16-key block). These are engine figures
on synthetic weights, not a consumer's end-to-end speed
(`docs/attention-rowsites/s5/bench.md`).

### Damped-greedy decoding

The 1.2 candidate's opt-in decoder was confirmed on Windows x64 through the
production CPU generation loop, using real Qwen2.5 0.5B-Instruct and
1.5B-Instruct artifacts. The paired population is 48 fixed prompts at both
100- and 300-token ceilings for each model: 192 greedy generations and 192
damped generations at `alpha=2`, anti-LM order `2`, and `top_k=6`.

| Cell | Greedy locks | Damped locks | Greedy rep-3 | Damped rep-3 |
|---|---:|---:|---:|---:|
| 0.5B / 100 | 3 | 0 | 0.09063 | 0.00172 |
| 0.5B / 300 | 3 | 0 | 0.13959 | 0.00368 |
| 1.5B / 100 | 0 | 0 | 0.01873 | 0.00087 |
| 1.5B / 300 | 0 | 0 | 0.04139 | 0.00196 |

The fixed-work, alternating-arm follow-up measured 4.276667 seconds greedy and
4.277000 seconds damped across six 30-token pairs on 0.5B: the 0.333 ms mean
difference was below the CLI timer's 1 ms reporting resolution. The original
separate-run Phase E timings are retained as raw evidence, not as a decoder-cost
comparison. The complete selector microbenchmark and full verification command
are recorded in the 1.2 review packet. Output quality is workload-dependent:
the measured corpus often became more coherent, while a repeated-list prompt
also demonstrated a real formatting regression. See
[the full confirmation and side-by-side text](calibration/t2199-phase-e-confirmation.md).

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

- **The AVX-512 CI leg has not yet executed on a hosted runner** — it
  probes and reports SKIPPED honestly until the scheduler provides capable
  hardware. The tier itself is fully proven off-CI: bit-identity (34,174
  checks, 0 failures, digest matching) and throughput (about 1.18x over
  AVX2) both measured on real AVX-512 silicon, above.
- **The AVX-512 measurement is one microarchitecture.** The 1.18x figure
  is Zen 4, whose double-pumped 512-bit execution bounds the gain; silicon
  with full-width datapaths (Zen 5, server Intel) would measure
  differently and has not been measured.

- **The tiled prefill GEMM has not yet executed on Windows**, and its
  MSVC/clang-cl AVX-512 path is held off until it has (above).
- **The tiled prefill GEMM is measured on one cloud host only**, on
  synthetic weights; no reference-hardware whole-prefill figure exists yet.

## What's next

Extending GPU certification to a newer NVIDIA generation (Blackwell) and
to integrated GPUs (pending the divergence investigation above), closing
the two gaps above, and measuring macOS beyond what GitHub's own CI
runners can exercise, are committed post-1.1 work — see the README's
roadmap section.
