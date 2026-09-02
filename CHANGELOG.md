# Changelog

All notable changes to SuperSLM (Layer 1) are recorded here.

## [Unreleased]

### Fixed

- **The Linux/ELF FP-free scan leg (`_X86_GPR_ALLOW`) now accepts `bswap`.** `superslm::Sha256::
  Final` (`src/sha256.cpp`) compiles, under the runner's own GCC 13.x (`-O3 -DNDEBUG`, matching
  the `linux-x64` job's `-DCMAKE_BUILD_TYPE=Release` recipe), to a `bswap` on the byte-swapped
  big-endian length write -- a pure integer byte-reversal (Intel SDM Vol. 2A: no rounding, no
  exception, no floating-point register read) absent from the checked-in GPR allow-list
  `tests/ci/check_fp_free_scan.py` documents as frozen and reviewed-diff-only. Added,
  individually vetted, following that list's own precedent (`shrd`/`shld`, `cpuid`, `rep`,
  `vzeroupper`, `xgetbv`).

  **The `linux-x64` job's own scan step has NOT actually run** -- this branch has never been
  pushed, and no GitHub Actions run of that job exists at this tip. What stands instead (T-2533,
  closing M-4n/M-5n) is the job's own two-step recipe reproduced end to end, off this exact tip,
  under the real runner's own compiler: a fresh `git clone` of this branch into WSL/Ubuntu (the
  same checkout convention `ubuntu-latest` uses), GCC 13.3.0-6ubuntu2~24.04.1 (the exact release
  T-2530's own review named for the hosted runner, fetched as `.deb` packages and extracted
  without root -- no system package install), `cmake -B build -DCMAKE_BUILD_TYPE=Release` +
  `cmake --build build --target superslm`, then `python3 tests/ci/scan_build_output.py
  --build-dir build --target superslm --isa x86-64`, whose own full output line is (T-2535
  correction, Poirot 2945361-t2534-superslm-ci-green-confirmation2.md M-2: the entry previously
  bolded only up through REFUSE and stopped, truncating the line's own scope-qualifying clause
  -- check (C) is non-gating by design so the gate verdict is unaffected either way, but a
  published "0 REJECT" without the clause that bounds it claims a larger cell than was
  measured): **`Totals: 17 object(s); 505 symbol(s) ACCEPT, 0 REJECT, 0 object(s) REFUSE
  (checks (A)/(B), gating); 344 symbol(s) reject under check (C) alone (non-gating
  diagnostic)`** -- an archive-level result under the runner's own exact compiler, not the
  8-symbol single-object spot check this entry previously cited (that check remains correct as
  far as it goes: `Sha256::Final` alone, 1 REJECT without `bswap`, 0 with it). The 505-symbol,
  344-check-(C)-reject corpus matches CI run 33545319929's own `linux-x64` step exactly (`504
  symbol(s) ACCEPT, 1 REJECT ... 344 symbol(s) reject under check (C) alone` -- the 504/1 split
  there is that run's own report against a since-fixed tip, T-2529's own `bswap` addition
  turning the 1 REJECT to 0 here without moving the check-(C) count, and the two runs' matching
  344 is the genuine corroboration): 447 (`5e128ee`'s own GCC-15.2.0 archive scan, unchanged by
  this diff -- it touches no file under `src/`) + 58 (T-2530's own review, the GCC-13-vs-15.2.0
  corpus-size difference) = 505, matching this run and the CI run's own `504 + 1` identically.
  The job's second step, `./build/superslm_tests`, was also run against this same GCC-13.3.0
  build: `superslm tests: 24310 checks, 0 failures`. Both steps of the job's own recipe pass on
  the reproduced cell; the job itself remains unrun.

  **This closes only the first of 1.3.0's own two deferral conditions for the Linux/ELF leg (no
  run had completed) -- the second is still outstanding, and the guarantee stays deferred.**
  1.3.0 also deferred on the design's own fiftieth population (`t2265-superslm-fp-free-open-
  design-2026-08-24.md` Sec5.4 closing paragraph) never having been built: a must-accept run
  against this wired job's own real archive, and a must-reject run against an archive that
  genuinely carries floating-point arithmetic. Neither has run.
  `tests/t2296-fp-free-open-red-suite/test_archive_gate.py`'s own `real_elf_archive` fixture
  states this plainly, at this same tip: the population "remains outstanding" (D-SLM5230,
  deferred to 1.3.1). An instrument whose must-reject has never fired has not been shown able to
  fail -- so the Linux/ELF no-floating-point guarantee is **enforced on the reproduced cell above
  and not yet delivered**; 1.3.1 is where it is delivered, once that population runs.

- **`tools/convert_tokenizer.py`'s `derive_model_name` now parses a checkpoint path's own
  separators directly, regardless of the OS running the converter.** A Windows-style checkpoint
  path (backslash-separated, this project's own HF hub cache convention) previously returned the
  whole path as the emitted CONFIG section's model label when the converter ran on a POSIX host
  -- `pathlib.Path` splits only on the running OS's own separator convention. User-visible: the
  label every converted `.sslm` artifact's CONFIG section carries.

### Added

- **The converter learns the checkpoint's own namespace convention and its QK-norm
  tensors (Ask 5 Track C, T-2539).** `_upstream_names` now detects, from a single anchor
  tensor, whether a checkpoint's transformer backbone is `model.`-prefixed (every
  incumbent through Qwen2.5) or bare (this ask's own candidate) -- a checkpoint matching
  neither convention, or both, is a named rejection, never a guess. Q/k/v projection
  biases and the two new QK-norm gain tensors per layer are each included only when the
  checkpoint's own key set carries that exact tensor. `sslm_convert_validate.
  check_required_groups` no longer demands `dynamic_biases` be non-empty -- a bias-free
  checkpoint is a legitimate architecture fact, not a calibration-bug symptom, since that
  group is built by a total, filter-free comprehension. `_derive_composition_constants`
  gains a `q_norm`/`k_norm` offline composition-constant loop, gated on presence,
  applying the identical formula the existing `attn_norm`/`mlp_norm` loop uses. No
  forward-path code, GPU shader, tokenizer converter, or ABI change -- the converter half
  of Qwen3-architecture support only (`SuperSLM_Plan.md` §22.5 Track C).

  **Product claim, executed against the real, pinned candidate** (Qwen3-Embedding-0.6B,
  revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`) -- **a LOAD-TIME claim only; see
  the staleness note below before treating this artifact as an input to anything.**
  Rejected before this round with 310 of 310 checkpoint tensors unmapped and 338 of 338
  map entries missing. After: `calibrate_checkpoint.py` then `convert_model.py` (with the
  engine's own compiled `sslm_verify` invoked, not skipped) ran end to end --
  `verified: independent loader accepted the artifact`, 9 sections (Config, Weights,
  Biases, RopeTables, WeightScales, CompositionConstants, KvLandingScales,
  KvLandingReciprocals, SigmoidLut), `config_geometry.ok: true`. The emitted `.sslm` is
  **633,576,276 bytes**, SHA-256
  **`5cf871fbfc2153e6296913c0ef602dfeafac005548adbb4d3a8b34db14ec2aae`**, and its own
  proof manifest's `weight_scales_evidence` carries all 56 `q_norm`/`k_norm` gain entries
  (28 layers x 2).

  **Calibration staleness (T-2543 S-4).** This artifact's numerical content is stale by
  construction and NOT to be reused as an input once Ask 5 Track B step 6 lands: the
  calibration forward (`_float_layer`) applies no QK-norm today, so every scale in this
  artifact was derived from a Qwen2.5-shaped forward trajectory over an architecturally
  Qwen3 checkpoint. `t2408`'s own Track B step 6 already rules this: landing the real
  QK-norm forward call site "re-prices production calibration and invalidates every
  cached calibrated artifact for this candidate," and this candidate's own calibrated
  artifacts "go stale and must be discarded, not reused," once that step lands. The
  SHA-256 above remains the correct identity of what this round produced; it names an
  artifact proven to convert and load, not one proven numerically correct.

  **T-2543 fix round (code review FIX-THEN-SHIP,
  `Claude/Poirot/2a46a85-t2540-ask5-trackc-review.md`).** The namespace-detection fix
  above now protects `sslm_convert_adapter.py`'s LoRA merge path too: before this fix, a
  bare-convention checkpoint reaching the merge loop matched zero adapter keys against a
  still-hardcoded `model.`-prefixed adapter key set, merging nothing, printing success,
  and writing a checkpoint byte-equal to the base -- a silent wrong model, where the
  pre-Track-C engine raised loudly on the identical input. Both the merge site and its
  sibling (`read_base_projection_weight`) now detect the checkpoint's own namespace via a
  new, shared `detect_namespace` helper. Asymmetric `q_norm`/`k_norm` presence (one
  tensor present, the other absent, on the same layer) is now a named converter-side
  rejection, matching design §4's "defined rejection, not two independent null checks."
  `_weight_scales_from_float_source` now reads a float source's own real population
  rather than `_weight_shapes`'s unconditional full set, closing a latent `KeyError` on
  any real pre-Ask-5 checkpoint reaching `calibrate_kv_landing_arm`. Full test suite
  reconciled: 1937 (T-2539's own tip) + 6 new T-2543 cells = **1943/1943 passed**
  (`-m "not upstream"`, real checkpoints present); 1939 passed/4 skipped CI-faithful.
  `pytest tests/ci/`: 423 passed, unchanged. Fix log:
  `Claude/Brunel/t2543-ask5-trackc-fix-round-2026-09-02.md` (records worktree). Build
  log: `Claude/Brunel/t2539-ask5-trackc-build-2026-09-02.md` (records worktree).

  **T-2549 close-out (confirmation review FIX-THEN-SHIP,
  `Claude/Poirot/e0fdd60-t2544-ask5-trackc-confirmation.md`).**
  Closes the confirmation review's one remaining Significant (the C-1 sibling's `ns`
  parameter is now required, not defaulted -- a reverted call site is a `TypeError` on
  every real caller, confirmed by direct execution) and four Minors (a genuine, non-
  duplicated safetensors writer; this entry's own trailing paragraph no longer ends on
  superseded totals; an unrecognized `qk_norm` fixture sentinel is now a named rejection;
  the required-norm-gains docstring phrase made exact). Full suite reconciled: 1943
  (T-2543's own final) + 1 new cell = **1944/1944 passed** (`-m "not upstream"`, real
  checkpoints present); 1940 passed/4 skipped CI-faithful. `pytest tests/ci/`: 423 passed,
  unchanged. The 633 MB product artifact's identity is unaffected -- no calibration re-
  run.
  Close-out log: `Claude/Brunel/t2549-ask5-trackc-close-out-2026-09-02.md` (records
  worktree).

## [1.3.0] - 2026-08-29

This release ships one of five requested consumer-driven changes: the FP-free load path
SuperEmbedder's first buildable encoder unit needs. The other four remain open follow-up work
for later 1.x releases.

### Added

- **A new guarantee: checks (A) and (B) decide, by disassembly, that the `superslm`
  CMake target's compiled object output contains no floating-point arithmetic instruction, over
  the archive the platform's own build produces — enforced on Windows/COFF for this release. The
  Linux/ELF leg is wired into CI (see the job below), not yet enforced: no run of it has completed
  and the design's own commissioning population for that leg is not yet built, so the guarantee
  is not made for Linux/ELF in 1.3.0 and is deferred to 1.3.1, once that leg's own run is green.
  Not enforced on macOS, where no Mach-O reader exists and macOS is ruled out of 1.3.0's own
  scope.** "Floating-point arithmetic instruction" means an instruction whose semantics compute a
  numeric result under IEEE-754 rules (addition, subtraction, multiplication, division, square
  root, fused multiply-add, rounding conversion, or a numeric comparison that reads operand bits
  as a float). Decided by disassembling every archive member: checks (A) and (B) accept
  known-safe move and bitwise-logical mnemonics from an explicit, checked-in list, and
  packed-integer mnemonics from a second explicit, checked-in allow-list — a frozen snapshot of
  the vocabulary's own `p`/`vp` naming-convention membership with a six-entry, individually
  vetted deny list removed, not a fresh per-mnemonic re-derivation — and reject everything else
  that touches the vector/FP register file, including any future packed-integer-shaped mnemonic
  that is not on that frozen list. **The guarantee covers only arithmetic present in SuperSLM's
  own compiled objects — it asserts nothing about arithmetic an external callee (the CRT, the
  STL, a consumer-installed callback) might itself perform.**
- **A new CI job, `fp-free-scan-gate`**: configures and builds the
  `superslm` target, scans the resulting archive with `scan_build_output.py`, and fails
  the workflow on any rejected symbol or an archive member that cannot be read or recognized.
  Defined in `.github/workflows/tests.yml`. Runs the arc's own red suite
  (`tests/t2296-fp-free-open-red-suite`) in the same job, against the
  same build. The `linux-x64` job also scans its own build's archive with the same driver; that
  leg is not enforced for this release.

### Changed

- **Internal hash containers.** `std::unordered_map`/`std::unordered_set`, used for the anti-LM's
  per-context n-gram tables (`src/damped_greedy_antilm.cpp`), the live-sequence registry
  (`src/sslm_abi.cpp`), duplicate tensor/constant-name detection (`src/model.cpp`), and the
  tokenizer's BPE-merge and Unicode-normalization tables (`src/tokenizer.cpp`), are replaced by
  this release's own open-addressing containers (`src/detail/int_hash.h`,
  `src/detail/context_hash.h`) — removes the standard library's own bucket-count floating-point
  division from the compiled corpus, which is what the guarantee above depends on not being
  present. Internal implementation detail; no public signature changes.
- **Seven compiled symbols restructured from a `switch` to a chain of direct conditional
  branches**, removing a compiler-emitted jump table from each symbol's own compiled extent:
  `SslmModelStatusName`, `ValidateSectionValues` (via its inlined `ValidateConfigGeometryJoin`),
  `SslmForwardStatusName`, `BuildProofManifestJsonImpl`, `ConfigGeometryStatusName`,
  `IsKnownSectionType`, and `ExpectedDtype`. Behaviourally identical in every case — the same
  inputs map to the same outputs.
- `AntiLmRetainedBytes`'s documentation (`include/superslm/sslm_damped_greedy.h`) is corrected:
  the prior footprint calibration was fit to `std::unordered_map`'s own bucket shape and is
  retracted as measured-false against this release's own containers, replaced with measured
  lower-bound ranges by `max_order`, sampled across vocabulary size and generation length.
  Comment-only — no declaration changed; this release carries no ABI change.
- **`tools/convert_tokenizer.py`'s emitted CONFIG model label is now derived from the checkpoint
  directory path** instead of being hardcoded to `qwen2.5-1.5b-instruct` — every converted
  checkpoint previously carried that label regardless of which model it actually was. Seven new
  tests cover the derivation (commit `87e0639`).

## [1.2.1] - 2026-08-24

Twelve correctness items closed against 1.2.0, red-first (`Claude/Plans/SuperSLM_1p2p1_Plan.md`
plan of record; test design `Claude/Curie/t2243-1p2p1-red-suite-2026-08-22.md`). Four items priced
in the same review (S1, M1, S3, a perf-footprint item T-2236) are deferred to a later release —
see that plan's own deferral table; they carry no line here.

### Fixed

- **`sslm_gpu_seq_restore` now rejects `SSLM_BUSY` while any sequence on the target model holds
  an unfenced, in-flight decode submission**, closing an ordering hazard between that in-flight
  work and the restore's own device round-trip (a fresh K/V buffer allocated and uploaded against
  the same device without waiting for the sibling's fence). Genuinely transient: drains the
  instant the in-flight sequence's own fence signals.
- **`sslm_gpu_model_unmap` now rejects while any adapter is still mapped against the model**
  (`SSLM_MODEL_HAS_LIVE_ADAPTERS` — see Changed, below). An adapter handle's retained model
  pointer is never dereferenced today but was left dangling by an unmap that ignored it.
- **The tokenizer's special-token table is now validated for longest-content-first ordering.** A
  hand-built artifact with non-monotonic special-token content lengths is rejected at load
  (`TokenizerRejected`) instead of accepted silently — restores parity with the writer-invariant
  checks the parser already runs for vocab-offset monotonicity and unicode-range sortedness.
- **The damped-greedy forward loop now enforces `out_tokens_capacity` before every token/logit-row
  write**, rejecting an undersized caller buffer (`SSLM_INVALID_ARGUMENT`) instead of writing past
  it — closes a memory-safety hole reproducible under ASan.
- **`RunGreedyDecodeLoop` (the plain-greedy sibling of the loop above) now enforces the same
  `out_tokens_capacity` bound**, closing the identical undersized-buffer hole on its own call
  path. No live overflow existed in any first-party caller, all of which already size the buffer
  to `max_new_tokens`; closed as the root class rather than as an active defect.
- **A non-DGC1 (greedy-only) artifact's workspace no longer grows unconditionally.** The
  `damped_indices` scratch region is now reserved only when the mapped model actually carries the
  damped-greedy feature — restores the pre-1.2.0 workspace-sizing formula for every caller that
  never opted into damped-greedy decoding. See also the retroactive disclosure, below: 1.2.0
  itself grew every caller's workspace unconditionally, undisclosed at the time.
- `sslm_gpu_ready` no longer silently discards a null-in-flight-token status
  (`RunLayerLoopGpuFinish`'s own caller-error rejection) as `SSLM_OK`/`*out_ready=0` — the real
  status now surfaces through `*out_status`. Affects only a state no legitimate public caller can
  reach through the documented API alone.
- **`sslm_seq_save` now serializes the carried residual for a sequence resting at ready-for-logits
  (post-`sslm_prefill`/`sslm_seq_adopt_prefix`, `layer_index == 0`), closing a shipped 1.2.0
  defect.** The prior predicate keyed residual presence off `layer_index != 0` alone, so a
  ready-for-logits sequence — which carries a real, load-bearing residual — saved zero residual
  bytes; `sslm_seq_restore` then reconstructed `ready_for_logits = true` over that all-zero
  residual, and the next `sslm_decode_step` produced whatever token the model's head weights map
  zero to, independent of any speculative-decoding mechanism (proven by execution: token 0 emitted
  vs. 97 wanted). Fixed by writing the residual unconditionally whenever `hidden_size > 0` —
  mirroring the GPU blob format's own identical fix (T-2114/C1). See Changed, below, for the new
  blob format this required.

### Added

- **`sslm_gpu_seq_bind_adapter(ctx, seq, adapter_or_null)`** binds (or, passed a null adapter,
  unbinds) a LoRA adapter to a GPU sequence handle *across* calls — distinct from the existing
  per-call `adapter_or_null` argument every decode call already takes. A bound adapter is read
  automatically by the recommended one-call bridge (`SslmGpuSeqDecodeStepForG5Bridge`) and by the
  chunk-prefill entry points; it does not change what a direct `sslm_decode_step_gpu`/
  `sslm_decode_step_batch_gpu` caller must still pass explicitly. This is the mechanism serial
  specialist-switching needs: decode under one adapter, rebind to a different one, decode again,
  on the same live sequence, without an unwanted extra token or losing K/V state. Rejects a
  model-mismatched or foreign-context adapter, and rejects mid-token (`SSLM_BUSY` — a drained rest,
  at either token boundary, always admits). Unbinds automatically on `sslm_gpu_seq_release`;
  survives `sslm_gpu_seq_reset`; does not round-trip through save/restore.

### Changed

- **New `SslmGpuStatus` members, appended last, no existing value moved:**
  `SSLM_MODEL_HAS_LIVE_ADAPTERS` (`sslm_gpu_model_unmap`, above) and
  `SSLM_ADAPTER_HAS_BOUND_SEQUENCES` (`sslm_gpu_adapter_unmap` now rejects while any sequence
  still holds a bind to that adapter, via the new bind verb above). Both are persistent-liveness
  conditions — they hold until the caller explicitly unmaps/unbinds, never draining on their own —
  distinct from the existing transient `SSLM_BUSY`.
- **`anti_lm_max_order` now has a ceiling of 82**, enforced on both the caller-supplied-params
  path (`ValidateDampedGreedyParams`) and `sslm_seq_restore`'s blob path; `83` and above are
  rejected `SSLM_INVALID_ARGUMENT` on either. Derived from the shipped fixed-point recurrence
  (`kBetaQ15`) as the last order whose contribution does not underflow to zero — a blob-format
  constraint as much as a caller-params one, since `sslm_seq_restore` reconstructs the identical
  object from an untrusted 4-byte field.
- **Retroactive disclosure (1.2.0):** the workspace-region growth this release now makes
  conditional on `damped_greedy_available` was, in 1.2.0, unconditional for every caller and was
  not disclosed as a narrowing at the time. 1.2.1 restores the pre-1.2 sizing for non-DGC1
  artifacts; see Fixed, above.
- `docs/api.md` updated: `sslm_gpu_seq_restore` added to the calls needing external
  serialization (it now performs real device work under a Busy-precedence guard, above); the new
  bind verb and both new statuses documented.
- **The CPU sequence save/restore format bumps to a new magic, `SSB4`.** It supersedes `SSB3`
  (shipped 1.2.0) with two changes: the residual is now serialized unconditionally whenever
  `hidden_size > 0` (see Fixed, above), and a new explicit `ready_for_logits` field is appended to
  the fixed header, so restore reads that state directly instead of inferring it from
  `layer_index`/`context_length` alone. `sslm_seq_save` writes only `SSB4`; `sslm_seq_restore`
  continues to accept shipped `SSB3` and `SSB2` blobs read-only, unchanged in this respect from
  1.2.0's own `SSB3`/`SSB2` compatibility promise. `sslm_seq_state_size`'s upper bound grows by 4
  bytes (the new field) to 128.
- **New `sslm_status` member `SSLM_RESTORE_RESIDUAL_LOST`, appended last, no existing value
  moved.** `sslm_seq_restore` now returns it for a legacy `SSB3` **or `SSB2`** blob in the one
  state the 1.2.0 defect above could produce (`layer_index == 0 && context_length > 0` with no
  pending-embed token saved) — the lost residual cannot be recovered, since it was never written,
  but the caller now gets a loud, diagnosable failure instead of a silently wrong restore. `SSB2`
  carries the identical defect at the identical field offsets and is extended in this release
  (D-SLM4114); the current `SSB4` format never produces this state at all.

## [1.2.0] - 2026-08-21

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
- Sequence saves now use `SSB3` to carry damped anti-LM history while restore
  remains backward-compatible with shipped `SSB2` blobs. Restore accepts
  state-size-capacity buffers with trailing bytes and rejects history longer
  than the saved context.
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
