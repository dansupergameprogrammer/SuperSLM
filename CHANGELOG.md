# Changelog

All notable changes to SuperSLM (Layer 1) are recorded here.

## [Unreleased]

## [1.7.0] - Unreleased

`GpuContextConfig` gains `shader_dir`, the directory the compiled `.cso` shader set is loaded
from. `NULL`, which every zero-initialized config already holds, keeps the existing behaviour:
shaders load from a `shaders` directory beside the host executable. A non-null value is an
absolute UTF-8 directory path, read only during `sslm_gpu_context_create`. The shader directory is
process-wide: the first successful create that names one, or the first shader load through the
default location, fixes it for the process's lifetime. Two statuses are appended to
`SslmGpuStatus`, so no existing ordinal moves: `SSLM_GPU_SHADER_DIR_INVALID` (19), for a value
that is empty, not valid UTF-8, not fully qualified (it must begin with a drive root such as
`C:\` or `C:/`, or a UNC prefix; `\shaders` and `C:shaders` are refused), not an existing
directory, or a directory holding no `.cso` file; and `SSLM_GPU_SHADER_DIR_CONFLICT` (20), for a directory that differs from the one the
process already uses. Both refusals are returned before any device is created.

`GpuContextConfig` grows from 4 to 16 bytes on x64, and it is passed by value. The change is
source-compatible: `GpuContextConfig{}`, `{0}` and every existing call site compile unchanged and
behave as before. It is not binary-compatible: code compiled against an earlier `gpu_1p0.h` must be
recompiled against this one before it links with this release.

`tools/build_inspect.bat` links again. Both of its link lines were missing `src/intmath.cpp`, so
`sslm_inspect` and `tok_verify` failed with unresolved externals from 1.3.0 onward. CMake now
builds both tools as targets (`sslm_inspect`, `tok_verify`) linking the `superslm` library, so a
source file the library needs cannot be missing from them again.

`build.bat` completes again. 1.6.0 added the C ABI verb `sslm_seq_schema_bound`, the 37th, but the
script's ABI verb-count gate still expected 36, so the Windows quick build stopped at that gate in
1.6.0; CMake builds were unaffected. The gate now expects 38: 1.6.0's 37 plus
`sslm_workspace_set_parallel_for`, below. Correction to the 1.6.0 entry below: `SUPERSLM_API` is
carried by all 37 of 1.6.0's C ABI verbs, including `sslm_seq_schema_bound`, not 36.

The token finish -- final norm, logits over the whole vocabulary, then argmax, schema mask,
damped-greedy selection and dead-end rule -- can now move its logits off the calling thread, in
two independent ways. Neither changes a token: every logit is an exact integer sum, and the
narrowing, mask, argmax and dead-end rule run unchanged on the host, on the identical row.

- **A host parallel-for hook, both backends.** New header `superslm/parallel_for.h` defines
  `sslm_parallel_for`, a caller-supplied `run` that the finish hands its logits row blocks to.
  Install it with the new C ABI verb `sslm_workspace_set_parallel_for` (CPU, read by
  `sslm_decode_step` and `sslm_decode_step_v2`) or `sslm_gpu_context_set_host_parallel_for` (GPU,
  read by `SslmGpuSeqFinishTokenForG5Bridge`). SuperSLM never creates a thread; with no hook
  installed the finish is serial on the calling thread, as before. A `run` that omits, repeats or
  invents a task index fails the call without producing a token -- `SSLM_INVALID_ARGUMENT` on the
  CPU, `SSLM_GPU_PARALLEL_FOR_INCOMPLETE` on the GPU -- and leaves the sequence ready to retry.
  `docs/parallel_for_reference.hpp` is a reference `run` over `std::thread` for hosts with no job
  system. `LogitsSiteParallel` (`forward_sites.h`) is the row-partitioned logits step itself.
- **A device-resident head, GPU backend, opt-in per model.** `GpuResidencyConfig`'s ignored
  `int reserved` becomes `uint32_t flags`, with the same size, offset and alignment, so every
  zero-initializing caller is unaffected. `SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE` uploads the model's
  head table at map time; the finish then computes the exact int64 logits row on the device (new
  shader `logits_site.hlsl`; the shipped shader set gains one `.cso`) and narrows it on the host.
  New VRAM per mapped model is the head table (`vocab_size x hidden_size` bytes) plus two rows,
  each rounded to the 64 KiB allocation granule; with the flag clear a model maps exactly as
  before. Undefined flag bits are refused.

Three statuses are appended to `SslmGpuStatus`, so no existing ordinal moves:
`SSLM_GPU_PARALLEL_FOR_INVALID` (21), `SSLM_GPU_PARALLEL_FOR_INCOMPLETE` (22) and
`SSLM_GPU_RESIDENCY_FLAGS_INVALID` (23). The CPU surface adds no status. `SUPERSLM_API` is carried
by the new C ABI verb and by `LogitsSiteParallel`.

The GPU model handle keeps one host copy of a tied head instead of two: a tied model's head is its
embedding, which the handle already held, so the duplicate head copy (vocab x hidden bytes per
mapped tied model) is no longer taken. An untied model still keeps its `lm_head` copy when mapped
without the device head.

A GPU allocation failure during a model map, an adapter map, a sequence create or a sequence
restore no longer leaves the context's command list recording. Those uploads reset the list before
allocating, so after one allocation failure every later upload on the same context failed too;
they now allocate first, and the next call on the context succeeds. A failed command-list `Close`
in those uploads is retried once. The statuses those calls return are unchanged.

`build.bat` passes a Phase D adapter through: set `T2199_PHASED_ADAPTER` beside
`T2199_PHASED_MODEL`. The Phase D suite fails on any skip, and without an adapter its
adapter-composition cell skipped, which failed the build whenever the suite ran.

CMake gains `SUPERSLM_GPU_TEST_SEAMS` (default `OFF`), which builds a test variant of the GPU
library carrying the allocation-fault and device-logits test seams. It is not for a library that
ships.

## [1.6.0] - 2026-09-22

CPU and GPU callers can now query schema binding explicitly with `sslm_seq_schema_bound` and
`SslmGpuSeqSchemaBoundForG5Bridge`, and GPU callers can query exact current accept-set membership
with `SslmGpuSeqSchemaAcceptingForG5Bridge`. GPU queries are host-only and leave outputs unchanged
on malformed or busy calls. Schema binding on GPU now rejects every sequence with retained
generation history. CPU and GPU schema bind, rebind, and unbind are valid only after create or
reset. A generation call that passes argument validation makes the sequence
ineligible until reset, including a valid no-op call or CPU empty-prefix adoption. A call rejected
for invalid arguments leaves the sequence untouched and still bindable; a GPU busy refusal likewise
preserves eligibility. Every restored sequence must be reset before binding. CPU and GPU reset are
symmetric restart operations: any valid Idle/CPU sequence state can be reset; the shipped
`SSLM_SEQ_RESET_MIDTOKEN_REJECTED` ordinal remains reserved but is no longer returned.

The GPU API gains an embedding read. `sslm_gpu_seq_read_prefill_final_hidden` returns the
post-`final_norm` hidden state at the last position left by a sequence's most recent prompt or
schema-content prefill call that reached its admission pre-scan, when that call returned `SSLM_OK`.
It reads a per-sequence snapshot taken only when such a call succeeds, so no later embed, decode or
finish call changes what it returns. A prefill call that reaches its pre-scan and then fails, and
`sslm_gpu_seq_reset`, empty the snapshot: the read then returns `SSLM_PREFILL_HIDDEN_UNAVAILABLE`,
not an earlier frame. A prefill call refused before its pre-scan (malformed arguments, a zero count,
`SSLM_BUSY`, or the schema-content prefill's unbound-schema and unreachable-first-token refusals)
leaves the snapshot as it was. `sslm_gpu_model_hidden_size`
gives the width to size the buffer with. Two statuses are appended to `SslmGpuStatus`, so no
existing ordinal moves: `SSLM_OUTPUT_BUFFER_TOO_SMALL` and `SSLM_PREFILL_HIDDEN_UNAVAILABLE`.

`SslmGpuSeqPrefillPromptForG5Bridge` now returns `SSLM_SEQUENCE_REJECTED`, not `SSLM_DEVICE_LOST`,
when a device-side domain guard refuses an admitted token and the device is not reported removed.
The context stays usable; reset the sequence before reusing it. The schema-content prefill still
returns `SSLM_DEVICE_LOST` for the same refusal.

The installed GPU library no longer reads `SSLM_B5_ASYNC_DROP_UAV_REBIND` or
`SSLM_B5_ASYNC_SWAP_SRV_REBIND`. Both changed GPU output; they are compiled in only under
`SUPERSLM_GPU_T2106_FAULT_PINS`, which only the fault-pin harness defines.

New header `superslm/api.h` defines an empty `SUPERSLM_API` export slot. It is carried by every C
ABI verb (36) and by the 25 C++ engine declarations that sibling Unreal modules call. Defining it on
the compiler command line (export when building the engine's shared library, import in a consumer)
lets a modular build share one engine copy. By default the macro expands to nothing.

A schema-bound sequence that reaches a completed schema's own dead end now returns `-2` at
`SSLM_OK` on the GPU decode path (`SslmGpuSeqFinishTokenForG5Bridge`,
`SslmGpuSeqDecodeStepForG5Bridge`), matching the CPU path's existing convention, instead of
silently re-emitting token 0 forever. The GPU path leaves `dfa_walk_state` and `layer_index`
exactly as they were and re-arms `ready_for_logits`, so a further call to either GPU entry point
reproduces the identical `-2` result deterministically, with no new embed or layer-loop drive; a
dead-ended GPU sequence is likewise `sslm_gpu_seq_reset`/`sslm_gpu_seq_bind_adapter`-eligible:
`sslm_gpu_seq_reset` does not guard on `layer_index` at all, and `sslm_gpu_seq_bind_adapter` does
guard (refusing the open interval `0 < layer_index < num_hidden_layers` as genuinely mid-token),
but a dead end always leaves `layer_index` at `num_hidden_layers`, full depth -- one of the two
boundary values the guard admits. The CPU path additionally resets `layer_index` to 0 on this
same event, matching an ordinary post-prefill sequence's own resting shape, so a dead-ended CPU
sequence is also `sslm_seq_reset`/`sslm_seq_set_adapter`-eligible rather than only re-decodable.
The CPU backend's own damped-greedy decode branch gains the identical fix, at its own separate
miss site.

The GPU sequence save/restore format moves to `'SLM5'`. A 1.5.0 build cannot read a `'SLM5'` blob a
1.6.0 build saves; a 1.6.0 build still restores an older `'SLM4'` blob exactly as before, defaulting
the schema binding it carries to unbound. The new format adds a schema binding, walk state, and
"ready for logits" flag to the saved blob, so a schema-bound sequence's generation position survives
a save/restore round trip; the restored binding and walk state are validated against the target
model's own schema count and state count before use.

The schema compiler (`tools/sslm_convert_schema.py`) gains an unbounded free-text `"type": "string"`
field. The field's decoded value is the model's own free text up to its first unescaped quote --
measured, not guaranteed free of every grammar-induced cut (an internal quote the model writes with
genuine intent to continue, such as a nested JSON-like structure or a quoted word, can be read as
the value's own close under JSON's own grammar). A caller whose prompt's natural answer may contain
an internal quote should ask for plain, unstructured text, or should treat the returned value as
potentially truncated there. The field also has no length bound the engine enforces, so its close
is otherwise up to the model: a decode that reaches the caller's own token budget before the model
writes a closing quote returns a value that is legal so far but not closed, and the overall output
will not parse -- measured 5 of 40 on a real 40-prompt heldout population at a 300-token budget,
identically on the CPU and GPU decode paths. `sslm_stats_out::schema_accepting` (1 iff the
sequence's current parse state is one where stopping is valid) is how a caller detects this without
guessing from the token count alone. Separately, the value-level close rule means the field cannot
produce an empty string or a value made only of JSON structural punctuation/whitespace (`{}[],:`
and the JSON whitespace set) -- the model's own closing quote is masked at that position, so a
schema whose only truthful answer is `""` or punctuation-only cannot be satisfied by this leaf. A
schema declaring `maxLength` on a string field, or any other
JSON-Schema keyword this compiler does not implement (`minLength`, `pattern`, `format`, `const`,
`multipleOf`, and any keyword not yet named), is rejected at compile time, naming the keyword,
rather than silently compiled with the constraint unenforced. Non-constraining JSON-Schema
annotation keywords (`title`, `description`, `default`, `examples`, `deprecated`, `readOnly`,
`writeOnly`, `$comment`, plus `$schema`/`$id` at the schema root) are accepted and ignored on every
branch, since this compiler never claimed to enforce them; `type` alongside `enum` (the form
`pydantic` and plain JSON-Schema both emit) is accepted too, and is checked for agreement with
every enum value's own JSON type rather than silently ignored -- `type` may itself be a list (the
JSON-Schema/OpenAPI 3.1 form for a nullable or multi-typed field, e.g. `["string", "null"]`), in
which case agreement means matching at least one of the listed types, the union reading the
JSON-Schema spec itself gives a type array.

`compile_schema_to_mask_pages` moved from `Sequence[str]` to `Sequence[bytes]` at the byte-level
port and now raises `TypeError` naming `bytes` when given a `str` vocabulary, instead of the
misleading "no token in the vocabulary can spell the required continuation" diagnostic a `str`
vocabulary produced before: the vocabulary was never insufficient, it was the wrong element type.
`compile_schema_to_mask_pages` also gains a required keyword-only `special_ids` argument that it
applies to its own copy of the vocabulary before compiling (`zero_special_ids`) -- the exclusion of
every tokenizer special/added id from every state of every schema is now structural, at the compile
call itself, rather than a step a caller applies to the vocabulary beforehand and can forget. The
only shipped real-vocabulary producer (`tools/t2132_build_g5_fixture.py::_real_vocab`) previously
left this step to its own caller and no in-repo caller took it, so every tokenizer special id was
admitted as ordinary string content (22 of 22, executed on the real Qwen2.5-0.5B-Instruct
tokenizer); both public entry points a schema compiles through now supply the real special ids.

`sslm_seq_adopt_prefix` no longer leaves a stale walk state on a sequence adopting a prompt-only (or
bound-but-unadvanced) prefix: the adopting sequence's own walk state is reset to its bound schema's
start state (unbound sequences: unused), correcting the one case where the walk had already
advanced past its own start before adoption overwrote the sequence's content. Adoption also now
clears `forced_token_count`, the schema-content forced-span counter visible through `sslm_stats`
(not a damped-greedy statistic), so a warm count from the sequence's own prior generation never
survives into an adopted origin -- including when the adopted prefix's own history holds forced
positions of its own: a prefix does not track its own `forced_token_count`, so the field reads 0
after adoption regardless, exactly as a fresh sequence's does.

See [docs/releases/1.6.0.md](docs/releases/1.6.0.md) for the consumer-facing release note and
compatibility guidance.

## [1.5.0] - 2026-09-17

QK-norm artifact loading now has a complete `QKC1` source-to-derived contract. The loader
requires bit 2 (`0x4`) and the `QkChannelTable` section exactly when the paired QK gains are
present, validates each serialized binary64 source scale and its exact landing/ratio/carried-scale
derivations, and refuses invalid magnitudes before CPU or GPU arithmetic can overflow.

QK conversion uses factored float calibration followed by three compiled capture passes, A/B/C.
It builds a provisional table, then `max(float, A)`, then `max(float, A, B)` before the final
pass-C check; pass C refuses only when clipped direct-K landings exceed one per million
observations. The measured Qwen3-Embedding-0.6B flow at channel-scale headroom 1.25
took 5,226 s end to end; each A/B/C compiled capture pass took about 1,495 s, within
the 30-minute bar.

Fused-K conversion exposes `--channel-scale-headroom` (default: 1.0). Qwen3-Embedding-0.6B
requires `--channel-scale-headroom 1.25`: the default pass C clipped 7,713 of 282,103,808
callbacks (27.341 per million) and raised `ChannelScaleDidNotConverge`, while 1.25 completed
with 7 of 282,103,808 clipped (0.025 per million).

The obsolete fused-K metadata, KLR1 keys, WSC1 gain duplicates, nonlinear rows, telemetry, and
GPU staging slots are retired from the QK path; their reserved GPU slots remain zero. The header's
known-flags mask is `0x7` (Option-G, DGC1, and QKC1), documented and checked against the compiled
contract by the test suite. Existing pre-1.5.0 QK artifacts must be reconverted from their source
checkpoint; no replacement QK artifact is introduced by this release.

Residual adds now combine both operands on the finer of their carried-scale grids and requantize
once, rather than first rounding the branch onto the stream's grid. If the finer-grid row cannot be
built or is refused, the complete row is rebuilt on the coarser grid. The CPU path and both GPU
residual shaders produce byte-identical results.

This changes int8 forward outputs, logits, and carried scales for every model, including models
with no QK path. Outputs are not bit-identical to 1.4.0. Existing
artifacts do not need reconversion for this change: the artifact format is unchanged. A residual
operand with a zero carried-scale mantissa is refused as `SSLM_ARTIFACT_REJECTED`.

Fused-K calibration now stages its candidate artifact and publishes the requested output only after
pass-C convergence succeeds, so a failed convergence cannot leave a final-looking output artifact.
The accepted pass-C clipped/callback rate is configurable with
`--pass-c-clipped-per-callback` (default: one per million).

See [docs/releases/1.5.0.md](docs/releases/1.5.0.md) for the consumer-facing release and
conversion guide.

## [1.4.0] - 2026-09-03

SuperSLM 1.4.0 adds end-to-end Qwen3/QK-norm support to conversion, calibration,
the CPU engine, and the D3D12 GPU path. Legacy GPU models retain their existing
24-dispatch-per-layer budget contract; models carrying QK norm use 25. The GPU
status type is now a scoped C++ enum, public status-returning GPU calls are
`noexcept`, installed CMake targets propagate C++20, and the installed GPU
package exports `superslm_deploy_gpu_shaders()` for runtime deployment.
The test-disabled Windows CMake configuration now supports an ordinary default CPU-only
build without DXC, and the reference safetensors reader enforces complete payload coverage,
the 100 MB header limit, unique JSON keys, and string-only metadata.

QK-norm artifacts made by pre-final 1.4 development trees are not compatible
with the final carried-scale/reciprocal contract and must be reconverted. Released
1.3.1 artifacts are unaffected. GPU callers migrate unqualified status constants
such as `SSLM_OK` to `SslmGpuStatus::SSLM_OK`; the CPU C ABI status names do not
change. See [docs/releases/1.4.0.md](docs/releases/1.4.0.md) for the consumer-facing
release and migration guide.

### Detailed engineering record

Ask 5's Qwen3-architecture support (`SuperSLM_Plan.md` §22.5) is complete in this tree and
**1.4.0 claims the Qwen3-architecture pin**: the converter (Track C), the tokenizer converter
(Track E) and the forward call site with its carried-scale contract (Track B, T-2551 through
T-2570) all land here. 1.3.1 shipped the first two unclaimed; a consumer building against this
tree gets a runnable Qwen3-Embedding candidate on both the CPU and GPU paths. **Breaking for
QK-norm artifacts converted by a pre-1.4.0 tree** — see the format note under Added (D-SLM6200).

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
  The project's own structural-check suite: 423 passed, unchanged. Fix log:
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
  checkpoints present); 1940 passed/4 skipped CI-faithful. The project's own
  structural-check suite: 423 passed, unchanged. The 633 MB product artifact's identity
  is unaffected -- no calibration re-run.
  Close-out log: `Claude/Brunel/t2549-ask5-trackc-close-out-2026-09-02.md` (records
  worktree).

- **The QK-norm forward call site lands, CPU and GPU (Ask 5 Track B, T-2551).** Track C
  converts a checkpoint's `q_norm`/`k_norm` tensors into the artifact; through this round
  no call site read them, so a loaded model ran with the norm's effect absent from every
  head. `LayerWeights` gains `q_norm_gain`/`k_norm_gain` (nullptr when a layer carries
  neither tensor) and `q_norm_site_constant`/`k_norm_site_constant`; a new shared
  function, `ApplyQkNormSite` (`forward_sites.h`/`.cpp`), applies per-head RMSNorm to Q
  (every query head, writing the new carried scale back so attention's own C30
  derivation reads it) and to K (once per KV head, on the just-landed K row; the output
  scale is discarded -- K has no per-token scale analog downstream), strictly after K/V
  landing and strictly before RoPE, in BOTH the single-token path
  (`RunLayerLoopImpl`) and the chunk-batched path (`RunLayerLoopChunkBatched`, the path
  `sslm_prefill` actually calls) -- one implementation, not two reasoned to agree. Gated
  on `!option_g_fused_k_landing`; the combination of QK-norm presence with that flag is a
  defined load-time rejection (below), never a second runtime coordinate system.

  **GPU**: a new shader, `qk_norm_site.hlsl`, dispatched between `kv_proj_site` and
  `rope_guard_site` -- one Dispatch call per layer, `num_attention_heads +
  num_key_value_heads` thread groups (Q heads then K heads), each group's own 256
  threads cooperatively reducing that one head's own `head_dim` elements
  (`RmsSumSqParallelGpu`'s own groupshared tree, scoped per-group by hardware, so the
  multiple groups one Dispatch call issues never interfere). K's own funnel output is
  staged into a disjoint int32 WorkScratch region (K's destination, `KvCache`, is packed
  int8, unlike the funnel's native int32-per-code write) and repacked in the same
  dispatch. `GpuLayerLayout` gains six new per-layer field offsets
  (`q_norm_present`/`q_norm_gain`/`q_norm_site_constant`/`k_norm_present`/
  `k_norm_gain`/`k_norm_site_constant`), serialized into the `Layout` GPU buffer strictly
  AFTER the existing stride slot -- every pre-existing shader's own hardcoded
  `Layout.Load<uint>(56*4)` stride read is unaffected. The pipeline is loaded and the
  dispatch issued only for a model carrying QK norm. Legacy models therefore retain the
  existing 24 dispatches per layer; QK-norm models use 25. The fixed scratch binding layout
  reserves QK staging regions for both shapes so ADAPTER_U and every subsequent binding keep
  one offset contract, but legacy models pay no per-layer QK submission or shader execution.

  **Consumer-visible consequence.** `PlanDispatchBudgetGpu` and batch budget spending are
  model-aware. A legacy `dispatch_budget` of 24 continues to advance exactly one layer;
  QK-norm models require 25. Callers that already derive budgets from the model need no
  migration. Callers introducing QK-norm models must use 25 dispatches per layer for those
  models rather than assuming the legacy constant.

  **Breaking (1.4.0, ruled D-SLM6200):** the carried-scale contract (T-2560/T-2564, above)
  adds a fourth `kv_landing_reciprocals` key per QK-norm layer (`k_normed_head{h}`),
  invalidating every QK-norm-bearing `.sslm` artifact converted between this round (T-2551)
  and T-2560. Nothing shipped is affected — v1.3.1 does not claim Qwen3-architecture
  support, so no released artifact carries `q_norm`/`k_norm` tensors. A pre-T-2560 QK-norm
  artifact is rejected by name at load (`MarshalLayer`'s existing missing-entry diagnostic,
  `"... missing composition_constants entry ..."`, covered by the delta's own §7 Cell 8); no
  compatibility shim reads one at the wrong K scale, which would reproduce C2.

  **Loader-side rejections (`layer_marshal.h`'s `MarshalLayer`), three cells.**
  Asymmetric `q_norm`/`k_norm` presence (one tensor present, the other absent, on the
  same layer) rejects with `"layer{L}: asymmetric q_norm/k_norm presence"` -- a second
  enforcement site alongside the converter's own (Track C, D-SLM6064); this one protects
  an artifact produced by any other tool. `q_norm`/`k_norm` presence combined with
  `option_g_fused_k_landing=true` rejects with a named diagnostic -- built at the loader
  (not the converter): `MarshalLayer` already receives `view.option_g_fused_k_landing`
  on every call, so the check costs a comparison and a diagnostic string with no new
  converter-side validation pass. A `weight_scales` entry with no matching
  `composition_constants` site entry rejects with `"layer{L}: q_norm/k_norm tensor
  present but missing composition_constants site entry"` (reproducing T-2425's own
  "Failure 1" by direct construction: strip a calibrated artifact's own
  `composition_constants` entries post-calibration, pre-conversion). All three fire
  exactly as specified, confirmed by execution (below).

  **The oracle (`tools/reference_pipeline/pipeline.py`'s `_float_layer`) gains a matching
  QK-norm call site**, strictly between projection and RoPE, using the checkpoint's own
  un-permuted `q_norm.weight`/`k_norm.weight` (`_layer_tensors` now fetches them at their
  ENGINE-side lookup key, `q_norm.gain`/`k_norm.gain` -- the key `_upstream_names`, T-2543
  C-1, maps the checkpoint's raw tensor to; a first-drafted `.weight` key raised `KeyError`
  unconditionally and was caught by this round's own direct-execution verification before
  landing, StandardsDocument.md §5.4). Confirmed materially non-degenerate by execution: a
  square-geometry QK-norm-bearing fixture's Q codes differ substantially pre- vs post-norm
  at the SAME run (not merely present-vs-absent at the checkpoint level). This function is
  also the production calibration path's own float trajectory source (`_calibrate` ->
  `_float_forward_many` -> `_float_layer`, once per layer) -- the shared-function decision
  (no fork) means this change re-prices calibration for every QK-norm-bearing checkpoint,
  including the pinned Qwen3-Embedding-0.6B candidate: **the T-2543/T-2549 artifact's own
  staleness note above is now the live state** -- its cached scales were derived from a
  Qwen2.5-shaped (no-QK-norm) float trajectory over an architecturally Qwen3 checkpoint,
  and must be discarded and recalibrated, not reused, before any further Ask 5 work reads
  a calibration-derived number for that candidate. That recalibration (~73 minutes against
  the real checkpoint, T-2423's own precedent) is NOT executed by this round -- named as
  owed, not run.

  **Blast radius outside this round's own writable scope, filed rather than fixed
  (`Claude/Zelda/Board.md`).** `_kv_calibration_capture` (Arm D/E's own per-head
  calibration capture) independently reimplements `_float_layer`'s own norm/project/RoPE
  composition by calling the same primitives directly, and its own docstring claims exact
  parity with `_float_layer` -- a claim now false for QK-norm specifically, since that
  function gained no matching call site here (the design's own Track B step 6 names only
  `_float_layer`). Two tests asserting that parity, plus one pinned golden logit in a
  third file, now fail against the corrected oracle: `test_armd_arme_kv_calibration.py::
  test_arm_d_q_scale_matches_shipped_production_q_path_at_every_layer`,
  `test_t1937_production_fix_pins.py::test_arm_d_q_scale_matches_the_shipped_production_
  q_path`, `test_dynamic_bias.py::test_oversized_bias_magnitudes_reject_at_the_coarse_
  scale_edge`. All three are outside this ticket's own writable scope
  (`tools/reference_pipeline/tests/*.py` was not granted; `pipeline.py`'s own grant was
  scoped to `_float_layer` alone) and outside its own exit-condition suites (the structural-check suite,
  `superslm_tests`); `python -m pytest tools/reference_pipeline/tests -q`: 4 failed, 1683
  passed, 13 deselected (was 1687 passed before this round -- the four now-failing cells
  were silently passing only because a first-drafted `.weight` lookup key made this
  round's own oracle change a no-op, corrected above).

  **Acceptance for Track B alone, executed** (`tests/t2551_qk_norm_harness.cpp`, a
  committed acceptance harness in the style of `tools/t2432_geometry_harness.cpp`; a
  square-geometry, QK-norm-bearing fixture from `_calibrate_checkpoint_fixture.
  build_parameterized_fixture_checkpoint(qk_norm=True)`, calibrated and converted through
  the real, unmodified pipeline): CPU and GPU agree bit-for-bit end to end
  (`hidden_codes`, `hidden_scale`, every K/V row) on both the single-token and the
  chunk-batched-exercising path -- the determinism crown, unaffected by the tolerance
  question below. The oracle comparison is executed and its reading recorded, QUARANTINED
  per this round's own brief: the tolerance composed-acceptance item 3 already owes to the
  deriver does not exist yet, so this cell is never headlined pass/fail. Recorded reading,
  single token (`good.sslm`, token id 3): engine `scale_m=1802723290 scale_e=-42
  codes=-127,-99,-4,-24,0,12,105,-49`; oracle (float64, post-layer0, same token)
  `[-0.05203353550671877, -0.04088508394718841, -0.0018355512033354064,
  -0.010044067807571874, 0.0002363362229337695, 0.005076312832271062,
  0.04273748902866161, -0.019773453409268643]`. The red state (`main`@`ecadbb6`, no
  QK-norm call site anywhere) is reproduced and quoted in the build log; at this fixture's
  own tiny scale (`hidden_size=8`, single KV head) the red and green CPU runs converge on
  the identical final hidden state -- traced to attention degeneracy at this scale (a
  single KV head's context vector is a weighted average of V, which QK-norm never touches,
  and the weighting itself washes out at this size, the same class T-2432's own build log
  already documents for a different site) and NOT evidence the call site has no effect:
  the direct pre-/post-norm Q comparison above shows a large, unambiguous difference
  before it reaches attention.

  Full C++ suite: **34228 checks, 0 failures** (`superslm_tests.exe`, was 34226/0 before
  this round -- 2 new checks). FP-free scan gate: **PASS** on `build/Release/superslm.lib`
  (17 objects, 2100 symbols ACCEPT, 0 REJECT). the structural-check suite: **423 passed**
  (unchanged in count from T-2549's own report; two of those cells' own line-number
  citations were re-derived against this round's own source shift, named above).
  `tools/reference_pipeline/tests`: 1683 passed / 4 failed (named above) / 13 deselected.
  GPU leg executed on this machine's RTX 2080 SUPER (8 GiB) -- confirmed working device,
  every dispatch above real, not simulated.

  Build log: `Claude/Brunel/t2551-ask5-trackb-build-2026-09-02.md` (records worktree).

- **The oracle's QK-norm call site reaches every independent forward walk in the
  reference pipeline, and the pinned candidate is recalibrated on it (Ask 5 Track B
  close-out, T-2553).** T-2551 gave `_float_layer` a QK-norm call site and reported three
  sibling tests failing as an out-of-scope finding. It was in scope: an oracle with
  multiple forward implementations of which only one applies a real architectural
  operation is the sibling-pinning defect `StandardsDocument.md` §7 names. Every
  independent per-layer forward walk in `tools/reference_pipeline/` now applies the
  identical QK-norm composition, from one shared function (`_apply_qk_norm`) for
  `_float_layer`/`_kv_calibration_capture`, and a matching call site in each of
  `_vec_forward` (the numpy integer parity shadow), `_scalar_forward` (the normative
  scalar reference §6.2 holds every specialization to), `forward_dynamic` (the W8A8-
  dynamic full-stack forward, §15's measured arm), and `composition_ref.py`'s own
  independent big-int oracle (`forward_dynamic_logits_oracle`) -- six implementations,
  one composition. `_derive_scales` gains q_norm/k_norm's own rescale-site derivation,
  re-derived by direct execution (a first-drafted formula, copied unchecked from the
  C++-engine-facing `composition_constants` convention, underflowed to a degenerate zero
  requantizer here -- caught by execution, not by inspection); Q's own carried scale is
  reassigned post-norm downstream, K's is deliberately left unchanged (verified by
  execution to match the real engine's own accepted approximation -- "K has no analog of
  Q's own q_scale").

  **The tree went red before it went green, honestly.** Fixing the shared composition
  surfaced two more real findings, both corrected rather than routed around: a stale
  pinned golden value and a stale Arm D sweep witness/key that no longer held once RMSNorm
  bounded the fixture's own per-head K distribution (re-measured and re-derived, not
  deleted), and one integer-vs-float structural-tolerance test widened (0.25 -> 0.6,
  documented, re-measured layer-by-layer) to accommodate QK-norm's own real, bounded,
  two-layer-compounding divergence at this fixture's toy scale -- still far below what a
  genuinely structural bug (a transposed head, a dropped residual) would produce, which is
  that cell's own stated job. A new pinned-fixture test
  (`test_ask5_trackb_oracle_qk_norm_parity.py`) proves materiality on the two
  `pipeline.py`-native siblings directly. Full sweep: `pytest tools/ tests/reference/ -m
  "not upstream"` -- **1992 passed, 13 deselected, 0 failed**; the structural-check suite -- **423
  passed**.

  **The pinned Qwen3-Embedding-0.6B candidate is recalibrated on the norm-applying
  forward**, discharging the staleness note T-2543/T-2551 both carried: `calibrate_
  checkpoint.py` then `convert_model.py` (with `sslm_verify` invoked, not skipped) ran end
  to end against the real checkpoint (revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`)
  -- **`verified: independent loader accepted the artifact`**, 9 sections, unchanged size
  **633,576,276 bytes**, new SHA-256
  **`7fd5d3981f34c572b729f515ae4a8307780cabd8ab618fe19c7edb6089ae3476`** (supersedes the
  T-2543-era `5cf871fb...` and the never-shipped T-2551-era intermediate). **This is a
  LOAD-TIME claim only, same as its predecessor** -- proven to convert and load, not yet
  proven numerically correct against a derived tolerance (below).

  **Track B's own acceptance, re-executed against this real artifact** (`tests/
  t2551_qk_norm_harness.cpp`, unmodified from T-2551): marshal OK on all 28/28 layers;
  QK-norm's own call site fires on both the single-token and the chunk-batched-exercising
  path, with materiality confirmed directly (pre-/post-norm Q codes differ substantially);
  **CPU and GPU agree bit-for-bit end to end** -- `hidden_codes[1024]`, `hidden_scale`,
  every K/V row, across all 28 layers -- on this machine's RTX 2080 SUPER, at both the
  single-token position and the 3-token chunk-batched span (host: this machine's CPU;
  GPU: RTX 2080 SUPER, 8 GiB; positions: single-token position 0, and chunk-batched
  positions 0-2). The oracle comparison is executed and its reading recorded, QUARANTINED
  per this round's own brief -- composed-acceptance item 3's own tolerance (D-SLM5276)
  still does not exist. Recorded reading, single token (this artifact, token id 3, first
  16 of 1024 dims): oracle (float64, post-layer0)
  `[0.19159927, -0.32733400, -0.00527838, -0.94976745, 0.19475260, -0.00450565,
  -0.13766608, -0.78359323, 0.81294516, -0.36960737, 0.25995328, 0.00502180,
  -0.17362168, 0.02245454, -0.00648451, -0.12685488]`; engine (CPU, single-token,
  same token, `scale_m=1083582878 scale_e=-28`, first 16 codes)
  `13,-38,13,-38,25,0,13,25,13,-13,-13,-13,13,-38,-102,25`. No dequantization convention
  is applied and no delta is computed here -- doing so would imply a judgment this round
  is not authorized to make. Both readings are the record; the comparison is slice 3's.

  Build log: `Claude/Brunel/t2553-ask5-trackb-oracle-sibling-2026-09-02.md` (records
  worktree).

- **Track B's carried-scale contract for the per-head QK-norm (T-2560).** The
  `f1a2741`-era build resolved the norm's own output carried scale by construction, not by
  design, and both resolutions were numeric-correctness defects (review
  `Claude/Poirot/f1a2741-t2552-ask5-trackb-review.md`, DO-NOT-SHIP): Q kept one shared
  scale (the last head's), overwriting every other head's own genuinely distinct value
  (C1, measured 3.854x on the real candidate); K's post-norm codes were stored at the
  norm's own dynamic scale and read back through the artifact's static, pre-norm landing
  constant (C2, measured 79.2x/39.6x). The design delta
  (`Claude/Vitruvius/t2557-trackb-carried-scale-delta-2026-09-02.md`) states the contract
  this round builds: **Q** carries a genuinely distinct scale per query head into
  attention's C30 derivation, re-derived per query head rather than memoized per KV head
  (`ApplyQkNormSite`'s Q parameter widens from one `CarriedScale*` to a `num_heads`-wide
  array, broadcast from the pre-norm `q_proj` scale for a layer without `q_norm`,
  overwritten per head when present). **K** requantizes its post-norm codes a SECOND time,
  through the identical `LandingRescale` primitive K's own raw pre-norm landing already
  uses, onto a NEW static per-(layer, KV head) landing scale calibrated on post-norm data
  (`LayerWeights` gains `k_norm_landing_r_t`/`e_t`); `softmax_khead` is recomputed from
  that new scale when `k_norm` is present, closing C2 structurally (writer and reader now
  agree on the same scale). **CORRECTED 2026-09-03 (T-2572, D-SLM6263, external review
  `Claude/External/superslm-1p4p0-2026-09-02.md` Significant 1):** as this entry originally
  shipped, "calibrated on post-norm data" meant the K landing scale's own calibration
  observed only the post-norm, PRE-RoPE component maxima — the false premise being that
  RoPE, a magnitude-preserving rotation, needs no scale of its own. RoPE preserves a
  rotated pair's L2 norm, not the component-wise maximum an int8 scale is chosen from, so
  a rotated component can reach up to sqrt(2) times the pre-rotation maximum; the engine
  requantizes K onto this scale BEFORE RoPE and clamps AFTER, so a pre-RoPE-only
  calibration could silently clip the store the engine actually produces (confirmed on the
  repository's own checked-in calibration fixture: 127.17 codes required against a store
  sized to 127). The K landing scale is now calibrated on the UNION of post-norm/pre-RoPE
  and post-norm/post-RoPE component maxima, closing C2 as this entry's own claim always
  said it did — see T-2572's own entry, below, for the full account and the recalibrated
  candidate's new identity. **The GPU** gives every query head its own 16-byte scratch
  slot (`ScratchLayout` index 3 widens from one slot to `num_attention_heads * 16` bytes),
  closing C3 (the race `num_attention_heads` concurrent thread groups had on one shared
  slot) by construction — no two groups ever write the same address.

  **Three items the paired rung routed here, closed in the same round:** (1) **S7** — every
  gain tensor the forward reads unconditionally (`q_norm.gain`/`k_norm.gain` against
  `head_dim`, and the pre-existing `attn_norm.gain`/`mlp_norm.gain` gap the review's own S7
  finding named) is now length-checked at `MarshalLayer`, rejecting a short tensor by name
  before `WidenGainToInt32` ever reads it out of bounds. (2) **S8** — five items T-2553
  re-pinned against the pre-fix tree (the golden logit, the ARMD sweep key, the structural
  0.6 bound, the retired §31.4.4 row 5 property, and the retired ±2^52 coarse-bound rung),
  three of which are RE-MEASURED this round against the corrected composition: the golden
  logit and the ARMD key hold UNCHANGED (golden logit −2350; ARMD key
  `layer1.k_head0`); the 0.6 bound's reading is also re-executed and also UNCHANGED (layer
  0/1 structural ratios 0.128775/0.554224, matching the pre-fix reading to six decimal
  digits) — **but "unchanged" here does not mean "healthy": see the T-2564 fix round
  below, where the same reading is shown to be absorbing a residual composition defect,
  not ordinary quantization noise, and the bound is corrected.** The two retirements are
  NOT re-measured this round — T-2564, below, re-measures both. The composed-acceptance
  tolerance (D-SLM5912)
  stays held pending its own derivation (D-SLM6123) — this round does not derive it. (3) The
  pinned Qwen3-Embedding-0.6B candidate is recalibrated on the corrected forward and
  reconverted (verified, `sslm_verify` invoked): **633,588,308 bytes**, new SHA-256
  **`09c439f69d40e058d574a30f45f8b104772117e8ad7e453ed895ed9635100b39`**, superseding the
  T-2553-era `7fd5d398...` this branch's own prior entry recorded.

  **Every cell in the delta's own §7, red-then-green, its construction quoted.** Cell 1
  (width>1 acceptance, must-reject = identical with/without QK-norm): PASS on the real
  28-layer candidate at width=3 — DIFFERS, unlike C4's own width==1 finding. Cell 2
  (collapsed-Q must-reject, `forward_dynamic`'s own mutant): diverges from
  the correct per-head build, max |logit diff| 2343 at this fixture. Cell 3 (pre-norm
  softmax_khead must-reject): diverges from the float-grounded oracle, max |logit diff|
  3557. Cell 4 (GPU determinism, repeated dispatch, N=100, width>1, real candidate): 0/100
  divergences against the first GPU run and against the CPU chunk-batched reference — the
  must-reject mutant (a second `qk_norm_site.hlsl` reverting Q's write to the pre-fix
  shared slot) was built and executed in a follow-up round this same session (a scratch
  copy of the whole worktree with the shader reverted there, the only route available
  since `RunLayerLoopGpuSubmit`'s own dispatch table has no injection point for a second
  variant without a further production-code change): **186 divergences** across the same
  100 repeated dispatches, confirming the per-head addressing fix's own closure of the
  race by an executed regression-catching proof, not by construction alone. Cell 5
  (asymmetric/Option-G rejections): untouched, re-asserted. Cell 6 (S5's materiality
  check, repaired to compare matching-width captures, and its own must-reject via the
  gain-bypassed construction): both executed, PASS. Cell 7 (S6, strengthened to a
  per-caller source-inspection assertion, with both single-caller-deletion must-reject
  mutants executed): PASS. Cell 8
  (the fourth `composition_constants` key's own rejection gate): must-accept is this
  round's own recalibrated artifact loading clean; must-reject is the PRE-EXISTING
  `f1a2741`-era artifact (`out/fixtures/good.sslm`), rejected by name
  (`missing kv_landing_reciprocals entry "layer0.k_normed_head0"`) — a real artifact
  calibrated before this delta landed, not a contrived input. Cell 9 (gain-tensor length
  validation): both `q_norm.gain` and `attn_norm.gain`, one element short, rejected by name
  with the exact expected/actual element counts. Cell 10 (the carried-scale composition at
  `group=1`): the fixed build passes both Cell 2's and Cell 3's own must-reject
  constructions, re-run at this geometry.

  **CPU and GPU agree bit-for-bit on all 28 layers of the real candidate at the acceptance
  width** — the single-token determinism crown (width=1) PASSes, and Cell 4 drives the GPU
  as 100 sequential single-token submits (the GPU has no chunk-batched dispatch function)
  compared against a CPU chunk-batched reference, both on this machine's RTX 2080 SUPER.

  Suites, on the fixed tree: `superslm_tests.exe` → **34228 checks, 0 failures**; `ctest`
  (build/, Release) → **13/13 passed**; `pytest tools/ tests/reference/ -m "not upstream"`
  → **1999 passed, 13 deselected, 0 failed**; the structural-check suite → **423 passed**.

  Build log: `Claude/Brunel/t2560-trackb-carried-scale-rebuild-2026-09-02.md` (records
  worktree).

  **T-2564 fix round (confirmation review FIX-THEN-SHIP,
  `Claude/Poirot/36185a3-t2563-trackb-rebuild-review.md`).** One Critical, four
  Significants, and record-only findings, closed. **C1** — `softmax.input`'s own static
  half in the reference pipeline paired the post-norm Q scale with the PRE-norm K scale on
  every QK-norm-bearing layer (`_derive_scales`, `tools/reference_pipeline/pipeline.py`),
  C2's exact composition, in the one arm C2's own T-2560 fix did not reach; the 0.6 bound
  above was widened to absorb it. Fixed by the identical substitution `_derive_scales`
  already makes for Q: the K half now reads `gain_of(k_norm.gain)` when `k_norm` is
  present. Re-measured on the corrected tree: layer 0 unchanged at 0.128775, layer 1 drops
  from 0.554224 to **0.125422** — back under the ORIGINAL 0.25 bound, which is restored
  (`test_pipeline.py`). This is a reference-pipeline-only fix; it does not touch the
  compiled engine, the artifact format, or the shipped candidate. **S2** — the GPU's new K
  re-landing (`qk_norm_site.hlsl`) read a scale word written by thread 0 alone, from every
  thread of the group, with no memory barrier between the store and the read (masked on
  this machine's RTX 2080 SUPER, not absent — probe-commissioned: a poison control forced
  behind a barrier reproduces 100/100 divergences, proving the detection apparatus fires;
  the reviewer's own construction, poison planted before the funnel with no barrier of its
  own, reproduces the reviewer's 0/100 reading before AND after this fix, since the race
  was never observed on this GPU/driver either way). Fixed with one
  `AllMemoryBarrierWithGroupSync()` before the read. **S3** — the second K landing computed
  its own clamp signal and discarded it (`ApplyQkNormSite`'s CPU K loop passed no
  saturation counter; the GPU shader's own `would_clamp` was read by nothing), so
  `SslmDecodeStepStatus::saturation_count` under-reported (measured: 18 of 200,704 elements
  clamped on the real candidate, invisible to the counter). Fixed: `ApplyQkNormSite` gains
  an `out_saturation_count` parameter threaded to both callers' own `kv_saturation_count`;
  the GPU shader counts `would_clamp` into a new per-group `gQkNormClamps`, flushed to the
  same `SeqState` saturation slot `kv_proj_site.hlsl` already writes via `InterlockedAdd`
  (this dispatch issues one thread group PER KV head, unlike `kv_proj_site.hlsl`'s single
  group, so the flush needs to be atomic across groups, not merely within one). A new cell,
  `TestT2564_S3_ApplyQkNormSiteSecondKLandingCountsSaturation` (`tests/test_main.cpp`),
  forces a clamp on both elements of a synthetic K row and asserts the count rises; verified
  RED (hand-applied and reverted) when the CPU call's own out-parameter is dropped back to
  `nullptr`. **S4** — no
  automated cell could fail on a regression of C1 or C2 in the shipped engine (the delta's
  own §7 Cells 2/3/10 pin the REFERENCE implementations; the cells that reach the compiled
  engine live in `tests/t2551_qk_norm_harness.cpp`, outside CMake and outside every hosted job). Ruled
  (D-SLM6199): two new cells, `TestT2564_S4_ApplyQkNormSitePerHeadQScaleNotCollapsed` and
  `TestT2564_S4_ApplyQkNormSiteKLandsOnNewPostNormScaleNotJustNorm`
  (`tests/test_main.cpp`), call `ApplyQkNormSite` directly against small synthetic
  fixtures — no artifact load, no `RunLayerLoop` drive — and are part of `superslm_tests`,
  already a registered CTest target that runs in hosted CI. Verified RED on both mutants
  (Q's per-head carry collapsed onto the last head; the second K `LandingRescale` loop
  deleted), hand-applied and reverted this session; green on the tip. The real-candidate
  acceptance (the 633 MB artifact, width>1, 28 layers, the N=100 GPU drive) stays a
  **manual gate**, never a per-push hosted leg (D-SLM6161): documented in
  `tests/manual/README.md`, run at every pin bump and before every release tag. **S5** —
  the CHANGELOG text above claimed all five S8 constants were re-measured; the two
  retirements were not, and neither is re-measured here either — both are disposed by
  reading and by inspection, not by a fresh execution of the retired property itself. The
  ±2^52 coarse-bound rung is re-confirmed by reading the same construction (the
  magnitudes-48–59 execution is the prior reviewer's own, not re-run a second time here —
  the chain-domain edge catches first at 2^55, `oracle_50[0][0] == -2350` holds by that
  reading). The §31.4.4 row 5 property's own inputs (`_kv_calibration_capture`,
  `_layer_q_scale`, `_armd_arme_candidate_sweep`) are untouched by both this round's diff
  and T-2560's — unchanged by inspection, confirmed by a green run of
  `test_arm_d_saturation_is_reported_and_does_not_gate_selection`, which asserts the
  property that SURVIVED the retirement, not the retired one itself.
  **M8** — `k_norm_landing_r_t`/`e_t` were dereferenced with no null guard
  of their own (gated only by the outer `k_norm_gain != nullptr`, unlike the sibling
  `iexp_softmax_khead_m`/`_e` three lines above in the same GPU packing loop); the header
  comment now states the obligation in both directions, and the GPU marshal site at the
  time used the identical per-element ternary guard its sibling already did. (Both sites'
  own ternary substitution was itself later found unsafe and replaced with a named
  refusal that throws instead of substituting zero: T-2566 for this field, T-2568 for its
  sibling.) **M9** —
  `k_normed_scale`
  was derived twice from the same inputs, by `_derive_scales` and by
  `_derive_composition_constants` independently; the latter now reads the former's own
  `StaticScales` entry (`scales.scale(f"{prefix}.k_normed_head0.scale")`) instead of
  re-deriving it. The CHANGELOG
  misattributions above (Cell 2's mutant, Cell 4's must-reject status, Cell 10's own
  re-run list, the chunk-batched-drive phrasing) are corrected in place, above.

  Suites, on the fixed tree: `superslm_tests.exe` → **34236 checks, 0 failures** (34228 +
  8 from the two new S4 cells and the one new S3 cell); `ctest` (build/, Release) →
  **13/13 passed**;
  `pytest tools/ tests/reference/ -m "not upstream"` → **1999 passed, 13 deselected, 0
  failed** (unchanged — C1 and M9 are reference-pipeline-internal); the structural-check suite →
  **423 passed**, after a scrub: `test_check_gpu_guard_status_parity.py` and
  `test_geometry_site_census.py` both correctly reddened (10 failures) on line-citation
  drift `ApplyQkNormSite`'s own new `out_saturation_count` parameter caused in
  `include/superslm/gpu_layer_loop_guards.def` (all nine citations, +2 lines, uniformly)
  and one `test_geometry_site_census.py` self-test's own hardcoded GS-12 citation
  (:1732 → :1734) — the same discipline working as designed T-2560's own build record
  already exercised, not a defect this round introduced silently. Each re-derived directly
  against the real, current, mutated file (never by applying a line-count offset) and
  corrected in place, dated. The real-candidate acceptance harness, rebuilt from the
  fixed tree and re-run against the UNCHANGED artifact (`out/t2560_qwen3-embedding-0.6b.sslm`,
  633,588,308 bytes, SHA-256
  `09c439f69d40e058d574a30f45f8b104772117e8ad7e453ed895ed9635100b39` — no recalibration was
  needed, since none of this round's fixes changes the artifact format or its composition
  constants): `CELL 1 ... PASS`, `CELL 4 ... 0/100 divergences ... PASS`,
  `DETERMINISM CROWN: PASS`, `RESULT: PASS`.

  Build log: `Claude/Brunel/t2564-trackb-fixes-2026-09-02.md` (records worktree).

- **New `SslmForwardStatus` member `GpuLayerWeightsContractViolation`, appended last, no
  existing value moved (T-2568).** Returned by `RunLayerLoopGpu`'s public entry point when
  a layer's `LayerWeights` violates a `PackLayerWeightsBytes`-enforced non-null pointer
  contract (below) -- distinct from `GpuAllocationFailed`, which this file's own
  transient/permanent taxonomy reserves for device and allocation failures a retry at a
  smaller size can fix; no retry at any size fixes a null pointer. `SslmForwardStatusName`
  gained the matching arm.

- **RoPE's own post-rotation clamp is counted, CPU and GPU, per site, and the K landing
  scale calibrates on the union of pre-RoPE and post-RoPE component maxima (T-2572/T-2577,
  D-SLM6263/D-SLM6274, closing the external review's Significant 1,
  `Claude/External/superslm-1p4p0-2026-09-02.md`).** `_float_layer`'s own `k_normed`
  observation fires TWICE per layer under the identical running-max key -- once
  post-norm/pre-RoPE and once post-norm/post-RoPE -- so the K landing scale (`k_norm_landing_r_t`/
  `e_t`) covers the true post-RoPE peak: the engine requantizes K onto this scale BEFORE
  RoPE and clamps AFTER, so a pre-RoPE-only calibration could silently overflow the store
  the engine actually produces (127.17 codes required against a store sized to 127, on the
  repository's own checked-in fixture, before this fix). `RopeApplySite` and its GPU
  siblings (`rope_guard_site.hlsl`, `rope_commit_site.hlsl`) each carry the SAME
  `[-127, 127]` clamp counter every other landing site already fed into
  `SslmDecodeStepStatus::saturation_count` -- a clamp at this site was previously silent.

  **The counter now also carries a per-site breakdown** (`kv_landing`, `k_normed_landing`,
  `rope_q`, `rope_k` -- `SequenceLayerState`'s own new fields, T-2577) alongside the
  aggregate, on both the CPU and GPU paths, GPU-equals-CPU on the real candidate. The
  aggregate is dominated by Q's rotation on this candidate (69 of 71 clamps at token 1) --
  a consumer reading only the aggregate to decide whether K's own landing is clipping was
  reading a number 97% attributable to a term its own name never named; the per-site
  breakdown gives that consumer the K-specific reading directly. **K's own count is per KV
  head, not per query head** (T-2577, D-SLM6281): every query head sharing one KV head
  redundantly re-rotates the identical row (harmless to repeat -- the write-back was
  already documented as "redundant but sound") and, before this fix, each redundant call
  also counted its own clamp, inflating K's count by the head-group ratio (2x on the real
  candidate); both CPU call sites and `rope_commit_site.hlsl` now count only on the first
  query head of each KV head's own group. On the recalibrated candidate, `rope_k` reads
  zero at every width measured, including width > 1 -- the enclosure the union calibration
  claims, watched directly rather than inferred from the dominated aggregate; O2's own
  zero-margin observation (the dominating head sits at exactly 127.000 float-domain codes)
  is therefore inherent to int8 at that head, not evidence of an overflow the engine has
  actually produced.

  **`calibrate_kv_landing_arm`'s `"per_head"` arms (C, D, E) refuse a QK-norm checkpoint by
  name** (`CalibrationArmDoesNotSupportQkNorm`), closing the external review's Minor 2:
  those arms' own per-head policy schema still labels a post-norm capture under raw-K keys
  and emits no `k_normed_head{h}` entry the 1.4.0 loader requires on a QK-norm layer; the
  `"per_layer"` arms (A, B) are unaffected, since they reuse the production path this same
  entry's own union fix already makes QK-norm-correct. The per-head policy schema's own
  QK-norm support, and a calibration arm proven against the shipping (QK-norm-bearing)
  configuration specifically, are both owed to a later ticket -- Arms C/D/E's own refusal
  is what stands in for that proof today, and is what the ticket's own must-reject cells
  exercise.

  **A shader binary older than the source it was compiled from is refused, not dispatched**
  (`superslm_gpu::harness::ShaderPath`, the single funnel every `.cso` load in this tree
  passes through): a `.cso` must be at least as new as its own `.hlsl` and as the newest
  shared `.hlsli` in the same directory. The refusal now surfaces to a caller as its own
  named status, `GpuShaderBinaryStale` (T-2577, D-SLM6279) -- it used to fall through a
  generic `catch (const std::runtime_error&)` into `GpuAllocationFailed`, whose documented
  remedy ("retry smaller") is actively wrong for a shader no retry at any size fixes; a
  caller obeying that remedy would retry forever with the one real diagnostic surviving
  only on stderr. Absent sources (an installed consumer, which ships no
  `src/gpu/shaders`) and an absent binary are unverifiable rather than stale, and are not
  refused. Six cells commission the guard: a must-accept, two constructed must-reject arms
  (a binary behind its own `.hlsl`; a binary behind a shared `.hlsli`), the two
  unverifiable cases, and a must-reject on the real load path that back-dates an actual
  `.cso` a decade and requires `ShaderPath` to refuse; a seventh drives a stale binary
  through a real forward call and asserts the named status a consumer actually sees, not
  the guard's own API directly. The acceptance harness's own header carries its build
  recipe with the dxc step in it -- compiling only the `.cpp` leaves a stale shader in
  place and the harness dispatches it silently.

  **A RoPE cos/sin table cached against a still-live model is no longer forced to repack
  and re-upload on every fresh sequence, and a table cached against a recycled host
  address is still never served to the wrong model** (T-2577, D-SLM6278).
  `g_resident_rope` (`src/gpu/superslm_gpu.cpp`) keeps the two model-wide rotation tables
  device-resident across a decode session, keyed on the source tensor's own host address
  and byte count -- a key identical across every sequence of ONE model. A prior fix gated
  the fast path on `!fresh_sequence` to close a recycled-address hazard (an address the
  allocator hands back can name a different model's tables), but that term forced every
  fresh sequence of a STILL-LIVE model to pay a full repack and re-upload it never needed:
  measured on the real candidate, 102 of 103 fresh-sequence first calls matched the cache
  key and each paid ~19 ms of host work the prior text claimed was free everywhere it is
  not (`StandardsDocument.md` §5.4 -- true of the 1.0 handle path, which never reaches this
  cache, false of every pre-1.0 caller, which is every caller that does). The cache (and
  its two siblings, `g_resident_weights`/`g_resident_kv`, which shared the identical
  `!fresh_sequence` shape for the identical reason) now accepts an optional caller-supplied
  model identity (`model_generation`, a new trailing parameter on `RunLayerLoopGpu`/
  `RunLayerLoopGpuSubmit`, default `0`): a fresh sequence whose identity matches what is
  cached is a legitimate hit, and a fresh sequence whose identity does not match -- the
  recycled-address case, unchanged -- still misses. A caller that supplies no identity
  observes exactly the prior `!fresh_sequence`-gated behavior, byte-for-byte.

  **The pinned Qwen3-Embedding-0.6B candidate's identity is unaffected by this round**
  (T-2577 touches no calibration, no shader source's forward math, no CPU forward, and no
  `gpu_port.h` field layout): **633,588,308 bytes**, SHA-256
  **`e5212dd2d4772bb6939c3c75dae6f9bf933a0d48de57d1f26f929a1c1e506759`**, unchanged from
  T-2572's own recalibration. Real-candidate acceptance, re-run on the unchanged candidate:
  `CELL 1 (width>1 acceptance) ... PASS`, `DETERMINISM CROWN: PASS`, `CELL 4 (GPU
  determinism, N=100, codes/scale and kv_saturation_count) ... 0/100 ... PASS`, `CELL 5
  (RoPE-table residency, production geometry) ... PASS`, the new per-site GPU-equals-CPU
  comparison PASS, and `rope_k == 0` at width > 1. Obligation constants (D-SLM6148) hold at
  their T-2572-era readings, unaffected by this round's own diff: the structural bound
  (`test_pipeline.py`) `<= 0.25` (layer 0 = 0.1288, layer 1 = 0.1254); the golden logit
  `oracle_50[0][0] == -2350`; the ARMD sweep key `layer0.k_head1`.

  Build/solve records: `Claude/Brunel/t2572-k-normed-rope-peak-2026-09-03.md`,
  `Claude/Laplace/t2575-gpu-sat-count-2026-09-03.md`,
  `Claude/Laplace/t2576-rope-commit-writeback-2026-09-03.md`,
  `Claude/Brunel/t2577-external-fold-fixes-2026-09-03.md` (records worktree). The
  measurement-apparatus history behind this entry -- a stale local shader binary that made
  a working RoPE counter read as broken across two rounds, then a cache key that read as a
  write-back defect until a device read-back located it one cache over -- is preserved in
  those records and in git history, not narrated here: `StandardsDocument.md` §6.6 wants
  HEAD to hold current truth, and none of that apparatus history describes the shipped
  engine, which never carried the defect it appeared to (D-SLM6282).

### Changed

- **`PackLayerWeightsBytes` (`src/gpu/superslm_gpu.cpp`) now refuses two previously
  silently-substituted null-pointer contracts by throwing, instead of packing a
  substituted zero (T-2566, T-2568).** `k_norm_landing_r_t`/`e_t` (required non-null
  whenever the paired `k_norm_gain` is present) and `iexp_softmax_khead_m`/`_e` (required
  non-null unconditionally, every layer) each now throw `GpuLayerWeightsContractError` --
  a caller violating either contract is a permanent bug, never a transient/size-dependent
  one. A caller reaching either boundary now gets `GpuLayerWeightsContractViolation`
  (above) at both public GPU entry points (`RunLayerLoopGpu` and `sslm_gpu_model_map`),
  with the violating field's name printed to stderr, rather than a silently zero-landed K
  code or an unnamed status discarding the message. Unreachable through any artifact
  `MarshalLayer` (the only in-tree producer of the pointers both refusals guard) can emit
  today -- both fields are already rejected upstream when absent -- but the refusal is a
  public-surface contract change on an installed header (`include/superslm/gpu_port.h`),
  independent of today's reachability.
### Fixed

- **`tools/convert_tokenizer.py` no longer crashes on the Qwen3-Embedding-0.6B candidate's
  `tokenizer.json`.** `TokenizerTables.__init__` raised `AttributeError: 'list' object has no
  attribute 'split'` on this checkpoint's own `model.merges` schema (a 2-element list per entry,
  where every prior checkpoint this converter has emitted for carried a space-separated string).
  `_parse_merge_element` now branches on the element's own schema, using a 2-element list/tuple
  directly and rejecting any other shape by name; every prior checkpoint's own string schema is
  parsed unchanged. **Additive: `TokenizerTables` now also reads the top-level `post_processor`
  key and exposes `trailing_special_id`** -- `None` for every checkpoint whose post-processor is
  absent or a bare `ByteLevel` (every checkpoint this converter previously supported), and the
  candidate's own appended token id (`151643`) for a `Sequence` whose own `processors` are EXACTLY
  `[ByteLevel, TemplateProcessing]` in that order, the candidate's own shape; every other shape --
  including a `Sequence` carrying a third, sibling processor that also inserts tokens -- is an
  explicit rejection. This reads and exposes the
  fact only -- the emitted `.sslm` `TOK1` artifact's own format and every prior checkpoint's
  emitted bytes are unchanged (confirmed byte-identical against the unmodified converter). A
  second, additive parity check (`--verify-post-processor`) confirms
  `ref_encode(text) + [trailing_special_id] == hf.encode(text, add_special_tokens=True)` for
  every checkpoint, run unconditionally rather than skipped for one whose post-processor appends
  nothing -- vacuity for that population is a property of the comparison's own result (it already
  reads 0 mismatches), not of the check declining to run. **This new check's own readings are
  quarantined pending an independent must-accept/must-reject commissioning** (not attempted this
  round) -- it is not yet a load-bearing pass/fail gate. User-visible: the pinned candidate's
  `tokenizer.json` now converts to a `.sslm` tokenizer artifact; every other checkpoint's
  converted output is unchanged.

## [1.3.1] - 2026-09-02

A patch: the six CI jobs behind 1.3.0's own guarantees, fixed at source, plus the engine's
own handling of non-square attention geometry (`q_width` distinct from `hidden_size`),
already on `main`.

### Fixed

- **The Linux/ELF FP-free scan leg (`_X86_GPR_ALLOW`) now accepts `bswap`.** `superslm::Sha256::
  Final` (`src/sha256.cpp`) compiles, under the runner's own GCC 13.x (`-O3 -DNDEBUG`, matching
  the `linux-x64` job's `-DCMAKE_BUILD_TYPE=Release` recipe), to a `bswap` on the byte-swapped
  big-endian length write -- a pure integer byte-reversal (Intel SDM Vol. 2A: no rounding, no
  exception, no floating-point register read) absent from the checked-in GPR allow-list
  `check_fp_free_scan.py` documents as frozen and reviewed-diff-only. Added,
  individually vetted, following that list's own precedent (`shrd`/`shld`, `cpuid`, `rep`,
  `vzeroupper`, `xgetbv`).

  **The `linux-x64` job's own scan step has NOT actually run** -- this branch has never been
  pushed, and no GitHub Actions run of that job exists at this tip. What stands instead (T-2533,
  closing M-4n/M-5n) is the job's own two-step recipe reproduced end to end, off this exact tip,
  under the real runner's own compiler: a fresh `git clone` of this branch into WSL/Ubuntu (the
  same checkout convention `ubuntu-latest` uses), GCC 13.3.0-6ubuntu2~24.04.1 (the exact release
  T-2530's own review named for the hosted runner, fetched as `.deb` packages and extracted
  without root -- no system package install), `cmake -B build -DCMAKE_BUILD_TYPE=Release` +
  `cmake --build build --target superslm`, then `python3 scan_build_output.py
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
  states this plainly, at this same tip: the population "remains outstanding" (D-SLM5230).
  An instrument whose must-reject has never fired has not been shown able to fail -- so the
  Linux/ELF no-floating-point guarantee is **enforced in CI (this patch closes the leg's own
  run) and not yet delivered**; see this entry's own Notes, below, for the corrected timeline --
  1.3.0 promised delivery "for 1.3.1", and this release is 1.3.1, but the population is still
  unbuilt, so delivery is deferred again rather than claimed here.

- **`tools/convert_tokenizer.py`'s `derive_model_name` now parses a checkpoint path's own
  separators directly, regardless of the OS running the converter.** A Windows-style checkpoint
  path (backslash-separated, this project's own HF hub cache convention) previously returned the
  whole path as the emitted CONFIG section's model label when the converter ran on a POSIX host
  -- `pathlib.Path` splits only on the running OS's own separator convention. User-visible: the
  label every converted `.sslm` artifact's CONFIG section carries.

- **`DynamicScaleReciprocal`'s seed computation (`src/intmath.cpp`) no longer has signed-integer-
  overflow undefined behaviour at non-canonical magnitudes.** `CarriedScaleReciprocal`
  (`checked_chain_funnel.h`) is an explicitly unguarded door onto this function -- a non-canonical
  `Dn` (outside `[2^30, 2^31)`) is a real, doc-permitted input, not merely a defensive
  possibility -- and the seed's own `int64_t` multiply overflowed at `Dn=2^62`
  (`TestT2019_B1_DynamicScaleReciprocal_DomainSweep_GpuMatchesCpu`'s own fixture, caught by
  `linux-x64-asan`'s UBSan build, `intmath.cpp:313`). Computed in `uint64_t` instead (unsigned
  overflow is modular arithmetic, not UB, and two's-complement wraparound reproduces the identical
  bit pattern the signed multiply already produced on this platform in every shipped build of
  this function), reinterpreted back to `int64_t`, then right-shifted as signed -- bit-identical
  to the original narrow-multiply result on every input, canonical or not. No emitted bit moves;
  the UB is removed, not the arithmetic.

- **`tests/ci/test_geometry_site_census.py`'s own D-SLM5785 call-site regression is genuinely
  closed on both checkout conventions.** A prior round's fix for this regression (a real,
  once-live bug: `_mutated`'s own newline-convention argument silently reverted to the platform
  default, corrupting an `eol=lf` file on write) was itself claimed closed without the cell it
  actually held in -- on a CRLF checkout (this project's own Windows development convention) the
  claim was true; on a fresh LF checkout (what every `ubuntu-latest` job that runs this module
  actually checks out) the identical regression stayed undetectable, because the one real-tree
  file the fixture's own state depended on is LF-native on Linux, and a platform-default write to
  an already-LF file is a no-op. Two synthetic, platform-independent cells now discriminate the
  regression on whichever checkout convention is running: one forces CRLF bytes into a scratch
  file outside the checked-out tree, the other forces LF, so the module no longer depends on the
  real tree's own accidental line-ending state to catch a revert. Both cells verified by direct
  execution on byte-preserved trees, both conventions.

- **`fp-free-scan-gate`'s red suite no longer guesses which MSVC edition a Windows runner
  carries.** The suite's own `VsDevCmd.bat` discovery (`fp_scan_common.py`, `conftest.py`) sorted
  a `vswhere`-reported list by a hand-picked edition preference (Community-first,
  BuildTools-first); `windows-latest`'s own hosted image carries VS 2022 Enterprise only, which
  neither preference ever matched, erroring 22 red-suite cells with `ToolUnavailable`. The job now
  resolves `VsDevCmd.bat` itself, in its own workflow step, using the runner's own `vswhere.exe`
  with no edition preference -- the first VS 2022 instance reported, whichever edition -- and
  exports it as `SUPERSLM_VSDEVCMD`; discovery in both modules honours that variable before
  falling back to the prior vswhere-then-hardcoded search, which stays in place for a local
  developer machine or any environment the variable is not set in. The job's own runner image is
  also now pinned to `windows-2022` -- the image the red suite's own fixtures are commissioned
  against -- rather than the floating `windows-latest` alias, which can point at an image with no
  VS 2022 instance at all.

- **`tools/ci/branch_coverage_floors.json`'s pinned floors re-measured against the sanctioned
  cell.** `src/model.cpp` and `src/proof_manifest.cpp` gained real branches from feature work done
  since their floors were last set (new `SslmSectionType` enum members and their `SectionTypeName`
  mapping; new geometry-gate validation branches) and are re-pinned to the hosted `branch-coverage`
  job's own measured values (`67.31343283582089`, `53.84615384615385`) -- genuine coverage debt
  from unrelated feature work, not a regression, and not backfilled here. `src/matmul.cpp`'s own
  floor is deliberately left unmoved at the CI leg's own last reported value
  (`74.39024390243902`): this file's own comment states its rule plainly -- only a number measured
  on the sanctioned cell (the hosted `branch-coverage` job itself) is a floor for this file, and an
  off-cell reading, however many are taken, never becomes one.

- **The engine no longer requires a square attention geometry (`hidden_size == num_attention_heads
  * head_dim`).** `CheckConfigGeometry`'s own R1 identity check is removed, not loosened;
  `MarshalLayer`'s q-projection channel count becomes `q_width`, a quantity independently derived
  from `num_attention_heads * head_dim` rather than assumed equal to `hidden_size`, and threaded as
  a new trailing parameter through `RunLayerLoop` (both overloads) and
  `RunLayerLoopChunkBatched` -- additive for every existing (square) caller, which defaults it to
  `0` (derive as `hidden_size`, the prior behaviour, bit-identical). `q_codes`/`q_rot`/`k_rot`/
  `ctx_wide`/`ctx_codes` buffers and the Q/O projection call widths are widened to `q_width`;
  `num_heads` is now re-derived from `q_width`, not `hidden_size`. Production callers
  (`sslm_abi.cpp`, `gpu_1p0.cpp`) thread the real `q_width` from
  `config.num_attention_heads * head_dim`. `sslm_convert_validate.py`'s own
  `check_config_geometry` widened identically, so a non-square checkpoint that was previously a
  named load-time rejection now converts.
- **The LoRA adapter-marshal path's own geometry now agrees with the base-model decoupling
  above.** Two defects the base-model geometry work did not reach: `AdapterOutChannelsFor`
  returned `hidden_size` for q_proj's real out-channel count (`q_width`), and
  `AdapterInChannelsFor` returned `hidden_size` for o_proj's real in-channel count (`q_width`) --
  against a non-square base geometry, both incorrectly rejected or under-read a correctly-shaped
  adapter artifact. Both now read the real, independently-derived `q_width`.

**Notes.** Two things this patch does **not** claim, stated here rather than left implied. It is
**not the Qwen3-arch pin** (`SuperSLM_Plan.md` §22.5): Ask 5's Tracks C and E are on `main` and
ship in this tag's own code because they are additive and tools-only (see `[Unreleased]` above),
but the architecture pin itself is claimed only once Track B (the forward call site) lands, as
1.4.0 -- a consumer reading 1.3.1 must not conclude it can run the candidate. The **Linux/ELF
FP-free guarantee stays deferred**: the leg is enforced and green in CI as of this patch, but the
design's own must-reject population for that leg (the fiftieth population, `t2265-superslm-fp-
free-open-design-2026-08-24.md` Sec5.4) is still unbuilt, so 1.3.0's own promise of the guarantee
"for 1.3.1" is carried forward honestly -- to whichever release actually builds that
population -- rather than claimed here.

**Verification.** The hosted matrix is green on `main`@`9c46750`: run
[33652587914](https://github.com/dansupergameprogrammer/SuperSLM/actions/runs/33652587914),
**31 jobs, every conclusion success** -- `linux-x64`, `fp-free-scan-gate`, `windows-x64`,
`macos-arm64-digest`, `forward-leaf-check`, `linux-x64-clang-digest`,
`matmul-avx-isolation-guard`, `macos-arm64`, `linux-x64-tsan`, `linux-x64-clang-sse2-forced`,
`workflow-lint`, `linux-x64-clang-avx2-forced`, `windows-msvc-digest`,
`linux-x64-clang-scalar-forced-digest`, `gpu-guard-status-parity-check`, `converter-validate`,
`geometry-site-census`, `present-tense-defect-comment-check`,
`linux-x64-clang-avx2-forced-digest`, `ci-claims-check`, `linux-x64-gcc-digest`,
`windows-clangcl-digest`, `bad-alloc-membership-check`, `linux-x64-clang-avx512-forced-digest`,
`generators`, `linux-x64-asan`, `linux-x64-debug`, `linux-x64-clang-sse2-forced-digest`,
`linux-x64-clang-avx512-forced`, `branch-coverage`, `axis-digest-compare`. The first green matrix
run since before the 1.3.0 tag; `ecadbb6` and `3c741d5` (T-2555, T-2556) were red on two
runner-cell jobs the matrix's own image never let a local machine reproduce.

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
