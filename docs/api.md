# API surfaces

SuperSLM ships two public APIs: the D3D12 GPU surface has ordinary C++ linkage,
while the CPU embedding surface is an `extern "C"` ABI. Both follow the same
status-code philosophy: a fallible call
returns a status enum with one distinct value per real failure cause, never
a single generic "failed" code, so a caller can tell "your input was
malformed" apart from "the object is in the wrong lifecycle state" apart
from "the content doesn't match what you told me it was" without
inspecting a side channel.

Both surfaces carry the same determinism guarantee at the level they
operate on: for a certified platform (see
[platform-support.md](platform-support.md)), the same model, prompt, and
decoding configuration produce identical output tokens on every call, and
on a certified GPU, identical intermediate layer state against the CPU
reference, bit-for-bit.

## The GPU handle API (`SslmGpu*`) — shipped

`include/superslm/gpu_1p0.h` is the contract. This is the D3D12-backed GPU
acceleration surface, Windows-only, and it is what
[the certified GPU numbers](platform-support.md) are measured through.

### Handles

Four opaque handle types own the API's state: a context (`SslmGpuContext`,
one per device), a mapped model (`SslmGpuModelHandle`), a mapped LoRA
adapter (`SslmGpuAdapterHandle`), and a decoding sequence
(`SslmGpuSequenceHandle`). Every fallible call takes the handles it needs
and returns a status; values the call produces — a new handle, a readiness
flag, a batch's per-sequence outcomes — come back through an out-parameter,
never through the return value itself.

### Lifecycle

- **Context**: `sslm_gpu_context_create` / `sslm_gpu_context_destroy`.
  `GpuContextConfig::shader_dir` chooses where the compiled `.cso` shader
  set is loaded from. Left `NULL` (a zero-initialized config), shaders load
  from a `shaders` directory beside the host executable, as in every earlier
  release. Otherwise it is an absolute directory path in UTF-8 holding the
  compiled set, read only during the create call. **The shader directory is
  process-wide**, whichever context supplies it: compiled pipelines are
  cached per process by shader name, so one process loads every shader from
  one directory for its whole lifetime. That directory is fixed by the
  first successful create that names one, or by the first shader load
  through the default location, whichever comes first. After that, a create
  with `NULL` uses it, a create naming the same directory in any spelling
  succeeds, and a create naming a different directory is refused with
  `SSLM_GPU_SHADER_DIR_CONFLICT`. A value that is empty, not valid UTF-8,
  not an existing directory, a directory with no `.cso` file, or not fully
  qualified is refused with `SSLM_GPU_SHADER_DIR_INVALID`. Fully qualified
  means it begins with a drive root (`X:\` or `X:/`) or a UNC prefix
  (`\\server\share`), so a relative path, `\shaders` (rooted on the current
  drive) and `C:shaders` (relative to drive C's current directory) are all
  refused. Both refusals happen
  before any device is created and leave `*out_ctx` null.
- **Model**: `sslm_gpu_model_map` maps an already-loaded model view onto a
  context; `sslm_gpu_model_unmap` releases it, and refuses (`Busy`) while
  any sequence still has decode work in flight against it.
  `GpuResidencyConfig::flags` (1.7.0) takes one option,
  `SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE`, which moves the token finish's logits
  onto the device for that model (see
  [The token finish](#the-token-finish) below). Any other bit is refused
  with `SSLM_GPU_RESIDENCY_FLAGS_INVALID` and no handle.
- **Adapter**: `sslm_gpu_adapter_map` maps a LoRA adapter artifact against
  an already-mapped model, rejecting a base-model mismatch; `sslm_gpu_
  adapter_unmap` releases it, with the same in-flight-work refusal as model
  unmap, plus a refusal (`SSLM_ADAPTER_HAS_BOUND_SEQUENCES`) while any
  sequence still holds a bind to it (`sslm_gpu_seq_bind_adapter`, below).
- **Model**: `sslm_gpu_model_unmap` also refuses
  (`SSLM_MODEL_HAS_LIVE_ADAPTERS`) while any adapter is still mapped
  against it, independent of whether any sequence is live.
- **Sequence**: `sslm_gpu_seq_create` / `sslm_gpu_seq_release`;
  `sslm_gpu_seq_embed_token` feeds a starting token; `sslm_gpu_seq_reset`
  clears a sequence back to empty; `sslm_gpu_seq_save` / `sslm_gpu_seq_
  restore` serialize a sequence's full state to a caller buffer and back,
  rejecting a restore against a model that isn't the one the state was
  saved from. The current `SLM5` blob adds a schema binding, walk state,
  and "ready for logits" flag to the `SLM4` shape, so a schema-bound
  sequence's generation position survives a save/restore round trip; the
  restored binding and walk state are validated against the target model's
  own schema count and state count before use. A save always writes the
  current `SLM5` format; restore also still accepts an older `SLM4` blob
  exactly as before, defaulting the schema binding it carries to unbound.
  Older `SSLM`, `SLM2`, and `SLM3` layouts are rejected on their versioned
  magic rather than misread.
- **Adapter binding**: `sslm_gpu_seq_bind_adapter` binds (or, passed a
  null adapter, unbinds) a LoRA adapter to a sequence handle *across*
  calls — distinct from the per-call `adapter_or_null` argument every
  decode call already takes. A bound adapter is read automatically by
  `SslmGpuSeqDecodeStepForG5Bridge` and by the prefill entry points below;
  it does not change what a direct `sslm_decode_step_gpu`/`sslm_decode_
  step_batch_gpu` caller must still pass explicitly. Rejects a
  model-mismatched adapter, a foreign-context adapter, or a call made
  mid-token (`Busy` — a drained rest, at either token boundary, always
  admits); unbinds automatically on `sslm_gpu_seq_release`; survives
  `sslm_gpu_seq_reset`; does not round-trip through save/restore — a
  restored sequence's binding is always null and is re-bound explicitly if
  wanted. `sslm_gpu_adapter_unmap` refuses (`SSLM_ADAPTER_HAS_BOUND_
  SEQUENCES`) while any sequence still holds a bind to that adapter.

### Decoding

- `sslm_decode_step_gpu` advances one sequence, with an optional
  per-sequence LoRA adapter and a caller-chosen layer-budget
  (`dispatch_budget`) — the mechanism behind sliceable inference. The same
  prompt decoded at any layer-budget granularity, down to one layer per
  call, produces bit-identical output on a certified GPU.
- `sslm_decode_step_batch_gpu` advances several sequences in one call, each
  with its own optional adapter, sharing one batch-wide layer budget. A
  rejection on one sequence in the batch (returned per-sequence in
  `out_statuses`) does not abort the others.
- `sslm_gpu_ready` polls (or, with `block`, waits for) a sequence's
  in-flight GPU work to complete.

### Reading the prefill hidden state

`sslm_gpu_seq_read_prefill_final_hidden` returns the post-`final_norm`
hidden state at the last position left by a sequence's most recent prompt
or schema-content prefill call that reached its admission pre-scan and
returned `SSLM_OK`. It reads a per-sequence snapshot taken only when such a
call succeeds, so no later embed, decode, or finish call changes what it
returns; `sslm_gpu_model_hidden_size` gives the width to size the caller's
buffer with (`SSLM_OUTPUT_BUFFER_TOO_SMALL` if it's short). A prefill call
that reaches its pre-scan and then fails, and `sslm_gpu_seq_reset`, empty
the snapshot — the read then returns `SSLM_PREFILL_HIDDEN_UNAVAILABLE`, not
an earlier frame. A prefill call refused before its pre-scan (malformed
arguments, a zero count, `SSLM_BUSY`, or the schema-content prefill's
unbound-schema and unreachable-first-token refusals) leaves the snapshot as
it was.

### The token finish

The token finish is the work after a token's last layer: the final norm,
the logits over the whole vocabulary, then the argmax, schema mask,
damped-greedy selection and dead-end rule. In the GPU API it is
`SslmGpuSeqFinishTokenForG5Bridge` (and the composed bridge calls that use
it); on the CPU it is the end of `sslm_decode_step`/`sslm_decode_step_v2`.
The logits are the expensive part, and 1.7.0 offers two independent ways to
take them off the calling thread. Neither changes a single token: each logit
is an exact integer sum, and the narrowing, mask, argmax and dead-end rule
run unchanged, on the host, on the identical row.

**A host parallel-for hook, both backends.** SuperSLM never creates a
thread. A host that wants the logits rows computed on several threads
installs an `sslm_parallel_for` (`include/superslm/parallel_for.h`) on a
CPU workspace with `sslm_workspace_set_parallel_for`, or on a GPU context
with `sslm_gpu_context_set_host_parallel_for`. The finish then splits the
rows into contiguous blocks and hands them to the hook's `run`. With no hook
installed the finish runs serially on the calling thread, as in every
earlier release, and its tokens are identical to 1.6.0; its measured cost
against 1.6.0 is in the
[1.7.0 release note](releases/1.7.0.md#no-hook-cost).

- `run` must invoke each task index in `[0, task_count)` exactly once, on
  any threads, and return only after every invocation has returned. The
  header states the whole contract.
- A `run` that omits, repeats or invents a task index, while still waiting
  for its invocations, fails the call without producing a token:
  `SSLM_INVALID_ARGUMENT` on the CPU, `SSLM_GPU_PARALLEL_FOR_INCOMPLETE` on
  the GPU. The sequence is left with its final hidden state ready for
  logits, so the same finish can be retried through a correct hook. A `run`
  that returns while an invocation is still running has undefined behaviour;
  that part of the contract cannot be checked.
- `max_tasks` is at most `SSLM_PARALLEL_FOR_MAX_TASKS` (256). A setter
  refuses a `reserved` field that is not 0, a `max_tasks` outside
  `[0, 256]`, or a null `run` with `max_tasks` above 1
  (`SSLM_INVALID_ARGUMENT` on the CPU, `SSLM_GPU_PARALLEL_FOR_INVALID` on
  the GPU). Passing `NULL` clears the hook.
- Only the finish's logits step reads the hook. Prefill, prefix prefill and
  every other call ignore it.
- A host with no job system can use the reference `run` in
  [`docs/parallel_for_reference.hpp`](parallel_for_reference.hpp): a small
  `std::thread` pool that meets the contract. It is documentation, not a
  library API, and the test suite compiles it as it stands.

**A device-resident head, GPU backend, opt-in per model.** Mapping a model
with `SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE` uploads its output head table (the
tied embedding, or `lm_head` when untied) to the device. The finish then
computes the exact int64 logits row on the device, reads it back and
narrows it on the host. With the flag set no separate host copy of the
head is taken: an untied model's `lm_head` is not copied to the host, and a
tied model's head is its embedding table, which the handle keeps on the host
in every case for token embedding. The context's host hook is not used for
that model. New VRAM per mapped model is
the head table (`vocab_size × hidden_size` bytes: 136,134,656 B for
Qwen2.5-0.5B and 233,373,696 B for 1.5B) plus a `hidden_size × 4` B input
row and a `vocab_size × 8` B output row, each rounded up to the 64 KiB
allocation granule; nothing is allocated per sequence or per token. With the
flag clear a model maps exactly as before and uses no new VRAM.

- The map loads `logits_site.cso` from the process's shader directory: a
  stale binary refuses the map with `SSLM_GPU_SHADER_BINARY_STALE`, a
  missing one with `SSLM_DEVICE_LOST`.
- An out-of-memory failure while allocating the head's device buffers, on a
  device that is not removed, refuses the map with
  `SSLM_GPU_ALLOCATION_FAILED`; the context stays usable, and the map can
  be retried on it, with or without the flag.

The GPU handle keeps one host copy of a tied head, with the flag set or
clear: a tied model's head is its embedding table, which the handle keeps on
the host in every case for token embedding, so no second copy is taken
(before 1.7.0 there were two). An untied model mapped without the flag keeps
a host copy of its `lm_head`.

### Thread safety

Calls against **different** sequence handles are safe to make from
different threads concurrently. Any call that submits GPU work — either
decode call, `sslm_gpu_ready` with `block` set, or `sslm_gpu_seq_restore`
(which uploads the restored sequence's K/V state to a fresh device
buffer, and refuses `Busy` while *any* sequence anywhere in the process
— any model, any context — has unfenced in-flight work, because the
decode dispatch path shares one process-global command allocator/list
regardless of which model or context submitted it) — must be externally
serialized by the caller relative to every other GPU-submitting call on
the same context; the API does not build an internal queue lock. Two
threads driving the *same* sequence handle concurrently is not a
supported use. A token finish on a model mapped with
`SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE` submits its logits dispatch on the
context too, so it is serialized the same way, as is
`sslm_gpu_context_set_host_parallel_for`.

### Status causes

`SslmGpuStatus` is a scoped C++ enum. Its values are written as
`SslmGpuStatus::SSLM_OK`, `SslmGpuStatus::SSLM_BUSY`, and so on; this keeps the
GPU header safe to include with the CPU C ABI header in either order. It distinguishes:
a dispatch budget too small to make
progress; the device busy with in-flight work on the handle you're
releasing, or with any unfenced in-flight work elsewhere in the process
when you're restoring; a context or model with handles
still live; a model with an adapter still mapped against it
(`SSLM_MODEL_HAS_LIVE_ADAPTERS`) or an adapter with a sequence still
bound to it (`SSLM_ADAPTER_HAS_BOUND_SEQUENCES`) — both persistent
conditions that hold until the caller explicitly unmaps/unbinds, unlike
the transient in-flight-work `Busy`; an adapter that doesn't match the
model it's mapped against, by content hash or by identity; a sequence's
saved KV state that doesn't match the buffer shape it's being restored
into; a lost/reset device; a batch call that ran out of its shared
budget; an out-of-range token id; a single sequence's decode step
rejected on structural grounds unrelated to device health (so a healthy
device serving other sequences in the same batch is distinguishable from
a real device loss); a restore whose blob doesn't match the model it's
being restored against; and `SSLM_GPU_SHADER_BINARY_STALE`, which means a
deployed `.cso` predates one of its HLSL inputs; and
`SslmGpuStatus::SSLM_GPU_ALLOCATION_FAILED`, which reports allocation failure
without allowing a C++ exception to cross the public `noexcept` boundary. A stale
shader requires a
matching shader rebuild/redeployment; the model, sequence, and device are
not condemned by it.

Two more statuses cover the prefill-hidden read below: `SSLM_OUTPUT_BUFFER_TOO_SMALL`,
a caller buffer too small for the hidden state's width; and
`SSLM_PREFILL_HIDDEN_UNAVAILABLE`, no live snapshot to read (see
[Reading the prefill hidden state](#reading-the-prefill-hidden-state) below).

Two cover the context's shader directory (see Lifecycle above):
`SSLM_GPU_SHADER_DIR_INVALID`, a `GpuContextConfig::shader_dir` that cannot
name a compiled shader set — fix the path; and `SSLM_GPU_SHADER_DIR_CONFLICT`,
a directory that differs from the one this process already loads shaders
from — supply the same directory, or `NULL`.

Three cover the token finish (see [The token finish](#the-token-finish)):
`SSLM_GPU_PARALLEL_FOR_INVALID`, a host parallel-for hook with an invalid
field, refused by `sslm_gpu_context_set_host_parallel_for`;
`SSLM_GPU_PARALLEL_FOR_INCOMPLETE`, a finish whose hook's `run` broke its
exactly-once contract, retryable through a correct hook; and
`SSLM_GPU_RESIDENCY_FLAGS_INVALID`, a model map with an undefined
`GpuResidencyConfig::flags` bit.

`SslmGpuSeqPrefillPromptForG5Bridge` and `SslmGpuSeqPrefillSchemaContentForG5Bridge`
diverge on one refusal: when a device-side domain guard refuses one of the admitted
tokens and the device is not reported removed, the prompt entry point returns
`SSLM_SEQUENCE_REJECTED` — the context and the sequence's own device state stay
usable, but a decode issued on the sequence without a reset reads a residual that is
not a resting state, so `sslm_gpu_seq_reset` is required before reuse. The
schema-content entry point reports the identical refusal as `SSLM_DEVICE_LOST`
instead, with the same reset requirement (see each function's own header comment,
`include/superslm/gpu_1p0.h`, for why the two calls are not unified).

## The CPU consumer API (`sslm_*`) — shipped

`include/superslm/sslm_abi.h` is the contract: a from-scratch, engine-
agnostic C ABI for embedding SuperSLM's CPU inference path directly in
another process — a game engine's own tooling, for instance — without the
GPU handle types above. It declares and implements 38 functions across the
same lifecycle shape as the GPU API (workspace and KV-pool sizing and
creation, model map/unmap, sequence and prefix lifecycle, decode,
tokenize/detokenize, stats) plus concepts the GPU API does not need:
caller-owned workspace and KV-pool memory (sized by the library and allocated
by the caller). A correctly sized workspace removes the ABI layer's transient
forward and damped-selection buffers; the engine's existing compute kernels retain their
documented, shape-stable internal scratch allocations. Damped greedy additionally grows its
per-sequence anti-LM state as tokens and new n-grams appear. The count-table
component is content-dependent and reported by `AntiLmRetainedBytes`; total
retained state also includes four bytes per generated-history token. Neither is
represented as caller workspace. Shared-prefix
"prefix" handles that let more than one sequence reuse one prefilled prompt
prefix, and schema binding (below).

`SSLM_ABI_ALIGNMENT_BYTES` (64 bytes) is the alignment `sslm_workspace_create`
and `sslm_kv_pool_create` both require of the caller-supplied buffer; passing
a misaligned buffer is rejected (`SSLM_MISALIGNED_BUFFER`) rather than
silently accepted.

### Lifecycle

- **Model**: `sslm_model_map` / `sslm_model_unmap`.
- **Workspace and KV pool sizing**: `sslm_workspace_size`, `sslm_kv_block_size`,
  `sslm_kv_pool_overhead_size`, and `sslm_seq_state_size` compute caller-buffer
  capacities (`sslm_seq_state_size` is an upper bound across live sequence states);
  `sslm_workspace_create`/`_destroy`
  and `sslm_kv_pool_create`/`_destroy` take those buffers and hand back a
  handle. A workspace is reusable across a sequence of calls but is not
  safe to share between two calls running concurrently; a caller driving
  multiple sequences concurrently needs one workspace per concurrently
  active call.
- **Prefix** (shared prompt prefix): `sslm_prefix_begin` / `sslm_prefix_release`,
  `sslm_prefix_prefill` (runs the shared prefix's own forward pass once),
  `sslm_prefix_freeze` (locks it for adoption by sequences).
- **Sequence**: `sslm_seq_create` / `sslm_seq_release`, `sslm_seq_reset`,
  `sslm_seq_adopt_prefix` (attaches a frozen prefix, so its forward pass is
  never repeated per sequence), `sslm_seq_save` / `sslm_seq_restore`
  (serializes a sequence's full state — including its schema binding and
  DFA walk state, see below — to a caller buffer and back). v1.2.1 writes
  `SSB4` blobs: the residual is serialized unconditionally whenever
  `hidden_size > 0` (a ready-for-logits sequence, `layer_index == 0`, carries
  a real residual and is no longer saved with it silently dropped — the fixed
  1.2.0 defect), and an explicit `ready_for_logits` field is carried in the
  header rather than inferred on restore. Restore continues to accept shipped
  `SSB3` (v1.2.0) and `SSB2` blobs read-only; a legacy `SSB3` blob resting at
  the one state the 1.2.0 defect could produce is rejected with
  `SSLM_RESTORE_RESIDUAL_LOST` rather than silently restored wrong. Restore
  accepts a buffer whose size is at
  least the encoded blob size, including a buffer sized by
  `sslm_seq_state_size`, and ignores trailing capacity. It also rejects anti-LM
  history longer than the blob's own saved context length;
  `sslm_seq_set_adapter` (attaches or detaches a LoRA adapter on a live
  sequence).
- **Adapter**: `sslm_adapter_map` / `sslm_adapter_release`, rejecting a
  base-model mismatch; `sslm_adapter_residency` reports its resident byte
  size.

### Decoding

- `sslm_prefill` runs a chunk of tokens (prompt or, once a schema is bound,
  forced schema content) through the forward pass in one batched call —
  proven bit-identical to processing the same tokens one at a time, at
  every chunk size (see the README's sliceable-inference section).
- `sslm_decode_step` is the v1.1-compatible greedy entry point. It advances a
  batch by one token and reads only `sslm_decode_params.layer_budget`, the
  complete four-byte shape released in v1.1. It never probes later fields, so
  an unchanged old binary remains safe.
- `sslm_decode_step_v2` is the extended greedy/damped-greedy entry point.
  Initialize its parameter block with `sslm_decode_params_init(model, mode,
  layer_budget, &params)`. For greedy this zeroes every damped-only field. For
  damped greedy it selects the ruled defaults (`alpha_q15=65536`, anti-LM order
  `2`, `top_k=min(6, vocab_size)`) and derives `q_ln2`/`q_b`/`q_c` from the
  mapped artifact's DGC1 scale; an artifact without that opt-in section returns
  `SSLM_ARTIFACT_REJECTED`. Callers that populate the struct manually set
  `struct_size = sizeof(sslm_decode_params)`; any other value is a
  defined `SSLM_INVALID_ARGUMENT` rejection before another extended field is
  read. The distinct symbol—not an unsafe in-place size probe—is what makes
  header/library skew explicit. `layer_budget` remains the caller-chosen layer
  budget, the mechanism behind sliceable inference. `mode` selects the
  decode-step's own selection mechanism:
  `SSLM_DECODE_MODE_GREEDY` (0, the default under zero-init) or
  `SSLM_DECODE_MODE_DAMPED_GREEDY` (1) — any other value is rejected, never
  silently treated as greedy. Selecting damped mode directly through this entry
  point also requires a valid DGC1 section; manually supplied scale constants are
  validated before forward work begins. Under damped-greedy mode, six more fields
  apply: `alpha_q15` (the Q15-scaled anti-repetition weight, an `int32_t`
  rejected outside `[0, 2^20)`), `anti_lm_max_order` (the anti-LM's own n,
  `>= 1`), `top_k` (candidates scored per step, `1 <= top_k <= vocab_size`),
  and `q_ln2`/`q_b`/`q_c` (runtime i-exp scale constants initialized from the
  artifact). All five are ignored under greedy mode. `out_tokens[i]` carries
  THREE reserved sentinel values alongside a real token id: `-1` means the
  call is still mid-token and safe to retry (call again with the same layer
  budget to continue); `-2` means that sequence's schema-bound walk has
  reached a state with no legal continuation at all — a per-sequence
  outcome, not a call failure, and safe to retry (nothing about that
  sequence changes until the caller does something else with it — rebind a
  schema, reset, etc.); and `-3` means this index named a sequence that is
  **not currently live** (concurrently released by another thread) — unlike
  `-1`, this is **not** safe to retry with the same state, since the caller
  no longer holds a live handle to that sequence at all. All three leave
  the overall call returning `SSLM_OK`. A numeric refusal on an otherwise
  valid model and valid params (a per-step gate declining, not an artifact
  defect) returns `SSLM_NUMERIC_STEP_REFUSED` rather than rejecting the
  model — safe to retry once the caller adjusts the parameters that
  triggered it.
- `sslm_workspace_set_parallel_for` (1.7.0) installs a host parallel-for hook
  on a workspace, or clears it with `NULL`. `sslm_decode_step` and
  `sslm_decode_step_v2` then split the token finish's logits rows across the
  hook's `run` when that workspace is passed; with no hook, or no workspace,
  the finish is serial on the calling thread as before. Tokens are identical
  either way. A hook that breaks its exactly-once contract fails that call
  with `SSLM_INVALID_ARGUMENT` and leaves the sequence ready to retry. See
  [The token finish](#the-token-finish) above for the contract and the
  reference `run`.
- `sslm_tokenize` / `sslm_detokenize_stream` convert between text and token
  ids; the streaming detokenizer carries a small caller-owned state struct
  across calls so a partial UTF-8 sequence at a call boundary is handled
  correctly.
- `sslm_stats` reports per-sequence counters: the decode-step ceiling and
  actual layers run, `forced_token_count` (how many tokens this sequence
  has had forced onto it by schema jump-forward rather than chosen by
  argmax), the resident KV block count, and `schema_accepting` (1 iff a
  schema is bound and the sequence's current parse state is one where
  stopping is valid; 0 if not, and 0 when no schema is bound).

### Schema-constrained generation

A schema is compiled offline (see [sslm_format.md](sslm_format.md)'s
`SchemaMasks` section) into a table of named, independently-compiled
per-token-id valid-continuation masks, indexed by parser state, and shipped
inside the `.sslm` artifact. `sslm_schema_lookup` resolves a schema by name;
`sslm_schema_count` / `sslm_schema_name` enumerate every schema an artifact
carries. `sslm_seq_set_schema` binds, rebinds, or unbinds a schema only after
`sslm_seq_create` or `sslm_seq_reset`. A generation call that passes argument
validation makes the sequence ineligible until reset, including a valid no-op
call or empty-prefix `sslm_seq_adopt_prefix`. A call rejected for invalid
arguments leaves the sequence untouched and still bindable. A restored sequence
must be reset before binding. The call returns `SSLM_SCHEMA_BIND_REJECTED`
without changing the binding or walk state when the sequence is ineligible.
`sslm_prefix_set_schema` does the same
for a prefix under construction, using `SSLM_SCHEMA_NONE` to mean
unconstrained. `sslm_seq_schema_bound`
reports whether a sequence is bound, so callers can distinguish an unbound
zero from a bound, non-accepting zero in `sslm_stats::schema_accepting`.
`sslm_seq_reset` accepts every valid sequence state, including partial and
full token depth; it restarts generation while preserving the schema binding.

Once bound, `sslm_prefill` and `sslm_decode_step` carry the constraint
automatically: a masked argmax step forbids the model from emitting a token
that would break the schema, including on the spans the schema forces
deterministically without a real choice ("jump-forward" — for instance, a
fixed key name or a closing brace your schema already dictates), which
`sslm_prefill` also drives. A rejected span (`SSLM_SCHEMA_SPAN_UNREACHABLE`)
partially consumes: every token before the rejected one is fully and
permanently admitted (forward pass run, KV written, `forced_token_count`
advanced), and only the rejected token and anything after it in that call
has no effect — the same partial-consumption contract `sslm_prefill`
already has for an unconstrained span. The determinism guarantee is
unchanged by any of this — a schema-constrained decode is exactly as
reproducible, on the same certified platform, as an unconstrained one.

### Status causes

`sslm_status` carries one success value (`SSLM_OK`) plus 26 distinct
rejection causes (an internal sentinel past the last real value is never
returned or accepted as an argument), in five groups: argument/precondition
rejections (a bad argument, a buffer too small, a misaligned buffer);
artifact/content rejections (a rejected artifact, an adapter that doesn't
match its base model, a restore whose content or KV shape doesn't match);
lifecycle rejections (a model, pool, or adapter with live handles still
attached to it; an adapter swap mid-token; a frozen
prefix reused; a KV pool with no room left); numeric/domain rejections (a
token id out of range, a context length exceeded, a legal decode-output
token id with no tokenizer entry for the padded-vocabulary case, or —
distinct from all of those — a per-step numeric gate declining on an
otherwise valid model and valid params, `SSLM_NUMERIC_STEP_REFUSED`,
`sslm_decode_step_v2`'s own damped-greedy mode only); and schema rejections
(an unknown schema name; binding a schema to a non-fresh sequence; a
schema-content span on an unbound sequence; a prefix or restore whose
schema doesn't match; a fixed span the schema's own DFA cannot reach; a
schema the offline compiler could not prove satisfiable), the shared
`SSLM_GPU_SHADER_BINARY_STALE` deployment-mismatch cause, plus one
process-level resource-exhaustion cause distinct from a caller-supplied
buffer running out.

## The GPU schema-constrained decoding surface — shipped

The G5 schema-constrained-decoding verbs (`SslmGpuModelHasSchemasForG5Bridge`,
`SslmGpuSchemaLookupForG5Bridge`, `SslmGpuSeqSetSchemaForG5Bridge`,
`SslmGpuSeqSchemaBoundForG5Bridge`, `SslmGpuSeqSchemaAcceptingForG5Bridge`,
`SslmGpuSeqWalkStateForG5Bridge`, `SslmGpuSeqPrefillPromptForG5Bridge`,
`SslmGpuSeqFinishTokenForG5Bridge`, `SslmGpuSeqDecodeStepForG5Bridge`,
`SslmGpuSeqPrefillSchemaContentForG5Bridge`) live on the same shipped
`include/superslm/gpu_1p0.h` surface as the rest of the GPU API above — the
GPU-side twins of the CPU ABI's schema lookup, binding, prefill, and decode
calls, proven bit-identical against the CPU path (matching digest across 80
real decode steps) on the certified NVIDIA GPU. The same check passed
bit-identical on the certified AMD GPU as well (measured 2026-08-17 on the
Radeon RX 7900 XTX) — see [Certified platforms](platform-support.md).
The two schema-state queries are host-only and never submit, poll, or wait on
GPU work. They return `SSLM_BUSY` while a sequence is Submitted. An Idle
sequence that has been drained but not yet finished still reports its
pre-finish acceptance membership; drain and finish before treating the query
as an output-finality decision. Schema bind/rebind/unbind is accepted only
after sequence creation or `sslm_gpu_seq_reset`. A generation call that passes
argument validation makes the sequence ineligible until reset,
including a valid no-op call. A call rejected for invalid arguments leaves the
sequence untouched and still bindable. A restored sequence must be reset before
binding. `SSLM_BUSY` and malformed-handle refusals likewise leave eligibility
unchanged; an ineligible Idle sequence returns
`SSLM_SEQUENCE_REJECTED` without changing its binding or walk state.
`SslmGpuSeqDecodeStepForG5Bridge` is the recommended one-call-per-decode-step
entry point; a caller that always uses it (rather than hand-composing the
lower-level embed/decode/ready calls) cannot reproduce a class of
duplicate-KV-commit bug this project's own build process found and fixed
while landing this surface.

`SslmGpuSeqPrefillPromptForG5Bridge` and `SslmGpuSeqPrefillSchemaContentForG5Bridge`
are bulk-throughput calls, not submission-slicing contracts: each still
validates its `dispatch_budget`/`dispatch_budget_per_token` parameter as
nonzero, but records and submits every admitted token as one chunk
(subject only to an internal, driver-stability sub-chunk split, unrelated
to the parameter's value) rather than issuing budget-sized round trips per
token. Per-call, per-token submission slicing by a dispatch budget remains
the decode path's own contract — `sslm_decode_step_gpu` and
`SslmGpuSeqDecodeStepForG5Bridge`'s layer-loop-to-depth step — unchanged
by either prefill call.

A schema-bound sequence that reaches its schema's own dead end (an
accepting state whose mask page is legitimately all-zero, or a synthetic
degenerate row where every admitted logit ties and the tie-break returns a
token with no transition) returns `*out_token == -2` at `SSLM_OK` from
`SslmGpuSeqFinishTokenForG5Bridge`/`SslmGpuSeqDecodeStepForG5Bridge` —
never `SSLM_SEQUENCE_REJECTED`, which those calls reserve for their own
precondition failures — matching the CPU path's identical `-2` convention
for the same event. The walk state and layer index are left exactly as
they were and `ready_for_logits` is re-armed, so a further call to either
GPU entry point reproduces the identical `-2` result deterministically,
consuming no new token from the caller: retrying after a dead end is
always safe. A dead-ended sequence is `sslm_gpu_seq_reset`- and
`sslm_gpu_seq_bind_adapter`-eligible (see the CHANGELOG), and is proven
bit-identical between the CPU and GPU paths at real scale (see
[Certified platforms](#certified-platforms)).
