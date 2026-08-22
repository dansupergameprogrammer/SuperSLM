# API surfaces

SuperSLM ships two public C APIs, at different points on its own build
timeline. Both follow the same status-code philosophy: a fallible call
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
- **Model**: `sslm_gpu_model_map` maps an already-loaded model view onto a
  context; `sslm_gpu_model_unmap` releases it, and refuses (`Busy`) while
  any sequence still has decode work in flight against it.
- **Adapter**: `sslm_gpu_adapter_map` maps a LoRA adapter artifact against
  an already-mapped model, rejecting a base-model mismatch; `sslm_gpu_
  adapter_unmap` releases it, with the same in-flight-work refusal as model
  unmap.
- **Sequence**: `sslm_gpu_seq_create` / `sslm_gpu_seq_release`;
  `sslm_gpu_seq_embed_token` feeds a starting token; `sslm_gpu_seq_reset`
  clears a sequence back to empty; `sslm_gpu_seq_save` / `sslm_gpu_seq_
  restore` serialize a sequence's full state to a caller buffer and back,
  rejecting a restore against a model that isn't the one the state was
  saved from.

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

### Thread safety

Calls against **different** sequence handles are safe to make from
different threads concurrently. Any call that submits GPU work — either
decode call, or `sslm_gpu_ready` with `block` set — must be externally
serialized by the caller relative to every other GPU-submitting call on
the same context; the API does not build an internal queue lock. Two
threads driving the *same* sequence handle concurrently is not a supported
use.

### Status causes

`SslmGpuStatus` distinguishes: a dispatch budget too small to make
progress; the device busy with in-flight work on the handle you're
releasing; a context or model with handles still live; an adapter that
doesn't match the model it's mapped against, by content hash or by
identity; a sequence's saved KV state that doesn't match the buffer shape
it's being restored into; a lost/reset device; a batch call that ran out
of its shared budget; an out-of-range token id; a single sequence's decode
step rejected on structural grounds unrelated to device health (so a
healthy device serving other sequences in the same batch is distinguishable
from a real device loss); and a restore whose blob doesn't match the model
it's being restored against.

## The CPU consumer API (`sslm_*`) — shipped

`include/superslm/sslm_abi.h` is the contract: a from-scratch, engine-
agnostic C ABI for embedding SuperSLM's CPU inference path directly in
another process — a game engine's own tooling, for instance — without the
GPU handle types above. It declares and implements 35 functions across the
same lifecycle shape as the GPU API (workspace and KV-pool sizing and
creation, model map/unmap, sequence and prefix lifecycle, decode,
tokenize/detokenize, stats) plus concepts the GPU API does not need:
caller-owned workspace and KV-pool memory (sized by the library and allocated
by the caller). A correctly sized workspace removes the ABI layer's transient
forward buffers; the engine's existing compute kernels retain their documented,
shape-stable internal scratch allocations. Damped greedy additionally grows its
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
  `sslm_kv_pool_overhead_size`, `sslm_seq_state_size` compute exact byte
  counts for caller-allocated buffers; `sslm_workspace_create`/`_destroy`
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
  DFA walk state, see below — to a caller buffer and back), and
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
  silently treated as greedy. Under damped-greedy mode, five more fields
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
- `sslm_tokenize` / `sslm_detokenize_stream` convert between text and token
  ids; the streaming detokenizer carries a small caller-owned state struct
  across calls so a partial UTF-8 sequence at a call boundary is handled
  correctly.
- `sslm_stats` reports per-sequence counters: the decode-step ceiling and
  actual layers run, `forced_token_count` (how many tokens this sequence
  has had forced onto it by schema jump-forward rather than chosen by
  argmax), the resident KV block count, and `schema_accepting` (1 iff the
  sequence's current parse state is one where stopping is valid; 0 if not,
  and 0 when no schema is bound — a caller never needs to special-case
  whether a schema is bound before reading it).

### Schema-constrained generation

A schema is compiled offline (see [sslm_format.md](sslm_format.md)'s
`SchemaMasks` section) into a table of named, independently-compiled
per-token-id valid-continuation masks, indexed by parser state, and shipped
inside the `.sslm` artifact. `sslm_schema_lookup` resolves a schema by name;
`sslm_schema_count` / `sslm_schema_name` enumerate every schema an artifact
carries. `sslm_seq_set_schema` binds a schema to a sequence (only valid at a
fresh or just-reset sequence — no mid-generation rebinding) and
`sslm_prefix_set_schema` does the same for a prefix under construction, both
using `SSLM_SCHEMA_NONE` to mean unconstrained.

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
attached to it; an adapter swap or sequence reset mid-token; a frozen
prefix reused; a KV pool with no room left); numeric/domain rejections (a
token id out of range, a context length exceeded, a legal decode-output
token id with no tokenizer entry for the padded-vocabulary case, or —
distinct from all of those — a per-step numeric gate declining on an
otherwise valid model and valid params, `SSLM_NUMERIC_STEP_REFUSED`,
`sslm_decode_step_v2`'s own damped-greedy mode only); and schema rejections
(an unknown schema name; binding a schema to a non-fresh sequence; a
schema-content span on an unbound sequence; a prefix or restore whose
schema doesn't match; a fixed span the schema's own DFA cannot reach; a
schema the offline compiler could not prove satisfiable) plus one
process-level resource-exhaustion cause distinct from a caller-supplied
buffer running out.

## The GPU schema-constrained decoding surface — shipped

The G5 schema-constrained-decoding verbs (`SslmGpuModelHasSchemasForG5Bridge`,
`SslmGpuSchemaLookupForG5Bridge`, `SslmGpuSeqSetSchemaForG5Bridge`,
`SslmGpuSeqWalkStateForG5Bridge`, `SslmGpuSeqPrefillPromptForG5Bridge`,
`SslmGpuSeqFinishTokenForG5Bridge`, `SslmGpuSeqDecodeStepForG5Bridge`,
`SslmGpuSeqPrefillSchemaContentForG5Bridge`) live on the same shipped
`include/superslm/gpu_1p0.h` surface as the rest of the GPU API above — the
GPU-side twins of the CPU ABI's schema lookup, binding, prefill, and decode
calls, proven bit-identical against the CPU path (matching digest across 80
real decode steps) on the certified NVIDIA GPU. The same check passed
bit-identical on the certified AMD GPU as well (measured 2026-08-17 on the
Radeon RX 7900 XTX) — see [Certified platforms](platform-support.md).
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
