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

## The CPU consumer API (`sslm_*`) — under active development

`include/superslm/sslm_abi.h` is the contract: a from-scratch, engine-
agnostic C ABI for embedding SuperSLM's CPU inference path directly in
another process — a game engine's own tooling, for instance — without the
GPU handle types above. It declares 29 functions across the same lifecycle
shape as the GPU API (workspace and KV-pool sizing and creation, model
map/unmap, sequence and prefix lifecycle, decode, tokenize/detokenize,
stats) plus concepts the GPU API does not need: caller-owned workspace and
KV-pool memory (sized by the library, allocated by the caller, no hidden
allocation on the hot path) and shared-prefix "prefix" handles that let
more than one sequence reuse one prefilled prompt prefix.

As of this writing, 10 of the 29 declared functions are implemented and
tested: workspace and KV-pool sizing and lifecycle (`sslm_workspace_size`,
`sslm_kv_block_size`, `sslm_kv_pool_overhead_size`, `sslm_seq_state_size`,
`sslm_workspace_create`/`_destroy`, `sslm_kv_pool_create`/`_destroy`) and
model lifecycle (`sslm_model_map`/`_unmap`). The remaining declarations —
prefix lifecycle, sequence lifecycle including save/restore and adapter
binding, prefill, the batched decode step, tokenize/detokenize, and stats
— are specified in the header and reviewed, and are landing as the
implementation completes; the header is the stable contract that
implementation is being written against.

### Status causes

`sslm_status` is an 18-cause taxonomy in four groups: argument/precondition
rejections (a bad argument, a buffer too small, a misaligned buffer);
artifact/content rejections (a rejected artifact, an adapter that doesn't
match its base model, a restore whose content or KV shape doesn't match);
lifecycle rejections (a model, pool, or adapter with live handles still
attached to it; an adapter swap or sequence reset mid-token; a frozen
prefix reused; a KV pool with no room left); and numeric/domain rejections
(a token id out of range, a context length exceeded, or — a case distinct
from "out of range" — a legal decode-output token id with no tokenizer
entry, for the padded-vocabulary case).

### What schema-constrained generation adds

Once schema-constrained generation lands (see the README), the same
`sslm_prefill` and `sslm_decode_step` calls this ABI already declares carry
the constraint: a schema bound to a sequence forbids the model from
emitting a token that would break it, including on the token spans the
schema forces deterministically ("jump-forward") without a real decode
step. The determinism guarantee is unchanged by this — a schema-
constrained decode is exactly as reproducible, on the same platform, as an
unconstrained one.
