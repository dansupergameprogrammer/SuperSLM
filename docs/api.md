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

Six opaque handle types own the API's state: a context (`SslmGpuContext`,
one per device), a mapped model (`SslmGpuModelHandle`), a mapped LoRA
adapter (`SslmGpuAdapterHandle`), a decoding sequence
(`SslmGpuSequenceHandle`), and, from 1.13.0, a K/V page pool
(`SslmGpuKvPoolHandle`) and a shared prompt prefix (`SslmGpuPrefixHandle`);
see [Paged KV memory on the GPU](#paged-kv-memory-on-the-gpu-1130). Every fallible call takes the handles it needs
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
  magic rather than misread. From 1.13.0 a sequence made from a page pool
  saves `SLM6` instead, which `sslm_gpu_seq_restore` refuses with
  `SSLM_GPU_HOLDER_MODE_MISMATCH` (restore it with
  `sslm_gpu_seq_restore_into_pool`); a sequence made by
  `sslm_gpu_seq_create` keeps saving `SLM5`.
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
  refuses a `reserved` bit the library does not implement, a `max_tasks`
  outside `[0, 256]`, or a null `run` with `max_tasks` above 1
  (`SSLM_INVALID_ARGUMENT` on the CPU, `SSLM_GPU_PARALLEL_FOR_INVALID` on
  the GPU). Passing `NULL` clears the hook.
- The finish's logits step reads the hook. With `reserved` = 0 nothing else
  does: prefill, prefix prefill and every other call ignore it.
- **One-row projections, opt-in (CPU backend).** Setting
  `SSLM_PARALLEL_FOR_MATVEC` (bit 1) in `reserved` also splits every
  one-row (M = 1) projection across `run`. Each decode layer has five
  groups: q alone, k and v together, o alone, gate and up together, and
  down alone. A prefill or prefix-prefill call that admits exactly one token
  (whatever `count` was sent) splits each of its seven projections alone.
  For a group of N output rows over input width K (N·K weight bytes), the
  task count is the smallest of `max_tasks`, ceil(N / 64) and
  floor(N·K / 256 KiB), and at least 1. Each task then takes
  ceil(N / count) rows rounded up to a multiple of 64, and the last task
  takes what is left. So 256 KiB caps the task count by the group's total
  weight bytes. It is not a floor on each task: the last task can be
  smaller. At N = K = 896 (Qwen2.5-0.5B's q and o) and `max_tasks` 4 the
  split is 320, 320 and 256 rows, and the last task streams 229,376 bytes.
  A group calls `run` only when this gives at least two tasks; otherwise it
  runs on the calling thread. On Qwen2.5-0.5B at `max_tasks` 4, k + v
  (256 rows, 229,376 bytes) runs on the calling thread and the other four
  groups split, so there are 97 `run` calls per decode token (4 per layer
  plus the finish). A call admitting two or more tokens is unchanged. The GPU
  setter accepts the bit and ignores it: the GPU backend reads its hook only
  for the finish. The macro's presence in `parallel_for.h` is the
  compile-time test, and a library without the feature refuses the bit.
- Rows are split into contiguous blocks whose size is a function of the
  matrix shape, `max_tasks` and the 256 KiB constant only, and every row is one
  exact integer sum on one thread: tokens, save blobs and digests are
  identical with any hook, any `max_tasks` and either bit setting.
- A `run` that breaks exactly-once inside a decode layer fails the call with
  `SSLM_INVALID_ARGUMENT`. The sequence rests at the start of that layer,
  with the layer's five saturation counts put back (the layers before it
  keep theirs), and the next call resumes there. A prefill or prefix-prefill
  call that fails before admitting a token, for any reason and with or
  without the bit, leaves the saturation counts as they were before the
  call; a call that admits a prefix and then stops keeps that prefix's
  counts.
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
that model. The flag requests three new device buffers per mapped model:
the head table (`vocab_size × hidden_size` bytes: 136,134,656 B for
Qwen2.5-0.5B and 233,373,696 B for 1.5B), a `hidden_size × 4` B input row
and a `vocab_size × 8` B output row, 137,353,728 B and 234,595,328 B in
total. That sum is a lower bound on the new VRAM: the driver adds alignment
and allocation overhead that differs by GPU and driver (measured
+137,433,088 B and +234,627,072 B on an RTX 2080 SUPER; +137,629,696 B and
RX 7900 XTX). Nothing is allocated per sequence or per token. With the
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

### Paged KV memory on the GPU (1.13.0)

From 1.13.0 the GPU backend has the CPU's paged, shared-prefix K/V memory
([Paged KV memory](#paged-kv-memory-1120) describes the model). The
paged-KV block at the end of `include/superslm/gpu_1p0.h` is the contract;
this section summarizes it. Sequences made by `sslm_gpu_seq_create` are
unchanged: each keeps its own whole-cap buffer and saves `SLM5`, and the
GPU has no block pool and no no-budget pooled mode.

**Page geometry.** A GPU page is `B` positions: the artifact's
`kv_block_size` when it divides the context cap, otherwise the cap. That is
16 for converted artifacts at cap 4,096. It is not the CPU's
`sslm_kv_page_positions` (64 at Qwen2.5-0.5B, cap 4,096), so a cohort takes
a different number of pages on each backend. A page is
`layers · 2 · kv_heads · B · head_dim` bytes (the GPU's K/V is int8): 96 KiB
at Qwen2.5-0.5B and 224 KiB at Qwen2.5-1.5B. A budgeted holder reserves the
CPU's `R(budget) = min(ceil(budget / B) + 1, ceil(cap / B))` pages, with the
GPU's `B`.

**Pools.**

- `sslm_gpu_kv_pool_create(ctx, model, page_count, &pool)` creates one
  device buffer of `page_count` pages, zero-filled before it returns, with a
  host mirror of the same size. A pool above the adapter's pool cap is
  `SSLM_GPU_KV_POOL_TOO_LARGE`, before anything is allocated. The cap is
  D3D12's required single-resource size for the adapter, max(128 MiB,
  min(1/4 of its reported dedicated video memory, 2048 MiB)): 2 GiB on an
  adapter reporting 8 GiB or more, which is 21,845 pages at Qwen2.5-0.5B and
  9,362 at Qwen2.5-1.5B. An RTX 2080 SUPER reports 8,374,976,512 bytes, so
  its cap is 2,093,744,128 bytes: 21,298 and 9,128 pages. It is never above 2^32 bytes, the reach of the shaders'
  32-bit offsets. A single resource above D3D12's required size is the
  driver's to accept or refuse; the RX 7900 XTX's driver removes the device.
  `sslm_gpu_kv_pool_max_page_count(ctx, model, &pages)` reports the largest
  admissible page count. A pool the device or the host
  cannot allocate is `SSLM_GPU_ALLOCATION_FAILED` on a device that is not
  removed, and a smaller pool may be tried at once.
- `sslm_gpu_kv_pool_destroy` destroys an empty pool, refuses with
  `SSLM_MODEL_HAS_LIVE_SEQUENCES` while anything drawn from it lives, and
  accepts a null pool. `sslm_gpu_model_unmap` refuses while the model has
  live pools.
- `sslm_gpu_kv_pool_stats(pool, &out)` fills an `SslmGpuKvPoolStats` (set
  `struct_size`): `page_count`, `free_pages`, `shared_pages` and
  `page_positions` (`B`), the CPU's pool stats field for field. Size
  admissions from `free_pages`, not from handle counts.

**Budgeted sequences and prefixes.**

- `sslm_gpu_seq_create_budgeted(ctx, model, pool, budget, &seq)` takes
  `1 ≤ budget ≤ cap` and draws `R(budget)` pages, or returns
  `SSLM_GPU_KV_POOL_EXHAUSTED` with nothing drawn. Its cap is the model's
  context cap (the artifact's `context_cap`); unlike `sslm_gpu_seq_create`,
  the host does not choose it. It writes up to its limit,
  `min(origin + budget, cap)`. Below the cap, a decode at the limit is
  `SSLM_GPU_KV_BUDGET_EXCEEDED` with no row written; a prefill that reaches
  the limit commits the tokens below it, sets the ready flag when any
  landed (so the next bridge decode finishes the last of them and ignores
  its token argument), leaves no prefill snapshot, and returns
  `SSLM_GPU_KV_BUDGET_EXCEEDED`. The schema-content bridge's `*consumed`
  counts the committed tokens; the prompt bridge reports no count (it is
  the room the budget had left). At the cap the existing cap behaviour
  applies. The other sequence calls (embed, both
  decodes, `sslm_gpu_ready`, the prefill and finish bridges, schema binding,
  adapter binding, reset, release) take it as they take a dedicated
  sequence; its reset and release touch only its own pages.
- `sslm_gpu_prefix_begin(ctx, model, pool, budget, &prefix)` makes a
  budgeted prefix (origin 0). `sslm_gpu_prefix_set_schema` binds a schema
  before any content. `sslm_gpu_prefix_prefill(ctx, prefix, tokens, count,
  kind, dispatch_budget, &consumed)` prefills through the prompt (`kind` 0)
  or schema-content (`kind` 1) path, base-only, with the bridges' partial
  consumption. `sslm_gpu_prefix_freeze` freezes it and
  `sslm_gpu_prefix_release` releases it.
- `sslm_gpu_prefix_begin_from(ctx, parent, budget, &prefix)` continues a
  frozen parent in the parent's pool: the parent's full pages shared, its
  partial page and its state copied, origin the parent's length.
- `sslm_gpu_seq_adopt_prefix(ctx, seq, prefix)` adopts a frozen prefix onto
  a pooled sequence with the CPU adopt's field and schema rules: full pages
  mapped shared, the tail rows copied. Every check runs before any state
  change. A prefix of another model is `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`;
  a dedicated sequence is `SSLM_GPU_HOLDER_MODE_MISMATCH`; a prefix that is
  not frozen, or of another pool of the same model, is
  `SSLM_GPU_PREFIX_REJECTED`; a schema mismatch is
  `SSLM_GPU_PREFIX_SCHEMA_MISMATCH`. One failure comes after the checks: if
  the upload of the prefix's partial last page fails, the adopt returns the
  fault's status and leaves the sequence reset (its earlier history gone,
  its bind permission granted again).
- A shared page is freed when the prefix and every holder that maps it have
  let go of it.
- A handle of another context or model, or a budget outside `[1, cap]`, is
  `SSLM_SEQUENCE_KV_BUFFER_MISMATCH` with a null out-handle. Every
  out-handle is null on every status but `SSLM_OK`.

**Save and restore.** A pooled sequence's `sslm_gpu_seq_save` writes `SLM6`
(little-endian; offsets as the 1.13.0 writer lays them out):

| Offset | Field |
|---|---|
| 0–119 | the `SLM5` header, magic `SLM6` |
| 120–131 | the `SLM5` tail (schema index, walk state, ready flag) |
| 132 | `kv_mode`, u32: 1 (budget; every `SLM6` is a budget blob) |
| 136 | `budget`, i32 |
| 140 | `origin`, i64 |
| 148 | the residual (the header's hidden-codes size) |
| then | `kv_positions`, u64: `L' = context_length + (1 if mid-token else 0)` |
| then | rows `[0, L')`, `[layer][K|V][kv_head][position][head_dim]` |

Mid-token means a layer index strictly between 0 and the layer count: a
sequence resting after a completed token carries no row at
`context_length`. As with `SLM5`, a save into a null or too-small buffer
writes the size it needs to `*out_blob_size` and returns the status the
`SLM5` save returns then (`SSLM_DEVICE_LOST`).

- `sslm_gpu_seq_restore_into_pool(ctx, model, pool, blob, size,
  prefix_or_null, &seq, &out_shared_pages)` restores an `SLM6` blob into a
  fresh pooled sequence with its budget and origin. The prefix is a sharing
  hint: when it matches, the restored sequence maps its pages again and
  `*out_shared_pages` counts them; a null or non-matching prefix gives a
  private restore with `*out_shared_pages = 0`. Its refusals, in order, each
  changing nothing: malformed handles and arguments; `SSLM_BUSY` while any
  sequence is Submitted; a blob whose magic is not `SLM6` (`SLM5`, `SLM4`
  or any other bytes), `SSLM_GPU_HOLDER_MODE_MISMATCH`; the body, with `sslm_gpu_seq_restore`'s
  statuses; `SSLM_GPU_KV_POOL_EXHAUSTED`, the first draw rolled back.
- `sslm_gpu_seq_restore` is unchanged except that it refuses `SLM6` with
  `SSLM_GPU_HOLDER_MODE_MISMATCH`, allocating nothing.
- 1.12.0 and earlier refuse `SLM6` on its magic with
  `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`. GPU and CPU blobs are separate
  formats; neither backend restores the other's.
- `sslm_gpu_prefix_save(ctx, prefix, blob, &size)` (1.16.0, unreleased) writes a frozen
  prefix as an `SPX1` blob, the CPU's precached-prefix layout with backend 1
  and the adapter's vendor and device IDs as producer. It reads the pool's
  host mirror and submits nothing. An unfrozen prefix, or one not at rest
  after a whole token, is `SSLM_GPU_PREFIX_REJECTED`; a short buffer is
  `SSLM_OUTPUT_BUFFER_TOO_SMALL` with the size written. Exported, and
  unreleased until 1.16.0.
- `sslm_gpu_prefix_restore(ctx, model, pool, blob, size, parent_or_null,
  &prefix, &shared)` (1.16.0, unreleased) validates the whole blob first (arguments,
  Busy, size and magic, digest, backend, producer
  (`SSLM_RESTORE_PRODUCER_MISMATCH`), model (`SSLM_RESTORE_MODEL_MISMATCH`),
  then kv precision, schema and field domains, each
  `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`), then draws `ceil(L/B) - s` pages
  (`SSLM_GPU_KV_POOL_EXHAUSTED`, nothing drawn) and shares the parent's `s`
  leading pages when it is a frozen prefix of this pool holding exactly the
  blob's origin with equal rows; otherwise `s = 0`. A blob restores only on
  the engine version and adapter that wrote it. Exported, and unreleased
  until 1.16.0.

**In flight.** `sslm_gpu_kv_pool_create`, `sslm_gpu_prefix_prefill`,
`sslm_gpu_prefix_begin_from`, `sslm_gpu_seq_adopt_prefix`,
`sslm_gpu_seq_restore_into_pool` and `sslm_gpu_prefix_restore` submit GPU work: they are serialized as
every submitting call is (below), and each returns `SSLM_BUSY`, changing
nothing, while any sequence in the process is Submitted. Every guarded call
on a pooled sequence has the same any-sequence scope, and a guarded call on
a dedicated sequence, a model or an adapter returns `SSLM_BUSY` while a
pooled sequence is Submitted. The other paged calls submit nothing and are
never refused Busy. (`sslm_gpu_prefix_restore` is exported, and unreleased
until 1.16.0.)

**Speed.** On the NVIDIA RTX 2080 SUPER (one card, one driver),
single-sequence decode through a pooled sequence's page table was no more
than 5% slower than a dedicated sequence of the same build on both rows
measured: at Qwen2.5-0.5B, cap 4,096, a cohort member (budget 512, an
adopted 1,200-token prefix) at width 1,712 read about 1% faster; at
Qwen2.5-1.5B, cap 32,768, a pooled sequence restored without a prefix at
width 16,384 read about 3.7% slower. It is a bar, not a speed-up. Other
widths and caps, other pool occupancies, batched decode and the AMD card
are not measured. That is the only GPU speed statement; the
[1.13.0 release note](releases/1.13.0.md) has the readings.

### Thread safety

Calls against **different** sequence handles are safe to make from
different threads concurrently. Every call that resets or submits work on a
device, and `sslm_gpu_ready` with `block` set, must be externally
serialized by the caller against every other such call on that device; the
API does not build an internal queue lock. Up to 1.12.0 the header named
the two decode calls and the blocking `sslm_gpu_ready`; from 1.13.0
`gpu_1p0.h` (*In-flight work*) lists every such call and the device it
uses: the decode calls, the blocking `sslm_gpu_ready`, the prefill bridges,
the prefix prefill, `sslm_gpu_prefix_begin_from`, `sslm_gpu_seq_adopt_prefix`
and `sslm_gpu_seq_restore_into_pool` on the process's submission device;
`sslm_gpu_model_map`, `sslm_gpu_adapter_map`, `sslm_gpu_seq_create`,
`sslm_gpu_kv_pool_create` and the finish of a model mapped with
`SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE` on the context's own device; and
`sslm_gpu_seq_restore` and `SslmGpuSeqDecodeStepForG5Bridge` on both.
`sslm_gpu_ready` with `block` clear may run beside a submitter.
`sslm_gpu_context_set_host_parallel_for` is serialized with the context's
calls too.

Within that rule the library keeps two promises (1.13.0). It never resets a
device's command allocator while work submitted on it has not been observed
complete. And a guarded call that would need to returns `SSLM_BUSY` and
changes nothing, or `SSLM_DEVICE_LOST` once a submission is stranded (its
completion cannot be confirmed), until the process restarts. So a second
call that reaches the submission device while an earlier submission is
unobserved (a decode of sequence A, then of sequence B before
`sslm_gpu_ready` has reported A) returns `SSLM_BUSY`; before 1.13.0 it was
undefined. Finish each submission before the next, or decode several
sequences in one `sslm_decode_step_batch_gpu` call. The header states the
scope of each guard and the order of every guarded call's refusals; for
example, `SslmGpuSeqPrefillSchemaContentForG5Bridge` on a Submitted
sequence returns `SSLM_BUSY` before its unreachable-first-token
`SSLM_SEQUENCE_REJECTED`, as it has since 1.6.0.

Two threads driving the *same* sequence handle concurrently is not a
supported use. From 1.13.0 the four decode entry points,
`sslm_decode_step_gpu`, each slot of `sslm_decode_step_batch_gpu`,
`SslmGpuSeqDecodeStepForG5Bridge` and `SslmGpuSeqFinishTokenForG5Bridge`,
take a per-handle claim before they read the handle and hold it until they
return, so two of them raced on one handle, in any pairing, stay
memory-safe: one runs and the other returns `SSLM_BUSY` having changed
nothing (a batch slot reads `SSLM_BUSY`). The claim lasts one call, so a
thread that submits, drains and finishes takes it afresh at each decode
call. `sslm_gpu_ready` takes no claim: two `sslm_gpu_ready` calls on one
handle, or one beside a decode entry point on that handle, are the caller's
to serialize, as are other calls raced on one handle (embed, the prefill
bridges, reset, save, schema binding, release).

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

Six cover paged K/V (1.13.0, ordinals 24 to 29; see
[Paged KV memory on the GPU](#paged-kv-memory-on-the-gpu-1130)):
`SSLM_GPU_KV_BUDGET_EXCEEDED`, a budgeted sequence or prefix writing at its
limit below the cap (a prefill commits the tokens below the limit first); `SSLM_GPU_KV_POOL_EXHAUSTED`, a pool too short for a
create, prefix, `begin_from` or restore, with nothing drawn;
`SSLM_GPU_KV_POOL_TOO_LARGE`, a pool above the adapter's pool cap (1.13.1; 2^32 bytes in 1.13.0);
`SSLM_GPU_HOLDER_MODE_MISMATCH`, a pooled and a dedicated holder mixed (an
adopt onto a dedicated sequence, an `SLM6` blob given to
`sslm_gpu_seq_restore`, a blob whose magic is not `SLM6` given to
`sslm_gpu_seq_restore_into_pool`); `SSLM_GPU_PREFIX_REJECTED`, a prefix verb
refused on the prefix's state (not frozen, another pool of the same model, prefill after
freeze, a schema bind it can no longer take); and
`SSLM_GPU_PREFIX_SCHEMA_MISMATCH`, an adopt whose schema binding does not
match the prefix's.

One covers precached prefixes (1.16.0, unreleased; ordinal 30):
`SSLM_RESTORE_PRODUCER_MISMATCH`, an `SPX1` blob given to
`sslm_gpu_prefix_restore` that another engine version wrote, or that was
written on another adapter (its vendor or device ID differs from the
context's). Nothing is drawn. The status is exported, and unreleased until
1.16.0, with the GPU twins.

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
GPU handle types above. It declares and implements 51 functions (38 before 1.12.0) across the
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
  capacities (`sslm_seq_state_size` is an upper bound across live sequence states and
  every save format; in 1.12.0 it is 20 bytes larger than in 1.11.0, see
  [Paged KV memory](#paged-kv-memory-1120));
  `sslm_workspace_create`/`_destroy`
  and `sslm_kv_pool_create`/`_destroy` take those buffers and hand back a
  handle. A workspace is reusable across a sequence of calls but is not
  safe to share between two calls running concurrently; a caller driving
  multiple sequences concurrently needs one workspace per concurrently
  active call.
- **Prefix** (shared prompt prefix): `sslm_prefix_begin` / `sslm_prefix_release`,
  `sslm_prefix_prefill` (runs the shared prefix's own forward pass once),
  `sslm_prefix_freeze` (locks it for adoption by sequences; since 1.12.0 the first freeze
  also returns the prefix's unused pages to its pool). `sslm_prefix_begin_budgeted` and
  `sslm_prefix_begin_from` (1.12.0) make budget-mode prefixes; see
  [Paged KV memory](#paged-kv-memory-1120).
- **Sequence**: `sslm_seq_create` / `sslm_seq_release`, `sslm_seq_reset`,
  `sslm_seq_adopt_prefix` (attaches a frozen prefix, so its forward pass is
  never repeated per sequence), `sslm_seq_save` / `sslm_seq_restore`
  (serializes a sequence's full state — including its schema binding and
  DFA walk state, see below — to a caller buffer and back). In 1.12.0 a
  sequence made by `sslm_seq_create` (or restored from a no-budget blob)
  still writes `SSB5`, byte-equal to 1.11.0's blob for the same state, and a
  budget-mode sequence writes `SSB6` (see
  [Paged KV memory](#paged-kv-memory-1120)); restore accepts both.
  `sslm_seq_create_budgeted` and `sslm_seq_restore_shared` (1.12.0) are the
  budget-mode create and the restore that can re-share a prefix's pages.
  Since v1.9.0 the no-budget save writes
  `SSB5` blobs: the `SSB4` layout plus the four per-site saturation counts
  (K/V landing, K channel landing, RoPE Q, RoPE K) that the sequence's
  saturation total sums, so a restored sequence keeps its per-site counts.
  `SSB4` (v1.2.1 to v1.8.1) serializes the residual unconditionally whenever
  `hidden_size > 0` (a ready-for-logits sequence, `layer_index == 0`, carries
  a real residual and is no longer saved with it silently dropped — the fixed
  1.2.0 defect), and carries an explicit `ready_for_logits` field in the
  header rather than inferring it on restore; `SSB5` keeps both. Restore
  accepts shipped `SSB4`, `SSB3` (v1.2.0) and `SSB2` blobs read-only. Those
  record only the saturation total, so a sequence restored from one has its
  saved total and per-site counts of 0 until it is reset. A legacy `SSB3` blob resting at
  the one state the 1.2.0 defect could produce is rejected with
  `SSLM_RESTORE_RESIDUAL_LOST` rather than silently restored wrong. For
  `SSB5` and older, restore accepts a buffer whose size is at
  least the encoded blob size, including a buffer sized by
  `sslm_seq_state_size`, and ignores trailing capacity; an `SSB6` blob must
  be passed with exactly its encoded size. It also rejects anti-LM
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
  validated before forward work begins. `sslm_model_map` refuses, with
  `SSLM_ARTIFACT_REJECTED`, an artifact whose DGC1 scale derives constants that
  damped-greedy decode would refuse (since v1.9.0 this includes the i-exp peak
  check decode applies), so a mapped artifact that offers damped greedy can
  always decode with the constants `sslm_decode_params_init` derives. Under damped-greedy mode, six more fields
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
  the finish is serial on the calling thread as before. With
  `SSLM_PARALLEL_FOR_MATVEC` set, each decode layer's one-row projections,
  and a prefill call that admits exactly one token, split across it too.
  Tokens are identical either way. A hook that breaks its exactly-once contract fails that call
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
  argmax), `kv_blocks_resident` (always 1, in every release; page accounting
  is `sslm_kv_pool_stats` and `sslm_seq_kv_stats`), and `schema_accepting` (1 iff a
  schema is bound and the sequence's current parse state is one where
  stopping is valid; 0 if not, and 0 when no schema is bound).

### Paged KV memory (1.12.0)

Since 1.12.0 a KV pool stores K/V in fixed-size **pages** rather than one
whole-context block per sequence. Each sequence and prefix addresses its K/V
through a page table, and a sequence that adopts a prefix can **share** the
prefix's full pages instead of copying them. Output is unchanged: tokens and K/V
values are bit-identical to 1.11.0 on every path, whichever verbs a host uses, as
read on Windows x64 with MSVC. The 1.11.0 verbs keep their signatures and their
results (the changes are listed under [Changed in 1.12.0](#changed-in-1120)
below), but a no-budget `sslm_seq_save` is slower than in 1.11.0 (see
[Speed claims in 1.12.0](#speed-claims-in-1120)); the memory savings come only
from the new budget-mode verbs.

The paged surface is declared in `include/superslm/sslm_abi_functions.inc`, and
`SSLM_HAS_PAGED_KV_ABI` is defined whenever it is present.

**Page geometry.** `B`, the positions per page, is a pure function of the
model's head dimension `D`, context cap and K/V element width. It is the
smallest power of two of at least 16 whose page run, `B · D · element_bytes`,
reaches 4 KiB (when `D ≤ 64` and the cap is at most 4,096) or 8 KiB
(otherwise), and the cap itself when that is larger than the cap. The
artifact's stored `kv_block_size` does not select it. Where `B` does not
divide the cap, a holder's last page is partial: its slack positions are never
addressed, since writes stop at the cap.

- `sslm_kv_page_positions(model)` returns `B` (0 on a null model). It is 64 at
  Qwen2.5-0.5B with cap 4,096 and at Qwen2.5-1.5B with cap 32,768, and 128 at
  Qwen2.5-0.5B with cap 32,768.
- `sslm_kv_page_size(model)` returns the bytes per page,
  `layers · 2 · kv_heads · B · head_dim · element_bytes` (0 on a null model
  or overflow). At Qwen2.5-0.5B with cap 4,096 that is 384 KiB (768 KiB at cap
  32,768), and at Qwen2.5-1.5B with cap 32,768 it is 896 KiB.
- `sslm_kv_pages_for_budget(model, budget)` returns `R(budget)`, the pages a
  budget holder reserves: `min(ceil(budget / B) + 1, ceil(cap / B))`. It
  returns 0 for a budget outside `[1, cap]`. The extra page covers a prefix
  whose length is not a multiple of `B`.

**Pools.** Three verbs build a pool over caller memory, and every holder verb,
old or new, takes any of them. The first two build page pools of the same kind.

- `sslm_kv_pool_create(model, buf, size, block_count, &pool)` builds
  `block_count · ceil(cap / B)` pages in the same buffer size as before at a
  cap `B` divides (one block is exactly `ceil(cap / B)` pages). At any other
  cap the buffer is larger by the slack of each block's partial last page,
  which `sslm_kv_pool_overhead_size` carries, so a pool sized as
  `N · sslm_kv_block_size + sslm_kv_pool_overhead_size(N)` admits N no-budget
  holders. A count whose page total exceeds `UINT32_MAX` is refused with
  `SSLM_INVALID_ARGUMENT`, before the buffer-size check (at cap 32,768, any
  `block_count` of 8,388,608 or more at `B = 64`, and of 16,777,216 or more at
  `B = 128`; computed from the page rule).
- `sslm_kv_pool_create_contiguous(model, buf, size, block_count, &pool)` builds
  the **contiguous pool option** (CPU only): `block_count` pages, each the
  whole context cap, which is 1.11.0's contiguous layout (one page per
  sequence). Size the buffer with
  `sslm_kv_pool_contiguous_size(model, block_count)`, which returns
  `block_count · (sslm_kv_block_size + 8) + 32` bytes, and 0 on a null model, a
  zero count or an overflow. The create refuses in `sslm_kv_pool_create`'s
  order and alignment rule. Every verb works on it as on a page pool with
  `B = cap`: a no-budget or budget holder takes one page, adopt and
  `sslm_prefix_begin_from` copy and share no page, and saves are the same in
  every pool kind, so blobs restore across pool kinds. It gives up what a page
  pool gives a budget holder, sharing a prefix's pages and reserving by
  budget, and it is outside the claims for reset and adopt cost, shared
  prefixes and residency below. `sslm_kv_page_positions`, `sslm_kv_page_size`
  and `sslm_kv_pages_for_budget` answer for the model's derived `B`, not for a
  contiguous pool's; `sslm_kv_pool_stats` reports the pool's own. A copy adopt
  between a contiguous pool and a page pool of one model is
  `SSLM_INVALID_ARGUMENT` wherever their page sizes differ, so move state
  between pool kinds by save and restore.
- `sslm_kv_page_pool_create(model, buf, size, page_count, &pool)` builds a
  pool of `page_count` pages. `buf_size` must be at least
  `page_count · sslm_kv_page_size + sslm_kv_page_pool_overhead_size`, and the
  buffer `SSLM_ABI_ALIGNMENT_BYTES`-aligned. Its refusals come in
  `sslm_kv_pool_create`'s order: arguments, a zero count, overflow,
  `SSLM_BUFFER_TOO_SMALL`, then `SSLM_MISALIGNED_BUFFER`. Every `page_count`
  up to `UINT32_MAX` is admissible by count.
- Both overhead verbs keep the block verb's convention (0 on a null model,
  `SIZE_MAX` where no buffer suffices). The pool's per-page bookkeeping (a
  free-list entry, a reference count, a written flag and a shared count, 13
  bytes per page) is allocated on the library heap at pool create, as the
  1.11.0 free list was, not in the overhead part of the caller's buffer. A heap
  failure there returns `SSLM_ALLOCATION_FAILED`.
- `sslm_kv_pool_destroy` destroys either kind and refuses with
  `SSLM_POOL_HAS_LIVE_HANDLES` while any handle made from the pool lives.

**Two kinds of holder.** Every sequence and prefix is one of these, fixed when
it is made:

- **No-budget** (`whole_reserve`): made by `sslm_seq_create`,
  `sslm_prefix_begin`, or a restore of a no-budget blob (`SSB5` and older). It
  reserves `ceil(cap / B)` pages, one old block plus the slack of its partial
  last page where `B` does not divide the cap, and behaves as in 1.11.0. Its
  limit is the cap.
- **Budget**: made by `sslm_seq_create_budgeted(model, &pool, budget, &seq)`,
  `sslm_prefix_begin_budgeted(model, &pool, budget, &prefix)`,
  `sslm_prefix_begin_from(parent, budget, &prefix)`, or a restore of a
  budget-mode `SSB6`. It declares `1 ≤ budget ≤ cap`, the positions it may
  write itself, and reserves `R(budget)` pages from the pool when it is made.
  A budget outside `[1, cap]`, or a pool of another model, is
  `SSLM_INVALID_ARGUMENT`. A pool with fewer free pages than the reservation
  is `SSLM_KV_POOL_EXHAUSTED`, with nothing drawn.

A holder's **origin** is the position its own writes start at: 0 after create
or reset, the prefix's length after `sslm_seq_adopt_prefix`, and the parent's
length for `sslm_prefix_begin_from`. A budget holder's **limit** is
`min(origin + budget, cap)`.

- Prefill and decode map pages only from the holder's own reservation. They
  never take a page from the pool, never take the pool's lock and never
  allocate, so no write below the limit is refused for lack of pages.
- A write at or past the limit, when the limit is below the cap, returns the
  new **`SSLM_KV_BUDGET_EXCEEDED`** (ordinal 29). Prefill admits the tokens
  below the limit and stops, with the same partial-consumption contract as the
  cap. Decode refuses at the token boundary with no row written, with the same
  batch semantics as the cap check. A sequence resting ready for logits at the
  limit still emits its one ready token, so a sequence resting ready at length
  `c` emits exactly `1 + (limit − c)` tokens before the refusal. When the
  limit is the cap, `SSLM_CONTEXT_CAP_EXCEEDED` fires instead, as before, so a
  no-budget holder never sees the new status.
- `sslm_seq_reset` keeps the reservation and sets the origin to 0. It never
  draws from the pool, so reset and reuse are never refused.
- The first `sslm_prefix_freeze` of any prefix, of either kind, returns its
  unused pages to the pool, so a frozen prefix holds `ceil(length / B)` pages.
  Later freezes return `SSLM_OK` and change nothing.
- `sslm_prefix_begin_from` makes a budget-mode child of a frozen parent of
  either kind (an unfrozen parent is `SSLM_INVALID_ARGUMENT`). It draws
  `R(budget)` from the parent's pool, shares the parent's full pages, copies
  its partial last page, and copies the parent's state, including its schema
  binding and walk state. A child of a non-empty parent cannot bind a schema
  of its own (`sslm_prefix_set_schema` refuses once a prefix has content), so
  only a schema bound at the root prefix carries schema content.

**Adopt.** The adopting sequence's kind decides, never the prefix's:

- A budget sequence **shares**: it maps the frozen prefix's full pages
  read-only and copies at most one partial page into its own reservation. It
  never draws from the pool. A prefix of another pool is
  `SSLM_INVALID_ARGUMENT`, checked with the model check before anything
  changes. This holds at a cap of more than one page: at a cap that fits in one
  page a holder is a single page, so the adopt copies it and shares none.
- A no-budget sequence **copies** the prefix's written rows into its own
  pages (proportional to the prefix's length, not to the cap), and may adopt
  from another pool of the same model, as before.
- The other refusals are 1.11.0's: an unfrozen prefix or another model is
  `SSLM_INVALID_ARGUMENT`, and a prefix whose schema progress the sequence's
  schema does not match is `SSLM_PREFIX_SCHEMA_MISMATCH`, which, as before,
  also leaves the sequence unable to bind a schema until it is reset.
- Prefixes stay base-only, so a shared page is valid under any adapter the
  adopting sequence binds.

**Page lifetime and admission.** A shared page is freed when the prefix and
every holder that maps it have released, reset or re-adopted. Releasing a
prefix therefore frees only the pages no other holder still maps, and a pool
can stay full after its prefixes are released. Read `free_pages` from
`sslm_kv_pool_stats` to size admissions, not the handle count.

- In a pool where every handle was made by a no-budget verb, `block_count = N`
  still admits N handles, as in 1.11.0. A frozen prefix now returns its unused
  pages, so such a pool can admit more handles than 1.11.0 did, never fewer.
- Once a budget-mode handle is in a pool, budget accounting governs admission:
  a budget holder that shares a released prefix's pages keeps them counted
  against the pool until it resets, adopts or releases.

**Save and restore.**

- A no-budget sequence's `sslm_seq_save` writes `SSB5`, byte-equal to the blob
  1.11.0 writes for the same state, so 1.9.0 to 1.11.0 read it. Its size is
  1.11.0's: the whole block.
- A budget sequence writes **`SSB6`**, which carries the rows `[0, L')` of its
  live length, the span of an adopted prefix included. Releases before 1.12.0
  refuse it on its magic.
- Restore reads `SSB6`, `SSB5`, `SSB4`, `SSB3` and `SSB2`.

The `SSB6` layout (little-endian):

| Offset | Field |
|---|---|
| 0–155 | the `SSB5` fixed header, unchanged (magic `SSB6`) |
| 156 | `kv_mode`, u32: 0 no-budget, 1 budget |
| 160 | `budget`, i32: the declared budget (the cap for a no-budget holder) |
| 164 | `origin`, i64 |
| 172 | the residual (`hidden_size` bytes when `hidden_size > 0`), then the anti-LM history, as in `SSB5` |
| then | `kv_positions`, u64: `L' = context_length + (1 if mid-token else 0)` |
| then | rows `[0, L')` in the flat layout with the cap replaced by `L'`: `[layer][K|V][kv_head][position][head_dim]` |

So an `SSB6` blob is `172 + residual + 4 · history + 8 + L' · bytes_per_token`
bytes, where `bytes_per_token = layers · 2 · kv_heads · head_dim ·
element_bytes`. A sequence with no tokens saves no K/V bytes (its header,
residual and history are saved as before). At a mid-token save, the layers
the sequence has not yet reached at position `context_length` are written as
zero bytes, so a blob depends only on the sequence's valid state, never on
page identities or reuse history.

Restore keeps the holder's contract:

- A budget-mode `SSB6` restores as the holder it was saved from: the same
  budget, origin and mode, so the same limit, the same remaining token count
  and, after its first reset or adopt, the same reservation.
- `SSB5` and older restore as no-budget holders with origin 0. A no-budget
  sequence's origin is therefore 0 after a save and restore, as in 1.11.0;
  only `sslm_seq_kv_stats` reads it, and its limit, token count and
  reservation do not depend on it.
- Every field is validated before any page is drawn. `SSB6` adds `kv_mode` in
  {0, 1}, `1 ≤ budget ≤ cap` (equal to the cap for a no-budget blob),
  `0 ≤ origin ≤ context_length ≤ min(origin + budget, cap)`, `kv_positions`
  equal to `L'`, and an **exact** blob size. An `SSB6` blob must therefore be
  passed with exactly the `*n` its save returned, not the
  `sslm_seq_state_size` capacity; older formats still accept trailing
  capacity. A refusal is `SSLM_INVALID_ARGUMENT` with nothing drawn.
- The per-site saturation counts restore verbatim from `SSB6` and `SSB5`; no
  relation between them and the total is checked.

`sslm_seq_restore_shared(model, &pool, blob, size, prefix_or_null, &seq,
&out_shared_pages)` is `sslm_seq_restore` with a sharing hint, and
`sslm_seq_restore` is this verb with no prefix. A budget-mode `SSB6` re-shares
the prefix's full pages, and `*out_shared_pages` receives their count, only when
all of these hold: the prefix is frozen, of this model and this pool, its length
equals the blob's origin, and its rows over the shared span compare byte-equal to
the blob's own rows (an exact compare, not a hash). Otherwise the restore is
private, returns `SSLM_OK` with `*out_shared_pages = 0`, and also draws pages
to hold the prefix span, which go back to the pool at the sequence's first
reset, adopt or release. A no-budget blob ignores the handle.
`out_shared_pages` may be `NULL`. A host that restores many sequences of one
cohort should pass each one its live prefix handle: without the handle each
restored sequence holds the prefix span privately until it resets or
re-adopts.

**Sizing.** `sslm_seq_state_size` stays an upper bound for every blob of
either format. Its fixed part is 180 bytes, against 160 in 1.11.0 (the `SSB6`
header's 16 more bytes, and the 8-byte `kv_positions` where `SSB5` has a
4-byte `kv_block_count`), so it is 20 bytes larger at every model.

**Diagnostics.** Both stats verbs take a caller struct whose `struct_size` the
caller sets to its `sizeof`; any other value is `SSLM_INVALID_ARGUMENT`.

- `sslm_kv_pool_stats(pool, &out)`: `page_count`, `free_pages`,
  `shared_pages` (the distinct pages at least one live holder maps as shared)
  and `page_positions` (`B`), read under the pool's lock.
- `sslm_seq_kv_stats(seq, &out)`: `mode`, `origin`, `budget`, `limit`,
  `context_length`, `mapped_private_pages` (including the pages a private
  restore drew for the prefix span), `shared_pages`, `reserve_pages` and
  `materialized_pages`. The holder's reservation is `reserve_pages +
  mapped_private_pages`. It takes the sequence's lifecycle lock.
- `sslm_stats`' `kv_blocks_resident` stays the constant 1.

**Threads.** Pages are reference-counted only by lifecycle verbs, under the
pool's lock; prefill and decode take no pool lock.

- A frozen prefix's pages are never written again, so any number of threads
  may adopt from it, `begin_from` it, and decode sequences that share it.
- Releasing a prefix while its sharers decode, reset or release is safe.
- Releasing a prefix while an adopt, a `begin_from` or a sharing restore is
  reading it is the caller's undefined behaviour, as for adopt in 1.11.0.
- `sslm_seq_kv_stats` must not race a prefill of the same sequence (the
  one-caller-per-sequence contract).
- Prefix construction is single-threaded, as before.

#### Changed in 1.12.0

These apply to every host, including one that calls no new verb:

- `sslm_seq_state_size` is 20 bytes larger (above).
- `sslm_kv_pool_create` refuses a `block_count` whose page total exceeds
  `UINT32_MAX`, with `SSLM_INVALID_ARGUMENT`.
- `sslm_kv_pool_overhead_size` is larger at a cap the page size does not
  divide, by `block_count · (ceil(cap / B) · B − cap)` positions of K/V;
  `sslm_kv_block_size` and the no-budget save blob are unchanged at every cap.
- `sslm_seq_save` of a no-budget sequence is slower than in 1.11.0 at every
  length measured, with the same bytes: 1.14 to 2.00 times as long on the
  release code (24 of 24 readings across four captures; readings, not resolved
  changes; see [Speed claims in 1.12.0](#speed-claims-in-1120)).
- The first `sslm_prefix_freeze` returns the prefix's unused pages to the
  pool, so a pool can admit more handles than before, never fewer.
- `sslm_seq_reset` no longer zero-fills the sequence's K/V memory. No read
  reaches the old rows: reads stop at the live length, and a no-budget save
  writes zeros past it, so the saved blob is unchanged.
- Restore refuses a blob, of any format, that is mid-token at the context cap
  (no save produces one), with `SSLM_INVALID_ARGUMENT`.
- `sslm_seq_restore` reads `SSB6`.

#### Speed claims in 1.12.0

The claims are for Windows x64, CPU only, at ordinary background load, on the
AVX2 tier. The [1.12.0 release note](releases/1.12.0.md) carries the measured
figures.

- **Single-conversation decode.** Attention reads K/V one page run at a time.
  Decode of one sequence over paged K/V is no more than 5% slower than the
  contiguous cache of the same build (the contiguous pool option), at
  Qwen2.5-0.5B up to its cap of 4,096 and at Qwen2.5-1.5B up to width 18,682.
  It is measured with whole-token calls and no host parallel-for hook, and it
  is not a comparison with 1.11.0.
- **Reset and adopt.** Their cost does not grow with the context cap, except
  that at `head_dim ≤ 64` a cap above 4,096 doubles the page and its one-page
  tail copy (0.5B: 384 KiB at caps up to 4,096, 768 KiB above), at a cap of
  more than one page: the same live length at cap 32,768 costs no more than a
  fixed bound of 2.0 times its cost at cap 4,096, measured at Qwen2.5-0.5B with
  the page size pinned equal (64 positions) at both caps, so the doubled page
  was not timed.
- **Batched decode.** `sslm_decode_step` over several sequences works and stays
  bit-identical to each sequence's solo run. No batched speed is claimed.
- **Lifecycle verbs against 1.11.0.** The release note's lifecycle table states
  the changes its comparison resolves (a factor of at least 4); all 22 are
  speed-ups. No-budget save is read separately, under *Not promised* below.

#### Not promised in 1.12.0

- **Memory savings for no-budget verbs.** A host that calls only the 1.11.0
  verbs holds one block's pages per handle, as before. Sharing and sizing by
  live length apply to budget-mode holders only.
- **No-budget save size.** A no-budget sequence's `SSB5` is still the whole
  block, so that 1.11.0 can read it.
- **No-budget save speed.** A no-budget `sslm_seq_save` takes 1.14 to 2.00 times
  as long as in 1.11.0 on the release code, at every length measured (24 of 24
  readings across four captures on the release machine; readings, not resolved
  changes: the release note's lifecycle comparison resolves a change only at 4
  times or more), with the same bytes. A later release is planned to remove the
  slowdown. (1.13.0 writes a no-budget block of 16 MiB or more with streaming
  stores on x64, with the same bytes; its
  [release note](releases/1.13.0.md) states what is measured.)
- **A higher context cap.** The cap is not removed: RoPE tables and the
  attention score scratch are still sized by it, and a write at the cap still
  fails. Paging only makes a high cap cheaper to reserve.
- **Speed beyond the claims above.** Nothing is claimed under load, on the
  GPU, or for batched decode.
- **GPU paging.** The GPU API (`SslmGpu*`, `sslm_gpu_*`) is unchanged: GPU
  sequences keep one whole-cap buffer each and have no prefix or adopt verbs.
  GPU paging is planned for a later release. (1.13.0 adds it: see
  [Paged KV memory on the GPU](#paged-kv-memory-on-the-gpu-1130).)
- **Automatic prefix matching.** Prefixes are explicit handles; the library
  does not find shared prefixes by itself.
- **Adapter-shaped prefixes.** Prefixes are computed without an adapter. An
  adapter shapes only the tokens after the prefix.
- **Shared reads.** Sequences that share pages each still read them in their
  own attention step; there is no batched shared-prefix attention.
- **Tokenization at a prefix seam.** The verbs take token ids. Splitting text
  so that the prefix's tokens and the continuation's tokens equal the whole
  text's tokens is the host's job.

### Frozen-prefix save and restore (1.14.0 CPU ABI)

`SSLM_HAS_PREFIX_SAVE_ABI` marks the availability of `sslm_prefix_save` and
`sslm_prefix_restore`. Save accepts a frozen prefix and writes canonical `SPX1`
bytes. Use the two-call buffer convention: a null or short buffer returns
`SSLM_BUFFER_TOO_SMALL` and sets the exact byte count in `*n`. An unfrozen
prefix is refused with `SSLM_INVALID_ARGUMENT`. A frozen prefix may be saved
concurrently by multiple callers; releasing it during a save is caller
undefined behavior.

`SPX1` carries the CPU backend, producer version, model hash, KV precision,
schema binding and walk state, current token, length and origin, hidden scale,
five saturation counts, residual, and canonical K/V rows. Its exact size is
`184 + hidden_size + length * bytes_per_token`. It omits page identities,
page size, budget, and allocation history. The SHA-256 digest detects accidental
changes to the bytes; it does not authenticate them. With a recomputed digest,
an edited blob can choose any model-resolvable schema and valid walk state,
length within the context cap, origin within length, valid current token,
int32 mantissa, any int64 exponent, any saturation counts, and arbitrary
residual and K/V bytes. Magic, backend, producer, model identity, and KV
precision must match exactly.

Restore accepts only a blob from this backend and producer. A producer version
mismatch returns `SSLM_RESTORE_PRODUCER_MISMATCH`; model, precision, and schema
mismatches have their distinct restore statuses. Every refusal clears `*out`.
`parent_or_null` is a sharing hint. A nested blob (origin above 0) shares the
hint's leading `floor(origin / B)` pages only when the hint is frozen, belongs
to the same model and the same pool, holds exactly `origin` positions, and its
rows over those pages, every layer, K and V, are byte-equal to the blob's. The
restore then draws `ceil(length / B) - floor(origin / B)` pages and fills them
from the blob, the parent's partial last page included, and reports the shared
count through `*out_shared_pages`. A null or nonmatching hint, or a root blob,
restores the same prefix privately (`SSLM_OK`, 0 shared) when the pool has
`ceil(length / B)` free pages. The restored child holds its own references to
the shared pages, so the parent may be released afterwards. A private nested
restore may return `SSLM_KV_POOL_EXHAUSTED` even where building the nested
prefix against its parent was admitted. `B` is the restore pool's page size
(the contiguous pool option's is the context cap). `SPX1` carries no page size,
so a blob restores into a pool of any page size; the restored prefix is then
adoptable only from pools of that page size, as for every prefix. Releasing a hint while restore reads
it is caller undefined behavior.

Every blob written by this library and restored by the same release version
(or the same development build) yields, on success, a prefix whose adopters
emit the same tokens as adopters of the prefill-built prefix. Accidental byte
changes are refused unless they collide under SHA-256. Development builds on
one version line may accept one another's blobs, outside that fidelity promise.
Model mapping refuses two schema names with the same nonzero FNV-1a 64-bit
hash, or a name hashing to zero, so a saved schema hash resolves uniquely.

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

`sslm_status` carries one success value (`SSLM_OK`) plus 30 distinct
rejection causes (an internal sentinel past the last real value is never
returned or accepted as an argument), in five groups: argument/precondition
rejections (a bad argument, a buffer too small, a misaligned buffer);
artifact/content rejections (a rejected artifact, an adapter that doesn't
match its base model, a restore whose content or KV shape doesn't match);
lifecycle rejections (a model, pool, or adapter with live handles still
attached to it; an adapter swap mid-token; a frozen
prefix reused; a KV pool with no room left); numeric/domain rejections (a
token id out of range, a context length exceeded, a budget-mode holder's
declared budget exceeded below the cap (`SSLM_KV_BUDGET_EXCEEDED`, ordinal 29,
1.12.0), a legal decode-output
token id with no tokenizer entry for the padded-vocabulary case, or —
distinct from all of those — a per-step numeric gate declining on an
otherwise valid model and valid params, `SSLM_NUMERIC_STEP_REFUSED`,
`sslm_decode_step_v2`'s own damped-greedy mode only); and schema rejections
(an unknown schema name; binding a schema to a non-fresh sequence; a
schema-content span on an unbound sequence; a prefix or restore whose
schema doesn't match; a fixed span the schema's own DFA cannot reach; a
schema the offline compiler could not prove satisfiable), the shared
`SSLM_GPU_SHADER_BINARY_STALE` deployment-mismatch cause, the
`SSLM_RESTORE_PRODUCER_MISMATCH` producer-version refusal (ordinal 30), plus one
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
