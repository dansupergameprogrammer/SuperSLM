#ifndef SSLM_GPU_1P0_H
#define SSLM_GPU_1P0_H
// SuperSLM GPU acceleration API (D3D12-backed) -- the public entry points for creating a GPU
// context; mapping a model and its LoRA adapters into device residency; creating, saving,
// restoring, and releasing per-sequence decode state; and running batched decode steps on the
// GPU.
//
// Every fallible call returns the scoped SslmGpuStatus enum and is noexcept; every value it
// produces (a handle, a ready
// flag, a decoded status, a batch's per-sequence outcome) is delivered through an out-parameter,
// never through the return value. Every function below is declared at global scope with
// ordinary C++ linkage -- not extern "C", not inside a namespace -- and a caller linking
// against this header's implementation must match that scope exactly.
//
// A per-sequence decode-time rejection (an out-of-domain input, a guard check failing on that
// sequence's own step) is reported as SslmGpuStatus::SSLM_SEQUENCE_REJECTED, kept distinct from
// SslmGpuStatus::SSLM_DEVICE_LOST: the device stays healthy and no other sequence in the same batch call is
// affected. sslm_gpu_seq_restore additionally rejects a restore blob whose recorded
// model-content hash does not match the target model handle
// (SslmGpuStatus::SSLM_RESTORE_MODEL_MISMATCH), so
// a sequence saved from one model cannot be silently replayed against a different one.

#include <stdint.h>
#include <stddef.h>

#include "superslm/parallel_for.h"  // sslm_parallel_for (sslm_gpu_context_set_host_parallel_for)

/* --- opaque handle types, design Sec4.1 --- */
typedef struct SslmGpuContext        SslmGpuContext;
typedef struct SslmGpuModelHandle    SslmGpuModelHandle;
typedef struct SslmGpuAdapterHandle  SslmGpuAdapterHandle;
typedef struct SslmGpuSequenceHandle SslmGpuSequenceHandle;

/* Same namespace-collision fix the suite's own header carries:
 * SslmModelView must be THE real superslm::SslmModelView, not a second
 * distinct global-scope type of the same name, so production callers can load a real
 * .sslm artifact through this API. Declared once here for every 1.0 production TU. */
namespace superslm { struct SslmModelView; }
using superslm::SslmModelView;

/* The two by-value configuration structs of the context and model-map calls. An
 * all-zero-initialized value (`{}` or `{0}`) is always a valid "no options requested"
 * config. GpuContextConfig's first field, `reserved`, is ignored; GpuResidencyConfig's only
 * field is `flags` (1.7.0, below).
 *
 * GpuContextConfig::shader_dir (1.7.0) selects the directory the compiled .cso shader set
 * is loaded from:
 * - NULL (the default): shaders load from `<directory of the host executable>\shaders`,
 *   unchanged from earlier releases.
 * - Otherwise: an absolute directory path, UTF-8, NUL-terminated, holding the compiled .cso
 *   set. It is read only during sslm_gpu_context_create; the caller may free it afterwards.
 * The shader directory is PROCESS-WIDE, whichever context supplies it: compiled pipelines are
 * cached per process by shader name, so a process loads every shader from one directory for its
 * whole lifetime. The directory is fixed by the first of (a) a successful create with a
 * non-NULL shader_dir, or (b) the first shader load through the default path. After that, a
 * create with NULL uses the fixed directory; a create naming the same directory (any spelling
 * of it) succeeds; a create naming a different directory returns SSLM_GPU_SHADER_DIR_CONFLICT.
 * See sslm_gpu_context_create below for validation.
 *
 * Adding shader_dir grew GpuContextConfig from 4 to 16 bytes (x64). This is source-compatible
 * and binary-incompatible: code compiled against an earlier header must be recompiled. */
typedef struct GpuContextConfig {
	int reserved;            /* ignored; zero-initialize */
	const char* shader_dir;  /* NULL = default. Otherwise an absolute directory path, UTF-8,
	                            NUL-terminated, holding the compiled .cso set. Read during
	                            sslm_gpu_context_create only; the caller may free it after. */
} GpuContextConfig;
/* GpuResidencyConfig::flags (1.7.0; the field was `int reserved` before, with the same size,
 * offset and alignment, so a zero-initializing caller is binary-compatible). 0 is the behaviour
 * of every earlier release. sslm_gpu_model_map refuses any bit other than those below with
 * SSLM_GPU_RESIDENCY_FLAGS_INVALID and creates no handle.
 *
 * SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE: upload the model's output head table (the tied embedding,
 * or lm_head when untied) to the device at map time, so the token finish
 * (SslmGpuSeqFinishTokenForG5Bridge) computes the exact int64 logits row on the device and reads
 * it back; the host then runs the same narrowing, mask, argmax and dead-end rule as without the
 * flag, so tokens are identical. Opt-in per model. New VRAM per mapped model: three device
 * buffers are requested, the head table (vocab_size x hidden_size bytes: 136,134,656 B at
 * Qwen2.5-0.5B, 233,373,696 B at 1.5B), a hidden_size x 4 B input row and a vocab_size x 8 B
 * output row (137,353,728 B and 234,595,328 B in total). That sum is a lower bound: the driver
 * adds alignment and allocation overhead that differs by GPU and driver (measured +137,433,088 B
 * and +234,627,072 B on an RTX 2080 SUPER; +137,629,696 B and +234,889,216 B on an RX 7900
 * XTX). Nothing per sequence. With the flag set no separate host copy of the head is taken: an untied model's lm_head is not copied to the host, and a tied model's head is its
 * embedding table, which the handle keeps on the host in every case for token embedding. The
 * host parallel-for hook (sslm_gpu_context_set_host_parallel_for) is not used for that model's
 * finish. */
typedef struct GpuResidencyConfig {
	uint32_t flags;  /* SSLM_GPU_RESIDENCY_* bits; 0 = no options */
} GpuResidencyConfig;
constexpr uint32_t SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE = 0x1u;

/* --- status enum ---
 * Scoped deliberately: the CPU C ABI also has SSLM_-prefixed status constants, and both
 * installed public headers must be includable in either order. Existing ordinal spellings
 * remain members of this type; qualify them as SslmGpuStatus::SSLM_OK, etc. */
enum class SslmGpuStatus : uint32_t {
    SSLM_OK = 0,
    SSLM_DISPATCH_BUDGET_TOO_SMALL,      /* substrate, gpu_port.h:472           */
    SSLM_BUSY,                           /* design Sec9                         */
    SSLM_CONTEXT_HAS_LIVE_HANDLES,       /* design Sec9 -- B1                   */
    SSLM_MODEL_HAS_LIVE_SEQUENCES,       /* design Sec9 -- B2                   */
    SSLM_ADAPTER_MODEL_MISMATCH,         /* design Sec9 -- B6                   */
    SSLM_ADAPTER_BASE_HASH_MISMATCH,     /* design Sec9 -- B6                   */
    SSLM_SEQUENCE_KV_BUFFER_MISMATCH,    /* design Sec9 -- B3                   */
    /* design Sec9 -- B1/B2. Four distinct causes resolve to this ONE status, indistinguishable
     * at the ABI. Since 1.8.0 it never means that memory ran out on a live device with the
     * submission clean: that is SSLM_GPU_ALLOCATION_FAILED (below).
     * (a) an ordinary, healthy rejection -- most visibly, the batched G5 prefill entry points
     *     (SslmGpuSeqPrefillPromptForG5Bridge/SslmGpuSeqPrefillSchemaContentForG5Bridge, below)
     *     returning it when a chunk's own derived admit count comes back short of what was
     *     requested at a saturated context cap; the shipped per-token decode loop's own
     *     cap-boundary behavior, mirrored. The device and the context stay fully usable; a
     *     caller may continue issuing calls against the same context and sequence.
     * (b) a device fault, or a GPU operation that failed for a reason other than memory -- a
     *     lost/removed device, a missing shader, a D3D12 call failing with an HRESULT other than
     *     E_OUTOFMEMORY (an exception caught and contained at this boundary from the GPU
     *     command-submission tail among them). The command list is recovered at the point the
     *     fault is caught (confirmed Closed, matching the invariant every other failure path in
     *     the same function already restores) and the context stays usable for a caller that
     *     continues issuing calls against it, PROVIDED neither exception below applies -- both are
     *     checked and resolved to this same status before returning, so a caller cannot tell the
     *     difference from the status alone and must not assume recovery from it:
     *     - a confirmed device-removed condition (`GetDeviceRemovedReason()` non-S_OK) is
     *       genuinely terminal for the context regardless of the command-list state, and no call
     *       against it can be trusted afterward. At 1.6.0, a removed device stays this process's
     *       submission device, so every later sslm_gpu_context_create in the process returns
     *       SSLM_DEVICE_LOST until the process restarts (T-2845, D-SLM7386); a 1.6.x point
     *       release gives every context its own device instead of sharing the process's one, so
     *       a fresh device is built once every context on the removed one has been destroyed.
     *     - a stranded submission. A failed `Close()` -- whatever its HRESULT, E_OUTOFMEMORY
     *       included -- is retried once and then confirmed (`Reset()` then `Close()`: a list the
     *       driver already closed accepts the `Reset()`, one still recording refuses it). A list
     *       that cannot be confirmed Closed is left recording, and every later call against its
     *       context fails too (`ID3D12CommandAllocator::Reset()` refuses an allocator whose list
     *       is still recording). Every decode and prefill submits on one process-wide command
     *       list, so for them this means every later submitting call in the process fails --
     *       terminal, independent of what `GetDeviceRemovedReason()` reports, since a stuck list
     *       is unusable whether or not the device itself is confirmed gone;
     *     - a fault raised AFTER submission (`ExecuteCommandLists` already queued the work; only
     *       the fence `Signal()` itself failed) is retried at a FRESH fence value minted for the
     *       retry, never the same value the failed call attempted -- a same-value retry cannot
     *       distinguish "the failed Signal() already advanced the fence" from "it did not," so it
     *       is not issued. If the retry `Signal()` itself fails, or the fence wait cannot be
     *       armed, this is ALSO a stranded submission and terminal: the buffers this call is about
     *       to release may still be in use by work the GPU has not finished, indistinguishable
     *       from here from a genuinely removed device, and `GetDeviceRemovedReason()` is what a
     *       caller must consult, exactly as the two cases above. Only when the retry `Signal()`
     *       succeeds and is waited out does the context genuinely recover on this path; a retry
     *       that recovers after the first `Signal()` failed with E_OUTOFMEMORY returns
     *       SSLM_GPU_ALLOCATION_FAILED instead.
     * (c) SslmGpuSeqPrefillSchemaContentForG5Bridge only: a device-side domain guard refused
     *     one of the admitted tokens, found by the post-chunk readback. The context and the
     *     device stay usable, but THE SEQUENCE DOES NOT, unlike cause (a): its live residual and
     *     layer index are not a resting state, and a decode call issued on it without a reset
     *     returns a token computed from that state. Call sslm_gpu_seq_reset before reusing the
     *     sequence. The prompt twin reports the same refusal as SSLM_SEQUENCE_REJECTED instead,
     *     with the same reset requirement (see that function's comment).
     * (d) 1.8.0: the process's submission device could not be set up, for a reason other than
     *     memory -- no D3D12 hardware adapter, device creation refused, or a D3D12 setup step
     *     failing with an HRESULT other than E_OUTOFMEMORY. Every call that submits GPU work
     *     returns this status, from every entry point, on every call, for the life of the
     *     process. A first-time setup that runs out of memory is not this cause: that call
     *     returns SSLM_GPU_ALLOCATION_FAILED, and the next call sets up again. */
    SSLM_DEVICE_LOST,
    SSLM_BATCH_BUDGET_EXHAUSTED,         /* design Sec9 -- B7                   */
    SSLM_TOKEN_ID_OUT_OF_RANGE,          /* design Sec9 -- B3.5                 */
    /* A per-sequence
     * decode-time rejection, or a prompt-prefill rejection found by
     * SslmGpuSeqPrefillPromptForG5Bridge's post-chunk readback, that the CPU-domain guard ladder
     * (RunLayerLoopGpuSubmit's own pre-submission checks, or DecodeStickyTag's own post-dispatch
     * decode) produced --
     * InvalidLayerBudget, ChainInputOutOfDomain, SoftmaxRowWidthOutOfDomain, and every other
     * superslm::SslmForwardStatus value that is neither Ok nor a device-level failure
     * (GpuAllocationFailed, which maps to SSLM_GPU_ALLOCATION_FAILED, and GpuDeviceRemoved and
     * GpuOperationFailed, which map to SSLM_DEVICE_LOST, instead -- 1.8.0).
     * Design Sec9 deliberately assigns no dedicated 1.0 status per individual guard reason
     * (this enum does not grow one enumerator per CPU-domain check); this ONE status is the
     * real distinction that matters to a caller -- THIS sequence's own decode step was
     * rejected on a numeric/structural ground, the shared device is healthy, and (design
     * Sec7's own per-sequence independence) no other sequence in the same batch call is
     * affected. Before this status existed, both mapping functions
     * (MapSubmitRejectionToGpuStatus/MapDecodedStatusToGpuStatus, gpu_1p0.cpp) collapsed
     * every one of these into SSLM_DEVICE_LOST, which made sslm_decode_step_batch_gpu's own
     * "one sequence's guard rejection does not abort the batch" contract (Sec7) impossible to
     * honor -- the batch loop's own DeviceLost-poisons-the-rest logic (gpu_1p0.cpp) cannot
     * tell a real device loss apart from a healthy device's per-sequence rejection when both
     * arrive through the same value. */
    SSLM_SEQUENCE_REJECTED,
    /* (design Sec4.2/Sec9/Sec22): `sslm_gpu_seq_restore` (Sec4.2/Sec5.3), the v4 blob's
     * own `model_content_hash` field does not match the TARGET model handle's own
     * `RawIntegrityHash()` -- a blob saved from one model restored against a different one, the
     * identity gap the N1 size-admissibility widening (Sec21) left open. Checked after the
     * size-derivation ladder (a malformed blob is rejected for that reason first) and before any
     * device work. Appended LAST, the same precedent SSLM_SEQUENCE_REJECTED already
     * set: no existing enumerator value moves. */
    SSLM_RESTORE_MODEL_MISMATCH,
    /* T-2243 (M2, D-SLM3965): `sslm_gpu_model_unmap` rejects while any adapter is still mapped
     * against this model (`SslmGpuModelHandle::live_adapters > 0`) -- a persistent-liveness
     * condition (it holds until every mapped adapter is explicitly unmapped, never draining on
     * its own), distinct from the transient `SSLM_BUSY` row above, the same distinction
     * `SSLM_MODEL_HAS_LIVE_SEQUENCES` already draws against `submitted_sequences`. Appended
     * LAST, the same precedent SSLM_RESTORE_MODEL_MISMATCH already set: no existing enumerator
     * value moves. */
    SSLM_MODEL_HAS_LIVE_ADAPTERS,
    /* T-2243 (S2, D-SLM3965): `sslm_gpu_adapter_unmap` rejects while any sequence still holds a
     * bind to this adapter (`SslmGpuAdapterHandle::bound_sequences > 0`, set by
     * `sslm_gpu_seq_bind_adapter`) -- the same persistent-liveness shape as
     * SSLM_MODEL_HAS_LIVE_ADAPTERS above, for the adapter's own bound-sequence set rather than
     * the model's own mapped-adapter set. Remedy: unbind every sequence still holding this
     * adapter (`sslm_gpu_seq_bind_adapter(ctx, seq, nullptr)`) and retry. Appended LAST. */
    SSLM_ADAPTER_HAS_BOUND_SEQUENCES,
    /* T-2578 confirmation remedy: ShaderPath rejected a deployed .cso because it predates
     * one of its HLSL inputs. The model, sequence, and device remain valid, but retrying the
     * same deployment cannot succeed; rebuild/redeploy the matching shader set first.
     * Appended LAST so every existing public GPU status keeps its ordinal. */
    SSLM_GPU_SHADER_BINARY_STALE,
    /* An allocation failed on a device that is not removed. The contract a caller relies on
     * (SuperSLM 1.8.0):
     * - When: during any GPU call, one of these fails, on any heap, while the device is not
     *   reported removed at the moment the call classifies the failure:
     *   - a host allocation (std::bad_alloc, std::length_error);
     *   - a D3D12 call returning E_OUTOFMEMORY -- resource creation, Map, Reset, pipeline
     *     creation, and a Close or Signal the engine confirmed recovered;
     *   - the first-time setup of the process's submission device, at the first call that
     *     submits GPU work. The next call sets up again (see SSLM_DEVICE_LOST cause (d)).
     * - What stays valid: the context and the device, with the submission command list closed;
     *   every other sequence, model and adapter handle. An entry point with an output handle
     *   (sslm_gpu_context_create, sslm_gpu_model_map, sslm_gpu_adapter_map, sslm_gpu_seq_create,
     *   sslm_gpu_seq_restore) leaves it null. In sslm_decode_step_batch_gpu it is that
     *   sequence's own out_statuses[i]; the other sequences are recorded and complete as usual.
     * - What the caller may do next: retry the call; it succeeds once memory is available.
     *   Retrying smaller is also valid: a smaller context_cap, or a model mapped without
     *   SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE. A sequence the failed call was advancing must be reset
     *   (sslm_gpu_seq_reset) or restored before its next generation call: a prompt or schema
     *   prefill may have committed earlier sub-chunks; its prefill snapshot is empty, so
     *   sslm_gpu_seq_read_prefill_final_hidden returns SSLM_PREFILL_HIDDEN_UNAVAILABLE; a decode
     *   step's finish may have written K/V the host mirror does not hold.
     * - Named residuals, both reported as SSLM_DEVICE_LOST, the conservative direction: an
     *   allocation failure racing an unrelated device removal; and a stranded submission --
     *   memory exhausted during command recording where the command list cannot be confirmed
     *   Closed after Close, a retry and a confirm step, or a fence Signal or wait that cannot be
     *   confirmed (SSLM_DEVICE_LOST cause (b)). */
    SSLM_GPU_ALLOCATION_FAILED,
    /* sslm_gpu_seq_read_prefill_final_hidden: `out_capacity` is below the required element count.
     * Nothing is written except `*out_required`. Appended LAST; no existing ordinal moves. */
    SSLM_OUTPUT_BUFFER_TOO_SMALL,
    /* sslm_gpu_seq_read_prefill_final_hidden: the sequence holds no prefill snapshot (see that
     * function's comment). Appended LAST; no existing ordinal moves. */
    SSLM_PREFILL_HIDDEN_UNAVAILABLE,
    /* sslm_gpu_context_create: GpuContextConfig::shader_dir is unusable -- empty, not valid
     * UTF-8, not fully qualified (see sslm_gpu_context_create, step 4), not an existing
     * directory, or a directory holding no .cso file. No
     * context and no device are created. Remedy: supply the absolute path of the directory that
     * holds the compiled shader set. Appended LAST; no existing ordinal moves. */
    SSLM_GPU_SHADER_DIR_INVALID,
    /* sslm_gpu_context_create: this process already loads shaders from a different directory
     * (see GpuContextConfig above). No context is created. Remedy: supply the same directory, or
     * NULL. Appended LAST; no existing ordinal moves. */
    SSLM_GPU_SHADER_DIR_CONFLICT,
    /* sslm_gpu_context_set_host_parallel_for: a field of the hook is invalid (see that
     * function's comment). The context's hook is unchanged. Appended LAST. */
    SSLM_GPU_PARALLEL_FOR_INVALID,
    /* SslmGpuSeqFinishTokenForG5Bridge: the host parallel-for hook's `run` omitted a task index,
     * invoked one twice, or invoked one outside [0, task_count) (superslm/parallel_for.h). No
     * token is produced; `ready_for_logits` is re-armed and the walk state is unchanged, so the
     * finish can be retried through a correct hook. Appended LAST. */
    SSLM_GPU_PARALLEL_FOR_INCOMPLETE,
    /* sslm_gpu_model_map: GpuResidencyConfig::flags carries a bit this release does not define.
     * No handle is created. Appended LAST. */
    SSLM_GPU_RESIDENCY_FLAGS_INVALID
};

/* Output handles: sslm_gpu_context_create, sslm_gpu_model_map, sslm_gpu_adapter_map,
 * sslm_gpu_seq_create and sslm_gpu_seq_restore each set a non-null output slot to nullptr on
 * every status other than SSLM_OK, including SSLM_GPU_ALLOCATION_FAILED. A null output slot is
 * never dereferenced and is itself a refusal: SSLM_SEQUENCE_KV_BUFFER_MISMATCH from
 * sslm_gpu_seq_create and sslm_gpu_seq_restore, SSLM_DEVICE_LOST from the other three. */

/* --- Sec4.1.1: context create/destroy. DEFINED as of B1 (src/gpu/gpu_1p0.cpp). ---
 * sslm_gpu_context_create checks, in order, before any device is created, and sets `*out_ctx`
 * to nullptr on every refusal:
 * 1. `out_ctx` null: SSLM_DEVICE_LOST.
 * 2. `cfg.shader_dir` NULL: no shader-directory check; the process's shader directory applies
 *    (GpuContextConfig above).
 * 3. `cfg.shader_dir` empty, or not valid UTF-8: SSLM_GPU_SHADER_DIR_INVALID.
 * 4. Not fully qualified -- the value does not begin with a drive root (`X:\` or `X:/`) or a
 *    UNC prefix (two leading separators): SSLM_GPU_SHADER_DIR_INVALID. This refuses a relative
 *    path, a path rooted on the current drive (`\shaders`), and a drive-relative path
 *    (`C:shaders`), each of which resolves against the process's current directory.
 * 5. Not an existing directory, after full-path normalization and trailing-separator removal:
 *    SSLM_GPU_SHADER_DIR_INVALID.
 * 6. No `*.cso` file in the directory: SSLM_GPU_SHADER_DIR_INVALID.
 * 7. The process's shader directory is already fixed to a different directory (ordinal,
 *    case-insensitive comparison of the normalized paths): SSLM_GPU_SHADER_DIR_CONFLICT.
 * 8. Device acquisition: SSLM_GPU_ALLOCATION_FAILED when it runs out of memory (a host
 *    allocation, or E_OUTOFMEMORY from device creation or any setup step; the call may be
 *    retried), SSLM_DEVICE_LOST on any other failure. On success, a non-NULL shader_dir fixes
 *    the process's shader directory if nothing fixed it yet. */
SslmGpuStatus sslm_gpu_context_create(GpuContextConfig cfg, SslmGpuContext** out_ctx) noexcept;
SslmGpuStatus sslm_gpu_context_destroy(SslmGpuContext* ctx) noexcept;

/* Installs a host parallel-for hook on the context, or clears it when `pf` is NULL
 * (superslm/parallel_for.h states the contract the hook's `run` must meet). Copies *pf.
 * - Used only by SslmGpuSeqFinishTokenForG5Bridge's host logits step, for a model mapped without
 *   SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE. With no hook the finish is serial on the calling thread,
 *   as in every earlier release. Tokens are identical with any hook and no hook.
 * - Returns SSLM_SEQUENCE_KV_BUFFER_MISMATCH for a null ctx, and SSLM_GPU_PARALLEL_FOR_INVALID,
 *   changing nothing, for pf->reserved != 0, pf->max_tasks < 0 or
 *   > SSLM_PARALLEL_FOR_MAX_TASKS, or pf->run NULL with pf->max_tasks > 1.
 * - Must be externally serialized with every other call on the context, as submission already
 *   is. pf->host_ctx must outlive every finish made on the context while the hook is installed. */
SslmGpuStatus sslm_gpu_context_set_host_parallel_for(SslmGpuContext* ctx,
                                                     const sslm_parallel_for* pf) noexcept;

/* --- Sec5.1: model map/unmap. Declared for B2. ---
 * sslm_gpu_model_map: `cfg.flags` with an undefined bit returns SSLM_GPU_RESIDENCY_FLAGS_INVALID
 * and creates no handle. An allocation failure anywhere in the map (1.8.0) -- host memory, or
 * E_OUTOFMEMORY while uploading the weights, RoPE tables or schema masks or, with
 * SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE set (GpuResidencyConfig above), while allocating the head's
 * device buffers -- on a device that is not removed, returns SSLM_GPU_ALLOCATION_FAILED with no
 * handle (SSLM_GPU_ALLOCATION_FAILED above); the context stays usable, and the map may be
 * retried on it, with or without the flag. The flag also loads `logits_site.cso` from the
 * process's shader directory at map time: a stale binary returns SSLM_GPU_SHADER_BINARY_STALE,
 * and a missing one SSLM_DEVICE_LOST, the statuses every other shader load already has. Every
 * other device failure during the map, and an artifact that cannot be marshaled, returns
 * SSLM_DEVICE_LOST, as before. */
SslmGpuStatus sslm_gpu_model_map(SslmGpuContext* ctx, const SslmModelView* base,
                                  GpuResidencyConfig cfg, SslmGpuModelHandle** out_model) noexcept;
SslmGpuStatus sslm_gpu_model_unmap(SslmGpuContext* ctx, SslmGpuModelHandle* model) noexcept;

/* --- Sec5.2: adapter map/unmap. Declared for B6.
 * `sslm_gpu_adapter_map`: on a base-hash mismatch against `model`, returns AdapterBaseHashMismatch,
 * `*out_adapter=nullptr`; on an allocation failure on a device that is not removed (1.8.0), returns
 * SSLM_GPU_ALLOCATION_FAILED, `*out_adapter=nullptr`, and the map may be retried; on any other
 * upload failure or a foreign `model` (mapped against a DIFFERENT context than `ctx`), returns
 * DeviceLost, `*out_adapter=nullptr`.
 *
 * `sslm_gpu_adapter_unmap`: releases the adapter's own residency and returns Ok -- CARRIES A `Busy`
 * PRECONDITION (design Sec5.2/Sec9, mirrors sslm_gpu_model_unmap's own identical precondition):
 * returns Busy while any Submitted sequence's own in-flight decode call still has `adapter` bound
 * -- releasing while such a submission is in flight would free device buffers (lora_ab_buf/
 * fold_buf) the GPU may still be reading through an already-recorded, not-yet-fenced command list, a
 * real use-after-free. REMEDY: drain every sequence that bound this adapter (poll sslm_gpu_ready to
 * completion) and retry -- the same remedy every other Busy-returning release call in this header
 * already expects a caller to apply. `adapter` mapped against a DIFFERENT context than `ctx`
 * returns DeviceLost, same disposition as the map call's own foreign-`model` case
 * above. `unmap(ctx, nullptr)` is a documented no-op, returns Ok, checked before the Busy/DeviceLost
 * preconditions above (matches sslm_gpu_model_unmap(ctx, nullptr)'s own precedent). --- */
SslmGpuStatus sslm_gpu_adapter_map(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const SslmModelView* adapter_artifact,
                                    SslmGpuAdapterHandle** out_adapter) noexcept;
SslmGpuStatus sslm_gpu_adapter_unmap(SslmGpuContext* ctx, SslmGpuAdapterHandle* adapter) noexcept;

/* --- Sec5.3: sequence create/release. Declared for B3. --- */
SslmGpuStatus sslm_gpu_seq_create(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                   int64_t context_cap, SslmGpuSequenceHandle** out_seq) noexcept;
SslmGpuStatus sslm_gpu_seq_release(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) noexcept;

/* --- T-2243 (S2, D-SLM3954/D-SLM3996, plan Sec6.1): bind a LoRA adapter to a sequence handle,
 * across calls -- distinct from the existing per-call `adapter_or_null` argument every decode
 * call already takes. `adapter_or_null == nullptr` unbinds, mirroring `sslm_seq_set_adapter`'s
 * own CPU-side convention. Checked in order:
 *  1. `!ctx || !seq || seq->ctx != ctx` -> SSLM_SEQUENCE_KV_BUFFER_MISMATCH (malformed handle).
 *  2. `0 < seq's own layer_index < model's own num_hidden_layers` (genuinely mid-token; a
 *     drained rest at 0 OR at num_hidden_layers both admit) -> SSLM_BUSY. Resolves through the
 *     caller's own continued decoding of the token already in progress -- not automatically.
 *  3. `adapter_or_null != nullptr`: a model-mismatched adapter -> SSLM_ADAPTER_MODEL_MISMATCH;
 *     a foreign-context adapter -> SSLM_SEQUENCE_KV_BUFFER_MISMATCH.
 *  4. Already bound to `adapter_or_null` -> SSLM_OK, no counter change (idempotent).
 *  5. Otherwise rebinds: decrements the previously-bound adapter's own `bound_sequences` (if
 *     any), increments `adapter_or_null`'s (if non-null), and returns SSLM_OK.
 * Unbound automatically on `sslm_gpu_seq_release`. Untouched by `sslm_gpu_seq_reset` (the same
 * "caller's own standing configuration survives reset" precedent `bound_schema_index` already
 * sets) and does NOT round-trip through `sslm_gpu_seq_save`/`sslm_gpu_seq_restore` -- a restored
 * handle's own binding is always null; a caller that wants one re-binds explicitly after
 * restore. --- */
SslmGpuStatus sslm_gpu_seq_bind_adapter(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                         const SslmGpuAdapterHandle* adapter_or_null) noexcept;

/* --- Sec5.3a: the production token-feed entry point. Host-only -- no dispatch, no
 * state transition to Submitted. DEFINED as of B3.5 (src/gpu/gpu_1p0.cpp), added at the
 * mini-fold of 2026-08-15. Precondition: seq state Idle (Busy against
 * Submitted). Effect on success: overwrites seq's own hidden_codes/hidden_scale via the
 * identical EmbedEntry primitive the CPU path calls, resets layer_index to 0, returns Ok.
 * On a hostile token_id (outside [0, vocab_size)): returns TokenIdOutOfRange, seq's own
 * state left untouched. */
SslmGpuStatus sslm_gpu_seq_embed_token(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                        int32_t token_id) noexcept;

/* --- Sec4.2: save/restore/reset. Declared for B3/B5.
 * `sslm_gpu_seq_save` writes a v5 blob ('SLM5', v1.6.0, T-2895/D-SLM7572), carrying `model`'s
 * own content hash (design Sec22), the four per-site K/V saturation counters whose sum is
 * kv_saturation_count, and a twelve-byte tail (bound_schema_index int32, dfa_walk_state
 * uint32, ready_for_logits uint32-as-bool) written immediately after the unchanged v4-sized
 * header and before the residual stream -- the schema binding and walk state a §3.9/§3.10
 * caller needs to resume a schema-bound sequence across a save/restore round trip.
 * `sslm_gpu_seq_restore` sizes the fresh handle to the blob's own recorded
 * context_cap (Sec21), rejects a v1/v2/v3 blob outright on magic, a malformed/inadmissible
 * size derivation, OR a model_content_hash that does not match `model`'s own hash --
 * SSLM_RESTORE_MODEL_MISMATCH, distinct from the generic malformed-blob disposition (Sec22) --
 * and still restores an 'SLM4' blob exactly as before: a v4 blob carries no tail, so its
 * restored schema binding defaults to unbound (bound_schema_index -1), no walk
 * (dfa_walk_state unused), not ready (ready_for_logits false). On an 'SLM5' blob, the
 * restored bound_schema_index and dfa_walk_state are validated against the target model's own
 * schema count and that schema's own state_count before use, mirroring
 * `sslm_seq_restore`'s own CPU-side validation.
 * `sslm_gpu_seq_restore`'s device statuses (1.8.0): the restore round-trips the blob's K/V bytes
 * through the process's submission device. An allocation failure on a device that is not
 * removed -- the fresh handle's buffers, the round trip, or that device's first-time setup
 * running out of memory -- returns SSLM_GPU_ALLOCATION_FAILED, and the restore may be retried; a
 * submission device that could not be set up for any other reason returns SSLM_DEVICE_LOST
 * (SSLM_DEVICE_LOST cause (d)), never SSLM_SEQUENCE_KV_BUFFER_MISMATCH, which stays the status
 * of a malformed blob; every other device failure returns SSLM_DEVICE_LOST. `*out_seq` is null
 * on every refusal.
 * --- */
SslmGpuStatus sslm_gpu_seq_save(SslmGpuContext* ctx, const SslmGpuSequenceHandle* seq,
                                 void* out_blob, size_t* out_blob_size) noexcept;
SslmGpuStatus sslm_gpu_seq_restore(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const void* blob, size_t blob_size,
                                    SslmGpuSequenceHandle** out_seq) noexcept;
/* Reset accepts every valid Idle sequence, including one resting at partial or full layer depth;
 * only a Submitted sequence returns SSLM_BUSY. It clears generation history and returns a bound
 * schema's walk to state 0 while preserving that binding. */
SslmGpuStatus sslm_gpu_seq_reset(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) noexcept;

/* --- Sec4.3: the two decode calls. Declared for B5/B7.
 * Thread-safety (design Sec5.4): safe to call concurrently from different threads
 * against DIFFERENT SslmGpuSequenceHandle values ("thread-safe execution over disjoint
 * sequences") -- BUT every call that submits GPU work (sslm_decode_step_gpu,
 * sslm_decode_step_batch_gpu, and sslm_gpu_ready(block=1) draining one) must be externally
 * serialized by the caller: this design does not build an internal queue-level lock (see
 * src/gpu/gpu_1p0.cpp's own SslmGpuContext comment for the full contract and the grounded,
 * currently-process-wide reason). Two threads driving the SAME sequence handle concurrently is
 * an unguarded caller error, not a supported use (see the same comment). */
SslmGpuStatus sslm_decode_step_gpu(
    SslmGpuContext* ctx,
    SslmGpuSequenceHandle* seq,
    const SslmGpuAdapterHandle* adapter_or_null,   /* per-sequence, Sec8 */
    uint32_t dispatch_budget) noexcept;

SslmGpuStatus sslm_decode_step_batch_gpu(
    SslmGpuContext* ctx,
    SslmGpuSequenceHandle* const* seqs,
    const SslmGpuAdapterHandle* const* adapters_or_null,  /* parallel array, per-sequence, Sec8 */
    uint32_t n_sequences,
    uint32_t dispatch_budget,          /* BATCH-WIDE -- design Sec7 */
    SslmGpuStatus* out_statuses) noexcept;       /* [n_sequences] -- design Sec7 */

/* --- Sec4.2: sslm_gpu_ready. Declared for B5. --- */
SslmGpuStatus sslm_gpu_ready(SslmGpuContext* ctx,
                              SslmGpuSequenceHandle* seq,
                              int32_t block,
                              int32_t* out_ready,
                              SslmGpuStatus* out_status) noexcept;

/* ============================================================================
 * G5: schema-constrained GPU decoding -- PROMOTED to this shipped surface (design Sec14.2).
 * Built and proven on include/superslm/gpu_1p0_g5_bridge.h, which stated it was NOT part of this
 * shipped surface -- a genuine product capability (a real host wanting schema-constrained
 * decoding on the GPU, the plan's own G-1 lineage) with proven bit-identical CPU/GPU parity (80
 * decode steps, matching SHA-256) had no
 * shipped entry point to reach it. Applying this design's own rung-7 precedent (Sec11.2: a
 * genuine capability promotes to production; only a state no legitimate host call could ever
 * reach stays test-only) points the opposite direction from the bridge's prior placement --
 * GPU parity being REQUIRED, with no dated deferral, independently forecloses staying test-only.
 * Declarations below are UNCHANGED from gpu_1p0_g5_bridge.h's own signatures (a relocation, not a
 * redesign) -- that header now includes this one for its remaining, non-verb content (the
 * kSslmGpuDfaWalkStateUnused sentinel) and stays valid for every existing includer. Definitions
 * are unchanged, src/gpu/gpu_1p0.cpp. Promotion mechanics beyond the declaration move (CMake
 * install/export list inclusion, a dedicated declaration-parity gate against this surface's own
 * suite mirror) are owed to the builder, not performed here -- this ruling authorizes exactly the
 * declaration relocation and its suite-side mirror; the mechanics stay owed to the
 * builder. --- */

/* True iff `model`'s own mapped artifact carried a SchemaMasks (SCM1) section -- an artifact
 * with none is a valid, unconstrained-only artifact (design Sec13.1), exactly the CPU ABI's own
 * disposition; every schema-bound call below is meaningless against such a model. */
bool SslmGpuModelHasSchemasForG5Bridge(SslmGpuModelHandle* model);

/* Resolves a compiled schema by name against `model`'s own host-side parsed SchemaMasksTable
 * (built once, at sslm_gpu_model_map time, from the SAME section bytes the CPU ABI parses) --
 * the GPU-1.0 twin of `sslm_schema_lookup`. Returns the schema's own index (>= 0) on a match,
 * -1 on no match. */
int32_t SslmGpuSchemaLookupForG5Bridge(SslmGpuModelHandle* model, const char* name);

/* Binds `schema_index` (as returned by the lookup above; -1 unbinds, mirroring
 * SSLM_SCHEMA_NONE) to `seq`. Bind, rebind, and unbind are valid ONLY after sequence creation or
 * `sslm_gpu_seq_reset`. A generation call that passes argument validation makes the
 * sequence ineligible until reset, including a valid no-op call. A call rejected for invalid
 * arguments leaves the sequence untouched and still bindable. A restored sequence must be reset
 * first. A Submitted sequence returns SSLM_BUSY and likewise leaves eligibility unchanged; any
 * other ineligible sequence returns SSLM_SEQUENCE_REJECTED without changing its binding or walk
 * state. A caller-malformed handle (`seq`/`model` null, `ctx` mismatch) returns
 * SSLM_SEQUENCE_KV_BUFFER_MISMATCH, the existing "malformed handle" bucket every 1.0 entry point
 * already uses. */
SslmGpuStatus SslmGpuSeqSetSchemaForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                              int32_t schema_index) noexcept;

/* Host-only sequence-state queries: neither submits GPU work nor polls/waits a fence. They return
 * SSLM_BUSY without changing the output while the sequence is Submitted, and
 * SSLM_SEQUENCE_KV_BUFFER_MISMATCH without changing it for a null output or malformed/foreign
 * handle. On SSLM_OK, `out_schema_bound` is 1 iff a schema is bound; `out_schema_accepting` is 1
 * iff a schema is bound and the current DFA state is in that schema's exact SCM1 accept set.
 * Unbound reads are 0. On a drained-but-not-yet-finished Idle sequence, accepting reports the
 * current pre-finish membership; drain and finish before using it to decide output finality. */
SslmGpuStatus SslmGpuSeqSchemaAcceptingForG5Bridge(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, int32_t* out_schema_accepting) noexcept;
SslmGpuStatus SslmGpuSeqSchemaBoundForG5Bridge(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, int32_t* out_schema_bound) noexcept;

/* Reads `seq`'s own current DFA-walk-state -- kSslmGpuDfaWalkStateUnused if no schema is bound. */
uint32_t SslmGpuSeqWalkStateForG5Bridge(SslmGpuSequenceHandle* seq);

/* The GPU-1.0 twin of `sslm_prefill(..., SSLM_SPAN_PROMPT, ...)` -- the REQUIRED way to prime a
 * fresh or reset sequence with a host prompt before decoding. Embeds and drives EVERY token in
 * `tokens` (including the last) to full depth, no walk-state touch, no masking. On success
 * (count > 0), sets an internal "ready for logits" flag mirroring `sslm_seq_s::ready_for_logits`
 * (src/sslm_abi.cpp) EXACTLY -- a caller's own next `SslmGpuSeqDecodeStepForG5Bridge` call
 * consumes this flag automatically.
 *
 * `dispatch_budget` is a bulk-throughput call, not a submission-slicing contract: it is
 * validated nonzero (SSLM_SEQUENCE_KV_BUFFER_MISMATCH otherwise) but this call records and
 * submits every admitted token in `tokens` as ONE chunk (subject only to an internal
 * driver-stability sub-chunk split, unrelated to this parameter's value) rather than issuing
 * `dispatch_budget`-sized round trips per token. Per-call, per-token submission slicing by a
 * dispatch budget remains the DECODE path's own contract
 * (`sslm_decode_step_gpu`/`SslmGpuSeqDecodeStepForG5Bridge`'s layer-loop-to-depth step),
 * unchanged by this call.
 *
 * Returns SSLM_SEQUENCE_REJECTED when a device-side domain guard refused one of the admitted
 * tokens (the guard family the CPU forward reports by name, found here by the post-chunk
 * readback) and the device was not reported removed when this call classified the refusal. The
 * context and the device remain usable. Tokens before the refused one are committed; the
 * refused token and every later one are not; the sequence's live residual and layer index are
 * unspecified and the prefill snapshot is empty. Call sslm_gpu_seq_reset before reusing the
 * sequence. Returns SSLM_GPU_ALLOCATION_FAILED (1.8.0) when memory ran out during the chunk, host
 * or GPU, on a device that is not removed and with the submission clean, including at the process
 * submission device's first-time setup -- the SSLM_GPU_ALLOCATION_FAILED contract above: earlier
 * sub-chunks may have committed and the prefill snapshot is empty, so reset (or restore) the
 * sequence before reusing it; retry once memory is available. Returns SSLM_DEVICE_LOST when the
 * committed count falls short for any other reason: a saturated context cap, a non-allocation
 * device or infrastructure fault in submission, fence wait or readback, a stranded submission,
 * or a removed device. This surface offers no query that tells a recovered context from a
 * removed device after SSLM_DEVICE_LOST. */
SslmGpuStatus SslmGpuSeqPrefillPromptForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                  const int32_t* tokens, int32_t count,
                                                  uint32_t dispatch_budget) noexcept;

/* Finishes a token once `seq`'s own layer loop has reached full depth (caller-ensures: drained
 * via `sslm_gpu_ready` to Idle, `seq`'s own layer_index == model->num_hidden_layers) -- runs
 * final_norm + logits (the SAME `RmsNormSite`/`LogitsSite` calls sslm_decode_step's own
 * finishing block uses) then, if a schema is bound, `superslm::ApplyMaskAndArgmax` indexed by
 * `seq`'s own walk-state -- advances the walk-state via `SchemaMasksTable::Transition`, exactly
 * `sslm_decode_step`'s own masked-argmax step. No schema bound: plain
 * `ArgmaxLowestIndexTieBreak`, byte-for-byte the pre-G5 path. On an ordinary produced token,
 * `*out_token` is that token id and `seq`'s own layer_index resets to 0. When the bound
 * schema's own walk-state is inside a `"type": "string"` leaf's content sub-automaton (v1.6.0,
 * Sec3.9 of `Claude/Plans/te266-gpu-path.md`, Wizard repo), see `schema_masks.h`'s own top-of-
 * file documentation for the field's value promise (the model's own free text up to its first
 * unescaped quote) and its consumer guidance -- a property of the compiled table itself,
 * identical on this path and on `sslm_decode_step`'s own CPU equivalent.
 *
 * SCHEMA DEAD END (v1.6.0, D-SLM3476): when a schema is bound and `Transition` finds no CSR row
 * entry for the masked-argmax winner at `seq`'s own walk-state (an accepting state's own
 * legitimately all-zero mask page, or a synthetic degenerate row where every admitted logit ties
 * at INT32_MIN and the lowest-index tie-break returns a token with no transition), this call
 * returns `*out_token == -2` at `SSLM_OK` -- never `SSLM_SEQUENCE_REJECTED`, which this bridge
 * reserves for the layer-loop-not-at-full-depth precondition failure below. This is the identical
 * status the CPU path returns for the same event (`sslm_abi.cpp`'s own two miss sites). On this
 * path `seq`'s own layer_index is NOT reset (the walk state and layer index are both left exactly
 * as they were), and `ready_for_logits` IS re-armed. A further call to THIS function re-runs
 * directly on the unchanged residual -- its own precondition (layer_index == num_hidden_layers)
 * is still satisfied, unconditionally, since layer_index was never reset; it does not read
 * `ready_for_logits` at all. A further call to `SslmGpuSeqDecodeStepForG5Bridge` is the one that
 * takes the `ready_for_logits` shortcut: composed call skips the embed and the layer drive and
 * calls this function directly instead. Either path deterministically reproduces the identical
 * `-2`/`SSLM_OK` result and consumes no new token from the caller -- retrying after a dead end is
 * therefore always safe and never re-drives the layer loop or embeds a stray token.
 *
 * Returns SSLM_SEQUENCE_REJECTED if the precondition (full depth reached) does not hold.
 *
 * THE LOGITS STEP (1.7.0). For a model mapped with SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE the exact
 * int64 logits row is computed on the device from the model's resident head table, read back,
 * and narrowed on the host with the same check the host path uses; a device submission failure
 * returns SSLM_DEVICE_LOST, except that an allocation failure on a live device with the submission
 * clean returns SSLM_GPU_ALLOCATION_FAILED (1.8.0). Otherwise the host computes it, through the
 * context's host parallel-for hook when one is installed
 * (sslm_gpu_context_set_host_parallel_for). A hook whose
 * `run` breaks its exactly-once contract returns SSLM_GPU_PARALLEL_FOR_INCOMPLETE with
 * `ready_for_logits` re-armed and the walk state and layer index unchanged. The row, and so the
 * token, is identical on every path. */
SslmGpuStatus SslmGpuSeqFinishTokenForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                int32_t* out_token) noexcept;

/* THE RECOMMENDED one-call-per-decode-step entry point -- the GPU-1.0 twin of
 * `sslm_decode_step`'s own composition (embed-if-needed, layer-loop-to-depth, finish), including
 * its `ready_for_logits` shortcut verbatim (src/sslm_abi.cpp): if a prior
 * `SslmGpuSeqPrefillPromptForG5Bridge`/`SslmGpuSeqPrefillSchemaContentForG5Bridge` call left that
 * flag set, `token_to_embed_if_needed` is IGNORED and this call finishes the already-computed
 * residual directly (no embed, no layer loop); otherwise it embeds `token_to_embed_if_needed`,
 * drives it to full depth, and finishes. A caller that always calls this once per decode step --
 * rather than hand-composing embed/`sslm_decode_step_gpu`/`sslm_gpu_ready`/
 * `SslmGpuSeqFinishTokenForG5Bridge` itself -- cannot reproduce the duplicate-KV-commit class of
 * bug an earlier build round found and fixed, by construction.
 *
 * Inherits `SslmGpuSeqFinishTokenForG5Bridge`'s own schema-dead-end contract (v1.6.0,
 * D-SLM3476) unchanged: `*out_token == -2` at `SSLM_OK` on a schema dead end, with
 * `ready_for_logits` re-armed by the underlying Finish call, so a caller that always retries
 * through this same entry point after a `-2` reproduces the identical result deterministically
 * and never re-embeds or re-drives the layer loop.
 *
 * Statuses (1.8.0): a failure while driving the token to full depth returns the status the
 * underlying `sslm_decode_step_gpu`/`sslm_gpu_ready` call reported -- SSLM_GPU_ALLOCATION_FAILED
 * for an allocation failure on a live device, SSLM_SEQUENCE_REJECTED for a guard refusal,
 * SSLM_DISPATCH_BUDGET_TOO_SMALL for a budget below one layer, SSLM_DEVICE_LOST for a device
 * failure -- where earlier releases reported every such failure as SSLM_DEVICE_LOST. */
SslmGpuStatus SslmGpuSeqDecodeStepForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                               int32_t token_to_embed_if_needed,
                                               uint32_t dispatch_budget, int32_t* out_token) noexcept;

/* Jump-forward's own GPU twin. Drives `count` FORCED tokens (already known -- never chosen, no
 * masking/argmax involved, exactly `PrefillWholeTokensImpl`'s own SSLM_SPAN_SCHEMA_CONTENT
 * branch, src/sslm_abi.cpp) through the full embed -> layer-loop-to-depth -> commit sequence.
 * Reachability is checked BEFORE each token's own forward pass (`SchemaMasksTable::Transition`
 * against `seq`'s own current walk-state) -- a token that leaves the DFA's language is rejected
 * (SSLM_SEQUENCE_REJECTED). Only the REJECTED token's own effects are withheld: its own
 * walk-state transition never applies, its own forward pass never runs, and `*consumed` is
 * never incremented for it. Tokens admitted BEFORE it in the same call already had their
 * walk-state advance, K/V write, and layer loop run for real, and `*consumed` already counts
 * them -- the documented partial-consumption contract (matching `sslm_prefill`'s own shape), not
 * full-call atomicity (design Sec14.3 -- RULED design text). Requires a schema already bound
 * (SSLM_SEQUENCE_REJECTED otherwise). Sets the SAME "ready for logits" flag
 * `SslmGpuSeqPrefillPromptForG5Bridge` sets whenever `*consumed > 0` -- including on the
 * rejection path, when earlier tokens in the same call were already admitted.
 *
 * `dispatch_budget_per_token` is a bulk-throughput call, not a submission-slicing contract: it
 * is validated nonzero (SSLM_SEQUENCE_KV_BUFFER_MISMATCH otherwise) but every admitted token in
 * `tokens` is recorded and submitted as part of ONE chunk (subject only to an internal
 * driver-stability sub-chunk split, unrelated to this parameter's value), never as
 * `dispatch_budget_per_token`-sized `sslm_decode_step_gpu` calls issued one token at a time.
 * Per-call, per-token submission slicing by a dispatch budget remains the DECODE path's own
 * contract (`sslm_decode_step_gpu`/`SslmGpuSeqDecodeStepForG5Bridge`'s layer-loop-to-depth
 * step), unchanged by this call.
 *
 * Returns SSLM_GPU_ALLOCATION_FAILED (1.8.0) when memory ran out during the chunk, on a device
 * that is not removed and with the submission clean -- the prompt twin's rule and the
 * SSLM_GPU_ALLOCATION_FAILED contract above; `*consumed` counts only the committed tokens, and
 * the sequence must be reset (or restored) before reuse.
 *
 * Returns SSLM_DEVICE_LOST for three distinct causes (see the status enum's own comment,
 * above), which the status does not tell apart:
 *  - an ordinary, healthy rejection when the chunk's own derived admit count comes back short
 *    of what was requested at a saturated context cap -- the context and device stay usable,
 *    and a caller may continue;
 *  - a real, contained non-allocation device fault from the GPU submit/finish tail, after which
 *    the command list is recovered and the context stays usable for a caller that continues,
 *    unless the device is confirmed removed or the submission is stranded, which is terminal;
 *  - a device-side domain guard refused one of the admitted tokens, found by the post-chunk
 *    readback. Tokens before the refused one are committed and `*consumed` counts them; the
 *    sequence's live residual and layer index are not a resting state, and the "ready for
 *    logits" flag may still be set from an earlier call, so a decode call issued without a reset
 *    returns a token computed from that state. The sequence must be reset before reuse.
 * The status alone cannot say which cause applied, so a caller that cannot rule out the third
 * resets the sequence before issuing anything further on it. (The prompt twin reports the guard
 * refusal as SSLM_SEQUENCE_REJECTED instead.) */
SslmGpuStatus SslmGpuSeqPrefillSchemaContentForG5Bridge(SslmGpuContext* ctx,
                                                          SslmGpuSequenceHandle* seq,
                                                          const int32_t* tokens, int32_t count,
                                                          uint32_t dispatch_budget_per_token,
                                                          int32_t* consumed) noexcept;

/* ============================================================================
 * The embedding read: the final hidden state a successful prefill leaves behind.
 * ============================================================================ */

/* The model's hidden width: the element count sslm_gpu_seq_read_prefill_final_hidden writes.
 * Needs no sequence and no prefill. A null `model` or `out_hidden_size` returns
 * SSLM_SEQUENCE_KV_BUFFER_MISMATCH and writes nothing. */
SslmGpuStatus sslm_gpu_model_hidden_size(const SslmGpuModelHandle* model,
                                         uint32_t* out_hidden_size) noexcept;

/* Post-final_norm hidden state at the last position of this sequence's most recent SUCCESSFUL
 * prefill: the most recent SslmGpuSeqPrefillPromptForG5Bridge or
 * SslmGpuSeqPrefillSchemaContentForG5Bridge call that reached its admission pre-scan returned
 * SSLM_OK, and no sslm_gpu_seq_reset has run since. Every other public call leaves what this
 * returns unchanged.
 *
 * out_capacity == 0 does NOT act as a prefill-independent size query: the no-snapshot check
 * (SSLM_PREFILL_HIDDEN_UNAVAILABLE) runs before the capacity check, so a sequence holding no
 * prefill snapshot refuses here at any capacity, including 0. Callers size their buffer from
 * sslm_gpu_model_hidden_size, which needs no sequence and no prefill, not from a zero-capacity
 * call to this verb. *out_required is written only once the handle, busy and snapshot checks
 * pass; SSLM_SEQUENCE_KV_BUFFER_MISMATCH, SSLM_BUSY and SSLM_PREFILL_HIDDEN_UNAVAILABLE write
 * nothing.
 *
 * Checked in order:
 *  1. `ctx` or `seq` null, `seq` not created against `ctx`, any of `out_required`,
 *     `out_scale_m`, `out_scale_e` null, or `out_codes` null with `out_capacity` nonzero ->
 *     SSLM_SEQUENCE_KV_BUFFER_MISMATCH, nothing written.
 *  2. `seq` Submitted -> SSLM_BUSY, nothing written.
 *  3. No prefill snapshot -> SSLM_PREFILL_HIDDEN_UNAVAILABLE, nothing written. A fresh,
 *     restored or reset sequence holds none, and neither does one whose most recent prefill
 *     call that reached its admission pre-scan did not return SSLM_OK.
 *  4. `*out_required` = the hidden width, on this and every later outcome. `out_capacity`
 *     below it -> SSLM_OUTPUT_BUFFER_TOO_SMALL, nothing else written.
 *  5. final_norm (the RmsNormSite call SslmGpuSeqFinishTokenForG5Bridge makes before logits)
 *     over the snapshot. A guard refusal -> SSLM_SEQUENCE_REJECTED, nothing written to
 *     `out_codes`, `out_scale_m` or `out_scale_e`.
 *  6. Writes exactly the hidden width of codes and the scale, returns SSLM_OK.
 *
 * Host-side: records no command list and adds no dispatch. Writes nothing on the sequence, so a
 * decode step issued after a read produces what it would have produced without it. `out_codes`
 * needs no alignment. Thread-safety as the rest of this header: calls on distinct sequences may
 * run concurrently; a read and another call driving the same sequence from two threads is a
 * caller error. */
SslmGpuStatus sslm_gpu_seq_read_prefill_final_hidden(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                     int8_t* out_codes, size_t out_capacity,
                                                     size_t* out_required,
                                                     int64_t* out_scale_m, int64_t* out_scale_e) noexcept;

#endif /* SSLM_GPU_1P0_H */
