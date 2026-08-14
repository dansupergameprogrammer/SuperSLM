#pragma once
// T-1986 GPU-serial port (Claude/Vitruvius/t1986-gpu-serial-port-design-2026-08-13.md,
// commit 0160876d14 -- Sec10 Coverage Model, Sec11 B1-B12 build decomposition): the
// production entry-point surface T-2019's red suite (Claude/Curie/
// t2019-gpu-serial-red-suite-2026-08-13.md) gates the build against.
//
// EVERY SYMBOL BELOW IS DECLARED, NOT DEFINED. This header exists so the red suite
// compiles; it links to nothing until the build seat (Brunel, B1-B12) provides a
// definition. This is test-authoring infrastructure -- the call SHAPE the suite
// needs -- not the shipped design: the build seat's own dispatch recording,
// descriptor-table binding, and D3D12 plumbing are unconstrained by this header
// beyond the input/output contract each red-suite test asserts against. Mirrors the
// precedent this project already used for the identical situation (T-1899's red
// suite for T-1894 Option-G, `artifact.h`/`checked_chain_funnel.h`/`forward_sites.h`,
// commit 7f55b9d): "contract extensions, declared not defined... link fails on
// exactly these N symbols."
//
// Function groups below are named by the B-step (Sec11) whose gate they discharge.
// Each one's own test cell(s) are filed in tests/test_main.cpp (search "T-2019") and
// derived in full in the Curie casebook cited above.

#include <cstddef>
#include <cstdint>

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"

namespace superslm_gpu {

// --- B1 (Sec7.1, Sec11 B1): per-primitive int64/128-bit battery. One GPU dispatch
// per call, reading back the primitive's own output type. Oracle (in the red suite):
// the real, shipped CPU function of the same name, called directly. ---
int64_t DynamicScaleReciprocalGpu(int64_t dn);
// Returns the RequantTokenCodeWide narrowed int8 code, widened back to int64_t for
// a uniform comparison surface across every primitive in this battery.
int64_t RequantTokenCodeWideGpu(int64_t x_i, int64_t r, int s);
bool BiasReconcileWideGpu(int64_t b, int64_t q_b, int64_t r_a, int64_t e_a, int64_t* out);
superslm::CarriedScale CombineCarriedScaleGpu(superslm::CarriedScale a, superslm::CarriedScale b);

// --- B1 (Sec7.2, Sec11 B1): standalone signed-right-shift battery. The three named
// sites (int32 arithmetic shift, int64 arithmetic shift, the 128-bit SShrFull
// cross-word shift) are internal to intmath.cpp with no public symbol this header
// can cite without inventing one the CPU side does not have -- grep of this tree at
// main@727e63e finds no public `SShrFull` (Curie casebook, Sec2). This battery's own
// negative-operand residual (O-3, Sec7.2: DynamicScaleReciprocal's seed `>> 31`) is
// realized as a fixture on `DynamicScaleReciprocalGpu` above rather than a separate
// symbol -- named explicitly in the casebook rather than silently dropped.

// --- B5 (Sec5.5, Sec11 B5): the two-schedule int64 abs-max reduction (SCHEME 0
// sequential, SCHEME 1 shared-memory binary tree) -- SCHEME 2/3 are not built for
// this operator/width (Sec5.5's own ruling) and carry no symbol here. Oracle: the
// real, shipped MaxAbsReduceWide. ---
int64_t MaxAbsReduceWideGpuScheme0(const int64_t* x, size_t n);
int64_t MaxAbsReduceWideGpuScheme1(const int64_t* x, size_t n);

// --- B2 (Sec6, Sec11 B2): the guard-path port, isolated tier -- the per-dispatch
// status word a standalone guard kernel would write, before any site integrates it.
// Covers the three genuine rejecting guards (RoundingDivideByPot's composed-exponent
// check, RequantChainChecked's mantissa/d_prime checks) and ApplyBiasReconcileRow's
// own compound predicate (Sec11 B2 (3), D-SLM3053), for k_bias and v_bias
// independently. ---
superslm::SslmForwardStatus CheckRoundingDivideByPotExponentDomainGpu(int64_t q_B, int64_t e_a);
superslm::SslmForwardStatus CheckBiasAccumulateMagnitudeDomainGpu(int64_t acc_i, int64_t b,
                                                                   int64_t q_b, int64_t r_a,
                                                                   int64_t e_a);
superslm::ChainResult RequantChainCheckedGpu(const int64_t* wide_row, size_t n,
                                              const superslm::CarriedScale* incoming,
                                              size_t n_incoming,
                                              superslm::CarriedScale site_constant);

// --- B7/B11 (Sec5.6, Sec11 B7/B11): the composed, device-resident, multi-layer
// forward -- the sequence-level sticky word, kv_proj's fused guard-combinator and
// split-word saturation counter, and RoPE's staged K commit, all included. Mirrors
// RunLayerLoop's own signature and SequenceLayerState contract exactly (Sec10 dim 7:
// "the CPU path remains the normative oracle" the GPU port is checked against, never
// the reverse) -- a caller supplies the identical inputs RunLayerLoop takes and reads
// back the identical struct shape. This is also B8's own per-token driver (a decode
// session is N calls at layer_budget == num_hidden_layers) and B4's per-site proof
// composes through it once every site lands. ---
superslm::SslmForwardStatus RunLayerLoopGpu(superslm::SequenceLayerState& seq,
                                             const superslm::LayerWeights* layers,
                                             uint32_t num_hidden_layers, uint32_t layer_budget,
                                             size_t hidden_size, size_t head_dim,
                                             size_t num_key_value_heads, size_t intermediate_size,
                                             int64_t context_cap,
                                             const superslm::SslmTensorManifest& rope_tables,
                                             uint8_t* workspace, size_t workspace_size);

// Read back the device-resident K/V cache in the SAME layout and argument order
// superslm::KeyRow/ValueRow already define (forward_sites.h) -- the GPU port's
// `workspace` is the device-resident twin of the identical buffer RunLayerLoop uses.
const int8_t* KeyRowGpu(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                         size_t num_kv_heads, size_t head_dim, size_t kv_head,
                         int64_t position);
const int8_t* ValueRowGpu(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                           size_t num_kv_heads, size_t head_dim, size_t kv_head,
                           int64_t position);

// --- B8 (Sec11 B8): device-to-host / host-to-device round-trip of the
// residual-stream/KV/kv_saturation_count/context_length state (Sec10 dim 9,
// D-SLM3034), mechanism-level -- SuperSLM_Plan.md Sec12's `sslm_seq_save`/`restore`
// C-ABI wrapper is not yet built on ANY backend (grep of this tree at main@727e63e
// finds no such symbol), so this red suite gates the device-resident round-trip
// mechanism the design commits to, not a specific ABI symbol name; named explicitly
// in the Curie casebook rather than silently assumed. `out_seq`/`out_workspace` are
// caller-owned, sized identically to the save call's own `seq`/`workspace`. ---
bool SaveGpuSequenceState(const superslm::SequenceLayerState& seq, const uint8_t* workspace,
                           size_t workspace_size, void* out_blob, size_t* out_blob_size);
bool RestoreGpuSequenceState(const void* blob, size_t blob_size, superslm::SequenceLayerState* out_seq,
                              uint8_t* out_workspace, size_t workspace_size);

// --- B3 (Sec5.1, Sec11 B3): descriptor-table binding substrate + int8-native
// packing. Maps a synthetic multi-array fixture (sized to a real tier's layer count
// x 42-array-pointer shape, Sec4) through the table-based binder and reads back
// every bound array's known-pattern content, including negative int8 values
// (Sec5.2's corrected sign-extension). ---
bool BindDescriptorTableAndReadback(const int8_t* const* array_pointers,
                                     const size_t* array_element_counts, size_t n_arrays,
                                     int8_t* out_readback_concat, size_t out_readback_capacity);
// Two-handle cell (Sec11 B3, added Sec14 Fold G1): map, write a distinguishing
// pattern, release; map a second, different handle into the freed descriptor-heap
// region; report whether the second handle's own shader read shows residue from the
// first.
bool DescriptorHeapRegionIsCleanAfterHandleRelease();

// --- B12 (Sec10 dim 5, Sec11 B12): GPU residency allocation-failure handling.
// `requested_bytes` sized past the adapter's own reported budget by the caller
// (IDXGIAdapter3::QueryVideoMemoryInfo). Returns true iff a defined failure status
// came back with no partial device-resident state left behind, confirmed by a
// subsequent clean map at a budget that fits. ---
bool GpuResidencyMapReturnsDefinedFailureOnOverBudgetRequest(uint64_t requested_bytes);

}  // namespace superslm_gpu
