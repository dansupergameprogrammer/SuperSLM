// Schema fast-decode plan (rev 1.2.1), slices M1 and J1: the surface the red suite calls, declared
// here so the cells compile on main (2e12a6e) and fail at their assertions instead of at the build.
//
// Three groups:
//   1. The two J1 verbs (§5.3), with the plan's own signatures. J1 appends them to
//      include/superslm/sslm_abi_functions.inc; the identical declarations here then stay legal
//      (a redeclaration with the same signature), so this header needs no edit when they land.
//   2. Two M1 internals the pure-function cells call (§4.1, §4.3): the static exactness predicate and
//      LogitsSiteLegalRows. The plan names LogitsSiteLegalRows and its parameters; the types below
//      follow LogitsSiteParallel's (forward_sites.h), with legal_ids as the CSR row's ascending u32
//      token ids. The predicate's name is the suite's (the plan describes it, §4.3, but names none).
//   3. The test-only seams of §11.0 (S-ROWS, S-LAYERROWS, S-ROW, S-CHOICE, S-BOUND, S-WALK), plus two
//      the plan implies without naming: a read of the forced-next table (cells 6.6, 10.1 (d)) and a
//      single-entry overwrite of it (cell 11.2, "the table is corrupted through a seam"). The plan
//      names what each seam does, never its signature; the signatures are the suite's. They live in
//      the test libraries only, under SUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS (CMakeLists.txt
//      defines it on every library a suite binary links), and are absent from the production library.
//
// Until M1 and J1 land, tests/schema_fast_decode/sfd_red_stubs.cpp defines every symbol below as
// "no behaviour" (a refusal, an empty log, nothing written through any pointer), behind the CMake
// option SUPERSLM_SFD_RED_STUBS (default ON). The builder defines the real symbols in the library and
// turns the option OFF in the same commit; a stub left ON beside a real definition is a duplicate
// symbol, so the link refuses the pair (the paged-kv T0 convention, tests/paged-kv/pkv_r9_stubs.cpp).
#ifndef SUPERSLM_TESTS_SCHEMA_FAST_DECODE_SFD_API_H
#define SUPERSLM_TESTS_SCHEMA_FAST_DECODE_SFD_API_H

#include <stddef.h>
#include <stdint.h>

#include "superslm/checked_chain_funnel.h"  // SslmForwardStatus
#include "superslm/parallel_for.h"
#include "superslm/sslm_abi.h"

// ---- 1. The J1 verbs (plan §5.3, verbatim) --------------------------------------------------------

extern "C" {

/* Read-only. Writes up to `cap` ids of the forced chain from the sequence's current walk state
   (ids may be NULL when cap == 0); *out_count = the full chain length (may exceed cap); 0 when no
   schema is bound or the state is unforced. *out_accepting = 1 iff the walk state is accepting.
   Changes nothing, including bind eligibility. */
SUPERSLM_API sslm_status sslm_seq_forced_tokens(sslm_model model, sslm_seq seq,
                                                int32_t* out_ids, int32_t cap, int32_t* out_count,
                                                int32_t* out_accepting);

/* Admits the pending choice token (§5.4), if any, then forced-chain tokens, up to
   `chunk_budget` tokens in all, in one batched layer pass through the prefill path.
   Writes the admitted *forced* ids to out_forced (cap >= chunk_budget required). */
SUPERSLM_API sslm_status sslm_seq_jump_forward(sslm_model model, sslm_seq seq, int32_t chunk_budget,
                                               const sslm_decode_params* params, /* NULL = greedy (§5.8) */
                                               sslm_workspace ws,
                                               int32_t* out_forced, int32_t cap, int32_t* out_forced_count);

}  // extern "C"

namespace superslm {

// ---- 2. M1 internals -----------------------------------------------------------------------------

// §4.3: the static exactness predicate, as a pure function of the model's hidden size. True iff
// hidden_size * 16384 < INT32_MAX, i.e. iff hidden_size <= 131,071 (cell 2.4). The map computes the
// model's flag from it (through S-BOUND's bound, below).
bool LegalRowFinishExact(size_t hidden_size) noexcept;

// §4.1 steps 1-2: for each id in legal_ids[0, n) (ascending, each < vocab_size), wide_logits[id] =
// GemmInt8AccumulateRow over head row id, then out_logits[id] = that value narrowed. Writes no other
// position of either array. §4.2: the split is SplitColumns over n (alignment 64), and `run` is
// called iff pf->max_tasks >= 2 and n >= 1; a hook that breaks exactly-once returns
// SslmForwardStatus::ParallelForIncomplete.
SslmForwardStatus LogitsSiteLegalRows(const int8_t* final_codes, size_t hidden_size,
                                      const int8_t* head_weights, size_t vocab_size,
                                      const uint32_t* legal_ids, size_t n, int64_t* wide_logits,
                                      int32_t* out_logits, const sslm_parallel_for* pf);

namespace test {
namespace sfd {

// ---- 3. The §11.0 seams ----------------------------------------------------------------------------
// Every log below is process-wide, appended under a lock, and cleared only by its Reset.

// S-ROWS: one entry per token finish (CPU decode step, one per sequence that reaches the finish), in
// call order: the number of head rows that finish computed (V for the full finish, n for the
// legal-row finish, 0 for a legal-row finish at n = 0).
void RowsLogReset() noexcept;
size_t RowsLogCount() noexcept;
uint64_t RowsLogAt(size_t i) noexcept;  // 0 past the end

// S-LAYERROWS: one entry per layer pass, in call order. kind 0: a batched pass
// (RunLayerLoopChunkBatched, from sslm_prefill, sslm_prefix_prefill or sslm_seq_jump_forward), rows =
// its token count M. kind 1: a decode step that runs the layer loop for a sequence (one entry per
// sequence per decode call that embeds or runs any layer), rows = 1. A finish-only decode (on
// ready_for_logits) logs nothing.
enum : int32_t { kLayerRowsBatched = 0, kLayerRowsDecode = 1 };
struct LayerRowsEntry {
	int32_t kind = -1;
	uint64_t rows = 0;
};
void LayerRowsLogReset() noexcept;
size_t LayerRowsLogCount() noexcept;
LayerRowsEntry LayerRowsLogAt(size_t i) noexcept;  // {-1, 0} past the end

// S-ROW: a copy of the post-mask logit_row of the last token finish (greedy: after the degenerate-row
// seam and ApplyMaskAndArgmax; damped greedy: after its masking loop, before scoring). Writes
// min(cap, V) entries to out and returns V; returns 0 (writing nothing) when no finish has run since
// the last reset.
void LastFinishRowReset() noexcept;
size_t LastFinishRow(int32_t* out, size_t cap) noexcept;

// S-CHOICE: single-shot. The next token finish of a schema-bound sequence produces `token` in place
// of the argmax, after masking, in greedy and in damped greedy (the caller must pass a token legal
// at that state; an illegal one takes the dead-end path, as an illegal argmax would). The step is
// otherwise unchanged: the walk takes `token`, damped greedy feeds it to AntiLmUpdate.
void ArmChoice(int32_t token) noexcept;
void DisarmChoice() noexcept;
bool ChoiceArmed() noexcept;  // false once consumed

// S-BOUND: the hidden-size bound the exactness predicate uses at sslm_model_map (default 131,071).
// A model mapped while the bound is below its hidden size keeps the full finish. Read at map only.
void SetExactnessHiddenBound(size_t bound) noexcept;
void ResetExactnessHiddenBound() noexcept;

// S-WALK: the number of schema walk steps (SchemaMasksTable::Transition calls) taken by the prefill
// admission pre-scan (PrefillWholeTokensImpl), across every caller of it, since the last reset.
void WalkCountReset() noexcept;
uint64_t WalkCount() noexcept;

// The forced-next table, read (cells 6.6, 10.1 (d)): schema `schema_index`, state `state`.
// *next_forced = next_forced(s), or -1 for an unforced state; *f_len = |F(s)|, 0 when unforced.
// SSLM_ARTIFACT_REJECTED when the model has no table; SSLM_INVALID_ARGUMENT for a null argument or an
// index out of range.
sslm_status ForcedTableEntry(sslm_model model, int32_t schema_index, uint32_t state,
                             int32_t* next_forced, uint32_t* f_len) noexcept;

// The forced-next table, overwritten in one entry (cell 11.2): next_forced(s) := next_forced, |F|
// unchanged. Test-only; the production table is immutable after map. Same refusals as the read.
sslm_status CorruptForcedTableEntry(sslm_model model, int32_t schema_index, uint32_t state,
                                    int32_t next_forced) noexcept;

}  // namespace sfd
}  // namespace test
}  // namespace superslm

#endif  // SUPERSLM_TESTS_SCHEMA_FAST_DECODE_SFD_API_H
