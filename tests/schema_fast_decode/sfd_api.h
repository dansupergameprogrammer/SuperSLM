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
// M1 and J1 define every symbol in groups 1-3 in the library. The red round's no-behaviour stubs
// (sfd_red_stubs.cpp) and their CMake option SUPERSLM_SFD_RED_STUBS were removed when the slices
// landed (CHANGELOG.md, CMakeLists.txt), so a missing definition is now a link error.
//
// Plan rev 1.3.3's slice TF is red first by the same pattern: group 4, its seams, defined with no
// behaviour by sfd_red_stubs_tf.cpp under option SUPERSLM_SFD_TF_RED_STUBS. The TF build defines them
// and turns the option OFF. Slice J1-X likewise: group 1x (its ABI) and group 5 (its seams), defined
// with no behaviour by sfd_red_stubs_x.cpp under option SUPERSLM_SFD_X_RED_STUBS.
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

#if defined(SUPERSLM_SFD_X_RED_STUBS)
/* Admits the pending choice token (§5.4), if any, then forced-chain tokens, up to
   `chunk_budget` tokens in all, in one batched layer pass through the prefill path.
   Writes the admitted *forced* ids to out_forced (cap >= chunk_budget required). */
SUPERSLM_API sslm_status sslm_seq_jump_forward(sslm_model model, sslm_seq seq, int32_t chunk_budget,
                                               const sslm_decode_params* params, /* NULL = greedy (§5.8) */
                                               sslm_workspace ws,
                                               int32_t* out_forced, int32_t cap, int32_t* out_forced_count);
#else
/* Plan rev 1.3.3 §5.10.2 (slice J1-X): the verb gains an options pointer. */
SUPERSLM_API sslm_status sslm_seq_jump_forward(sslm_model model, sslm_seq seq, int32_t chunk_budget,
                                               const sslm_decode_params* params,  /* NULL = greedy (§5.8) */
                                               const sslm_jump_options* opts,     /* NULL = canonical */
                                               sslm_workspace ws,
                                               int32_t* out_forced, int32_t cap, int32_t* out_forced_count);
#endif

}  // extern "C"

// ---- 1x. Slice J1-X's ABI (plan rev 1.3.3 §5.10.2), red first -------------------------------------
// J1-X changes the verb's signature (an options pointer after params) and adds sslm_jump_mode and
// sslm_jump_options to include/superslm/sslm_abi.h. Until it lands (CMake option
// SUPERSLM_SFD_X_RED_STUBS, default ON) the shipped 8-argument verb above stays the library's, the two
// types are defined here verbatim from §5.10.2, and the 9-argument form is a C++-linkage overload that
// sfd_red_stubs_x.cpp defines with no behaviour: it writes nothing and returns SSLM_STATUS_NEXT_FREE,
// the sentinel no verb ever returns, so every J1-X cell is red at its first exact call ("exact mode
// unimplemented") and no cell can pass on the stub's status. The J1-X build adds the types and the
// 9-argument extern "C" verb to the ABI headers and turns the option OFF in the same commit; this header
// then declares the same 9-argument verb (above), and every suite call goes through the helpers below,
// so no cell needs editing. (Plan §9's "the first commit updates sfd_api.h to the new signature and adds
// NULL at every call site" cannot be done before the library has the signature without breaking the
// link of every existing cell; the helpers are that NULL, applied when the option turns OFF.)
#if defined(SUPERSLM_SFD_X_RED_STUBS)
extern "C" {
typedef enum sslm_jump_mode { SSLM_JUMP_CANONICAL = 0, SSLM_JUMP_EXACT = 1 } sslm_jump_mode;

typedef struct sslm_jump_options {
	uint32_t struct_size;  /* sizeof(sslm_jump_options) */
	int32_t mode;          /* sslm_jump_mode */
	const int32_t* draft;  /* EXACT only. NULL: the canonical chain from the state after the pending token */
	int32_t draft_count;   /* EXACT only. >= 0; at most chunk_budget - 1 tokens are used */
} sslm_jump_options;
}  // extern "C"

sslm_status sslm_seq_jump_forward(sslm_model model, sslm_seq seq, int32_t chunk_budget,
                                  const sslm_decode_params* params, const sslm_jump_options* opts, sslm_workspace ws,
                                  int32_t* out_forced, int32_t cap, int32_t* out_forced_count);
#endif

namespace superslm {
namespace test {
namespace sfd {
// The canonical verb (NULL opts), in whichever signature the library has: every M1, J1 and TF cell
// calls the verb through this, so the J1-X signature change needs no cell edit.
inline sslm_status JumpCanonical(sslm_model model, sslm_seq seq, int32_t chunk_budget, const sslm_decode_params* params,
                                 sslm_workspace ws, int32_t* out_forced, int32_t cap, int32_t* out_forced_count) {
#if defined(SUPERSLM_SFD_X_RED_STUBS)
	return sslm_seq_jump_forward(model, seq, chunk_budget, params, ws, out_forced, cap, out_forced_count);
#else
	return sslm_seq_jump_forward(model, seq, chunk_budget, params, nullptr, ws, out_forced, cap, out_forced_count);
#endif
}
// The 9-argument verb (the J1-X cells).
inline sslm_status JumpWithOptions(sslm_model model, sslm_seq seq, int32_t chunk_budget, const sslm_decode_params* params,
                                   const sslm_jump_options* opts, sslm_workspace ws, int32_t* out_forced, int32_t cap,
                                   int32_t* out_forced_count) {
	return sslm_seq_jump_forward(model, seq, chunk_budget, params, opts, ws, out_forced, cap, out_forced_count);
}
}  // namespace sfd
}  // namespace test
}  // namespace superslm

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
// its token count M (when a one-row pass threads -- bit 1 and two or more tasks -- the verb runs an
// admission of M = 2..7 rows as M such passes of one row each, §16 Q6 C). kind 1: a decode step that runs the layer loop for a sequence (one entry per
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

// ---- 4. The TF seams (plan rev 1.3.3 §4.5, §11.5) ------------------------------------------------
// Slice TF makes the decode step's finish a transaction. Its cells (TF.2, TF.7) need two seams the
// plan names by what they do (TF.2: "by seam or by construction"; TF.7: "the guard stubbed through a
// seam"); the signatures are the suite's. Until TF lands they are defined with no behaviour by
// sfd_red_stubs_tf.cpp (CMake option SUPERSLM_SFD_TF_RED_STUBS, default ON); the TF build defines them
// in the library beside the §11.0 seams and turns the option OFF in the same commit.

// S-FINISHEXIT (TF.2, TF.6). The finish's four status exits, none reachable by construction on F-SFD:
//   kFinalNorm     the final norm returns a non-Ok forward status (the decode step's existing
//                  `if (fst != Ok) return MapForwardStatus(fst)` after RmsNormSite);
//   kLogitsReject  the logits site (full or legal-row) returns a non-Ok forward status other than
//                  ParallelForIncomplete, as NarrowRowChecked's row reject does;
//   kDampedNotOk   damped greedy's scorer returns ok = false (the decode returns SSLM_INVALID_ARGUMENT);
//   kDampedRefused damped greedy's scorer sets `refused` (the decode returns the refusal's status).
// Armed, the seam lets `skip` finishes that reach the exit's site run normally, then fails every later
// one there, until DisarmFinishExit. It is persistent on purpose: TF.2's deterministic refusal must
// repeat on two retries. The damped exits are reached only by damped finishes, and a greedy finish
// does not count toward `skip` for them. "A finish" is one sequence's finish in a decode step (in a
// batch, one per sequence that reaches it, in batch order) and, once J1-X lands, each check row's
// FinishRow (plan §5.10.3: the decode step's own finish, factored and shared).
enum class FinishExit : int32_t { kNone = 0, kFinalNorm = 1, kLogitsReject = 2, kDampedNotOk = 3, kDampedRefused = 4 };
void ArmFinishExit(FinishExit exit, int32_t skip) noexcept;
void DisarmFinishExit() noexcept;
uint64_t FinishExitFired() noexcept;  // finishes the seam has failed since the last ArmFinishExit

// The TF guard, stubbed (TF.7). While disabled, a failed finish leaves whatever its exit leaves with
// no rest at the finish's entry (main's behaviour before TF); enabled is the default.
void SetFinishGuardDisabled(bool disabled) noexcept;

// ---- 5. The J1-X seams (plan rev 1.3.3 §11.4, "New seams"; cells 6.X3, 6.X7, 11.X1-11.X3) ----------
// The plan names S-VERIFY, S-SATROW and the S-ROWS check tag, and implies two more by what cells 11.X2
// and 11.X3 do to the build ("the walk is corrupted through a seam", "HolderUnmapTail stubbed through a
// seam"). The signatures are the suite's. Until J1-X lands they are defined with no behaviour by
// sfd_red_stubs_x.cpp (option SUPERSLM_SFD_X_RED_STUBS, default ON); the J1-X build defines them in the
// library beside the §11.0 seams and turns the option OFF.

// S-VERIFY (11.X1): forces the check's comparison (§5.10.3 step 3). kAlwaysEqual: every check row
// "confirms" its drafted token; kNeverEqual: none does (j = 0). Process-wide, until reset to kNone.
enum class VerifyForce : int32_t { kNone = 0, kAlwaysEqual = 1, kNeverEqual = 2 };
void SetVerifyForce(VerifyForce force) noexcept;

// S-SATROW (6.X3, 5.X3): single-shot. The next layer pass that computes row `row` adds one count to each
// of the sequence's five saturation counters (kv_saturation_count and the four per-site counts), both in
// the pass's per-row output and in the totals, as if that row had saturated once at every site. "Row r"
// is row r of a batched pass (RunLayerLoopChunkBatched, from any caller), the r-th one-row pass of a call
// under Q6 C's split, or, for r = 0, a decode step's own one-row layer pass (so an off run can carry the
// same injection at the same output position). Consumed by that pass whether or not its rows are kept.
void ArmSatRow(int32_t row) noexcept;
void DisarmSatRow() noexcept;
bool SatRowArmed() noexcept;  // false once consumed

// The S-ROWS check tag (6.X7, 8.X7): true iff S-ROWS entry i was written by the verb's check (FinishRow
// at a check row) rather than by a decode step's finish. False past the end.
bool RowsLogIsCheck(size_t i) noexcept;

// S-ROW's check slot (§5.10.3, "the check's capture goes to its own tagged slot"): the post-mask row of
// the last check finish, with LastFinishRow's contract. LastFinishRow keeps the decode finish's capture.
void LastCheckRowReset() noexcept;
size_t LastCheckRow(int32_t* out, size_t cap) noexcept;

// 11.X2: "the walk is corrupted through a seam". While set, exact mode's draft walk (§5.10.3) truncates
// the draft only for out-of-range ids and the capacity clip: special, byteless and illegal tokens pass
// it, so only the pre-scan (G26) stands between a hostile draft and the pass.
void SetDraftWalkLegalityDisabled(bool disabled) noexcept;

// 11.X3: HolderUnmapTail (§5.10.4) stubbed: while set, it returns without unmapping anything.
void SetHolderUnmapTailStubbed(bool stubbed) noexcept;

}  // namespace sfd
}  // namespace test
}  // namespace superslm

#endif  // SUPERSLM_TESTS_SCHEMA_FAST_DECODE_SFD_API_H
