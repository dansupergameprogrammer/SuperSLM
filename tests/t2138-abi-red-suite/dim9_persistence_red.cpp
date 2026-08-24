// T-2138 (Curie) -- Dim 9 (Persistence round-trip and version evolution), design Sec7.3/Sec9
// C5 gate/Sec10 dim9. 3 cells. RED BY LINK.
#include "fixture_common.h"

#include <cstring>

using namespace superslm;

// --- Cell 1 (design Sec9 C5 gate: "round-trip bit-equality including a mid-token
// (non-zero layer-index) suspend point"). The base save/restore obligation dim7/dim8 both
// cross-cite rather than duplicate. ---
static void TestDim9_C1_SaveRestoreMidTokenRoundTripBitEqual(sslm_model model, sslm_seq seq,
                                                              sslm_kv_pool* pool) {
	int32_t prompt[3] = {0, 1, 2};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, prompt, 3, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
	      SSLM_OK);
	CHECK(EnterMidToken(model, seq));  // leaves the sequence resting mid-token, non-zero layer_index.

	SeqBlobBuffer blob(model);
	CHECK(sslm_seq_save(seq, blob.bytes.data(), &blob.size) == SSLM_OK);
	sslm_seq restored = nullptr;
	CHECK(sslm_seq_restore(model, pool, blob.bytes.data(), blob.size, &restored) == SSLM_OK);

	// FEATURE ORACLE: a second save immediately after restore reproduces the identical blob
	// bytes -- bit-equal round-trip, not merely "restore succeeds" (the same discipline
	// tests/t2130-g5-red-suite/dim9_persistence_red.cpp's own M1 cell already establishes for
	// the sibling G5 surface, reused here for the base mid-token case).
	SeqBlobBuffer blob2(model);
	CHECK(sslm_seq_save(restored, blob2.bytes.data(), &blob2.size) == SSLM_OK);
	CHECK(blob2.size == blob.size);
	CHECK(std::memcmp(blob.bytes.data(), blob2.bytes.data(), blob.size) == 0);
	CHECK(sslm_seq_release(restored) == SSLM_OK);
}

// --- Cell 2 (design Sec6/Sec7.3/Sec10 dim2/dim7/dim9, filed jointly per dim7's own
// instruction: "the handle-type half... N/A; the persisted-state half... gets a real cell,
// filed jointly with dimension 2's own blob-rejection cell"). sslm_seq_restore given a
// well-formed GPU-format ('SLM3'-magic) blob rejects on magic mismatch (SSLM_RESTORE_MODEL_
// MISMATCH is the closest-named enumerator this suite's own taxonomy has for "the blob does not
// belong to this surface" -- the design's own §6 does not carve out a dedicated
// cross-surface-magic enumerator distinct from RESTORE_MODEL_MISMATCH, so this cell asserts
// against that one, consistent with the design's own "never a false accept, never UB from
// mis-parsing the GPU header's field layout as the CPU one" requirement regardless of which
// named enumerator the rejection surfaces as). ---
static void TestDim9_C2_RestoreRejectsWellFormedGpuFormatBlobOnMagicMismatch(sslm_model model,
                                                                             sslm_kv_pool* pool) {
	// A synthetic buffer carrying the GPU ABI's own 'SLM3' magic (design Sec7.3's own citation:
	// GpuSeqBlobHeader, superslm_gpu.cpp) followed by plausible-looking but CPU-format-
	// incompatible field bytes -- genuinely well-formed AS a GPU blob's own header shape, never
	// merely random/truncated bytes (which SSLM_INVALID_ARGUMENT or a generic parse failure
	// would already catch; this cell's own subject is specifically the MAGIC discrimination).
	unsigned char gpu_format_blob[128] = {0};
	gpu_format_blob[0] = 'S';
	gpu_format_blob[1] = 'L';
	gpu_format_blob[2] = 'M';
	gpu_format_blob[3] = '3';
	sslm_seq restored = nullptr;
	const sslm_status st =
	    sslm_seq_restore(model, pool, gpu_format_blob, sizeof(gpu_format_blob), &restored);
	// Never a false accept: SSLM_OK is categorically wrong here regardless of which specific
	// rejection enumerator fires.
	CHECK(st != SSLM_OK);
	CHECK(restored == nullptr);
}

// --- Cell 3 (design Sec10 dim9, Mendeleev 4.5 folded, the reconciled magic-per-version
// hard-reject strategy: "construct a byte buffer that is a valid v1 blob... with its magic
// bytes corrupted to an unrecognized value, and confirm sslm_seq_restore rejects on the magic
// check before parsing any field"). A REAL v1 blob (from a genuine sslm_seq_save) with its
// magic corrupted -- distinct from cell 2's cross-format ('SLM3') construction, since this
// cell's own subject is an UNRECOGNIZED magic, not a recognized-but-foreign one. ---
static void TestDim9_C3_CorruptedMagicOnRealV1BlobRejectedBeforeFieldParsing(sslm_model model,
                                                                              sslm_seq seq,
                                                                              sslm_kv_pool* pool) {
	int32_t prompt[2] = {0, 1};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, prompt, 2, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
	      SSLM_OK);
	SeqBlobBuffer blob(model);
	CHECK(sslm_seq_save(seq, blob.bytes.data(), &blob.size) == SSLM_OK);
	// Corrupt the magic bytes (design Sec7.3: "magic: 4 bytes, 'SSB1'") to an unrecognized
	// 4-byte value, leaving every OTHER byte of the real v1 blob untouched -- if the
	// implementation parsed fields before checking the magic, the rest of the blob would still
	// parse as a structurally valid (if now content-mismatched) sequence, masking this cell's
	// own defect class; the design's own hard-reject strategy requires the magic check to fire
	// FIRST, before any such downstream field is ever touched.
	blob.bytes[0] = 0xDE;
	blob.bytes[1] = 0xAD;
	blob.bytes[2] = 0xBE;
	blob.bytes[3] = 0xEF;
	sslm_seq restored = nullptr;
	const sslm_status st = sslm_seq_restore(model, pool, blob.bytes.data(), blob.size, &restored);
	CHECK(st != SSLM_OK);
	CHECK(restored == nullptr);
}

// T-2260 (D-SLM4073, Option A + Sec6 safety net; plan Claude/Vitruvius/
// t2260-ssb3-residual-options-2026-08-23.md) -- three cells, this fold.
namespace {
inline uint32_t T2260ReadLE32(const uint8_t* p) {
	return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
	       (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
inline uint64_t T2260ReadLE64(const uint8_t* p) {
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
	return v;
}
}  // namespace

// --- R1 (D-SLM4065's own defect, reproduced end to end): save a sequence resting at
// ready-for-logits (post-prefill, layer_index == 0) -- the state whose residual the pre-fix
// save path silently dropped -- restore it, and decode one step. Asserts the restored decode's
// own token matches a LIVE reference continuation from an identically-prefilled sequence, never
// saved/restored -- the same identity D-SLM4065's own repro checked by hand (token 0 emitted vs
// 97 wanted, pre-fix). ---
static void TestT2260_R1_ReadyForLogitsSaveRestoreMatchesLiveContinuation(sslm_model model,
                                                                          sslm_kv_pool* pool) {
	int32_t prompt[3] = {0, 1, 2};
	int32_t consumed = 0;

	// The live reference: prefill, then decode one step directly -- no save/restore at all.
	SinglePool ref_sp;
	sslm_seq ref_seq = nullptr;
	CHECK(MakeSinglePool(model, &ref_sp));
	if (ref_sp.pool) CHECK(sslm_seq_create(model, &ref_sp.pool, &ref_seq) == SSLM_OK);
	CHECK_MSG(ref_seq != nullptr, "T2260-R1: reference sequence create");
	if (!ref_seq) return;
	CHECK(sslm_prefill(model, ref_seq, prompt, 3, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
	      SSLM_OK);
	sslm_decode_params ref_params{};
	ref_params.struct_size = sizeof(ref_params);
	ref_params.layer_budget = 1;
	sslm_seq ref_batch[1] = {ref_seq};
	int32_t ref_token = -1;
	CHECK_MSG(sslm_decode_step(model, ref_batch, 1, &ref_params, nullptr, &ref_token) == SSLM_OK,
	          "T2260-R1: live reference decode step");

	// The save/restore arm: an identically-prefilled sequence, saved AT the ready-for-logits
	// resting point (before any decode_step call), restored into a fresh handle, then decoded.
	sslm_seq seq = nullptr;
	if (pool) CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK_MSG(seq != nullptr, "T2260-R1: save-arm sequence create");
	if (!seq) {
		CHECK(sslm_seq_release(ref_seq) == SSLM_OK);
		return;
	}
	CHECK(sslm_prefill(model, seq, prompt, 3, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	SeqBlobBuffer blob(model);
	CHECK_MSG(sslm_seq_save(seq, blob.bytes.data(), &blob.size) == SSLM_OK,
	          "T2260-R1: save at ready-for-logits");
	sslm_seq restored = nullptr;
	CHECK_MSG(sslm_seq_restore(model, pool, blob.bytes.data(), blob.size, &restored) == SSLM_OK,
	          "T2260-R1: restore");
	CHECK_MSG(restored != nullptr, "T2260-R1: restore produced a live handle");
	if (restored) {
		sslm_decode_params params{};
		params.struct_size = sizeof(params);
		params.layer_budget = 1;
		sslm_seq batch[1] = {restored};
		int32_t restored_token = -1;
		CHECK_MSG(sslm_decode_step(model, batch, 1, &params, nullptr, &restored_token) == SSLM_OK,
		          "T2260-R1: restored decode step");
		CHECK_MSG(restored_token == ref_token,
		          "T2260-R1: restored decode must match the live reference -- ref=%d restored=%d "
		          "(D-SLM4065's own defect: a restored post-prefill sequence used to carry "
		          "ready_for_logits=true over an all-zero residual, producing whatever token the "
		          "head weights map zero to)",
		          ref_token, restored_token);
		CHECK(sslm_seq_release(restored) == SSLM_OK);
	}
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_seq_release(ref_seq) == SSLM_OK);
}

// --- R2 (plan Sec6 safety net): a legacy-'SSB3'-shaped blob in the one unrecoverable state
// (fresh-post-prefill/adopt, current_token == the "no pending embed" sentinel) must be rejected
// loudly (SSLM_RESTORE_RESIDUAL_LOST) rather than silently restored. Constructed by hand from a
// REAL 'SSB4' blob (produced by this build's own sslm_seq_save): the fixed-header bytes at
// offset [4, 120) are BYTE-IDENTICAL between 'SSB4' and 'SSB3' by this fold's own design (only
// the magic and the trailing ready_for_logits field differ) -- so the legacy blob is magic
// 'SSB3' + those 116 bytes verbatim + (no ready_for_logits field, no residual bytes -- the
// pre-fix save path wrote zero residual bytes for exactly this state, which is the defect) +
// the real anti-LM history / kv_block_count / kv_blocks tail, copied verbatim from the real
// blob. ---
static void TestT2260_R2_LegacySsb3AffectedStateRejectedLoudly(sslm_model model,
                                                                sslm_kv_pool* pool) {
	int32_t prompt[2] = {0, 1};
	int32_t consumed = 0;
	sslm_seq seq = nullptr;
	if (pool) CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK_MSG(seq != nullptr, "T2260-R2: sequence create");
	if (!seq) return;
	CHECK(sslm_prefill(model, seq, prompt, 2, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	SeqBlobBuffer real_blob(model);
	CHECK_MSG(sslm_seq_save(seq, real_blob.bytes.data(), &real_blob.size) == SSLM_OK,
	          "T2260-R2: save the real 'SSB4' blob to transform");
	CHECK_MSG(real_blob.size >= 124, "T2260-R2: real blob at least covers the SSB4 fixed header");

	const uint8_t* real = real_blob.bytes.data();
	// Confirm this fixture actually reached the state R2 needs -- current_token's own sentinel
	// (-1) at its shared offset (72, identical in SSB3/SSB4), and layer_index == 0 (offset 68).
	const uint32_t layer_index = T2260ReadLE32(real + 68);
	const int32_t current_token = static_cast<int32_t>(T2260ReadLE32(real + 72));
	CHECK_MSG(layer_index == 0 && current_token == -1,
	          "T2260-R2: setup precondition -- must rest at ready-for-logits (layer_index=%u "
	          "current_token=%d)",
	          layer_index, current_token);

	const uint64_t anti_lm_history_count = T2260ReadLE64(real + 112);
	const size_t anti_lm_history_bytes = static_cast<size_t>(anti_lm_history_count) * 4;
	const size_t block_size = sslm_kv_block_size(model);
	// real_blob layout: [124 fixed header][residual][anti_lm history][4 kv_block_count][kv_blocks]
	CHECK_MSG(real_blob.size >= 124 + anti_lm_history_bytes + 4 + block_size,
	          "T2260-R2: real blob large enough to locate its own tail sections");
	const size_t tail_offset = real_blob.size - anti_lm_history_bytes - 4 - block_size;
	CHECK_MSG(tail_offset >= 124, "T2260-R2: derived residual region is non-negative");

	std::vector<uint8_t> legacy;
	legacy.push_back('S');
	legacy.push_back('S');
	legacy.push_back('B');
	legacy.push_back('3');
	// Fixed-header bytes [4, 120) -- byte-identical layout between 'SSB3' and 'SSB4'.
	legacy.insert(legacy.end(), real + 4, real + 120);
	// NO ready_for_logits field (that is the 'SSB4'-only addition), NO residual bytes (the
	// pre-fix 'SSB3' save path wrote zero for this exact state -- the defect this cell proves is
	// now caught). Anti-LM history + kv_block_count + kv_blocks, copied verbatim.
	legacy.insert(legacy.end(), real + tail_offset, real + real_blob.size);

	sslm_seq restored = nullptr;
	const sslm_status st = sslm_seq_restore(model, pool, legacy.data(), legacy.size(), &restored);
	CHECK_MSG(st == SSLM_RESTORE_RESIDUAL_LOST,
	          "T2260-R2: a legacy 'SSB3' blob in the affected state must reject "
	          "SSLM_RESTORE_RESIDUAL_LOST, not silently restore ready_for_logits=true over a "
	          "zeroed residual -- got status %d", static_cast<int>(st));
	CHECK_MSG(restored == nullptr, "T2260-R2: a rejected restore must not hand back a live handle");

	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// --- R3 (round-trip regression): 'SSB4' save/restore still works at a mid-token state and at a
// genuinely fresh/empty state; a real legacy 'SSB2'-shaped blob (hand-constructed the same way
// R2's 'SSB3' construction is, from a real 'SSB4' blob's own shared-layout header bytes) is
// still accepted -- the shipped SSB2-compatibility promise, unaffected by this fold. ---
static void TestT2260_R3_Ssb4RoundTripPlusLegacySsb2StillAccepted(sslm_model model,
                                                                   sslm_kv_pool* pool) {
	// Fresh/empty sequence, never prefilled -- layer_index == 0, context_length == 0.
	{
		sslm_seq seq = nullptr;
		if (pool) CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
		if (seq) {
			SeqBlobBuffer blob(model);
			CHECK_MSG(sslm_seq_save(seq, blob.bytes.data(), &blob.size) == SSLM_OK,
			          "T2260-R3: save a fresh/empty sequence");
			// Release the original before restoring -- this sub-case's own pool holds only ONE
			// block (main()'s own MakeSinglePool wiring for R3), so the original and the
			// restored handle cannot both be live at once.
			CHECK(sslm_seq_release(seq) == SSLM_OK);
			sslm_seq restored = nullptr;
			const sslm_status restore_st =
			    sslm_seq_restore(model, pool, blob.bytes.data(), blob.size, &restored);
			CHECK_MSG(restore_st == SSLM_OK,
			          "T2260-R3: restore a fresh/empty sequence -- got status %d",
			          static_cast<int>(restore_st));
			if (restored) CHECK(sslm_seq_release(restored) == SSLM_OK);
		}
	}
	// Mid-token: the existing C1 cell's own shape, re-run to confirm 'SSB4' still round-trips it.
	{
		SinglePool sp;
		sslm_seq seq = nullptr;
		if (MakePool(model, 2, &sp)) CHECK(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
		if (seq) {
			int32_t prompt[3] = {0, 1, 2};
			int32_t consumed = 0;
			CHECK(sslm_prefill(model, seq, prompt, 3, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
			      SSLM_OK);
			CHECK(EnterMidToken(model, seq));
			SeqBlobBuffer blob(model);
			CHECK_MSG(sslm_seq_save(seq, blob.bytes.data(), &blob.size) == SSLM_OK,
			          "T2260-R3: save mid-token");
			sslm_seq restored = nullptr;
			CHECK_MSG(sslm_seq_restore(model, &sp.pool, blob.bytes.data(), blob.size, &restored) ==
			              SSLM_OK,
			          "T2260-R3: restore mid-token");
			if (restored) CHECK(sslm_seq_release(restored) == SSLM_OK);
			CHECK(sslm_seq_release(seq) == SSLM_OK);
		}
		if (sp.pool) CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
	}
	// Legacy 'SSB2': hand-constructed from a real 'SSB4' mid-token blob -- 'SSB2' omits the
	// trailing anti_lm_order/anti_lm_history_count pair entirely (its own 108-byte fixed header,
	// vs 'SSB3'/'SSB4''s 120), so the legacy blob is magic 'SSB2' + the first 104 bytes of the
	// real header (offset [4, 108), identical layout to 'SSB3'/'SSB4' for every field SSB2 also
	// carries) + the residual (present, mid-token) + kv_block_count + kv_blocks -- no anti-LM
	// history, matching SSB2's own documented absence of that state.
	{
		SinglePool sp;
		sslm_seq seq = nullptr;
		if (MakePool(model, 2, &sp)) CHECK(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
		if (seq) {
			int32_t prompt[3] = {0, 1, 2};
			int32_t consumed = 0;
			CHECK(sslm_prefill(model, seq, prompt, 3, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
			      SSLM_OK);
			CHECK(EnterMidToken(model, seq));
			SeqBlobBuffer real_blob(model);
			CHECK_MSG(sslm_seq_save(seq, real_blob.bytes.data(), &real_blob.size) == SSLM_OK,
			          "T2260-R3: save the real 'SSB4' mid-token blob to transform into 'SSB2'");
			const uint8_t* real = real_blob.bytes.data();
			const uint64_t anti_lm_history_count = T2260ReadLE64(real + 112);
			CHECK_MSG(anti_lm_history_count == 0,
			          "T2260-R3: this fixture uses no damped-greedy state -- the 'SSB2' "
			          "construction below assumes zero anti-LM history to drop");
			const size_t block_size = sslm_kv_block_size(model);
			const size_t tail_offset = real_blob.size - 4 - block_size;  // kv_block_count + kv_blocks
			const size_t residual_offset = 124;  // fixed header(120) + ready_for_logits(4)
			CHECK_MSG(tail_offset >= residual_offset, "T2260-R3: derived residual region sane");

			std::vector<uint8_t> legacy;
			legacy.push_back('S');
			legacy.push_back('S');
			legacy.push_back('B');
			legacy.push_back('2');
			legacy.insert(legacy.end(), real + 4, real + 108);  // shared 104-byte header prefix
			legacy.insert(legacy.end(), real + residual_offset, real + tail_offset);  // residual
			legacy.insert(legacy.end(), real + tail_offset, real + real_blob.size);  // kv tail

			sslm_seq restored = nullptr;
			CHECK_MSG(sslm_seq_restore(model, &sp.pool, legacy.data(), legacy.size(), &restored) ==
			              SSLM_OK,
			          "T2260-R3: a real legacy 'SSB2'-shaped blob must still restore -- the "
			          "shipped SSB2-compatibility promise, unaffected by this fold");
			if (restored) CHECK(sslm_seq_release(restored) == SSLM_OK);
			CHECK(sslm_seq_release(seq) == SSLM_OK);
		}
		if (sp.pool) CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
	}
}

// REAL INVOCATION DRIVER (house pattern) -- supersedes the address-only convention. C1/C3 each
// get their OWN fresh sequence and pool (C1's own restored handle and C3's own attempted-but-
// rejected restore both need a real, bound pool per the buffer-mapping ruling); C2 needs only a
// pool to attempt the (rejected) restore into.
int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim9 C1-C3 not run");
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}
	std::vector<uint8_t> bytes;
	CHECK(ReadFileBytes(g_model_path, &bytes));
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
	if (model) {
		// C1 -- needs 2 concurrent blocks: seq stays live while restored is created from it.
		{
			SinglePool sp;
			sslm_seq seq = nullptr;
			if (MakePool(model, 2, &sp)) CHECK(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
			if (seq) {
				TestDim9_C1_SaveRestoreMidTokenRoundTripBitEqual(model, seq, &sp.pool);
				CHECK(sslm_seq_release(seq) == SSLM_OK);
			}
			if (sp.pool) CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
		}
		// C2
		{
			SinglePool sp;
			if (MakeSinglePool(model, &sp)) {
				TestDim9_C2_RestoreRejectsWellFormedGpuFormatBlobOnMagicMismatch(model, &sp.pool);
				CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
			}
		}
		// C3
		{
			SinglePool sp;
			sslm_seq seq = nullptr;
			if (MakeSinglePool(model, &sp)) CHECK(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
			if (seq) {
				TestDim9_C3_CorruptedMagicOnRealV1BlobRejectedBeforeFieldParsing(model, seq,
				                                                                 &sp.pool);
				CHECK(sslm_seq_release(seq) == SSLM_OK);
			}
			if (sp.pool) CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
		}
		// T-2260 (D-SLM4073): R1/R2 each need their own dedicated pool (R1 needs two concurrent
		// blocks -- the save-arm sequence and its restored handle -- alongside R1's own separate
		// live-reference pool; R2 needs one, its own restore attempt is rejected so nothing extra
		// is ever drawn from it).
		{
			SinglePool sp;
			if (MakePool(model, 2, &sp)) {
				TestT2260_R1_ReadyForLogitsSaveRestoreMatchesLiveContinuation(model, &sp.pool);
				CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
			}
		}
		{
			SinglePool sp;
			if (MakeSinglePool(model, &sp)) {
				TestT2260_R2_LegacySsb3AffectedStateRejectedLoudly(model, &sp.pool);
				CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
			}
		}
		{
			SinglePool sp;
			if (MakeSinglePool(model, &sp)) {
				TestT2260_R3_Ssb4RoundTripPlusLegacySsb2StillAccepted(model, &sp.pool);
				CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
			}
		}
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
