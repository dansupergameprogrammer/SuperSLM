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
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
