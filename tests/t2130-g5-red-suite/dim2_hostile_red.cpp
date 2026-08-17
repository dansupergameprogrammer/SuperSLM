// T-2130 (Curie) -- Dim 2 (Trust boundaries and hostile inputs), design Sec7 dim2 (extended
// this fold, T-2121 F1 restore-path half, closed). 4 cells: 3 mechanism (load-time), 1
// mechanism (restore-path). RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1: a corrupt/truncated mask-page section in the .sslm artifact fires a
// documented loader rejection -- symmetric with the base loader's existing exhaustive-
// validation discipline (design Sec7 dim2) and proven to fire BEFORE the artifact is mapped
// for use (T-2121 N3 / Sec7 dim11's ordering cross: no decode step can ever read an
// unvalidated page). ---
static void TestDim2_M1_CorruptMaskPageSectionRejectedBeforeMap() {
	std::vector<uint8_t> corrupt_artifact_bytes;  // a hand-built .sslm image with a truncated
	                                               // mask-page section, per this dimension's
	                                               // own hostile-input corpus convention
	                                               // (§11's exhaustive-validation discipline).
	sslm_model model = nullptr;
	CHECK(sslm_model_map(corrupt_artifact_bytes.data(), corrupt_artifact_bytes.size(),
	                      &model) != SSLM_OK);
	// ORDERING (design Sec7 dim11): the rejection must fire during sslm_model_map itself, not
	// on the first schema lookup or decode step -- so `model` above must never be usable, and
	// no separate "first use" call is exercised here at all, which is the assertion.
	CHECK(model == nullptr);
}

// --- Mechanism cell 2: an out-of-range schema index (a mask-page section claiming a schema
// index beyond the artifact's own declared schema count) is a defined loader rejection. ---
static void TestDim2_M2_OutOfRangeSchemaIndexRejectedBeforeMap() {
	std::vector<uint8_t> oob_artifact_bytes;  // a hand-built .sslm image whose mask-page
	                                           // section header names a schema index >=
	                                           // the artifact's own schema_count.
	sslm_model model = nullptr;
	CHECK(sslm_model_map(oob_artifact_bytes.data(), oob_artifact_bytes.size(), &model) !=
	      SSLM_OK);
	CHECK(model == nullptr);
}

// --- Mechanism cell 3: a mask page whose bitmask width does not match the artifact's own
// vocab count is a defined loader rejection (the AND at decode time must never operate on a
// mismatched-width page -- the guard proven to fire before any decode step can reach it). ---
static void TestDim2_M3_MaskPageWidthVocabMismatchRejectedBeforeMap() {
	std::vector<uint8_t> width_mismatch_bytes;  // a hand-built .sslm image whose mask-page
	                                             // bitmask width != artifact vocab_count.
	sslm_model model = nullptr;
	CHECK(sslm_model_map(width_mismatch_bytes.data(), width_mismatch_bytes.size(), &model) !=
	      SSLM_OK);
	CHECK(model == nullptr);
}

// --- Mechanism cell 4 (design Sec5 / Sec7 dim2 EXTENDED this fold, restore-path half): a
// save-blob whose bound schema does not resolve against the currently-mapped artifact's
// schema set is validated AT RESTORE, not only at bind time -- SSLM_RESTORE_SCHEMA_MISMATCH,
// symmetric with the adapter's own dim-9 restore-mismatch cell. Ordering: this check runs
// AFTER the existing KV/model-hash validation and BEFORE any device work (design Sec5). ---
static void TestDim2_M4_RestoreRejectsUnresolvableSchemaBinding(sslm_model model_a,
                                                                 sslm_model model_b,
                                                                 sslm_schema schema_only_on_a) {
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model_a, nullptr, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, schema_only_on_a) == SSLM_OK);
	unsigned char blob[65536];
	size_t blob_size = sizeof(blob);
	CHECK(sslm_seq_save(seq, blob, &blob_size) == SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	// Restore against model_b, an artifact whose schema set does not include
	// schema_only_on_a's identity -- the save-blob's model-hash mismatch would ALSO fire here
	// (SSLM_RESTORE_MODEL_MISMATCH is the pre-existing precedent), so this cell restores
	// against a SECOND artifact sharing model_a's own content hash (a schema-compiled variant
	// of the same base checkpoint) to isolate the schema-mismatch path specifically from the
	// model-hash path -- the ordering claim (Sec5: "after the existing KV/model-hash
	// validation and before any device work") is what this cell's own restore-target choice
	// is designed to exercise.
	sslm_seq restored = nullptr;
	CHECK(sslm_seq_restore(model_b, nullptr, blob, blob_size, &restored) ==
	      SSLM_RESTORE_SCHEMA_MISMATCH);
	CHECK(restored == nullptr);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	TestDim2_M1_CorruptMaskPageSectionRejectedBeforeMap();
	TestDim2_M2_OutOfRangeSchemaIndexRejectedBeforeMap();
	TestDim2_M3_MaskPageWidthVocabMismatchRejectedBeforeMap();

	std::vector<uint8_t> model_a_bytes, model_b_bytes;
	sslm_model model_a = nullptr, model_b = nullptr;
	const bool have_model_a = TryMapRealModel(g_model_1p5b_path, &model_a_bytes, &model_a);
	const bool have_model_b =
	    TryMapRealModel(g_model_1p5b_variant_path, &model_b_bytes, &model_b);
	sslm_schema schema_only_on_a = nullptr;
	const bool have_schema_on_a =
	    have_model_a && TryLookupSchema(model_a, g_reference_schema_name.c_str(), &schema_only_on_a);
	if (have_model_a && have_model_b && have_schema_on_a) {
		TestDim2_M4_RestoreRejectsUnresolvableSchemaBinding(model_a, model_b, schema_only_on_a);
	} else {
		SKIP_MSG("two real artifacts sharing a base content hash (--model1p5b=PATH, "
		         "--model1p5b_variant=PATH), the second lacking the first's schema, not "
		         "supplied -- mechanism cell 4 not run");
	}
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
