// T-2130 (Curie) -- Dim 9 (Persistence round-trip and version evolution), design Sec7 dim9
// (extended this fold, T-2121 F1, closed). Mask-page sections (artifact-level) are versioned
// exactly as every other .sslm section is, per the general artifact-evolution cell (plan
// Sec17 dim9) -- design Sec7 dim9 states this half "was already correct" and owes no new
// cell here. The cell that WAS missing is the sequence-level save-blob's own schema-binding
// and DFA-walk-state round-trip. 1 cell, plus the restore-rejection half already exercised in
// dim2 (cross-cited, not duplicated). RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec5/Sec7 dim9, the cell that was missing): the save-blob's
// schema-binding field and DFA-walk-state field round-trip bit-equal through
// sslm_seq_save/sslm_seq_restore -- symmetric with the adapter's own dim-9 cell ("the
// save-blob's adapter-binding field round-trips bit-equal"). The restore-time rejection for
// an unresolvable binding (SSLM_RESTORE_SCHEMA_MISMATCH) is exercised in
// dim2_hostile_red.cpp's own mechanism cell 4, cross-cited here rather than duplicated. ---
static void TestDim9_M1_SaveBlobSchemaBindingAndWalkStateRoundTripBitEqual(
    sslm_model model, sslm_schema reference_schema) {
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, reference_schema) == SSLM_OK);
	// Advance the walk state to something non-start/non-trivial before the round-trip, so a
	// round-trip that silently resets it (rather than preserving it) is observable.
	int32_t forced_tokens[3] = {0, 1, 2};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, forced_tokens, 3, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr,
	                    &consumed) == SSLM_OK);
	unsigned char blob[65536];
	size_t blob_size = sizeof(blob);
	CHECK(sslm_seq_save(seq, blob, &blob_size) == SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);

	sslm_seq restored = nullptr;
	CHECK(sslm_seq_restore(model, nullptr, blob, blob_size, &restored) == SSLM_OK);

	// FEATURE ORACLE: the restored sequence's own schema binding resolves to reference_schema
	// (re-binding is REJECTED -- sslm_seq_set_schema requires a fresh/reset walk state,
	// proving the binding survived the round-trip rather than being cleared) and its
	// DFA-walk-state continues correctly: the next forced token the schema's own DFA admits
	// from the mid-walk state is accepted, not rejected as though the walk had reset to start.
	CHECK(sslm_seq_set_schema(restored, reference_schema) == SSLM_SCHEMA_BIND_REJECTED);
	int32_t next_forced_token[1] = {3};
	int32_t consumed2 = 0;
	CHECK(sslm_prefill(model, restored, next_forced_token, 1, 8, SSLM_SPAN_SCHEMA_CONTENT,
	                    nullptr, &consumed2) == SSLM_OK);

	// And a SECOND save immediately after restore reproduces the identical blob bytes --
	// bit-equal round-trip, not merely "restore succeeds".
	unsigned char blob2[65536];
	size_t blob2_size = sizeof(blob2);
	CHECK(sslm_seq_save(restored, blob2, &blob2_size) == SSLM_OK);
	CHECK(blob2_size == blob_size);
	CHECK(std::memcmp(blob, blob2, blob_size) == 0);
	CHECK(sslm_seq_release(restored) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	std::vector<uint8_t> model_bytes;
	sslm_model model = nullptr;
	const bool have_model = TryMapRealModel(g_model_1p5b_path, &model_bytes, &model);
	sslm_schema schema_ref = nullptr;
	const bool have_schema_ref =
	    have_model && TryLookupSchema(model, g_reference_schema_name.c_str(), &schema_ref);
	if (have_schema_ref) {
		TestDim9_M1_SaveBlobSchemaBindingAndWalkStateRoundTripBitEqual(model, schema_ref);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied -- mechanism cell 1 "
		         "not run");
	}
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
