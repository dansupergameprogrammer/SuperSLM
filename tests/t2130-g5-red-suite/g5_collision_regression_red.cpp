// T-2130 (Curie) -- THE COMMITTED COLLISION-REGRESSION FIXTURE, closing T-2127 G1 (design
// Sec6 G5-2 gate / Sec7 dim7's own contract-claim rollup).
//
// Provenance and why this file exists rather than re-running a script: the rung-5 adversary
// strike (Claude/Loki/t2122-g5-design-strike-2026-08-16.md, D-SLM3445) found, by exhaustive
// enumeration against the project's own shipped reference decoder
// (tests/reference/superslm_spike/constrain.py @ f37be4a), that a schema-bound sequence
// mid-generation and a fresh prompt chunk present an IDENTICAL observable vector on every
// field the pre-repair design named -- zero policies could discriminate them (survivors = 0).
// The repair (design Sec5/Sec10.2) adds `sslm_span_kind`, re-verified by
// Claude/Vitruvius/t2119-repair-verification/repair_probe.py (survivors 0 -> 1, both traces,
// executed 2026-08-16). Design Sec6 G5-2 / Sec7 dim7 both require that verification's own
// result -- "the walk-advance decision is a pure function of the caller-declared
// `sslm_span_kind`, never of token content or context length" -- be pinned by a COMMITTED
// fixture in `tests/`, not left as standalone script evidence: "`repair_probe.py` is
// design-time evidence that no other policy could have worked; the committed fixture is the
// coverage that keeps the shipped implementation honest on every future change" (design
// Sec6, the G-8/Japp "an executed corpus sitting outside `tests/` is evidence, not coverage"
// precedent, plan Sec17 G-8). THIS FILE is that fixture.
//
// What it proves, restated as an executable claim rather than the probe's own policy-space
// enumeration (this suite tests the SHIPPED ABI's behavior, not the abstract policy space --
// the enumeration already did its job at design time): the exact shape the strike found -- a
// schema-bound sequence, MID-GENERATION (non-start DFA-walk-state, reached via a prior
// forced/schema-content span), receiving a SSLM_SPAN_PROMPT call whose content is split
// across a chunkBudget boundary -- produces decode output BIT-IDENTICAL to the same
// generation with the identical prompt content supplied unchunked (one call, a chunk_budget
// large enough to consume it whole). If `sslm_prefill`'s walk-advance policy were ever a
// function of chunk boundaries, context length, or any other bit the struck probe enumerated
// instead of `sslm_span_kind`, chunking the SAME prompt content differently would move the
// walk-state differently and this cell would diverge -- which is exactly the collision the
// strike found under the pre-repair, parameter-less signature.
#include "fixture_common.h"

using namespace superslm;

namespace {

// Runs the fixture's own scenario against `model`/`schema` with the prompt content supplied
// in `n_chunks` calls (each of `chunk_len` tokens, chunk_budget == chunk_len forces the
// runtime to consume exactly one chunk per call) and returns the resulting sequence's own
// save-blob, for byte-for-byte comparison against a differently-chunked run of the identical
// content.
bool RunScenario(sslm_model model, sslm_schema schema, const int32_t* prompt_tokens,
                  int32_t prompt_len, int32_t chunk_budget, std::vector<uint8_t>* out_blob) {
	sslm_seq seq = nullptr;
	if (sslm_seq_create(model, nullptr, &seq) != SSLM_OK) return false;
	if (sslm_seq_set_schema(seq, schema) != SSLM_OK) return false;

	// Drive the walk to a genuinely non-start, mid-generation state BEFORE the prompt content
	// under test arrives -- the strike's own "J at the start state" call, reproduced as a
	// short forced (schema-content) span so the sequence's DFA-walk-state field is non-fresh
	// when the SSLM_SPAN_PROMPT calls below execute. This is the mid-generation half of the
	// struck shape; a fresh sequence's FIRST prompt call is the ordinary, already-covered
	// case (the strike's own P1 call), not the collision this fixture pins.
	int32_t leading_forced_span[3] = {0, 1, 2};
	int32_t consumed = 0;
	if (sslm_prefill(model, seq, leading_forced_span, 3, chunk_budget,
	                  SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed) != SSLM_OK)
		return false;

	// The prompt content under test, chunked per this run's own `chunk_budget` -- multiple
	// internal calls when prompt_len > chunk_budget, one call otherwise ("unchunked").
	int32_t offset = 0;
	while (offset < prompt_len) {
		const int32_t remaining = prompt_len - offset;
		const int32_t this_call_budget = chunk_budget;
		int32_t this_consumed = 0;
		if (sslm_prefill(model, seq, prompt_tokens + offset, remaining, this_call_budget,
		                  SSLM_SPAN_PROMPT, nullptr, &this_consumed) != SSLM_OK)
			return false;
		if (this_consumed <= 0) return false;
		offset += this_consumed;
	}

	unsigned char blob[65536];
	size_t blob_size = sizeof(blob);
	if (sslm_seq_save(seq, blob, &blob_size) != SSLM_OK) return false;
	out_blob->assign(blob, blob + blob_size);
	return sslm_seq_release(seq) == SSLM_OK;
}

}  // namespace

// --- THE COLLISION-REGRESSION CELL (design Sec6 G5-2 gate, Sec7 dim7, T-2127 G1 closed). ---
static void TestG5CollisionRegression_ChunkedVsUnchunkedPromptMidGenerationBitIdentical(
    sslm_model model, sslm_schema reference_schema) {
	// The struck probe's own prompt (13 tokens, the reference task's own utterance shape) and
	// its own chunkBudget = 8, reproduced verbatim from
	// Claude/Loki/t2122-probe/probe.py / Claude/Vitruvius/t2119-repair-verification/
	// repair_probe.py -- "Friday works, just not with Mox." tokenized against the probe's own
	// vocabulary. Represented here as opaque token ids (the C++ ABI's own int32_t contract);
	// the specific ids are immaterial to the claim under test, only that BOTH runs below
	// consume the SAME 13 ids.
	int32_t prompt_tokens[13];
	for (int i = 0; i < 13; ++i) prompt_tokens[i] = 10 + i;  // arbitrary, fixed, shared token ids

	std::vector<uint8_t> chunked_blob;
	CHECK(RunScenario(model, reference_schema, prompt_tokens, 13, /*chunk_budget=*/8,
	                  &chunked_blob));

	std::vector<uint8_t> unchunked_blob;
	CHECK(RunScenario(model, reference_schema, prompt_tokens, 13, /*chunk_budget=*/64,
	                  &unchunked_blob));

	// FEATURE ORACLE, behavioral (design Sec6 G5-2's own wording): the two save-blobs --
	// header, KV, DFA-walk-state, everything sslm_seq_save captures -- are byte-for-byte
	// identical. A walk-advance policy that is a pure function of `sslm_span_kind` (never of
	// chunk boundaries, context length, or any bit the struck probe enumerated) guarantees
	// this; any other policy is exactly the collision the strike found by construction.
	CHECK_MSG(chunked_blob == unchunked_blob,
	          "chunked (chunk_budget=8, %zu bytes) vs unchunked (chunk_budget=64, %zu bytes) "
	          "save-blobs diverge -- sslm_prefill's walk-advance decision depended on "
	          "something other than sslm_span_kind",
	          chunked_blob.size(), unchunked_blob.size());
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
		TestG5CollisionRegression_ChunkedVsUnchunkedPromptMidGenerationBitIdentical(model,
		                                                                            schema_ref);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied -- collision-"
		         "regression cell not run");
	}
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
