// T-2916 (Curie) -- TE-370 S3's cells: the SLM5 restore validation's three bounds checks
// (`Claude/Poirot/te370-final-code-review-2026-09-21.md` Sec2 S3, Wizard repo) are correct at
// source (`src/gpu/gpu_1p0.cpp:2200-2224`, `sslm_gpu_seq_restoreImpl`) but unpinned: no test in
// the range feeds a forged tail, so a deletion of any one of them would leave every suite in the
// range green. This cell feeds all three forged tails directly.
//
// GREEN AT 0062c99 (checks exist): tampering a genuine, freshly-saved SLM5 blob's own v5 tail
// (`Claude/Poirot/...`'s own cited offsets: int32 bound_schema_index, uint32 dfa_walk_state,
// uint32 ready_for_logits -- immediately after the fixed-size v4 header,
// `src/gpu/superslm_gpu.cpp:4290-4298`) at each of the three checks' own trigger conditions
// restores `SSLM_SEQUENCE_KV_BUFFER_MISMATCH` today. Proven by MUTANT, not by reading the source:
// `make_mut_slm5_bounds.py` deletes one check at a time from a scratch copy of `gpu_1p0.cpp`
// (structural match, refuses on anything but exactly one hit), and this SAME cell binary, linked
// against each mutant instead of the tree's own object, must go RED (a tampered blob that should
// have been refused instead restores `SSLM_OK`, or a status other than the documented refusal).
//
// TAIL OFFSETS. Not read from any private header (`GpuSeqBlobHeader` lives in an anonymous
// namespace in superslm_gpu.cpp, inaccessible to a test translation unit) -- computed by
// COMPILING a field-for-field mirror of that struct (D:/_t2916/probe/sizeof_check.cpp, executed):
// sizeof(header)=120, so the v5 tail is bound_schema_index@120 (int32), dfa_walk_state@124
// (uint32), ready_for_logits@128 (uint32). If the real header's layout ever changes, `kSanity*`
// below (a full, untampered round trip through the SAME offsets) catches the drift before any
// tamper cell is trusted -- it is the first thing this file asserts.
//
// Run: cell_gpu_slm5_bounds_tamper.exe --model=PATH
#include "fixture_common.h"

namespace {

constexpr size_t kHeaderSize = 120;                    // verified by compilation, see header comment
constexpr size_t kTailSchemaIndexOffset = kHeaderSize;  // int32, offset 120
constexpr size_t kTailWalkStateOffset = kHeaderSize + 4;  // uint32, offset 124
constexpr size_t kTailReadyOffset = kHeaderSize + 8;      // uint32, offset 128
constexpr uint32_t kUnusedWalkSentinel = 0xFFFFFFFFu;

void WriteLE(std::vector<uint8_t>& blob, size_t offset, const void* value, size_t n) {
	CHECK_MSG(offset + n <= blob.size(), "tamper offset %zu+%zu exceeds blob size %zu -- the blob "
	          "is smaller than this file's own hardcoded tail layout expects (kSanity* below "
	          "would already have failed)",
	          offset, n, blob.size());
	if (offset + n <= blob.size()) std::memcpy(blob.data() + offset, value, n);
}
void WriteSchemaIndex(std::vector<uint8_t>& blob, int32_t v) { WriteLE(blob, kTailSchemaIndexOffset, &v, sizeof(v)); }
void WriteWalkState(std::vector<uint8_t>& blob, uint32_t v) { WriteLE(blob, kTailWalkStateOffset, &v, sizeof(v)); }

struct GDec {
	SslmGpuStatus st = SSLM_OK;
	int32_t out = -1;
};
GDec Decode(const GpuModelFixture& fx, SslmGpuSequenceHandle* s, int32_t tok) {
	GDec d;
	d.st = SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, s, tok, fx.one_layer_budget, &d.out);
	return d;
}

// A schema-bound sequence with real, non-zero schema-content progress -- the P2 construction
// from `cell_gpu_slm5_saverestore.cpp::Build(pt=2)`, reconciled to D-SLM7625:
// bind schema 0, prefill the prompt, consume one legal schema-content token, then
// two decode steps.
SslmGpuSequenceHandle* BuildBoundProgressed(const GpuModelFixture& fx, const IndependentSchema& ds,
                                             const std::vector<int32_t>& P, int32_t other) {
	SslmGpuSequenceHandle* s = nullptr;
	CHECK(sslm_gpu_seq_create(fx.ctx, fx.model, fx.model_cap, &s) == SSLM_OK && s);
	if (!s) return nullptr;
	CHECK(SslmGpuSeqSetSchemaForG5Bridge(fx.ctx, s, 0) == SSLM_OK);
	CHECK(Prefill(fx, s, P) == SSLM_OK);
	const int32_t t0 = ds.FirstLegal(0);
	int32_t consumed = -1;
	CHECK(SslmGpuSeqPrefillSchemaContentForG5Bridge(fx.ctx, s, &t0, 1, fx.one_layer_budget, &consumed) == SSLM_OK && consumed == 1);
	int32_t last = -1;
	auto step = [&]() {
		GDec d = Decode(fx, s, last >= 0 ? last : other);
		CHECK_MSG(d.st == SSLM_OK, "P2 decode returned %s", StatusName(d.st));
		if (d.st == SSLM_OK && d.out >= 0) last = d.out;
	};
	step();
	step();
	CHECK_MSG(SslmGpuSeqWalkStateForG5Bridge(s) == 1u,
	          "P2 walk=%u, want progressed state 1", SslmGpuSeqWalkStateForG5Bridge(s));
	return s;
}

std::vector<uint8_t> Save(const GpuModelFixture& fx, SslmGpuSequenceHandle* s) {
	size_t need = 0;
	CHECK_MSG(sslm_gpu_seq_save(fx.ctx, s, nullptr, &need) == SSLM_DEVICE_LOST && need > 0,
	          "P2 save size probe did not return a nonzero size");
	std::vector<uint8_t> blob(need);
	size_t n = need;
	CHECK(sslm_gpu_seq_save(fx.ctx, s, blob.data(), &n) == SSLM_OK);
	return blob;
}

}  // namespace

int main(int argc, char** argv) {
	std::string model_path;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--model=", 0) == 0) model_path = a.substr(8);
	}
	if (model_path.empty()) {
		std::printf("SKIP cell_gpu_slm5_bounds_tamper -- needs --model=PATH\n");
		std::printf("checks=0 failures=0 skips=1\n");
		return 0;
	}

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	GpuModelFixture fx;
	CHECK_MSG(fx.Open(model_path, ctx), "GPU open failed");
	IndependentSchema ds;
	CHECK(ds.Build(fx.bytes, "g5_minimal_one_field"));
	if (GFailures > 0 || !ds.entry) {
		std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
		return 1;
	}
	const std::vector<int32_t> P = fx.TokensP();
	const int32_t other = fx.OtherToken();

	// --- Build one genuine, bound, progressed original and save it -- the untampered blob every
	// tamper cell below starts from. ---
	SslmGpuSequenceHandle* original = BuildBoundProgressed(fx, ds, P, other);
	if (!original || GFailures) {
		std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
		return 1;
	}
	std::vector<uint8_t> good_blob = Save(fx, original);
	sslm_gpu_seq_release(ctx, original);

	// kSanity1/2: the offsets this file hardcodes actually name the real tail -- an untampered
	// blob, saved and restored through the SAME buffer this file tampers, must round-trip at
	// SSLM_OK with the SAME schema/walk it was saved with. If the header layout ever drifts, this
	// is what catches it, before any tamper cell below is trusted at all.
	{
		std::vector<uint8_t> blob_copy = good_blob;  // untouched -- the tamper helpers below are
		                                              // never called on this copy
		SslmGpuSequenceHandle* r = nullptr;
		const SslmGpuStatus rs = sslm_gpu_seq_restore(ctx, fx.model, blob_copy.data(), blob_copy.size(), &r);
		CHECK_MSG(rs == SSLM_OK && r, "kSanity1: untampered blob failed to restore (%s) -- the "
		          "fixture/construction is broken, not the checks under test", StatusName(rs));
		if (r) {
			const uint32_t restored_walk = SslmGpuSeqWalkStateForG5Bridge(r);
			CHECK_MSG(restored_walk != kUnusedWalkSentinel && restored_walk != 0u,
			          "kSanity2: the original must have real schema-content progress (walk=%u) "
			          "for the walk>=state_count tamper below to have somewhere real to land",
			          restored_walk);
			sslm_gpu_seq_release(ctx, r);
		}
	}

	// --- S3 check 1: an out-of-range restored_schema_index. Any value larger than any real
	// model's own schema count refuses.
	//
	// T-2916 FOUND, T-2917 ACTED ON (D-SLM7600): the explicit `>= model->schemas.Count()`
	// pre-check this input used to exercise had its own deletion mutant survive (T-2916,
	// `make_mut_slm5_bounds.py INDEXCOUNT`, since removed from that generator) -- provably dead,
	// not merely redundant: `SchemaMasksTable::ByIndex(size_t)` (include/superslm/schema_masks.h)
	// is itself bounds-safe (`index < entries_.size() ? &entries_[index] : nullptr`), and EVERY
	// consumer of `bound_schema_index` in this file (this restore path, plus lines
	// ~2258/2475/3214) routes through `ByIndex` and null-checks the result. An index >= Count()
	// can, by that contract, never make `ByIndex` return non-null. The conductor ruled the check
	// removed and replaced with a comment naming the null-check as the bound (`gpu_1p0.cpp`,
	// `sslm_gpu_seq_restoreImpl`); this assertion is UNCHANGED and now exercises check 2's own
	// `!resolved_entry` guard directly -- kept as a behavioral regression pin (an out-of-range
	// index is always refused, whichever guard is doing the refusing), not as a mutation-proof
	// of a specific line, since no such line exists here to mutate any more. ---
	{
		std::vector<uint8_t> blob = good_blob;
		WriteSchemaIndex(blob, 999999);
		SslmGpuSequenceHandle* r = nullptr;
		const SslmGpuStatus rs = sslm_gpu_seq_restore(ctx, fx.model, blob.data(), blob.size(), &r);
		CHECK_MSG(rs == SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
		          "S3 check 1 (index >= schemas.Count()): forged schema_index=999999 must be "
		          "refused with SSLM_SEQUENCE_KV_BUFFER_MISMATCH, got %s", StatusName(rs));
		CHECK_MSG(!r, "S3 check 1: a refused restore must not also hand back a live handle");
		if (r) sslm_gpu_seq_release(ctx, r);
	}

	// --- S3 check 2: dfa_walk_state >= resolved_entry->state_count, with a VALID schema index
	// (0) -- isolates this check from check 1 (both must never fire from the same tamper). ---
	{
		std::vector<uint8_t> blob = good_blob;
		WriteSchemaIndex(blob, 0);  // valid -- this schema's own real index
		WriteWalkState(blob, ds.entry->state_count);  // exactly at the boundary (>=)
		SslmGpuSequenceHandle* r = nullptr;
		const SslmGpuStatus rs = sslm_gpu_seq_restore(ctx, fx.model, blob.data(), blob.size(), &r);
		CHECK_MSG(rs == SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
		          "S3 check 2 (walk >= state_count): forged walk_state=%u (== this schema's own "
		          "state_count) with a VALID schema index must be refused with "
		          "SSLM_SEQUENCE_KV_BUFFER_MISMATCH, got %s", ds.entry->state_count, StatusName(rs));
		CHECK_MSG(!r, "S3 check 2: a refused restore must not also hand back a live handle");
		if (r) sslm_gpu_seq_release(ctx, r);
	}

	// --- S3 check 3: an unbound blob (schema_index < 0) carrying a non-sentinel walk -- the
	// symmetric malformation check. ---
	{
		std::vector<uint8_t> blob = good_blob;
		WriteSchemaIndex(blob, -1);
		WriteWalkState(blob, 5u);  // any real state id, deliberately not the unused sentinel
		SslmGpuSequenceHandle* r = nullptr;
		const SslmGpuStatus rs = sslm_gpu_seq_restore(ctx, fx.model, blob.data(), blob.size(), &r);
		CHECK_MSG(rs == SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
		          "S3 check 3 (unbound blob, non-sentinel walk): forged schema_index=-1 with "
		          "walk_state=5 (not the 0xFFFFFFFF sentinel) must be refused with "
		          "SSLM_SEQUENCE_KV_BUFFER_MISMATCH, got %s", StatusName(rs));
		CHECK_MSG(!r, "S3 check 3: a refused restore must not also hand back a live handle");
		if (r) sslm_gpu_seq_release(ctx, r);
	}

	// Positive control, symmetric with kSanity above: an unbound blob carrying the CORRECT
	// sentinel is not a malformation and must restore cleanly (proves check 3 is about the
	// sentinel mismatch, not about unbound blobs in general).
	{
		std::vector<uint8_t> blob = good_blob;
		WriteSchemaIndex(blob, -1);
		WriteWalkState(blob, kUnusedWalkSentinel);
		SslmGpuSequenceHandle* r = nullptr;
		const SslmGpuStatus rs = sslm_gpu_seq_restore(ctx, fx.model, blob.data(), blob.size(), &r);
		CHECK_MSG(rs == SSLM_OK && r, "positive control: unbound blob WITH the correct sentinel "
		          "must restore cleanly, got %s", StatusName(rs));
		if (r) sslm_gpu_seq_release(ctx, r);
	}

	fx.Close();
	sslm_gpu_context_destroy(ctx);
	std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}
