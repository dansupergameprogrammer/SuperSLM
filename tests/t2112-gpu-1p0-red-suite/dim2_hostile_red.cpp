// T-2112 (Curie) -- Dim 2 (Trust boundaries and hostile inputs), design Sec11 dim2. 3 cells,
// +1 (T-2114, S3) -- see that cell's own header comment. +2 (T-2113 P2, design Sec22, D-SLM3415,
// `TestDim2_P2_ForeignBlobRestoreModelMismatch`/`TestDim2_P3_SameShapeDifferentContentRestore
// ModelMismatch`) -- authored by THIS BUILD SEAT per the design fold's own explicit instruction
// ("the build seat owes ... the two new dim-2 cells in the T-2112 suite"), the one named exception
// to Brunel's own "does not author the test suite" boundary. +1 (T-2124, D-SLM3446 P1-4,
// `TestDim2_M4_CrossContextHandleRejectedAtEveryBoundary`) -- the external-review cross-context
// validation gap, authored by the build seat under the SAME routed exception this file's own P2/P3
// cells already establish (a fix round routes its own pin in the same round, per this project's
// standing rule).
// RED BY LINK (see dim1_lifetime_red.cpp's own header for the shared explanation).
#include "fixture_common.h"
#include "../sslm_model_hostile_fixtures.h"
#include "../sslm_kvc1_hostile_fixtures.h"

using namespace superslm;

// --- Mechanism cell 1: every new guard in Sec9's table has a hostile-input cell that forces it.
static void TestDim2_M1_NewGuardsForcedByHostileInputs(SslmGpuContext* ctx,
                                                        SslmGpuModelHandle* model_a,
                                                        SslmGpuModelHandle* model_b,
                                                        SslmGpuSequenceHandle* seq_bound_to_a,
                                                        const SslmModelView* base_hash_mismatch_adapter) {
	// AdapterModelMismatch: an adapter mapped against model_b passed to a sequence bound to
	// model_a (design Sec5.2, Sec9).
	SslmGpuAdapterHandle* adapter_for_b = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model_b, base_hash_mismatch_adapter, &adapter_for_b) != SSLM_OK
	      || adapter_for_b != nullptr);
	if (adapter_for_b) {
		CHECK(sslm_decode_step_gpu(ctx, seq_bound_to_a, adapter_for_b, 24u) ==
		      SSLM_ADAPTER_MODEL_MISMATCH);
	}
	// AdapterBaseHashMismatch: an adapter whose base-hash does not match model_a (B0b's existing
	// hostile-fixture population, reused rather than invented, design Sec2 dim2 mechanism).
	SslmGpuAdapterHandle* hostile = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model_a, base_hash_mismatch_adapter, &hostile) ==
	      SSLM_ADAPTER_BASE_HASH_MISMATCH);
	CHECK(hostile == nullptr);
	// ContextHasLiveHandles: destroying ctx while model_a/model_b/seq_bound_to_a are still live.
	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_CONTEXT_HAS_LIVE_HANDLES);
}

// --- Mechanism cell 2 (N2 repair, 2026-08-15: renamed -- the prior name, "NineGuardLadder...",
// claimed all nine gpu_layer_loop_guards.def guards were reproduced through the new entry point;
// the body only ever forced ONE (DispatchBudgetTooSmall). Renamed to what the body actually
// delivers; the other eight guards' own carry-forward through sslm_decode_step_gpu specifically
// (as opposed to their EXISTENCE, which check_gpu_guard_status_parity.py's own structural
// derivation already confirms, and their internal correctness, which tests/test_main.cpp already
// covers against RunLayerLoopGpu directly) is a genuine, un-swept gap -- named here rather than
// silently claimed covered by a name promising more than nine cases' worth of assertion.) ---
static void TestDim2_M2_OneRepresentativeGuardCarriedForwardThroughDecodeStepGpu(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* seq_with_oversized_context) {
	// A malformed budget below the whole-layer floor is the cheapest carried-forward guard to
	// force through the new entry point (DispatchBudgetTooSmall, unchanged from the substrate).
	// ROUTED, not fixed here: the other eight of gpu_layer_loop_guards.def's own nine guards each
	// need their own hostile fixture to force through sslm_decode_step_gpu specifically -- larger
	// scope than this repair pass, a follow-up coverage task for whoever next extends this file.
	CHECK(sslm_decode_step_gpu(ctx, seq_with_oversized_context, nullptr, /*dispatch_budget=*/1u) ==
	      SSLM_DISPATCH_BUDGET_TOO_SMALL);
}

// --- Mechanism cell 4 (T-2124, D-SLM3446 P1-4, external review): every public entry point that
// takes both a context and a handle validates the handle was actually created/mapped against THAT
// context -- before this fix, `model->ctx == ctx` (and the adapter/sequence analogues) was never
// checked anywhere, so a caller could bind a sequence to a model mapped on a DIFFERENT context (a
// different D3D12 device) with no rejection. `ctx_b`/`model_b` are a second, independent context
// and its own model handle, real but otherwise unused by this cell -- every call below passes
// `ctx_b` alongside a handle that belongs to `ctx_a`, and every one must reject through the same
// disposition that call already uses for a malformed/null handle (never a new status, per this
// call's own routing). `seq_a`/`model_a` are left live and unmodified by every rejected call --
// checked at the end by a real, successful call against the CORRECT context, proving no rejected
// call corrupted the handle it was wrongly aimed at. ---
static void TestDim2_M4_CrossContextHandleRejectedAtEveryBoundary(
    SslmGpuContext* ctx_a, SslmGpuContext* ctx_b, SslmGpuModelHandle* model_a,
    SslmGpuSequenceHandle* seq_a, SslmGpuAdapterHandle* adapter_a,
    const SslmModelView* adapter_artifact, uint32_t num_hidden_layers) {
	// sslm_gpu_model_unmap: model_a belongs to ctx_a, called with ctx_b.
	CHECK(sslm_gpu_model_unmap(ctx_b, model_a) == SSLM_DEVICE_LOST);
	// sslm_gpu_adapter_unmap: adapter_a belongs to ctx_a, called with ctx_b.
	CHECK(sslm_gpu_adapter_unmap(ctx_b, adapter_a) == SSLM_DEVICE_LOST);
	// sslm_gpu_seq_create: model_a belongs to ctx_a, called with ctx_b.
	SslmGpuSequenceHandle* rejected_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx_b, model_a, 64, &rejected_seq) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH);
	CHECK(rejected_seq == nullptr);
	// sslm_gpu_seq_release: seq_a belongs to ctx_a, called with ctx_b -- must NOT delete seq_a
	// (checked below by a real, successful call against ctx_a).
	CHECK(sslm_gpu_seq_release(ctx_b, seq_a) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH);
	// sslm_gpu_seq_embed_token: seq_a belongs to ctx_a, called with ctx_b.
	CHECK(sslm_gpu_seq_embed_token(ctx_b, seq_a, 5) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH);
	// sslm_decode_step_gpu: seq_a belongs to ctx_a, called with ctx_b.
	CHECK(sslm_decode_step_gpu(ctx_b, seq_a, nullptr, 24u) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH);
	// sslm_decode_step_batch_gpu: the per-sequence channel (out_statuses[0]), never the call-level
	// return value (design Sec4.3) -- seq_a belongs to ctx_a, the batch called with ctx_b.
	SslmGpuSequenceHandle* const seqs_one[] = {seq_a};
	SslmGpuStatus out_statuses[1] = {SSLM_OK};
	CHECK(sslm_decode_step_batch_gpu(ctx_b, seqs_one, nullptr, 1u, 24u, out_statuses) == SSLM_OK);
	CHECK(out_statuses[0] == SSLM_SEQUENCE_KV_BUFFER_MISMATCH);
	// sslm_gpu_ready: seq_a belongs to ctx_a, called with ctx_b.
	int32_t ready = 0;
	SslmGpuStatus ready_status = SSLM_OK;
	CHECK(sslm_gpu_ready(ctx_b, seq_a, /*block=*/0, &ready, &ready_status) ==
	      SSLM_SEQUENCE_KV_BUFFER_MISMATCH);
	// sslm_gpu_seq_save: seq_a belongs to ctx_a, called with ctx_b -- the ctx check is the first
	// thing this call does (before the blob is ever touched), so a dummy 1-byte buffer suffices.
	uint8_t save_probe = 0;
	size_t save_size = sizeof(save_probe);
	CHECK(sslm_gpu_seq_save(ctx_b, seq_a, &save_probe, &save_size) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH);
	// sslm_gpu_seq_restore: model_a belongs to ctx_a, called with ctx_b -- the ctx check runs
	// before the blob header is parsed, so a dummy non-null blob pointer suffices.
	uint8_t restore_probe = 0;
	SslmGpuSequenceHandle* restored = nullptr;
	CHECK(sslm_gpu_seq_restore(ctx_b, model_a, &restore_probe, sizeof(restore_probe), &restored) ==
	      SSLM_SEQUENCE_KV_BUFFER_MISMATCH);
	CHECK(restored == nullptr);
	// sslm_gpu_seq_reset: seq_a belongs to ctx_a, called with ctx_b.
	CHECK(sslm_gpu_seq_reset(ctx_b, seq_a) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH);
	// sslm_gpu_adapter_map: model_a belongs to ctx_a, called with ctx_b -- the real adapter
	// artifact's own content is irrelevant here (the ctx check runs before it is read, same
	// ordering as every other malformed-input check this call already has).
	SslmGpuAdapterHandle* rejected_adapter = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx_b, model_a, adapter_artifact, &rejected_adapter) ==
	      SSLM_DEVICE_LOST);
	CHECK(rejected_adapter == nullptr);

	// Every rejected call above must have left model_a/seq_a/adapter_a completely unmodified --
	// proven by a real, successful call against the CORRECT context (ctx_a) for each. Uses
	// FullTokenBudget (not a bare per-layer 24u), matching this suite's own established
	// adapter-bound decode convention (dim8_composition_red.cpp's own RunFullTokenStep calls).
	CHECK(RunFullTokenStep(ctx_a, seq_a, adapter_a, num_hidden_layers, 5));
}

// --- Product cell: a real adapter artifact with a declared rank exceeding its own tensor's shape
// (the exact T-2104 S1 fixture shape, D-SLM3320) passed through sslm_gpu_adapter_map against the
// real 1.5B model -- rejected, never read past the tensor.
static void TestDim2_P1_OversizedRankAdapterRejectedNeverReadPastTensor(
    SslmGpuContext* ctx, SslmGpuModelHandle* model_1p5b, const SslmModelView* oversized_rank_adapter) {
	if (g_model_1p5b_path.empty() || g_adapter_path.empty()) {
		SKIP_MSG("real 1.5B artifact / hostile adapter artifact not supplied -- product cell not run");
		return;
	}
	SslmGpuAdapterHandle* out_adapter = nullptr;
	// The fixture (design's own T-2104 S1 shape: declared rank field larger than the tensor the
	// adapter blob actually carries) is constructed by the same generator T-2104's own CPU-side
	// hostile-fixture population uses (tests/fixtures/sslm_model_hostile_fixtures.h's own
	// convention) -- reused, not reinvented, per this seat's Sec5.4 real-workload rule and the
	// StandardsDocument Sec7 sibling-reuse discipline (audited: that generator already targets
	// exactly this defect class for the CPU-side adapter loader, T-2104's own citation).
	const SslmGpuStatus st = sslm_gpu_adapter_map(ctx, model_1p5b, oversized_rank_adapter, &out_adapter);
	CHECK(st == SSLM_ADAPTER_BASE_HASH_MISMATCH || st != SSLM_OK);
	CHECK(out_adapter == nullptr);
}

// T-2113 (P2, design Sec4.2/Sec9/Sec22, routed `Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-
// review.md` Sec15, D-SLM3415, owed BY THIS BUILD SEAT per the fold's own explicit "the build seat
// owes ... the two new dim-2 cells in the T-2112 suite" instruction -- the one exception to
// Brunel's own "does not author the test suite" boundary, stated by the routing itself). Small
// integer-arithmetic helper for TestDim2_P2 below: `gcd`, real and exact, never a floating
// approximation. ---
static int64_t GpuSeqBlobKvGcd(int64_t a, int64_t b) {
	while (b != 0) {
		const int64_t t = b;
		b = a % b;
		a = t;
	}
	return a;
}

// --- Product cell (design Sec22's own P2 dim-2 addition): "a blob saved from the real 0.5B
// artifact restored against the real 1.5B model" -- design's own named example. Asserts
// SSLM_RESTORE_MODEL_MISMATCH, not the generic malformed-blob disposition.
//
// Read at source before writing this cell (StandardsDocument.md Sec5.4: exactness verified by
// execution, never by construction): sslm_gpu_seq_restore's own size-derivation ladder (design
// Sec4.2/Sec21, gpu_1p0.cpp) runs BEFORE the model_content_hash check this cell exists to force,
// and derives the fresh handle's own context_cap from `workspace_size / (num_hidden_layers *
// num_key_value_heads * head_dim * 2)` computed against the TARGET (1.5B) model's own dimensions --
// NOT the 0.5B model's, which is what the blob was actually saved at. The real 0.5B and 1.5B
// artifacts have DIFFERENT per-cap-unit byte counts (different num_hidden_layers/
// num_key_value_heads/head_dim), so an arbitrarily-chosen 0.5B context_cap (e.g. 64) produces a
// workspace_size that does NOT divide evenly against the 1.5B model's own per-cap-unit bytes --
// the size ladder would reject the blob as MALFORMED before the hash check is ever reached,
// which would silently make this cell assert the wrong (still-rejecting, but wrong-CHANNEL)
// status. The fix is arithmetic, not a different fixture: this cell picks the 0.5B sequence's own
// context_cap as `per_unit(1.5B) / gcd(per_unit(0.5B), per_unit(1.5B))` -- the smallest positive
// integer that makes `per_unit(0.5B) * context_cap` an exact multiple of `per_unit(1.5B)` by
// construction, so the blob's own workspace_size divides evenly against the TARGET model's
// dimensions and the size ladder admits it, reaching the hash check this cell actually exists to
// pin. `hidden_codes_size` also mismatches (0.5B's own hidden_size vs. 1.5B's, design's own
// parenthetical) -- irrelevant here, since the hash check runs and rejects before
// RestoreGpuSequenceState (where hidden_codes_size would otherwise be checked) is ever reached.
static void TestDim2_P2_ForeignBlobRestoreModelMismatch(SslmGpuContext* ctx,
                                                          SslmGpuModelHandle* model_0p5b,
                                                          SslmGpuModelHandle* model_1p5b,
                                                          const SslmModelView& view_0p5b,
                                                          const SslmModelView& view_1p5b) {
	// `SslmGpuModelHandle` is an OPAQUE handle from this suite's own declared surface (design
	// Sec4.1) -- its real fields are private to gpu_1p0.cpp, not readable from test code. Every
	// dimension this cell's own cap arithmetic needs is already available from the LOADED
	// `SslmModelView` config (the same real artifact each handle was mapped from), never through
	// the handle pointer.
	const int64_t per_unit_0p5b = static_cast<int64_t>(view_0p5b.config.num_hidden_layers) *
	                               view_0p5b.config.num_key_value_heads * view_0p5b.config.head_dim * 2;
	const int64_t per_unit_1p5b = static_cast<int64_t>(view_1p5b.config.num_hidden_layers) *
	                               view_1p5b.config.num_key_value_heads * view_1p5b.config.head_dim * 2;
	CHECK_MSG(per_unit_0p5b > 0 && per_unit_1p5b > 0,
	          "P2 fixture bug: real artifacts must have nonzero per-cap-unit byte counts "
	          "(0.5B=%lld, 1.5B=%lld)",
	          (long long)per_unit_0p5b, (long long)per_unit_1p5b);
	const int64_t g = GpuSeqBlobKvGcd(per_unit_0p5b, per_unit_1p5b);
	const int64_t cap_0p5b = per_unit_1p5b / g;  // exact by construction, see this cell's own header.
	const int64_t derived_cap_against_1p5b = per_unit_0p5b / g;
	CHECK_MSG(cap_0p5b > 0 && derived_cap_against_1p5b > 0 &&
	              derived_cap_against_1p5b <= static_cast<int64_t>(view_1p5b.config.context_cap),
	          "P2 fixture bug: derived context_cap (%lld) must be positive and admissible against "
	          "the 1.5B model's own context_cap (%lld) for this cell to reach the hash check rather "
	          "than a spurious size-ladder rejection",
	          (long long)derived_cap_against_1p5b, (long long)view_1p5b.config.context_cap);

	SslmGpuSequenceHandle* seq = nullptr;
	CHECK_MSG(sslm_gpu_seq_create(ctx, model_0p5b, cap_0p5b, &seq) == SSLM_OK,
	          "P2: 0.5B fixture sequence create at the exactly-divisible cap (%lld)",
	          (long long)cap_0p5b);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);

	size_t required_size = 0;
	{
		uint8_t probe = 0;
		sslm_gpu_seq_save(ctx, seq, &probe, &required_size);
	}
	std::vector<uint8_t> blob(required_size);
	size_t blob_size = blob.size();
	CHECK(sslm_gpu_seq_save(ctx, seq, blob.data(), &blob_size) == SSLM_OK);

	// Sanity control, real and executed: the SAME blob restores cleanly against the model it was
	// actually saved from -- proving the size-ladder admissibility this cell's own arithmetic set
	// up is genuinely exact (not an accidental pass), before the cross-model call below is trusted.
	SslmGpuSequenceHandle* restored_same_model = nullptr;
	CHECK_MSG(sslm_gpu_seq_restore(ctx, model_0p5b, blob.data(), blob_size, &restored_same_model) ==
	              SSLM_OK,
	          "P2 control: the 0.5B blob must restore cleanly against the SAME (0.5B) model it was "
	          "saved from -- if this fails, the cell's own cap arithmetic is wrong, not the "
	          "cross-model assertion below");
	if (restored_same_model) CHECK(sslm_gpu_seq_release(ctx, restored_same_model) == SSLM_OK);

	SslmGpuSequenceHandle* restored = nullptr;
	const SslmGpuStatus st = sslm_gpu_seq_restore(ctx, model_1p5b, blob.data(), blob_size, &restored);
	CHECK_MSG(st == SSLM_RESTORE_MODEL_MISMATCH,
	          "P2: a blob saved from the real 0.5B artifact, restored against the real 1.5B model, "
	          "must return SSLM_RESTORE_MODEL_MISMATCH -- got %d",
	          (int)st);
	CHECK(restored == nullptr);

	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- Product cell (design Sec22's own second dim-2 addition): "two real artifacts sharing declared
// hidden_size/context_cap... so only the hash differs" -- the cell that actually forces
// model_content_hash to do work no other field already does. `g_model_1p5b_path` (instruct) and
// `g_model_1p5b_variant_path` (a same-tier, same-architecture, genuinely different-weights real
// artifact -- e.g. the real qwen2.5-1.5b-shopkeeper-lora-v2-merged-t2102.sslm (an adapter merged
// into the 1.5B tier's own weights, same architecture) alongside qwen2.5-1.5b-instruct.sslm) share every
// declared config dimension (hidden_size, num_hidden_layers, num_key_value_heads, head_dim,
// context_cap), so the size-derivation ladder admits the cross-restore unconditionally and
// hidden_codes_size matches too -- only RawIntegrityHash() differs. ---
static void TestDim2_P3_SameShapeDifferentContentRestoreModelMismatch(
    SslmGpuContext* ctx, SslmGpuModelHandle* model_a, SslmGpuModelHandle* model_b,
    const SslmModelView& view_a, const SslmModelView& view_b) {
	// `SslmGpuModelHandle` is opaque from this suite's own declared surface -- see TestDim2_P2's
	// own header comment. The shape-parity fixture check reads the loaded `SslmModelView` configs
	// instead, the real artifacts each handle was mapped from.
	CHECK_MSG(view_a.config.hidden_size == view_b.config.hidden_size &&
	              view_a.config.num_hidden_layers == view_b.config.num_hidden_layers &&
	              view_a.config.num_key_value_heads == view_b.config.num_key_value_heads &&
	              view_a.config.head_dim == view_b.config.head_dim,
	          "P3 fixture bug: the two supplied artifacts must share every declared config "
	          "dimension for this cell to exercise the hash check alone -- got a genuine shape "
	          "mismatch instead, which would be P2's own class, not P3's");

	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_a, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);

	size_t required_size = 0;
	{
		uint8_t probe = 0;
		sslm_gpu_seq_save(ctx, seq, &probe, &required_size);
	}
	std::vector<uint8_t> blob(required_size);
	size_t blob_size = blob.size();
	CHECK(sslm_gpu_seq_save(ctx, seq, blob.data(), &blob_size) == SSLM_OK);

	// Sanity control: restoring against the SAME model (model_a) the blob was saved from succeeds.
	SslmGpuSequenceHandle* restored_same = nullptr;
	CHECK_MSG(sslm_gpu_seq_restore(ctx, model_a, blob.data(), blob_size, &restored_same) == SSLM_OK,
	          "P3 control: the blob must restore cleanly against the SAME model it was saved from");
	if (restored_same) CHECK(sslm_gpu_seq_release(ctx, restored_same) == SSLM_OK);

	// The real assertion: restoring the IDENTICAL blob against model_b -- same declared shape,
	// different content hash -- must be rejected, and specifically under SSLM_RESTORE_MODEL_MISMATCH,
	// never a silent success (the defect this whole fold exists to close) and never a generic
	// malformed-blob status (every size check above passes; only the hash differs).
	SslmGpuSequenceHandle* restored_b = nullptr;
	const SslmGpuStatus st = sslm_gpu_seq_restore(ctx, model_b, blob.data(), blob_size, &restored_b);
	CHECK_MSG(st == SSLM_RESTORE_MODEL_MISMATCH,
	          "P3: a blob saved from model_a, restored against SAME-SHAPE model_b (different "
	          "content hash), must return SSLM_RESTORE_MODEL_MISMATCH -- got %d -- if this reads "
	          "SSLM_OK, model_content_hash is not being checked at all",
	          (int)st);
	CHECK(restored_b == nullptr);

	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- Product cell (T-2114, S3, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md):
// sslm_gpu_model_map used to copy `vocab_size * hidden_size` bytes out of the "embed" tensor's
// own data with no check against that tensor's own declared `elem_count` -- a short `embed`
// section (a parser-legal artifact whose CFG1.vocab_size disagrees with WGT1's own "embed"
// tensor shape, the exact gap model.cpp:1192's own cross-check does not close, per the review)
// read megabytes past the mapped buffer on every load. Built directly from the CPU-side
// manifest/keyed-constant builders (sslm_model_hostile_fixtures.h/sslm_kvc1_hostile_fixtures.h)
// rather than a full on-disk artifact -- num_hidden_layers=0 in the hand-built config means
// sslm_gpu_model_map's own per-layer MarshalLayer loop is zero iterations, so no per-layer
// weight tensors are needed at all; the "embed" tensor and its composition_constants entry are
// the whole fixture. ---
static void TestDim2_S3_ShortEmbedTensorRejectedAtMapTime(SslmGpuContext* ctx) {
	using namespace superslm_test;

	// "embed" tensor: 4 int8 bytes, genuinely that small and internally consistent (a
	// well-formed WGT1 manifest the parser accepts) -- config below declares vocab_size=10,
	// hidden_size=10 (embed_bytes_needed=100), so the tensor is 96 bytes short of what
	// sslm_gpu_model_map's own (pre-S3) unchecked copy would have read.
	const BuiltManifest wgt = BuildManifest(superslm::kWeightsMagic, /*element_size=*/1,
	                                         {{"embed", {4}}});
	SslmSectionView wgt_view = MakeManifestSectionView(SslmSectionType::Weights, SslmDtype::Int8, wgt.bytes);
	SslmTensorManifest weights;
	std::string werr;
	CHECK_MSG(SslmTensorManifest::Parse(wgt_view, weights, &werr) == SslmModelStatus::Ok,
	          "S3 fixture: the hand-built WGT1 manifest itself must parse Ok (spec-faithful, not "
	          "the defect under test) -- %s",
	          werr.c_str());

	const BuiltKvc1 kvc = BuildKvc1(/*declared_value_words=*/2, {{"embed", {1073741824LL, 0LL}}});
	SslmSectionView kvc_view = MakeConstantsSectionView(SslmSectionType::CompositionConstants, kvc.bytes);
	SslmKeyedConstants composition_constants;
	std::string kerr;
	CHECK_MSG(SslmKeyedConstants::Parse(kvc_view, composition_constants, &kerr) == SslmModelStatus::Ok,
	          "S3 fixture: the hand-built composition-constants section itself must parse Ok -- %s",
	          kerr.c_str());

	SslmModelView view{};
	view.config.hidden_size = 10;
	view.config.vocab_size = 10;
	view.config.num_hidden_layers = 0;  // zero-iteration MarshalLayer loop -- no layer tensors needed
	view.config.context_cap = 64;
	view.has_config = true;
	view.weights = std::move(weights);
	view.has_weights = true;
	view.composition_constants = std::move(composition_constants);
	view.has_composition_constants = true;

	SslmGpuModelHandle* model = nullptr;
	const SslmGpuStatus st = sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model);
	CHECK_MSG(st != SSLM_OK, "S3: sslm_gpu_model_map must reject a short embed tensor "
	          "(elem_count=4, vocab_size*hidden_size=100), not read past it -- got status %d", (int)st);
	CHECK_MSG(model == nullptr, "S3: a rejected map must not deliver a live handle");
}

// T-2114 (M2): see dim1_lifetime_red.cpp's own header comment -- the local re-declaration
// this file used to complete here is retired; sslm_gpu_1p0.h now defines both types complete.

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_0 = (void*)&TestDim2_M1_NewGuardsForcedByHostileInputs; (void)addr_0;
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_1 = (void*)&TestDim2_M2_OneRepresentativeGuardCarriedForwardThroughDecodeStepGpu; (void)addr_1;
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_2 = (void*)&TestDim2_P1_OversizedRankAdapterRejectedNeverReadPastTensor; (void)addr_2;
	// Force emission (StandardsDocument.md Sec5.4): see the pattern above.
	volatile void* addr_3 = (void*)&TestDim2_S3_ShortEmbedTensorRejectedAtMapTime; (void)addr_3;
	// Force emission (StandardsDocument.md Sec5.4): see the pattern above.
	volatile void* addr_4 = (void*)&TestDim2_P2_ForeignBlobRestoreModelMismatch; (void)addr_4;
	// Force emission (StandardsDocument.md Sec5.4): see the pattern above.
	volatile void* addr_5 = (void*)&TestDim2_P3_SameShapeDifferentContentRestoreModelMismatch; (void)addr_5;
	// Force emission (StandardsDocument.md Sec5.4): see the pattern above.
	volatile void* addr_6 = (void*)&TestDim2_M4_CrossContextHandleRejectedAtEveryBoundary; (void)addr_6;

	// S3's own cell is fully synthetic (no real artifact needed) -- run it unconditionally,
	// including when no --model*/--adapter args are supplied, rather than gating it behind the
	// same real-artifact requirement the OTHER cells in this file need.
	{
		SslmGpuContext* s3_ctx = nullptr;
		CHECK(sslm_gpu_context_create(GpuContextConfig{}, &s3_ctx) == SSLM_OK);
		TestDim2_S3_ShortEmbedTensorRejectedAtMapTime(s3_ctx);
		CHECK(sslm_gpu_context_destroy(s3_ctx) == SSLM_OK);
	}

	if (g_model_1p5b_path.empty() || g_model_0p5b_path.empty() || g_adapter_path.empty()) {
		SKIP_MSG("dim2 needs --model1p5b=PATH --model0p5b=PATH --adapter=PATH -- not run");
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}
	std::vector<uint8_t> bytes_1p5b, bytes_0p5b, bytes_adapter;
	SslmModelView view_1p5b{}, view_0p5b{}, view_adapter{};
	std::string err;
	const bool have_all =
	    LoadRealModel(g_model_1p5b_path, &view_1p5b, &bytes_1p5b, &err) &&
	    LoadRealModel(g_model_0p5b_path, &view_0p5b, &bytes_0p5b, &err) &&
	    LoadRealModel(g_adapter_path, &view_adapter, &bytes_adapter, &err);
	if (!have_all) {
		SKIP_MSG("dim2 could not load one of the three real artifacts: %s", err.c_str());
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}

	// M1: model_a=0.5B, model_b=1.5B, adapter=the real shopkeeper (D-SLM3320's own real-adapter
	// convention, no synthetic fixture needed) -- shopkeeper's own base-hash matches 1.5B and
	// mismatches 0.5B, giving BOTH assertions the cell's own body needs against ONE real
	// artifact: mapped against model_b(1.5B) it succeeds, and binding that mapping to a sequence
	// bound to model_a(0.5B) fires AdapterModelMismatch (pointer inequality, not a re-hash);
	// mapped directly against model_a(0.5B) it fires AdapterBaseHashMismatch (the real check).
	{
		SslmGpuContext* ctx = nullptr;
		CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
		SslmGpuModelHandle* model_a = nullptr;
		SslmGpuModelHandle* model_b = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view_0p5b, GpuResidencyConfig{}, &model_a) == SSLM_OK);
		CHECK(sslm_gpu_model_map(ctx, &view_1p5b, GpuResidencyConfig{}, &model_b) == SSLM_OK);
		SslmGpuSequenceHandle* seq_bound_to_a = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model_a, 64, &seq_bound_to_a) == SSLM_OK);
		TestDim2_M1_NewGuardsForcedByHostileInputs(ctx, model_a, model_b, seq_bound_to_a, &view_adapter);
		// TestDim2_M1's own last assertion is `sslm_gpu_context_destroy` REJECTING (live
		// handles) -- ctx/model_a/model_b/seq_bound_to_a are therefore all still live; real
		// cleanup, in the guard-satisfying order, happens here. NOT asserted SSLM_OK: the
		// cell's own body (unedited, per this driver's own discipline) maps `adapter_for_b`
		// against model_b and never unmaps it or returns the pointer to this caller -- a
		// leaked handle internal to the cell's own fixture, not a defect this driver's
		// cleanup can close without touching the CHECK-decorated body. `ctx->live_handles`
		// stays nonzero for that one leaked adapter handle even after every handle this
		// driver DOES own is released, so this final destroy is best-effort (its own D3D12
		// device is reclaimed on process exit regardless) rather than a correctness check.
		CHECK(sslm_gpu_seq_release(ctx, seq_bound_to_a) == SSLM_OK);
		CHECK(sslm_gpu_model_unmap(ctx, model_a) == SSLM_OK);
		CHECK(sslm_gpu_model_unmap(ctx, model_b) == SSLM_OK);
		sslm_gpu_context_destroy(ctx);
	}

	// M2: any real sequence -- the guard under test (DispatchBudgetTooSmall) fires before any
	// adapter/model check runs, so which real model the sequence is bound to is immaterial.
	{
		SslmGpuContext* ctx = nullptr;
		CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view_1p5b, GpuResidencyConfig{}, &model) == SSLM_OK);
		SslmGpuSequenceHandle* seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
		TestDim2_M2_OneRepresentativeGuardCarriedForwardThroughDecodeStepGpu(ctx, seq);
		CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
		CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	}

	// M4 (T-2124, D-SLM3446 P1-4): two independent real contexts, ctx_a owns model_a/seq_a/adapter_a,
	// ctx_b is the foreign context every call below is wrongly aimed at.
	{
		SslmGpuContext* ctx_a = nullptr;
		SslmGpuContext* ctx_b = nullptr;
		CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx_a) == SSLM_OK);
		CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx_b) == SSLM_OK);
		SslmGpuModelHandle* model_a = nullptr;
		CHECK(sslm_gpu_model_map(ctx_a, &view_1p5b, GpuResidencyConfig{}, &model_a) == SSLM_OK);
		SslmGpuSequenceHandle* seq_a = nullptr;
		CHECK(sslm_gpu_seq_create(ctx_a, model_a, 64, &seq_a) == SSLM_OK);
		SslmGpuAdapterHandle* adapter_a = nullptr;
		CHECK(sslm_gpu_adapter_map(ctx_a, model_a, &view_adapter, &adapter_a) == SSLM_OK);
		TestDim2_M4_CrossContextHandleRejectedAtEveryBoundary(
		    ctx_a, ctx_b, model_a, seq_a, adapter_a, &view_adapter,
		    view_1p5b.config.num_hidden_layers);
		CHECK(sslm_gpu_seq_release(ctx_a, seq_a) == SSLM_OK);
		CHECK(sslm_gpu_adapter_unmap(ctx_a, adapter_a) == SSLM_OK);
		CHECK(sslm_gpu_model_unmap(ctx_a, model_a) == SSLM_OK);
		CHECK(sslm_gpu_context_destroy(ctx_a) == SSLM_OK);
		CHECK(sslm_gpu_context_destroy(ctx_b) == SSLM_OK);
	}

	// P2 (design Sec22): foreign-blob rejection -- 0.5B save, restore against the real 1.5B model.
	{
		SslmGpuContext* ctx = nullptr;
		CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
		SslmGpuModelHandle* model_0p5b = nullptr;
		SslmGpuModelHandle* model_1p5b = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view_0p5b, GpuResidencyConfig{}, &model_0p5b) == SSLM_OK);
		CHECK(sslm_gpu_model_map(ctx, &view_1p5b, GpuResidencyConfig{}, &model_1p5b) == SSLM_OK);
		TestDim2_P2_ForeignBlobRestoreModelMismatch(ctx, model_0p5b, model_1p5b, view_0p5b, view_1p5b);
		CHECK(sslm_gpu_model_unmap(ctx, model_0p5b) == SSLM_OK);
		CHECK(sslm_gpu_model_unmap(ctx, model_1p5b) == SSLM_OK);
		CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	}

	// P3 (design Sec22, Q1 repair 2026-08-15): same-shape, different-content rejection -- "the
	// arc's own only discriminating model-identity pin" (T-2114 third confirmation, casebook
	// @7b627d5). Per the reviewer's own named fix: a cell whose PURPOSE is discrimination FAILS,
	// naming the missing fixture and how to produce it, when that fixture is absent -- it never
	// SKIPs. A silent skip here is exactly the second reduced-invocation-as-result shape this arc
	// has found (the first was dim8's own D-SLM3412/P1, §7.4 of this suite's own artifact): a
	// "clean-clone" run supplying the documented three-artifact baseline
	// (--model1p5b/--model0p5b/--adapter) already reports dim2 GREEN without
	// --model1p5bvariant=PATH -- the one flag beyond that baseline this specific cell needs -- so
	// model_content_hash's own discrimination silently never executes while the run looks
	// complete. Every OTHER skip in this suite gates on the documented three-artifact baseline
	// itself (an artifact-less CI runner, this suite's own founding convention,
	// fixture_common.h's own header comment) -- this is the one cell that needs a FOURTH,
	// undocumented-by-default artifact, which is why it is the one that fails rather than skips.
	if (!g_model_1p5b_variant_path.empty()) {
		std::vector<uint8_t> bytes_variant;
		SslmModelView view_variant{};
		std::string variant_err;
		if (LoadRealModel(g_model_1p5b_variant_path, &view_variant, &bytes_variant, &variant_err)) {
			SslmGpuContext* ctx = nullptr;
			CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
			SslmGpuModelHandle* model_a = nullptr;
			SslmGpuModelHandle* model_b = nullptr;
			CHECK(sslm_gpu_model_map(ctx, &view_1p5b, GpuResidencyConfig{}, &model_a) == SSLM_OK);
			CHECK(sslm_gpu_model_map(ctx, &view_variant, GpuResidencyConfig{}, &model_b) == SSLM_OK);
			TestDim2_P3_SameShapeDifferentContentRestoreModelMismatch(ctx, model_a, model_b, view_1p5b,
			                                                           view_variant);
			CHECK(sslm_gpu_model_unmap(ctx, model_a) == SSLM_OK);
			CHECK(sslm_gpu_model_unmap(ctx, model_b) == SSLM_OK);
			CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
		} else {
			CHECK_MSG(false,
			          "P3 (the arc's only discriminating model-identity pin) FAILS, not skips: "
			          "--model1p5bvariant=%s was supplied but could not be loaded -- %s. Supply "
			          "a real, on-disk artifact sharing the 1.5B tier's own declared config shape "
			          "(hidden_size/num_hidden_layers/num_key_value_heads/head_dim) but a "
			          "genuinely different content hash -- e.g. the real "
			          "qwen2.5-1.5b-shopkeeper-lora-v2-merged-t2102.sslm alongside "
			          "qwen2.5-1.5b-instruct.sslm.",
			          g_model_1p5b_variant_path.c_str(), variant_err.c_str());
		}
	} else {
		CHECK_MSG(false,
		          "P3 (the arc's only discriminating model-identity pin) FAILS, not skips, when "
		          "--model1p5bvariant=PATH is absent -- a silent skip here would let a "
		          "clean-clone run supplying only the documented three-artifact baseline "
		          "(--model1p5b/--model0p5b/--adapter) report dim2 GREEN while "
		          "model_content_hash's own discrimination is never exercised (Q1, casebook "
		          "@7b627d5). Supply a second real artifact sharing the 1.5B tier's own declared "
		          "config shape (hidden_size/num_hidden_layers/num_key_value_heads/head_dim) but "
		          "a genuinely different content hash, e.g. "
		          "--model1p5bvariant=<path to qwen2.5-1.5b-shopkeeper-lora-v2-merged-t2102.sslm> "
		          "alongside --model1p5b=<path to qwen2.5-1.5b-instruct.sslm>.");
	}

	// P1: NOT driven this reconciliation, named rather than silently skipped-without-comment
	// (StandardsDocument.md Sec5.6). `oversized_rank_adapter` is the T-2104 S1 hostile-fixture
	// shape (a declared rank field exceeding the tensor the adapter blob actually carries) --
	// its generator (`BuildAdp1`/`BuildArtifact`/`BuildFoldManifestOneEntry`/
	// `BuildAdapterWeightsManifest`, tests/test_main.cpp) is private to that translation unit,
	// not exposed through fixture_common.h or any shared header this suite includes. Building
	// one real malformed adapter artifact byte-for-byte is fixture-authoring work (Curie's own
	// craft, StandardsDocument.md/Brunel's own "does not author the test suite" boundary), not a
	// drive-existing-cells-with-real-artifacts pass -- every other cell this reconciliation (and
	// B6a's own dim1/dim3/dim6/dim9/dim11 reconciliation before it) drives needed only REAL,
	// already-on-disk artifacts, never a newly-authored malformed one. Routed, not attempted.
	SKIP_MSG("P1 (oversized-rank hostile adapter) needs a shared fixture generator "
	         "(tests/test_main.cpp's BuildAdp1/BuildArtifact family is private to that TU) -- "
	         "not built this reconciliation, routed to Curie/a fixture-extraction pass");

	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
