// T-2180 (Rung 6, Brunel build-seat verification tool -- not part of the committed T-2178 red
// suite, matching this project's own established precedent for a builder's own bench-proof
// harness, tools/t2169_rung2b_selfcheck.cpp): drives the batched GPU prefill through the PUBLIC
// G5 bridge (SslmGpuSeqPrefillPromptForG5Bridge, now wired per Rungs 3/4) at the size sweep
// {1,2,4,8,16,256} plus boundary splits, asserting bit-identity against the reference (N
// single-token bridge calls) at every size -- this is the public-bridge-surface generalization of
// cell_bitidentity.cpp's own Cell1/Cell2 (which sweep {1,3,6}/splits of a 6-token chunk), covering
// the exact size list T-2180's own build log is required to report against, including sizes that
// force multiple internal TDR-safe sub-chunk splits (kT2169TdrSafeMaxChunkTokens = 4, so sizes 8,
// 16, 256 each force >= 2 internal sub-chunk submissions).
//
// Usage: t2180_rung6_public_bridge_sweep <model.sslm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/model.h"

using namespace superslm;

namespace {

constexpr uint32_t kDispatchBudget = 24;

struct Snapshot {
	std::vector<int8_t> hidden_codes;
	int64_t hidden_scale_m = 0, hidden_scale_e = 0;
	uint32_t layer_index = 0;
	uint64_t kv_saturation_count = 0;
	int64_t context_length = 0;
};

bool CaptureSnapshot(SslmGpuSequenceHandle* seq, Snapshot* out) {
	const size_t hidden_size = SslmGpuSeqHandleHiddenSizeForBench(seq);
	const int8_t* codes = SslmGpuSeqHandleHiddenCodesForBench(seq);
	if (!codes) return false;
	out->hidden_codes.assign(codes, codes + hidden_size);
	const CarriedScale* scale = SslmGpuSeqHandleHiddenScaleForBench(seq);
	out->hidden_scale_m = scale->m;
	out->hidden_scale_e = scale->e;
	out->layer_index = *SslmGpuSeqHandleLayerIndexForBench(seq);
	out->kv_saturation_count = *SslmGpuSeqHandleKvSaturationForBench(seq);
	out->context_length = *SslmGpuSeqHandleContextLengthForBench(seq);
	return true;
}

bool SnapshotsEqual(const Snapshot& a, const Snapshot& b) {
	return a.hidden_codes == b.hidden_codes && a.hidden_scale_m == b.hidden_scale_m &&
	       a.hidden_scale_e == b.hidden_scale_e && a.layer_index == b.layer_index &&
	       a.kv_saturation_count == b.kv_saturation_count && a.context_length == b.context_length;
}

std::vector<int32_t> MakeTokens(int32_t start, size_t n) {
	std::vector<int32_t> v(n);
	for (size_t i = 0; i < n; ++i) v[i] = start + static_cast<int32_t>(i);
	return v;
}

// Reference: N separate single-token public-bridge calls. Candidate: ONE public-bridge call
// carrying all N tokens. Both start from an identical prime.
bool RunOneSizeCell(SslmGpuContext* ctx, SslmGpuModelHandle* model, int64_t context_cap,
                     const std::vector<int32_t>& prime, size_t n, int32_t token_base, int* out_checks,
                     int* out_failures) {
	const std::vector<int32_t> tokens = MakeTokens(token_base, n);

	SslmGpuSequenceHandle* ref_seq = nullptr;
	if (sslm_gpu_seq_create(ctx, model, context_cap, &ref_seq) != SSLM_OK || !ref_seq) return false;
	if (!prime.empty()) {
		SslmGpuSeqPrefillPromptForG5Bridge(ctx, ref_seq, prime.data(),
		                                    static_cast<int32_t>(prime.size()), kDispatchBudget);
	}
	SslmGpuStatus ref_status = SSLM_OK;
	for (int32_t t : tokens) {
		ref_status = SslmGpuSeqPrefillPromptForG5Bridge(ctx, ref_seq, &t, 1, kDispatchBudget);
		if (ref_status != SSLM_OK) break;
	}
	Snapshot ref_snap;
	CaptureSnapshot(ref_seq, &ref_snap);
	int32_t ref_out = -1;
	const SslmGpuStatus ref_step =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, ref_seq, 999, kDispatchBudget, &ref_out);
	sslm_gpu_seq_release(ctx, ref_seq);

	SslmGpuSequenceHandle* cand_seq = nullptr;
	if (sslm_gpu_seq_create(ctx, model, context_cap, &cand_seq) != SSLM_OK || !cand_seq) return false;
	if (!prime.empty()) {
		SslmGpuSeqPrefillPromptForG5Bridge(ctx, cand_seq, prime.data(),
		                                    static_cast<int32_t>(prime.size()), kDispatchBudget);
	}
	const SslmGpuStatus cand_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, cand_seq, tokens.data(), static_cast<int32_t>(tokens.size()), kDispatchBudget);
	Snapshot cand_snap;
	CaptureSnapshot(cand_seq, &cand_snap);
	int32_t cand_out = -1;
	const SslmGpuStatus cand_step =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, cand_seq, 999, kDispatchBudget, &cand_out);
	sslm_gpu_seq_release(ctx, cand_seq);

	++(*out_checks);
	bool ok = (ref_status == cand_status) && SnapshotsEqual(ref_snap, cand_snap) &&
	          (ref_step == cand_step) && (ref_out == cand_out);
	if (!ok) {
		++(*out_failures);
		std::printf("  [n=%zu] DIVERGENCE: ref_status=%d cand_status=%d ref_step=%d/%d cand_step=%d/%d "
		            "snap_eq=%d\n",
		            n, (int)ref_status, (int)cand_status, (int)ref_step, ref_out, (int)cand_step,
		            cand_out, (int)SnapshotsEqual(ref_snap, cand_snap));
	} else {
		std::printf("  [n=%zu] BIT-IDENTICAL (status=%d, ctxlen=%lld, step_out=%d)\n", n, (int)ref_status,
		            (long long)cand_snap.context_length, cand_out);
	}
	return ok;
}

bool RunBoundarySplitCell(SslmGpuContext* ctx, SslmGpuModelHandle* model, int64_t context_cap,
                           size_t n, size_t split_k, int* out_checks, int* out_failures) {
	const std::vector<int32_t> tokens = MakeTokens(500, n);

	SslmGpuSequenceHandle* whole_seq = nullptr;
	if (sslm_gpu_seq_create(ctx, model, context_cap, &whole_seq) != SSLM_OK) return false;
	SslmGpuSeqPrefillPromptForG5Bridge(ctx, whole_seq, tokens.data(), static_cast<int32_t>(n),
	                                    kDispatchBudget);
	Snapshot whole_snap;
	CaptureSnapshot(whole_seq, &whole_snap);
	sslm_gpu_seq_release(ctx, whole_seq);

	SslmGpuSequenceHandle* split_seq = nullptr;
	if (sslm_gpu_seq_create(ctx, model, context_cap, &split_seq) != SSLM_OK) return false;
	SslmGpuSeqPrefillPromptForG5Bridge(ctx, split_seq, tokens.data(), static_cast<int32_t>(split_k),
	                                    kDispatchBudget);
	SslmGpuSeqPrefillPromptForG5Bridge(ctx, split_seq, tokens.data() + split_k,
	                                    static_cast<int32_t>(n - split_k), kDispatchBudget);
	Snapshot split_snap;
	CaptureSnapshot(split_seq, &split_snap);
	sslm_gpu_seq_release(ctx, split_seq);

	++(*out_checks);
	const bool ok = SnapshotsEqual(whole_snap, split_snap);
	if (!ok) {
		++(*out_failures);
		std::printf("  [split n=%zu k=%zu] DIVERGENCE\n", n, split_k);
	} else {
		std::printf("  [split n=%zu k=%zu] BIT-IDENTICAL (ctxlen=%lld)\n", n, split_k,
		            (long long)split_snap.context_length);
	}
	return ok;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model.sslm>\n", argv[0]);
		return 2;
	}
	std::vector<uint8_t> bytes;
	SslmModelView view{};
	std::string err;
	{
		std::FILE* f = std::fopen(argv[1], "rb");
		if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
		std::fseek(f, 0, SEEK_END);
		const long sz = std::ftell(f);
		std::fseek(f, 0, SEEK_SET);
		bytes.resize(static_cast<size_t>(sz));
		std::fread(bytes.data(), 1, bytes.size(), f);
		std::fclose(f);
	}
	if (SslmModel::Load(bytes.data(), bytes.size(), view, &err) != SslmModelStatus::Ok) {
		std::fprintf(stderr, "load failed: %s\n", err.c_str());
		return 2;
	}

	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
		std::fprintf(stderr, "context create failed\n");
		return 2;
	}
	SslmGpuModelHandle* model = nullptr;
	if (sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) != SSLM_OK || !model) {
		std::fprintf(stderr, "model map failed\n");
		return 2;
	}

	int checks = 0, failures = 0;
	const std::vector<int32_t> prime = {11, 22, 33};
	const int64_t kContextCap = 512;  // covers the 256-token sweep size plus prime/split margin.

	std::printf("=== Size sweep {1,2,4,8,16,256} through the PUBLIC bridge (SslmGpuSeqPrefillPromptForG5Bridge) ===\n");
	for (size_t n : std::vector<size_t>{1, 2, 4, 8, 16, 256}) {
		RunOneSizeCell(ctx, model, kContextCap, prime, n, 100, &checks, &failures);
	}

	std::printf("=== Boundary splits (n=16, k in {1,4,7,8,9,15}; straddles the sub-chunk bound of 4) ===\n");
	for (size_t k : std::vector<size_t>{1, 4, 7, 8, 9, 15}) {
		RunBoundarySplitCell(ctx, model, kContextCap, 16, k, &checks, &failures);
	}

	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);
	std::printf("\n=== T-2180 Rung 6 public-bridge bit-identity sweep: checks=%d failures=%d ===\n", checks,
	            failures);
	return failures ? 1 : 0;
}
