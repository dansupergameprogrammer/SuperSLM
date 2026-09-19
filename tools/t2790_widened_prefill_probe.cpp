// T-2790 spike probe (not a release tool). Derived from the TE-266 planning probe
// (Claude/Vitruvius/te266-probe/te266_gpu_prefill_probe.cpp): drives SuperSLM's public GPU handle path
// -- context, model map, ONE sequence at a caller-chosen context_cap, reset per document,
// SslmGpuSeqPrefillPromptForG5Bridge over the whole document -- reads the last position's residual
// through the bench accessors and applies the model's own final_norm on the host, emitting the
// T-2701 final-hidden frame per row (byte-compatible with the frozen per-token frames).
//
// mode 0: the shipped per-token chunk path (the handle-path baseline).
// mode 1: the row-widened prefill (SUPERSLM_T2790_WIDE_H rows per chunk), selected inside
//         SubmitAdmittedChunkForG5Bridge. Everything else -- pre-scan, snapshot, status mapping,
//         copy-back, this probe's own timing -- is the same code in both modes.
//
// Timing: the first document is run once untimed (pipeline creation, residency warm-up) in both
// modes, then every document is timed. Per-site GPU timings of the widened sites are printed in mode 1.
//
// Usage: t2790_probe_h<H> <model.sslm> <token-rows.txt> <frames.bin> <timings.tsv> <context_cap> <mode>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "sslm_marshal.h"

namespace superslm_gpu {
void T2790SetWidenedEnabled(bool enabled);
uint32_t T2790Rows();
void T2790ResetTimings();
void T2790GetTimings(double* site_ms_25, uint64_t* chunks, double* record_ms, double* wait_ms,
                     double* gpu_busy_ms);
}  // namespace superslm_gpu

using namespace superslm;
using enum SslmGpuStatus;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;
using superslm_marshal::WidenGainToInt32;
using Clock = std::chrono::steady_clock;

namespace {
bool ParseTokenIds(const std::string& spec, std::vector<int32_t>* out) {
	for (size_t pos = 0; pos < spec.size();) {
		const size_t end = spec.find(',', pos);
		const std::string field = spec.substr(pos, end == std::string::npos ? end : end - pos);
		if (field.empty()) return false;
		try { out->push_back(static_cast<int32_t>(std::stol(field))); } catch (...) { return false; }
		if (end == std::string::npos) break;
		pos = end + 1;
	}
	return !out->empty();
}
double Ms(Clock::time_point a, Clock::time_point b) {
	return std::chrono::duration<double, std::milli>(b - a).count();
}
const char* const kSiteNames[25] = {
    "attn_norm", "q_proj_gemm", "q_proj_tail", "kv_proj_gemm", "kv_proj_tail", "qk_norm", "rope_guard",
    "rope_commit", "attention_score", "softmax", "context_accumulate", "ctx_fold", "o_proj_gemm",
    "o_proj_tail", "attn_residual", "mlp_norm", "gate_proj_gemm", "gate_proj_tail", "up_proj_gemm",
    "up_proj_tail", "mlp_act", "down_proj_gemm", "down_proj_tail", "mlp_residual", "commit"};
}  // namespace

int main(int argc, char** argv) {
	if (argc != 7)
		return std::fprintf(stderr, "usage: %s <model> <rows.txt> <frames.bin> <timings.tsv> <context_cap> <mode 0|1>\n", argv[0]), 2;
	const int64_t cap = std::atoll(argv[5]);
	const int mode = std::atoi(argv[6]);
	superslm_gpu::T2790SetWidenedEnabled(mode == 1);
	std::vector<uint8_t> bytes;
	if (!ReadFile(argv[1], bytes)) return std::fprintf(stderr, "cannot read model\n"), 1;
	SslmModelView view;
	std::string error;
	if (SslmModel::Load(bytes.data(), bytes.size(), view, &error) != SslmModelStatus::Ok)
		return std::fprintf(stderr, "model load: %s\n", error.c_str()), 1;
	const SslmTensorView* final_gain = view.weights.Tensor("final_norm.gain");
	if (!final_gain) return std::fprintf(stderr, "missing final_norm.gain\n"), 1;
	bool ok = true;
	const CarriedScale final_scale = ReadCarriedScale(view.composition_constants, "final_norm", &ok);
	if (!ok) return std::fprintf(stderr, "missing final_norm constant\n"), 1;
	const std::vector<int32_t> final_gains = WidenGainToInt32(*final_gain);
	const size_t hidden = view.config.hidden_size;

	GpuContextConfig cc{};
	GpuResidencyConfig rc{};
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* gm = nullptr;
	SslmGpuSequenceHandle* seq = nullptr;
	if (sslm_gpu_context_create(cc, &ctx) != SSLM_OK) return std::fprintf(stderr, "context_create failed\n"), 1;
	if (sslm_gpu_model_map(ctx, &view, rc, &gm) != SSLM_OK) return std::fprintf(stderr, "model_map failed\n"), 1;
	if (sslm_gpu_seq_create(ctx, gm, cap, &seq) != SSLM_OK) return std::fprintf(stderr, "seq_create failed\n"), 1;

	std::vector<std::vector<int32_t>> docs;
	{
		std::ifstream in(argv[2]);
		if (!in) return std::fprintf(stderr, "cannot open rows\n"), 1;
		std::string line;
		while (std::getline(in, line)) {
			if (line.empty()) continue;
			std::vector<int32_t> t;
			if (!ParseTokenIds(line, &t)) return std::fprintf(stderr, "bad row %zu\n", docs.size()), 2;
			docs.push_back(std::move(t));
		}
	}
	auto run_doc = [&](const std::vector<int32_t>& t) -> SslmGpuStatus {
		if (sslm_gpu_seq_reset(ctx, seq) != SSLM_OK) return SSLM_DEVICE_LOST;
		return SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, t.data(), static_cast<int32_t>(t.size()),
		                                           superslm_gpu::kDispatchesPerLayer);
	};
	if (!docs.empty()) {  // untimed warm-up, both modes
		const SslmGpuStatus st = run_doc(docs[0]);
		if (st != SSLM_OK) return std::fprintf(stderr, "warm-up prefill failed status %u\n", static_cast<unsigned>(st)), 1;
	}
	superslm_gpu::T2790ResetTimings();

	std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
	std::ofstream tim(argv[4], std::ios::trunc);
	if (!out || !tim) return std::fprintf(stderr, "cannot open outputs\n"), 1;
	tim << "row\ttokens\tprefill_ms\n";
	size_t tokens_total = 0;
	double prefill_total = 0;
	const auto t_run0 = Clock::now();
	for (size_t row = 0; row < docs.size(); ++row) {
		const auto b = Clock::now();
		const SslmGpuStatus st = run_doc(docs[row]);
		const auto c = Clock::now();
		if (st != SSLM_OK) return std::fprintf(stderr, "prefill failed row %zu status %u\n", row, static_cast<unsigned>(st)), 1;
		const int8_t* resid = SslmGpuSeqHandleHiddenCodesForBench(seq);
		const CarriedScale incoming = *SslmGpuSeqHandleHiddenScaleForBench(seq);
		std::vector<int8_t> codes(hidden);
		CarriedScale fs{};
		const SslmForwardStatus fst = RmsNormSite(resid, final_gains.data(), hidden, incoming, final_scale,
		                                          codes.data(), &fs, "final_norm");
		if (fst != SslmForwardStatus::Ok) return std::fprintf(stderr, "final_norm row %zu: %s\n", row, SslmForwardStatusName(fst)), 1;
		const uint64_t magic = UINT64_C(0x54474D5331373032), count = hidden;
		out.write(reinterpret_cast<const char*>(&magic), 8);
		out.write(reinterpret_cast<const char*>(&count), 8);
		out.write(reinterpret_cast<const char*>(&fs.m), 8);
		out.write(reinterpret_cast<const char*>(&fs.e), 8);
		out.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(hidden));
		tim << row << "\t" << docs[row].size() << "\t" << Ms(b, c) << "\n";
		prefill_total += Ms(b, c);
		tokens_total += docs[row].size();
	}
	const double run_s = Ms(t_run0, Clock::now()) / 1000.0;
	std::printf("mode=%d rows_per_chunk=%u context_cap=%lld docs=%zu tokens=%zu run_s=%.3f tok_per_s=%.3f "
	            "prefill_ms=%.1f\n",
	            mode, superslm_gpu::T2790Rows(), static_cast<long long>(cap), docs.size(), tokens_total, run_s,
	            run_s > 0 ? tokens_total / run_s : 0.0, prefill_total);
	if (mode == 1) {
		double site[25];
		uint64_t chunks = 0;
		double rec = 0, wait = 0, busy = 0;
		superslm_gpu::T2790GetTimings(site, &chunks, &rec, &wait, &busy);
		double sum = 0;
		for (double v : site) sum += v;
		std::printf("chunks=%llu host_record_ms=%.1f (%.3f/chunk) fence_wait_ms=%.1f gpu_busy_ms=%.1f (%.3f/chunk)\n",
		            static_cast<unsigned long long>(chunks), rec, chunks ? rec / chunks : 0.0, wait, busy,
		            chunks ? busy / chunks : 0.0);
		std::printf("site\tms_total\tms_per_chunk\tshare\n");
		for (int s = 0; s < 25; ++s) {
			std::printf("%s\t%.1f\t%.4f\t%.3f\n", kSiteNames[s], site[s], chunks ? site[s] / chunks : 0.0,
			            sum > 0 ? site[s] / sum : 0.0);
		}
	}
	sslm_gpu_seq_release(ctx, seq);
	sslm_gpu_model_unmap(ctx, gm);
	sslm_gpu_context_destroy(ctx);
	return 0;
}
