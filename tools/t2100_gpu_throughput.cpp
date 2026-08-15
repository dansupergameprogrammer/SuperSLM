// T-2100: decode-step throughput, CPU oracle vs GPU port, real artifact, real hardware.
//
// WHY THIS EXISTS. The GPU-serial port has been proven bit-identical to the CPU oracle for
// twelve consecutive rounds -- on ONE token, every time. No round ever measured a token per
// second, so the arc could say "the port is correct" and could not say "the port is worth
// having." This harness answers the second question with the same setup C5 already uses:
// identical artifact, identical marshalling, identical entry points.
//
// WHAT IT MEASURES. N successive decode steps. Each step is one token through all 28 layers --
// exactly one `RunLayerLoop`/`RunLayerLoopGpu` call, the same call a generation loop makes --
// with the K/V workspace persisting across steps so `context_length` advances the way it does in
// real decode. One warmup step per side is run and discarded (device init, PSO compile, first-
// touch page faults); the reported figure is the mean over the timed steps.
//
// WHAT IT DOES NOT MEASURE, stated so no one reads more into the number than it carries: this
// times `RunLayerLoopGpu` AS IT EXISTS, which fence-waits on every call (a real, measured cost --
// see the phase decomposition below, `submit_wait_ms`). The K/V readback itself is NOT a
// significant contributor to that wait as of T-2101 (commit `d79f6b7`): the workspace is
// device-resident and each call reads back only the rows it actually wrote (~14 KiB at the 1.5B
// tier, not the whole 448 MiB workspace) -- a prior version of this comment claimed the opposite
// and was wrong from that commit forward, inside this same arc (code review
// `6d9e04e-t2101-gpu-throughput-review.md`, S2). The real remaining cost, decomposed and named by
// D-SLM3312/D-SLM3313, is GPU-busy time concentrated in the GEMM sites -- not host<->device
// marshalling, and not per-call submission overhead (`record_ms`/`readback_ms` are both small; see
// the phase table). The number below is a floor on what THIS calling convention (one synchronous
// dispatch chain per token, no pipelining across tokens) can do, not a ceiling on the kernels
// themselves -- D-SLM3314/D-SLM3315's own multi-group fix already moved that floor once.
//
// Throwaway harness, same precedent as tools/t2039_c5_harness.cpp.
// Usage: t2100_gpu_throughput <model.sslm> [steps] [token_id]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "sslm_marshal.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::PreflightScanWscFolds;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;

namespace {

double SecondsSince(const std::chrono::steady_clock::time_point& t0) {
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model.sslm> [steps] [token_id]\n", argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const int steps = argc >= 3 ? std::atoi(argv[2]) : 16;
	const int32_t token_id = argc >= 4 ? std::atoi(argv[3]) : 0;
	if (steps < 1) {
		std::fprintf(stderr, "steps must be >= 1\n");
		return 2;
	}

	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) {
		std::fprintf(stderr, "FAILED: could not read \"%s\"\n", model_path.c_str());
		return 1;
	}
	SslmModelView model_view;
	std::string model_err;
	const SslmModelStatus load_status =
	    SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err);
	if (load_status != SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=model_load: status=%s diagnostic=\"%s\"\n",
		             SslmModelStatusName(load_status), model_err.c_str());
		return 1;
	}
	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const size_t head_dim = model_view.config.head_dim;
	const size_t intermediate_size = model_view.config.intermediate_size;
	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	std::printf("model loaded: hidden_size=%zu layers=%u heads=%u/%u head_dim=%zu intermediate=%zu\n",
	            hidden_size, num_hidden_layers, num_heads, num_kv_heads, head_dim, intermediate_size);

	PreflightScanWscFolds(model_view);
	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l], &marshal_err)) {
			std::fprintf(stderr, "FAILED at stage=layer_weights_marshal: layer=%u diagnostic=\"%s\"\n", l,
			             marshal_err.c_str());
			return 1;
		}
	}

	const SslmTensorView* embed_w = model_view.weights.Tensor("embed");
	if (!embed_w) {
		std::fprintf(stderr, "FAILED: missing embed tensor\n");
		return 1;
	}
	bool ok = true;
	CarriedScale embed_site_constant = ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED: missing embed site constant\n");
		return 1;
	}
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);

	std::vector<int8_t> embed_codes(hidden_size);
	CarriedScale embed_scale{};
	const SslmForwardStatus est =
	    EmbedEntry(token_id, model_view.config.vocab_size, embed_weights, hidden_size,
	               embed_site_constant, embed_codes.data(), &embed_scale);
	if (est != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=embed: status=%s\n", SslmForwardStatusName(est));
		return 1;
	}

	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;
	std::printf("workspace (KV cache) bytes = %zu (%.2f MiB)\n", kv_bytes,
	            kv_bytes / (1024.0 * 1024.0));
	std::printf("timing %d decode steps per side (1 warmup step each, discarded)\n\n", steps);

	// Each side runs its own persistent workspace so context_length advances across steps,
	// exactly as a real decode loop does.
	struct Side {
		const char* name;
		bool gpu;
	};
	const Side sides[2] = {{"CPU (RunLayerLoop)", false}, {"GPU (RunLayerLoopGpu)", true}};

	// T-2100: per-step CPU/GPU equality across MULTIPLE steps. C5 proves bit-identity on ONE call,
	// which cannot exercise a weight-residency cache at all -- the first call is always a miss that
	// populates it, so every residency fast path is untested by a single-call harness. These arrays
	// capture each side's own per-step residual so the comparison below is over steps 2..N as well.
	std::vector<std::vector<int8_t>> step_codes[2];
	std::vector<int64_t> step_ctx[2];

	// T-2101 (dispatch-overhead decomposition, D-SLM3302/D-SLM3304's own follow-up): accumulated
	// per-phase GPU-side timing across every timed step, so the report below is a MEAN over the
	// same population the headline s/token figure is -- not one call's own possibly-atypical split.
	double gpu_record_ms_sum = 0.0, gpu_submit_wait_ms_sum = 0.0, gpu_busy_ms_sum = 0.0,
	       gpu_readback_ms_sum = 0.0;

	// T-2101 follow-up (per-site decomposition, D-SLM3312; per-dispatch parallelism, D-SLM3313's
	// own follow-up): the fixed, unconditional per-layer dispatch order `RunLayerLoopGpu` issues
	// (`superslm_gpu.cpp`'s own `bind_and_dispatch` call sequence) -- dispatch `d` within a call is
	// site `d % kSitesPerLayer` of layer `d / kSitesPerLayer`. 22 sites/layer now, not 17:
	// q_proj/o_proj/gate_proj/up_proj/down_proj each split into their own multi-group GEMM
	// dispatch immediately followed by their own (now leaner) requant dispatch; kv_proj stays
	// fused and single-dispatch. Summed across every timed GPU step below.
	constexpr int kSitesPerLayer = 22;
	const char* const kSiteNames[kSitesPerLayer] = {
	    "attn_norm",     "q_proj_gemm",         "q_proj",        "kv_proj",
	    "rope",          "attention_score",     "softmax",       "context_accumulate",
	    "ctx_fold",      "o_proj_gemm",         "o_proj",        "attn_residual",
	    "mlp_norm",      "gate_proj_gemm",      "gate_proj",     "up_proj_gemm",
	    "up_proj",       "mlp_act",             "down_proj_gemm", "down_proj",
	    "mlp_residual",  "commit"};
	double site_ms_sum[kSitesPerLayer] = {0.0};

	double per_token[2] = {0.0, 0.0};
	for (int s = 0; s < 2; ++s) {
		std::vector<uint8_t> ws(kv_bytes, 0);
		std::vector<int8_t> codes(hidden_size);
		SequenceLayerState seq;
		seq.hidden_codes = codes.data();
		seq.layer_index = 0;

		auto one_step = [&]() -> SslmForwardStatus {
			std::memcpy(codes.data(), embed_codes.data(), hidden_size);
			seq.hidden_scale = embed_scale;
			seq.layer_index = 0;
			if (sides[s].gpu) {
				return superslm_gpu::RunLayerLoopGpu(seq, layers.data(), num_hidden_layers,
				                                     num_hidden_layers, hidden_size, head_dim,
				                                     num_kv_heads, intermediate_size, context_cap,
				                                     model_view.rope_tables, ws.data(), ws.size());
			}
			return RunLayerLoop(seq, layers.data(), num_hidden_layers, num_hidden_layers, hidden_size,
			                    head_dim, num_kv_heads, intermediate_size, context_cap,
			                    model_view.rope_tables, ws.data(), ws.size());
		};

		const SslmForwardStatus warm = one_step();
		if (warm != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "%s: warmup step returned %s\n", sides[s].name,
			             SslmForwardStatusName(warm));
			return 1;
		}

		const auto t0 = std::chrono::steady_clock::now();
		double worst = 0.0, best = 1e30;
		for (int i = 0; i < steps; ++i) {
			const auto ts = std::chrono::steady_clock::now();
			const SslmForwardStatus st = one_step();
			const double dt = SecondsSince(ts);
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "%s: step %d returned %s\n", sides[s].name, i,
				             SslmForwardStatusName(st));
				return 1;
			}
			if (dt > worst) worst = dt;
			if (dt < best) best = dt;
			step_codes[s].push_back(codes);
			step_ctx[s].push_back(seq.context_length);
			if (sides[s].gpu) {
				const auto timing = superslm_gpu::LastCallTiming();
				gpu_record_ms_sum += timing.record_ms;
				gpu_submit_wait_ms_sum += timing.submit_wait_ms;
				gpu_busy_ms_sum += timing.gpu_busy_ms;
				gpu_readback_ms_sum += timing.readback_ms;
				const auto per_dispatch = superslm_gpu::LastCallPerDispatchTimingsMs();
				for (size_t d = 0; d < per_dispatch.size(); ++d) {
					site_ms_sum[d % kSitesPerLayer] += per_dispatch[d];
				}
			}
		}
		const double total = SecondsSince(t0);
		per_token[s] = total / steps;
		std::printf("%-24s %6.4f s/token   %8.2f tok/s   (min %.4f, max %.4f, total %.2f s over %d steps)\n",
		            sides[s].name, per_token[s], 1.0 / per_token[s], best, worst, total, steps);
	}

	// --- Per-step equality, every step, not just the first. ---
	int mismatched_steps = 0;
	for (size_t i = 0; i < step_codes[0].size() && i < step_codes[1].size(); ++i) {
		if (step_codes[0][i] != step_codes[1][i] || step_ctx[0][i] != step_ctx[1][i]) {
			if (mismatched_steps == 0) std::printf("\nDIVERGENCE first at timed step %zu\n", i);
			++mismatched_steps;
		}
	}
	std::printf("\nper-step CPU/GPU equality over %zu timed steps: %s (%d mismatched)\n",
	            step_codes[0].size(), mismatched_steps == 0 ? "IDENTICAL" : "DIVERGED",
	            mismatched_steps);

	std::printf("\nGPU vs CPU: %.2fx %s\n", per_token[0] / per_token[1],
	            per_token[1] < per_token[0] ? "FASTER" : "SLOWER");

	// T-2101 (dispatch-overhead decomposition): mean per-phase split over the SAME `steps` timed
	// GPU calls the headline figure above is computed from. `gpu_busy_ms` is the ONLY GPU-only
	// number here (a timestamp-query bracket around the call's own first/last dispatch); the other
	// three are CPU wall-clock splits of the surrounding work. The four should sum to
	// approximately the GPU side's own mean s/token above -- reported so the reader can check that,
	// not asserted.
	const double mean_record_ms = gpu_record_ms_sum / steps;
	const double mean_submit_wait_ms = gpu_submit_wait_ms_sum / steps;
	const double mean_gpu_busy_ms = gpu_busy_ms_sum / steps;
	const double mean_readback_ms = gpu_readback_ms_sum / steps;
	const double mean_sum_ms = mean_record_ms + mean_submit_wait_ms + mean_readback_ms;
	// Roofline: 1.5 GB of int8 weights read once per token against this card's own advertised
	// 496 GB/s memory bandwidth (RESUME's own derived, not measured, figure) -- restated here in
	// milliseconds for direct comparison against `mean_gpu_busy_ms`.
	const double roofline_ms = (1.5e9 / 496e9) * 1000.0;
	std::printf(
	    "\nGPU per-token phase decomposition (mean over %d timed steps):\n"
	    "  record       %8.3f ms  (CPU: build the command list -- pack-or-skip, every Upload/"
	    "MakeBuffer, every Dispatch's own recording)\n"
	    "  submit+wait  %8.3f ms  (CPU: ExecuteCommandLists -> fence signaled -- submission + "
	    "actual GPU execution + driver/OS scheduling, NOT GPU-only)\n"
	    "  GPU-busy     %8.3f ms  (GPU timestamp query, first dispatch -> last dispatch -- THE "
	    "number the roofline verdict below is judged against)\n"
	    "  readback     %8.3f ms  (CPU: map + memcpy the small readback buffers into seq/"
	    "workspace)\n"
	    "  sum (record+submit_wait+readback, GPU-busy excluded -- it is INSIDE submit+wait, not "
	    "additional) = %.3f ms, against the measured %.3f ms/token above\n"
	    "  roofline (1.5 GB weights / 496 GB/s, DERIVED not measured) = %.3f ms\n"
	    "  GPU-busy / roofline = %.2fx\n",
	    steps, mean_record_ms, mean_submit_wait_ms, mean_gpu_busy_ms, mean_readback_ms, mean_sum_ms,
	    per_token[1] * 1000.0, roofline_ms, mean_gpu_busy_ms / roofline_ms);

	// T-2101 follow-up (per-site decomposition, D-SLM3312's own follow-up): mean ms/token per site,
	// summed across all 28 layers and averaged over the same `steps` timed GPU calls as everything
	// above, sorted descending. Achieved bandwidth is reported for the six GEMM weight-matrix reads
	// -- T-2101's own per-dispatch split (D-SLM3313's own follow-up) moved that read traffic into
	// the five `*_gemm` sites specifically (q/o/gate/up/down_proj_gemm; kv_proj stays fused, its
	// own GEMM traffic charged to "kv_proj" as before) -- the now-leaner requant/bias-only sites of
	// the same base name touch only the already-computed wide row (O(out_channels) int64s, already
	// counted nowhere near bandwidth-interesting) and get no bytes figure. Everything else touches
	// only O(hidden_size) or O(context_length, still tiny this early in decode) data.
	struct SiteRow {
		int idx;
		double ms;
		double bytes_per_layer;  // 0 if not a GEMM site with a known dominant weight matrix
	};
	const size_t H = hidden_size, I = intermediate_size, KV = num_kv_heads * head_dim;
	std::vector<SiteRow> rows(kSitesPerLayer);
	for (int i = 0; i < kSitesPerLayer; ++i) {
		double bytes_per_layer = 0.0;
		const std::string name = kSiteNames[i];
		if (name == "q_proj_gemm" || name == "o_proj_gemm") bytes_per_layer = static_cast<double>(H) * H;
		else if (name == "kv_proj") bytes_per_layer = 2.0 * static_cast<double>(KV) * H;
		else if (name == "gate_proj_gemm" || name == "up_proj_gemm")
			bytes_per_layer = static_cast<double>(I) * H;
		else if (name == "down_proj_gemm") bytes_per_layer = static_cast<double>(H) * I;
		rows[i] = SiteRow{i, site_ms_sum[i] / steps, bytes_per_layer};
	}
	std::sort(rows.begin(), rows.end(), [](const SiteRow& a, const SiteRow& b) { return a.ms > b.ms; });
	double site_ms_total = 0.0;
	for (const auto& r : rows) site_ms_total += r.ms;
	std::printf(
	    "\nGPU per-site decomposition (mean ms/token, summed across %u layers, sorted descending; "
	    "sum of all %d sites = %.3f ms against GPU-busy %.3f ms above):\n"
	    "  %-20s %10s %8s %14s %12s\n",
	    num_hidden_layers, kSitesPerLayer, site_ms_total, mean_gpu_busy_ms, "site", "ms/token", "share",
	    "GB read/tok", "achieved GB/s");
	for (const auto& r : rows) {
		const double share_pct = 100.0 * r.ms / mean_gpu_busy_ms;
		if (r.bytes_per_layer > 0.0) {
			const double gb_per_token = r.bytes_per_layer * num_hidden_layers / 1e9;
			const double achieved_gbps = r.ms > 0.0 ? (gb_per_token / (r.ms / 1000.0)) : 0.0;
			std::printf("  %-20s %10.3f %7.1f%% %14.4f %12.2f\n", kSiteNames[r.idx], r.ms, share_pct,
			            gb_per_token, achieved_gbps);
		} else {
			std::printf("  %-20s %10.3f %7.1f%% %14s %12s\n", kSiteNames[r.idx], r.ms, share_pct, "--",
			            "--");
		}
	}
	// Uniformity verdict: coefficient of variation across the `kSitesPerLayer` (22, not 17) site
	// means. Low CoV = spread ~uniformly (per-dispatch fixed cost dominates); high CoV =
	// concentrated in specific sites (those sites' own kernels are genuinely slow).
	//
	// T-2101 (O1, code review 6d9e04e-t2101-gpu-throughput-review.md; M4, confirmation pass @
	// f7026db, correcting this paragraph's own inverted direction): this threshold's own
	// `site_cov > 1.0` cutoff was set (D-SLM3313) against a 17-site population. T-2101's own fix
	// round split five sites' GEMM step into its own `*_gemm` dispatch -- THOSE five carry the real
	// GEMM cost and are the population's own HEAVY rows (down_proj_gemm alone was 36.3% of GPU-busy
	// time in round 1, D-SLM3314's own §34.2); it is the ORIGINAL-NAME requant-only sites (q_proj,
	// o_proj, gate_proj, up_proj, down_proj) that became near-zero, since their own GEMM work moved
	// out from under them. Adding five near-zero values (the requant sites) to a population whose
	// heavy rows did not shrink pulls the MEAN down without proportionally shrinking the stdev,
	// which mechanically RAISES CoV relative to what the same underlying concentration would have
	// read against the original 17 -- the conclusion (marking rather than recalibrating) survives
	// this correction unchanged, only the stated MECHANISM was backwards. Every run this ticket
	// executed (D-SLM3313's own 1.82, T-2101's own 1.76 and 1.42) reads well clear of the 1.0
	// cutoff either way, so the verdict itself has not flipped -- but a future population change
	// close to the threshold should re-derive the cutoff rather than trust this one across another
	// headcount change.
	double site_mean = site_ms_total / kSitesPerLayer, site_var = 0.0;
	for (int i = 0; i < kSitesPerLayer; ++i) {
		const double d = (site_ms_sum[i] / steps) - site_mean;
		site_var += d * d;
	}
	const double site_stdev = std::sqrt(site_var / kSitesPerLayer);
	const double site_cov = site_mean > 0.0 ? site_stdev / site_mean : 0.0;
	std::printf(
	    "\nUniformity: mean %.4f ms/site/token, stdev %.4f ms, coefficient of variation %.2f -- "
	    "%s\n",
	    site_mean, site_stdev, site_cov,
	    site_cov > 1.0 ? "CONCENTRATED (a small number of sites dominate; see the top rows above)"
	                   : "roughly UNIFORM (no small subset of sites dominates)");
	// T-2101 (S2, code review 6d9e04e-t2101-gpu-throughput-review.md): this NOTE used to claim
	// RunLayerLoopGpu reads the complete K/V workspace back every call -- true when T-2100 wrote it,
	// false since commit d79f6b7 (this same arc, T-2101's own device-resident K/V fix), which made
	// the readback targeted: only the rows a call actually wrote, not the whole workspace. Rewritten
	// to state the CURRENT calling convention rather than repeat a claim one of this file's own
	// prior commits falsified.
	//
	// T-2101 (S5, confirmation pass @ f7026db, correcting the FIRST rewrite of this NOTE, which
	// itself said "the fence wait is real GPU-busy time" -- a category error `gpu_port.h`'s own
	// `GpuCallTiming` contract and the phase print eleven lines above both go out of their way to
	// prevent: `submit_wait_ms` is defined there as submission overhead PLUS actual GPU execution
	// PLUS driver/OS scheduling, explicitly NOT GPU-only; `gpu_busy_ms` is the GPU-only figure. The
	// two are not the same quantity by definition, whatever their measured relationship on any one
	// run. What §32's own three runs actually established, stated as the measurement it is rather
	// than a redefinition of the phase: `submit_wait_ms` tracked `gpu_busy_ms` within about 0.4 ms
	// IN THOSE RUNS -- an empirical closeness at that measurement, not a guarantee that holds by
	// construction. The async-lifecycle sentence is dropped rather than requalified: §34's own
	// per-site breakdown already puts the remaining cost in the GEMM kernels themselves (achieved
	// bandwidth still ~24x below the derived roofline even after the multi-group fix), which
	// pipelining submission across tokens does not reduce -- that claim belongs with the per-site
	// table's own honest accounting, not repeated here as a sequencing argument this NOTE cannot
	// support.
	std::printf(
	    "NOTE: RunLayerLoopGpu fence-waits on every call (the workspace itself, %.2f MiB at this\n"
	    "tier, is device-resident and read back only in the small targeted rows each call actually\n"
	    "wrote -- see the phase decomposition's own record_ms/readback_ms above, both small). The\n"
	    "phase decomposition above measures gpu_busy_ms (GPU-only, timestamp-query-derived) and\n"
	    "submit_wait_ms (submission + GPU execution + driver/OS scheduling, NOT GPU-only by\n"
	    "definition) separately for exactly this reason -- read the per-site table above for where\n"
	    "the GPU-busy time actually goes, not this NOTE.\n",
	    kv_bytes / (1024.0 * 1024.0));
	return 0;
}
