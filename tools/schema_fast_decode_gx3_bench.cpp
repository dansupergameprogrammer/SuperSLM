// Schema fast-decode plan §7.1, step GX.3, headless: the GPU backend's per-submit fixed cost, measured with
// the existing API only (no engine change). The question behind it: does a slice that runs M token rows cost
// clearly less than M one-row slices? The GPU path has no M-row dispatch (a chunk records each token's own
// full-depth dispatch body in turn, src/gpu/superslm_gpu.cpp SubmitOneSubChunkToFullDepthForG5Bridge), so the
// only saving a chunk can show is per submission: record, submit, fence and readback. Two readings:
//
//   chunk   SslmGpuSeqPrefillPromptForG5Bridge with M = 1..4 tokens in one call (full depth, one sub-chunk:
//           kT2169TdrSafeMaxChunkTokens is 4) against the same M tokens as M one-token calls. Each sample
//           starts from a reset sequence with the same untimed prefix, so every arm runs at the same depth.
//   batch   sslm_decode_step_batch_gpu over M = 1..6 sequences, 4 layers each, against M one-row
//           sslm_decode_step_gpu slices of 4 layers. Every sequence is parked at layer 20 (untimed), so a
//           batch-wide budget of M x 4 layers gives each sequence exactly its last 4 layers, and both arms
//           time the same layers [20, 24). The batch call submits and drains each sequence on its own
//           (gpu_1p0.cpp's named limitation), so this reading measures that wrapper, not a fused submission.
//
// Per M: `pairs` interleaved pairs after `warmup` unmeasured ones, the arm that runs first alternating per
// pair and the sizes rotated by one per round. Wall time covers the whole call (the prompt bridge blocks
// on its last sub-chunk; a decode slice is timed through its blocking sslm_gpu_ready). gpu_busy_ms is
// superslm_gpu::LastCallTiming() after each submission (the GPU's own timestamp window, which excludes
// recording, submission and readback), summed over the one-token arm; the batch arm reports none, because
// LastCallTiming keeps only the last of its M submissions. Ratios are per pair: chunk / singles and
// batch / one-row slices, with the median, IQR and the n = 15 sign-test interval [R(4), R(12)].
//
//   slices  (--slices-only, instead of the two above) one one-row sslm_decode_step_gpu slice of L layers per sample,
//           L in --slice-layers (default 4,8,12,24), the layers [N-L, N) after an untimed run to N-L: the slice's wall
//           time against its own gpu_busy_ms, which gives the per-submit overhead at that slice depth. `pairs` rounds
//           after `warmup`, L rotated by one per round.
//
// Usage: sslm_schema_fast_decode_gx3_bench <model.sslm> [--pairs 15] [--warmup 1] [--json PATH]
//                                           [--slices-only [--slice-layers 4,8,12,24]]
// The compiled shaders must sit in a shaders\ folder beside the executable (superslm_gpu ShaderPath).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
using enum SslmGpuStatus;

namespace {

using Clock = std::chrono::steady_clock;
double MsSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

struct Stat {
	size_t n = 0;
	double median = 0, q1 = 0, q3 = 0, lo = 0, hi = 0;
};
double Quantile(std::vector<double> v, double q) {
	std::sort(v.begin(), v.end());
	const double pos = q * static_cast<double>(v.size() - 1);
	const size_t i = static_cast<size_t>(pos);
	const double f = pos - static_cast<double>(i);
	return i + 1 < v.size() ? v[i] * (1 - f) + v[i + 1] * f : v[i];
}
Stat Summarize(const std::vector<double>& v) {
	Stat s;
	s.n = v.size();
	if (v.empty()) return s;
	s.median = Quantile(v, 0.5);
	s.q1 = Quantile(v, 0.25);
	s.q3 = Quantile(v, 0.75);
	std::vector<double> sorted = v;
	std::sort(sorted.begin(), sorted.end());
	// The n = 15 sign-test interval is [R(4), R(12)] (1-based order statistics); other n report the IQR ends.
	if (sorted.size() == 15) {
		s.lo = sorted[3];
		s.hi = sorted[11];
	} else {
		s.lo = s.q1;
		s.hi = s.q3;
	}
	return s;
}
std::string JStat(const Stat& s) {
	char b[256];
	std::snprintf(b, sizeof b, "{\"n\":%zu,\"median\":%.4f,\"q1\":%.4f,\"q3\":%.4f,\"sign_lo\":%.4f,\"sign_hi\":%.4f}", s.n,
	              s.median, s.q1, s.q3, s.lo, s.hi);
	return b;
}
const char* Arg(int argc, char** argv, const char* name, const char* def) {
	for (int i = 2; i + 1 < argc; ++i)
		if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
	return def;
}

bool LoadModel(const char* path, superslm::SslmModelView* view, std::vector<uint8_t>* bytes) {
	std::FILE* f = std::fopen(path, "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	bytes->resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	const size_t got = sz > 0 ? std::fread(bytes->data(), 1, bytes->size(), f) : 0;
	std::fclose(f);
	if (got != bytes->size()) return false;
	std::string err;
	if (superslm::SslmModel::Load(bytes->data(), bytes->size(), *view, &err) != superslm::SslmModelStatus::Ok) {
		std::fprintf(stderr, "model load failed: %s\n", err.c_str());
		return false;
	}
	return true;
}

[[noreturn]] void Die(const char* what, SslmGpuStatus st) {
	std::fprintf(stderr, "%s failed: status %d\n", what, static_cast<int>(st));
	std::exit(3);
}
void Ok(SslmGpuStatus st, const char* what) {
	if (st != SSLM_OK) Die(what, st);
}
SslmGpuStatus Drain(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
	int32_t ready = 0;
	SslmGpuStatus out = SSLM_OK, st = SSLM_OK;
	while (!ready) st = sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &out);
	return st != SSLM_OK ? st : out;
}

// A dispatch budget that buys exactly `layers` whole layers whether the model records 24 or 25 dispatches
// per layer (floor(budget / per_layer) = layers for both).
uint32_t BudgetFor(uint32_t layers) { return layers * superslm_gpu::kDispatchesPerLayer; }

constexpr int kPrefixTokens = 41;  // the example extraction prompt's depth
int32_t PrefixToken(int i) { return (i * 37 + 11) % 1000 + 100; }
int32_t SpanToken(int i) { return (i * 53 + 7) % 1000 + 100; }

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model.sslm> [--pairs 15] [--warmup 1] [--json PATH]\n", argv[0]);
		return 2;
	}
	const int pairs = std::atoi(Arg(argc, argv, "--pairs", "15"));
	const int warmup = std::atoi(Arg(argc, argv, "--warmup", "1"));
	const char* json_path = Arg(argc, argv, "--json", nullptr);
	bool slices_only = false;
	for (int i = 2; i < argc; ++i)
		if (std::strcmp(argv[i], "--slices-only") == 0) slices_only = true;
	std::vector<int> slice_layers;
	for (const char* p = Arg(argc, argv, "--slice-layers", "4,8,12,24"); *p;) {
		slice_layers.push_back(std::atoi(p));
		while (*p && *p != ',') ++p;
		if (*p == ',') ++p;
	}

	std::vector<uint8_t> bytes;
	superslm::SslmModelView view;
	if (!LoadModel(argv[1], &view, &bytes)) {
		std::fprintf(stderr, "cannot load %s\n", argv[1]);
		return 2;
	}
	const uint32_t layers = view.config.num_hidden_layers;
	const int64_t cap = static_cast<int64_t>(view.config.context_cap);
	std::printf("artifact %s layers %u context_cap %lld dispatches_per_layer_budget %u\n", argv[1], layers,
	            static_cast<long long>(cap), superslm_gpu::kDispatchesPerLayer);
	if (layers < 8 || layers % 4 != 0) {
		std::fprintf(stderr, "needs a layer count that is a multiple of 4 (got %u)\n", layers);
		return 2;
	}

	GpuContextConfig cfg{};
	GpuResidencyConfig rcfg{};
	SslmGpuContext* ctx = nullptr;
	Ok(sslm_gpu_context_create(cfg, &ctx), "sslm_gpu_context_create");
	SslmGpuModelHandle* model = nullptr;
	Ok(sslm_gpu_model_map(ctx, &view, rcfg, &model), "sslm_gpu_model_map");
	constexpr int kMaxSeqs = 6;
	SslmGpuSequenceHandle* seqs[kMaxSeqs] = {};
	for (int i = 0; i < kMaxSeqs; ++i) Ok(sslm_gpu_seq_create(ctx, model, cap, &seqs[i]), "sslm_gpu_seq_create");

	std::vector<int32_t> prefix(kPrefixTokens);
	for (int i = 0; i < kPrefixTokens; ++i) prefix[static_cast<size_t>(i)] = PrefixToken(i);
	auto fresh = [&](SslmGpuSequenceHandle* s) {
		Ok(sslm_gpu_seq_reset(ctx, s), "sslm_gpu_seq_reset");
		Ok(SslmGpuSeqPrefillPromptForG5Bridge(ctx, s, prefix.data(), kPrefixTokens, 0xFFFFFFFFu), "prefix prefill");
	};

	// ---- slices (--slices-only): one one-row decode slice of L layers, layers [N-L, N), against its own GPU time ----
	// The sequence takes a token and runs untimed to layer N-L; the timed slice is one sslm_decode_step_gpu of L layers
	// and its blocking sslm_gpu_ready. L rotates by one per round; the context grows by one token per sample.
	struct SliceRow {
		std::vector<double> wall, gpu, overhead;
	};
	std::vector<SliceRow> slices(slice_layers.size());
	if (slices_only) {
		for (int l : slice_layers) {
			if (l < 1 || static_cast<uint32_t>(l) > layers) {
				std::fprintf(stderr, "--slice-layers %d is outside 1..%u\n", l, layers);
				return 2;
			}
		}
		fresh(seqs[0]);
		int32_t next = 0;
		for (int round = 0; round < warmup + pairs; ++round) {
			for (size_t q = 0; q < slice_layers.size(); ++q) {
				const size_t si = (q + static_cast<size_t>(round)) % slice_layers.size();
				const uint32_t l = static_cast<uint32_t>(slice_layers[si]);
				Ok(sslm_gpu_seq_embed_token(ctx, seqs[0], SpanToken(next++)), "slice embed");
				if (l < layers) {
					Ok(sslm_decode_step_gpu(ctx, seqs[0], nullptr, BudgetFor(layers - l)), "slice park");
					Ok(Drain(ctx, seqs[0]), "slice park drain");
				}
				const auto t0 = Clock::now();
				Ok(sslm_decode_step_gpu(ctx, seqs[0], nullptr, BudgetFor(l)), "slice decode");
				Ok(Drain(ctx, seqs[0]), "slice drain");
				const double wall = MsSince(t0);
				const double gpu = superslm_gpu::LastCallTiming().gpu_busy_ms;
				if (round < warmup) continue;
				slices[si].wall.push_back(wall);
				slices[si].gpu.push_back(gpu);
				slices[si].overhead.push_back(wall - gpu);
			}
		}
	}

	// ---- chunk: M = 1..4 tokens in one prompt-bridge call against M one-token calls ----
	const std::vector<int> chunk_sizes = {1, 2, 3, 4};
	struct ChunkRow {
		std::vector<double> chunk_wall, chunk_gpu, single_wall, single_gpu, r_wall, r_gpu, chunk_overhead, single_overhead;
	};
	std::vector<ChunkRow> chunk(chunk_sizes.size());
	auto run_chunk = [&](int m, double* wall, double* gpu) {
		std::vector<int32_t> span(static_cast<size_t>(m));
		for (int i = 0; i < m; ++i) span[static_cast<size_t>(i)] = SpanToken(i);
		fresh(seqs[0]);
		const auto t0 = Clock::now();
		Ok(SslmGpuSeqPrefillPromptForG5Bridge(ctx, seqs[0], span.data(), m, 0xFFFFFFFFu), "chunk prefill");
		*wall = MsSince(t0);
		*gpu = superslm_gpu::LastCallTiming().gpu_busy_ms;
	};
	auto run_singles = [&](int m, double* wall, double* gpu) {
		fresh(seqs[0]);
		*wall = 0;
		*gpu = 0;
		for (int i = 0; i < m; ++i) {
			const int32_t tok = SpanToken(i);
			const auto t0 = Clock::now();
			Ok(SslmGpuSeqPrefillPromptForG5Bridge(ctx, seqs[0], &tok, 1, 0xFFFFFFFFu), "single prefill");
			*wall += MsSince(t0);
			*gpu += superslm_gpu::LastCallTiming().gpu_busy_ms;
		}
	};
	for (int round = 0; !slices_only && round < warmup + pairs; ++round) {
		for (size_t q = 0; q < chunk_sizes.size(); ++q) {
			const size_t si = (q + static_cast<size_t>(round)) % chunk_sizes.size();
			const int m = chunk_sizes[si];
			double cw = 0, cg = 0, sw = 0, sg = 0;
			if (round % 2 == 0) {
				run_chunk(m, &cw, &cg);
				run_singles(m, &sw, &sg);
			} else {
				run_singles(m, &sw, &sg);
				run_chunk(m, &cw, &cg);
			}
			if (round < warmup) continue;
			ChunkRow& r = chunk[si];
			r.chunk_wall.push_back(cw);
			r.chunk_gpu.push_back(cg);
			r.single_wall.push_back(sw);
			r.single_gpu.push_back(sg);
			r.r_wall.push_back(cw / sw);
			r.r_gpu.push_back(sg > 0 ? cg / sg : 0);
			r.chunk_overhead.push_back(cw - cg);
			r.single_overhead.push_back(sw - sg);
		}
	}

	// ---- batch: M = 1..6 sequences, 4 layers each (layers [L-4, L)), against M one-row slices ----
	const uint32_t park = layers - 4;
	std::vector<int32_t> next_token(kMaxSeqs, 0);
	auto park_at = [&](SslmGpuSequenceHandle* s, int idx) {
		// A sequence at full depth (or fresh) takes its next token, then runs untimed to layer L-4.
		Ok(sslm_gpu_seq_embed_token(ctx, s, SpanToken(next_token[static_cast<size_t>(idx)]++)), "embed");
		Ok(sslm_decode_step_gpu(ctx, s, nullptr, BudgetFor(park)), "park decode");
		Ok(Drain(ctx, s), "park drain");
	};
	if (!slices_only) for (int i = 0; i < kMaxSeqs; ++i) fresh(seqs[i]);
	const std::vector<int> batch_sizes = {1, 2, 3, 4, 5, 6};
	struct BatchRow {
		std::vector<double> batch_wall, one_wall, one_gpu, r_wall;
	};
	std::vector<BatchRow> batch(batch_sizes.size());
	auto run_batch = [&](int m, double* wall) {
		for (int i = 0; i < m; ++i) park_at(seqs[i], i);
		const SslmGpuAdapterHandle* adapters[kMaxSeqs] = {};
		SslmGpuStatus statuses[kMaxSeqs] = {};
		const auto t0 = Clock::now();
		Ok(sslm_decode_step_batch_gpu(ctx, seqs, adapters, static_cast<uint32_t>(m), BudgetFor(4) * static_cast<uint32_t>(m),
		                              statuses),
		   "batch decode");
		for (int i = 0; i < m; ++i) {
			Ok(statuses[i], "batch member");
			Ok(Drain(ctx, seqs[i]), "batch drain");
		}
		*wall = MsSince(t0);
	};
	auto run_one_rows = [&](int m, double* wall, double* gpu) {
		for (int i = 0; i < m; ++i) park_at(seqs[i], i);
		*wall = 0;
		*gpu = 0;
		for (int i = 0; i < m; ++i) {
			const auto t0 = Clock::now();
			Ok(sslm_decode_step_gpu(ctx, seqs[i], nullptr, BudgetFor(4)), "one-row decode");
			Ok(Drain(ctx, seqs[i]), "one-row drain");
			*wall += MsSince(t0);
			*gpu += superslm_gpu::LastCallTiming().gpu_busy_ms;
		}
	};
	for (int round = 0; !slices_only && round < warmup + pairs; ++round) {
		for (size_t q = 0; q < batch_sizes.size(); ++q) {
			const size_t si = (q + static_cast<size_t>(round)) % batch_sizes.size();
			const int m = batch_sizes[si];
			double bw = 0, ow = 0, og = 0;
			if (round % 2 == 0) {
				run_batch(m, &bw);
				run_one_rows(m, &ow, &og);
			} else {
				run_one_rows(m, &ow, &og);
				run_batch(m, &bw);
			}
			if (round < warmup) continue;
			BatchRow& r = batch[si];
			r.batch_wall.push_back(bw);
			r.one_wall.push_back(ow);
			r.one_gpu.push_back(og);
			r.r_wall.push_back(bw / ow);
		}
	}

	for (int i = 0; i < kMaxSeqs; ++i) sslm_gpu_seq_release(ctx, seqs[i]);
	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);

	if (slices_only) {
		std::string sj = "{\"mode\":\"gx3-slices\",\"rounds\":" + std::to_string(pairs) + ",\"warmup\":" + std::to_string(warmup) +
		                 ",\"layers\":" + std::to_string(layers) + ",\"prefix_tokens\":" + std::to_string(kPrefixTokens) + ",\"slices\":[";
		std::printf("slices (one one-row slice of L layers, layers [N-L, N)): L | wall ms | gpu_busy ms | wall - gpu_busy ms\n");
		for (size_t i = 0; i < slice_layers.size(); ++i) {
			const Stat w = Summarize(slices[i].wall), g = Summarize(slices[i].gpu), o = Summarize(slices[i].overhead);
			std::printf("  L=%d | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f]\n", slice_layers[i], w.median, w.lo, w.hi,
			            g.median, g.lo, g.hi, o.median, o.lo, o.hi);
			sj += std::string(i ? "," : "") + "{\"L\":" + std::to_string(slice_layers[i]) + ",\"wall_ms\":" + JStat(w) +
			      ",\"gpu_busy_ms\":" + JStat(g) + ",\"overhead_ms\":" + JStat(o) + "}";
		}
		sj += "]}";
		if (json_path) {
			std::FILE* f = std::fopen(json_path, "wb");
			if (!f) {
				std::fprintf(stderr, "cannot write %s\n", json_path);
				return 4;
			}
			std::fwrite(sj.data(), 1, sj.size(), f);
			std::fclose(f);
		}
		return 0;
	}
	std::string js = "{\"mode\":\"gx3\",\"pairs\":" + std::to_string(pairs) + ",\"warmup\":" + std::to_string(warmup) +
	                 ",\"layers\":" + std::to_string(layers) + ",\"prefix_tokens\":" + std::to_string(kPrefixTokens) +
	                 ",\"sub_chunk_max_tokens\":" + std::to_string(superslm_gpu::kT2169TdrSafeMaxChunkTokens) + ",\"chunk\":[";
	std::printf("chunk (full depth, %u layers): M | chunk wall / gpu ms | M singles wall / gpu ms | ratio wall [sign] | ratio gpu | "
	            "overhead (wall - gpu) chunk / singles ms\n",
	            layers);
	for (size_t i = 0; i < chunk_sizes.size(); ++i) {
		const ChunkRow& r = chunk[i];
		const Stat cw = Summarize(r.chunk_wall), cg = Summarize(r.chunk_gpu), sw = Summarize(r.single_wall),
		           sg = Summarize(r.single_gpu), rw = Summarize(r.r_wall), rg = Summarize(r.r_gpu),
		           co = Summarize(r.chunk_overhead), so = Summarize(r.single_overhead);
		std::printf("  M=%d | %.3f / %.3f | %.3f / %.3f | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] | %.3f / %.3f\n", chunk_sizes[i],
		            cw.median, cg.median, sw.median, sg.median, rw.median, rw.lo, rw.hi, rg.median, rg.lo, rg.hi, co.median,
		            so.median);
		js += std::string(i ? "," : "") + "{\"M\":" + std::to_string(chunk_sizes[i]) + ",\"submits\":" +
		      std::to_string((chunk_sizes[i] + static_cast<int>(superslm_gpu::kT2169TdrSafeMaxChunkTokens) - 1) /
		                     static_cast<int>(superslm_gpu::kT2169TdrSafeMaxChunkTokens)) +
		      ",\"chunk_wall_ms\":" + JStat(cw) + ",\"chunk_gpu_busy_ms\":" + JStat(cg) + ",\"singles_wall_ms\":" + JStat(sw) +
		      ",\"singles_gpu_busy_ms\":" + JStat(sg) + ",\"ratio_wall\":" + JStat(rw) + ",\"ratio_gpu_busy\":" + JStat(rg) +
		      ",\"chunk_overhead_ms\":" + JStat(co) + ",\"singles_overhead_ms\":" + JStat(so) + "}";
	}
	js += "],\"batch\":[";
	std::printf("batch (4 layers each, layers [%u, %u)): M | batch wall ms | M one-row slices wall / gpu ms | ratio wall [sign]\n",
	            park, layers);
	for (size_t i = 0; i < batch_sizes.size(); ++i) {
		const BatchRow& r = batch[i];
		const Stat bw = Summarize(r.batch_wall), ow = Summarize(r.one_wall), og = Summarize(r.one_gpu), rw = Summarize(r.r_wall);
		std::printf("  M=%d | %.3f | %.3f / %.3f | %.4f [%.4f, %.4f]\n", batch_sizes[i], bw.median, ow.median, og.median,
		            rw.median, rw.lo, rw.hi);
		js += std::string(i ? "," : "") + "{\"M\":" + std::to_string(batch_sizes[i]) + ",\"batch_wall_ms\":" + JStat(bw) +
		      ",\"one_row_wall_ms\":" + JStat(ow) + ",\"one_row_gpu_busy_ms\":" + JStat(og) + ",\"ratio_wall\":" + JStat(rw) + "}";
	}
	js += "]}";
	if (json_path) {
		std::FILE* f = std::fopen(json_path, "wb");
		if (!f) {
			std::fprintf(stderr, "cannot write %s\n", json_path);
			return 4;
		}
		std::fwrite(js.data(), 1, js.size(), f);
		std::fclose(f);
	}
	return 0;
}
