// Multi-row GPU plan rev 1.3.2, cell 10.2 (slice MR4): the cost table's box bench. It measures C(M), the wall time of
// one whole-depth pass of an M-row chunk, for M = 1..8 through the published verbs, and writes the table header
// include/superslm/gpu_rows_cost_table.h in its own layout (kGpuRowsCostTableMeasured = true), and the CSV that
// tests/multirow-gpu/mr4_box.py check102 grades.
//
// Per M: 1 warm-up and --runs (30) timed runs. A run resets one dedicated sequence, feeds --p0 prompt tokens one row
// at a time (untimed), then times:
//   C   sslm_gpu_seq_embed_rows(PROMPT, M) + one sslm_decode_step_gpu of L x DispatchesPerLayer + sslm_gpu_ready(block):
//       the chunk from opening to its close (M = 1 is the one-row body);
//   F   SslmGpuSeqFinishTokenForG5Bridge on the closed chunk: the last row's finish, one row's logits and token.
// Filed per M: the median, p95 and max of C, and of the "with logits" column C + M x F.
//
// The logits column is a STAND-IN until MR5 widens the logits stage (plan §15 item 10): this build has no per-row
// logits for a chunk, so G(M) is taken as M one-row finishes. That overstates G(M) if MR5's stage amortizes across
// rows; the header records it as measured from the stand-in.
//
// The table: C(M) at its MAXIMUM over the timed runs (MR-D15); r(M) = ceil(1000 x maxC(M) / maxC(1)) and
// g(M) = ceil(1000 x (max(C + M x F) - maxC(M)) / maxC(1)), integer per-mille, rounded up (costs round up, §5.1), the
// same arithmetic as mr4_box.py check102.
//
// The CSV (--csv): `model,M,logits,runs,warmup,median_us,p95_us,max_us`, one row per M and logits 0 (C(M)) and 1 (the
// with-logits column), in whole us (rounded up). --model-name names the model column (mr4_box.py: 0.5B, 1.5B); a
// second run appends with --csv-append. A box run is 30 runs and 1 warm-up, the defaults.
//
// Usage: multirow_cost_bench --model=PATH [--shaders=DIR] [--adapter=PATH] [--device-head] [--runs=30] [--warmup=1]
//                            [--p0=16] [--out=PATH] [--json=PATH] [--csv=PATH [--csv-append]] [--model-name=NAME]
//   --out   where the header is written (default: not written; the table is printed either way)
//   --json  the per-run samples and summaries
// Exit 0 when every run's chunk closed OK with all M rows committed.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"

using enum SslmGpuStatus;

namespace {

using Clock = std::chrono::steady_clock;
double UsSince(Clock::time_point t0) { return std::chrono::duration<double, std::micro>(Clock::now() - t0).count(); }

bool ReadFileBytes(const std::string& path, std::vector<uint8_t>* out) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	const size_t n = sz > 0 ? std::fread(out->data(), 1, out->size(), f) : 0;
	std::fclose(f);
	return n == out->size();
}

int32_t TokenAt(int i, int32_t vocab) {
	return static_cast<int32_t>((1103515245u * static_cast<uint32_t>(i + 7) + 12345u) % static_cast<uint32_t>(vocab));
}

struct Env {
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	SslmGpuAdapterHandle* adapter = nullptr;
	SslmGpuSequenceHandle* seq = nullptr;
	uint32_t L = 0, dpl = 0;
	int32_t vocab = 0;
};

SslmGpuStatus StepDrain(const Env& e, uint32_t budget) {
	SslmGpuStatus st = sslm_decode_step_gpu(e.ctx, e.seq, e.adapter, budget);
	if (st != SSLM_OK) return st;
	int32_t ready = 0;
	SslmGpuStatus out = SSLM_OK;
	if ((st = sslm_gpu_ready(e.ctx, e.seq, 1, &ready, &out)) != SSLM_OK) return st;
	return ready ? out : SSLM_DEVICE_LOST;
}

// One prompt row, untimed: embed and the whole depth, resting there with no finish (as a prompt row does).
SslmGpuStatus PromptRow(const Env& e, int32_t tok) {
	const SslmGpuStatus st = sslm_gpu_seq_embed_token(e.ctx, e.seq, tok);
	return st != SSLM_OK ? st : StepDrain(e, e.L * e.dpl);
}

struct Sample {
	double c_us = 0, f_us = 0;
};

// One run at M rows: the timed chunk and the timed finish after it.
bool Run(const Env& e, uint32_t m, int p0, Sample* s, std::string* why) {
	char buf[160];
	if (sslm_gpu_seq_reset(e.ctx, e.seq) != SSLM_OK) return *why = "reset", false;
	for (int i = 0; i < p0; ++i) {
		const SslmGpuStatus st = PromptRow(e, TokenAt(i, e.vocab));
		if (st != SSLM_OK) {
			std::snprintf(buf, sizeof buf, "prefix row %d -> %d", i, static_cast<int>(st));
			return *why = buf, false;
		}
	}
	int32_t rows[SSLM_GPU_MAX_ROWS];
	for (uint32_t r = 0; r < m; ++r) rows[r] = TokenAt(p0 + 100 + static_cast<int>(r), e.vocab);
	SslmGpuRowsSpec spec{};
	spec.struct_size = sizeof spec;
	spec.kind = SSLM_GPU_ROWS_PROMPT;
	spec.tokens = rows;
	spec.count = static_cast<int32_t>(m);
	int32_t admitted = 0, committed = 0, open = 1;
	auto t0 = Clock::now();
	SslmGpuStatus st = sslm_gpu_seq_embed_rows(e.ctx, e.seq, &spec, &admitted);
	if (st == SSLM_OK) st = StepDrain(e, e.L * e.dpl);
	s->c_us = UsSince(t0);
	sslm_gpu_seq_last_rows_committed(e.seq, &committed);
	sslm_gpu_seq_rows_open(e.seq, &open);
	if (st != SSLM_OK || admitted != static_cast<int32_t>(m) || committed != static_cast<int32_t>(m) || open != 0) {
		std::snprintf(buf, sizeof buf, "chunk -> %d, admitted %d, committed %d, open %d", static_cast<int>(st), admitted,
		              committed, open);
		return *why = buf, false;
	}
	int32_t tok = -1;
	t0 = Clock::now();
	st = SslmGpuSeqFinishTokenForG5Bridge(e.ctx, e.seq, &tok);
	s->f_us = UsSince(t0);
	if (st != SSLM_OK) {
		std::snprintf(buf, sizeof buf, "finish -> %d", static_cast<int>(st));
		return *why = buf, false;
	}
	return true;
}

struct Stats {
	double median = 0, p95 = 0, max = 0;
};

Stats Summarize(std::vector<double> v) {
	Stats s;
	if (v.empty()) return s;
	std::sort(v.begin(), v.end());
	const size_t n = v.size();
	s.median = n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
	s.p95 = v[std::min(n - 1, static_cast<size_t>((95 * n + 99) / 100) - 1)];  // nearest rank
	s.max = v.back();
	return s;
}

int64_t CeilPerMille(double num, double den) {
	const double v = 1000.0 * num / den;
	const int64_t n = static_cast<int64_t>(v);
	return (static_cast<double>(n) < v) ? n + 1 : n;
}

}  // namespace

int main(int argc, char** argv) {
	std::string model_path, adapter_path, shaders, out_path, json_path, csv_path, model_name = "model";
	bool device_head = false, csv_append = false;
	int runs = 30, warmup = 1, p0 = 16;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--model=", 0) == 0) model_path = a.substr(8);
		else if (a.rfind("--adapter=", 0) == 0) adapter_path = a.substr(10);
		else if (a.rfind("--shaders=", 0) == 0) shaders = a.substr(10);
		else if (a.rfind("--out=", 0) == 0) out_path = a.substr(6);
		else if (a.rfind("--json=", 0) == 0) json_path = a.substr(7);
		else if (a.rfind("--csv=", 0) == 0) csv_path = a.substr(6);
		else if (a == "--csv-append") csv_append = true;
		else if (a.rfind("--model-name=", 0) == 0) model_name = a.substr(13);
		else if (a.rfind("--runs=", 0) == 0) runs = std::atoi(a.c_str() + 7);
		else if (a.rfind("--warmup=", 0) == 0) warmup = std::atoi(a.c_str() + 9);
		else if (a.rfind("--p0=", 0) == 0) p0 = std::atoi(a.c_str() + 5);
		else if (a == "--device-head") device_head = true;
		else {
			std::printf("usage: multirow_cost_bench --model=PATH [--shaders=DIR] [--adapter=PATH] [--device-head] "
			            "[--runs=30] [--warmup=1] [--p0=16] [--out=PATH] [--json=PATH] [--csv=PATH [--csv-append]] [--model-name=NAME]\n");
			return 2;
		}
	}
	if (runs < 1 || warmup < 0 || p0 < 0) {
		std::printf("FAIL --runs must be >= 1, --warmup and --p0 >= 0\n");
		return 2;
	}
	std::vector<uint8_t> model_bytes, adapter_bytes;
	superslm::SslmModelView view{}, adapter_view{};
	std::string err;
	if (model_path.empty() || !ReadFileBytes(model_path, &model_bytes) ||
	    superslm::SslmModel::Load(model_bytes.data(), model_bytes.size(), view, &err) != superslm::SslmModelStatus::Ok) {
		std::printf("FAIL cannot load model '%s' %s\n", model_path.c_str(), err.c_str());
		return 1;
	}
	if (!adapter_path.empty() &&
	    (!ReadFileBytes(adapter_path, &adapter_bytes) ||
	     superslm::SslmModel::Load(adapter_bytes.data(), adapter_bytes.size(), adapter_view, &err) !=
	         superslm::SslmModelStatus::Ok)) {
		std::printf("FAIL cannot load adapter '%s' %s\n", adapter_path.c_str(), err.c_str());
		return 1;
	}
	Env e;
	e.vocab = static_cast<int32_t>(view.config.vocab_size);
	e.L = view.config.num_hidden_layers;
	bool qk = false;
	for (uint32_t l = 0; l < e.L; ++l)
		if (view.weights.Tensor("layer" + std::to_string(l) + ".q_norm.gain") != nullptr) qk = true;
	e.dpl = superslm_gpu::DispatchesPerLayer(qk);
	GpuContextConfig cfg{};
	cfg.shader_dir = shaders.empty() ? nullptr : shaders.c_str();
	GpuResidencyConfig res{};
	res.flags = device_head ? SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE : 0u;
	if (sslm_gpu_context_create(cfg, &e.ctx) != SSLM_OK || sslm_gpu_model_map(e.ctx, &view, res, &e.model) != SSLM_OK) {
		std::printf("FAIL context or model map\n");
		return 1;
	}
	if (!adapter_path.empty() && sslm_gpu_adapter_map(e.ctx, e.model, &adapter_view, &e.adapter) != SSLM_OK) {
		std::printf("FAIL adapter map\n");
		return 1;
	}
	if (sslm_gpu_seq_create(e.ctx, e.model, p0 + 2 * SSLM_GPU_MAX_ROWS, &e.seq) != SSLM_OK ||
	    (e.adapter && sslm_gpu_seq_bind_adapter(e.ctx, e.seq, e.adapter) != SSLM_OK)) {
		std::printf("FAIL sequence\n");
		return 1;
	}
	std::printf("multirow_cost_bench: L = %u, %u dispatches per layer, p0 = %d, %d warm-up + %d timed runs per M%s%s\n",
	            e.L, e.dpl, p0, warmup, runs, device_head ? ", device head" : ", host head",
	            e.adapter ? ", adapter bound" : "");

	constexpr uint32_t kMaxM = SSLM_GPU_MAX_ROWS;
	std::vector<Sample> samples[kMaxM + 1];
	Stats c_stats[kMaxM + 1], cl_stats[kMaxM + 1], mf_stats[kMaxM + 1];
	for (uint32_t m = 1; m <= kMaxM; ++m) {
		for (int i = 0; i < warmup + runs; ++i) {
			Sample s;
			std::string why;
			if (!Run(e, m, p0, &s, &why)) {
				std::printf("FAIL M = %u run %d: %s\n", m, i, why.c_str());
				return 1;
			}
			if (i >= warmup) samples[m].push_back(s);
		}
		std::vector<double> c, cl, mf;
		for (const Sample& s : samples[m]) {
			c.push_back(s.c_us);
			mf.push_back(m * s.f_us);
			cl.push_back(s.c_us + m * s.f_us);
		}
		c_stats[m] = Summarize(c);
		cl_stats[m] = Summarize(cl);
		mf_stats[m] = Summarize(mf);
	}

	int64_t r[kMaxM + 1] = {0}, g[kMaxM + 1] = {0};
	std::printf("\n  M | C(M) us median    p95      max | with logits (stand-in) median    p95      max | r(M) g(M)\n");
	for (uint32_t m = 1; m <= kMaxM; ++m) {
		r[m] = m == 1 ? 1000 : CeilPerMille(c_stats[m].max, c_stats[1].max);
		g[m] = CeilPerMille(cl_stats[m].max - c_stats[m].max, c_stats[1].max);
		std::printf("  %u | %13.0f %8.0f %8.0f | %29.0f %8.0f %8.0f | %4lld %4lld\n", m, c_stats[m].median,
		            c_stats[m].p95, c_stats[m].max, cl_stats[m].median, cl_stats[m].p95, cl_stats[m].max,
		            static_cast<long long>(r[m]), static_cast<long long>(g[m]));
	}
	std::printf("  port rule C(4) / (4 x C(1)) at the median: %.3f (MR0: <= 0.6)\n",
	            c_stats[4].median / (4.0 * c_stats[1].median));

	if (!json_path.empty()) {
		std::FILE* f = std::fopen(json_path.c_str(), "w");
		if (!f) {
			std::printf("FAIL cannot write %s\n", json_path.c_str());
			return 1;
		}
		std::fprintf(f, "{\"L\": %u, \"p0\": %d, \"warmup\": %d, \"runs\": %d, \"device_head\": %s, \"rows\": [", e.L, p0,
		             warmup, runs, device_head ? "true" : "false");
		for (uint32_t m = 1; m <= kMaxM; ++m) {
			std::fprintf(f, "%s\n {\"M\": %u, \"r_permille\": %lld, \"g_permille\": %lld, \"c_us\": [", m == 1 ? "" : ",", m,
			             static_cast<long long>(r[m]), static_cast<long long>(g[m]));
			for (size_t i = 0; i < samples[m].size(); ++i) std::fprintf(f, "%s%.1f", i ? ", " : "", samples[m][i].c_us);
			std::fprintf(f, "], \"finish_us\": [");
			for (size_t i = 0; i < samples[m].size(); ++i) std::fprintf(f, "%s%.1f", i ? ", " : "", samples[m][i].f_us);
			std::fprintf(f, "], \"c\": {\"median\": %.1f, \"p95\": %.1f, \"max\": %.1f}", c_stats[m].median, c_stats[m].p95,
			             c_stats[m].max);
			std::fprintf(f, ", \"with_logits_standin\": {\"median\": %.1f, \"p95\": %.1f, \"max\": %.1f}}", cl_stats[m].median,
			             cl_stats[m].p95, cl_stats[m].max);
		}
		std::fprintf(f, "\n]}\n");
		std::fclose(f);
	}

	if (!csv_path.empty()) {
		std::FILE* f = std::fopen(csv_path.c_str(), csv_append ? "ab" : "wb");
		if (!f) {
			std::printf("FAIL cannot write %s\n", csv_path.c_str());
			return 1;
		}
		if (!csv_append) std::fprintf(f, "model,M,logits,runs,warmup,median_us,p95_us,max_us\n");
		auto up = [](double v) { return static_cast<long long>(CeilPerMille(v, 1000.0)); };  // whole us, rounded up
		for (uint32_t m = 1; m <= kMaxM; ++m) {
			for (int lg = 0; lg < 2; ++lg) {
				const Stats& st = lg ? cl_stats[m] : c_stats[m];
				std::fprintf(f, "%s,%u,%d,%d,%d,%lld,%lld,%lld\n", model_name.c_str(), m, lg, runs, warmup, up(st.median),
				             up(st.p95), up(st.max));
			}
		}
		std::fclose(f);
		std::printf("wrote %s\n", csv_path.c_str());
	}

	if (!out_path.empty()) {
		std::FILE* f = std::fopen(out_path.c_str(), "wb");
		if (!f) {
			std::printf("FAIL cannot write %s\n", out_path.c_str());
			return 1;
		}
		std::fprintf(f,
		             "// Multi-row GPU plan rev 1.3.2, slice MR4: the cost table in C18 (plan §4.7, §5.1, cell 10.2; rulings item 12).\n"
		             "//\n"
		             "// WRITTEN BY tools/multirow_cost_bench (model %s): L = %u, p0 = %d, %d warm-up + %d timed runs per M, %s.\n"
		             "// C(1) max = %.0f us. r(M) = ceil(1000 x maxC(M) / maxC(1)); g(M) = ceil(1000 x (max(C + M x F) - maxC(M)) / maxC(1)),\n"
		             "// F the closing row's one-row finish: a stand-in for the per-row logits stage until MR5.\n"
		             "// Host-only data for the plugin's integer chooser (MR6); index M - 1. See the estimates' header for the layout.\n"
		             "#ifndef SUPERSLM_GPU_ROWS_COST_TABLE_H\n#define SUPERSLM_GPU_ROWS_COST_TABLE_H\n\n#include <cstdint>\n\n"
		             "namespace superslm_gpu {\n\n"
		             "constexpr bool kGpuRowsCostTableMeasured = true;\n"
		             "constexpr uint32_t kGpuRowCostUs = 6200;  // GpuRowCostMs's default (6.2 ms) in us: C(1), the M = 1 entry\n\n",
		             model_name.c_str(), e.L, p0, warmup, runs, device_head ? "device head" : "host head", c_stats[1].max);
		const char* names[2] = {"kGpuRowsCostRatioPerMille", "kGpuRowsLogitsRatioPerMille"};
		for (int t = 0; t < 2; ++t) {
			std::fprintf(f, "constexpr uint32_t %s[8] = {\n", names[t]);
			for (uint32_t m = 1; m <= kMaxM; ++m)
				std::fprintf(f, "    %lld,  // M = %u\n", static_cast<long long>(t == 0 ? r[m] : g[m]), m);
			std::fprintf(f, "};\n\n");
		}
		std::fprintf(f, "}  // namespace superslm_gpu\n\n#endif  // SUPERSLM_GPU_ROWS_COST_TABLE_H\n");
		std::fclose(f);
		std::printf("wrote %s\n", out_path.c_str());
	}
	sslm_gpu_seq_release(e.ctx, e.seq);
	std::printf("PASS multirow_cost_bench: %u M values x %d runs\n", kMaxM, runs);
	return 0;
}
