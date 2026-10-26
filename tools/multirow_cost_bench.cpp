// Multi-row GPU plan rev 1.3.2, cell 10.2 (slice MR4): the cost table's box bench. It measures C(M), the GPU time of
// one whole-depth pass of an M-row chunk, for M = 1..8 through the published verbs, and writes the table header
// include/superslm/gpu_rows_cost_table.h in its own layout, and the CSV that tests/multirow-gpu/mr4_box.py check102
// grades.
//
// Per M: 1 warm-up and --runs (30) timed runs. A run resets one dedicated sequence, feeds --p0 prompt tokens one row
// at a time (untimed), then:
//   C      sslm_gpu_seq_embed_rows(PROMPT, M) + one sslm_decode_step_gpu of L x DispatchesPerLayer + sslm_gpu_ready
//          (block): the chunk from opening to its close, one submission (M = 1 is the one-row body). C is that
//          submission's GPU window, superslm_gpu::LastCallTiming().gpu_busy_ms, read after the close: the recorder's
//          timestamp pair around its first and last dispatch, on the GPU's clock. That is what the table prices
//          (GpuRowCostMs is GPU work), so r(M) is built from it (MR4 review S2, ruling 18). The wall time of the same
//          span (host embed, recording, submission, fence wait, read-back, commit) is filed beside it as information.
//   F      SslmGpuSeqFinishTokenForG5Bridge on the closed chunk: the last row's finish, wall time.
//
// The logits = 1 rows (MR5, rulings 18a, 27, 33 S7): an EXACT_DRAFT chunk of M rows [p, d1 .. d(M-1)] whose drafts are
// the model's own greedy continuation (read once through the Off bridge), so every row agrees and the close finishes
// all M rows. Per run:
//   X      sslm_gpu_seq_embed_rows(EXACT_DRAFT, M) + one sslm_decode_step_gpu of L x DispatchesPerLayer + sslm_gpu_ready
//          (block): the chunk from opening to its Exact close. Its GPU time is the layers' submission
//          (LastCallTiming().gpu_busy_ms) plus the widened logits stage's own bracket (LastLogitsStageTiming().gpu_ms);
//          its wall time is filed beside it.
//   S      the widened logits stage alone (LastLogitsStageTiming().gpu_ms): the CSV's logits_stage_gpu_us, at its
//          maximum over the timed runs, whole us rounded up. mr4_box.py prices g(M) = ceil(S(M) x 1000 / C(1)).
//          M >= 2 only: a one-row close takes Off's own one-row logits call, untimed (ruling 37), so the M = 1 row's
//          column is empty and g(1) is the table's (kGpuRowsLogitsRatioPerMille[0]).
//   HF     the close's host finishes (LastLogitsStageTiming().host_finish_ms: final_norm and each row's logits path,
//          narrowing, mask and argmax; the stage excluded): the CSV's host_finish_us, its maximum, report only (the GPU
//          frame budget stays GPU-only; the host finish runs on the worker thread).
// The Exact rows need the head on the device (EXACT_DRAFT's admission), so without --device-head they run on a second
// mapping of the model that has it. They run unbound, where the close's finish takes the device head and the stage
// runs. --schema=NAME also runs them bound to that schema (the certified models' shape: on a tied model the bound close
// finishes on the host's legal rows and runs no stage); host_finish_us is then the bound close's, and the unbound
// one is printed beside it. The written header (--out) carries the measured r(M) and g(M) (g(1) from the table).
//
// The table: C(M) at its MAXIMUM over the timed runs (MR-D15); r(M) = ceil(1000 x maxC_gpu(M) / maxC_gpu(1)), integer
// per-mille, rounded up (costs round up, §5.1).
//
// The CSV (--csv): `model,M,logits,runs,warmup,median_us,p95_us,max_us,clock,wall_median_us,wall_p95_us,wall_max_us,
// logits_stage_gpu_us,host_finish_us`, one row per M and logits 0 and 1, whole us rounded up. logits 0: the timing
// columns are C's GPU time (clock `gpu`), the wall_* columns its wall time, the last two empty. logits 1: X's GPU time
// (clock `gpu`), its wall time, S's maximum and HF's maximum. --model-name names the model column (mr4_box.py: 0.5B,
// 1.5B); a second run appends with --csv-append. A box run is 30 runs and 1 warm-up, the defaults.
//
// Usage: multirow_cost_bench --model=PATH [--shaders=DIR] [--adapter=PATH] [--device-head] [--schema=NAME] [--runs=30]
//                            [--warmup=1] [--p0=16] [--out=PATH] [--json=PATH] [--csv=PATH [--csv-append]] [--model-name=NAME]
//   --out   where the header is written (default: not written; the table is printed either way)
//   --json  the per-run samples and summaries
//   --quick 3 timed runs at p0 = 3 (tools/multirow_wide_smoke.sh --quick; never a box run)
// Exit 0 when every run's chunk closed OK with all M rows committed.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"
#include "superslm/gpu_rows_cost_table.h"  // g(M)'s estimates, carried into the written table
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
	SslmGpuModelHandle* model_x = nullptr;    // the Exact rows' mapping: the head on the device
	SslmGpuAdapterHandle* adapter_x = nullptr;
	SslmGpuSequenceHandle* seq_x = nullptr;
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
	double c_us = 0, c_gpu_us = 0, f_us = 0;  // C's wall and GPU time, the finish's wall time
};

// The Exact rows (logits = 1): X's wall and GPU time (layers + stage), S, HF.
struct XSample {
	double x_us = 0, x_gpu_us = 0, stage_us = 0, host_us = 0;
};

// The Exact sequence at its start: reset, bound to schema_index (-1: unbound), the prefix rows, and the Off finish that leaves
// the pending token p in hand.
SslmGpuStatus ExactStart(const Env& e, int32_t schema_index, int p0, int32_t* p) {
	SslmGpuStatus st = sslm_gpu_seq_reset(e.ctx, e.seq_x);
	if (st == SSLM_OK) st = SslmGpuSeqSetSchemaForG5Bridge(e.ctx, e.seq_x, schema_index);  // -1 unbinds (a reset keeps it)
	if (st == SSLM_OK && e.adapter_x) st = sslm_gpu_seq_bind_adapter(e.ctx, e.seq_x, e.adapter_x);
	for (int i = 0; st == SSLM_OK && i < p0; ++i) {
		st = sslm_gpu_seq_embed_token(e.ctx, e.seq_x, TokenAt(i, e.vocab));
		if (st != SSLM_OK) break;
		if ((st = sslm_decode_step_gpu(e.ctx, e.seq_x, e.adapter_x, e.L * e.dpl)) != SSLM_OK) break;
		int32_t ready = 0;
		SslmGpuStatus out = SSLM_OK;
		if ((st = sslm_gpu_ready(e.ctx, e.seq_x, 1, &ready, &out)) == SSLM_OK) st = ready ? out : SSLM_DEVICE_LOST;
	}
	if (st == SSLM_OK) st = SslmGpuSeqFinishTokenForG5Bridge(e.ctx, e.seq_x, p);
	return st;
}

// The drafts: the model's own greedy continuation of p through the Off bridge, one token at a time, up to `want`
// rows in all (fewer at a dead end or a failed finish).
bool ExactChain(const Env& e, int32_t schema_index, int p0, uint32_t want, std::vector<int32_t>* chain, std::string* why) {
	int32_t tok = -1;
	SslmGpuStatus st = ExactStart(e, schema_index, p0, &tok);
	if (st != SSLM_OK || tok < 0) return *why = "the prefix's finish -> " + std::to_string(static_cast<int>(st)), false;
	chain->assign(1, tok);
	while (chain->size() < want) {
		st = sslm_gpu_seq_embed_token(e.ctx, e.seq_x, tok);
		if (st == SSLM_OK) st = sslm_decode_step_gpu(e.ctx, e.seq_x, e.adapter_x, e.L * e.dpl);
		int32_t ready = 0;
		SslmGpuStatus out = SSLM_OK;
		if (st == SSLM_OK && (st = sslm_gpu_ready(e.ctx, e.seq_x, 1, &ready, &out)) == SSLM_OK) st = ready ? out : SSLM_DEVICE_LOST;
		if (st == SSLM_OK) st = SslmGpuSeqFinishTokenForG5Bridge(e.ctx, e.seq_x, &tok);
		if (st != SSLM_OK || tok < 0) break;
		chain->push_back(tok);
	}
	return true;
}

// One Exact run at M rows: the timed EXACT_DRAFT chunk [chain[0] .. chain[M-1]] to its close, all M rows agreeing.
bool RunExact(const Env& e, int32_t schema_index, uint32_t m, int p0, const std::vector<int32_t>& chain, XSample* s,
              std::string* why) {
	char buf[200];
	int32_t p = -1;
	SslmGpuStatus st = ExactStart(e, schema_index, p0, &p);
	if (st != SSLM_OK || p != chain[0]) {
		std::snprintf(buf, sizeof buf, "exact start -> %d, p %d (the chain's %d)", static_cast<int>(st), p, chain[0]);
		return *why = buf, false;
	}
	SslmGpuRowsSpec spec{};
	spec.struct_size = sizeof spec;
	spec.kind = SSLM_GPU_ROWS_EXACT_DRAFT;
	spec.tokens = chain.data();
	spec.count = static_cast<int32_t>(m);
	const superslm_gpu::GpuLogitsStageTiming before = superslm_gpu::LastLogitsStageTiming();
	int32_t admitted = 0, committed = 0, open = 1, produced = -1;
	const auto t0 = Clock::now();
	st = sslm_gpu_seq_embed_rows(e.ctx, e.seq_x, &spec, &admitted);
	if (st == SSLM_OK) st = sslm_decode_step_gpu(e.ctx, e.seq_x, e.adapter_x, e.L * e.dpl);
	int32_t ready = 0;
	SslmGpuStatus out = SSLM_OK;
	if (st == SSLM_OK && (st = sslm_gpu_ready(e.ctx, e.seq_x, 1, &ready, &out)) == SSLM_OK) st = ready ? out : SSLM_DEVICE_LOST;
	s->x_us = UsSince(t0);
	const double layers_gpu_us = superslm_gpu::LastCallTiming().gpu_busy_ms * 1000.0;
	const superslm_gpu::GpuLogitsStageTiming after = superslm_gpu::LastLogitsStageTiming();
	sslm_gpu_seq_last_rows_committed(e.seq_x, &committed);
	sslm_gpu_seq_rows_open(e.seq_x, &open);
	const SslmGpuStatus pst = sslm_gpu_seq_rows_produced(e.seq_x, &produced);
	if (st != SSLM_OK || admitted != static_cast<int32_t>(m) || committed != static_cast<int32_t>(m) || open != 0 ||
	    pst != SSLM_OK) {
		std::snprintf(buf, sizeof buf, "exact chunk -> %d, admitted %d, committed %d, open %d, rows_produced %d",
		              static_cast<int>(st), admitted, committed, open, static_cast<int>(pst));
		return *why = buf, false;
	}
	if (after.closes != before.closes + 1 || after.host_rows != m) {
		std::snprintf(buf, sizeof buf, "the close published %llu host timings for %u rows (want 1 for %u)",
		              static_cast<unsigned long long>(after.closes - before.closes), after.host_rows, m);
		return *why = buf, false;
	}
	s->host_us = after.host_finish_ms * 1000.0;
	s->stage_us = after.stages == before.stages + 1 ? after.gpu_ms * 1000.0 : 0.0;  // no stage: a host finish path
	// Unbound, the close's finish takes the device head: one timed stage of M rows for M >= 2; at M = 1 Off's own
	// one-row call and no stage (ruling 37: g(1) is the table's).
	const bool stage_ok = m == 1 ? after.stages == before.stages
	                             : after.stages == before.stages + 1 && after.rows == m && s->stage_us > 0;
	if (schema_index < 0 && !stage_ok) {
		std::snprintf(buf, sizeof buf, "unbound: %llu logits stages of %u rows, %.1f us (want %s)",
		              static_cast<unsigned long long>(after.stages - before.stages), after.rows, s->stage_us,
		              m == 1 ? "none at M = 1" : "1 of M rows, timed");
		return *why = buf, false;
	}
	s->x_gpu_us = layers_gpu_us + s->stage_us;
	return true;
}

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
	s->c_gpu_us = superslm_gpu::LastCallTiming().gpu_busy_ms * 1000.0;  // the close's submission (published at ready)
	sslm_gpu_seq_last_rows_committed(e.seq, &committed);
	sslm_gpu_seq_rows_open(e.seq, &open);
	if (st == SSLM_OK && !(s->c_gpu_us > 0)) {
		std::snprintf(buf, sizeof buf, "no GPU timing for the chunk (gpu_busy_ms %.4f): no timestamp support?", s->c_gpu_us / 1000.0);
		return *why = buf, false;
	}
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
	std::string schema_name;
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
		else if (a.rfind("--schema=", 0) == 0) schema_name = a.substr(9);
		else if (a == "--quick") runs = 3, p0 = 3;  // the Wine smoke's reduced form
		else {
			std::printf("usage: multirow_cost_bench --model=PATH [--shaders=DIR] [--adapter=PATH] [--device-head] "
			            "[--schema=NAME] [--runs=30] [--warmup=1] [--p0=16] [--out=PATH] [--json=PATH] [--csv=PATH [--csv-append]] "
			            "[--model-name=NAME]\n");
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
	// The Exact rows' mapping: the head on the device (EXACT_DRAFT's admission requires it).
	if (device_head) {
		e.model_x = e.model;
		e.adapter_x = e.adapter;
	} else {
		GpuResidencyConfig res_x{};
		res_x.flags = SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE;
		if (sslm_gpu_model_map(e.ctx, &view, res_x, &e.model_x) != SSLM_OK ||
		    (!adapter_path.empty() && sslm_gpu_adapter_map(e.ctx, e.model_x, &adapter_view, &e.adapter_x) != SSLM_OK)) {
			std::printf("FAIL the device-head mapping for the Exact rows\n");
			return 1;
		}
	}
	if (sslm_gpu_seq_create(e.ctx, e.model_x, p0 + 3 * SSLM_GPU_MAX_ROWS, &e.seq_x) != SSLM_OK) {
		std::printf("FAIL the Exact sequence\n");
		return 1;
	}
	int32_t schema_index = -1;
	if (!schema_name.empty()) {
		schema_index = SslmGpuSchemaLookupForG5Bridge(e.model_x, schema_name.c_str());
		if (schema_index < 0) {
			std::printf("FAIL --schema=%s: the model has no such schema\n", schema_name.c_str());
			return 1;
		}
	}
	std::printf("multirow_cost_bench: L = %u, %u dispatches per layer, p0 = %d, %d warm-up + %d timed runs per M%s%s%s%s\n",
	            e.L, e.dpl, p0, warmup, runs, device_head ? ", device head" : ", host head",
	            e.adapter ? ", adapter bound" : "", schema_name.empty() ? "" : ", Exact rows also bound to ",
	            schema_name.c_str());
	// The drafts, unbound and (with --schema) bound: the model's own continuation, so every row of a run agrees.
	std::vector<int32_t> chain_u, chain_b;
	{
		std::string why;
		if (!ExactChain(e, -1, p0, SSLM_GPU_MAX_ROWS, &chain_u, &why) ||
		    (schema_index >= 0 && !ExactChain(e, schema_index, p0, SSLM_GPU_MAX_ROWS, &chain_b, &why))) {
			std::printf("FAIL the drafts: %s\n", why.c_str());
			return 1;
		}
		if (chain_u.size() < SSLM_GPU_MAX_ROWS || (schema_index >= 0 && chain_b.size() < SSLM_GPU_MAX_ROWS)) {
			std::printf("FAIL the drafts: %zu unbound, %zu bound tokens before a dead end (want %d each; pick a longer "
			            "schema)\n", chain_u.size(), chain_b.size(), SSLM_GPU_MAX_ROWS);
			return 1;
		}
	}

	constexpr uint32_t kMaxM = SSLM_GPU_MAX_ROWS;
	std::vector<Sample> samples[kMaxM + 1];
	std::vector<XSample> xsamples[kMaxM + 1], bsamples[kMaxM + 1];  // the Exact rows, unbound and bound
	Stats c_stats[kMaxM + 1], cw_stats[kMaxM + 1];                   // C gpu, C wall
	Stats x_stats[kMaxM + 1], xw_stats[kMaxM + 1], s_stats[kMaxM + 1], hf_stats[kMaxM + 1], hfb_stats[kMaxM + 1];
	for (uint32_t m = 1; m <= kMaxM; ++m) {
		for (int i = 0; i < warmup + runs; ++i) {
			Sample s;
			XSample x, b;
			std::string why;
			if (!Run(e, m, p0, &s, &why)) {
				std::printf("FAIL M = %u run %d: %s\n", m, i, why.c_str());
				return 1;
			}
			if (!RunExact(e, -1, m, p0, chain_u, &x, &why) ||
			    (schema_index >= 0 && !RunExact(e, schema_index, m, p0, chain_b, &b, &why))) {
				std::printf("FAIL M = %u run %d (Exact): %s\n", m, i, why.c_str());
				return 1;
			}
			if (i >= warmup) {
				samples[m].push_back(s);
				xsamples[m].push_back(x);
				if (schema_index >= 0) bsamples[m].push_back(b);
			}
		}
		std::vector<double> c, cw, xg, xw, st, hf, hfb;
		for (const Sample& s : samples[m]) {
			c.push_back(s.c_gpu_us);
			cw.push_back(s.c_us);
		}
		for (const XSample& x : xsamples[m]) {
			xg.push_back(x.x_gpu_us);
			xw.push_back(x.x_us);
			st.push_back(x.stage_us);
			hf.push_back(x.host_us);
		}
		for (const XSample& b : bsamples[m]) hfb.push_back(b.host_us);
		c_stats[m] = Summarize(c);
		cw_stats[m] = Summarize(cw);
		x_stats[m] = Summarize(xg);
		xw_stats[m] = Summarize(xw);
		s_stats[m] = Summarize(st);
		hf_stats[m] = Summarize(hf);
		hfb_stats[m] = Summarize(hfb);
	}

	int64_t r[kMaxM + 1] = {0};
	int64_t g[kMaxM + 1] = {0};
	std::printf("\n  M | C(M) GPU us median    p95      max | wall median    p95      max | r(M) | Exact X GPU max | S max | g(M) |"
	            " host finish max%s\n", schema_index >= 0 ? " (unbound / bound)" : "");
	for (uint32_t m = 1; m <= kMaxM; ++m) {
		r[m] = m == 1 ? 1000 : CeilPerMille(c_stats[m].max, c_stats[1].max);
		g[m] = m == 1 ? superslm_gpu::kGpuRowsLogitsRatioPerMille[0]  // no stage at M = 1: the table's (ruling 37)
		              : CeilPerMille(std::ceil(s_stats[m].max), std::ceil(c_stats[1].max));
		std::printf("  %u | %17.0f %8.0f %8.0f | %11.0f %8.0f %8.0f | %4lld | %15.0f | %5.0f | %4lld | %8.0f", m,
		            c_stats[m].median, c_stats[m].p95, c_stats[m].max, cw_stats[m].median, cw_stats[m].p95, cw_stats[m].max,
		            static_cast<long long>(r[m]), x_stats[m].max, s_stats[m].max, static_cast<long long>(g[m]), hf_stats[m].max);
		if (schema_index >= 0) std::printf(" / %.0f", hfb_stats[m].max);
		std::printf("\n");
	}
	std::printf("  port rule C(4) / (4 x C(1)) on GPU time at the median: %.3f (MR0: <= 0.6); g(M) = ceil(S(M) x 1000 / C(1)),"
	            " M >= 2; g(1) from the table (no stage at M = 1, ruling 37)\n",
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
			std::fprintf(f, "%s\n {\"M\": %u, \"r_permille\": %lld, \"c_gpu_us\": [", m == 1 ? "" : ",", m,
			             static_cast<long long>(r[m]));
			for (size_t i = 0; i < samples[m].size(); ++i) std::fprintf(f, "%s%.1f", i ? ", " : "", samples[m][i].c_gpu_us);
			std::fprintf(f, "], \"c_wall_us\": [");
			for (size_t i = 0; i < samples[m].size(); ++i) std::fprintf(f, "%s%.1f", i ? ", " : "", samples[m][i].c_us);
			std::fprintf(f, "], \"finish_us\": [");
			for (size_t i = 0; i < samples[m].size(); ++i) std::fprintf(f, "%s%.1f", i ? ", " : "", samples[m][i].f_us);
			std::fprintf(f, "], \"c\": {\"median\": %.1f, \"p95\": %.1f, \"max\": %.1f}", c_stats[m].median, c_stats[m].p95,
			             c_stats[m].max);
			std::fprintf(f, ", \"c_wall\": {\"median\": %.1f, \"p95\": %.1f, \"max\": %.1f}", cw_stats[m].median, cw_stats[m].p95,
			             cw_stats[m].max);
			std::fprintf(f, ", \"exact_gpu_us\": [");
			for (size_t i = 0; i < xsamples[m].size(); ++i) std::fprintf(f, "%s%.1f", i ? ", " : "", xsamples[m][i].x_gpu_us);
			std::fprintf(f, "], \"exact_wall_us\": [");
			for (size_t i = 0; i < xsamples[m].size(); ++i) std::fprintf(f, "%s%.1f", i ? ", " : "", xsamples[m][i].x_us);
			std::fprintf(f, "], \"logits_stage_gpu_us\": [");
			for (size_t i = 0; i < xsamples[m].size(); ++i) std::fprintf(f, "%s%.1f", i ? ", " : "", xsamples[m][i].stage_us);
			std::fprintf(f, "], \"host_finish_us\": [");
			for (size_t i = 0; i < xsamples[m].size(); ++i) std::fprintf(f, "%s%.1f", i ? ", " : "", xsamples[m][i].host_us);
			std::fprintf(f, "], \"host_finish_bound_us\": [");
			for (size_t i = 0; i < bsamples[m].size(); ++i) std::fprintf(f, "%s%.1f", i ? ", " : "", bsamples[m][i].host_us);
			std::fprintf(f, "], \"g_permille\": %lld}", static_cast<long long>(g[m]));
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
		if (!csv_append)
			std::fprintf(f, "model,M,logits,runs,warmup,median_us,p95_us,max_us,clock,wall_median_us,wall_p95_us,wall_max_us,"
			                "logits_stage_gpu_us,host_finish_us\n");
		auto up = [](double v) { return static_cast<long long>(CeilPerMille(v, 1000.0)); };  // whole us, rounded up
		for (uint32_t m = 1; m <= kMaxM; ++m) {
			for (int lg = 0; lg < 2; ++lg) {
				const Stats& st = lg ? x_stats[m] : c_stats[m];  // logits 1: the Exact chunk's GPU time, layers + stage
				const Stats& wall = lg ? xw_stats[m] : cw_stats[m];
				std::fprintf(f, "%s,%u,%d,%d,%d,%lld,%lld,%lld,gpu,%lld,%lld,%lld,", model_name.c_str(), m, lg, runs, warmup,
				             up(st.median), up(st.p95), up(st.max), up(wall.median), up(wall.p95), up(wall.max));
				if (lg) {
					const Stats& hf = schema_index >= 0 ? hfb_stats[m] : hf_stats[m];  // the certified (bound) shape when named
					if (m == 1) std::fprintf(f, ",%lld\n", up(hf.max));  // no stage at M = 1: g(1) from the table (ruling 37)
					else std::fprintf(f, "%lld,%lld\n", up(s_stats[m].max), up(hf.max));
				} else {
					std::fprintf(f, ",\n");
				}
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
		             "// C(1) = %.0f us of GPU time at its max (superslm_gpu::LastCallTiming().gpu_busy_ms; wall time is not used).\n"
		             "// r(M) = ceil(1000 x maxC(M) / maxC(1)), measured. g(M) = ceil(1000 x S(M) / maxC(1)) for M >= 2, S the widened\n"
		             "// logits stage's GPU time at its max (rulings 18a, 27); g(1) is carried from the table this bench was built\n"
		             "// against (no stage runs at M = 1, ruling 37).\n"
		             "// Host-only data for the plugin's integer chooser (MR6); index M - 1. The layout is include/superslm/gpu_rows_cost_table.h's.\n"
		             "#ifndef SUPERSLM_GPU_ROWS_COST_TABLE_H\n#define SUPERSLM_GPU_ROWS_COST_TABLE_H\n\n#include <cstdint>\n\n"
		             "namespace superslm_gpu {\n\n"
		             "constexpr bool kGpuRowsCostTableMeasured = true;     // r(M), on this card\n"
		             "constexpr bool kGpuRowsLogitsRatioMeasured = true;   // g(M), M >= 2, on this card; g(1) from the table\n"
		             "constexpr uint32_t kGpuRowCostUs = 6200;  // GpuRowCostMs's default (6.2 ms) in us: C(1), the M = 1 entry\n\n",
		             model_name.c_str(), e.L, p0, warmup, runs, device_head ? "device head" : "host head", c_stats[1].max);
		const char* names[2] = {"kGpuRowsCostRatioPerMille", "kGpuRowsLogitsRatioPerMille"};
		for (int t = 0; t < 2; ++t) {
			std::fprintf(f, "constexpr uint32_t %s[8] = {\n", names[t]);
			for (uint32_t m = 1; m <= kMaxM; ++m)
				std::fprintf(f, "    %lld,  // M = %u%s\n",
				             static_cast<long long>(t == 0 ? r[m] : g[m]), m, t == 1 && m == 1 ? ": from the table" : "");
			std::fprintf(f, "};\n\n");
		}
		std::fprintf(f, "}  // namespace superslm_gpu\n\n#endif  // SUPERSLM_GPU_ROWS_COST_TABLE_H\n");
		std::fclose(f);
		std::printf("wrote %s\n", out_path.c_str());
	}
	sslm_gpu_seq_release(e.ctx, e.seq);
	sslm_gpu_seq_release(e.ctx, e.seq_x);
	std::printf("PASS multirow_cost_bench: %u M values x %d runs\n", kMaxM, runs);
	return 0;
}
