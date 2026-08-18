// t2132_diag_layer_bisect.cpp -- DISPOSABLE (T-2132, Brunel). Bounded diagnosis: localizes the
// >18-step CPU/GPU numeric divergence recorded in
// Claude/Brunel/t2132-g5-build-2026-08-16.md ("Session 2 ... the next step is a per-layer/
// per-kernel hidden-state comparison at step 19"). Includes neither sslm_abi.h nor gpu_1p0.h --
// see t2132_diag_layer_bisect_shared.h.
//
// Usage: t2132_diag_layer_bisect <model.sslm> <prompt> <schema_name> <target_step>
// Exits 0 if CPU/GPU hidden state matches through every layer at target_step, 1 if a divergent
// layer is found (reported to stdout), 2 on setup failure.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "t2132_diag_layer_bisect_shared.h"

namespace {

bool ReadFile(const char* path, std::vector<uint8_t>* out) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) return false;
	const std::streamsize sz = f.tellg();
	if (sz < 0) return false;
	f.seekg(0, std::ios::beg);
	out->resize(static_cast<size_t>(sz));
	if (sz > 0 && !f.read(reinterpret_cast<char*>(out->data()), sz)) return false;
	return true;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 5) {
		std::fprintf(stderr, "usage: %s <model.sslm> <prompt> <schema_name> <target_step>\n", argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string prompt = argv[2];
	const std::string schema_name = argv[3];
	const int32_t target_step = std::atoi(argv[4]);

	std::vector<uint8_t> bytes;
	if (!ReadFile(model_path.c_str(), &bytes)) {
		std::fprintf(stderr, "could not read %s\n", model_path.c_str());
		return 2;
	}

	std::fprintf(stderr, "=== T-2132 DIAG: layer-bisect at step=%d ===\n", target_step);
	std::fprintf(stderr, "artifact: %s\nprompt: \"%s\"\nschema: %s\n\n", model_path.c_str(),
	             prompt.c_str(), schema_name.c_str());

	std::vector<int32_t> prompt_tokens;
	DiagStepResult cpu = RunCpuDiagStep(bytes.data(), bytes.size(), prompt, schema_name, target_step,
	                                     &prompt_tokens);
	if (!cpu.setup_ok) {
		std::fprintf(stderr, "CPU DIAG FAILED: %s\n", cpu.last_error.c_str());
		return 2;
	}
	std::fprintf(stderr, "[CPU] %zu layer snapshots captured, produced_token=%d\n",
	             cpu.layer_snapshots.size(), cpu.produced_token);

	DiagStepResult gpu = RunGpuDiagStep(bytes.data(), bytes.size(), prompt_tokens, schema_name,
	                                     target_step);
	if (!gpu.setup_ok) {
		std::fprintf(stderr, "GPU DIAG FAILED: %s\n", gpu.last_error.c_str());
		return 2;
	}
	std::fprintf(stderr, "[GPU] %zu layer snapshots captured, produced_token=%d\n",
	             gpu.layer_snapshots.size(), gpu.produced_token);

	// ---- Prefix pass: did the RAW hidden state (not just the argmax-chosen token) already
	// differ at some step BEFORE the bisected one? ----
	std::fprintf(stderr, "\n--- prefix steps 0..%d (raw hidden state, pre-final_norm) ---\n",
	             target_step - 1);
	int first_divergent_prefix_step = -1;
	if (cpu.prefix_step_snapshots.size() != gpu.prefix_step_snapshots.size()) {
		std::fprintf(stderr, "STRUCTURAL MISMATCH: CPU captured %zu prefix-step snapshots, GPU captured %zu.\n",
		             cpu.prefix_step_snapshots.size(), gpu.prefix_step_snapshots.size());
	} else {
		for (size_t i = 0; i < cpu.prefix_step_snapshots.size(); ++i) {
			const DiagLayerSnapshot& c = cpu.prefix_step_snapshots[i];
			const DiagLayerSnapshot& g = gpu.prefix_step_snapshots[i];
			const bool scale_match = (c.scale_m == g.scale_m) && (c.scale_e == g.scale_e);
			const bool size_match = c.hidden_codes.size() == g.hidden_codes.size();
			int first_diff_idx = -1;
			int max_abs_delta = 0;
			if (size_match) {
				for (size_t k = 0; k < c.hidden_codes.size(); ++k) {
					const int delta = static_cast<int>(c.hidden_codes[k]) - static_cast<int>(g.hidden_codes[k]);
					if (delta != 0) {
						if (first_diff_idx < 0) first_diff_idx = static_cast<int>(k);
						const int ad = delta < 0 ? -delta : delta;
						if (ad > max_abs_delta) max_abs_delta = ad;
					}
				}
			}
			const bool codes_match = size_match && first_diff_idx < 0;
			const bool is_prefill_sentinel = c.layer_index_after == 0xFFFFFFFFu;
			if ((!scale_match || !codes_match) && first_divergent_prefix_step < 0) {
				first_divergent_prefix_step = is_prefill_sentinel ? -1 : static_cast<int>(c.layer_index_after);
			}
			const char* ctx_note = c.context_length == g.context_length ? "SAME" : "DIFF";
			if (is_prefill_sentinel) {
				std::fprintf(stderr,
				             "step=PREFILL(pre-decode) | ctx_len cpu=%lld gpu=%lld %s | cpu_scale=(%d,%d) gpu_scale=(%d,%d) %s | codes %s | first-diff idx=%d | max|delta|=%d\n",
				             (long long)c.context_length, (long long)g.context_length, ctx_note, c.scale_m,
				             c.scale_e, g.scale_m, g.scale_e, scale_match ? "SAME" : "DIFF",
				             codes_match ? "SAME" : "DIFF", first_diff_idx, max_abs_delta);
			} else {
				std::fprintf(stderr,
				             "step=%u | ctx_len cpu=%lld gpu=%lld %s | cpu_scale=(%d,%d) gpu_scale=(%d,%d) %s | codes %s | first-diff idx=%d | max|delta|=%d\n",
				             c.layer_index_after, (long long)c.context_length, (long long)g.context_length,
				             ctx_note, c.scale_m, c.scale_e, g.scale_m, g.scale_e,
				             scale_match ? "SAME" : "DIFF", codes_match ? "SAME" : "DIFF", first_diff_idx,
				             max_abs_delta);
			}
		}
		if (first_divergent_prefix_step < 0) {
			std::fprintf(stderr,
			             "\nEvery prefix step's raw hidden state matches byte-for-byte -- the "
			             "divergence genuinely first appears inside the bisected step's own layer "
			             "loop, not from drift carried in from an earlier step.\n");
		} else {
			std::fprintf(stderr,
			             "\n*** Raw hidden state ALREADY differs at step=%d, %d step(s) before the "
			             "bisected step -- this is the TRUE first-divergence step; the bisected "
			             "step's own layer-0 divergence is downstream of THIS. ***\n",
			             first_divergent_prefix_step, target_step - first_divergent_prefix_step);
		}
	}

	if (cpu.layer_snapshots.size() != gpu.layer_snapshots.size()) {
		std::fprintf(stderr,
		             "STRUCTURAL MISMATCH: CPU captured %zu layer snapshots, GPU captured %zu -- "
		             "the two paths did not advance one layer per call the same number of times.\n",
		             cpu.layer_snapshots.size(), gpu.layer_snapshots.size());
		return 2;
	}

	std::fprintf(stderr, "\nlayer | scale(m,e) match | hidden_codes | first-diff idx | max|delta|\n");
	int first_divergent_layer = -1;
	for (size_t i = 0; i < cpu.layer_snapshots.size(); ++i) {
		const DiagLayerSnapshot& c = cpu.layer_snapshots[i];
		const DiagLayerSnapshot& g = gpu.layer_snapshots[i];
		const bool scale_match = (c.scale_m == g.scale_m) && (c.scale_e == g.scale_e);
		const bool size_match = c.hidden_codes.size() == g.hidden_codes.size();
		int first_diff_idx = -1;
		int max_abs_delta = 0;
		if (size_match) {
			for (size_t k = 0; k < c.hidden_codes.size(); ++k) {
				const int delta = static_cast<int>(c.hidden_codes[k]) - static_cast<int>(g.hidden_codes[k]);
				if (delta != 0) {
					if (first_diff_idx < 0) first_diff_idx = static_cast<int>(k);
					const int ad = delta < 0 ? -delta : delta;
					if (ad > max_abs_delta) max_abs_delta = ad;
				}
			}
		}
		const bool codes_match = size_match && first_diff_idx < 0;
		std::fprintf(stderr, "L%02u(CPUidx%zu/GPUidx%zu) | cpu=(%d,%d) gpu=(%d,%d) %s | %s | %d | %d\n",
		             c.layer_index_after, i, i, c.scale_m, c.scale_e, g.scale_m, g.scale_e,
		             scale_match ? "SAME" : "DIFF", codes_match ? "SAME" : "DIFF", first_diff_idx,
		             max_abs_delta);
		if ((!scale_match || !codes_match) && first_divergent_layer < 0) {
			first_divergent_layer = static_cast<int>(c.layer_index_after);
		}
	}

	std::fprintf(stderr, "\nCPU produced_token=%d, GPU produced_token=%d, %s\n", cpu.produced_token,
	             gpu.produced_token, cpu.produced_token == gpu.produced_token ? "MATCH" : "DIFFER");

	if (first_divergent_layer < 0) {
		std::fprintf(stderr,
		             "\n=== RESULT: hidden state matches every layer through this step; produced "
		             "tokens %s. Divergence (if any) is NOT in the layer loop at this step. ===\n",
		             cpu.produced_token == gpu.produced_token ? "also match" : "still DIFFER -- see finish-token stage");
		return cpu.produced_token == gpu.produced_token ? 0 : 1;
	}

	std::fprintf(stderr,
	             "\n=== RESULT: FIRST DIVERGENT LAYER at step=%d is layer_index_after=%d (the %d%s "
	             "of %zu layers run this step) ===\n",
	             target_step, first_divergent_layer, first_divergent_layer,
	             first_divergent_layer == 1 ? "st" : (first_divergent_layer == 2 ? "nd" : "th"),
	             cpu.layer_snapshots.size());
	return 1;
}
