// T-2700 converter-only compiled post-RoPE K capture.
//
// Usage:
//   t2700_fused_k_capture <artifact.sslm> <453-token-prefix.txt> <report.tsv>
//                         <suffix.txt> [suffix.txt ...]
//   t2700_fused_k_capture <artifact.sslm> <453-token-prefix.txt> <report.tsv>
//                         --suffix-list <one-suffix-path-per-line.txt>
//
// This is deliberately the slice-2 ABI snapshot/clone route.  Its one private
// binder only attaches a caller-owned observation sink before the first
// sequence exists; normal ABI forwards never use that binder or a sink.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "superslm/sha256.h"
#include "superslm/sslm_abi.h"

// Private converter-tool seam defined in sslm_abi.cpp.  It is intentionally
// absent from the public ABI header and refuses once any sequence is live.
extern "C" sslm_status sslm_t2700_set_fused_k_capture_sink(
    sslm_model model, superslm::FusedKCaptureSink* sink);

namespace {

struct AlignedBytes {
	std::vector<uint8_t> storage;
	void* data = nullptr;
	explicit AlignedBytes(size_t size) : storage(size + SSLM_ABI_ALIGNMENT_BYTES - 1) {
		uintptr_t address = reinterpret_cast<uintptr_t>(storage.data());
		address = (address + SSLM_ABI_ALIGNMENT_BYTES - 1) & ~(uintptr_t(SSLM_ABI_ALIGNMENT_BYTES - 1));
		data = reinterpret_cast<void*>(address);
	}
};

struct Peak {
	uint64_t raw_abs = 0;
	double real = 0.0;
	uint64_t landing_saturation_count = 0;
	bool seen = false;
};

struct Capture {
	uint32_t layers = 0;
	uint32_t heads = 0;
	uint32_t channels = 0;
	std::vector<Peak> peaks;
	std::vector<superslm::CarriedScale> scales;
	std::vector<bool> scale_seen;
	uint64_t callback_count = 0;
	uint64_t landing_saturation_count = 0;

	Capture(uint32_t in_layers, uint32_t in_heads, uint32_t in_channels)
	    : layers(in_layers), heads(in_heads), channels(in_channels),
	      peaks(static_cast<size_t>(in_layers) * in_heads * in_channels),
	      scales(in_layers), scale_seen(in_layers, false) {}

	static uint64_t AbsI64(int64_t value) {
		return value < 0 ? static_cast<uint64_t>(-(value + 1)) + 1 : static_cast<uint64_t>(value);
	}

	static void Observe(void* raw_context, uint32_t layer, size_t head, size_t channel,
	                    int64_t /*wide*/, int64_t rotated, superslm::CarriedScale scale,
	                    int64_t landing_raw) {
		auto* self = static_cast<Capture*>(raw_context);
		if (layer >= self->layers || head >= self->heads || channel >= self->channels) return;
		if (!self->scale_seen[layer]) {
			self->scales[layer] = scale;
			self->scale_seen[layer] = true;
		} else if (self->scales[layer].m != scale.m || self->scales[layer].e != scale.e) {
			return;  // impossible for one marshaled layer; retain the first authoritative pair.
		}
		const uint64_t raw_abs = AbsI64(rotated);
		// `rotated` is the Q30-rounded output of RopeApplyPairWide.  The sink
		// supplies its already-adjusted source scale, so this is exactly the
		// post-RoPE real quantity the float path observes.
		const double real = std::ldexp(static_cast<double>(raw_abs) * static_cast<double>(scale.m),
		                               static_cast<int>(scale.e));
		Peak& peak = self->peaks[(static_cast<size_t>(layer) * self->heads + head) * self->channels + channel];
		peak.raw_abs = std::max(peak.raw_abs, raw_abs);
		peak.real = std::max(peak.real, real);
		peak.seen = true;
		++self->callback_count;
		if (landing_raw < -127 || landing_raw > 127) {
			++self->landing_saturation_count;
			++peak.landing_saturation_count;
		}
	}
};

bool ReadTokens(const char* path, std::vector<int32_t>* out) {
	std::ifstream input(path);
	int64_t value = 0;
	while (input >> value) {
		if (value < INT32_MIN || value > INT32_MAX) return false;
		out->push_back(static_cast<int32_t>(value));
	}
	return input.eof() && !out->empty();
}

bool ReadSuffixList(const char* path, std::vector<std::string>* out) {
	std::ifstream input(path);
	std::string line;
	while (std::getline(input, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		if (!line.empty()) out->push_back(line);
	}
	return input.eof() && !out->empty();
}

bool Save(sslm_seq sequence, std::vector<uint8_t>* out) {
	size_t size = 0;
	if (sslm_seq_save(sequence, nullptr, &size) != SSLM_BUFFER_TOO_SMALL || size == 0) return false;
	out->assign(size, 0);
	return sslm_seq_save(sequence, out->data(), &size) == SSLM_OK && size == out->size();
}

std::string Digest(const std::vector<uint8_t>& bytes) {
	uint8_t digest[32] = {};
	superslm::Sha256Hash(bytes.data(), bytes.size(), digest);
	return superslm::ToHex(digest);
}

bool Prefill(sslm_model model, sslm_seq sequence, const std::vector<int32_t>& tokens,
	             int32_t budget, sslm_workspace workspace) {
	int32_t consumed = 0;
	return sslm_prefill(model, sequence, tokens.data(), static_cast<int32_t>(tokens.size()), budget,
	                    SSLM_SPAN_PROMPT, workspace, &consumed) == SSLM_OK &&
	       consumed == static_cast<int32_t>(tokens.size());
}

bool WriteReport(const char* path, const Capture& capture, const std::string& snapshot_digest,
	                 size_t snapshot_bytes, double prefix_seconds, double suffix_seconds, size_t suffixes) {
	std::ofstream output(path, std::ios::binary | std::ios::trunc);
	if (!output) return false;
	output << "T2700_FUSED_K_CAPTURE_V1\n";
	output << "summary\tprefix_tokens\t453\n";
	output << "summary\tsnapshot_sha256\t" << snapshot_digest << "\n";
	output << "summary\tsnapshot_bytes\t" << snapshot_bytes << "\n";
	output << "summary\tprefix_seconds\t" << prefix_seconds << "\n";
	output << "summary\tsuffixes\t" << suffixes << "\n";
	output << "summary\tsuffix_seconds\t" << suffix_seconds << "\n";
	output << "summary\tcallback_count\t" << capture.callback_count << "\n";
	output << "summary\tlanding_saturation_count\t" << capture.landing_saturation_count << "\n";
	output << "layer\thead\tchannel\traw_abs_peak\twide_scale_m\twide_scale_e\treal_peak\tlanding_saturation_count\n";
	for (uint32_t layer = 0; layer < capture.layers; ++layer) {
		if (!capture.scale_seen[layer]) return false;
		for (uint32_t head = 0; head < capture.heads; ++head) {
			for (uint32_t channel = 0; channel < capture.channels; ++channel) {
				const Peak& peak = capture.peaks[(static_cast<size_t>(layer) * capture.heads + head) * capture.channels + channel];
				if (!peak.seen || !std::isfinite(peak.real)) return false;
				output << layer << '\t' << head << '\t' << channel << '\t' << peak.raw_abs << '\t'
				       << capture.scales[layer].m << '\t' << capture.scales[layer].e << '\t'
				       << std::hexfloat << peak.real << std::defaultfloat << '\t'
				       << peak.landing_saturation_count << '\n';
			}
		}
	}
	return static_cast<bool>(output);
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 5) {
		std::fprintf(stderr, "usage: t2700_fused_k_capture <artifact.sslm> <453-token-prefix.txt> <report.tsv> <suffix.txt> [suffix.txt ...]\n"
		                     "   or: t2700_fused_k_capture <artifact.sslm> <453-token-prefix.txt> <report.tsv> --suffix-list <paths.txt>\n");
		return 2;
	}
	std::ifstream artifact_file(argv[1], std::ios::binary);
	std::vector<uint8_t> artifact((std::istreambuf_iterator<char>(artifact_file)), {});
	if (artifact.empty()) { std::fprintf(stderr, "artifact read failed\n"); return 2; }
	std::vector<int32_t> prefix;
	if (!ReadTokens(argv[2], &prefix) || prefix.size() != 453) {
		std::fprintf(stderr, "prefix must be exactly 453 signed int32 token IDs\n"); return 2;
	}
	std::vector<std::string> suffix_paths;
	if (argc == 6 && std::strcmp(argv[4], "--suffix-list") == 0) {
		if (!ReadSuffixList(argv[5], &suffix_paths)) {
			std::fprintf(stderr, "invalid suffix path list: %s\n", argv[5]); return 2;
		}
	} else {
		for (int index = 4; index < argc; ++index) suffix_paths.emplace_back(argv[index]);
	}
	std::vector<std::vector<int32_t>> suffixes(suffix_paths.size());
	for (size_t index = 0; index < suffix_paths.size(); ++index) {
		if (!ReadTokens(suffix_paths[index].c_str(), &suffixes[index])) {
			std::fprintf(stderr, "invalid suffix token file: %s\n", suffix_paths[index].c_str()); return 2;
		}
	}

	superslm::SslmModelView view;
	std::string diagnostic;
	if (superslm::SslmModel::Load(artifact.data(), artifact.size(), view, &diagnostic) != superslm::SslmModelStatus::Ok) {
		std::fprintf(stderr, "artifact rejected: %s\n", diagnostic.c_str()); return 1;
	}
	sslm_model model = nullptr;
	if (sslm_model_map(artifact.data(), artifact.size(), &model) != SSLM_OK) {
		std::fprintf(stderr, "ABI model map rejected artifact\n"); return 1;
	}
	Capture capture(view.config.num_hidden_layers, view.config.num_key_value_heads, view.config.head_dim);
	superslm::FusedKCaptureSink sink{&capture, &Capture::Observe};
	if (sslm_t2700_set_fused_k_capture_sink(model, &sink) != SSLM_OK) {
		std::fprintf(stderr, "capture sink bind rejected\n"); sslm_model_unmap(model); return 1;
	}
	const sslm_config config{1, static_cast<int32_t>(prefix.size()),
	                         static_cast<int32_t>(view.config.num_hidden_layers), 0};
	const size_t workspace_size = sslm_workspace_size(model, &config);
	const size_t pool_size = sslm_kv_pool_overhead_size(model, 2) + 2 * sslm_kv_block_size(model);
	AlignedBytes workspace_memory(workspace_size);
	AlignedBytes pool_memory(pool_size);
	sslm_workspace workspace = nullptr;
	sslm_kv_pool pool = nullptr;
	if (workspace_size == 0 || pool_size == 0 ||
	    sslm_workspace_create(model, &config, workspace_memory.data, workspace_size, &workspace) != SSLM_OK ||
	    sslm_kv_pool_create(model, pool_memory.data, pool_size, 2, &pool) != SSLM_OK) {
		std::fprintf(stderr, "ABI workspace/pool construction failed\n");
		if (workspace) sslm_workspace_destroy(workspace); sslm_model_unmap(model); return 1;
	}
	sslm_seq parent = nullptr;
	if (sslm_seq_create(model, &pool, &parent) != SSLM_OK) {
		std::fprintf(stderr, "parent sequence creation failed\n");
		sslm_kv_pool_destroy(pool); sslm_workspace_destroy(workspace); sslm_model_unmap(model); return 1;
	}
	const auto prefix_start = std::chrono::steady_clock::now();
	if (!Prefill(model, parent, prefix, config.max_chunk_budget, workspace)) {
		std::fprintf(stderr, "prefix prefill failed\n"); return 1;
	}
	std::vector<uint8_t> snapshot;
	if (!Save(parent, &snapshot)) { std::fprintf(stderr, "prefix snapshot failed\n"); return 1; }
	const double prefix_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - prefix_start).count();
	const std::string snapshot_digest = Digest(snapshot);
	double suffix_seconds = 0.0;
	for (const auto& suffix : suffixes) {
		const auto start = std::chrono::steady_clock::now();
		sslm_seq clone = nullptr;
		if (sslm_seq_restore(model, &pool, snapshot.data(), snapshot.size(), &clone) != SSLM_OK) {
			std::fprintf(stderr, "snapshot restore failed\n"); return 1;
		}
		// `t2693_prefix_clone` owns the byte-identity/non-mutating-clone
		// obligation.  Re-serializing both 1.8 GB sequence blocks for every
		// conversion suffix would time a diagnostic copy loop rather than the
		// production capture, so this driver performs the real restore/prefill
		// only.  The clone's independent pool block is released immediately.
		if (!Prefill(model, clone, suffix, config.max_chunk_budget, workspace)) {
			std::fprintf(stderr, "clone suffix prefill failed\n"); return 1;
		}
		sslm_seq_release(clone);
		suffix_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	}
	const bool wrote = WriteReport(argv[3], capture, snapshot_digest, snapshot.size(), prefix_seconds,
	                               suffix_seconds, suffixes.size());
	std::printf("prefix_tokens=453 snapshot_sha256=%s prefix_seconds=%.6f suffixes=%zu suffix_seconds=%.6f callback_count=%llu landing_saturation_count=%llu\n",
	            snapshot_digest.c_str(), prefix_seconds, suffixes.size(), suffix_seconds,
	            static_cast<unsigned long long>(capture.callback_count),
	            static_cast<unsigned long long>(capture.landing_saturation_count));
	sslm_seq_release(parent);
	sslm_kv_pool_destroy(pool);
	sslm_workspace_destroy(workspace);
	sslm_model_unmap(model);
	if (!wrote) { std::fprintf(stderr, "capture report write failed\n"); return 1; }
	return 0;
}
