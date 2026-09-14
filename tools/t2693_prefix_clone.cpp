// T-2693 slice-2 compiled-prefix probe.
//
// Usage: t2693_prefix_clone <artifact.sslm> <453-token-prefix.txt> <suffix.txt> [suffix.txt ...]
// Token files are whitespace-separated int32 IDs.  The tool uses the shipped ABI's SSB4
// serializer: it deep-copies the full CPU sequence state (residual/SequenceLayerState and
// its KV block), never a handle or pointer, then drives the existing sslm_prefill path.

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "superslm/model.h"
#include "superslm/sha256.h"
#include "superslm/sslm_abi.h"

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

bool ReadTokens(const char* path, std::vector<int32_t>* out) {
	std::ifstream input(path);
	int64_t value = 0;
	while (input >> value) {
		if (value < INT32_MIN || value > INT32_MAX) return false;
		out->push_back(static_cast<int32_t>(value));
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

}  // namespace

int main(int argc, char** argv) {
	if (argc < 4) {
		std::fprintf(stderr, "usage: t2693_prefix_clone <artifact.sslm> <453-token-prefix.txt> <suffix.txt> [suffix.txt ...]\n");
		return 2;
	}
	std::ifstream artifact_file(argv[1], std::ios::binary);
	std::vector<uint8_t> artifact((std::istreambuf_iterator<char>(artifact_file)), {});
	if (artifact.empty()) {
		std::fprintf(stderr, "artifact read failed\n");
		return 2;
	}
	std::vector<int32_t> prefix;
	if (!ReadTokens(argv[2], &prefix) || prefix.size() != 453) {
		std::fprintf(stderr, "prefix must be exactly 453 signed int32 token IDs\n");
		return 2;
	}
	std::vector<std::vector<int32_t>> suffixes(static_cast<size_t>(argc - 3));
	for (int index = 3; index < argc; ++index) {
		if (!ReadTokens(argv[index], &suffixes[static_cast<size_t>(index - 3)])) {
			std::fprintf(stderr, "invalid suffix token file: %s\n", argv[index]);
			return 2;
		}
	}

	superslm::SslmModelView view;
	std::string diagnostic;
	if (superslm::SslmModel::Load(artifact.data(), artifact.size(), view, &diagnostic) != superslm::SslmModelStatus::Ok) {
		std::fprintf(stderr, "artifact rejected: %s\n", diagnostic.c_str());
		return 1;
	}
	sslm_model model = nullptr;
	if (sslm_model_map(artifact.data(), artifact.size(), &model) != SSLM_OK) {
		std::fprintf(stderr, "ABI model map rejected artifact\n");
		return 1;
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
		if (workspace) sslm_workspace_destroy(workspace);
		sslm_model_unmap(model);
		return 1;
	}
	sslm_seq parent = nullptr;
	if (sslm_seq_create(model, &pool, &parent) != SSLM_OK) {
		std::fprintf(stderr, "parent sequence creation failed\n");
		sslm_kv_pool_destroy(pool); sslm_workspace_destroy(workspace); sslm_model_unmap(model);
		return 1;
	}
	const auto prefix_start = std::chrono::steady_clock::now();
	if (!Prefill(model, parent, prefix, config.max_chunk_budget, workspace)) {
		std::fprintf(stderr, "prefix prefill failed\n");
		sslm_seq_release(parent); sslm_kv_pool_destroy(pool); sslm_workspace_destroy(workspace); sslm_model_unmap(model);
		return 1;
	}
	std::vector<uint8_t> snapshot;
	if (!Save(parent, &snapshot)) {
		std::fprintf(stderr, "prefix snapshot failed\n");
		sslm_seq_release(parent); sslm_kv_pool_destroy(pool); sslm_workspace_destroy(workspace); sslm_model_unmap(model);
		return 1;
	}
	const double prefix_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - prefix_start).count();
	const std::string snapshot_digest = Digest(snapshot);
	double suffix_seconds = 0.0;
	for (const auto& suffix : suffixes) {
		const auto start = std::chrono::steady_clock::now();
		sslm_seq clone = nullptr;
		if (sslm_seq_restore(model, &pool, snapshot.data(), snapshot.size(), &clone) != SSLM_OK) {
			std::fprintf(stderr, "snapshot restore failed\n"); return 1;
		}
		std::vector<uint8_t> restored;
		if (!Save(clone, &restored) || restored != snapshot || !Prefill(model, clone, suffix, config.max_chunk_budget, workspace)) {
			std::fprintf(stderr, "snapshot byte identity or clone suffix prefill failed\n"); return 1;
		}
		sslm_seq_release(clone);
		std::vector<uint8_t> parent_after;
		if (!Save(parent, &parent_after) || parent_after != snapshot) {
			std::fprintf(stderr, "parent snapshot mutated by clone\n"); return 1;
		}
		suffix_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	}
	char report[1024] = {};
	std::snprintf(report, sizeof(report),
	              "prefix_tokens=453 snapshot_sha256=%s snapshot_bytes=%zu prefix_seconds=%.6f suffixes=%zu suffix_seconds=%.6f suffix_seconds_per_record=%.6f parent_unmodified=true\n",
	              snapshot_digest.c_str(), snapshot.size(), prefix_seconds, suffixes.size(), suffix_seconds,
	              suffix_seconds / static_cast<double>(suffixes.size()));
	std::printf("%s", report);
	char* report_path = nullptr;
	size_t report_path_length = 0;
	if (_dupenv_s(&report_path, &report_path_length, "T2693_PREFIX_CLONE_REPORT") == 0 && report_path) {
		std::ofstream output(report_path, std::ios::binary);
		output << report;
		std::free(report_path);
		if (!output) {
			std::fprintf(stderr, "failed to write T2693_PREFIX_CLONE_REPORT\n");
			return 1;
		}
	}
	sslm_seq_release(parent);
	sslm_kv_pool_destroy(pool);
	sslm_workspace_destroy(workspace);
	sslm_model_unmap(model);
	return 0;
}
