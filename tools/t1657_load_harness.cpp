// Throwaway harness for T-1657/T-1663: loads the real qwen2.5-1.5b-instruct.sslm
// artifact through SslmModel::Load (the same entry point every runtime consumer
// uses) and reports the status and diagnostic. Not part of the build.bat/CMake
// build graph -- compiled and run directly for this session's own verification.
//
// Usage: t1657_load_harness <artifact.sslm>
#include <cstdio>
#include <fstream>
#include <string>

#include "superslm/model.h"

using namespace superslm;

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: t1657_load_harness <artifact.sslm>\n");
		return 2;
	}
	const char* path = argv[1];

	std::ifstream f(path, std::ios::binary);
	if (!f) {
		std::fprintf(stderr, "could not open %s\n", path);
		return 2;
	}
	std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	std::fprintf(stdout, "artifact: %s (%zu bytes)\n", path, bytes.size());

	SslmModelView view;
	std::string err;
	const SslmModelStatus status =
	    SslmModel::Load(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), view, &err);

	std::fprintf(stdout, "Load status = %d (%s)\n", static_cast<int>(status), SslmModelStatusName(status));
	std::fprintf(stdout, "Load diagnostic = \"%s\"\n", err.c_str());

	if (status == SslmModelStatus::Ok) {
		std::fprintf(stdout, "view.has_config = %d\n", view.has_config ? 1 : 0);
		std::fprintf(stdout, "view.has_sigmoid_lut = %d\n", view.has_sigmoid_lut ? 1 : 0);
		std::fprintf(stdout, "view.has_weights = %d\n", view.has_weights ? 1 : 0);
		std::fprintf(stdout, "view.has_biases = %d\n", view.has_biases ? 1 : 0);
		std::fprintf(stdout, "view.has_rope_tables = %d\n", view.has_rope_tables ? 1 : 0);
		std::fprintf(stdout, "view.has_weight_scales = %d\n", view.has_weight_scales ? 1 : 0);
		std::fprintf(stdout, "view.has_composition_constants = %d\n",
		             view.has_composition_constants ? 1 : 0);
		std::fprintf(stdout, "view.has_kv_landing_scales = %d\n", view.has_kv_landing_scales ? 1 : 0);
		std::fprintf(stdout, "view.has_kv_landing_reciprocals = %d\n",
		             view.has_kv_landing_reciprocals ? 1 : 0);
		std::fprintf(stdout, "view.has_calibration_band = %d\n", view.has_calibration_band ? 1 : 0);
		std::fprintf(stdout, "view.has_tokenizer = %d\n", view.has_tokenizer ? 1 : 0);
		if (view.has_config) {
			std::fprintf(stdout, "config.hidden_size = %d\n", view.config.hidden_size);
			std::fprintf(stdout, "config.num_attention_heads = %d\n", view.config.num_attention_heads);
			std::fprintf(stdout, "config.num_key_value_heads = %d\n", view.config.num_key_value_heads);
			std::fprintf(stdout, "config.num_hidden_layers = %d\n", view.config.num_hidden_layers);
			std::fprintf(stdout, "config.vocab_size = %d\n", view.config.vocab_size);
		}
		if (view.has_biases) {
			std::fprintf(stdout, "biases.Tensors().size() = %zu\n", view.biases.Tensors().size());
			for (const SslmTensorView& t : view.biases.Tensors()) {
				std::fprintf(stdout, "  bias tensor \"%.*s\": elem_count=%llu\n",
				             static_cast<int>(t.name.size()), t.name.data(),
				             static_cast<unsigned long long>(t.elem_count));
			}
		}
	}

	return status == SslmModelStatus::Ok ? 0 : 1;
}
