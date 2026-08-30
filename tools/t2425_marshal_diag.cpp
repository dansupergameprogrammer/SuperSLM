// t2425_marshal_diag.cpp -- disposable spike probe (T-2425). Calls layer_marshal.h's own
// MarshalLayer DIRECTLY against layer 0, printing its diagnostic string on failure -- the ABI's
// own sslm_model_map returns only a 3-status-value shape with no diagnostic out-parameter
// (t2423_model_map_harness.cpp's own header comment), so this is the tool that actually shows
// WHICH check inside MarshalLayer fired, rather than trusting the generic SSLM_ARTIFACT_REJECTED
// this design's own asymmetric-presence check and every other required-tensor check both map to.
//
// Usage: t2425_marshal_diag <artifact.sslm>
#include <cstdio>
#include <fstream>
#include <string>

#include "superslm/layer_marshal.h"
#include "superslm/model.h"

using namespace superslm;
using namespace superslm_marshal;

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: t2425_marshal_diag <artifact.sslm>\n");
		return 2;
	}
	std::ifstream f(argv[1], std::ios::binary);
	if (!f) {
		std::fprintf(stderr, "could not open %s\n", argv[1]);
		return 2;
	}
	std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

	SslmModelView view;
	std::string load_err;
	SslmModelStatus lst = SslmModel::Load(reinterpret_cast<const uint8_t*>(bytes.data()),
	                                       bytes.size(), view, &load_err);
	if (lst != SslmModelStatus::Ok) {
		std::fprintf(stdout, "SslmModel::Load REJECTED: %s\n", load_err.c_str());
		return 1;
	}
	std::fprintf(stdout, "SslmModel::Load OK\n");

	int fail_count = 0;
	for (uint32_t l = 0; l < view.config.num_hidden_layers; ++l) {
		LayerBacking backing;
		LayerWeights out{};
		std::string err;
		const bool ok = MarshalLayer(view, l, view.config.num_attention_heads,
		                             view.config.num_key_value_heads, backing, out, &err);
		if (!ok) {
			std::fprintf(stdout, "MarshalLayer(layer=%u) FAILED: %s\n", l, err.c_str());
			++fail_count;
		}
	}
	if (fail_count == 0) {
		std::fprintf(stdout, "MarshalLayer: OK for all %u layers\n", view.config.num_hidden_layers);
		return 0;
	}
	std::fprintf(stdout, "MarshalLayer: %d of %u layers failed\n", fail_count,
	             view.config.num_hidden_layers);
	return 1;
}
