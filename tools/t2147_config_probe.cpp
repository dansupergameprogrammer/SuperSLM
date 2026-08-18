// t2147_config_probe.cpp -- quick one-off: print the real artifact's config fields needed for
// the D-SLM3488 bandwidth arithmetic (T-2147 re-measurement).
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "superslm/model.h"

int main(int argc, char** argv) {
	if (argc < 2) { std::fprintf(stderr, "usage: %s <artifact.sslm>\n", argv[0]); return 1; }
	std::ifstream f(argv[1], std::ios::binary);
	f.seekg(0, std::ios::end);
	const long sz = f.tellg();
	f.seekg(0, std::ios::beg);
	std::vector<uint8_t> bytes(static_cast<size_t>(sz));
	f.read(reinterpret_cast<char*>(bytes.data()), sz);
	superslm::SslmModelView view;
	std::string err;
	const superslm::SslmModelStatus st = superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err);
	if (st != superslm::SslmModelStatus::Ok) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }
	const auto& c = view.config;
	std::printf("hidden_size=%u intermediate_size=%u num_hidden_layers=%u head_dim=%u num_key_value_heads=%u vocab_size=%u context_cap=%lld\n",
	            c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.head_dim,
	            c.num_key_value_heads, c.vocab_size, (long long)c.context_cap);
	return 0;
}
