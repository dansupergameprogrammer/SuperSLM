// Loki probe T-1656/limb-1: does the real Qwen2.5-1.5B-Instruct .sslm reach a
// loaded SslmModelView, or is it refused at ValidateBiasesDomain?
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "superslm/model.h"

using namespace superslm;

int main(int argc, char** argv) {
	const char* path = (argc > 1) ? argv[1]
	                             : "D:\\hf_cache\\superslm_artifacts\\qwen2.5-1.5b-instruct.sslm";
	FILE* f = fopen(path, "rb");
	if (!f) { std::printf("open failed: %s\n", path); return 2; }
	fseek(f, 0, SEEK_END);
	const long long n = _ftelli64(f);
	fseek(f, 0, SEEK_SET);
	std::vector<unsigned char> buf(static_cast<size_t>(n));
	const size_t got = fread(buf.data(), 1, buf.size(), f);
	fclose(f);
	std::printf("file bytes: %lld  read: %zu\n", n, got);
	if (got != buf.size()) { std::printf("short read\n"); return 2; }

	SslmModelView view;
	std::string err;
	const SslmModelStatus st = SslmModel::Load(buf.data(), buf.size(), view, &err);
	std::printf("Load status = %d (%s)\n", static_cast<int>(st), SslmModelStatusName(st));
	std::printf("err = %s\n", err.c_str());
	std::printf("has_config=%d has_weights=%d has_biases=%d has_rope=%d has_wscales=%d\n",
	            (int)view.has_config, (int)view.has_weights, (int)view.has_biases,
	            (int)view.has_rope_tables, (int)view.has_weight_scales);
	if (st == SslmModelStatus::Ok) {
		std::printf("VERDICT: artifact LOADED -- bias codes are reachable by a forward path.\n");
	} else {
		std::printf("VERDICT: artifact REFUSED at load.\n");
	}
	return 0;
}
