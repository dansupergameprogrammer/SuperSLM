// t1788_tok_count.cpp -- throwaway tool, NOT part of the ticket's own record: prints the engine
// tokenizer's own token count for a prompt string, so candidate held-out prompts can be checked
// for engine/HF tokenization agreement BEFORE spending a full probe run on them.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/tokenizer.h"
#include "sslm_marshal.h"

using namespace superslm;
using superslm_marshal::ReadFile;

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: %s <tokenizer.sslm> \"<prompt>\"\n", argv[0]);
		return 2;
	}
	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(argv[1], tok_bytes)) {
		std::fprintf(stderr, "FAILED: tokenizer_file_read\n");
		return 1;
	}
	SslmArtifact tok_artifact;
	SslmError tok_open_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact, &tok_open_err) !=
	    SslmStatus::Ok) {
		std::fprintf(stderr, "FAILED: tokenizer_artifact_open\n");
		return 1;
	}
	TokenizerView tokenizer;
	std::string tok_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tok_err)) {
		std::fprintf(stderr, "FAILED: tokenizer_view_open: %s\n", tok_err.c_str());
		return 1;
	}
	const std::vector<int32_t> ids = tokenizer.Encode(argv[2]);
	std::printf("n_tokens=%zu\n", ids.size());
	for (int32_t id : ids) std::printf("%d\n", id);
	return 0;
}
