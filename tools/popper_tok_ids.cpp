// Popper T-1802 probe: print the token ids the .sslm tokenizer produces for a
// prompt read from a UTF-8 file, so the int8 arm's tokenization can be compared
// against the witness arm's AutoTokenizer ids.
#include "superslm/artifact.h"
#include "superslm/tokenizer.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace superslm;
int main(int argc, char** argv) {
	if (argc < 3) { std::printf("usage: popper_tok_ids <tokenizer.sslm> <prompt_file>\n"); return 2; }
	SslmArtifact art; SslmError e;
	if (SslmArtifact::OpenFromFile(argv[1], art, &e) != SslmStatus::Ok) {
		std::printf("artifact load failed: %s\n", e.message.c_str()); return 1; }
	TokenizerView tok; std::string err;
	if (!TokenizerView::Open(art, tok, &err)) { std::printf("tokenizer open failed: %s\n", err.c_str()); return 1; }
	std::ifstream f(argv[2], std::ios::binary);
	std::stringstream ss; ss << f.rdbuf();
	std::string prompt = ss.str();
	std::vector<int32_t> ids = tok.Encode(prompt);
	std::printf("n_tokens: %zu\nids:", ids.size());
	for (size_t i = 0; i < ids.size(); ++i) std::printf(" %d", ids[i]);
	std::printf("\n");
	return 0;
}
