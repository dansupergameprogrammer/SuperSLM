// t2139_c7_unmapped_pin.cpp -- design commit 212de7742c's own same-round pin: a forced
// out-of-tokenizer-range id (in [tok_vocab, cfg_vocab), a legal decode-output padding row with
// no tokenizer entry) fed to sslm_detokenize_stream is rejected with SSLM_TOKEN_ID_UNMAPPED,
// *out_n/utf8 left untouched, state unperturbed -- distinct from SSLM_TOKEN_ID_OUT_OF_RANGE
// (>= cfg_vocab, checked at sslm_prefill/sslm_decode_step, not this call) and from a plain
// SSLM_INVALID_ARGUMENT (id >= cfg_vocab, or negative, at THIS call).
//
// Usage: t2139_c7_unmapped_pin.exe <path-to-real-padded-vocab.sslm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "superslm/sslm_abi.h"

namespace {
bool ReadFile(const char* path, std::vector<uint8_t>* out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	f.seekg(0, std::ios::end);
	const std::streamoff size = f.tellg();
	if (size < 0) return false;
	f.seekg(0, std::ios::beg);
	out->resize(static_cast<size_t>(size));
	if (size > 0) f.read(reinterpret_cast<char*>(out->data()), size);
	return static_cast<bool>(f) || f.eof();
}
}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <path-to-real-padded-vocab.sslm>\n", argv[0]);
		return 1;
	}
	std::vector<uint8_t> bytes;
	if (!ReadFile(argv[1], &bytes)) {
		std::fprintf(stderr, "FAIL: could not read %s\n", argv[1]);
		return 1;
	}

	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (st != SSLM_OK || !model) {
		std::fprintf(stderr, "FAIL: sslm_model_map returned %d -- expected SSLM_OK on a real "
		                      "padded-vocab artifact (design commit 212de7742c)\n",
		             static_cast<int>(st));
		return 1;
	}
	std::printf("sslm_model_map on the real padded-vocab artifact: PASS (SSLM_OK)\n");

	// This artifact's own real shape (confirmed by prior inspection this ticket's own build log
	// records): CFG1.vocab_size=151936, TOK1.vocab_count=151665. A forced id in
	// [151665, 151936) -- e.g. 151700 -- is a legal decode-output padding row with no tokenizer
	// entry.
	const int32_t padding_id = 151700;
	sslm_detok_state state = {0};
	char buf[256];
	// Poison the buffer and out_n so a silent pass-through would be visible.
	std::memset(buf, 0xCD, sizeof(buf));
	int32_t out_n = static_cast<int32_t>(sizeof(buf));
	const sslm_detok_state state_before = state;

	st = sslm_detokenize_stream(model, &state, &padding_id, 1, buf, &out_n);
	if (st != SSLM_TOKEN_ID_UNMAPPED) {
		std::fprintf(stderr,
		             "FAIL: sslm_detokenize_stream(padding id %d) returned %d, expected "
		             "SSLM_TOKEN_ID_UNMAPPED\n",
		             padding_id, static_cast<int>(st));
		return 1;
	}
	if (out_n != 0) {
		std::fprintf(stderr, "FAIL: SSLM_TOKEN_ID_UNMAPPED must leave *out_n untouched at 0, got %d\n",
		             out_n);
		return 1;
	}
	if (std::memcmp(&state, &state_before, sizeof(state)) != 0) {
		std::fprintf(stderr, "FAIL: SSLM_TOKEN_ID_UNMAPPED must leave state unperturbed\n");
		return 1;
	}
	std::printf("sslm_detokenize_stream(padding id %d): PASS (SSLM_TOKEN_ID_UNMAPPED, output/state "
	            "unperturbed)\n",
	            padding_id);

	// Distinctness: an id >= cfg_vocab (151936) is a plain SSLM_INVALID_ARGUMENT, not
	// SSLM_TOKEN_ID_UNMAPPED -- never a legal decode output at all.
	const int32_t out_of_range_id = 151936;
	out_n = static_cast<int32_t>(sizeof(buf));
	st = sslm_detokenize_stream(model, &state, &out_of_range_id, 1, buf, &out_n);
	if (st != SSLM_INVALID_ARGUMENT) {
		std::fprintf(stderr,
		             "FAIL: sslm_detokenize_stream(id=cfg_vocab=%d) returned %d, expected "
		             "SSLM_INVALID_ARGUMENT (distinct from SSLM_TOKEN_ID_UNMAPPED)\n",
		             out_of_range_id, static_cast<int>(st));
		return 1;
	}
	std::printf("sslm_detokenize_stream(id=cfg_vocab=%d): PASS (SSLM_INVALID_ARGUMENT, distinct "
	            "from SSLM_TOKEN_ID_UNMAPPED)\n",
	            out_of_range_id);

	// A real, in-tokenizer id still works normally (the positive path, unaffected).
	const int32_t real_id = 1;
	out_n = static_cast<int32_t>(sizeof(buf));
	st = sslm_detokenize_stream(model, &state, &real_id, 1, buf, &out_n);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_detokenize_stream(real id %d) returned %d, expected SSLM_OK\n",
		             real_id, static_cast<int>(st));
		return 1;
	}
	std::printf("sslm_detokenize_stream(real id %d): PASS (SSLM_OK, %d bytes)\n", real_id, out_n);

	sslm_model_unmap(model);
	std::printf("t2139_c7_unmapped_pin: PASS\n");
	return 0;
}
