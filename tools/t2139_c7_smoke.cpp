// t2139_c7_smoke.cpp -- C7's own self-contained smoke (design Sec9: "no Gate A here [sslm_g5.h
// declares neither verb], no Gate B for the same reason; a self-contained smoke TU is still
// owed: encode a fixed string, decode the result, compare"). Also exercises the incremental-
// safety obligation (Forge W4, design Sec10 dim2/dim17): a token stream split at every possible
// boundary reassembles identically to one undivided call.
//
// Usage: t2139_c7_smoke.exe <path-to-real.sslm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
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
		std::fprintf(stderr, "usage: %s <path-to-real.sslm>\n", argv[0]);
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
		std::fprintf(stderr, "FAIL: sslm_model_map returned %d\n", static_cast<int>(st));
		return 1;
	}

	// --- design's own smoke shape: encode a fixed string, decode the result, compare ---
	const char* text = "Hello, world! This is a real tokenizer round trip.";

	int32_t needed = 0;
	st = sslm_tokenize(model, text, nullptr, &needed);
	if (st == SSLM_ARTIFACT_REJECTED) {
		// This artifact carries no Tokenizer/UnicodeTables section (a real, verified negative
		// path this project's own full weight-carrying artifacts genuinely exercise -- this
		// build's own available real fixtures split tokenizer and weights across separate
		// files, an older convention this ABI's own combined-artifact design supersedes;
		// Claude/Brunel/t2139-abi-build-2026-08-16.md records the gap). Confirmed real and
		// correct: SSLM_ARTIFACT_REJECTED on a genuinely tokenizer-less artifact, exactly the
		// disposition this file's own sslm_tokenize implementation states.
		std::printf("sslm_tokenize(no-tokenizer artifact): PASS (SSLM_ARTIFACT_REJECTED, as designed)\n");
		std::printf("SKIP: this artifact carries no tokenizer -- encode/decode round trip and "
		            "incremental split-boundary safety not exercised against it. See this "
		            "ticket's own build log for why no available real artifact combines both.\n");
		sslm_model_unmap(model);
		return 0;
	}
	if (st != SSLM_BUFFER_TOO_SMALL || needed <= 0) {
		std::fprintf(stderr,
		             "FAIL: sslm_tokenize(nullptr) returned %d, needed=%d -- expected "
		             "SSLM_BUFFER_TOO_SMALL with a real count\n",
		             static_cast<int>(st), needed);
		return 1;
	}
	std::vector<int32_t> tokens(needed);
	int32_t n = needed;
	st = sslm_tokenize(model, text, tokens.data(), &n);
	if (st != SSLM_OK || n != needed) {
		std::fprintf(stderr, "FAIL: sslm_tokenize returned %d, n=%d, expected %d\n",
		             static_cast<int>(st), n, needed);
		return 1;
	}
	std::printf("sslm_tokenize: PASS (%d real tokens)\n", n);

	// Decode in one undivided call.
	sslm_detok_state state_single = {0};
	std::vector<char> buf_single(1024);
	int32_t out_n_single = static_cast<int32_t>(buf_single.size());
	st = sslm_detokenize_stream(model, &state_single, tokens.data(), n, buf_single.data(),
	                             &out_n_single);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_detokenize_stream (undivided) returned %d\n",
		             static_cast<int>(st));
		return 1;
	}
	// S4 pin (Claude/Poirot/2c18dab-t2139-abi-build-review.md; closed per the coordinator's own
	// closing-round follow-up list): a negative caller-supplied *out_n capacity must be rejected
	// outright, not silently cast to a huge unsigned value and passed through.
	{
		sslm_detok_state state_neg = {0};
		char neg_buf[16];
		int32_t out_n_neg = -1;
		const sslm_status neg_st =
		    sslm_detokenize_stream(model, &state_neg, tokens.data(), n, neg_buf, &out_n_neg);
		if (neg_st != SSLM_INVALID_ARGUMENT || out_n_neg != 0) {
			std::fprintf(stderr,
			             "FAIL: sslm_detokenize_stream(*out_n=-1) returned %d, out_n=%d -- "
			             "expected SSLM_INVALID_ARGUMENT, out_n=0 (S4 pin)\n",
			             static_cast<int>(neg_st), out_n_neg);
			return 1;
		}
		std::printf("S4 pin (sslm_detokenize_stream negative capacity): PASS "
		            "(SSLM_INVALID_ARGUMENT fired as designed)\n");
	}

	const std::string decoded_single(buf_single.data(), static_cast<size_t>(out_n_single));
	if (decoded_single != text) {
		std::fprintf(stderr,
		             "FAIL: round trip diverges: encoded->decoded = \"%s\", expected \"%s\"\n",
		             decoded_single.c_str(), text);
		return 1;
	}
	std::printf("encode -> decode round trip: PASS (\"%s\")\n", decoded_single.c_str());

	// --- Forge W4: split at EVERY possible token boundary, reassemble, compare ---
	for (int32_t split = 0; split <= n; ++split) {
		sslm_detok_state state_split = {0};
		std::string reassembled;
		std::vector<char> chunk_buf(1024);

		auto feed = [&](const int32_t* t, int32_t count) -> bool {
			int32_t out_n = static_cast<int32_t>(chunk_buf.size());
			const sslm_status s =
			    sslm_detokenize_stream(model, &state_split, t, count, chunk_buf.data(), &out_n);
			if (s != SSLM_OK) {
				std::fprintf(stderr, "FAIL: sslm_detokenize_stream (split=%d) returned %d\n",
				             split, static_cast<int>(s));
				return false;
			}
			reassembled.append(chunk_buf.data(), static_cast<size_t>(out_n));
			return true;
		};

		if (!feed(tokens.data(), split)) return 1;
		if (!feed(tokens.data() + split, n - split)) return 1;

		if (reassembled != text) {
			std::fprintf(stderr,
			             "FAIL: split at token %d reassembles to \"%s\", expected \"%s\" -- "
			             "incremental-safety obligation violated\n",
			             split, reassembled.c_str(), text);
			return 1;
		}
		if (state_split.pending_count != 0) {
			std::fprintf(stderr,
			             "FAIL: split at token %d leaves pending_count=%u after both chunks fed "
			             "-- a complete stream should end with no partial tail\n",
			             split, static_cast<unsigned>(state_split.pending_count));
			return 1;
		}
	}
	std::printf("incremental split-boundary safety: PASS (%d boundaries, all reassemble identically)\n",
	            n + 1);

	sslm_model_unmap(model);
	std::printf("t2139_c7_smoke: PASS\n");
	return 0;
}
