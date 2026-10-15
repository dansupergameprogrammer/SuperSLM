// Paged-KV plan rev 16.22 cell 9.3, the pkv_odd restore side: the OLD reader. Built against a frozen
// release library (v1.11.0; tests/paged-kv/r9_old_reader.ps1), never against the tree under work. It
// restores a legacy blob the paged build saved (cell 9.3/T2's output), decodes the tokens the release's
// own record continues with, and compares them. Uses only the release's public C ABI.
//
// Usage: r9_old_reader FIXTURE.sslm BLOB EXPECTED_TOKENS (comma-separated)
// Exit 0 when the restore succeeds and every token matches; 1 otherwise, with the reason on stdout.

#include "superslm/sslm_abi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <new>
#include <string>
#include <vector>

static bool ReadAll(const char* path, std::vector<unsigned char>* out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	out->assign(std::istreambuf_iterator<char>(f), {});
	return !out->empty();
}

int main(int argc, char** argv) {
	if (argc != 4) {
		std::printf("usage: r9_old_reader FIXTURE BLOB EXPECTED_TOKENS\n");
		return 2;
	}
	std::vector<unsigned char> model_bytes, blob;
	if (!ReadAll(argv[1], &model_bytes) || !ReadAll(argv[2], &blob)) {
		std::printf("INFRASTRUCTURE: cannot read %s or %s\n", argv[1], argv[2]);
		return 3;
	}
	std::vector<int> want;
	for (const char* p = argv[3]; *p;) {
		want.push_back(std::atoi(p));
		const char* c = std::strchr(p, ',');
		if (!c) break;
		p = c + 1;
	}
	void* mbuf = ::operator new(model_bytes.size(), std::align_val_t(64));
	std::memcpy(mbuf, model_bytes.data(), model_bytes.size());
	sslm_model model = nullptr;
	if (sslm_model_map(mbuf, model_bytes.size(), &model) != SSLM_OK) {
		std::printf("INFRASTRUCTURE: the release cannot map %s\n", argv[1]);
		return 3;
	}
	const size_t size = sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, 1);
	const size_t rounded = (size + 63) / 64 * 64;
	void* pbuf = ::operator new(rounded, std::align_val_t(64));
	sslm_kv_pool pool = nullptr;
	if (sslm_kv_pool_create(model, pbuf, rounded, 1, &pool) != SSLM_OK) {
		std::printf("INFRASTRUCTURE: pool create\n");
		return 3;
	}
	sslm_seq s = nullptr;
	const sslm_status rs = sslm_seq_restore(model, &pool, blob.data(), blob.size(), &s);
	if (rs != SSLM_OK) {
		std::printf("FAIL: the release refuses the paged build's blob (%zu bytes): status %d\n", blob.size(), static_cast<int>(rs));
		return 1;
	}
	std::vector<int> got;
	for (size_t i = 0; i < want.size(); ++i) {
		sslm_decode_params p{};
		p.layer_budget = 1;
		sslm_seq b[1] = {s};
		int32_t tok = -1;
		for (int guard = 0; tok < 0 && guard < 4096; ++guard)
			if (sslm_decode_step(model, b, 1, &p, nullptr, &tok) != SSLM_OK) break;
		got.push_back(tok);
	}
	const bool same = got == want;
	std::printf("%s: the release restored %zu bytes and decoded %zu tokens%s\n", same ? "PASS" : "FAIL", blob.size(),
	            got.size(), same ? ", equal to its own continuation" : ", not its own continuation");
	sslm_seq_release(s);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);
	return same ? 0 : 1;
}
