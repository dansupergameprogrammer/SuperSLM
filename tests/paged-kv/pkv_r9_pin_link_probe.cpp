// Paged-KV plan rev 16.21 cell 4.12, the link half of "absent from production": this unit references
// the test-only pin and (rev 16.23, D-SLM8437) the geometry seam, and is linked against the production
// `superslm` library only (superslm_pkv_r9_pin_link_probe in tests/paged-kv/paged_kv.cmake). The link must
// FAIL with both unresolved; tests/paged-kv/r9_pin_reach.py builds it and grades the failure. A link that
// succeeds means production carries a test seam.

#include "superslm/sslm_abi.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

extern "C" sslm_status KvPagesTestPinPagePositions(sslm_model model, int64_t page_positions);
extern "C" sslm_status KvPagesTestPoolGeometry(const void* artifact, size_t size, uint32_t block_count, void* out);

int main() {
	const sslm_status st = KvPagesTestPinPagePositions(nullptr, 16);
	const sslm_status g = KvPagesTestPoolGeometry(nullptr, 0, 1, nullptr);
	std::printf("pin and seam linked against production: %d %d\n", static_cast<int>(st), static_cast<int>(g));
	return 0;
}
