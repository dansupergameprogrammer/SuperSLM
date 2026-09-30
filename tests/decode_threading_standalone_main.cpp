// Decode-threading plan (rev 1.2) §8 4.5 (F9): the scalar tier's run of the decode-threading cells.
// The scalar-forced library has no full-suite binary, so this main runs the two decode-threading
// translation units alone, against superslm_scalar_forced_seams. Exit status 0 iff every check passed.
#include <cstdio>

void RunDecodeThreadingCells(int& checks, int& failures);
void RunDecodeThreadingD1Cells(int& checks, int& failures);

int main() {
#if !defined(SUPERSLM_FORCE_SCALAR_MATMUL)
#error "decode_threading_standalone_main.cpp is the scalar-forced tier's runner"
#endif
	int checks = 0, failures = 0;
	RunDecodeThreadingCells(checks, failures);
	RunDecodeThreadingD1Cells(checks, failures);
	std::printf("superslm_decode_threading_scalar_forced (SUPERSLM_FORCE_SCALAR_MATMUL): %d checks, %d failures\n",
	            checks, failures);
	return failures == 0 && checks > 0 ? 0 : 1;
}
