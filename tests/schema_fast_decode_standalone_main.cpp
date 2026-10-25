// Schema fast-decode plan (rev 1.2.1) §9, slices M1 and J1: the scalar tier's run of the red suite.
// The scalar-forced library has no full-suite binary, so this main runs tests/test_schema_fast_decode.cpp
// alone, against superslm_scalar_forced_seams (the decode_threading_standalone_main.cpp precedent).
// Exit status 0 iff every check passed.
#include <cstdio>

void RunSchemaFastDecodeCells(int& checks, int& failures);

int main() {
#if !defined(SUPERSLM_FORCE_SCALAR_MATMUL)
#error "schema_fast_decode_standalone_main.cpp is the scalar-forced tier's runner"
#endif
	int checks = 0, failures = 0;
	RunSchemaFastDecodeCells(checks, failures);
	std::printf("superslm_schema_fast_decode_scalar_forced (SUPERSLM_FORCE_SCALAR_MATMUL): %d checks, %d failures\n",
	            checks, failures);
	return failures == 0 && checks > 0 ? 0 : 1;
}
