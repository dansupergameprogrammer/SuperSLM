// T-2814 (Curie) -- X2's loader: links sslm_api_consumer.dll, which links sslm_engine.dll, so the engine
// DLL, a consumer DLL and a loader run in one process (plan Sec3.7 item 6, dimension 8). Exits 0 only on
// ALL=PASS.
#include <cstdio>

extern "C" __declspec(dllimport) int x2_run();

int main() {
	const int fail = x2_run();
	std::printf("%s\n", fail == 0 ? "ALL=PASS" : "ALL=FAIL");
	return fail == 0 ? 0 : 1;
}
