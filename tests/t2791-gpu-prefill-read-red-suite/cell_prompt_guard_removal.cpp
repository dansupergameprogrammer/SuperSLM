// T-2801 (Curie) -- plan Sec3.4 row 5's Q4-1e: a genuine device removal against the prompt prefill,
// in its own process (it removes the process-wide harness device, which nothing afterwards in the
// process can use). Built against the SHIPPING configuration of the GPU library: it needs no seam.
//
// WHAT THIS CELL IS, AND IS NOT. It is a device-loss liveness and cleanliness cell: after
// ID3D12Device5::RemoveDevice, a prompt prefill returns SSLM_DEVICE_LOST and never
// SSLM_SEQUENCE_REJECTED, nothing hangs, and the handles still release, unmap and destroy with
// SSLM_OK. It CANNOT distinguish plan Sec3.6's classifications: on a removed device the fault surfaces
// at the upload, before any readback, so no sticky tag is ever decoded and the guard flag is never
// set. Every Sec3.6 mutant, and v1.5.0 itself, passes it. The device-alive conjunct's killer is
// cell_prompt_guard_status's Q4-1d (the simulated removed-device query on a guard-tripping prefill).
//
// Cells, on G-an (--gan; its hash is pinned) with sequence cap 64:
//   E0 must-accept: before the removal, a passing prefill [0, 1] returns SSLM_OK
//   E1 RemoveDevice; GetDeviceRemovedReason() is then not S_OK (SETUP: no ID3D12Device5 fails as SETUP)
//   E2 reset, then the passing prefill [0, 1] -> SSLM_DEVICE_LOST
//   E3 reset, then the guard-tripping prefill [0, 1, 3] -> SSLM_DEVICE_LOST (a removed device is never
//      reported as a guard refusal, even on ids that trip one on a live device)
//   E4 release, unmap and context destroy -> SSLM_OK
// A watchdog fails the cell (exit 7) if any phase does not return within 60 s.
//
// Links and runs at v1.5.0 (no verb, no seam): GREEN there, as it must be on every Sec3.6 build.
//
// Run: cell_prompt_guard_removal.exe --gan=PATH
#include "fixture_common.h"
#include "d3d12_harness.h"

#include <atomic>
#include <chrono>
#include <thread>

namespace {

constexpr const char* kGanSha256 = "cf48079cd3b50eb053c8f8cd57d4feb0f102d8ec9622aaf86300a15daf96e76b";

std::atomic<int> g_phase{0};
const char* kPhaseName[] = {"setup", "must-accept prefill", "remove", "passing prefill", "tripping prefill",
                            "teardown", "done"};
constexpr int kDone = 6;

void Watchdog(int seconds) {
	std::thread([seconds] {
		int last = -1;
		auto t0 = std::chrono::steady_clock::now();
		for (;;) {
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
			const int p = g_phase.load();
			if (p != last) {
				last = p;
				t0 = std::chrono::steady_clock::now();
			}
			if (p == kDone) return;
			if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(seconds)) {
				std::printf("FAIL: HANG -- phase '%s' did not return within %d s\n", kPhaseName[p], seconds);
				std::fflush(stdout);
				std::_Exit(7);
			}
		}
	}).detach();
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_gan_path.empty()) {
		SKIP_MSG("Q4-1e needs --gan=PATH");
		return FinishSuite("cell_prompt_guard_removal");
	}
	std::vector<uint8_t> b;
	const bool hash_ok = ReadFileBytes(g_gan_path, &b) && Sha256Hex(b.data(), b.size()) == kGanSha256;
	CHECK_MSG(hash_ok, "SETUP: --gan is not make_gan_fixture.py's output (want sha256 %s)", kGanSha256);
	if (!hash_ok) return FinishSuite("cell_prompt_guard_removal");

	Watchdog(60);
	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
		std::printf("FATAL: sslm_gpu_context_create failed\n");
		return 2;
	}
	GpuModelFixture g;
	if (!g.Open(g_gan_path, ctx)) {
		CHECK_MSG(false, "SETUP: G-an did not load and map");
		sslm_gpu_context_destroy(ctx);
		return FinishSuite("cell_prompt_guard_removal");
	}
	SslmGpuSequenceHandle* s = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, g.model, 64, &s) == SSLM_OK && s);
	const std::vector<int32_t> passing = {0, 1};
	const std::vector<int32_t> tripping = {0, 1, 3};  // cell_prompt_guard_status's Q4-1a ids

	g_phase = 1;
	CHECK(sslm_gpu_seq_reset(ctx, s) == SSLM_OK);
	const SslmGpuStatus e0 = Prefill(g, s, passing);
	std::printf("    E0 before removal, %s: %s\n", "[0,1]", StatusName(e0));
	CHECK_MSG(e0 == SSLM_OK, "E0 (must-accept): the passing prefill on a live device returned %s", StatusName(e0));

	g_phase = 2;
	auto& dev = superslm_gpu::harness::GetDevice();
	Microsoft::WRL::ComPtr<ID3D12Device5> d5;
	const bool have5 = SUCCEEDED(dev.dev.As(&d5));
	CHECK_MSG(have5, "SETUP: the harness device exposes no ID3D12Device5; Q4-1e cannot construct a removal");
	if (have5) {
		d5->RemoveDevice();
		const HRESULT rr = dev.dev->GetDeviceRemovedReason();
		std::printf("    E1 RemoveDevice -> GetDeviceRemovedReason 0x%08lx\n", static_cast<unsigned long>(rr));
		CHECK_MSG(rr != S_OK, "SETUP E1: the device does not report removed after RemoveDevice");

		g_phase = 3;
		CHECK(sslm_gpu_seq_reset(ctx, s) == SSLM_OK);
		const SslmGpuStatus e2 = Prefill(g, s, passing);
		std::printf("    E2 after removal, passing [0,1]: %s\n", StatusName(e2));
		CHECK_MSG(e2 == SSLM_DEVICE_LOST, "E2: a passing prefill on a removed device returned %s, want SSLM_DEVICE_LOST",
		          StatusName(e2));

		g_phase = 4;
		CHECK(sslm_gpu_seq_reset(ctx, s) == SSLM_OK);
		const SslmGpuStatus e3 = Prefill(g, s, tripping);
		std::printf("    E3 after removal, tripping [0,1,3]: %s\n", StatusName(e3));
		CHECK_MSG(e3 == SSLM_DEVICE_LOST,
		          "E3: a guard-tripping prefill on a removed device returned %s, want SSLM_DEVICE_LOST", StatusName(e3));
	}

	g_phase = 5;
	CHECK_MSG(sslm_gpu_seq_release(ctx, s) == SSLM_OK, "E4: release after removal");
	CHECK_MSG(sslm_gpu_model_unmap(ctx, g.model) == SSLM_OK, "E4: unmap after removal");
	g.model = nullptr;
	CHECK_MSG(sslm_gpu_context_destroy(ctx) == SSLM_OK, "E4: context destroy after removal");
	g_phase = kDone;
	return FinishSuite("cell_prompt_guard_removal");
}
