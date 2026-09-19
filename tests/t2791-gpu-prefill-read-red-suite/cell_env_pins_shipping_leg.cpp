// T-2791 (Curie) -- plan Sec3.2 and Sec3.4 row 7, the EXECUTED leg of "Environment reads removed":
// "The installed superslm_gpu library then contains no environment read that changes a GPU
// output." Built against the engine's GPU sources compiled with NO test or bench definitions --
// the shipping configuration, exactly what the installed superslm_gpu target compiles
// (build_red_suite.bat, variant gpu_plain).
//
//   E1 A prefill's completed residual and scale, and the token the decode wrapper then produces,
//      are identical with SSLM_B5_ASYNC_SWAP_SRV_REBIND=1, with SSLM_B5_ASYNC_DROP_UAV_REBIND=1,
//      and with neither set.
// The engine reads each variable once per process (a function-local static in the common submit
// body, src/gpu/superslm_gpu.cpp), so each environment runs in its own child process: this
// executable re-launches itself with --child=OUTFILE after setting (or clearing) the variables.
//
// ORACLE: the clean-environment child of the same binary; no reading comes from the variable-set
// runs themselves. VITALITY: at v1.5.0 the swap pin binds the RoPE cos/sin tables into each other's
// slots (plan Sec2.5), so the swap child's bytes differ and this cell is RED AT RUN -- the red
// reading is itself the proof this cell can fail. The source-level half of row 7 (no ungated read
// in any installed translation unit, and the pins still wired for the harness that needs them) is
// tests/ci/test_t2791_gpu_fault_pin_env_gate.py.
//
// This cell reads the hidden state through the bench bridge, not the verb, so it links at v1.5.0.
//
// Run: cell_env_pins_shipping_leg.exe [--synthetic=PATH] [--qwen3=PATH]   (the first supplied is used)
#include <process.h>

#include "fixture_common.h"

namespace {

int Child(const std::string& out_path) {
	const std::string path = !g_synthetic_path.empty() ? g_synthetic_path : g_qwen3_path;
	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) return 3;
	GpuModelFixture fx;
	if (!fx.Open(path, ctx)) return 3;
	SslmGpuSequenceHandle* s = nullptr;
	if (sslm_gpu_seq_create(ctx, fx.model, 64, &s) != SSLM_OK || !s) return 3;
	std::string rec;
	const std::vector<int32_t> row = fx.Run(8, fx.TokensP()[0]);  // positions 0..7: RoPE is not the identity
	if (sslm_gpu_seq_reset(ctx, s) != SSLM_OK || Prefill(fx, s, row) != SSLM_OK) return 3;
	const SeqState st = CaptureState(s);
	rec += Sha256Hex(st.codes.data(), st.codes.size());
	rec += " m=" + std::to_string(st.m) + " e=" + std::to_string(st.e);
	int32_t tok = -1;
	const SslmGpuStatus ds = SslmGpuSeqDecodeStepForG5Bridge(ctx, s, fx.OtherToken(), fx.one_layer_budget, &tok);
	int32_t tok2 = -1;
	const SslmGpuStatus ds2 = SslmGpuSeqDecodeStepForG5Bridge(ctx, s, tok, fx.one_layer_budget, &tok2);
	rec += " decode=" + std::string(StatusName(ds)) + "," + StatusName(ds2) + " tokens=" + std::to_string(tok) + "," +
	       std::to_string(tok2);
	sslm_gpu_seq_release(ctx, s);
	fx.Close();
	sslm_gpu_context_destroy(ctx);
	std::FILE* f = std::fopen(out_path.c_str(), "wb");
	if (!f) return 3;
	std::fputs(rec.c_str(), f);
	std::fclose(f);
	return 0;
}

std::string RunChild(const char* self, const std::string& out, const char* drop, const char* swap) {
	_putenv_s("SSLM_B5_ASYNC_DROP_UAV_REBIND", drop);
	_putenv_s("SSLM_B5_ASYNC_SWAP_SRV_REBIND", swap);
	const std::string child_flag = "--child=" + out;
	const std::string syn = "--synthetic=" + g_synthetic_path;
	const std::string q3 = "--qwen3=" + g_qwen3_path;
	const std::string qself = std::string("\"") + self + "\"";
	const std::string qchild = "\"" + child_flag + "\"", qsyn = "\"" + syn + "\"", qq3 = "\"" + q3 + "\"";
	const intptr_t rc = _spawnl(_P_WAIT, self, qself.c_str(), qchild.c_str(), qsyn.c_str(), qq3.c_str(), nullptr);
	_putenv_s("SSLM_B5_ASYNC_DROP_UAV_REBIND", "");
	_putenv_s("SSLM_B5_ASYNC_SWAP_SRV_REBIND", "");
	std::vector<uint8_t> b;
	if (rc != 0 || !ReadFileBytes(out, &b)) return "child-failed(rc=" + std::to_string(rc) + ")";
	return std::string(b.begin(), b.end());
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	for (int i = 1; i < argc; ++i) {
		if (std::strncmp(argv[i], "--child=", 8) == 0) return Child(argv[i] + 8);
	}
	if (g_synthetic_path.empty() && g_qwen3_path.empty()) {
		SKIP_MSG("E1 needs --synthetic=PATH or --qwen3=PATH");
		return FinishSuite("cell_env_pins_shipping_leg");
	}
	const std::string base = std::string(argv[0]) + ".env-leg";
	const std::string clean = RunChild(argv[0], base + ".clean.txt", "", "");
	const std::string drop = RunChild(argv[0], base + ".drop.txt", "1", "");
	const std::string swap = RunChild(argv[0], base + ".swap.txt", "", "1");
	std::printf("    clean: %s\n    drop : %s\n    swap : %s\n", clean.c_str(), drop.c_str(), swap.c_str());
	CHECK_MSG(clean.rfind("child-failed", 0) != 0, "E1 SETUP: the clean-environment child failed");
	CHECK_MSG(drop == clean, "E1: SSLM_B5_ASYNC_DROP_UAV_REBIND=1 changes the shipping build's GPU output");
	CHECK_MSG(swap == clean, "E1: SSLM_B5_ASYNC_SWAP_SRV_REBIND=1 changes the shipping build's GPU output");
	return FinishSuite("cell_env_pins_shipping_leg");
}
