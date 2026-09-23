// TE-399 (Curie) -- red cell for the property D-SLM7753 rules into the engine as v1.7.1: "the
// engine library writes nothing to stdout." TE-393 (Claude/Loki/te393-u3-strike-2026-09-23.md,
// D-SLM7752) found the one call site (src/gpu/d3d12_harness.h:335, std::wprintf, gated on
// SSLM_GPU_ADAPTER_INDEX being set) and showed it corrupts a host that reads stdout as a data
// channel. The property this cell pins is the library's stdout across a real session, not that
// one call site: a cell that greps for wprintf proves nothing about a future printf, cout, puts
// or fwrite(stdout) anywhere reachable from the shipped libraries.
//
// One process, one pipeline traversal through the public 1.0 GPU C API exactly as a host uses it
// (include/superslm/gpu_1p0.h): sslm_gpu_context_create, sslm_gpu_model_map,
// sslm_gpu_model_hidden_size, sslm_gpu_seq_create/reset, SslmGpuSeqPrefillPromptForG5Bridge (the
// prompt prefill), sslm_gpu_seq_read_prefill_final_hidden (the read), then release/unmap/destroy.
// stdout is redirected to a temp file for the whole traversal and the captured byte count is the
// oracle: a host that writes data to stdout is corrupted by ANY nonzero byte on this channel, so
// the assertion is exact-zero, not "no # adapter: substring" -- a future, differently-worded print
// must fail this cell too.
//
// Two invocations, not two device opens in one process: `harness::GetDevice()` (the process
// submission device TE-393 sec.4 names as Init() call #2) is a per-process magic static, so a
// second sslm_gpu_context_create in the SAME process after an earlier submission would not
// re-trigger its print and would under-count the defect. Each invocation of this binary is
// therefore its own fresh process, driven by run_cell.ps1 with SSLM_GPU_ADAPTER_INDEX unset for
// one run and set to a valid index for the other, comparing byte counts -- the same shape TE-393's
// own probe used (Claude/Loki/te393-u3-strike-probe/te393_stdout_probe.cpp) and for the same
// reason.
//
// RED AT f43ab15 (SuperSLM v1.7.0): the --set run captures 2 * (10 + len(adapter name) + 1) UTF-16
// bytes (two "# adapter: <name>\n" lines, d3d12_harness.h:330-336) where 0 is required. The --unset
// run captures 0 bytes already (override_requested gates the print), so it is not itself a
// standalone red cell (Curie disciplines: red before green -- a cell that already passes before
// the fix is not proof of anything); it is the control leg of the same property and is graded
// alongside the --set leg by run_cell.ps1, never committed as an independently-red cell.
//
// Test-design record: Claude/Curie/te399-slm171-stdout-red-2026-09-23.md.
//
// Usage: cell_gpu_pipeline_stdout_empty --model=PATH --mode=unset|set [--adapter-index=N]
#include <fcntl.h>
#include <io.h>
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_g5_bridge.h"
#include "superslm/model.h"

using namespace superslm;
using enum SslmGpuStatus;

namespace {

int GChecks = 0;
int GFailures = 0;

#define CHECK_MSG(cond, ...)                                               \
	do {                                                                     \
		++GChecks;                                                            \
		if (!(cond)) {                                                        \
			++GFailures;                                                        \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond);        \
			std::printf(__VA_ARGS__);                                           \
			std::printf("\n");                                                  \
		}                                                                      \
	} while (0)

int FinishSuite(const char* name) {
	const bool pass = GFailures == 0;
	std::printf("%s: checks=%d failures=%d -> %s\n", name, GChecks, GFailures, pass ? "PASS" : "FAIL");
	return pass ? 0 : 1;
}

// Sets an environment variable for the duration of one scope and restores whatever was there
// before (including "unset") on destruction. Same shape as tests/test_main.cpp's ScopedEnvVar
// (T-2116), reproduced here because this binary is its own standalone process/TU, never linked
// against test_main.cpp.
struct ScopedEnvVar {
	std::string name;
	bool had_prev = false;
	std::string prev;
	ScopedEnvVar(const char* n, const char* value) : name(n) {
		char buf[256]{};
		DWORD got = GetEnvironmentVariableA(name.c_str(), buf, sizeof(buf));
		had_prev = (got > 0 && got < sizeof(buf));
		if (had_prev) prev = buf;
		SetEnvironmentVariableA(name.c_str(), value);
	}
	~ScopedEnvVar() { SetEnvironmentVariableA(name.c_str(), had_prev ? prev.c_str() : nullptr); }
};

// RAII stdout redirect to a temp file, restored on scope exit via the saved duplicate file
// descriptor -- including if the guarded code throws. Same shape as tests/test_main.cpp's
// ScopedStdoutCapture (T-2116), reproduced here for the same standalone-TU reason as above.
struct ScopedStdoutCapture {
	int saved_fd = -1;
	std::string path;
	bool active = false;
	ScopedStdoutCapture() {
		char dir[MAX_PATH]{};
		char file[MAX_PATH]{};
		if (GetTempPathA(MAX_PATH, dir) == 0) return;
		if (GetTempFileNameA(dir, "te399c", 0, file) == 0) return;
		path = file;
		std::fflush(stdout);
		saved_fd = _dup(_fileno(stdout));
		if (saved_fd == -1) return;
		if (!std::freopen(path.c_str(), "wb", stdout)) {
			_close(saved_fd);
			saved_fd = -1;
			return;
		}
		active = true;
	}
	~ScopedStdoutCapture() {
		if (!active) return;
		std::fflush(stdout);
		_dup2(saved_fd, _fileno(stdout));
		_close(saved_fd);
		std::remove(path.c_str());
	}
	std::string ReadCaptured() {
		if (!active) return "";
		std::fflush(stdout);
		std::ifstream f(path, std::ios::binary);
		std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		return s;
	}
};

std::string HexPrefix(const std::string& s, size_t n) {
	static const char* d = "0123456789abcdef";
	std::string out;
	for (size_t i = 0; i < s.size() && i < n; ++i) {
		out.push_back(d[static_cast<uint8_t>(s[i]) >> 4]);
		out.push_back(d[static_cast<uint8_t>(s[i]) & 15]);
		out.push_back(' ');
	}
	return out;
}

int64_t Pow2Cap(size_t n) {
	int64_t c = 64;
	while (static_cast<size_t>(c) < n) c <<= 1;
	return c;
}

}  // namespace

int main(int argc, char** argv) {
	std::string model_path, mode;
	int adapter_index = 0;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--model=", 0) == 0) model_path = a.substr(8);
		else if (a.rfind("--mode=", 0) == 0) mode = a.substr(7);
		else if (a.rfind("--adapter-index=", 0) == 0) adapter_index = std::atoi(a.c_str() + 16);
	}
	if (model_path.empty() || (mode != "unset" && mode != "set")) {
		std::fprintf(stderr,
		              "usage: cell_gpu_pipeline_stdout_empty --model=PATH --mode=unset|set "
		              "[--adapter-index=N]\n");
		return 2;
	}

	// The env var is set BEFORE the first GPU call this process ever makes -- both
	// harness::Device instances this pipeline reaches (the context's own, and the process-wide
	// harness::GetDevice() submission singleton) read it at their own, once-per-process Init().
	std::unique_ptr<ScopedEnvVar> env_guard;
	if (mode == "set") {
		env_guard = std::make_unique<ScopedEnvVar>("SSLM_GPU_ADAPTER_INDEX",
		                                            std::to_string(adapter_index).c_str());
	} else {
		env_guard = std::make_unique<ScopedEnvVar>("SSLM_GPU_ADAPTER_INDEX", nullptr);
	}

	std::ifstream mf(model_path, std::ios::binary);
	std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
	if (bytes.empty()) {
		std::fprintf(stderr, "SETUP: cannot read --model=%s\n", model_path.c_str());
		return 2;
	}
	SslmModelView view;
	std::string err;
	if (SslmModel::Load(bytes.data(), bytes.size(), view, &err) != SslmModelStatus::Ok) {
		std::fprintf(stderr, "SETUP: model load failed: %s\n", err.c_str());
		return 2;
	}

	// The pipeline a host runs (§4.2's shape): context, map, encode (reset, prompt prefill,
	// read), unmap, destroy -- through the public C API only, nothing internal to the library.
	SslmGpuStatus pipeline_status = SSLM_OK;
	std::string captured;
	{
		ScopedStdoutCapture cap;

		SslmGpuContext* ctx = nullptr;
		pipeline_status = sslm_gpu_context_create(GpuContextConfig{}, &ctx);
		if (pipeline_status == SSLM_OK && ctx) {
			SslmGpuModelHandle* gm = nullptr;
			pipeline_status = sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &gm);
			if (pipeline_status == SSLM_OK && gm) {
				uint32_t hidden = 0;
				pipeline_status = sslm_gpu_model_hidden_size(gm, &hidden);
				if (pipeline_status == SSLM_OK) {
					const int32_t vocab = static_cast<int32_t>(view.config.vocab_size);
					std::vector<int32_t> ids = {vocab > 3 ? vocab / 3 : 0, vocab > 7 ? vocab / 7 : 0,
					                            vocab - 1};
					SslmGpuSequenceHandle* seq = nullptr;
					pipeline_status = sslm_gpu_seq_create(ctx, gm, Pow2Cap(ids.size()), &seq);
					if (pipeline_status == SSLM_OK && seq) {
						pipeline_status = sslm_gpu_seq_reset(ctx, seq);
						if (pipeline_status == SSLM_OK) {
							pipeline_status = SslmGpuSeqPrefillPromptForG5Bridge(
							    ctx, seq, ids.data(), static_cast<int32_t>(ids.size()), 1024u);
						}
						if (pipeline_status == SSLM_OK) {
							std::vector<int8_t> codes(hidden);
							size_t required = 0;
							int64_t m = 0, e = 0;
							pipeline_status = sslm_gpu_seq_read_prefill_final_hidden(
							    ctx, seq, codes.data(), codes.size(), &required, &m, &e);
						}
						sslm_gpu_seq_release(ctx, seq);
					}
				}
				sslm_gpu_model_unmap(ctx, gm);
			}
			sslm_gpu_context_destroy(ctx);
		}

		captured = cap.ReadCaptured();
	}  // ScopedStdoutCapture restores real stdout here -- everything below prints normally.

	if (pipeline_status == SSLM_DEVICE_LOST) {
		std::printf(
		    "cell_gpu_pipeline_stdout_empty: SKIPPED, named -- no usable D3D12 hardware adapter "
		    "on this machine (sslm_gpu_context_create returned SSLM_DEVICE_LOST)\n");
		return 3;
	}
	CHECK_MSG(pipeline_status == SSLM_OK,
	          "the pipeline itself must complete OK so the stdout capture spans a real open+use, "
	          "not an early refusal; got status %u",
	          static_cast<unsigned>(pipeline_status));
	CHECK_MSG(captured.empty(),
	          "mode=%s: the public GPU pipeline (context create, model map, prompt prefill, the "
	          "read) wrote %zu bytes to stdout; first 48 hex: %s",
	          mode.c_str(), captured.size(), HexPrefix(captured, 48).c_str());

	return FinishSuite("cell_gpu_pipeline_stdout_empty");
}
