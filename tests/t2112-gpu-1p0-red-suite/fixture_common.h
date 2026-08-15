// T-2112 (Curie) -- shared CHECK harness and real-artifact loading helper for the 1.0 GPU core
// red suite. Reuses this repo's own established conventions rather than inventing new ones:
// the CHECK/CHECK_MSG counter pair is tests/t2018-slora-serial/t2018_offline_red.cpp's own
// convention (itself tests/test_main.cpp's own convention, cited there); the real-artifact
// load path (SslmModel::Load over a file read into memory) is tools/t2100_gpu_throughput.cpp's
// own convention (T-2100, the substrate's own throughput harness), reused verbatim because
// every "real 1.5B artifact" product cell in this suite needs the identical load path that
// harness already proved works.
//
// Model/adapter paths are NEVER hardcoded (StandardsDocument.md Sec5.2, cold-read self-contained
// / no project-specific-only assumption an artifact lives at one fixed path): every product cell
// that needs a real artifact takes it from g_model_path/g_adapter_path, populated from argv by
// each suite binary's own main(), matching tools/t2100_gpu_throughput.cpp's own
// `<model.sslm> [steps] [token_id]` argv convention. A cell whose product half needs an artifact
// argv did not supply reports SKIP (counted separately from PASS/FAIL/RED so a suite run without
// artifacts on the invoking machine is still legible), never a silent pass.
#ifndef SSLM_T2112_FIXTURE_COMMON_H
#define SSLM_T2112_FIXTURE_COMMON_H

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/model.h"

// The declared 1.0 API surface this whole suite is written against (Claude/Vitruvius/
// t2107-gpu-core-1p0-design-2026-08-14.md Sec4/Sec5, post T-2111-rung-5 fold). Promoted copy,
// this directory -- see sslm_gpu_1p0.h's own header comment.
#include "sslm_gpu_1p0.h"

static int GChecks = 0;
static int GFailures = 0;
static int GSkips = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		} \
	} while (0)

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

#define SKIP_MSG(...) \
	do { \
		++GSkips; \
		std::printf("SKIP %s:%d -- ", __FILE__, __LINE__); \
		std::printf(__VA_ARGS__); \
		std::printf("\n"); \
	} while (0)

// argv-populated real-artifact paths (StandardsDocument.md Sec5.4 real-workload rule: every
// applicable dimension's product cell runs the real 1.5B artifact where the design's own Sec11
// says so, never a hidden_size=2 stand-in). Empty means "not supplied on this invocation" --
// the cell SKIPs its product half rather than silently passing or fabricating a result.
static std::string g_model_1p5b_path;
static std::string g_model_0p5b_path;
static std::string g_adapter_path;   // real adapter artifact (e.g. shopkeeper-v2), for dim8/dim10

// Reuses tools/t2100_gpu_throughput.cpp's own load path verbatim (T-2100, this repo).
inline bool LoadRealModel(const std::string& path, superslm::SslmModelView* out_view,
                           std::vector<uint8_t>* out_bytes, std::string* out_err) {
	if (path.empty()) return false;
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) {
		if (out_err) *out_err = "could not open " + path;
		return false;
	}
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out_bytes->resize(sz > 0 ? (size_t)sz : 0);
	if (sz > 0) {
		const size_t n = std::fread(out_bytes->data(), 1, (size_t)sz, f);
		std::fclose(f);
		if (n != (size_t)sz) {
			if (out_err) *out_err = "short read on " + path;
			return false;
		}
	} else {
		std::fclose(f);
	}
	const superslm::SslmModelStatus st =
	    superslm::SslmModel::Load(out_bytes->data(), out_bytes->size(), *out_view, out_err);
	return st == superslm::SslmModelStatus::Ok;
}

// Parses this suite's own argv convention: <binary> [--model1p5b=PATH] [--model0p5b=PATH]
// [--adapter=PATH]. Every flag optional; a cell whose fixture is missing SKIPs rather than
// asserting against nothing.
inline void ParseFixtureArgs(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		auto take = [&](const char* flag) -> const char* {
			const size_t n = std::strlen(flag);
			return a.compare(0, n, flag) == 0 ? a.c_str() + n : nullptr;
		};
		if (const char* v = take("--model1p5b=")) g_model_1p5b_path = v;
		else if (const char* v = take("--model0p5b=")) g_model_0p5b_path = v;
		else if (const char* v = take("--adapter=")) g_adapter_path = v;
	}
}

#endif  // SSLM_T2112_FIXTURE_COMMON_H
