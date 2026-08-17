// T-2130 (Curie) -- shared CHECK harness and real-artifact loading helper for the G5
// constrained-decoding red suite. Reuses this repo's own established conventions rather than
// inventing new ones: the CHECK/CHECK_MSG/SKIP_MSG counter triple and the argv fixture
// convention are tests/t2112-gpu-1p0-red-suite/fixture_common.h's own shapes (itself
// tests/t2018-slora-serial/t2018_offline_red.cpp's convention, itself tests/test_main.cpp's);
// the real-artifact load path (SslmModel::Load over a file read into memory) is
// tools/t2100_gpu_throughput.cpp's own convention, reused verbatim.
//
// Model/schema/adapter paths are NEVER hardcoded (StandardsDocument.md Sec5.2, cold-read
// self-contained): every product cell that needs a real artifact takes it from
// g_model_*_path / g_adapter_path, populated from argv by each suite binary's own main(). A
// cell whose product half needs an artifact argv did not supply reports SKIP (counted
// separately from PASS/FAIL/RED so a suite run without artifacts on the invoking machine is
// still legible), never a silent pass.
#ifndef SSLM_T2130_FIXTURE_COMMON_H
#define SSLM_T2130_FIXTURE_COMMON_H

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/model.h"

// The declared G5 ABI surface this whole suite is written against (Claude/Vitruvius/
// t2119-g5-constrained-decoding-design-2026-08-16.md Sec5, post rung-6 confirmation fold).
// Promoted copy, this directory -- see sslm_g5.h's own header comment.
#include "sslm_g5.h"

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

// argv-populated real-artifact paths (StandardsDocument.md Sec5.4 real-workload rule).
// Empty means "not supplied on this invocation" -- the cell SKIPs its product half rather
// than silently passing or fabricating a result.
static std::string g_model_1p5b_path;    // real 1.5B artifact, WITH a compiled schema set
static std::string g_adversarial_schema_model_path;  // artifact compiled with the adversarial
                                                       // deep-nesting schema corpus (G5-1 gate)
static std::string g_adapter_path;       // real adapter artifact (shopkeeper-v2), for dim8/dim10
static std::string g_reference_schema_name = "shopkeeper_intent_extraction";  // design Sec3's
                                                       // reference task, the plan's own D-SLM32
                                                       // reference schema name -- overridable.
static std::string g_model_1p5b_variant_path;  // a SECOND artifact, content-hash-matched to
                                                // --model1p5b but compiled with a DIFFERENT (or
                                                // absent) schema set -- dim2 mechanism cell 4's
                                                // own "model_b" fixture (T-2132 harness fix).
static std::string g_secondary_schema_name = "g5_minimal_one_field";  // a schema distinct from
                                                // g_reference_schema_name, for cells needing two
                                                // non-identical schema handles (T-2132).

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

// Parses this suite's own argv convention: <binary> [--model1p5b=PATH] [--adversarial=PATH]
// [--adapter=PATH] [--schema=NAME] [--model1p5b_variant=PATH] [--schema2=NAME]. Every flag
// optional; a cell whose fixture is missing SKIPs rather than asserting against nothing.
inline void ParseFixtureArgs(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		auto take = [&](const char* flag) -> const char* {
			const size_t n = std::strlen(flag);
			return a.compare(0, n, flag) == 0 ? a.c_str() + n : nullptr;
		};
		if (const char* v = take("--model1p5b_variant=")) g_model_1p5b_variant_path = v;
		else if (const char* v = take("--model1p5b=")) g_model_1p5b_path = v;
		else if (const char* v = take("--adversarial=")) g_adversarial_schema_model_path = v;
		else if (const char* v = take("--adapter=")) g_adapter_path = v;
		else if (const char* v = take("--schema2=")) g_secondary_schema_name = v;
		else if (const char* v = take("--schema=")) g_reference_schema_name = v;
	}
}

// --- Real-artifact fixture helpers (T-2132/Curie build-harness fix): each suite main() below
// uses these to actually CALL its cells with real handles when a real artifact is supplied on
// argv, and to SKIP -- never silently pass, never assert against a null handle -- the cells it
// cannot construct when one isn't. This replaces the prior address-take-only mains, whose
// clean-link runtime output was unconditionally checks=0 regardless of what the ABI does (see
// Claude/Brunel/t2132-g5-build-2026-08-16.md, "the harness's own construction" finding). Calling
// sslm_model_map here is safe today (it already ships, T-2139); sslm_schema_lookup and every
// other genuinely G5-only verb remain unresolved externals until G5-2 lands, so these helpers
// change nothing about this suite's RED BY LINK state pre-G5 -- a function that only takes the
// address of a symbol and a function that calls it both force the identical linker resolution.

// Loads a file fully into `keepalive` and maps it via sslm_model_map. `keepalive` must outlive
// every call made against the returned handle (sslm_model_map does not copy the bytes). Returns
// false -- never asserts -- when the path is empty, the file cannot be read, or the artifact is
// rejected; callers SKIP rather than call a cell with a null model handle.
inline bool TryMapRealModel(const std::string& path, std::vector<uint8_t>* keepalive,
                             sslm_model* out_model) {
	*out_model = nullptr;
	if (path.empty()) return false;
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	keepalive->resize(sz > 0 ? (size_t)sz : 0);
	if (sz > 0) {
		const size_t n = std::fread(keepalive->data(), 1, (size_t)sz, f);
		std::fclose(f);
		if (n != (size_t)sz) return false;
	} else {
		std::fclose(f);
	}
	return sslm_model_map(keepalive->data(), keepalive->size(), out_model) == SSLM_OK;
}

// Looks up a named schema on an already-mapped model. Returns false (leaves *out null) when the
// model is null or the lookup fails -- including "unresolved external", pre-G5-2, which never
// reaches runtime at all (the whole binary fails to link, per this suite's own RED BY LINK
// contract; this function's mere presence changes nothing about that).
inline bool TryLookupSchema(sslm_model model, const char* name, sslm_schema* out) {
	*out = nullptr;
	if (!model) return false;
	return sslm_schema_lookup(model, name, out) == SSLM_OK;
}

#endif  // SSLM_T2130_FIXTURE_COMMON_H
