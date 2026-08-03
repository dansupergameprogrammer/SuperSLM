// T-1691 primitive probe -- test-support tool, not production code (T-1683
// source-attribution campaign, Claude/Vitruvius/superslm-t1683-source-
// attribution-design-2026-08-02.md S7 red-first proof part 2).
//
// Curie's own red suite for T-1691 (the parity-shadow/precision-shadow
// decomposition) needs to cross-check the Python parity shadow's
// independently-coded primitive reimplementations against the REAL compiled
// C++ functions -- NormalizeScale, DynamicScaleReciprocal, MaxAbsReduceWide,
// RequantTokenCodeWide, ClampRopeCode, and RequantChainChecked's own C29
// domain check -- before either shadow is trusted on real data (design S7
// red-first proof part 2). This tool exposes those already-real functions
// through a CLI so a Python test can drive them as a subprocess and compare
// stdout against its own independent computation, the same "drive a built
// binary as a subprocess and parse its stdout" convention T-1689's own
// red-first proof already established (tools/sslm_layer_trace.cpp).
//
// This file is the SAME shape as tests/cert_intmath.cpp -- a standalone
// certifier-style probe, not part of the CMake-built superslm_tests binary
// and not part of the CI suite; built by tests/build_t1691_primitive_probe.bat
// (mirrors build_cert.bat's MSVC invocation, extended with the additional
// core sources this probe's own functions live in). It links the CLEAN
// superslm core sources unmodified -- no test-only injection, no stub -- so
// every value it prints is the real engine's own arithmetic.
//
// Usage:
//   t1691_primitive_probe normalize_scale <d_prime>
//   t1691_primitive_probe dynamic_scale_reciprocal <dn>
//   t1691_primitive_probe max_abs_reduce_wide <n> <x0> <x1> ... <x(n-1)>
//   t1691_primitive_probe requant_token_code_wide <x_i> <r> <s>
//   t1691_primitive_probe clamp_rope_code <raw>
//   t1691_primitive_probe requant_chain_checked <n> <x0> <x1> ... <x(n-1)>
//
// Every subcommand prints one line of `key=value` pairs on success and exits
// 0; a malformed invocation prints a diagnostic to stderr and exits 2 -- it
// never silently prints a default value for a subcommand it does not
// recognise (D-SLM715's own standard: a probe that cannot fail proves
// nothing).

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/intmath.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

using namespace superslm;

namespace {

// requant_chain_checked's own site_constant/incoming: a canonical identity
// CarriedScale (m*2^e == 1.0, m in [2^30, 2^31) per checked_chain_funnel.h's
// own canonical-form contract). This value affects only the carried-scale
// fold (C26), never the C29 domain check itself (design S7 red-first proof
// part 2's own statement: "the constraint... does not depend on... incoming
// or site_constant"), so its exact choice does not change what this
// subcommand certifies.
constexpr CarriedScale kIdentityScale{.m = int64_t{1} << 30, .e = -30};

int Fail(const char* msg) {
	std::fprintf(stderr, "t1691_primitive_probe: %s\n", msg);
	return 2;
}

int CmdNormalizeScale(int argc, char** argv) {
	if (argc != 3) return Fail("normalize_scale <d_prime>");
	const int64_t d_prime = std::strtoll(argv[2], nullptr, 10);
	const NormalizedScale ns = NormalizeScale(d_prime);
	std::printf("dn=%lld s=%d\n", static_cast<long long>(ns.dn), ns.s);
	return 0;
}

int CmdDynamicScaleReciprocal(int argc, char** argv) {
	if (argc != 3) return Fail("dynamic_scale_reciprocal <dn>");
	const int64_t dn = std::strtoll(argv[2], nullptr, 10);
	const int64_t r = DynamicScaleReciprocal(dn);
	std::printf("r=%lld\n", static_cast<long long>(r));
	return 0;
}

int CmdMaxAbsReduceWide(int argc, char** argv) {
	if (argc < 3) return Fail("max_abs_reduce_wide <n> <x0> ... <x(n-1)>");
	const size_t n = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
	if (argc != static_cast<int>(3 + n)) return Fail("max_abs_reduce_wide: argument count != n");
	std::vector<int64_t> x(n);
	for (size_t i = 0; i < n; ++i) x[i] = std::strtoll(argv[3 + i], nullptr, 10);
	const int64_t d = MaxAbsReduceWide(x.data(), n);
	std::printf("d=%lld\n", static_cast<long long>(d));
	return 0;
}

int CmdRequantTokenCodeWide(int argc, char** argv) {
	if (argc != 5) return Fail("requant_token_code_wide <x_i> <r> <s>");
	const int64_t x_i = std::strtoll(argv[2], nullptr, 10);
	const int64_t r = std::strtoll(argv[3], nullptr, 10);
	const int s = static_cast<int>(std::strtol(argv[4], nullptr, 10));
	const int8_t code = RequantTokenCodeWide(x_i, r, s);
	std::printf("code=%d\n", static_cast<int>(code));
	return 0;
}

int CmdClampRopeCode(int argc, char** argv) {
	if (argc != 3) return Fail("clamp_rope_code <raw>");
	const int64_t raw = std::strtoll(argv[2], nullptr, 10);
	const int64_t code = ClampRopeCode(raw);
	std::printf("code=%lld\n", static_cast<long long>(code));
	return 0;
}

int CmdRequantChainChecked(int argc, char** argv) {
	if (argc < 3) return Fail("requant_chain_checked <n> <x0> ... <x(n-1)>");
	const size_t n = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
	if (argc != static_cast<int>(3 + n)) return Fail("requant_chain_checked: argument count != n");
	std::vector<int64_t> wide_row(n);
	for (size_t i = 0; i < n; ++i) wide_row[i] = std::strtoll(argv[3 + i], nullptr, 10);
	std::vector<int8_t> out_codes(n);
	CarriedScale out_scale;
	const ChainResult result = RequantChainChecked(
	    wide_row.data(), n, std::span<const CarriedScale>{}, kIdentityScale, out_codes.data(),
	    &out_scale, "t1691_primitive_probe", 0, nullptr);
	std::printf("status=%s", SslmForwardStatusName(result.status));
	if (result.status == SslmForwardStatus::Ok) {
		std::printf(" out_scale.m=%lld out_scale.e=%lld codes=",
		            static_cast<long long>(out_scale.m), static_cast<long long>(out_scale.e));
		for (size_t i = 0; i < n; ++i) {
			std::printf("%d%s", static_cast<int>(out_codes[i]), (i + 1 < n) ? "," : "");
		}
	}
	std::printf("\n");
	return 0;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		return Fail(
		    "usage: t1691_primitive_probe "
		    "<normalize_scale|dynamic_scale_reciprocal|max_abs_reduce_wide|"
		    "requant_token_code_wide|clamp_rope_code|requant_chain_checked> ...");
	}
	const char* cmd = argv[1];
	if (std::strcmp(cmd, "normalize_scale") == 0) return CmdNormalizeScale(argc, argv);
	if (std::strcmp(cmd, "dynamic_scale_reciprocal") == 0) return CmdDynamicScaleReciprocal(argc, argv);
	if (std::strcmp(cmd, "max_abs_reduce_wide") == 0) return CmdMaxAbsReduceWide(argc, argv);
	if (std::strcmp(cmd, "requant_token_code_wide") == 0) return CmdRequantTokenCodeWide(argc, argv);
	if (std::strcmp(cmd, "clamp_rope_code") == 0) return CmdClampRopeCode(argc, argv);
	if (std::strcmp(cmd, "requant_chain_checked") == 0) return CmdRequantChainChecked(argc, argv);
	std::fprintf(stderr, "t1691_primitive_probe: unrecognized subcommand \"%s\"\n", cmd);
	return 2;
}
