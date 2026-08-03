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
// T-1691 site kind 2/4 (the attention interior, build log
// Claude/Brunel/superslm-t1691-site-comparison-build2-2026-08-03.md) extends
// this probe with six further subcommands -- the attention interior's own
// new fixed-point primitives (RopeApplyPair, LandingRescale, BiasReconcileWide,
// CombineCarriedScale, IExpScaleConstants, SoftmaxRowQ15) -- for the SAME
// reason and by the SAME convention as the six already here.
//
// T-1691 site kind 3/4 (the MLP, this build) extends this probe with ONE
// further subcommand -- `silu_sigmoid_q15`, calling the REAL compiled
// `SiluSigmoidQ15` (src/silu_lut.cpp) linked against the SAME canonical SIL1
// table (`kSiluLutCanonicalTable`, include/superslm/silu_lut_canonical.h)
// `SslmModel::Load` pins every loaded artifact's own SIL1 section against
// byte-for-byte -- so this probe's own table is never a stand-in, it is the
// identical content any real artifact's own runtime table carries. This is
// T-183/D-SLM312's own most-scrutinised arithmetic (the SwiGLU activation
// site); no other MLP primitive needs its own subcommand -- gate_proj/
// up_proj/down_proj reuse ProjectAndFunnel (already probed indirectly via
// q_proj/o_proj's own real-engine composite comparison) and mlp_residual
// reuses ResidualReconcileSite (same).
//
// Usage:
//   t1691_primitive_probe normalize_scale <d_prime>
//   t1691_primitive_probe dynamic_scale_reciprocal <dn>
//   t1691_primitive_probe max_abs_reduce_wide <n> <x0> <x1> ... <x(n-1)>
//   t1691_primitive_probe requant_token_code_wide <x_i> <r> <s>
//   t1691_primitive_probe clamp_rope_code <raw>
//   t1691_primitive_probe requant_chain_checked <n> <x0> <x1> ... <x(n-1)>
//   t1691_primitive_probe rope_apply_pair <x> <y> <cos_q30> <sin_q30>
//   t1691_primitive_probe landing_rescale <branch_code> <m_a> <r_t> <e_a> <e_t>
//   t1691_primitive_probe bias_reconcile_wide <b> <q_b> <r_a> <e_a>
//   t1691_primitive_probe combine_carried_scale <a_m> <a_e> <b_m> <b_e>
//   t1691_primitive_probe iexp_scale_constants <m> <e> <ln2_q> <b_q> <ca_q>
//   t1691_primitive_probe softmax_row_q15 <q_ln2> <q_b> <q_c> <n> <s0> ... <s(n-1)>
//   t1691_primitive_probe silu_sigmoid_q15 <code> <m> <e>
//
// Every subcommand prints one line of `key=value` pairs on success and exits
// 0; a malformed invocation prints a diagnostic to stderr and exits 2 -- it
// never silently prints a default value for a subcommand it does not
// recognise (D-SLM715's own standard: a probe that cannot fail proves
// nothing).

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/intmath.h"
#include "superslm/silu_lut.h"
#include "superslm/silu_lut_canonical.h"

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

int CmdRopeApplyPair(int argc, char** argv) {
	if (argc != 6) return Fail("rope_apply_pair <x> <y> <cos_q30> <sin_q30>");
	const int32_t x = static_cast<int32_t>(std::strtol(argv[2], nullptr, 10));
	const int32_t y = static_cast<int32_t>(std::strtol(argv[3], nullptr, 10));
	const int32_t cos_q30 = static_cast<int32_t>(std::strtol(argv[4], nullptr, 10));
	const int32_t sin_q30 = static_cast<int32_t>(std::strtol(argv[5], nullptr, 10));
	const RopePair p = RopeApplyPair(x, y, cos_q30, sin_q30);
	std::printf("x=%lld y=%lld\n", static_cast<long long>(p.x), static_cast<long long>(p.y));
	return 0;
}

int CmdLandingRescale(int argc, char** argv) {
	if (argc != 7) return Fail("landing_rescale <branch_code> <m_a> <r_t> <e_a> <e_t>");
	const int64_t branch_code = std::strtoll(argv[2], nullptr, 10);
	const int64_t m_a = std::strtoll(argv[3], nullptr, 10);
	const int64_t r_t = std::strtoll(argv[4], nullptr, 10);
	const int64_t e_a = std::strtoll(argv[5], nullptr, 10);
	const int64_t e_t = std::strtoll(argv[6], nullptr, 10);
	uint64_t sat_count = 0;
	bool magnitude_exceeded = false;
	const int64_t raw =
	    LandingRescale(branch_code, m_a, r_t, e_a, e_t, &sat_count, &magnitude_exceeded);
	std::printf("raw=%lld magnitude_exceeded=%d saturation_count=%llu\n",
	            static_cast<long long>(raw), magnitude_exceeded ? 1 : 0,
	            static_cast<unsigned long long>(sat_count));
	return 0;
}

int CmdBiasReconcileWide(int argc, char** argv) {
	if (argc != 6) return Fail("bias_reconcile_wide <b> <q_b> <r_a> <e_a>");
	const int64_t b = std::strtoll(argv[2], nullptr, 10);
	const int64_t q_b = std::strtoll(argv[3], nullptr, 10);
	const int64_t r_a = std::strtoll(argv[4], nullptr, 10);
	const int64_t e_a = std::strtoll(argv[5], nullptr, 10);
	int64_t out = 0;
	const bool fits = BiasReconcileWide(b, q_b, r_a, e_a, &out);
	std::printf("out=%lld fits=%d\n", static_cast<long long>(out), fits ? 1 : 0);
	return 0;
}

int CmdCombineCarriedScale(int argc, char** argv) {
	if (argc != 6) return Fail("combine_carried_scale <a_m> <a_e> <b_m> <b_e>");
	const CarriedScale a{std::strtoll(argv[2], nullptr, 10), std::strtoll(argv[3], nullptr, 10)};
	const CarriedScale b{std::strtoll(argv[4], nullptr, 10), std::strtoll(argv[5], nullptr, 10)};
	const CarriedScale r = CombineCarriedScale(a, b);
	std::printf("m=%lld e=%lld\n", static_cast<long long>(r.m), static_cast<long long>(r.e));
	return 0;
}

int CmdIExpScaleConstants(int argc, char** argv) {
	if (argc != 7) return Fail("iexp_scale_constants <m> <e> <ln2_q> <b_q> <ca_q>");
	const int64_t m = std::strtoll(argv[2], nullptr, 10);
	const int64_t e = std::strtoll(argv[3], nullptr, 10);
	const int64_t ln2_q = std::strtoll(argv[4], nullptr, 10);
	const int64_t b_q = std::strtoll(argv[5], nullptr, 10);
	const int64_t ca_q = std::strtoll(argv[6], nullptr, 10);
	int64_t out_q_ln2 = 0, out_q_b = 0, out_q_c = 0;
	const IExpScaleDomain domain =
	    IExpScaleConstants(m, e, ln2_q, 30, b_q, 30, ca_q, 30, &out_q_ln2, &out_q_b, &out_q_c);
	const char* domain_name = "kUnknown";
	switch (domain) {
		case IExpScaleDomain::kOk: domain_name = "kOk"; break;
		case IExpScaleDomain::kBadCoefficient: domain_name = "kBadCoefficient"; break;
		case IExpScaleDomain::kNegativeShift: domain_name = "kNegativeShift"; break;
		case IExpScaleDomain::kNotRepresentable: domain_name = "kNotRepresentable"; break;
	}
	std::printf("domain=%s q_ln2=%lld q_b=%lld q_c=%lld\n", domain_name,
	            static_cast<long long>(out_q_ln2), static_cast<long long>(out_q_b),
	            static_cast<long long>(out_q_c));
	return 0;
}

int CmdSoftmaxRowQ15(int argc, char** argv) {
	if (argc < 6) return Fail("softmax_row_q15 <q_ln2> <q_b> <q_c> <n> <s0> ... <s(n-1)>");
	const int64_t q_ln2 = std::strtoll(argv[2], nullptr, 10);
	const int64_t q_b = std::strtoll(argv[3], nullptr, 10);
	const int64_t q_c = std::strtoll(argv[4], nullptr, 10);
	const size_t n = static_cast<size_t>(std::strtoull(argv[5], nullptr, 10));
	if (argc != static_cast<int>(6 + n)) return Fail("softmax_row_q15: argument count != n");
	std::vector<int64_t> scores(n);
	for (size_t i = 0; i < n; ++i) scores[i] = std::strtoll(argv[6 + i], nullptr, 10);
	std::vector<int64_t> probs(n);
	const bool well_formed = SoftmaxRowQ15(scores.data(), n, q_ln2, q_b, q_c, probs.data());
	std::printf("well_formed=%d probs=", well_formed ? 1 : 0);
	for (size_t i = 0; i < n; ++i) {
		std::printf("%lld%s", static_cast<long long>(probs[i]), (i + 1 < n) ? "," : "");
	}
	std::printf("\n");
	return 0;
}

int CmdSiluSigmoidQ15(int argc, char** argv) {
	if (argc != 5) return Fail("silu_sigmoid_q15 <code> <m> <e>");
	const int8_t code = static_cast<int8_t>(std::strtol(argv[2], nullptr, 10));
	const int64_t m = std::strtoll(argv[3], nullptr, 10);
	const int e = static_cast<int>(std::strtol(argv[4], nullptr, 10));
	const int32_t sig = SiluSigmoidQ15(kSiluLutCanonicalTable, code, m, e);
	std::printf("sig=%d\n", sig);
	return 0;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		return Fail(
		    "usage: t1691_primitive_probe "
		    "<normalize_scale|dynamic_scale_reciprocal|max_abs_reduce_wide|"
		    "requant_token_code_wide|clamp_rope_code|requant_chain_checked|"
		    "rope_apply_pair|landing_rescale|bias_reconcile_wide|"
		    "combine_carried_scale|iexp_scale_constants|softmax_row_q15|"
		    "silu_sigmoid_q15> ...");
	}
	const char* cmd = argv[1];
	if (std::strcmp(cmd, "normalize_scale") == 0) return CmdNormalizeScale(argc, argv);
	if (std::strcmp(cmd, "dynamic_scale_reciprocal") == 0) return CmdDynamicScaleReciprocal(argc, argv);
	if (std::strcmp(cmd, "max_abs_reduce_wide") == 0) return CmdMaxAbsReduceWide(argc, argv);
	if (std::strcmp(cmd, "requant_token_code_wide") == 0) return CmdRequantTokenCodeWide(argc, argv);
	if (std::strcmp(cmd, "clamp_rope_code") == 0) return CmdClampRopeCode(argc, argv);
	if (std::strcmp(cmd, "requant_chain_checked") == 0) return CmdRequantChainChecked(argc, argv);
	if (std::strcmp(cmd, "rope_apply_pair") == 0) return CmdRopeApplyPair(argc, argv);
	if (std::strcmp(cmd, "landing_rescale") == 0) return CmdLandingRescale(argc, argv);
	if (std::strcmp(cmd, "bias_reconcile_wide") == 0) return CmdBiasReconcileWide(argc, argv);
	if (std::strcmp(cmd, "combine_carried_scale") == 0) return CmdCombineCarriedScale(argc, argv);
	if (std::strcmp(cmd, "iexp_scale_constants") == 0) return CmdIExpScaleConstants(argc, argv);
	if (std::strcmp(cmd, "softmax_row_q15") == 0) return CmdSoftmaxRowQ15(argc, argv);
	if (std::strcmp(cmd, "silu_sigmoid_q15") == 0) return CmdSiluSigmoidQ15(argc, argv);
	std::fprintf(stderr, "t1691_primitive_probe: unrecognized subcommand \"%s\"\n", cmd);
	return 2;
}
