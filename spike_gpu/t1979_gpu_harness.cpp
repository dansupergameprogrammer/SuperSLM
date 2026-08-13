// t1979_gpu_harness.cpp -- T-1979 GPU walking-skeleton spike, GPU-side runner.
//
// Disposable spike harness (branch brunel/t1979-gpu-skeleton, never merges).
// Reads the CPU-side dump (t1979_dump_qproj.exe's own output: real layer0
// q_proj weights/fold/bias from the real 1.5B artifact, a real attn_norm
// input vector, and the CPU production reference result captured via the
// trace hook), uploads it to the GPU via the proven gpu.hpp D3D12 harness
// pattern (Claude/Laplace/gpu-determinism), dispatches ONE compute shader
// (shaders/qproj_site.hlsl) that performs the whole composed
// ProjectAndFunnel(q_proj) site on-device, reads the result back, and
// compares it BYTE-FOR-BYTE against the CPU reference. Prints PASS/FAIL
// plainly; this result is a decision-bearing number and gets a blind
// instrument review before it is banked (per the commissioning brief) --
// this tool states the outcome, it does not certify it.
//
// Usage: t1979_gpu_harness <dump.bin> <qproj_site.cso>
#include "gpu.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

template <typename T>
bool ReadRaw(std::ifstream& f, T& out) {
	f.read(reinterpret_cast<char*>(&out), sizeof(out));
	return static_cast<bool>(f);
}

template <typename T>
bool ReadVec(std::ifstream& f, std::vector<T>& out, size_t n) {
	out.resize(n);
	if (n == 0) return true;
	f.read(reinterpret_cast<char*>(out.data()), n * sizeof(T));
	return static_cast<bool>(f);
}

struct Dump {
	uint64_t hidden_size = 0;
	std::vector<int8_t> in_codes;
	int64_t in_scale_m = 0, in_scale_e = 0;
	std::vector<int8_t> q_weight;         // hidden_size*hidden_size
	std::vector<int32_t> fold_identity, fold_mult, fold_shift;
	uint8_t has_bias = 0;
	std::vector<int64_t> q_bias;
	int64_t site_m = 0, site_e = 0;
	std::vector<int64_t> cpu_wide_row;
	int64_t cpu_d_prime = 0, cpu_dn = 0;
	int32_t cpu_s = 0;
	int64_t cpu_r = 0;
	std::vector<int8_t> cpu_out_codes;
	int64_t cpu_out_scale_m = 0, cpu_out_scale_e = 0;
};

bool LoadDump(const char* path, Dump& d) {
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		std::fprintf(stderr, "cannot open dump \"%s\"\n", path);
		return false;
	}
	if (!ReadRaw(f, d.hidden_size)) return false;
	const size_t hs = static_cast<size_t>(d.hidden_size);
	if (!ReadVec(f, d.in_codes, hs)) return false;
	if (!ReadRaw(f, d.in_scale_m) || !ReadRaw(f, d.in_scale_e)) return false;
	if (!ReadVec(f, d.q_weight, hs * hs)) return false;
	if (!ReadVec(f, d.fold_identity, hs)) return false;
	if (!ReadVec(f, d.fold_mult, hs)) return false;
	if (!ReadVec(f, d.fold_shift, hs)) return false;
	if (!ReadRaw(f, d.has_bias)) return false;
	if (!ReadVec(f, d.q_bias, hs)) return false;
	if (!ReadRaw(f, d.site_m) || !ReadRaw(f, d.site_e)) return false;
	if (!ReadVec(f, d.cpu_wide_row, hs)) return false;
	if (!ReadRaw(f, d.cpu_d_prime)) return false;
	if (!ReadRaw(f, d.cpu_dn)) return false;
	if (!ReadRaw(f, d.cpu_s)) return false;
	if (!ReadRaw(f, d.cpu_r)) return false;
	if (!ReadVec(f, d.cpu_out_codes, hs)) return false;
	if (!ReadRaw(f, d.cpu_out_scale_m) || !ReadRaw(f, d.cpu_out_scale_e)) return false;
	return true;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: %s <dump.bin> <qproj_site.cso>\n", argv[0]);
		return 2;
	}
	Dump d;
	if (!LoadDump(argv[1], d)) {
		std::fprintf(stderr, "FAILED at stage=dump_read\n");
		return 1;
	}
	const size_t hs = static_cast<size_t>(d.hidden_size);
	std::printf("dump loaded: hidden_size=%zu has_bias=%d\n", hs, static_cast<int>(d.has_bias));

	// Widen int8 -> int32 for GPU upload (the spike's own documented
	// simplification -- data-layout/packing is a real port question, not
	// solved here; see the build log).
	std::vector<int32_t> act_i32(hs), weight_i32(hs * hs);
	for (size_t i = 0; i < hs; ++i) act_i32[i] = d.in_codes[i];
	for (size_t i = 0; i < hs * hs; ++i) weight_i32[i] = d.q_weight[i];

	std::vector<int64_t> scalars = {d.in_scale_m, d.in_scale_e, d.site_m, d.site_e};

	Gpu g;
	g.init();
	g.reportFeatures();

	auto cso = readFile(argv[2]);
	// b0: 2 root constants (HiddenSize, HasBias); t0..t6: 7 root SRVs; u0: 1 UAV.
	auto rs = g.makeRootSig(/*numRootConst=*/2, /*numSRV=*/7, /*hasUAV=*/true);
	auto pso = g.makePSO(rs.Get(), cso);

	auto actBuf = g.upload(act_i32.data(), act_i32.size() * sizeof(int32_t));
	auto weightBuf = g.upload(weight_i32.data(), weight_i32.size() * sizeof(int32_t));
	auto idBuf = g.upload(d.fold_identity.data(), d.fold_identity.size() * sizeof(int32_t));
	auto multBuf = g.upload(d.fold_mult.data(), d.fold_mult.size() * sizeof(int32_t));
	auto shiftBuf = g.upload(d.fold_shift.data(), d.fold_shift.size() * sizeof(int32_t));
	auto biasBuf = g.upload(d.q_bias.data(), d.q_bias.size() * sizeof(int64_t));
	auto scalarsBuf = g.upload(scalars.data(), scalars.size() * sizeof(int64_t));

	const std::vector<uint32_t> rootConsts = {static_cast<uint32_t>(d.hidden_size),
	                                           static_cast<uint32_t>(d.has_bias)};
	const std::vector<ID3D12Resource*> srvs = {actBuf.Get(), weightBuf.Get(),   idBuf.Get(),
	                                            multBuf.Get(), shiftBuf.Get(),  biasBuf.Get(),
	                                            scalarsBuf.Get()};

	const uint64_t out_codes_base = hs * 8 + 32;
	const uint64_t out_bytes = out_codes_base + hs * 4 + 16;

	auto uav = g.makeBuffer(out_bytes, D3D12_HEAP_TYPE_DEFAULT,
	                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
	                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	std::printf("dispatching: one thread, one compute call, %llu-byte readback...\n",
	            static_cast<unsigned long long>(out_bytes));
	auto out = g.dispatch(rs.Get(), pso.Get(), rootConsts, srvs, uav.Get(), out_bytes, 1, 1, 1);
	std::printf("dispatch complete, %zu bytes read back\n", out.size());

	// --- Parse GPU output per the shader's own documented layout. ----------
	auto RdI64 = [&](uint64_t off) -> int64_t {
		int64_t v;
		std::memcpy(&v, out.data() + off, sizeof(v));
		return v;
	};
	auto RdI32 = [&](uint64_t off) -> int32_t {
		int32_t v;
		std::memcpy(&v, out.data() + off, sizeof(v));
		return v;
	};

	std::vector<int64_t> gpu_wide_row(hs);
	for (size_t i = 0; i < hs; ++i) gpu_wide_row[i] = RdI64(i * 8);
	const int64_t gpu_d_prime = RdI64(hs * 8 + 0);
	const int64_t gpu_dn = RdI64(hs * 8 + 8);
	const int32_t gpu_s = static_cast<int32_t>(RdI64(hs * 8 + 16));
	const int64_t gpu_r = RdI64(hs * 8 + 24);
	std::vector<int8_t> gpu_out_codes(hs);
	for (size_t i = 0; i < hs; ++i) {
		int32_t v = RdI32(static_cast<uint64_t>(out_codes_base) + i * 4);
		gpu_out_codes[i] = static_cast<int8_t>(v);
	}
	const int64_t gpu_out_scale_m = RdI64(out_codes_base + hs * 4 + 0);
	const int64_t gpu_out_scale_e = RdI64(out_codes_base + hs * 4 + 8);

	// --- Compare, element-by-element, against the CPU production reference. -
	int wide_row_mismatches = 0;
	for (size_t i = 0; i < hs; ++i) {
		if (gpu_wide_row[i] != d.cpu_wide_row[i]) {
			if (wide_row_mismatches < 5) {
				std::fprintf(stderr, "  wide_row[%zu] MISMATCH gpu=%lld cpu=%lld\n", i,
				             static_cast<long long>(gpu_wide_row[i]),
				             static_cast<long long>(d.cpu_wide_row[i]));
			}
			++wide_row_mismatches;
		}
	}
	const bool d_prime_match = gpu_d_prime == d.cpu_d_prime;
	const bool dn_match = gpu_dn == d.cpu_dn;
	const bool s_match = gpu_s == d.cpu_s;
	const bool r_match = gpu_r == d.cpu_r;

	int code_mismatches = 0;
	for (size_t i = 0; i < hs; ++i) {
		if (gpu_out_codes[i] != d.cpu_out_codes[i]) {
			if (code_mismatches < 5) {
				std::fprintf(stderr, "  out_codes[%zu] MISMATCH gpu=%d cpu=%d\n", i,
				             static_cast<int>(gpu_out_codes[i]), static_cast<int>(d.cpu_out_codes[i]));
			}
			++code_mismatches;
		}
	}
	const bool scale_match =
	    gpu_out_scale_m == d.cpu_out_scale_m && gpu_out_scale_e == d.cpu_out_scale_e;

	std::printf(
	    "GEMM+WSC1+BIA1 stage (wide_row, %zu elements): %s (%d mismatches)\n", hs,
	    wide_row_mismatches == 0 ? "MATCH" : "DIFFER", wide_row_mismatches);
	std::printf("funnel intermediates: d_prime=%s dn=%s s=%s r=%s\n",
	            d_prime_match ? "MATCH" : "DIFFER", dn_match ? "MATCH" : "DIFFER",
	            s_match ? "MATCH" : "DIFFER", r_match ? "MATCH" : "DIFFER");
	std::printf("  gpu: d_prime=%lld dn=%lld s=%d r=%lld\n", static_cast<long long>(gpu_d_prime),
	            static_cast<long long>(gpu_dn), gpu_s, static_cast<long long>(gpu_r));
	std::printf("  cpu: d_prime=%lld dn=%lld s=%d r=%lld\n", static_cast<long long>(d.cpu_d_prime),
	            static_cast<long long>(d.cpu_dn), d.cpu_s, static_cast<long long>(d.cpu_r));
	std::printf("out_codes (%zu elements): %s (%d mismatches)\n", hs,
	            code_mismatches == 0 ? "MATCH" : "DIFFER", code_mismatches);
	std::printf("out_scale: %s  gpu=(%lld,%lld)  cpu=(%lld,%lld)\n",
	            scale_match ? "MATCH" : "DIFFER", static_cast<long long>(gpu_out_scale_m),
	            static_cast<long long>(gpu_out_scale_e), static_cast<long long>(d.cpu_out_scale_m),
	            static_cast<long long>(d.cpu_out_scale_e));

	const bool all_match = wide_row_mismatches == 0 && d_prime_match && dn_match && s_match &&
	                        r_match && code_mismatches == 0 && scale_match;
	std::printf("\nT1979_RESULT: %s\n", all_match ? "PASS (bit-identical GPU vs CPU)" :
	                                                  "FAIL (see mismatches above)");
	return all_match ? 0 : 1;
}
