// Precached-prefixes plan §7 4.6 [F3] (rev 1.1, PC-5; rev 1.2): byte identity, on the real 0.5B and 1.5B artifacts at
// L = 1,000, comparing every layer, head and position (the existing evidence is one position, S21).
//   (a) the same prefix built by prefill on the CPU and on this GPU: the 'SPX1' rows and state fields are byte-equal,
//       the header's backend, producer and digest excepted (bytes [4, 60): backend, engine version, adapter vendor
//       and device, digest);
//   (b) the same prefix on the RTX 2080 SUPER and on the RX 7900 XTX: blobs byte-equal past the producer and digest
//       (the same [4, 60) excepted; both cards run one build, so the version agrees anyway);
//   (c) a blob saved on one GPU restored on the other, with the producer check skipped by a test seam
//       (SetGpuPrefixRestoreSkipAdapterCheckForTest, gpu_port.h): adopter tokens equal the target's prefill twin's.
// A difference is a finding, not a failed step (plan §7 4.6, §10 R7): PC-5's refusal stays, and the difference goes to
// the principal as evidence against docs/platform-support.md's CPU/GPU K/V identity claim. The cell is still graded:
// it goes red on a difference, and its output names the first differing (layer, K|V, head, position) and the count.
//
// The prefix: the first 1,000 ids of the prompt file (tests/t3035-precache-refs/prompts/world.ids for the 0.5B,
// world.1p5.ids for the 1.5B; 1,012 ids each), one batched GPU prefill, frozen. The CPU twin: the same ids through
// the CPU ABI (sslm_prefix_prefill, 64 per call) on the same artifact's bytes, in a page pool at the CPU's own B.
// SPX1 is canonical ([layer][K|V][head][pos < L][d]), so the two page geometries do not enter the comparison.
//
// Run, one model per process (plan §9 F3 on the box; the work-machine kit on the 7900 XTX):
//   pcg_cell_4_6.exe --model=FILE --tag=0p5|1p5 --ids=FILE --out=DIR       (a): writes DIR/gpu-<tag>-<vvvv_dddd>.spx1
//                                                                           and DIR/cpu-<tag>.spx1
//       [--other=FILE]    (b): the other card's gpu-<tag>-*.spx1, compared with this card's
//       [--foreign=FILE]  (c): the other card's blob, restored here through the seam (and refused without it)
// Configuration: alloc (the seam is a test-build seam, SUPERSLM_GPU_ALLOC_FAULT_INJECTION).
#include "pcg_f3_common.h"
#include "g1e_cpu_twin.h"

#include <climits>
#include <cstdio>

#if !defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
#error "pcg_cell_4_6 needs the adapter-check seam: build it in the alloc configuration"
#endif

using namespace pcg;
using namespace pcg::f3;

namespace {

constexpr int64_t kLen = 1000;
constexpr int64_t kAdopt = 64;

// Byte identity of two SPX1 blobs of one geometry outside [4, 60), row by row: every (layer, K|V, head, position)
// row of D bytes, the residual, and the header's state fields. Returns true when identical.
bool SameOutsideProducer(const Env& e, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, const char* what) {
	const Spx1 va = Spx1Parse(a, e.geo), vb = Spx1Parse(b, e.geo);
	CHECK_MSG(va.magic && vb.magic && va.size_exact && vb.size_exact && va.digest_ok && vb.digest_ok,
	          "%s: not two valid SPX1 blobs of this geometry (magic %d/%d, size %d/%d, digest %d/%d)", what, va.magic,
	          vb.magic, va.size_exact, vb.size_exact, va.digest_ok, vb.digest_ok);
	if (!(va.magic && vb.magic && va.size_exact && vb.size_exact && va.length == vb.length)) {
		CHECK_MSG(va.length == vb.length, "%s: lengths %lld and %lld", what, static_cast<long long>(va.length),
		          static_cast<long long>(vb.length));
		return false;
	}
	// The header's state fields: [0, 4) and [60, 184).
	size_t header_diff = 0, first_header = SIZE_MAX;
	for (size_t i = 0; i < kHeader; ++i)
		if ((i < 4 || i >= 60) && a[i] != b[i]) {
			++header_diff;
			if (first_header == SIZE_MAX) first_header = i;
		}
	const bool residual = std::memcmp(a.data() + kHeader, b.data() + kHeader, e.geo.hidden) == 0;
	const int64_t L = va.length;
	size_t rows = 0, bad = 0;
	uint32_t fl = 0, fhalf = 0, fh = 0;
	int64_t fpos = -1;
	for (uint32_t l = 0; l < e.geo.L; ++l)
		for (uint32_t half = 0; half < 2; ++half)
			for (uint32_t h = 0; h < e.geo.Hkv; ++h)
				for (int64_t pos = 0; pos < L; ++pos) {
					++rows;
					const size_t o = Spx1RowOff(e.geo, L, l, half, h, pos);
					if (std::memcmp(a.data() + o, b.data() + o, e.geo.D) != 0) {
						if (bad++ == 0) {
							fl = l;
							fhalf = half;
							fh = h;
							fpos = pos;
						}
					}
				}
	const bool same_bytes = a.size() == b.size() && [&] {
		for (size_t i = 0; i < a.size(); ++i)
			if ((i < 4 || i >= 60) && a[i] != b[i]) return false;
		return true;
	}();
	std::printf("%s: L = %lld, %zu rows of %u bytes compared (%u layers x K,V x %u heads x %lld positions): %zu differ%s; "
	            "residual %s; header state fields %s; whole blob outside [4, 60) %s\n",
	            what, static_cast<long long>(L), rows, e.geo.D, e.geo.L, e.geo.Hkv, static_cast<long long>(L), bad,
	            bad ? (" (first: layer " + std::to_string(fl) + (fhalf ? " V" : " K") + " head " + std::to_string(fh) +
	                   " position " + std::to_string(fpos) + ")")
	                      .c_str()
	                : "",
	            residual ? "equal" : "DIFFERS",
	            header_diff ? ("DIFFER (" + std::to_string(header_diff) + " bytes, first at " + std::to_string(first_header) + ")").c_str()
	                        : "equal",
	            same_bytes ? "IDENTICAL" : "DIFFERENT");
	CHECK_MSG(bad == 0, "%s: %zu of %zu K/V rows differ (a finding: PC-5's refusal stays)", what, bad, rows);
	CHECK_MSG(residual, "%s: the residual differs (a finding)", what);
	CHECK_MSG(header_diff == 0, "%s: %zu header state bytes differ (a finding)", what, header_diff);
	CHECK_MSG(same_bytes, "%s: the blobs differ outside [4, 60) (a finding)", what);
	return same_bytes && bad == 0 && residual && header_diff == 0;
}

std::vector<uint8_t> CpuTwin(const Env& e, const std::vector<int32_t>& tokens) {
	cpu::Model m(e.fx.bytes);
	if (!m.ok) return {};
	const int64_t B = sslm_kv_page_positions(m.model);
	CHECK_MSG(B > 0, "CPU twin: sslm_kv_page_positions reads %lld", static_cast<long long>(B));
	if (B <= 0) return {};
	const uint32_t pages = static_cast<uint32_t>((static_cast<int64_t>(tokens.size()) + B - 1) / B + 2);
	cpu::Pool pool(m.model, pages);
	if (!pool.pool) return {};
	sslm_prefix cp = nullptr;
	sslm_status cs = sslm_prefix_begin_budgeted(m.model, &pool.pool, static_cast<int32_t>(tokens.size()), &cp);
	if (cs == SSLM_OK) {
		pool.Prefix(cp);
		cs = cpu::PrefixPrefillAll(m.model, cp, tokens);
	}
	if (cs == SSLM_OK) cs = sslm_prefix_freeze(cp);
	CHECK_MSG(cs == SSLM_OK, "CPU twin prefix of %zu tokens: %d", tokens.size(), static_cast<int>(cs));
	if (cs != SSLM_OK) return {};
	size_t n = 0;
	(void)sslm_prefix_save(cp, nullptr, &n);
	std::vector<uint8_t> blob(n);
	cs = sslm_prefix_save(cp, blob.data(), &n);
	CHECK_MSG(cs == SSLM_OK, "CPU twin save: %d", static_cast<int>(cs));
	if (cs != SSLM_OK) return {};
	blob.resize(n);
	return blob;
}

void Cell(const Env& e, int argc, char** argv) {
	const std::string tag = Arg(argc, argv, "--tag"), ids_path = Arg(argc, argv, "--ids"), out = Arg(argc, argv, "--out");
	const std::string other = Arg(argc, argv, "--other"), foreign = Arg(argc, argv, "--foreign");
	CHECK_MSG(!tag.empty() && !ids_path.empty() && !out.empty(), "pass --tag=0p5|1p5, --ids=FILE and --out=DIR");
	if (tag.empty() || ids_path.empty() || out.empty()) return;
	std::vector<int32_t> tokens = ReadIds(ids_path);
	CHECK_MSG(static_cast<int64_t>(tokens.size()) >= kLen, "%s has %zu ids; 4.6 needs %lld", ids_path.c_str(), tokens.size(),
	          static_cast<long long>(kLen));
	if (static_cast<int64_t>(tokens.size()) < kLen) return;
	tokens.resize(static_cast<size_t>(kLen));
	const std::string adapter = AdapterTag();
	// The pool: the prefix, a restored copy of the foreign blob, two adopters, and room to spare.
	const uint32_t pages = static_cast<uint32_t>(2 * e.geo.PrefixPages(kLen) + 2 * e.geo.R(kAdopt) + 8);
	Rig rig(e, pages);
	if (!rig.pool || !RequireGpuGeometry(e, rig.pool, ("4.6 " + tag).c_str())) return;

	const auto t0 = std::chrono::steady_clock::now();
	SslmGpuPrefixHandle* p = BatchedPrefix(rig, tokens);
	if (!p) return;
	const std::vector<uint8_t> gpu = SavePrefix(e, p);
	if (gpu.empty()) return;
	const std::string gpu_file = out + "/gpu-" + tag + "-" + adapter + ".spx1";
	CHECK_MSG(WriteBytes(gpu_file, gpu), "4.6: could not write %s", gpu_file.c_str());
	std::printf("4.6 %s: GPU prefix of %lld tokens on %s built in %.0f ms; blob %zu bytes, sha256 %s -> %s\n", tag.c_str(),
	            static_cast<long long>(kLen), adapter.c_str(), Ms(t0, std::chrono::steady_clock::now()), gpu.size(),
	            Sha(gpu).c_str(), gpu_file.c_str());

	// (a) CPU <-> this GPU.
	const auto t1 = std::chrono::steady_clock::now();
	const std::vector<uint8_t> cpu_blob = CpuTwin(e, tokens);
	if (!cpu_blob.empty()) {
		const std::string cpu_file = out + "/cpu-" + tag + ".spx1";
		CHECK_MSG(WriteBytes(cpu_file, cpu_blob), "4.6: could not write %s", cpu_file.c_str());
		std::printf("4.6 %s: CPU twin built in %.0f ms; blob %zu bytes, sha256 %s -> %s\n", tag.c_str(),
		            Ms(t1, std::chrono::steady_clock::now()), cpu_blob.size(), Sha(cpu_blob).c_str(), cpu_file.c_str());
		const Spx1 vc = Spx1Parse(cpu_blob, e.geo);
		CHECK_MSG(vc.backend == 0 && vc.vendor == 0 && vc.device == 0, "4.6 (a) %s: the CPU blob's producer is backend %u, "
		          "adapter %#x:%#x", tag.c_str(), vc.backend, vc.vendor, vc.device);
		const bool same = SameOutsideProducer(e, cpu_blob, gpu, ("4.6 (a) " + tag + " CPU vs GPU " + adapter).c_str());
		std::printf("4.6 (a) %s: CPU vs GPU %s: %s\n", tag.c_str(), adapter.c_str(), same ? "IDENTICAL" : "DIFFERENT (finding)");
	}

	// (b) this card <-> the other card.
	if (!other.empty()) {
		std::vector<uint8_t> ob;
		CHECK_MSG(ReadBytes(other, &ob) && !ob.empty(), "4.6 (b): %s is missing", other.c_str());
		if (!ob.empty()) {
			const Spx1 vo = Spx1Parse(ob, e.geo), vg = Spx1Parse(gpu, e.geo);
			CHECK_MSG(vo.backend == 1 && (vo.vendor != vg.vendor || vo.device != vg.device),
			          "4.6 (b) %s: %s is not another card's GPU blob (backend %u, adapter %04x:%04x; this card %04x:%04x)",
			          tag.c_str(), other.c_str(), vo.backend, vo.vendor, vo.device, vg.vendor, vg.device);
			char ids[32];
			std::snprintf(ids, sizeof ids, "%04x_%04x", vo.vendor, vo.device);
			const bool same = SameOutsideProducer(e, ob, gpu, ("4.6 (b) " + tag + " " + ids + " vs " + adapter).c_str());
			std::printf("4.6 (b) %s: %s vs %s: %s\n", tag.c_str(), ids, adapter.c_str(), same ? "IDENTICAL" : "DIFFERENT (finding)");
		}
	}

	// (c) the other card's blob restored here through the seam: adopters equal this card's prefill twin's.
	if (!foreign.empty()) {
		std::vector<uint8_t> fb;
		CHECK_MSG(ReadBytes(foreign, &fb) && !fb.empty(), "4.6 (c): %s is missing", foreign.c_str());
		if (fb.empty()) return;
		const Spx1 vf = Spx1Parse(fb, e.geo), vg = Spx1Parse(gpu, e.geo);
		CHECK_MSG(vf.backend == 1 && (vf.vendor != vg.vendor || vf.device != vg.device) && vf.length == kLen,
		          "4.6 (c) %s: %s is not another card's GPU blob of L = %lld", tag.c_str(), foreign.c_str(),
		          static_cast<long long>(kLen));
		// Control: without the seam the producer step refuses it (PC-5), so the seam is what lets it in.
		Refused(RestorePrefix(rig, fb), kProducerMismatch, "4.6 (c) " + tag + " control: another card's blob without the seam");
		SetGpuPrefixRestoreSkipAdapterCheckForTest(true);
		const RestoredPrefix r = RestorePrefix(rig, fb);
		SetGpuPrefixRestoreSkipAdapterCheckForTest(false);
		CHECK_MSG(r.status == GS::SSLM_OK && r.p, "4.6 (c) %s: the other card's blob through the seam: %s", tag.c_str(),
		          Name(r.status));
		// The seam is off again: the same blob is refused.
		Refused(RestorePrefix(rig, fb), kProducerMismatch, "4.6 (c) " + tag + " control: the seam cleared");
		if (!r.p) return;
		const std::vector<int32_t> twin = AdopterOutcomes(rig, p, kLen, kAdopt, static_cast<int32_t>(kAdopt + 16));
		const std::vector<int32_t> got = AdopterOutcomes(rig, r.p, kLen, kAdopt, static_cast<int32_t>(kAdopt + 16));
		CHECK_MSG(AllTokens(twin) && twin.size() == static_cast<size_t>(kAdopt),
		          "4.6 (c) %s: the prefill twin's adopter did not decode %lld tokens", tag.c_str(), static_cast<long long>(kAdopt));
		size_t first = 0;
		while (first < twin.size() && first < got.size() && twin[first] == got[first]) ++first;
		CHECK_MSG(got == twin, "4.6 (c) %s: the adopter of the %04x:%04x blob differs from this card's prefill twin from "
		          "token %zu (a finding)", tag.c_str(), vf.vendor, vf.device, first);
		std::printf("4.6 (c) %s: %04x:%04x's blob restored on %s through the seam; %lld adopter tokens %s the prefill twin's\n",
		            tag.c_str(), vf.vendor, vf.device, adapter.c_str(), static_cast<long long>(kAdopt),
		            got == twin ? "EQUAL" : "DIFFER FROM (finding)");
	}
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		const std::string model = Arg(argc, argv, "--model");
		CHECK_MSG(!model.empty(), "pass --model=FILE (the real 0.5B or 1.5B artifact)");
		Env e;
		if (!model.empty() && e.Open(model, false)) Cell(e, argc, argv);
	}
	return FinishSuite("pcg_cell_4_6");
}
