// Streamed adapters (plan: superslm-lora-residency rev 1.2, §7, steps L1/L2): the engine cells the
// cloud can run. CPU: adapter info, the map's peak bound, the bind lock, lifetime. Device-free GPU
// half: the prepare core (src/adapter_core.h) that sslm_gpu_adapter_prepare wraps, its info fields,
// packed bytes and peak bound. The GPU verbs themselves (prepare, map_prepared, the mid-token
// guard, the GPU statuses) are box cells.
//
// Graded by the writer's records (lr_fixtures.h), statuses written in the test, and the allocation
// seam (lr_alloc_seam.h). Red on the base (main at 1.15.0): the new verbs are stubs returning a
// status outside the list, the core header does not exist, and the base's map accepts what the
// prescreen refuses.
#include "lr_alloc_seam.h"
#include "lr_fixtures.h"

#include <atomic>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <thread>

#if __has_include("adapter_core.h")
#include "adapter_core.h"
#define LR_HAVE_CORE 1
#else
#define LR_HAVE_CORE 0
#endif

namespace {
constexpr int kUnbuilt = 99;
}

#ifndef SSLM_HAS_ADAPTER_INFO_ABI
// The red suite's stubs: the declarations the plan publishes (§3.1), returning the unbuilt status.
typedef struct sslm_adapter_info_s {
	uint32_t struct_size, adp1_version, rank, target_modules_mask, covered_layers, use_rslora;
	double lora_alpha;
	uint8_t base_artifact_hash[32], artifact_hash[32];
	uint64_t host_resident_bytes, device_resident_bytes, upload_staging_bytes;
	char name[64];
} sslm_adapter_info;
#define SSLM_ADAPTER_INFO_V1_SIZE ((uint32_t)sizeof(sslm_adapter_info))
extern "C" {
sslm_status sslm_adapter_info_get(sslm_adapter, sslm_adapter_info*) { return static_cast<sslm_status>(kUnbuilt); }
sslm_status sslm_adapter_map_peak_bound(sslm_model, uint64_t, uint64_t*) { return static_cast<sslm_status>(kUnbuilt); }
}
#endif

namespace {
using namespace lr;

const Fixture& Def() { return GetFixture("pkv_def"); }

void Status(sslm_status got, sslm_status want, const char* what) {
	PKV_CHECK_MSG(got == want, "%s: got %d (%s), want %d (%s)", what, static_cast<int>(got), AbiStatusName(got),
	              static_cast<int>(want), AbiStatusName(want));
}

sslm_adapter Map(const Fixture& fx, const std::vector<uint8_t>& b, sslm_status want = SSLM_OK, const char* what = "map") {
	sslm_adapter a = nullptr;
	Status(sslm_adapter_map(b.data(), b.size(), fx.model, &a), want, what);
	return a;
}

sslm_adapter_info Info(sslm_adapter a, sslm_status want = SSLM_OK) {
	sslm_adapter_info i;
	std::memset(&i, 0, sizeof(i));
	i.struct_size = sizeof(i);
	Status(sslm_adapter_info_get(a, &i), want, "sslm_adapter_info_get");
	return i;
}

uint64_t Bound(const Fixture& fx, uint64_t size) {
	uint64_t b = 0;
	Status(sslm_adapter_map_peak_bound(fx.model, size, &b), SSLM_OK, "sslm_adapter_map_peak_bound");
	return b;
}

std::string Truncated63(const std::string& s) { return s.substr(0, 63); }

std::vector<int32_t> Tokens(uint32_t seed, int n) {
	std::vector<int32_t> t;
	uint32_t x = seed * 747796405u + 1;
	for (int i = 0; i < n; ++i) {
		x = x * 1664525u + 1013904223u;
		t.push_back(static_cast<int32_t>((x >> 16) % 250) + 3);
	}
	return t;
}

// ---- the seam's own must-reject (§8) -------------------------------------------------------------

void CellSeam() {
	int64_t net = 0, allocs = 0;
	{
		lr_seam::Scope s;
		volatile char* p = new char[1];
		p[0] = 1;
		net = s.NetLive();
		allocs = s.Allocations();
		delete[] p;
		PKV_CHECK_EQ(s.NetLive(), 0);
		PKV_CHECK_EQ(s.PeakAboveEntry(), 1);
	}
	PKV_CHECK_EQ(net, 1);
	PKV_CHECK_EQ(allocs, 1);
	const int64_t before = lr_seam::Count();
	volatile char* q = new char[1];
	delete[] q;
	PKV_CHECK_MSG(lr_seam::Count() == before, "an allocation on a disarmed thread was counted");
}
PKV_CELL("0.1/L1:seam", "L1", CellSeam);

// ---- 2.1: info on the CPU, against the writer's records ------------------------------------------

void CellCpuInfo() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	for (const AdapterSpec& spec : SyntheticSet()) {
		const WrittenAdapter w = WriteAdapter(fx, spec);
		sslm_adapter a = Map(fx, w.bytes, SSLM_OK, spec.label.c_str());
		if (!a) continue;
		const sslm_adapter_info i = Info(a);
		PKV_CHECK_EQ(i.struct_size, sizeof(sslm_adapter_info));
		PKV_CHECK_EQ(i.adp1_version, 1u);
		PKV_CHECK_EQ(i.rank, spec.rank);
		PKV_CHECK_EQ(i.target_modules_mask, spec.mask);
		PKV_CHECK_EQ(i.covered_layers, w.covered_layers);
		PKV_CHECK_EQ(i.use_rslora, spec.rslora);
		PKV_CHECK_MSG(i.lora_alpha == spec.alpha, "%s alpha %f != %f", spec.label.c_str(), i.lora_alpha, spec.alpha);
		PKV_CHECK_MSG(std::memcmp(i.base_artifact_hash, w.base_hash.data(), 32) == 0, "%s base hash", spec.label.c_str());
		PKV_CHECK_MSG(std::memcmp(i.artifact_hash, w.artifact_hash.data(), 32) == 0, "%s artifact hash", spec.label.c_str());
		PKV_CHECK_MSG(std::string(i.name) == Truncated63(spec.name) && i.name[63] == '\0', "%s name \"%.64s\"",
		              spec.label.c_str(), i.name);
		PKV_CHECK_EQ(i.device_resident_bytes, 0u);
		PKV_CHECK_EQ(i.upload_staging_bytes, 0u);
		PKV_CHECK_MSG(i.host_resident_bytes >= w.bytes.size(), "%s host bytes %llu below the artifact's %zu",
		              spec.label.c_str(), static_cast<unsigned long long>(i.host_resident_bytes), w.bytes.size());
		Status(sslm_adapter_release(a), SSLM_OK, "release");
	}
}
PKV_CELL("2.1/L2:cpu-info", "L2", CellCpuInfo);

void CellCpuInfoRefusals() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const WrittenAdapter w = WriteAdapter(fx, SpecA());
	sslm_adapter a = Map(fx, w.bytes);
	if (!a) return;
	sslm_adapter_info i;
	std::memset(&i, 0, sizeof(i));
	i.struct_size = sizeof(i);
	Status(sslm_adapter_info_get(nullptr, &i), SSLM_INVALID_ARGUMENT, "info(null adapter)");
	Status(sslm_adapter_info_get(a, nullptr), SSLM_INVALID_ARGUMENT, "info(null out)");
	for (uint32_t size : {0u, SSLM_ADAPTER_INFO_V1_SIZE - 1}) {
		std::memset(&i, 0xcd, sizeof(i));
		i.struct_size = size;
		Status(sslm_adapter_info_get(a, &i), SSLM_INVALID_ARGUMENT, "info(short struct_size)");
		PKV_CHECK_MSG(i.rank == 0xcdcdcdcdu, "a refused info call wrote the struct (struct_size %u)", size);
	}
	// A larger struct_size is accepted; the tail past v1 is left as the caller set it.
	struct Wide {
		sslm_adapter_info v1;
		uint8_t tail[32];
	} wide;
	std::memset(&wide, 0xab, sizeof(wide));
	wide.v1.struct_size = sizeof(wide);
	Status(sslm_adapter_info_get(a, &wide.v1), SSLM_OK, "info(larger struct_size)");
	bool canary = true;
	for (uint8_t b : wide.tail) canary = canary && b == 0xab;
	PKV_CHECK_MSG(canary, "info wrote past the v1 struct");
	PKV_CHECK_EQ(wide.v1.struct_size, sizeof(wide));
	PKV_CHECK_EQ(wide.v1.rank, 8u);
	uint64_t b = 0;
	Status(sslm_adapter_map_peak_bound(nullptr, 100, &b), SSLM_INVALID_ARGUMENT, "peak_bound(null base)");
	Status(sslm_adapter_map_peak_bound(fx.model, 100, nullptr), SSLM_INVALID_ARGUMENT, "peak_bound(null out)");
	Status(sslm_adapter_release(a), SSLM_OK, "release");
}
PKV_CELL("2.1/L2:cpu-info-refusals", "L2", CellCpuInfoRefusals);

// ---- 2.2 and 9.4: hostile inputs to the CPU map (statuses as main gives them) and to the prepare core

struct Hostile {
	std::string label;
	std::vector<uint8_t> bytes;
	sslm_status cpu;  // sslm_adapter_map, written here (main's, which the red run on main confirms)
	int core;         // 0 ok, 1 rejected, 2 base mismatch
};

std::vector<Hostile> HostileSet(const Fixture& fx) {
	std::vector<Hostile> h;
	const WrittenAdapter base = WriteAdapter(fx, SpecB());
	const std::vector<uint8_t>& b = base.bytes;
	h.push_back({"truncated-0", {}, SSLM_ARTIFACT_REJECTED, 1});
	h.push_back({"truncated-64", std::vector<uint8_t>(b.begin(), b.begin() + 64), SSLM_ARTIFACT_REJECTED, 1});
	h.push_back({"truncated-half", std::vector<uint8_t>(b.begin(), b.begin() + b.size() / 2), SSLM_ARTIFACT_REJECTED, 1});
	{
		std::vector<uint8_t> f = b;
		f[f.size() - 7] ^= 0x10;  // a payload byte; the integrity hash no longer matches
		h.push_back({"flipped-byte", f, SSLM_ARTIFACT_REJECTED, 1});
	}
	{
		AdapterSpec s = SpecA();
		s.mask |= 1u << 9;
		h.push_back({"unknown-mask-bit", WriteAdapter(fx, s).bytes, SSLM_ARTIFACT_REJECTED, 1});
	}
	{
		AdapterSpec s = SpecA();
		s.version = 2;
		h.push_back({"adp1-version-2", WriteAdapter(fx, s).bytes, SSLM_ARTIFACT_REJECTED, 1});
	}
	{
		AdapterSpec s = SpecA();
		s.foreign_base = true;
		h.push_back({"other-base", WriteAdapter(fx, s).bytes, SSLM_ADAPTER_MODEL_MISMATCH, 2});
	}
	{
		AdapterSpec s = SpecR1();
		s.rank = 0;
		s.mask = 0;  // a rank-0 adapter can carry no tensor (a zero shape is refused)
		s.cover = {};
		h.push_back({"rank-0", WriteAdapter(fx, s).bytes, SSLM_OK, 0});
	}
	h.push_back({"model-artifact", std::vector<uint8_t>(fx.bytes.p, fx.bytes.p + fx.size), SSLM_ARTIFACT_REJECTED, 1});
	return h;
}

void CellHostile() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	for (Hostile& c : HostileSet(fx)) {
		sslm_adapter a = nullptr;
		const sslm_status st = sslm_adapter_map(c.bytes.empty() ? reinterpret_cast<const uint8_t*>("") : c.bytes.data(),
		                                        c.bytes.size(), fx.model, &a);
		PKV_CHECK_MSG(st == c.cpu, "%s: sslm_adapter_map %d (%s), want %d (%s)", c.label.c_str(), static_cast<int>(st),
		              AbiStatusName(st), static_cast<int>(c.cpu), AbiStatusName(c.cpu));
		if (st != SSLM_OK) PKV_CHECK_MSG(a == nullptr, "%s: refused map wrote a handle", c.label.c_str());
		if (a) sslm_adapter_release(a);
#if LR_HAVE_CORE
		superslm_adapter::BaseModelGeometry geo;
		const Geo g = GeoOf(fx);
		geo.num_hidden_layers = g.layers;
		geo.hidden_size = g.hidden;
		geo.intermediate_size = g.inter;
		geo.kv_hidden_size = g.kv_width;
		geo.q_width = g.q_width;
		geo.base_artifact_hash = FixtureHash(fx);
		superslm_adapter::PreparedAdapterCore core;
		const auto r = superslm_adapter::PrepareAdapterCore(geo, c.bytes.data(), c.bytes.size(), core);
		const int got = r == superslm_adapter::PrepareResult::kOk ? 0
		                : r == superslm_adapter::PrepareResult::kRejected ? 1 : 2;
		PKV_CHECK_MSG(got == c.core, "%s: prepare core %d, want %d", c.label.c_str(), got, c.core);
#else
		PKV_CHECK_MSG(false, "%s: the prepare core (src/adapter_core.h) is not built", c.label.c_str());
#endif
	}
}
PKV_CELL("2.2/L2:hostile", "L2", CellHostile);

// The prescreen (§3.1 as built): what the map now refuses before parsing, each with the accepted
// neighbour that shows the limit is exactly where docs/api.md says.
void CellPrescreen() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	using superslm_test::MakeSection;
	auto check = [&](const char* label, const AdapterSpec& s, sslm_status want) {
		const WrittenAdapter w = WriteAdapter(fx, s);
		sslm_adapter a = Map(fx, w.bytes, want, label);
		if (a) sslm_adapter_release(a);
	};
	{
		AdapterSpec s = SpecA();
		s.extra_sections.push_back(MakeSection(superslm::SslmSectionType::Biases, superslm::SslmDtype::Int32,
		                                       superslm_test::BuildManifest(reinterpret_cast<const uint8_t*>("BIA1"), 4,
		                                                                    {{"b0", {4}}})
		                                           .bytes));
		check("seventh section (Biases)", s, SSLM_ARTIFACT_REJECTED);
	}
	{
		AdapterSpec s = SpecB();  // full coverage: 14 per layer, the cap
		check("weights at the cap", s, SSLM_OK);
		s.extra_weight_tensors = 1;
		check("weights one over the cap", s, SSLM_ARTIFACT_REJECTED);
	}
	{
		AdapterSpec s = SpecA();
		s.name = std::string(4096, 'n');
		check("name of 4096 bytes", s, SSLM_OK);
		s.name = std::string(4097, 'n');
		check("name of 4097 bytes", s, SSLM_ARTIFACT_REJECTED);
	}
	{
		AdapterSpec s = SpecA();  // 2 per layer of DFS1, cap 7: five unused entries more is at the cap
		s.extra_fold_entries = 7 * GeoOf(fx).layers - 2 * GeoOf(fx).layers + 1;
		check("fold entries one over the cap", s, SSLM_ARTIFACT_REJECTED);
	}
}
PKV_CELL("2.2b/L2:prescreen", "L2", CellPrescreen);

// ---- 4.1: bytes ------------------------------------------------------------------------------------

void CellCpuBytes() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	for (const AdapterSpec& spec : SyntheticSet()) {
		const WrittenAdapter w = WriteAdapter(fx, spec);
		sslm_adapter a = nullptr;
		int64_t owned = 0;
		sslm_status st;
		{
			lr_seam::Scope s;
			st = sslm_adapter_map(w.bytes.data(), w.bytes.size(), fx.model, &a);
			owned = s.NetLive();
		}
		Status(st, SSLM_OK, spec.label.c_str());
		if (!a) continue;
		const sslm_adapter_info i = Info(a);
		PKV_CHECK_MSG(static_cast<int64_t>(i.host_resident_bytes) == owned,
		              "%s: host_resident_bytes %llu, the seam's bytes the handle owns %lld", spec.label.c_str(),
		              static_cast<unsigned long long>(i.host_resident_bytes), static_cast<long long>(owned));
		const int64_t live = lr_seam::Live();
		Status(sslm_adapter_release(a), SSLM_OK, "release");
		PKV_CHECK_MSG(lr_seam::Live() == live - owned, "%s: release freed %lld counted bytes, the map left %lld",
		              spec.label.c_str(), static_cast<long long>(live - lr_seam::Live()), static_cast<long long>(owned));
	}
}
PKV_CELL("4.1/L2:cpu-bytes", "L2", CellCpuBytes);

#if LR_HAVE_CORE
superslm_adapter::BaseModelGeometry CoreGeo(const Fixture& fx) {
	superslm_adapter::BaseModelGeometry geo;
	const Geo g = GeoOf(fx);
	geo.num_hidden_layers = g.layers;
	geo.hidden_size = g.hidden;
	geo.intermediate_size = g.inter;
	geo.kv_hidden_size = g.kv_width;
	geo.q_width = g.q_width;
	geo.base_artifact_hash = FixtureHash(fx);
	return geo;
}
#endif

void CellCoreBytesAndInfo() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
#if LR_HAVE_CORE
	for (const AdapterSpec& spec : SyntheticSet()) {
		const WrittenAdapter w = WriteAdapter(fx, spec);
		superslm_adapter::PreparedAdapterCore* core = nullptr;
		superslm_adapter::PrepareResult r;
		int64_t owned = 0;
		{
			lr_seam::Scope s;
			core = new superslm_adapter::PreparedAdapterCore();
			r = superslm_adapter::PrepareAdapterCore(CoreGeo(fx), w.bytes.data(), w.bytes.size(), *core);
			owned = s.NetLive();
		}
		PKV_CHECK_MSG(r == superslm_adapter::PrepareResult::kOk, "%s: prepare core refused", spec.label.c_str());
		const uint64_t want_ab = w.ab_bytes ? w.ab_bytes : 1, want_fold = w.fold_bytes ? w.fold_bytes : 1;
		PKV_CHECK_EQ(core->ab_bytes.size(), want_ab);
		PKV_CHECK_EQ(core->fold_bytes.size(), want_fold);
		// The prepared object's own bytes: no parsed view, no artifact copy (11.2's mutant keeping the
		// view would add the artifact's size here).
		const uint64_t heap = superslm_adapter::PreparedCoreHeapBytes(*core) + sizeof(superslm_adapter::PreparedAdapterCore);
		PKV_CHECK_MSG(static_cast<int64_t>(heap) == owned, "%s: prepared bytes %llu, the seam's %lld", spec.label.c_str(),
		              static_cast<unsigned long long>(heap), static_cast<long long>(owned));
		PKV_CHECK_MSG(heap < w.bytes.size() + 4096 || w.ab_bytes + w.fold_bytes + 4096 > heap,
		              "%s: the prepared object holds more than its packed bytes", spec.label.c_str());
		// Info fields, from the core, against the writer.
		sslm_adapter_info i;
		std::memset(&i, 0, sizeof(i));
		i.struct_size = sizeof(i);
		superslm_adapter::FillAdapterInfoCommon(core->meta, core->covered_layers, core->artifact_hash, &i);
		PKV_CHECK_EQ(i.rank, spec.rank);
		PKV_CHECK_EQ(i.target_modules_mask, spec.mask);
		PKV_CHECK_EQ(i.covered_layers, w.covered_layers);
		PKV_CHECK_MSG(std::memcmp(i.artifact_hash, w.artifact_hash.data(), 32) == 0, "%s artifact hash", spec.label.c_str());
		PKV_CHECK_MSG(std::string(i.name) == Truncated63(spec.name), "%s name", spec.label.c_str());
		// Predicted device bytes: each buffer rounded up to 64 KiB (§3.1).
		const uint64_t predicted = superslm_adapter::RoundUp64KiB(core->ab_bytes.size()) +
		                           superslm_adapter::RoundUp64KiB(core->fold_bytes.size());
		PKV_CHECK_EQ(predicted, (want_ab + 65535) / 65536 * 65536 + (want_fold + 65535) / 65536 * 65536);
		// Slots: every covered (layer, projection) present at the writer's shape; nothing else.
		uint32_t present = 0;
		for (const auto& layer : core->slots)
			for (const auto& slot : layer) present += slot.present ? 1u : 0u;
		PKV_CHECK_EQ(present, static_cast<uint32_t>(Coverage(spec, GeoOf(fx)).size()));
		delete core;
	}
#else
	PKV_CHECK_MSG(false, "the prepare core (src/adapter_core.h) is not built");
#endif
}
PKV_CELL("4.1/L2:prepare-core-bytes", "L2", CellCoreBytesAndInfo);

// ---- 7.2's engine half: the bounds hold on every fixture and on the shapes that maximise each term --

struct Shape {
	std::string label;
	std::vector<uint8_t> bytes;
};

std::vector<Shape> BoundShapes(const Fixture& fx) {
	std::vector<Shape> v;
	for (const AdapterSpec& s : SyntheticSet()) v.push_back({s.label, WriteAdapter(fx, s).bytes});
	for (Hostile& h : HostileSet(fx)) v.push_back({"hostile " + h.label, h.bytes});
	{
		AdapterSpec s = SpecB();  // every manifest at its cap, the longest name, a rank-64 table
		s.rank = 64;
		s.name = std::string(4096, 'x');
		v.push_back({"max-entries max-name", WriteAdapter(fx, s).bytes});
	}
	{
		AdapterSpec s = SpecB();  // all but the last pair covered, then a refused entry last
		s.all_layers = false;
		for (uint32_t l = 0; l < GeoOf(fx).layers; ++l)
			for (uint32_t p = 0; p < 7; ++p)
				if (!(l + 1 == GeoOf(fx).layers && p == 6)) s.cover.push_back({l, p});
		s.extra_fold_entries = 1;
		s.name = std::string(4096, 'y');
		v.push_back({"rejection on the last entry", WriteAdapter(fx, s).bytes});
	}
	{
		AdapterSpec s = SpecB();  // weight shape refused on the last tensor checked: rank mismatch
		s.rank = 2;
		WrittenAdapter w = WriteAdapter(fx, s);
		v.push_back({"rank-2 full", w.bytes});
	}
	return v;
}

void CellBoundCpu() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	for (const Shape& sh : BoundShapes(fx)) {
		const uint64_t bound = Bound(fx, sh.bytes.size());
		sslm_adapter a = nullptr;
		int64_t peak = 0;
		{
			lr_seam::Scope s;
			sslm_adapter_map(sh.bytes.empty() ? reinterpret_cast<const uint8_t*>("") : sh.bytes.data(), sh.bytes.size(),
			                 fx.model, &a);
			peak = s.PeakAboveEntry();
		}
		PKV_CHECK_MSG(peak >= 0 && static_cast<uint64_t>(peak) <= bound,
		              "%s (%zu B): map peak %lld above the bound %llu", sh.label.c_str(), sh.bytes.size(),
		              static_cast<long long>(peak), static_cast<unsigned long long>(bound));
		std::printf("  7.2 cpu  %-30s %9zu B  peak %9lld  bound %9llu  slack %lld\n", sh.label.c_str(), sh.bytes.size(),
		            static_cast<long long>(peak), static_cast<unsigned long long>(bound),
		            static_cast<long long>(bound) - static_cast<long long>(peak));
		if (a) sslm_adapter_release(a);
	}
}
PKV_CELL("7.2/L2:bound-cpu", "L2", CellBoundCpu);

void CellBoundPrepare() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
#if LR_HAVE_CORE
	for (const Shape& sh : BoundShapes(fx)) {
		const uint64_t bound =
		    superslm_adapter::AdapterPeakTermsFor(GeoOf(fx).layers, sh.bytes.size(),
		                                          sizeof(superslm_adapter::PreparedAdapterCore), true)
		        .Sum();
		int64_t peak = 0;
		{
			lr_seam::Scope s;
			auto* core = new superslm_adapter::PreparedAdapterCore();
			superslm_adapter::PrepareAdapterCore(CoreGeo(fx), sh.bytes.data(), sh.bytes.size(), *core);
			peak = s.PeakAboveEntry();
			delete core;
		}
		PKV_CHECK_MSG(peak >= 0 && static_cast<uint64_t>(peak) <= bound,
		              "%s (%zu B): prepare peak %lld above the bound %llu", sh.label.c_str(), sh.bytes.size(),
		              static_cast<long long>(peak), static_cast<unsigned long long>(bound));
		std::printf("  7.2 prep %-30s %9zu B  peak %9lld  bound %9llu  slack %lld\n", sh.label.c_str(), sh.bytes.size(),
		            static_cast<long long>(peak), static_cast<unsigned long long>(bound),
		            static_cast<long long>(bound) - static_cast<long long>(peak));
	}
#else
	PKV_CHECK_MSG(false, "the prepare core (src/adapter_core.h) is not built");
#endif
}
PKV_CELL("7.2/L2:bound-prepare", "L2", CellBoundPrepare);

// The per-term mutants (7.2, 11.3), evaluated in place: for each term, the bound without it against
// the peak of the shape that maximises it. The copy term is the one no other term's slack covers, so
// its mutant must be killed; the rest are reported (a survivor is a term smaller than the others'
// combined slack, stated in docs/api.md, not tuned away).
void CellBoundTermMutants() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
#if LR_HAVE_CORE
	AdapterSpec s = SpecB();
	s.rank = 64;
	s.name = std::string(4096, 'x');
	const WrittenAdapter w = WriteAdapter(fx, s);
	int64_t cpu_peak = 0, prep_peak = 0;
	sslm_adapter a = nullptr;
	{
		lr_seam::Scope sc;
		sslm_adapter_map(w.bytes.data(), w.bytes.size(), fx.model, &a);
		cpu_peak = sc.PeakAboveEntry();
	}
	if (a) sslm_adapter_release(a);
	{
		lr_seam::Scope sc;
		auto* core = new superslm_adapter::PreparedAdapterCore();
		superslm_adapter::PrepareAdapterCore(CoreGeo(fx), w.bytes.data(), w.bytes.size(), *core);
		prep_peak = sc.PeakAboveEntry();
		delete core;
	}
	struct Term {
		const char* name;
		uint64_t superslm_adapter::AdapterPeakTerms::*field;
	};
	const Term terms[] = {{"copy", &superslm_adapter::AdapterPeakTerms::artifact_copy},
	                      {"sections", &superslm_adapter::AdapterPeakTerms::section_rows},
	                      {"manifests", &superslm_adapter::AdapterPeakTerms::manifests},
	                      {"name", &superslm_adapter::AdapterPeakTerms::adapter_name},
	                      {"layers", &superslm_adapter::AdapterPeakTerms::layer_table},
	                      {"modules", &superslm_adapter::AdapterPeakTerms::claimed_modules},
	                      {"messages", &superslm_adapter::AdapterPeakTerms::messages},
	                      {"handle", &superslm_adapter::AdapterPeakTerms::handle},
	                      {"packed", &superslm_adapter::AdapterPeakTerms::packed},
	                      {"slots", &superslm_adapter::AdapterPeakTerms::slot_table}};
	for (bool gpu : {false, true}) {
		const auto full = superslm_adapter::AdapterPeakTermsFor(
		    GeoOf(fx).layers, w.bytes.size(), gpu ? sizeof(superslm_adapter::PreparedAdapterCore) : 0, gpu);
		const int64_t peak = gpu ? prep_peak : cpu_peak;
		for (const Term& t : terms) {
			if (full.*t.field == 0) continue;
			auto m = full;
			m.*t.field = 0;
			const bool killed = static_cast<int64_t>(m.Sum()) < peak;
			std::printf("  7.2 mutant %-4s drop %-9s %s\n", gpu ? "prep" : "cpu", t.name, killed ? "killed" : "survives");
			if (std::strcmp(t.name, "copy") == 0 || (gpu && std::strcmp(t.name, "packed") == 0))
				PKV_CHECK_MSG(killed, "the %s bound without its %s term still covers the peak", gpu ? "prepare" : "map",
				              t.name);
		}
	}
#else
	PKV_CHECK_MSG(false, "the prepare core (src/adapter_core.h) is not built");
#endif
}
PKV_CELL("7.2/L2:bound-term-mutants", "L2", CellBoundTermMutants);

// ---- 1.1: lifetime ---------------------------------------------------------------------------------

void CellLifetime() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	// A private mapping of the base, so its release status is this cell's to read.
	AlignedBuf mb;
	mb.Reset(fx.size);
	std::memcpy(mb.p, fx.bytes.p, fx.size);
	sslm_model model = nullptr;
	Status(sslm_model_map(mb.p, fx.size, &model), SSLM_OK, "private model map");
	if (!model) return;
	std::vector<WrittenAdapter> set;
	for (const AdapterSpec& s : SyntheticSet()) set.push_back(WriteAdapter(fx, s));
	{
		LegacyPool pool(model, 2);
		Status(pool.status, SSLM_OK, "pool");
		sslm_seq seq = nullptr;
		Status(sslm_seq_create(model, &pool.pool, &seq), SSLM_OK, "seq");
		if (seq) {
			const int64_t live0 = lr_seam::Live();
			int failures = 0;
			for (int cycle = 0; cycle < 1000 && failures == 0; ++cycle) {
				lr_seam::Arm();
				const WrittenAdapter& w = set[cycle % set.size()];
				sslm_adapter a = nullptr;
				if (sslm_adapter_map(w.bytes.data(), w.bytes.size(), model, &a) != SSLM_OK) ++failures;
				sslm_adapter_info i;
				std::memset(&i, 0, sizeof(i));
				i.struct_size = sizeof(i);
				if (sslm_adapter_info_get(a, &i) != SSLM_OK || i.rank != w.spec.rank) ++failures;
				if (sslm_seq_set_adapter(seq, a) != SSLM_OK) ++failures;
				lr_seam::Disarm();
				if (cycle % 50 == 0) {
					Status(sslm_seq_reset(seq), SSLM_OK, "reset");
					const auto t = Tokens(cycle, 8);
					Status(PrefillAll(model, seq, t, 8), SSLM_OK, "prefill");
					for (int k = 0; k < 4; ++k) PKV_CHECK(NextToken(model, seq) >= 0);
				}
				lr_seam::Arm();
				if (sslm_seq_set_adapter(seq, nullptr) != SSLM_OK) ++failures;
				if (sslm_adapter_release(a) != SSLM_OK) ++failures;
				lr_seam::Disarm();
			}
			PKV_CHECK_EQ(failures, 0);
			PKV_CHECK_MSG(lr_seam::Live() == live0, "1,000 map/info/bind/release cycles left %lld counted bytes",
			              static_cast<long long>(lr_seam::Live() - live0));
			sslm_seq_release(seq);
		}
	}
	Status(sslm_model_unmap(model), SSLM_OK, "the base's release after the cycles");
}
PKV_CELL("1.1/L2:lifetime", "L2", CellLifetime);

// ---- 3.1 and 3.2: concurrency (graded by TSan in the lora-residency TSan build) --------------------

void CellBindLock() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const WrittenAdapter wa = WriteAdapter(fx, SpecA()), wb = WriteAdapter(fx, SpecB());
	sslm_adapter a = Map(fx, wa.bytes), b = Map(fx, wb.bytes);
	if (!a || !b) return;
	LegacyPool pool(fx.model, 2);
	sslm_seq s1 = nullptr, s2 = nullptr;
	Status(sslm_seq_create(fx.model, &pool.pool, &s1), SSLM_OK, "s1");
	Status(sslm_seq_create(fx.model, &pool.pool, &s2), SSLM_OK, "s2");
	if (!s1 || !s2) return;
	Status(sslm_seq_set_adapter(s2, b), SSLM_OK, "s2 bind");
	Status(PrefillAll(fx.model, s2, Tokens(3, 16), 16), SSLM_OK, "s2 prefill");
	std::atomic<int> bad{0};
	auto binder = [&](sslm_adapter x) {
		for (int i = 0; i < 2000; ++i) {
			if (sslm_seq_set_adapter(s1, (i & 1) ? nullptr : x) != SSLM_OK) bad.fetch_add(1);
		}
	};
	std::thread t1(binder, a), t2(binder, b);
	std::thread t3([&] {
		for (int i = 0; i < 24; ++i)
			if (NextToken(fx.model, s2) < 0) bad.fetch_add(1);
	});
	for (int i = 0; i < 20; ++i) {
		sslm_adapter m = nullptr;
		if (sslm_adapter_map(wa.bytes.data(), wa.bytes.size(), fx.model, &m) != SSLM_OK) bad.fetch_add(1);
		if (m) sslm_adapter_release(m);
	}
	t1.join();
	t2.join();
	t3.join();
	PKV_CHECK_EQ(bad.load(), 0);
	Status(sslm_seq_set_adapter(s1, nullptr), SSLM_OK, "s1 unbind");
	Status(sslm_seq_set_adapter(s2, nullptr), SSLM_OK, "s2 unbind");
	// Every bind the two threads made was undone in order: both adapters release.
	Status(sslm_adapter_release(a), SSLM_OK, "release a after the bind race");
	Status(sslm_adapter_release(b), SSLM_OK, "release b after the bind race");
	sslm_seq_release(s1);
	sslm_seq_release(s2);
}
PKV_CELL("3.1/L2:bind-lock", "L2", CellBindLock);

void CellCoreThreads() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
#if LR_HAVE_CORE
	const WrittenAdapter w = WriteAdapter(fx, SpecB());
	const auto geo = CoreGeo(fx);
	superslm_adapter::PreparedAdapterCore single;
	PKV_CHECK(superslm_adapter::PrepareAdapterCore(geo, w.bytes.data(), w.bytes.size(), single) ==
	          superslm_adapter::PrepareResult::kOk);
	std::vector<superslm_adapter::PreparedAdapterCore> out(4);
	std::vector<std::thread> ts;
	for (int t = 0; t < 4; ++t)
		ts.emplace_back([&, t] {
			for (int r = 0; r < 8; ++r) {
				superslm_adapter::PreparedAdapterCore c;
				superslm_adapter::PrepareAdapterCore(geo, w.bytes.data(), w.bytes.size(), c);
				if (r == 7) out[t] = std::move(c);
			}
		});
	for (auto& t : ts) t.join();
	for (const auto& c : out) {
		PKV_CHECK(c.ab_bytes == single.ab_bytes);
		PKV_CHECK(c.fold_bytes == single.fold_bytes);
		PKV_CHECK_EQ(c.slots.size(), single.slots.size());
	}
#else
	PKV_CHECK_MSG(false, "the prepare core (src/adapter_core.h) is not built");
#endif
}
PKV_CELL("3.2/L2:core-threads", "L2", CellCoreThreads);

// ---- 7.7: docs/api.md names each new verb ------------------------------------------------------------

void CellDocs() {
#ifdef LR_SOURCE_DIR
	std::ifstream f(std::string(LR_SOURCE_DIR) + "/docs/api.md");
	std::stringstream ss;
	ss << f.rdbuf();
	const std::string doc = ss.str();
	PKV_CHECK_MSG(!doc.empty(), "docs/api.md unreadable");
	for (const char* verb : {"sslm_adapter_info_get", "sslm_adapter_map_peak_bound", "SSLM_HAS_ADAPTER_INFO_ABI"})
		PKV_CHECK_MSG(doc.find(verb) != std::string::npos, "docs/api.md does not name %s", verb);
#else
	PKV_CHECK_MSG(false, "LR_SOURCE_DIR is not defined");
#endif
}
PKV_CELL("7.7/L2:docs", "L2", CellDocs);

}  // namespace
