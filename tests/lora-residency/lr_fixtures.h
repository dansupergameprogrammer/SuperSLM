// Streamed adapters (plan: superslm-lora-residency rev 1.2, §7 "Fixtures"): the L1 fixture writer.
// Synthetic adapters over a generated paged-KV fixture (tools/gen_paged_kv_fixture.py), built field
// by field from ADP1's layout (docs/sslm_format.md) and the manifest writers of
// tests/sslm_model_hostile_fixtures.h, never by calling the code under test. Each adapter's ADP1
// fields and expected packed sizes are recorded here by the writer, so the info cells grade the
// engine against the writer, not against itself.
//
// The plan names S-a and S-b at 0.5B; the cloud has no 0.5B artifact, so the shapes are written at
// pkv_def's geometry (2 layers, hidden 192, 4 query and 2 KV heads of 48, intermediate 512). What
// the cells grade (rank, mask, coverage, byte formulas, bounds) depends on the shape, not the size.
#ifndef SUPERSLM_TESTS_LORA_RESIDENCY_LR_FIXTURES_H
#define SUPERSLM_TESTS_LORA_RESIDENCY_LR_FIXTURES_H

#include "pkv_common.h"
#include "sslm_model_hostile_fixtures.h"
#include "superslm/artifact.h"
#include "superslm/model.h"

#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace lr {

using namespace pkv;

// ADP1's mask bit order (tools/sslm_convert_adapter.py PEFT_ADAPTABLE_PROJECTIONS, model.h's
// kPeftAdaptableProjections): q,o,gate,up,down,k,v.
enum Proj : uint32_t { kQ = 0, kO, kGate, kUp, kDown, kK, kV };
inline const char* const kProjName[7] = {"q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "k_proj", "v_proj"};

struct Geo {
	uint32_t layers, hidden, q_width, kv_width, inter;
};
inline Geo GeoOf(const Fixture& fx) {
	return {fx.geo.layers, 192u, 4u * fx.geo.head_dim, fx.geo.kv_heads * fx.geo.head_dim, 512u};
}
inline uint32_t InOf(const Geo& g, uint32_t p) {
	return p == kDown ? g.inter : p == kO ? g.q_width : g.hidden;
}
inline uint32_t OutOf(const Geo& g, uint32_t p) {
	if (p == kQ) return g.q_width;
	if (p == kO || p == kDown) return g.hidden;
	if (p == kK || p == kV) return g.kv_width;
	return g.inter;
}

struct AdapterSpec {
	std::string label;
	uint32_t rank = 8;
	uint32_t mask = 0;
	uint32_t version = 1;
	double alpha = 16.0;
	uint32_t rslora = 0;
	std::string name = "lr-adapter";
	// (layer, proj) pairs covered; empty with a nonzero mask means every layer for every mask bit.
	std::vector<std::pair<uint32_t, uint32_t>> cover;
	bool all_layers = true;
	uint32_t seed = 1;
	// Hostile knobs.
	bool foreign_base = false;           // ADP1 names another base
	std::vector<superslm_test::FixtureSection> extra_sections;
	uint32_t extra_weight_tensors = 0;   // unused tensors appended to Weights
	uint32_t extra_fold_entries = 0;     // unused-name entries appended to DFS1 (each rejected by populate)
};

// What the writer knows about what it wrote.
struct WrittenAdapter {
	AdapterSpec spec;
	std::vector<uint8_t> bytes;
	std::array<uint8_t, 32> base_hash{};
	std::array<uint8_t, 32> artifact_hash{};
	uint32_t covered_layers = 0;
	uint64_t ab_bytes = 0;    // Σ covered rank × (in + out)
	uint64_t fold_bytes = 0;  // Σ covered 12 × (out + rank)
};

inline std::vector<std::pair<uint32_t, uint32_t>> Coverage(const AdapterSpec& s, const Geo& g) {
	if (!s.cover.empty() || !s.all_layers) return s.cover;
	std::vector<std::pair<uint32_t, uint32_t>> c;
	for (uint32_t l = 0; l < g.layers; ++l)
		for (uint32_t p = 0; p < 7; ++p)
			if (s.mask & (1u << p)) c.push_back({l, p});
	return c;
}

inline std::array<uint8_t, 32> FixtureHash(const Fixture& fx) {
	std::array<uint8_t, 32> h{};
	if (fx.size >= 64) std::memcpy(h.data(), fx.bytes.p + superslm::kIntegrityHashOffset, 32);
	return h;
}

inline WrittenAdapter WriteAdapter(const Fixture& fx, const AdapterSpec& spec) {
	using namespace superslm_test;
	WrittenAdapter w;
	w.spec = spec;
	const Geo g = GeoOf(fx);
	superslm::SslmArtifact art;
	superslm::SslmError err;
	if (superslm::SslmArtifact::OpenFromMemory(fx.bytes.p, fx.size, art, &err) != superslm::SslmStatus::Ok) {
		PKV_CHECK_MSG(false, "lr fixture writer: %s does not open", fx.stem.c_str());
		return w;
	}
	const superslm::SslmSectionView* cfg = art.Section(superslm::SslmSectionType::Config);
	const superslm::SslmSectionView* sil = art.Section(superslm::SslmSectionType::SigmoidLut);
	w.base_hash = FixtureHash(fx);

	std::vector<uint8_t> adp = {'A', 'D', 'P', '1'};
	WriteU32LE(adp, spec.version);
	WriteU32LE(adp, spec.rank);
	WriteU32LE(adp, spec.mask);
	std::array<uint8_t, 32> declared = w.base_hash;
	if (spec.foreign_base) declared[0] ^= 0x5a;
	adp.insert(adp.end(), declared.begin(), declared.end());
	uint64_t bits = 0;
	std::memcpy(&bits, &spec.alpha, 8);
	WriteU64LE(adp, bits);
	WriteU32LE(adp, spec.rslora);
	WriteU32LE(adp, 0);
	WriteU32LE(adp, static_cast<uint32_t>(spec.name.size()));
	WriteBytes(adp, spec.name);

	const auto cover = Coverage(spec, g);
	std::vector<ManifestTensorSpec> wt, dt, ut;
	std::vector<bool> layer_covered(g.layers, false);
	for (const auto& [l, p] : cover) {
		const std::string n = "layer" + std::to_string(l) + "." + kProjName[p];
		const uint32_t in = InOf(g, p), out = OutOf(g, p);
		wt.push_back({n + ".lora_A", {spec.rank, in}});
		wt.push_back({n + ".lora_B", {out, spec.rank}});
		dt.push_back({n, {out * 3}});
		ut.push_back({n, {spec.rank * 3}});
		w.ab_bytes += uint64_t(spec.rank) * (in + out);
		w.fold_bytes += 12ull * (out + spec.rank);
		if (l < g.layers) layer_covered[l] = true;
	}
	for (uint32_t i = 0; i < spec.extra_weight_tensors; ++i) wt.push_back({"unused" + std::to_string(i), {1}});
	for (uint32_t i = 0; i < spec.extra_fold_entries; ++i) dt.push_back({"layer0.zz" + std::to_string(i), {3}});
	for (bool c : layer_covered) w.covered_layers += c ? 1 : 0;

	BuiltManifest wm = BuildManifest(reinterpret_cast<const uint8_t*>("WGT1"), 1, wt);
	BuiltManifest dm = BuildManifest(superslm::kDeltaFoldScalesMagic, 4, dt);
	BuiltManifest um = BuildManifest(superslm::kUFoldScalesMagic, 4, ut);
	// Weights: small deterministic int8 codes. Fold rows: identity triples {1, 0, 0}.
	uint32_t x = spec.seed * 2654435761u + 1;
	for (size_t t = 0; t < wm.tensor_data_off.size(); ++t)
		for (uint64_t i = 0; i < wm.tensor_elem_count[t]; ++i) {
			x = x * 1664525u + 1013904223u;
			wm.bytes[wm.tensor_data_off[t] + i] = static_cast<uint8_t>(static_cast<int8_t>((x >> 24) % 9) - 4);
		}
	for (BuiltManifest* m : {&dm, &um})
		for (size_t t = 0; t < m->tensor_data_off.size(); ++t)
			for (uint64_t r = 0; r < m->tensor_elem_count[t] / 3; ++r) {
				const int32_t triple[3] = {1, 0, 0};
				std::memcpy(m->bytes.data() + m->tensor_data_off[t] + r * 12, triple, 12);
			}
	auto copy = [](superslm::SslmSectionType t, const superslm::SslmSectionView* v) {
		return MakeSection(t, v->dtype, std::vector<uint8_t>(v->data, v->data + v->byte_size));
	};
	std::vector<FixtureSection> sections = {
	    copy(superslm::SslmSectionType::Config, cfg),
	    copy(superslm::SslmSectionType::SigmoidLut, sil),
	    MakeSection(superslm::SslmSectionType::Provenance, superslm::SslmDtype::Raw, adp),
	    MakeSection(superslm::SslmSectionType::Weights, superslm::SslmDtype::Int8, wm.bytes),
	    MakeSection(superslm::SslmSectionType::DeltaFoldScales, superslm::SslmDtype::Int32, dm.bytes),
	    MakeSection(superslm::SslmSectionType::UFoldScales, superslm::SslmDtype::Int32, um.bytes)};
	for (const FixtureSection& e : spec.extra_sections) sections.push_back(e);
	w.bytes = BuildArtifact(sections).bytes;
	std::memcpy(w.artifact_hash.data(), w.bytes.data() + superslm::kIntegrityHashOffset, 32);
	return w;
}

// Recomputes the integrity hash after a mutation (the container's own rule: SHA-256 of the file
// with the 32 hash bytes zeroed).
inline void Rehash(std::vector<uint8_t>& b) {
	std::memset(b.data() + superslm::kIntegrityHashOffset, 0, 32);
	uint8_t d[32];
	superslm::Sha256Hash(b.data(), b.size(), d);
	std::memcpy(b.data() + superslm::kIntegrityHashOffset, d, 32);
}

// The plan's synthetic set (§7), at pkv_def's geometry.
inline AdapterSpec SpecA() {  // S-a: rank 8, q and v
	AdapterSpec s;
	s.label = "S-a";
	s.rank = 8;
	s.mask = (1u << kQ) | (1u << kV);
	s.name = "S-a";
	s.seed = 11;
	return s;
}
inline AdapterSpec SpecB() {  // S-b: rank 16, all seven projections
	AdapterSpec s;
	s.label = "S-b";
	s.rank = 16;
	s.mask = 0x7f;
	s.alpha = 32.0;
	s.rslora = 1;
	s.name = "S-b, an adapter whose name runs well past the sixty-three bytes the info struct keeps";
	s.seed = 12;
	return s;
}
inline AdapterSpec SpecMask0() {  // S-mask0: mask 0, no entries
	AdapterSpec s;
	s.label = "S-mask0";
	s.rank = 4;
	s.mask = 0;
	s.name = "S-mask0";
	return s;
}
inline AdapterSpec SpecR1() {  // S-r1: rank 1, layer 0's q only
	AdapterSpec s;
	s.label = "S-r1";
	s.rank = 1;
	s.mask = 1u << kQ;
	s.all_layers = false;
	s.cover = {{0, kQ}};
	s.name = "S-r1";
	s.alpha = 1.0;
	return s;
}
inline std::vector<AdapterSpec> SyntheticSet() { return {SpecA(), SpecB(), SpecMask0(), SpecR1()}; }

}  // namespace lr

#endif  // SUPERSLM_TESTS_LORA_RESIDENCY_LR_FIXTURES_H
