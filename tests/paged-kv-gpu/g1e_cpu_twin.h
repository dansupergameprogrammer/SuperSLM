// Paged-KV plan (rev 16.2) step G1, group g1e: the CPU twin 8.17 grades against ("tokens byte-equal
// CPU's sslm_seq_restore_shared twin"), driven through the shipped CPU C ABI (include/superslm/
// sslm_abi.h; the CPU slice's paged verbs are defined on this tree). Only g1e_cell_8_17.cpp includes
// it, so only that executable links the CPU ABI library (the CMake `superslm` static library, which
// t2791's hand-built cpu.lib does not cover: it leaves out src/sslm_abi.cpp).
//
// The drivers are pkv_common.h / pkv_budget_c_helpers.h's (Token at layer budget 1, PrefillAll,
// PrefixPrefillAll, the legacy EnterMidToken, SaveBlob), restated over g1e's CHECK harness. The
// adapter is pkv_budget_b_helpers.h's BuildAdapterBytes, restated for an SslmModelView: one rank-1
// adapter on layer0.q_proj and layer1.v_proj, bound to the fixture's own integrity hash, mapped on
// both backends from the same bytes.
#ifndef SSLM_PKV_GPU_G1E_CPU_TWIN_H
#define SSLM_PKV_GPU_G1E_CPU_TWIN_H

#include "g1e_helpers.h"

#include "sslm_model_hostile_fixtures.h"  // tests/: superslm_test::BuildManifest, BuildArtifact, MakeSection
#include "superslm/artifact.h"
#include "superslm/sslm_abi.h"

#include <new>

namespace g1e {
namespace cpu {

struct Aligned {
	void* p = nullptr;
	size_t n = 0;
	explicit Aligned(size_t size, uint8_t fill = 0) : n(size) {
		p = ::operator new(size ? size : 1, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		std::memset(p, fill, size);
	}
	~Aligned() { ::operator delete(p, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)); }
	Aligned(const Aligned&) = delete;
	Aligned& operator=(const Aligned&) = delete;
};

struct Model {
	std::unique_ptr<Aligned> bytes;
	sslm_model model = nullptr;
	bool ok = false;
	explicit Model(const std::vector<uint8_t>& raw) {
		bytes = std::make_unique<Aligned>(raw.size());
		std::memcpy(bytes->p, raw.data(), raw.size());
		const sslm_status st = sslm_model_map(bytes->p, raw.size(), &model);
		CHECK_MSG(st == SSLM_OK, "CPU sslm_model_map: %d", static_cast<int>(st));
		ok = st == SSLM_OK;
	}
	~Model() {
		if (model) sslm_model_unmap(model);
	}
};

struct Pool {
	std::unique_ptr<Aligned> mem;
	sslm_kv_pool pool = nullptr;
	std::vector<sslm_seq> seqs;
	std::vector<sslm_prefix> prefixes;
	Pool(sslm_model m, uint32_t pages) {
		mem = std::make_unique<Aligned>(sslm_kv_page_size(m) * pages + sslm_kv_page_pool_overhead_size(m, pages));
		const sslm_status st = sslm_kv_page_pool_create(m, mem->p, mem->n, pages, &pool);
		CHECK_MSG(st == SSLM_OK, "CPU page pool of %u pages: %d", pages, static_cast<int>(st));
	}
	~Pool() {
		for (auto it = seqs.rbegin(); it != seqs.rend(); ++it) sslm_seq_release(*it);
		for (auto it = prefixes.rbegin(); it != prefixes.rend(); ++it) sslm_prefix_release(*it);
		if (pool) sslm_kv_pool_destroy(pool);
	}
	sslm_seq Seq(sslm_seq s) {
		if (s) seqs.push_back(s);
		return s;
	}
	sslm_prefix Prefix(sslm_prefix p) {
		if (p) prefixes.push_back(p);
		return p;
	}
};

inline int32_t Token(sslm_model m, sslm_seq s, sslm_status* st = nullptr) {
	sslm_decode_params p{};
	p.layer_budget = 1;
	sslm_seq b[1] = {s};
	int32_t tok = -1;
	sslm_status r = SSLM_OK;
	for (int guard = 0; tok == -1 && guard < 4096; ++guard) {
		r = sslm_decode_step(m, b, 1, &p, nullptr, &tok);
		if (r != SSLM_OK) break;
	}
	if (st) *st = r;
	return r == SSLM_OK ? tok : -1;
}
inline std::vector<int32_t> DecodeN(sslm_model m, sslm_seq s, int64_t n) {
	std::vector<int32_t> out;
	for (int64_t i = 0; i < n; ++i) {
		sslm_status st = SSLM_OK;
		const int32_t t = Token(m, s, &st);
		if (st != SSLM_OK || t < 0) {
			if (st == SSLM_OK && t == -2) out.push_back(-2);
			break;
		}
		out.push_back(t);
	}
	return out;
}
// StepOutcomes' CPU form (g1e_helpers.h): exactly `n` steps, the token, -2, or -1000 - status each.
inline std::vector<int32_t> StepOutcomes(sslm_model m, sslm_seq s, int64_t n) {
	std::vector<int32_t> out;
	for (int64_t i = 0; i < n; ++i) {
		sslm_status st = SSLM_OK;
		const int32_t t = Token(m, s, &st);
		out.push_back(st == SSLM_OK ? t : -1000 - static_cast<int32_t>(st));
	}
	return out;
}
inline sslm_status PrefillAll(sslm_model m, sslm_seq s, const std::vector<int32_t>& t, sslm_span_kind kind = SSLM_SPAN_PROMPT) {
	size_t at = 0;
	sslm_status st = SSLM_OK;
	while (at < t.size()) {
		int32_t consumed = 0;
		st = sslm_prefill(m, s, t.data() + at, static_cast<int32_t>(t.size() - at), 64, kind, nullptr, &consumed);
		at += static_cast<size_t>(consumed > 0 ? consumed : 0);
		if (st != SSLM_OK || consumed <= 0) break;
	}
	return st;
}
inline sslm_status PrefixPrefillAll(sslm_model m, sslm_prefix p, const std::vector<int32_t>& t,
                                    sslm_span_kind kind = SSLM_SPAN_PROMPT) {
	size_t at = 0;
	sslm_status st = SSLM_OK;
	while (at < t.size()) {
		int32_t consumed = 0;
		st = sslm_prefix_prefill(m, p, t.data() + at, static_cast<int32_t>(t.size() - at), 64, kind, nullptr, &consumed);
		at += static_cast<size_t>(consumed > 0 ? consumed : 0);
		if (st != SSLM_OK || consumed <= 0) break;
	}
	return st;
}
// One layer of the next token (layer budget 1); a ready sequence first emits its pending token.
inline bool EnterMidToken(sslm_model m, sslm_seq s) {
	sslm_decode_params p{};
	p.layer_budget = 1;
	sslm_seq b[1] = {s};
	int32_t tok = -1;
	sslm_status st = sslm_decode_step(m, b, 1, &p, nullptr, &tok);
	if (st == SSLM_OK && tok >= 0) st = sslm_decode_step(m, b, 1, &p, nullptr, &tok);
	CHECK_MSG(st == SSLM_OK && tok == -1, "CPU entering mid-token: %d, token %d", static_cast<int>(st), tok);
	return st == SSLM_OK && tok == -1;
}
inline std::vector<uint8_t> Save(sslm_seq s) {
	size_t need = 0;
	std::vector<uint8_t> out;
	if (sslm_seq_save(s, nullptr, &need) != SSLM_BUFFER_TOO_SMALL || need == 0) {
		CHECK_MSG(false, "CPU sslm_seq_save sizing");
		return out;
	}
	out.assign(need, 0);
	size_t wrote = need;
	const sslm_status st = sslm_seq_save(s, out.data(), &wrote);
	CHECK_MSG(st == SSLM_OK, "CPU sslm_seq_save: %d", static_cast<int>(st));
	out.resize(st == SSLM_OK ? wrote : 0);
	return out;
}

// The CPU persona chain, the world left live (8.12's BuildChain): world budget `world_len`, persona
// begin_from(world, 200). `world_tail` tokens (a schema-content span) follow the world's prompt.
struct Chain {
	sslm_prefix world = nullptr, persona = nullptr;
	bool ok = false;
};
inline Chain BuildChain(sslm_model m, Pool& pool, sslm_schema schema = nullptr, const std::vector<int32_t>& world_tail = {}) {
	Chain c;
	const int32_t wl = 1000 + static_cast<int32_t>(world_tail.size());
	sslm_status st = sslm_prefix_begin_budgeted(m, &pool.pool, wl, &c.world);
	if (st == SSLM_OK && schema) st = sslm_prefix_set_schema(c.world, schema);
	if (st == SSLM_OK) st = PrefixPrefillAll(m, c.world, WorldTokens());
	if (st == SSLM_OK && !world_tail.empty()) st = PrefixPrefillAll(m, c.world, world_tail, SSLM_SPAN_SCHEMA_CONTENT);
	if (st == SSLM_OK) st = sslm_prefix_freeze(c.world);
	if (st == SSLM_OK) st = sslm_prefix_begin_from(c.world, 200, &c.persona);
	if (st == SSLM_OK) st = PrefixPrefillAll(m, c.persona, PersonaTokens(0));
	if (st == SSLM_OK) st = sslm_prefix_freeze(c.persona);
	CHECK_MSG(st == SSLM_OK, "CPU chain: %d", static_cast<int>(st));
	pool.Prefix(c.world);
	pool.Prefix(c.persona);
	c.ok = st == SSLM_OK;
	return c;
}

// pkv_budget_b_helpers.h's BuildAdapterBytes, restated over the fixture's bytes and view.
inline std::vector<uint8_t> BuildAdapterBytes(const std::vector<uint8_t>& fixture, const superslm::SslmModelView& view) {
	using namespace superslm_test;
	superslm::SslmArtifact art;
	superslm::SslmError err;
	if (superslm::SslmArtifact::OpenFromMemory(fixture.data(), fixture.size(), art, &err) != superslm::SslmStatus::Ok) {
		CHECK_MSG(false, "adapter: the fixture does not open as an artifact");
		return {};
	}
	const superslm::SslmSectionView* cfg = art.Section(superslm::SslmSectionType::Config);
	const superslm::SslmSectionView* sil = art.Section(superslm::SslmSectionType::SigmoidLut);
	if (!cfg || !sil) {
		CHECK_MSG(false, "adapter: the fixture has no Config or SigmoidLut section");
		return {};
	}
	std::vector<uint8_t> adp = {'A', 'D', 'P', '1'};
	WriteU32LE(adp, 1);             // version
	WriteU32LE(adp, 1);             // rank
	WriteU32LE(adp, 0x1u | 0x40u);  // q_proj, v_proj
	const auto hash = art.RawIntegrityHash();
	adp.insert(adp.end(), hash.begin(), hash.end());
	const double alpha = 8.0;
	uint64_t bits = 0;
	std::memcpy(&bits, &alpha, 8);
	WriteU64LE(adp, bits);
	WriteU32LE(adp, 0);  // use_rslora
	WriteU32LE(adp, 0);  // reserved
	const std::string name = "pkv-gpu-g1e";
	WriteU32LE(adp, static_cast<uint32_t>(name.size()));
	WriteBytes(adp, name);
	struct Proj {
		std::string name;
		uint32_t in, out;
	};
	const uint32_t hidden = view.config.hidden_size, hd = view.config.head_dim;
	const std::vector<Proj> projs = {{"layer0.q_proj", hidden, view.config.num_attention_heads * hd},
	                                 {"layer1.v_proj", hidden, view.config.num_key_value_heads * hd}};
	std::vector<ManifestTensorSpec> wt, dt, ut;
	for (const Proj& p : projs) {
		wt.push_back({p.name + ".lora_A", {1, p.in}});
		wt.push_back({p.name + ".lora_B", {p.out, 1}});
		dt.push_back({p.name, {p.out * 3}});
		ut.push_back({p.name, {3}});
	}
	BuiltManifest w = BuildManifest(reinterpret_cast<const uint8_t*>("WGT1"), 1, wt);
	BuiltManifest d = BuildManifest(superslm::kDeltaFoldScalesMagic, 4, dt);
	BuiltManifest u = BuildManifest(superslm::kUFoldScalesMagic, 4, ut);
	for (BuiltManifest* m : {&d, &u})
		for (size_t t = 0; t < m->tensor_data_off.size(); ++t)
			for (uint64_t r = 0; r < m->tensor_elem_count[t] / 3; ++r) {
				const int32_t triple[3] = {1, 0, 0};
				std::memcpy(m->bytes.data() + m->tensor_data_off[t] + r * 12, triple, 12);
			}
	auto copy = [](superslm::SslmSectionType t, const superslm::SslmSectionView* v) {
		return MakeSection(t, v->dtype, std::vector<uint8_t>(v->data, v->data + v->byte_size));
	};
	BuiltArtifact a = BuildArtifact({copy(superslm::SslmSectionType::Config, cfg),
	                                 copy(superslm::SslmSectionType::SigmoidLut, sil),
	                                 MakeSection(superslm::SslmSectionType::Provenance, superslm::SslmDtype::Raw, adp),
	                                 MakeSection(superslm::SslmSectionType::Weights, superslm::SslmDtype::Int8, w.bytes),
	                                 MakeSection(superslm::SslmSectionType::DeltaFoldScales, superslm::SslmDtype::Int32, d.bytes),
	                                 MakeSection(superslm::SslmSectionType::UFoldScales, superslm::SslmDtype::Int32, u.bytes)});
	return a.bytes;
}

}  // namespace cpu
}  // namespace g1e

#endif  // SSLM_PKV_GPU_G1E_CPU_TWIN_H
