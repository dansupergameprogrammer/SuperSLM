// Paged-KV plan (rev 16.2) §7 cell 8.17 [G5] (rev 5; coverage audit round 3, G6):
// "sslm_gpu_seq_restore_into_pool x a mid-token SLM6; x a schema-bound prefix (bound_schema_index and
// dfa_walk_state come from the blob while the pages are shared); x an adapter bound on the restored
// holder. Each: tokens byte-equal CPU's sslm_seq_restore_shared twin and the unsaved GPU twin."
// Group g1e, step G1. The GPU twin of 8.12 (tests/paged-kv/c5_dim8_composition.cpp Cell812), whose
// three constructions it repeats on both backends from the same fixture and token streams.
//
// Each part builds the persona chain with the world live (world 1,000, or 1,001 with a schema-content
// token; persona begin_from(world, 200)), a budget-512 holder adopts the persona and decodes, the
// holder is saved and restored WITH the persona handle. Sharing is graded by admission: the
// destination pool leaves exactly R(512) pages free for the restore, so a private restore (R + E) is
// refused there and SSLM_OK is itself the evidence of the share. The same construction runs on the
// CPU through the shipped C ABI (sslm_seq_restore_shared), and on the GPU the unsaved original keeps
// decoding beside its restored twin. Verdict per part: GPU twin tokens == unsaved GPU original's ==
// CPU twin's.
//   (a) x a mid-token SLM6: 20 tokens, then one layer of the next (layer_index 1 of 2), save, restore;
//       10 tokens each (the first finishes the pending token).
//   (b) x a schema-bound prefix: the world is bound to the fixture's S8 schema "g5_minimal_one_field"
//       and takes one admitted schema-content token at 1,000, so the adopter's origin, 1,201, is
//       mid-page; the holder binds the schema too and adopts. 3 greedy tokens, save, restore; the
//       restored holder's bound flag, walk state and accepting flag equal the original's (they come
//       from the blob, not the handle), and the next 4 tokens equal on all three.
//   (c) x an adapter bound on the restored holder: 20 tokens, save, restore; the same adapter (built
//       from the fixture's bytes) bound on the original and the twin (sslm_gpu_seq_bind_adapter) and on
//       the CPU original and twin (sslm_seq_set_adapter); 12 tokens each; the GPU original's and twin's
//       re-saved SLM6 are byte-equal (the adapter does not round-trip, so their state is one state).
//       The construction is checked not to be vacuous: the adapter changes the tokens.
//
//   CONSTRUCTION (G5 cells fix, the 8.16 precedent in g1a_g4_prefix.cpp): pkv_def's S8 schema accepts
//   one document spelled by one token, and (b)'s world takes exactly that token, so the walk the persona
//   carries is at the accepting state with no outgoing transition: every side's decode is the defined
//   dead end (-2 at SSLM_OK, retry-stable). DecodeN stops at the first -2, so the three sides gave [-2]
//   and the size check (3 / 4 tokens) could not hold. (b) therefore grades one outcome per step
//   (StepOutcomes: the token, -2, or -1000 - status), 3 before the save and 4 after, byte-equal across
//   the GPU original, the GPU twin and the CPU twin, with no refused step. A restore that took the walk
//   from the handle or dropped it would restart the walk and emit the admitted token; one that dropped
//   the binding would decode unmasked and emit real tokens: either differs from the CPU twin's -2s.
//
// Mutants killed: (a) a restore that refuses a mid-token blob's share or draws E when sharing; (b) a
// shared restore taking the walk state or binding from the handle; (c) an adapter applied to rows the
// restored holder shares (they are base-only), or a share under the adapter.
//
// RED NOW: link-red (G3, G4 and G5 GPU verbs). It also links the CPU ABI library (the CMake `superslm`
// target), which the CPU slice already defines. After G4: red at run until G5.
// GREEN: every check passes on both GPUs, debug layer clean.
//
// Run: g1e_cell_8_17.exe --fixture-dir=DIR
#include "g1e_cpu_twin.h"

using namespace g1e;

namespace {

constexpr int32_t kBudget = 512;
constexpr int64_t kWorldLen = 1000, kPersonaLen = 200;
const char* const kSchema = "g5_minimal_one_field";

// Pages of the chain with the world live, by §3.4 (8.12's ChainPages).
int64_t ChainPages(const Geo& g, int64_t wl, int64_t pl) { return g.PrefixPages(wl) + (g.PrefixPages(wl + pl) - g.SharedPages(wl)); }

// The GPU chain with the world live; `world_tail` tokens are a schema-content span after the prompt.
GChain GpuChain(Rig& rig, int32_t schema_index, const std::vector<int32_t>& world_tail) {
	GChain c;
	const uint32_t db = rig.e.full_budget;
	const int32_t wl = static_cast<int32_t>(kWorldLen + world_tail.size());
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, wl, &c.world);
	if (st == GS::SSLM_OK && schema_index >= 0) st = sslm_gpu_prefix_set_schema(rig.ctx, c.world, schema_index);
	if (st == GS::SSLM_OK) st = PrefixPrefillAll(rig.ctx, c.world, WorldTokens(), 0, db);
	if (st == GS::SSLM_OK && !world_tail.empty()) st = PrefixPrefillAll(rig.ctx, c.world, world_tail, 1, db);
	if (st == GS::SSLM_OK) st = sslm_gpu_prefix_freeze(rig.ctx, c.world);
	if (st == GS::SSLM_OK) st = sslm_gpu_prefix_begin_from(rig.ctx, c.world, static_cast<int32_t>(kPersonaLen), &c.persona);
	if (st == GS::SSLM_OK) st = PrefixPrefillAll(rig.ctx, c.persona, PersonaTokens(0), 0, db);
	if (st == GS::SSLM_OK) st = sslm_gpu_prefix_freeze(rig.ctx, c.persona);
	CHECK_MSG(st == GS::SSLM_OK, "8.17 GPU chain: %s", StatusName(st));
	if (!Real(c.world)) c.world = nullptr;
	if (!Real(c.persona)) c.persona = nullptr;
	c.ok = st == GS::SSLM_OK;
	rig.chain = c;
	return c;
}

std::string Ids(const std::vector<int32_t>& t) {
	std::string s;
	for (size_t i = 0; i < t.size(); ++i) s += (i ? "," : "") + std::to_string(t[i]);
	return s;
}

void Compare3(const std::vector<int32_t>& gpu_orig, const std::vector<int32_t>& gpu_twin, const std::vector<int32_t>& cpu_twin,
              size_t want, const char* what) {
	CHECK_MSG(gpu_twin.size() == want && gpu_twin == gpu_orig, "%s: GPU twin [%s] differs from the unsaved GPU original [%s]",
	          what, Ids(gpu_twin).c_str(), Ids(gpu_orig).c_str());
	CHECK_MSG(gpu_twin == cpu_twin, "%s: GPU twin [%s] differs from the CPU restore_shared twin [%s]", what, Ids(gpu_twin).c_str(),
	          Ids(cpu_twin).c_str());
}

// ---- (a) x a mid-token SLM6 --------------------------------------------------------------------------
void PartA(const Env& e, cpu::Model& cm) {
	const Geo& g = e.geo;
	const uint32_t P = static_cast<uint32_t>(ChainPages(g, kWorldLen, kPersonaLen) + 2 * g.R(kBudget));
	std::vector<int32_t> gpu_orig, gpu_twin, cpu_twin, gpu_pre, cpu_pre;
	{
		Rig rig(e, P);
		const GChain c = rig.pool ? GpuChain(rig, -1, {}) : GChain{};
		GSeq s = c.ok ? rig.Budgeted(kBudget) : GSeq{};
		if (!s.h) return;
		CHECK(Adopt(rig, s, c.persona, kWorldLen + kPersonaLen) == GS::SSLM_OK);
		gpu_pre = DecodeN(e, s, 20);
		if (!EnterMidToken(e, s)) return;
		const std::vector<uint8_t> blob = Save(e, s.h);
		Slm6SelfCheck(blob, g, kBudget, kWorldLen + kPersonaLen, ContextLength(s.h), "8.17 (a)");
		// kills: restore-into-pool refusing a mid-token blob's share, or drawing E when sharing
		Restored r = RestoreIntoPool(rig, blob, c.persona, s);
		CHECK_MSG(r.status == GS::SSLM_OK, "8.17 (a): restore_into_pool of the mid-token SLM6 with the persona: %s",
		          StatusName(r.status));
		if (!r.seq.h) return;
		gpu_orig = DecodeN(e, s, 10);
		GSeq t = r.seq;
		gpu_twin = DecodeN(e, t, 10);
	}
	{
		cpu::Pool pool(cm.model, P);
		const cpu::Chain c = cpu::BuildChain(cm.model, pool);
		sslm_seq s = nullptr;
		if (!c.ok || sslm_seq_create_budgeted(cm.model, &pool.pool, kBudget, &s) != SSLM_OK) return;
		pool.Seq(s);
		CHECK(sslm_seq_adopt_prefix(s, c.persona) == SSLM_OK);
		cpu_pre = cpu::DecodeN(cm.model, s, 20);
		cpu::EnterMidToken(cm.model, s);
		const std::vector<uint8_t> blob = cpu::Save(s);
		sslm_seq r = nullptr;
		CHECK(sslm_seq_restore_shared(cm.model, &pool.pool, blob.data(), blob.size(), c.persona, &r, nullptr) == SSLM_OK);
		if (!pool.Seq(r)) return;
		cpu_twin = cpu::DecodeN(cm.model, r, 10);
	}
	CHECK_MSG(gpu_pre == cpu_pre, "8.17 (a): construction: the 20 tokens before the save differ between GPU and CPU");
	Compare3(gpu_orig, gpu_twin, cpu_twin, 10, "8.17 (a) mid-token SLM6");
}

// ---- (b) x a schema-bound prefix ------------------------------------------------------------------------
void PartB(const Env& e, cpu::Model& cm) {
	const Geo& g = e.geo;
	IndependentSchema sch;
	CHECK_MSG(sch.Build(e.fx.bytes, kSchema), "8.17 (b): the fixture has no schema %s", kSchema);
	if (!sch.entry) return;
	const int32_t admitted = sch.FirstLegal(0);
	CHECK_MSG(admitted >= 0, "8.17 (b): no token is admitted at the schema's state 0");
	if (admitted < 0) return;
	const int32_t gidx = SslmGpuSchemaLookupForG5Bridge(e.model(), kSchema);
	CHECK_MSG(gidx >= 0, "8.17 (b): SslmGpuSchemaLookupForG5Bridge(%s) = %d", kSchema, gidx);
	if (gidx < 0) return;
	const uint32_t P = static_cast<uint32_t>(ChainPages(g, kWorldLen + 1, kPersonaLen) + 2 * g.R(kBudget));
	const int64_t origin = kWorldLen + 1 + kPersonaLen;  // 1,201: mid-page
	std::vector<int32_t> gpu_orig, gpu_twin, cpu_twin, gpu_pre, cpu_pre;
	{
		Rig rig(e, P);
		const GChain c = rig.pool ? GpuChain(rig, gidx, {admitted}) : GChain{};
		GSeq s = c.ok ? rig.Budgeted(kBudget) : GSeq{};
		if (!s.h) return;
		CHECK(SslmGpuSeqSetSchemaForG5Bridge(e.ctx, s.h, gidx) == GS::SSLM_OK);
		CHECK(Adopt(rig, s, c.persona, origin) == GS::SSLM_OK);
		gpu_pre = StepOutcomes(e, s, 3);
		const std::vector<uint8_t> blob = Save(e, s.h);
		Slm6SelfCheck(blob, g, kBudget, origin, ContextLength(s.h), "8.17 (b)");
		Restored r = RestoreIntoPool(rig, blob, c.persona, s);
		CHECK_MSG(r.status == GS::SSLM_OK, "8.17 (b): restore_into_pool with the schema-bound persona: %s", StatusName(r.status));
		if (!r.seq.h) return;
		// kills: a shared restore taking the walk state or the binding from the handle instead of the blob
		int32_t ob = -1, tb = -2, oa = -1, ta = -2;
		CHECK(SslmGpuSeqSchemaBoundForG5Bridge(e.ctx, s.h, &ob) == GS::SSLM_OK);
		CHECK(SslmGpuSeqSchemaBoundForG5Bridge(e.ctx, r.seq.h, &tb) == GS::SSLM_OK);
		CHECK(SslmGpuSeqSchemaAcceptingForG5Bridge(e.ctx, s.h, &oa) == GS::SSLM_OK);
		CHECK(SslmGpuSeqSchemaAcceptingForG5Bridge(e.ctx, r.seq.h, &ta) == GS::SSLM_OK);
		CHECK_MSG(ob == 1 && tb == ob, "8.17 (b): bound flag: original %d, restored %d", ob, tb);
		CHECK_MSG(ta == oa, "8.17 (b): accepting flag: original %d, restored %d", oa, ta);
		const uint32_t ow = SslmGpuSeqWalkStateForG5Bridge(s.h), tw = SslmGpuSeqWalkStateForG5Bridge(r.seq.h);
		CHECK_MSG(tw == ow, "8.17 (b): walk state: original %u, restored %u", ow, tw);
		gpu_orig = StepOutcomes(e, s, 4);
		GSeq t = r.seq;
		gpu_twin = StepOutcomes(e, t, 4);
	}
	{
		cpu::Pool pool(cm.model, P);
		sslm_schema schema = nullptr;
		CHECK(sslm_schema_lookup(cm.model, kSchema, &schema) == SSLM_OK);
		const cpu::Chain c = schema ? cpu::BuildChain(cm.model, pool, schema, {admitted}) : cpu::Chain{};
		sslm_seq s = nullptr;
		if (!c.ok || sslm_seq_create_budgeted(cm.model, &pool.pool, kBudget, &s) != SSLM_OK) return;
		pool.Seq(s);
		CHECK(sslm_seq_set_schema(s, schema) == SSLM_OK);
		CHECK(sslm_seq_adopt_prefix(s, c.persona) == SSLM_OK);
		cpu_pre = cpu::StepOutcomes(cm.model, s, 3);
		const std::vector<uint8_t> blob = cpu::Save(s);
		sslm_seq r = nullptr;
		CHECK(sslm_seq_restore_shared(cm.model, &pool.pool, blob.data(), blob.size(), c.persona, &r, nullptr) == SSLM_OK);
		if (!pool.Seq(r)) return;
		cpu_twin = cpu::StepOutcomes(cm.model, r, 4);
	}
	CHECK_MSG(gpu_pre == cpu_pre && gpu_pre.size() == 3, "8.17 (b): construction: the 3 outcomes before the save [%s] differ from the CPU's [%s]",
	          Ids(gpu_pre).c_str(), Ids(cpu_pre).c_str());
	// No step may be a refusal (-1000 - status): the outcomes are tokens or the schema dead end.
	bool refused = false;
	for (const auto* v : {&gpu_pre, &gpu_orig, &gpu_twin, &cpu_twin})
		for (int32_t t : *v) refused = refused || t <= -1000;
	CHECK_MSG(!refused, "8.17 (b): a decode step was refused (outcome <= -1000)");
	Compare3(gpu_orig, gpu_twin, cpu_twin, 4, "8.17 (b) schema-bound prefix");
}

// ---- (c) x an adapter bound on the restored holder ---------------------------------------------------------
void PartC(const Env& e, cpu::Model& cm) {
	const Geo& g = e.geo;
	const std::vector<uint8_t> abytes = cpu::BuildAdapterBytes(e.fx.bytes, e.fx.view);
	if (abytes.empty()) return;
	superslm::SslmModelView aview{};
	std::string err;
	CHECK_MSG(superslm::SslmModel::Load(abytes.data(), abytes.size(), aview, &err) == superslm::SslmModelStatus::Ok,
	          "8.17 (c): the adapter does not load: %s", err.c_str());
	SslmGpuAdapterHandle* ga = nullptr;
	CHECK(sslm_gpu_adapter_map(e.ctx, e.model(), &aview, &ga) == GS::SSLM_OK && ga);
	cpu::Aligned abuf(abytes.size());
	std::memcpy(abuf.p, abytes.data(), abytes.size());
	sslm_adapter ca = nullptr;
	CHECK(sslm_adapter_map(abuf.p, abytes.size(), cm.model, &ca) == SSLM_OK && ca);
	if (!ga || !ca) {
		if (ga) sslm_gpu_adapter_unmap(e.ctx, ga);
		if (ca) sslm_adapter_release(ca);
		return;
	}
	const uint32_t P = static_cast<uint32_t>(ChainPages(g, kWorldLen, kPersonaLen) + 2 * g.R(kBudget));
	std::vector<int32_t> gpu_orig, gpu_twin, cpu_twin, unadapted;
	{
		Rig rig(e, P);
		const GChain c = rig.pool ? GpuChain(rig, -1, {}) : GChain{};
		GSeq s = c.ok ? rig.Budgeted(kBudget) : GSeq{};
		if (s.h) {
			CHECK(Adopt(rig, s, c.persona, kWorldLen + kPersonaLen) == GS::SSLM_OK);
			DecodeN(e, s, 20);
			const std::vector<uint8_t> blob = Save(e, s.h);
			Restored r = RestoreIntoPool(rig, blob, c.persona, s);
			CHECK_MSG(r.status == GS::SSLM_OK, "8.17 (c): restore_into_pool with the persona: %s", StatusName(r.status));
			if (r.seq.h) {
				CHECK(sslm_gpu_seq_bind_adapter(e.ctx, s.h, ga) == GS::SSLM_OK);
				CHECK(sslm_gpu_seq_bind_adapter(e.ctx, r.seq.h, ga) == GS::SSLM_OK);
				GSeq t = r.seq;
				gpu_orig = DecodeN(e, s, 12);
				gpu_twin = DecodeN(e, t, 12);
				// kills: an adapter applied to (or a share made under) the rows the restored holder shares
				CHECK_MSG(Save(e, s.h) == Save(e, r.seq.h), "8.17 (c): the GPU original's and twin's re-saved SLM6 differ");
				CHECK(sslm_gpu_seq_bind_adapter(e.ctx, s.h, nullptr) == GS::SSLM_OK);
				CHECK(sslm_gpu_seq_bind_adapter(e.ctx, r.seq.h, nullptr) == GS::SSLM_OK);
				rig.ReleaseSeq(r.seq.h);
				// The non-vacuity control: the same restore, unadapted, from the same save point.
				Restored u = RestoreIntoPool(rig, blob, c.persona, r.seq);
				if (u.seq.h) {
					GSeq ut = u.seq;
					unadapted = DecodeN(e, ut, 12);
				}
			}
		}
	}
	{
		cpu::Pool pool(cm.model, P);
		const cpu::Chain c = cpu::BuildChain(cm.model, pool);
		sslm_seq s = nullptr;
		if (c.ok && sslm_seq_create_budgeted(cm.model, &pool.pool, kBudget, &s) == SSLM_OK) {
			pool.Seq(s);
			CHECK(sslm_seq_adopt_prefix(s, c.persona) == SSLM_OK);
			cpu::DecodeN(cm.model, s, 20);
			const std::vector<uint8_t> blob = cpu::Save(s);
			sslm_seq r = nullptr;
			CHECK(sslm_seq_restore_shared(cm.model, &pool.pool, blob.data(), blob.size(), c.persona, &r, nullptr) == SSLM_OK);
			if (pool.Seq(r)) {
				CHECK(sslm_seq_set_adapter(r, ca) == SSLM_OK);
				cpu_twin = cpu::DecodeN(cm.model, r, 12);
				CHECK(sslm_seq_set_adapter(r, nullptr) == SSLM_OK);
			}
		}
	}
	CHECK_MSG(!unadapted.empty() && unadapted != gpu_twin, "8.17 (c): construction: the adapter does not change the tokens, so the "
	                                                       "part is vacuous");
	Compare3(gpu_orig, gpu_twin, cpu_twin, 12, "8.17 (c) adapter bound on the restored holder");
	sslm_gpu_adapter_unmap(e.ctx, ga);
	sslm_adapter_release(ca);
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) {
			cpu::Model cm(e.fx.bytes);
			if (cm.ok) {
				PartA(e, cm);
				PartB(e, cm);
				PartC(e, cm);
			}
		}
	}
	return FinishSuite("g1e_cell_8_17");
}
