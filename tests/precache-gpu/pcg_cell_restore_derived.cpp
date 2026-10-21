// Precached-prefixes plan cell 8.16 [F2] (the precache plan's GPU restore rows: audit G5 (b); §4 item 2; the red
// suite's reading 15): a restored GPU prefix's derived state, against its prefill twin. SPX1 carries no ready flag, no
// snapshot flag and no bind permission, so restore derives them from L: ready iff L > 0, the prefill snapshot
// valid iff L > 0, the bind permission iff L = 0. On GPU prefixes restored at L = 0 and at L > 0, and their
// prefill twins:
//   - sslm_gpu_prefix_set_schema on a begin_from child is admitted iff L = 0 (a frozen prefix refuses
//     set_schema itself, so the permission is read through the child, which inherits it);
//   - an adopter is ready for logits iff L > 0 (read from the adopter's own 'SLM6' save, the ready word at
//     offset 128) and reads the prefill snapshot iff L > 0;
//   - every prefix verb's disposition equals the twin's (the GPU twin of 8.11): prefill on the frozen prefix,
//     a second freeze (idempotent), set_schema on the frozen prefix, begin_from, and the pool's destroy while
//     both live (SSLM_MODEL_HAS_LIVE_SEQUENCES, the GPU pool's refusal; one call,
//     since twin and copy share the rig's pool).
// The L = 0 twin is built 6.8 leg D's way (rev 1.18.4): an empty prefix given a zero-count prefill, then
// frozen. That prefill clears the bind permission, so this is the one history on which the freeze-time
// setting matters; a twin frozen with no prefill call would leave it unexercised. 6.8's leg D stays: it grades
// the original against its own copy, which this cell does not.
//
// Mutants (apply_gpu_mutant.py):
//   bind-as-ready       restore's bind permission derived as for L > 0 -> the L = 0 restore's child refuses
//                       set_schema -> red;
//   freeze-set-removed  the freeze-time setting removed (the bind permission left as the zero-count prefill
//                       set it) -> the L = 0 twin's child refuses set_schema while the restore's is admitted
//                       -> red.
// GREEN at ac2fc2c7 (the freeze normalises from L). Before it, the freeze-set-removed reading is the tree's
// own state, so this cell was red there on the L = 0 twin.
//
// Run: pcg_cell_restore_derived.exe --fixture-dir=DIR (pkv_def).
#include "pcg_helpers.h"

using namespace pcg;

namespace {

constexpr size_t kSlm6Magic = 0, kSlm6Ready = 128;

// The adopter's ready-for-logits word, from its own sequence save ('SLM6': a pooled sequence). -1 when the
// save fails or is not an SLM6.
int ReadyWord(Rig& rig, SslmGpuSequenceHandle* s, const char* what) {
	size_t need = 0;
	(void)sslm_gpu_seq_save(rig.ctx, s, nullptr, &need);  // reports the size it needs (g1a's convention)
	if (need < kSlm6Ready + 4) {
		CHECK_MSG(false, "%s: sequence save size query reported %zu bytes", what, need);
		return -1;
	}
	std::vector<uint8_t> b(need);
	size_t n = need;
	const GS st = sslm_gpu_seq_save(rig.ctx, s, b.data(), &n);
	uint32_t magic = 0, ready = 0;
	if (st == GS::SSLM_OK) {
		std::memcpy(&magic, b.data() + kSlm6Magic, 4);
		std::memcpy(&ready, b.data() + kSlm6Ready, 4);
	}
	CHECK_MSG(st == GS::SSLM_OK && magic == 0x364D4C53u, "%s: sequence save %s, magic 0x%08x (want 'SLM6')", what, Name(st),
	          magic);
	return st == GS::SSLM_OK && magic == 0x364D4C53u ? static_cast<int>(ready != 0) : -1;
}

struct Derived {
	GS begin_from = GS::SSLM_DEVICE_LOST;
	GS child_set_schema = GS::SSLM_DEVICE_LOST;
	GS adopt = GS::SSLM_DEVICE_LOST;
	int ready = -1;
	GS snapshot = GS::SSLM_DEVICE_LOST;
	GS prefill_frozen = GS::SSLM_DEVICE_LOST;
	GS freeze_again = GS::SSLM_DEVICE_LOST;
	GS set_schema_frozen = GS::SSLM_DEVICE_LOST;
};

Derived Read(Rig& rig, SslmGpuPrefixHandle* p, const char* what) {
	Derived d;
	SslmGpuPrefixHandle* k = nullptr;
	d.begin_from = sslm_gpu_prefix_begin_from(rig.ctx, p, 16, &k);
	k = rig.Prefix(k);
	if (k) d.child_set_schema = sslm_gpu_prefix_set_schema(rig.ctx, k, 0);
	GSeq s = rig.Budgeted(256);
	if (s.h) {
		d.adopt = sslm_gpu_seq_adopt_prefix(rig.ctx, s.h, p);
		if (d.adopt == GS::SSLM_OK) {
			d.ready = ReadyWord(rig, s.h, what);
			uint32_t hidden = 0;
			CHECK(sslm_gpu_model_hidden_size(rig.model, &hidden) == GS::SSLM_OK && hidden > 0);
			std::vector<int8_t> codes(hidden, 0);
			size_t required = 0;
			int64_t m = 0, ex = 0;
			d.snapshot = sslm_gpu_seq_read_prefill_final_hidden(rig.ctx, s.h, codes.data(), codes.size(), &required, &m, &ex);
		}
		rig.ReleaseSeq(s.h);
	}
	const int32_t tok = 5;
	int32_t c = -1;
	d.prefill_frozen = sslm_gpu_prefix_prefill(rig.ctx, p, &tok, 1, 0, rig.e.full_budget, &c);
	d.freeze_again = sslm_gpu_prefix_freeze(rig.ctx, p);
	d.set_schema_frozen = sslm_gpu_prefix_set_schema(rig.ctx, p, 0);
	std::printf("%s: begin_from %s, child set_schema %s, adopt %s, ready %d, snapshot %s, prefill (frozen) %s, "
	            "freeze again %s, set_schema (frozen) %s\n",
	            what, Name(d.begin_from), Name(d.child_set_schema), Name(d.adopt), d.ready, Name(d.snapshot),
	            Name(d.prefill_frozen), Name(d.freeze_again), Name(d.set_schema_frozen));
	return d;
}

void Leg(const Env& e, int32_t L) {
	Rig rig(e, 32);
	if (!rig.pool) return;
	const std::string tag = "L = " + std::to_string(L);
	SslmGpuPrefixHandle* raw = nullptr;
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, 64, &raw);
	CHECK_MSG(st == GS::SSLM_OK, "%s: prefix_begin: %s", tag.c_str(), StatusName(st));
	if (st != GS::SSLM_OK) return;
	SslmGpuPrefixHandle* twin = rig.Prefix(raw);
	if (L == 0) {
		// 6.8 leg D's history: a zero-count prefill, which clears the bind permission.
		const int32_t unused = 0;
		int32_t c = -1;
		st = sslm_gpu_prefix_prefill(rig.ctx, twin, &unused, 0, 0, e.full_budget, &c);
		CHECK_MSG(st == GS::SSLM_OK && c == 0, "%s: the zero-count prefill: %s, consumed %d", tag.c_str(), StatusName(st), c);
	} else {
		const std::vector<int32_t> t = PrefixTokens(L, 11);
		st = PrefixPrefillAll(rig.ctx, twin, t, 0, e.full_budget);
		CHECK_MSG(st == GS::SSLM_OK, "%s: the twin's prefill: %s", tag.c_str(), StatusName(st));
	}
	CHECK_MSG(sslm_gpu_prefix_freeze(rig.ctx, twin) == GS::SSLM_OK, "%s: freeze", tag.c_str());
	GS sst = GS::SSLM_DEVICE_LOST;
	const std::vector<uint8_t> blob = SavePrefix(e, twin, &sst);
	CHECK_MSG(sst == GS::SSLM_OK && !blob.empty(), "%s: save %s", tag.c_str(), Name(sst));
	if (blob.empty()) return;
	const RestoredPrefix r = RestorePrefix(rig, blob);
	CHECK_MSG(r.status == GS::SSLM_OK && r.p, "%s: restore %s", tag.c_str(), Name(r.status));
	if (!r.p) return;

	const Derived a = Read(rig, twin, (tag + ", twin").c_str());
	const Derived b = Read(rig, r.p, (tag + ", restored").c_str());
	const bool want_bind = L == 0;
	const bool want_ready = L > 0;

	// The derived state, on each.
	CHECK_MSG((a.child_set_schema == GS::SSLM_OK) == want_bind, "%s: the twin's child set_schema %s (admitted iff L = 0)",
	          tag.c_str(), Name(a.child_set_schema));
	CHECK_MSG((b.child_set_schema == GS::SSLM_OK) == want_bind,
	          "%s: the restored prefix's child set_schema %s (admitted iff L = 0)", tag.c_str(), Name(b.child_set_schema));
	CHECK_MSG(a.adopt == GS::SSLM_OK && b.adopt == GS::SSLM_OK, "%s: adopt: twin %s, restored %s", tag.c_str(), Name(a.adopt),
	          Name(b.adopt));
	CHECK_MSG(a.ready == (want_ready ? 1 : 0), "%s: the twin's adopter ready for logits %d (iff L > 0)", tag.c_str(), a.ready);
	CHECK_MSG(b.ready == (want_ready ? 1 : 0), "%s: the restored prefix's adopter ready for logits %d (iff L > 0)",
	          tag.c_str(), b.ready);
	CHECK_MSG((a.snapshot == GS::SSLM_OK) == want_ready && (b.snapshot == GS::SSLM_OK) == want_ready,
	          "%s: adopters read the prefill snapshot iff L > 0: twin %s, restored %s", tag.c_str(), Name(a.snapshot),
	          Name(b.snapshot));

	// Every verb's disposition equals the twin's.
	CHECK_MSG(a.begin_from == b.begin_from, "%s: begin_from: twin %s, restored %s", tag.c_str(), Name(a.begin_from),
	          Name(b.begin_from));
	CHECK_MSG(a.child_set_schema == b.child_set_schema, "%s: child set_schema: twin %s, restored %s", tag.c_str(),
	          Name(a.child_set_schema), Name(b.child_set_schema));
	CHECK_MSG(a.snapshot == b.snapshot, "%s: snapshot read: twin %s, restored %s", tag.c_str(), Name(a.snapshot),
	          Name(b.snapshot));
	CHECK_MSG(a.prefill_frozen == b.prefill_frozen && a.prefill_frozen != GS::SSLM_OK,
	          "%s: prefill on the frozen prefix: twin %s, restored %s (refused on both)", tag.c_str(), Name(a.prefill_frozen),
	          Name(b.prefill_frozen));
	CHECK_MSG(a.freeze_again == GS::SSLM_OK && b.freeze_again == GS::SSLM_OK, "%s: a second freeze: twin %s, restored %s",
	          tag.c_str(), Name(a.freeze_again), Name(b.freeze_again));
	CHECK_MSG(a.set_schema_frozen == b.set_schema_frozen && a.set_schema_frozen != GS::SSLM_OK,
	          "%s: set_schema on the frozen prefix: twin %s, restored %s (refused on both)", tag.c_str(),
	          Name(a.set_schema_frozen), Name(b.set_schema_frozen));
	const GS destroy = sslm_gpu_kv_pool_destroy(rig.ctx, rig.pool);
	CHECK_MSG(destroy == GS::SSLM_MODEL_HAS_LIVE_SEQUENCES, "%s: the pool's destroy while both live: %s", tag.c_str(),
	          Name(destroy));
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) {
			Leg(e, 0);
			Leg(e, 10);
		}
	}
	return FinishSuite("pcg_cell_restore_derived");
}
