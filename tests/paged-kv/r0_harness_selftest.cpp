// Paged-KV plan (rev 16.1) step R0: the harness's own checks, run on the base (legacy verbs only).
//
// These are not §7 cells. They show the instruments read what they claim before any cell rests on
// them: the scenario runner reproduces the v1.11.0 reference on the build under test, the pinned
// blobs decode and restore and continue as their writers did, the SSB6 hand-builder keeps the rows,
// and the legacy-create admission count counts. They stay green through every step: a legacy
// holder's behaviour and blob are the reference's (§3.7, "byte-equal to the blob 1.9.0 would write").

#include "pkv_common.h"

namespace {

using namespace pkv;

// The lifecycle and prefix-length scenarios on the build under test equal v1.11.0 record for
// record: tokens, context length, K/V rows and the whole SSB5 blob.
void SelfScenarios() {
	for (const char* stem : {"pkv_def", "pkv_odd"}) {
		const Fixture& fx = GetFixture(stem);
		if (!fx.ok) continue;
		for (const char* sc : {"lifecycle", "prefix_lengths", "persist"})
			ExpectMatchesReference("v1.11.0", fx, sc, RunScenario(fx, sc), Compare::kTokensRowsAndBlob);
	}
}

// Each pinned blob decodes, restores into a legacy pool and continues with the writer's own
// continuation tokens: v1.8.1's SSB4 and v1.11.0's SSB5 alike.
void SelfPins() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	for (const char* tag : {"v1.8.1", "v1.11.0"}) {
		for (const char* sc : {"persist", "saturating"}) {
			const char* stage = std::strcmp(sc, "persist") == 0 ? "saved" : "saturated";
			const std::vector<uint8_t> blob = Pin(tag, sc, stage);
			PKV_CHECK_MSG(IsMagic(blob, std::strcmp(tag, "v1.8.1") == 0 ? "SSB4" : "SSB5"), "%s %s magic", tag, sc);
			const RefRecord* saved = RefLookup(Reference(tag, fx), sc, stage);
			const RefRecord* cont = RefLookup(Reference(tag, fx), sc, "continuation8");
			if (!saved || !cont) continue;
			PKV_CHECK_MSG(Sha(blob.data(), blob.size()) == saved->blob_sha, "%s %s pin digest", tag, sc);
			LegacyPool pool(fx.model, 1);
			sslm_seq s = nullptr;
			PKV_CHECK_EQ(sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &s), SSLM_OK);
			if (!s) continue;
			std::vector<int32_t> toks;
			for (int i = 0; i < 8; ++i) toks.push_back(NextToken(fx.model, s));
			PKV_CHECK_MSG(toks == cont->tokens, "%s %s continuation after restore", tag, sc);
			sslm_seq_release(s);
		}
	}
}

// BlobRows reads SSB5 rows that digest to the reference's rows_sha, and Ssb6WholeReserveFromSsb5
// carries the same rows into its canonical section.
void SelfBlobRows() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const std::vector<uint8_t> ssb5 = Pin("v1.11.0", "lifecycle", "adopt300");
	const RefRecord* rec = RefLookup(Reference("v1.11.0", fx), "lifecycle", "adopt300");
	if (!rec) return;
	std::vector<uint8_t> rows;
	PKV_CHECK(BlobRows(ssb5, fx, HiddenSize(fx), rec->context_length, &rows));
	PKV_CHECK_MSG(Sha(rows.data(), rows.size()) == rec->rows_sha, "SSB5 rows digest");
	const std::vector<uint8_t> ssb6 = Ssb6WholeReserveFromSsb5(ssb5, fx, 300);
	PKV_CHECK(IsMagic(ssb6, "SSB6"));
	const BlobView v = ParseBlobWithHidden(ssb6, HiddenSize(fx));
	PKV_CHECK(v.ok && v.kv_mode == 0 && v.budget == fx.geo.context_cap && v.origin == 300);
	PKV_CHECK_EQ(v.kv_positions, 300);
	PKV_CHECK_EQ(ssb6.size(), kSsb6Header + HiddenSize(fx) + 8 + 300 * fx.BytesPerToken());
	std::vector<uint8_t> rows6;
	PKV_CHECK(BlobRows(ssb6, fx, HiddenSize(fx), rec->context_length, &rows6));
	PKV_CHECK_MSG(rows6 == rows, "SSB6 hand-built rows equal the SSB5 rows");
}

// The legacy-create admission count admits exactly the pool's block count, and refuses with
// SSLM_KV_POOL_EXHAUSTED.
void SelfAdmissionCount() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	for (uint32_t blocks : {1u, 3u}) {
		LegacyPool pool(fx.model, blocks);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		sslm_status refusal = SSLM_OK;
		PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool, &refusal), blocks);
		PKV_CHECK_EQ(refusal, SSLM_KV_POOL_EXHAUSTED);
	}
	// §8's arithmetic, written in the test, on the fixture's own geometry.
	PKV_CHECK_EQ(fx.B(), 16);
	PKV_CHECK_EQ(fx.CapPages(), 256);
	PKV_CHECK_EQ(fx.R(512), 33);
	PKV_CHECK_EQ(fx.R(4096), 256);
	PKV_CHECK_EQ(fx.PageBytes() * 256, sslm_kv_block_size(fx.model));
	PKV_CHECK_EQ(GetFixture("pkv_odd").B(), 4100);
}

PKV_CELL("R0.self.scenarios", "R0", SelfScenarios);
PKV_CELL("R0.self.pins", "R0", SelfPins);
PKV_CELL("R0.self.blob_rows", "R0", SelfBlobRows);
PKV_CELL("R0.self.admission_count", "R0", SelfAdmissionCount);

}  // namespace
