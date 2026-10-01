// Paged-KV plan (rev 16.1) step R0: the harness's own checks, run on the base (legacy verbs only).
//
// These are not §7 cells. They show the instruments read what they claim before any cell rests on
// them: the scenario runner reproduces the v1.11.0 reference on the build under test, the pinned
// blobs decode and restore and continue as their writers did, the SSB6 hand-builder keeps the rows,
// and the legacy-create admission count counts. They stay green through every step: a legacy
// holder's behaviour and blob are the reference's (§3.7, "byte-equal to the blob 1.9.0 would write").

#include "pkv_common.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

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

// A .ref reads the same with LF and CRLF line endings. A Windows checkout with core.autocrlf=true
// turned every .ref into CRLF before .gitattributes pinned *.ref to eol=lf, and std::getline keeps
// the '\r', so the header's fixture sha (and an empty token list) read differently and every cell
// on the reference false-failed under POSIX. The check: each reference, normalized to LF and
// re-written with CRLF into temp files, parses (in binary mode, so no platform's text mode hides
// the '\r') to the same fixture sha and the same records, field for field.
bool SameRecord(const RefRecord& a, const RefRecord& b) {
	return a.context_length == b.context_length && a.saturation == b.saturation && a.blob_sha == b.blob_sha &&
	       a.rows_sha == b.rows_sha && a.tokens == b.tokens;
}

void SelfRefCrlf() {
	namespace fs = std::filesystem;
	std::error_code ec;
	const fs::path tmp = fs::temp_directory_path(ec);
	PKV_CHECK_MSG(!ec, "R0.self.ref_crlf: no temp directory (%s)", ec.message().c_str());
	if (ec) return;
	const std::string nonce = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
	for (const char* key : {"v1.11.0_pkv_def", "v1.8.1_pkv_def", "v1.11.0_pkv_32k"}) {
		std::ifstream src(std::string(PKV_REFERENCE_DIR) + "/" + key + ".ref", std::ios::binary);
		PKV_CHECK_MSG(static_cast<bool>(src), "R0.self.ref_crlf: reference %s.ref missing", key);
		if (!src) continue;
		const std::string raw((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
		std::string lf, crlf;
		for (char c : raw)
			if (c != '\r') lf.push_back(c);
		for (char c : lf) {
			if (c == '\n') crlf.push_back('\r');
			crlf.push_back(c);
		}
		PKV_CHECK_MSG(crlf.size() > lf.size(), "R0.self.ref_crlf: %s has no line to end with CRLF", key);
		RefFile parsed[2];
		const std::string* text[2] = {&lf, &crlf};
		for (int i = 0; i < 2; ++i) {
			const fs::path path = tmp / ("pkv_r0_ref_crlf_" + nonce + "_" + key + (i ? "_crlf" : "_lf") + ".ref");
			{
				std::ofstream out(path, std::ios::binary);
				out << *text[i];
				PKV_CHECK_MSG(static_cast<bool>(out), "R0.self.ref_crlf: cannot write %s", path.string().c_str());
			}
			std::ifstream in(path, std::ios::binary);
			ParseRef(in, &parsed[i]);
			in.close();
			fs::remove(path, ec);
		}
		const RefFile& a = parsed[0];
		const RefFile& b = parsed[1];
		PKV_CHECK_MSG(a.ok && a.fixture_sha.size() == 64, "R0.self.ref_crlf: %s (LF) parses, sha '%s'", key,
		              a.fixture_sha.c_str());
		// kills: a reader that keeps getline's '\r' (the sha gains a trailing '\r', 65 characters)
		PKV_CHECK_MSG(b.ok && b.fixture_sha == a.fixture_sha, "R0.self.ref_crlf: %s (CRLF) fixture sha '%s' != '%s'", key,
		              b.fixture_sha.c_str(), a.fixture_sha.c_str());
		PKV_CHECK_MSG(b.records.size() == a.records.size(), "R0.self.ref_crlf: %s record counts %zu (CRLF) vs %zu (LF)",
		              key, b.records.size(), a.records.size());
		int differ = 0;
		for (const auto& [k, rec] : a.records) {
			auto it = b.records.find(k);
			if (it == b.records.end() || !SameRecord(rec, it->second)) ++differ;
		}
		// kills: the same reader on a record with an empty token list ("tokens=\r" read as one token 0)
		PKV_CHECK_MSG(differ == 0, "R0.self.ref_crlf: %s: %d records differ between LF and CRLF", key, differ);
	}
}

PKV_CELL("R0.self.scenarios", "R0", SelfScenarios);
PKV_CELL("R0.self.pins", "R0", SelfPins);
PKV_CELL("R0.self.blob_rows", "R0", SelfBlobRows);
PKV_CELL("R0.self.admission_count", "R0", SelfAdmissionCount);
PKV_CELL("R0.self.ref_crlf", "R0", SelfRefCrlf);

}  // namespace
