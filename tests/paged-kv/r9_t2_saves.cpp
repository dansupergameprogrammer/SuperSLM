// Paged-KV plan rev 16.22 §24.6 (D-SLM8427, the T-3190 strike folded): legacy saves at a cap the page
// size does not divide. The repair keeps sslm_kv_block_size at 1.11.0's flat block at every cap and
// moves a legacy holder's page slack into sslm_kv_pool_overhead_size, so the legacy SSB5 keeps
// 1.9.0's size and bytes. These cells pin that at pkv_odd (cap 4,100, unpinned: B = 256, 17 pages,
// 252 positions of slack), where rev 16.21's grown block made the blob 96,768 bytes longer.
//
//   9.2b/T2   v1.11.0's and v1.8.1's pins of persist saved, lifecycle adopt300 and saturating saturated
//             at pkv_odd: restore, rows and continuation, the re-save byte-equal (or equal-length for
//             v1.8.1), a shared restore with 0 shared pages, and the forward digests over a 0xCD pool
//   9.5/T2:int16  a regression pin (green on 2ebe5e3): pkv_def at Int16 rests at the cap; the SSB5 is 3,162,464
//             bytes, sslm_seq_state_size 3,162,484, the margin exactly 20 (rev 16.24, D-SLM8443)
//   2.4/T2    the same pins restore into exactly block + overhead(1) bytes (1,671,208); one byte short
//             of each pin is refused with nothing drawn
//   7.11/T2   a legacy SSB5 is exactly 160 + hidden_size + 4 * history + the flat block, at L' in
//             {0, 1, B, 1,000}, on pkv_def (cap 4,096) and pkv_odd (cap 4,100)
//
// On 2ebe5e3 pkv_odd runs at B = cap (one page), which is v1.11.0's layout, so every save assertion
// here holds there; each cell first requires pkv_odd at the rule's B = 256 (its subject is a paged
// holder with a partial last page), and that requirement is what is red on 2ebe5e3. The save
// assertions are the ones a T2 build that sizes the legacy section by its pages fails (the strike's
// executed construction, reproduced by the plan's probes/legacy_save_census.py section C).

#include "pkv_r9.h"

#include <fstream>

namespace {

using namespace pkv;
using namespace pkv::r9;

struct Stage { const char* scenario; const char* stage; };
const Stage kStages[] = {{"persist", "saved"}, {"lifecycle", "adopt300"}, {"saturating", "saturated"}};
const char* const kTags[] = {"v1.11.0", "v1.8.1"};

bool AtRuleB(const Fixture& odd) {
	const int64_t got = sslm_kv_page_positions(odd.model);
	PKV_CHECK_MSG(got == 256, "pkv_odd runs at B = %lld; the cell's subject is the rule's B = 256 with a partial last page",
	              static_cast<long long>(got));
	return got == 256;
}

// The record after `scenario stage` in the .ref file's own order, for its continuation tokens.
std::vector<int32_t> ContinuationTokens(const char* tag, const char* stem, const char* scenario, const char* stage) {
	std::ifstream f(std::string(PKV_REFERENCE_DIR) + "/" + tag + "_" + stem + ".ref");
	std::string line;
	bool next = false;
	const std::string key = std::string(scenario) + " " + stage + " ";
	while (std::getline(f, line)) {
		if (next && line.rfind(std::string(scenario) + " ", 0) == 0) {
			const size_t at = line.find("tokens=");
			std::vector<int32_t> out;
			if (at == std::string::npos) return out;
			std::string list = line.substr(at + 7);
			size_t p = 0;
			while (p < list.size()) {
				const size_t q = list.find(',', p);
				const std::string t = list.substr(p, q == std::string::npos ? std::string::npos : q - p);
				if (!t.empty() && t != "-") out.push_back(std::stoi(t));
				if (q == std::string::npos) break;
				p = q + 1;
			}
			return out;
		}
		if (line.rfind(key, 0) == 0) next = true;
	}
	return {};
}

// A legacy pool sized by the verbs, as 9.2b states, over a 0xCD-filled buffer.
struct VerbSizedPool {
	AlignedBuf mem;
	sslm_kv_pool pool = nullptr;
	sslm_status status = SSLM_INVALID_ARGUMENT;
	VerbSizedPool(sslm_model model, uint32_t blocks) {
		mem.Reset(sslm_kv_block_size(model) * blocks + sslm_kv_pool_overhead_size(model, blocks), 0xCD);
		status = sslm_kv_pool_create(model, mem.p, mem.n, blocks, &pool);
	}
	~VerbSizedPool() {
		if (pool) sslm_kv_pool_destroy(pool);
	}
};

void Cell92b() {
	const Fixture& odd = GetFixture("pkv_odd");
	if (!odd.ok) return;
	AtRuleB(odd);
	const int64_t B = 256;
	for (const Stage& st : kStages) {
		const std::vector<uint8_t> v111 = PinFor("v1.11.0", "pkv_odd", st.scenario, st.stage);
		for (const char* tag : kTags) {
			const std::vector<uint8_t> pin = std::string(tag) == "v1.11.0" ? v111 : PinFor(tag, "pkv_odd", st.scenario, st.stage);
			const RefRecord* rec = RefLookup(Reference(tag, odd), st.scenario, st.stage);
			if (pin.empty() || !rec) continue;
			const std::vector<int32_t> cont = ContinuationTokens(tag, "pkv_odd", st.scenario, st.stage);
			VerbSizedPool pool(odd.model, 2);
			PKV_CHECK_EQ(pool.status, SSLM_OK);
			if (!pool.pool) continue;
			sslm_seq r = nullptr;
			const sslm_status rs = sslm_seq_restore(odd.model, &pool.pool, pin.data(), pin.size(), &r);
			// kills: a gate at ceil(cap/B) * page_bytes refuses the pin, short by 96,768 bytes
			PKV_CHECK_MSG(rs == SSLM_OK, "%s %s %s (%zu bytes) is refused with %s", tag, st.scenario, st.stage, pin.size(),
			              AbiStatusName(rs));
			if (!r) continue;
			std::vector<uint8_t> rows;
			PKV_CHECK_MSG(SeqRows(pool.pool, r, odd.geo, 1, B, rec->context_length, Rot::kRotated, &rows) &&
			                  Sha(rows.data(), rows.size()) == rec->rows_sha,
			              "%s %s %s: rows [0, %lld) read at B = 256 rotated are not the record's", tag, st.scenario, st.stage,
			              static_cast<long long>(rec->context_length));
			// the re-save before any decode: v1.11.0's bytes (a v1.8.1 pin: v1.11.0's length)
			const std::vector<uint8_t> resave = bc::Save(r);
			if (std::string(tag) == "v1.11.0")
				PKV_CHECK_MSG(resave == pin, "%s %s: the re-save is %zu bytes, not the pin's %zu bytes", st.scenario, st.stage,
				              resave.size(), pin.size());
			else
				PKV_CHECK_MSG(resave.size() == v111.size(), "%s %s from v1.8.1: re-save %zu bytes, v1.11.0's %zu", st.scenario,
				              st.stage, resave.size(), v111.size());
			PKV_CHECK_MSG(bc::DecodeN(odd.model, r, static_cast<int64_t>(cont.size())) == cont,
			              "%s %s %s: the continuation's %zu tokens differ from the record's", tag, st.scenario, st.stage,
			              cont.size());
			sslm_seq_release(r);
			// A6: restore_shared with a live frozen prefix handle restores it the same way, 0 shared pages
			sslm_prefix px = nullptr;
			if (sslm_prefix_begin(odd.model, &pool.pool, &px) == SSLM_OK) {
				PrefixPrefillAll(odd.model, px, Stream(51, 40, odd.vocab), 16);
				sslm_prefix_freeze(px);
				uint32_t shared = 99;
				sslm_seq r2 = nullptr;
				PKV_CHECK_EQ(sslm_seq_restore_shared(odd.model, &pool.pool, pin.data(), pin.size(), px, &r2, &shared), SSLM_OK);
				PKV_CHECK_EQ(shared, 0u);
				if (r2) {
					std::vector<uint8_t> rows2;
					PKV_CHECK(SeqRows(pool.pool, r2, odd.geo, 1, B, rec->context_length, Rot::kRotated, &rows2) && rows2 == rows);
					sslm_seq_release(r2);
				}
				sslm_prefix_release(px);
			}
		}
	}
	// And forward: every legacy save the paged build makes in these scenarios, over 0xCD pool buffers,
	// hashes to v1.11.0's digest for its stage. kills: a gather that copies page bytes past L'.
	for (const char* sc : {"persist", "lifecycle", "saturating"})
		ExpectMatchesReference("v1.11.0", odd, sc, RunScenario(odd, sc, 0xCD), Compare::kTokensRowsAndBlob);
}

void Cell24() {
	const Fixture& odd = GetFixture("pkv_odd");
	if (!odd.ok) return;
	AtRuleB(odd);
	const Lay y = LayoutOf(odd.geo, 1, 256);
	PKV_CHECK_EQ(y.PoolBytes(1), 1671208u);
	PKV_CHECK_EQ(sslm_kv_block_size(odd.model) + sslm_kv_pool_overhead_size(odd.model, 1), 1671208u);
	for (const Stage& st : kStages)
		for (const char* tag : kTags) {
			const std::vector<uint8_t> pin = PinFor(tag, "pkv_odd", st.scenario, st.stage);
			if (pin.empty()) continue;
			{
				BlockPoolOf pool(odd.model, 1, y);  // exactly 1,671,208 bytes
				PKV_CHECK_EQ(pool.status, SSLM_OK);
				sslm_seq r = nullptr;
				if (pool.pool)
					PKV_CHECK_MSG(sslm_seq_restore(odd.model, &pool.pool, pin.data(), pin.size(), &r) == SSLM_OK,
					              "%s %s %s does not restore into a one-block pool of 1,671,208 bytes", tag, st.scenario,
					              st.stage);
				if (r) sslm_seq_release(r);
			}
			{
				BlockPoolOf pool(odd.model, 1, y);
				sslm_seq r = nullptr;
				const std::vector<uint8_t> shorter(pin.begin(), pin.end() - 1);
				if (pool.pool) {
					PKV_CHECK_MSG(sslm_seq_restore(odd.model, &pool.pool, shorter.data(), shorter.size(), &r) ==
					                  SSLM_INVALID_ARGUMENT,
					              "%s %s %s one byte short is not refused", tag, st.scenario, st.stage);
					// nothing drawn: the one block still admits one legacy create
					sslm_status refusal = SSLM_OK;
					PKV_CHECK_EQ(CountCreates(odd.model, &pool.pool, 0, &refusal), 1);
				}
				if (r) sslm_seq_release(r);
			}
		}
}

void Cell711() {
	struct Fx { const char* stem; uint32_t D; };
	for (Fx f : {Fx{"pkv_def", 48}, Fx{"pkv_odd", 48}}) {
		const Fixture& fx = GetFixture(f.stem);
		if (!fx.ok) continue;
		const int64_t B = RuleB(f.D, static_cast<uint64_t>(fx.geo.context_cap), 1);
		if (std::string(f.stem) == "pkv_odd") AtRuleB(fx);
		const Lay y = LayoutOf(fx.geo, 1, B);
		PKV_CHECK_EQ(y.block_bytes, size_t{2} * fx.geo.context_cap * 2 * 48 * 2);  // L * cap * H_kv * D * 2 * elem
		for (int64_t L : {int64_t{0}, int64_t{1}, B, int64_t{1000}}) {
			BlockPoolOf pool(fx.model, 1, y);
			sslm_seq s = nullptr;
			if (!pool.pool || sslm_seq_create(fx.model, &pool.pool, &s) != SSLM_OK) {
				PKV_CHECK_MSG(false, "%s: legacy holder not made (%s)", f.stem, AbiStatusName(pool.status));
				continue;
			}
			if (L > 0) PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(41, static_cast<int32_t>(L), fx.vocab), 64), SSLM_OK);
			const std::vector<uint8_t> b = bc::Save(s);
			const uint64_t hist = Le64(b, 112);
			// kills: a section of ceil(cap/B) * page_bytes, 96,768 bytes longer at pkv_odd
			PKV_CHECK_MSG(IsMagic(b, "SSB5") && b.size() == 160 + 192 + 4 * hist + y.block_bytes,
			              "%s L' = %lld: the legacy save is %zu bytes; 160 + 192 + 4 * %llu + %zu owed", f.stem,
			              static_cast<long long>(L), b.size(), static_cast<unsigned long long>(hist), y.block_bytes);
			sslm_seq_release(s);
		}
	}
	PKV_CHECK_EQ(160u + 192u + LayoutOf(Geometry{2, 2, 48, 4100}, 1, 256).block_bytes, 1574752u);
}

// 9.3 [T2], the pkv_odd restore side (rev 16.22): the paged build's legacy save of persist saved,
// written for the OLD reader (tests/paged-kv/r9_old_reader.ps1 builds tests/paged-kv/pkv_r9_old_reader.cpp
// against the frozen v1.11.0 library, restores this file and continues it). The release's reader checks
// `n <` only, so it cannot see a grown section; 9.2b's digest is the separating assertion, and this
// cell's own digest check repeats it on the file it hands over.
void Cell93Write() {
	const Fixture& odd = GetFixture("pkv_odd");
	if (!odd.ok) return;
	AtRuleB(odd);
	const char* out_dir = std::getenv("SUPERSLM_PAGED_KV_R9_OUT");
	PKV_CHECK_MSG(out_dir && *out_dir, "SUPERSLM_PAGED_KV_R9_OUT is not set (tests/paged-kv/r9_old_reader.ps1 sets it)");
	if (!out_dir || !*out_dir) return;
	const RefRecord* rec = RefLookup(Reference("v1.11.0", odd), "persist", "saved");
	VerbSizedPool pool(odd.model, 1);
	sslm_seq s = nullptr;
	if (!pool.pool || sslm_seq_create(odd.model, &pool.pool, &s) != SSLM_OK) return;
	PKV_CHECK_EQ(PrefillAll(odd.model, s, Stream(31, 100, odd.vocab), 64), SSLM_OK);
	bc::DecodeN(odd.model, s, 4);
	const std::vector<uint8_t> blob = bc::Save(s);
	sslm_seq_release(s);
	PKV_CHECK_MSG(rec && Sha(blob.data(), blob.size()) == rec->blob_sha, "the paged save of persist saved is not v1.11.0's blob");
	std::ofstream f(std::string(out_dir) + "/pkv_odd_persist_saved_paged.ssb", std::ios::binary | std::ios::trunc);
	f.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
	PKV_CHECK_MSG(static_cast<bool>(f), "the blob could not be written to %s", out_dir);
}

// 9.5 [T2] Int16 (plan rev 16.24 §26.3, §26.8 item 4; D-SLM8443). A regression pin, green on 2ebe5e3: R9 does not
// change the formula, and T2 changes the sizing quantities around it (D-SLM8427). pkv_def's Int16 variant, a legacy
// holder driven by 9.5/C4's damped-greedy drive to rest at context_length = cap with history cap, then saved. Each
// side computed here:
//   the legacy SSB5 at rest:   160 + hidden + 4 * history + cap * L * 2 * H_kv * D * elem = 3,162,464
//   sslm_seq_state_size:       180 + hidden + 4 * cap     + cap * L * 2 * H_kv * D * elem = 3,162,484
// so the margin is exactly 20 (the SSB6 header's 20 more bytes: a 24-byte tail where SSB5 carries 4).
// Kills: the block term without elem (1,589,620, the bound fails by 1,572,844), and 1.11.0's state size (the bound
// holds, the exact 20 fails).
void Cell95Int16() {
	const Fixture& def = GetFixture("pkv_def");
	if (!def.ok) return;
	Fixture fx;
	Variant v;
	v.kv_precision = 1;
	sslm_status ms = SSLM_OK;
	PKV_CHECK_MSG(MakeVariant(def, v, &fx, &ms), "pkv_def at Int16 maps (%s)", AbiStatusName(ms));
	if (!fx.ok) return;
	const uint64_t cap = 4096, elem = 2, hidden = HiddenSize(fx);
	const uint64_t block = cap * fx.geo.layers * 2u * fx.geo.kv_heads * fx.geo.head_dim * elem;
	const uint64_t want_blob = 160 + hidden + 4 * cap + block, want_state = 180 + hidden + 4 * cap + block;
	PKV_CHECK_EQ(want_blob, 3162464u);
	PKV_CHECK_EQ(want_state, 3162484u);
	const size_t state = sslm_seq_state_size(fx.model);
	PKV_CHECK_MSG(state == want_state, "sslm_seq_state_size at Int16 is %zu, %llu owed", state,
	              static_cast<unsigned long long>(want_state));
	LegacyPool pool(fx.model, 1);
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(pool.status == SSLM_OK ? sslm_seq_create(fx.model, &pool.pool, &s) : pool.status, SSLM_OK);
	if (!s) return;
	const int32_t seed = 5;
	int32_t consumed = 0;
	PKV_CHECK_EQ(sslm_prefill(fx.model, s, &seed, 1, 1, SSLM_SPAN_PROMPT, nullptr, &consumed), SSLM_OK);
	sslm_decode_params p{};
	PKV_CHECK_EQ(sslm_decode_params_init(fx.model, SSLM_DECODE_MODE_DAMPED_GREEDY, static_cast<int32_t>(fx.geo.layers), &p),
	             SSLM_OK);
	sslm_seq b[1] = {s};
	int64_t emitted = 0;
	sslm_status st = SSLM_OK;
	while (emitted < static_cast<int64_t>(cap)) {  // the ready token, then whole tokens until it rests at the cap
		int32_t tok = -1;
		st = sslm_decode_step_v2(fx.model, b, 1, &p, nullptr, &tok);
		if (st != SSLM_OK) break;
		if (tok >= 0) ++emitted;
	}
	PKV_CHECK_MSG(st == SSLM_OK, "the damped drive was refused at %lld tokens (%s)", static_cast<long long>(emitted),
	              AbiStatusName(st));
	const std::vector<uint8_t> rest = bc::Save(s);
	PKV_CHECK_MSG(rest.size() > 120 && Le64(rest, 60) == cap && Le64(rest, 112) == cap,
	              "the holder does not rest at context_length = cap with history cap");
	PKV_CHECK_MSG(IsMagic(rest, "SSB5") && rest.size() == want_blob, "the resting legacy save is %zu bytes, %llu owed",
	              rest.size(), static_cast<unsigned long long>(want_blob));
	PKV_CHECK_MSG(state >= rest.size() && state - rest.size() == 20, "the margin is %lld, exactly 20 owed",
	              static_cast<long long>(state) - static_cast<long long>(rest.size()));
	sslm_seq_release(s);
}

PKV_CELL("9.2b/T2", "T2", Cell92b);
PKV_CELL("9.5/T2:int16", "T2", Cell95Int16);
PKV_CELL("9.3/T2:write", "T2", Cell93Write);
PKV_CELL("2.4/T2", "T2", Cell24);
PKV_CELL("7.11/T2", "T2", Cell711);

}  // namespace
