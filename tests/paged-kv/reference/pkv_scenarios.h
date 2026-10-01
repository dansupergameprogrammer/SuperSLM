// Paged-KV plan (rev 16.1) §8 and step R0: the scenario scripts the byte-equality oracle and the
// paged-KV cells share.
//
// A scenario is a fixed sequence of legacy-verb calls (pool, create, prefill, decode, reset, prefix,
// adopt, save, restore) on one of tools/gen_paged_kv_fixture.py's models. Each stage it reaches
// emits one record: the stage name, the tokens decoded since the last record, the context length,
// the SHA-256 of the whole saved blob, and the SHA-256 of the K/V rows [0, context_length) read out
// of the blob by the flat layout's own formula (written here, not called from the engine).
//
// tools: pkv_reference.cpp compiles this header against a frozen tag's sslm_abi.h and writes the
// records to tests/paged-kv/reference/*.ref. The cells compile the same header against the paged
// build and compare record for record. Every verb used here exists unchanged from v1.8.1 to
// v1.11.0, so one script runs on the legacy-blob writer (v1.8.1, SSB4) and the reference (v1.11.0,
// SSB5) alike. The flat path through the new code is never the reference (§8).

#ifndef SUPERSLM_PKV_SCENARIOS_H
#define SUPERSLM_PKV_SCENARIOS_H

#include "superslm/sha256.h"
#include "superslm/sslm_abi.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace pkv {

// The model geometry the row extractor needs. Read from the blob's fixed header and the fixture's
// own constants: every pkv fixture has 2 layers and 2 KV heads (tools/gen_paged_kv_fixture.py).
struct Geometry {
	uint32_t layers;
	uint32_t kv_heads;
	uint32_t head_dim;
	int64_t context_cap;
};

struct Record {
	std::string stage;
	std::vector<int32_t> tokens;  // decoded since the previous record
	int64_t context_length = -1;  // -1: no save at this stage
	int64_t saturation = -1;      // the blob's aggregate saturation count, -1: no save
	std::string blob_sha;         // hex, or "-" when no save
	std::string rows_sha;         // hex of rows [0, context_length), or "-"
};

inline std::string Hex(const uint8_t* d, size_t n) {
	static const char* k = "0123456789abcdef";
	std::string s;
	for (size_t i = 0; i < n; ++i) {
		s.push_back(k[d[i] >> 4]);
		s.push_back(k[d[i] & 15]);
	}
	return s;
}

inline std::string Sha(const uint8_t* d, size_t n) {
	uint8_t out[32];
	superslm::Sha256Hash(d, n, out);
	return Hex(out, 32);
}

// Deterministic token streams: an LCG over the vocabulary, one stream per seed.
inline std::vector<int32_t> Stream(uint32_t seed, int32_t n, int32_t vocab) {
	std::vector<int32_t> t(static_cast<size_t>(n));
	uint32_t x = seed * 2654435761u + 12345u;
	for (int32_t i = 0; i < n; ++i) {
		x = x * 1664525u + 1013904223u;
		t[static_cast<size_t>(i)] = static_cast<int32_t>((x >> 8) % static_cast<uint32_t>(vocab));
	}
	return t;
}

// Context length is the fixed header's LE64 at offset 60 (magic 4, model_hash 32, kv_precision 4,
// schema_name_hash 8, dfa_walk_state 4, adapter_binding_id 8, then context_length), and the
// aggregate saturation count the LE64 at offset 92 (after layer_index 4, current_token 4 and
// hidden_scale 16): the same offsets in SSB4 and SSB5. The K/V block is the blob's last `block`
// bytes in both.
inline int64_t BlobLe64(const std::vector<uint8_t>& blob, size_t at) {
	int64_t v = 0;
	if (blob.size() < at + 8) return -1;
	std::memcpy(&v, blob.data() + at, sizeof v);
	return v;
}
inline int64_t BlobContextLength(const std::vector<uint8_t>& blob) { return BlobLe64(blob, 60); }
inline int64_t BlobSaturationCount(const std::vector<uint8_t>& blob) { return BlobLe64(blob, 92); }

// The K/V rows of positions [0, L) in position-major order (for each position: layer, K then V,
// KV head, head_dim bytes), addressed by the flat layout's formula
// (per-(layer, half, head)-major, position-minor) written here, independent of the engine.
inline std::vector<uint8_t> RowsFromBlock(const uint8_t* block, const Geometry& g, int64_t L) {
	std::vector<uint8_t> rows;
	rows.reserve(static_cast<size_t>(L) * g.layers * 2u * g.kv_heads * g.head_dim);
	const size_t half = static_cast<size_t>(g.context_cap) * g.kv_heads * g.head_dim;
	for (int64_t p = 0; p < L; ++p)
		for (uint32_t layer = 0; layer < g.layers; ++layer)
			for (uint32_t kv = 0; kv < 2; ++kv)
				for (uint32_t h = 0; h < g.kv_heads; ++h) {
					const uint8_t* r = block + layer * half * 2u + kv * half +
					                   static_cast<size_t>(h) * g.context_cap * g.head_dim +
					                   static_cast<size_t>(p) * g.head_dim;
					rows.insert(rows.end(), r, r + g.head_dim);
				}
	return rows;
}

// The scenario driver over the legacy verbs. `fail` records the first failing call and stops the
// scenario; a scenario that fails emits no further records.
struct Driver {
	sslm_model model = nullptr;
	Geometry geo{};
	int32_t vocab = 256;
	std::string fail;
	std::vector<Record> records;
	std::vector<int32_t> pending;  // tokens decoded since the last record

	bool Ok(sslm_status s, const char* what) {
		if (s == SSLM_OK) return true;
		if (fail.empty()) fail = std::string(what) + " -> " + std::to_string(static_cast<int>(s));
		return false;
	}

	// A legacy pool of `blocks` whole-sequence blocks, over 64-byte-aligned caller memory.
	struct Pool {
		sslm_kv_pool pool = nullptr;
		void* mem = nullptr;
		~Pool() {
			if (pool) sslm_kv_pool_destroy(pool);
			if (mem) ::operator delete(mem, std::align_val_t(64));
		}
	};
	bool MakePool(uint32_t blocks, Pool* out) {
		const size_t size = sslm_kv_block_size(model) * blocks + sslm_kv_pool_overhead_size(model, blocks);
		const size_t rounded = (size + 63) / 64 * 64;
		out->mem = ::operator new(rounded, std::align_val_t(64));  // std::aligned_alloc is absent on MSVC
		return Ok(sslm_kv_pool_create(model, out->mem, rounded, blocks, &out->pool), "pool_create");
	}

	bool Prefill(sslm_seq s, const std::vector<int32_t>& t, int32_t chunk) {
		size_t at = 0;
		while (at < t.size()) {
			int32_t consumed = 0;
			const int32_t n = static_cast<int32_t>(t.size() - at);
			if (!Ok(sslm_prefill(model, s, t.data() + at, n, chunk, SSLM_SPAN_PROMPT, nullptr, &consumed),
			        "prefill"))
				return false;
			if (consumed <= 0) return Ok(SSLM_INVALID_ARGUMENT, "prefill consumed 0");
			at += static_cast<size_t>(consumed);
		}
		return true;
	}

	bool PrefixPrefill(sslm_prefix p, const std::vector<int32_t>& t, int32_t chunk) {
		size_t at = 0;
		while (at < t.size()) {
			int32_t consumed = 0;
			const int32_t n = static_cast<int32_t>(t.size() - at);
			if (!Ok(sslm_prefix_prefill(model, p, t.data() + at, n, chunk, SSLM_SPAN_PROMPT, nullptr,
			                            &consumed),
			        "prefix_prefill"))
				return false;
			if (consumed <= 0) return Ok(SSLM_INVALID_ARGUMENT, "prefix_prefill consumed 0");
			at += static_cast<size_t>(consumed);
		}
		return true;
	}

	// `n` greedy tokens, one layer per call, appended to the pending record.
	bool Decode(sslm_seq s, int32_t n) {
		for (int32_t i = 0; i < n; ++i) {
			sslm_decode_params p{};
			p.layer_budget = 1;
			sslm_seq b[1] = {s};
			int32_t tok = -1;
			for (int guard = 0; tok < 0 && guard < 4096; ++guard)
				if (!Ok(sslm_decode_step(model, b, 1, &p, nullptr, &tok), "decode_step")) return false;
			if (tok < 0) return Ok(SSLM_INVALID_ARGUMENT, "decode produced no token");
			pending.push_back(tok);
		}
		return true;
	}

	bool Save(sslm_seq s, std::vector<uint8_t>* blob) {
		size_t need = 0;
		if (sslm_seq_save(s, nullptr, &need) != SSLM_BUFFER_TOO_SMALL || need == 0)
			return Ok(SSLM_INVALID_ARGUMENT, "save size query");
		blob->assign(need, 0);
		size_t wrote = blob->size();
		if (!Ok(sslm_seq_save(s, blob->data(), &wrote), "save")) return false;
		blob->resize(wrote);
		return true;
	}

	// Emit a record. With `s`, save it and digest the blob and its rows; without, tokens only.
	bool Mark(const char* stage, sslm_seq s, std::vector<uint8_t>* keep = nullptr) {
		if (!fail.empty()) return false;
		Record r;
		r.stage = stage;
		r.tokens.swap(pending);
		if (s) {
			std::vector<uint8_t> blob;
			if (!Save(s, &blob)) return false;
			const size_t block = sslm_kv_block_size(model);
			r.context_length = BlobContextLength(blob);
			r.saturation = BlobSaturationCount(blob);
			if (blob.size() < block || r.context_length < 0 || r.context_length > geo.context_cap)
				return Ok(SSLM_INVALID_ARGUMENT, "blob shape");
			r.blob_sha = Sha(blob.data(), blob.size());
			const std::vector<uint8_t> rows =
			    RowsFromBlock(blob.data() + blob.size() - block, geo, r.context_length);
			r.rows_sha = Sha(rows.data(), rows.size());
			if (keep) keep->swap(blob);
		} else {
			r.blob_sha = "-";
			r.rows_sha = "-";
		}
		records.push_back(std::move(r));
		return true;
	}
};

// Pinned blobs a scenario hands out (stage name -> the whole saved blob), for the 9.2-family cells.
struct Pinned {
	std::string stage;
	std::vector<uint8_t> blob;
};

// ---- the scenarios --------------------------------------------------------------------------

// 1.1 / 4.4 / 4.6 (the legacy half): one handle driven fresh -> prefill 100 -> reset -> prefill
// 40 -> reset -> adopt a 300-token prefix -> decode, crossing 16-position page boundaries at every
// stage (B = 16 on cap 4096; B = cap on pkv_odd).
inline void Lifecycle(Driver& d, std::vector<Pinned>* pins) {
	Driver::Pool pool;
	if (!d.MakePool(2, &pool)) return;
	sslm_prefix px = nullptr;
	sslm_seq s = nullptr;
	if (!d.Ok(sslm_prefix_begin(d.model, &pool.pool, &px), "prefix_begin")) return;
	if (!d.PrefixPrefill(px, Stream(3, 300, d.vocab), 64)) return;
	if (!d.Ok(sslm_prefix_freeze(px), "prefix_freeze")) return;
	if (!d.Ok(sslm_seq_create(d.model, &pool.pool, &s), "seq_create")) return;
	std::vector<uint8_t> kept;
	if (!d.Prefill(s, Stream(1, 100, d.vocab), 64) || !d.Mark("prefill100", s)) goto out;
	if (!d.Decode(s, 4) || !d.Mark("prefill100+decode4", s, &kept)) goto out;
	if (pins) pins->push_back({"prefill100+decode4", kept});
	if (!d.Decode(s, 8) || !d.Mark("continue8", nullptr)) goto out;
	if (!d.Ok(sslm_seq_reset(s), "reset") || !d.Mark("reset1", s)) goto out;
	if (!d.Prefill(s, Stream(2, 40, d.vocab), 16) || !d.Decode(s, 4) || !d.Mark("prefill40+decode4", s))
		goto out;
	if (!d.Ok(sslm_seq_reset(s), "reset") || !d.Ok(sslm_seq_adopt_prefix(s, px), "adopt") ||
	    !d.Mark("adopt300", s, &kept))
		goto out;
	if (pins) pins->push_back({"adopt300", kept});
	if (!d.Decode(s, 8) || !d.Mark("adopt300+decode8", s)) goto out;
	if (!d.Prefill(s, Stream(4, 37, d.vocab), 7) || !d.Decode(s, 4) || !d.Mark("adopt+prefill37+decode4", s))
		goto out;
out:
	if (s) sslm_seq_release(s);
	if (px) sslm_prefix_release(px);
}

// 6.1 (C2 engine, C4 legacy ABI): a long prefill in 64-token chunks, so every chunk crosses page
// boundaries, then decode. Rows [0, L) pinned by digest.
inline void LongPrefill(Driver& d, std::vector<Pinned>*) {
	Driver::Pool pool;
	if (!d.MakePool(1, &pool)) return;
	sslm_seq s = nullptr;
	if (!d.Ok(sslm_seq_create(d.model, &pool.pool, &s), "seq_create")) return;
	if (d.Prefill(s, Stream(11, 1500, d.vocab), 64) && d.Mark("prefill1500", s) && d.Decode(s, 16))
		d.Mark("prefill1500+decode16", s);
	sslm_seq_release(s);
}

// 10.1 / 10.2: ten sequences that share a 700-token preamble, each run with no sharing (the whole
// prompt prefilled into its own block), then 32 tokens decoded. Each sequence's tokens are one
// record. A second pass adopts the frozen preamble (v1.11.0 copies it) and must give the same
// tokens; it is recorded under its own stage so the reference itself shows the two agree.
inline void SharedPreamble(Driver& d, std::vector<Pinned>*) {
	Driver::Pool pool;
	if (!d.MakePool(2, &pool)) return;
	const std::vector<int32_t> pre = Stream(21, 700, d.vocab);
	for (int i = 0; i < 10; ++i) {
		std::vector<int32_t> prompt = pre;
		const std::vector<int32_t> suffix = Stream(100 + static_cast<uint32_t>(i), 20, d.vocab);
		prompt.insert(prompt.end(), suffix.begin(), suffix.end());
		sslm_seq s = nullptr;
		if (!d.Ok(sslm_seq_create(d.model, &pool.pool, &s), "seq_create")) return;
		const bool ok = d.Prefill(s, prompt, 64) && d.Decode(s, 32);
		std::string stage = "unshared" + std::to_string(i);
		if (ok) d.Mark(stage.c_str(), s);
		sslm_seq_release(s);
		if (!ok) return;
	}
	sslm_prefix px = nullptr;
	if (!d.Ok(sslm_prefix_begin(d.model, &pool.pool, &px), "prefix_begin")) return;
	if (d.PrefixPrefill(px, pre, 64) && d.Ok(sslm_prefix_freeze(px), "prefix_freeze")) {
		for (int i = 0; i < 10; ++i) {
			sslm_seq s = nullptr;
			if (!d.Ok(sslm_seq_create(d.model, &pool.pool, &s), "seq_create")) break;
			const bool ok = d.Ok(sslm_seq_adopt_prefix(s, px), "adopt") &&
			                d.Prefill(s, Stream(100 + static_cast<uint32_t>(i), 20, d.vocab), 64) &&
			                d.Decode(s, 32);
			std::string stage = "adopted" + std::to_string(i);
			if (ok) d.Mark(stage.c_str(), s);
			sslm_seq_release(s);
			if (!ok) break;
		}
	}
	sslm_prefix_release(px);
}

// 9.2 / 9.3 / 9.7: a saved blob and its continuation, for the restore side. The pinned blob is the
// state after prefill 100 + decode 4; the continuation is the next 8 tokens on the writer's binary.
// A restore-then-save round trip on the writer's own binary is recorded too.
inline void Persist(Driver& d, std::vector<Pinned>* pins) {
	Driver::Pool pool;
	if (!d.MakePool(2, &pool)) return;
	sslm_seq s = nullptr;
	if (!d.Ok(sslm_seq_create(d.model, &pool.pool, &s), "seq_create")) return;
	std::vector<uint8_t> blob;
	if (d.Prefill(s, Stream(31, 100, d.vocab), 64) && d.Decode(s, 4) && d.Mark("saved", s, &blob)) {
		if (pins) pins->push_back({"saved", blob});
		if (d.Decode(s, 8)) d.Mark("continuation8", nullptr);
		sslm_seq r = nullptr;
		if (d.Ok(sslm_seq_restore(d.model, &pool.pool, blob.data(), blob.size(), &r), "restore")) {
			if (d.Mark("restored", r) && d.Decode(r, 8)) d.Mark("restored+continuation8", nullptr);
			sslm_seq_release(r);
		}
	}
	sslm_seq_release(s);
}

// 9.7 (ii): a state with a nonzero aggregate saturation count. Tokens repeated in a short cycle
// drive the fixture's K-peaked calibration into saturation; the record carries the count so the
// reference shows it is nonzero.
inline void Saturating(Driver& d, std::vector<Pinned>* pins) {
	Driver::Pool pool;
	if (!d.MakePool(1, &pool)) return;
	sslm_seq s = nullptr;
	if (!d.Ok(sslm_seq_create(d.model, &pool.pool, &s), "seq_create")) return;
	std::vector<int32_t> prompt;
	for (int32_t t = 0; t < d.vocab && prompt.size() < 12; ++t) prompt.push_back(t);
	std::vector<uint8_t> blob;
	if (d.Prefill(s, prompt, static_cast<int32_t>(prompt.size())) && d.Decode(s, 4) &&
	    d.Mark("saturated", s, &blob)) {
		if (pins) pins->push_back({"saturated", blob});
		if (d.Decode(s, 8)) d.Mark("continuation8", nullptr);
	}
	sslm_seq_release(s);
}

// 4.1 / 6.1 (C2 engine, C4 legacy ABI): one fresh sequence per width W in {1, 15, 16, 17, 32,
// 1,064 (a 1,000-token prefix + 64), cap - 1, cap}: prefill W, record the rows; decode the ready
// token (it writes no row); below the cap decode one more, which writes row W and attends over
// W + 1 positions, and record the rows again. The 32k fixture stops at 1,064 (its cap-sized widths
// cost minutes and no cell asks for them).
inline void Widths(Driver& d, std::vector<Pinned>*) {
	Driver::Pool pool;
	if (!d.MakePool(1, &pool)) return;
	std::vector<int64_t> widths = {1, 15, 16, 17, 32, 1064};
	if (d.geo.context_cap <= 4100) {
		widths.push_back(d.geo.context_cap - 1);
		widths.push_back(d.geo.context_cap);
	}
	for (int64_t w : widths) {
		sslm_seq s = nullptr;
		if (!d.Ok(sslm_seq_create(d.model, &pool.pool, &s), "seq_create")) return;
		const std::string tag = "w" + std::to_string(w);
		bool ok = d.Prefill(s, Stream(41, static_cast<int32_t>(w), d.vocab), 64) && d.Mark((tag + "_prefill").c_str(), s) &&
		          d.Decode(s, 1) && d.Mark((tag + "_ready").c_str(), nullptr);
		if (ok && w < d.geo.context_cap) ok = d.Decode(s, 1) && d.Mark((tag + "_decode1").c_str(), s);
		sslm_seq_release(s);
		if (!ok) return;
	}
}

// 4.2 / 6.1 (the adopt half): a prefix of length P in {0, 1, 15, 16, 17, 1,000 (mid-page),
// 1,008 (aligned)} frozen and adopted by a fresh sequence, which decodes 8 tokens, then prefills 20
// more and decodes 4. P = 0 freezes an empty prefix; its adopter has no ready token (decode on it is
// refused, as on a fresh sequence), so it skips the first decode.
inline void PrefixLengths(Driver& d, std::vector<Pinned>*) {
	Driver::Pool pool;
	if (!d.MakePool(2, &pool)) return;
	for (int32_t len : {0, 1, 15, 16, 17, 1000, 1008}) {
		sslm_prefix px = nullptr;
		sslm_seq s = nullptr;
		if (!d.Ok(sslm_prefix_begin(d.model, &pool.pool, &px), "prefix_begin")) return;
		const std::string tag = "p" + std::to_string(len);
		bool ok = (len == 0 || d.PrefixPrefill(px, Stream(51, len, d.vocab), 64)) &&
		          d.Ok(sslm_prefix_freeze(px), "prefix_freeze") &&
		          d.Ok(sslm_seq_create(d.model, &pool.pool, &s), "seq_create") &&
		          d.Ok(sslm_seq_adopt_prefix(s, px), "adopt") && d.Mark((tag + "_adopted").c_str(), s) &&
		          (len == 0 || (d.Decode(s, 8) && d.Mark((tag + "_decode8").c_str(), s))) &&
		          d.Prefill(s, Stream(52, 20, d.vocab), 64) && d.Decode(s, 4) &&
		          d.Mark((tag + "_prefill20+decode4").c_str(), s);
		if (s) sslm_seq_release(s);
		sslm_prefix_release(px);
		if (!ok) return;
	}
}

// 10.1 / 10.4 / 10.6 (the cohort's no-sharing reference): ten prompts of a 1,000-token world plus a
// 200-token persona, each prefilled whole into its own block with no sharing, decoded to 1,456 (257
// tokens) and to 1,712 (256 more), the positions a budget-512 sequence adopting the persona reaches
// mid-budget and at its limit.
inline void Cohort(Driver& d, std::vector<Pinned>*) {
	Driver::Pool pool;
	if (!d.MakePool(1, &pool)) return;
	const std::vector<int32_t> world = Stream(61, 1000, d.vocab);
	for (int i = 0; i < 10; ++i) {
		std::vector<int32_t> prompt = world;
		const std::vector<int32_t> persona = Stream(71 + static_cast<uint32_t>(i), 200, d.vocab);
		prompt.insert(prompt.end(), persona.begin(), persona.end());
		sslm_seq s = nullptr;
		if (!d.Ok(sslm_seq_create(d.model, &pool.pool, &s), "seq_create")) return;
		const std::string tag = "persona" + std::to_string(i);
		const bool ok = d.Prefill(s, prompt, 64) && d.Decode(s, 257) && d.Mark((tag + "_1456").c_str(), s) &&
		                d.Decode(s, 256) && d.Mark((tag + "_1712").c_str(), s);
		sslm_seq_release(s);
		if (!ok) return;
	}
}

struct Scenario {
	const char* name;
	void (*run)(Driver&, std::vector<Pinned>*);
};

inline const Scenario* Scenarios(size_t* n) {
	static const Scenario k[] = {
	    {"lifecycle", Lifecycle}, {"long_prefill", LongPrefill}, {"shared_preamble", SharedPreamble},
	    {"persist", Persist},     {"saturating", Saturating},         {"widths", Widths},
	    {"prefix_lengths", PrefixLengths}, {"cohort", Cohort},
	};
	*n = sizeof k / sizeof k[0];
	return k;
}

// One record per line:
// `<scenario> <stage> L=<n> sat=<n> blob=<hex|-> rows=<hex|-> tokens=<a,b,...>`.
inline std::string FormatRecord(const char* scenario, const Record& r) {
	std::string s = std::string(scenario) + " " + r.stage + " L=" + std::to_string(r.context_length) +
	                " sat=" + std::to_string(r.saturation) + " blob=" + r.blob_sha +
	                " rows=" + r.rows_sha + " tokens=";
	for (size_t i = 0; i < r.tokens.size(); ++i) {
		if (i) s.push_back(',');
		s += std::to_string(r.tokens[i]);
	}
	return s;
}

}  // namespace pkv

#endif  // SUPERSLM_PKV_SCENARIOS_H
