// Paged-KV plan (rev 16.1) step C1: the engine-level harness the C2 cells share (4.1, 4.6, 6.1,
// 11.1, 11.9).
//
// Engine-level cells drive the C++ layer loops directly, through the view overloads of
// pkv_engine_api.h, over pages the test owns. The fixture's C++ side (SslmModel::Load, MarshalLayer,
// the embed/final_norm/head resolution) is built here from pkv_common.h's GetFixture bytes, the way
// tests/test_decode_threading_d1.cpp and sslm_abi.cpp's BuildEngineCache build it, and a whole token
// is finished (final_norm, logits, the lowest-index argmax) as sslm_decode_step finishes one.
//
// Everything a verdict reads is the test's own: the page buffer is the test's memory, every row is
// addressed by §3.1's formula written below (never by calling KeyRow), and the reference is the R0
// `widths` scenario (rows_sha, saturation and tokens), recorded by v1.11.0's ABI. The flat path
// through the new code is never the reference (§8).
//
// The page table is scattered (a fixed shuffle of the physical pages), so a build that addresses
// page p at p * page_bytes, ignoring the table, reads and writes the wrong pages. One physical page
// past the mapped set is never in the table: it is the "foreign" page every cell reads raw to see
// that nothing landed outside the view. With one logical page the table is exactly {0} and
// page_bytes the whole block, which is the flat wrapper's own one-page view (§3.2 item 2).

#ifndef SUPERSLM_TESTS_PKV_ENGINE_HELPERS_H
#define SUPERSLM_TESTS_PKV_ENGINE_HELPERS_H

#include "pkv_common.h"
#include "pkv_engine_api.h"

#include "superslm/forward_sites.h"
#include "superslm/layer_marshal.h"
#include "superslm/model.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace pkv_engine {

using superslm::CarriedScale;
using superslm::GemmThreading;
using superslm::KvPageView;
using superslm::SequenceLayerState;
using superslm::SslmForwardStatus;

constexpr uint8_t kFill = 0x5A;   // every page buffer starts as this; a byte still 0x5A was never written
constexpr int32_t kChunk = 64;    // the widths scenario's prefill chunk (pkv_scenarios.h)
constexpr uint32_t kWidthsSeed = 41;

inline const char* StatusName(SslmForwardStatus s) { return superslm::SslmForwardStatusName(s); }

// ---- the fixture's C++ side -------------------------------------------------------------------

struct EngineFixture {
	const pkv::Fixture* fx = nullptr;
	bool ok = false;
	superslm::SslmModelView view;
	std::vector<superslm_marshal::LayerBacking> backings;
	std::vector<superslm::LayerWeights> layers;
	const int8_t* embed = nullptr;
	const int8_t* head = nullptr;
	CarriedScale embed_sc{};
	CarriedScale final_sc{};
	std::vector<int32_t> final_gain;

	const superslm::SslmModelConfig& C() const { return view.config; }
	uint32_t L() const { return view.config.num_hidden_layers; }
	size_t Hidden() const { return view.config.hidden_size; }
	size_t D() const { return view.config.head_dim; }
	size_t Hkv() const { return view.config.num_key_value_heads; }
	size_t QWidth() const { return size_t{view.config.num_attention_heads} * view.config.head_dim; }
	int64_t Cap() const { return static_cast<int64_t>(view.config.context_cap); }
	// §3.1's per-page product L * 2 * H_kv * B * D (int8 K/V), written here.
	size_t PageBytes(int64_t B) const { return size_t{L()} * 2u * Hkv() * static_cast<size_t>(B) * D(); }
	// forward_sites.cpp's direct_qk condition, read off the marshalled weights: every layer must take
	// the QkQ31Score branch on pkv_qk and the GemmInt8AccumulateRow branch on pkv_def (cell 4.6).
	bool DirectQk() const {
		bool all = !layers.empty();
		for (const auto& lw : layers)
			all = all && lw.k_norm_gain != nullptr && lw.k_channel_r_t != nullptr && lw.k_channel_e_t != nullptr &&
			      lw.k_channel_ratio != nullptr;
		return all;
	}
};

inline const EngineFixture& GetEngineFixture(const char* stem) {
	static std::map<std::string, std::unique_ptr<EngineFixture>> cache;
	auto it = cache.find(stem);
	if (it != cache.end()) {
		PKV_CHECK_MSG(it->second->ok, "engine fixture %s did not load (see its first failure)", stem);
		return *it->second;
	}
	auto e = std::make_unique<EngineFixture>();
	const pkv::Fixture& fx = pkv::GetFixture(stem);
	e->fx = &fx;
	if (fx.ok) {
		std::string err;
		const bool loaded =
		    superslm::SslmModel::Load(fx.bytes.p, fx.size, e->view, &err) == superslm::SslmModelStatus::Ok;
		PKV_CHECK_MSG(loaded, "SslmModel::Load(%s): %s", stem, err.c_str());
		bool ok = loaded && e->view.config.kv_precision == superslm::SslmKvPrecision::Int8;
		PKV_CHECK_MSG(!loaded || ok, "%s: the engine cells are written for int8 K/V", stem);
		if (ok) {
			const uint32_t n = e->view.config.num_hidden_layers;
			e->backings.resize(n);
			e->layers.resize(n);
			for (uint32_t l = 0; l < n && ok; ++l) {
				std::string merr;
				ok = superslm_marshal::MarshalLayer(e->view, l, e->view.config.num_attention_heads,
				                                    e->view.config.num_key_value_heads, e->backings[l], e->layers[l],
				                                    &merr);
				PKV_CHECK_MSG(ok, "MarshalLayer(%s, %u): %s", stem, l, merr.c_str());
			}
		}
		if (ok) {
			// sslm_abi.cpp's BuildEngineCache, mirrored.
			const superslm::SslmTensorView* embed_w = e->view.weights.Tensor("embed");
			const superslm::SslmTensorView* gain_w = e->view.weights.Tensor("final_norm.gain");
			const superslm::SslmTensorView* head_w =
			    e->view.config.tie_word_embeddings ? embed_w : e->view.weights.Tensor("lm_head");
			bool sc_ok = true;
			e->embed_sc = superslm_marshal::ReadCarriedScale(e->view.composition_constants, "embed", &sc_ok);
			e->final_sc = superslm_marshal::ReadCarriedScale(e->view.composition_constants, "final_norm", &sc_ok);
			ok = embed_w && gain_w && head_w && sc_ok;
			PKV_CHECK_MSG(ok, "%s: embed, final_norm.gain, head or a site constant is missing", stem);
			if (ok) {
				e->final_gain = superslm_marshal::WidenGainToInt32(*gain_w);
				e->embed = reinterpret_cast<const int8_t*>(embed_w->data);
				e->head = reinterpret_cast<const int8_t*>(head_w->data);
			}
		}
		e->ok = ok;
	}
	const EngineFixture& ref = *e;
	cache[stem] = std::move(e);
	return ref;
}

// ---- the test's own pages ---------------------------------------------------------------------

// `logical` = ceil(cap / B) pages addressed through `table` (a fixed shuffle of physical pages
// [0, logical)); physical page `logical` is the foreign page, never in the table. All of it is
// the test's memory, pre-filled with kFill.
struct PagedKv {
	int64_t B = 0;
	size_t D = 0, H = 0;
	uint32_t L = 0;
	uint32_t logical = 0;
	size_t page_bytes = 0;
	pkv::AlignedBuf mem;
	std::vector<uint32_t> table;

	PagedKv(const EngineFixture& e, int64_t page_positions) {
		B = page_positions;
		D = e.D();
		H = e.Hkv();
		L = e.L();
		logical = static_cast<uint32_t>((e.Cap() + B - 1) / B);
		page_bytes = e.PageBytes(B);
		mem.Reset(page_bytes * (size_t{logical} + 1), kFill);
		table.resize(logical);
		for (uint32_t i = 0; i < logical; ++i) table[i] = i;
		uint32_t x = 2463534242u;  // Fisher-Yates under a fixed xorshift: deterministic, far from identity
		for (uint32_t i = logical; i > 1; --i) {
			x ^= x << 13;
			x ^= x >> 17;
			x ^= x << 5;
			std::swap(table[i - 1], table[x % i]);
		}
	}
	uint32_t ForeignPage() const { return logical; }
	uint8_t* Page(uint32_t physical) { return mem.p + size_t{physical} * page_bytes; }
	// The pages a call that ends at `positions` must see mapped: ceil(positions / B).
	uint32_t PagesFor(int64_t positions) const { return static_cast<uint32_t>((positions + B - 1) / B); }
	KvPageView View(uint32_t mapped) const {
		return KvPageView{mem.p, table.data(), mapped, B, page_bytes};
	}
	// §3.1's address of (l, half, h, pos, 0), written here:
	//   pool_base + table[pos / B] * page_bytes + ((l*2 + half)*H_kv + h)*B*D + (pos % B)*D.
	size_t Offset(uint32_t l, uint32_t half, size_t h, int64_t pos) const {
		return size_t{table[static_cast<size_t>(pos / B)]} * page_bytes +
		       ((size_t{l} * 2u + half) * H + h) * static_cast<size_t>(B) * D + static_cast<size_t>(pos % B) * D;
	}
	const uint8_t* Row(uint32_t l, uint32_t half, size_t h, int64_t pos) const { return mem.p + Offset(l, half, h, pos); }
	// Rows [0, n) in pkv_scenarios.h's position-major order (position, layer, K then V, head), so the
	// digest compares with the reference's rows_sha.
	std::vector<uint8_t> RowsPositionMajor(int64_t n) const {
		std::vector<uint8_t> rows;
		rows.reserve(static_cast<size_t>(n) * L * 2u * H * D);
		for (int64_t p = 0; p < n; ++p)
			for (uint32_t l = 0; l < L; ++l)
				for (uint32_t half = 0; half < 2; ++half)
					for (size_t h = 0; h < H; ++h) {
						const uint8_t* r = Row(l, half, h, p);
						rows.insert(rows.end(), r, r + D);
					}
		return rows;
	}
	// Every byte that is not a row of [0, n) still holds kFill: nothing was written outside the rows
	// the calls owned, in a mapped page or the foreign one. Returns the first offending offset, or -1.
	long long FirstStrayByte(int64_t n) const {
		std::vector<uint8_t> owned(mem.n, 0);
		for (int64_t p = 0; p < n; ++p)
			for (uint32_t l = 0; l < L; ++l)
				for (uint32_t half = 0; half < 2; ++half)
					for (size_t h = 0; h < H; ++h) std::memset(owned.data() + Offset(l, half, h, p), 1, D);
		for (size_t i = 0; i < mem.n; ++i)
			if (!owned[i] && mem.p[i] != kFill) return static_cast<long long>(i);
		return -1;
	}
	std::vector<uint8_t> Snapshot() const { return std::vector<uint8_t>(mem.p, mem.p + mem.n); }
	bool Equals(const std::vector<uint8_t>& snap) const {
		return snap.size() == mem.n && std::memcmp(snap.data(), mem.p, mem.n) == 0;
	}
};

// ---- whole tokens through the view overloads -----------------------------------------------------

// The single-token view loop over every layer, as sslm_decode_step calls the flat one (option G off).
inline SslmForwardStatus StepToken(const EngineFixture& e, SequenceLayerState& seq, const KvPageView& view,
                                   size_t token_index) {
	return superslm::RunLayerLoop(seq, e.layers.data(), e.L(), e.L(), e.Hidden(), e.D(), e.Hkv(), e.C().intermediate_size,
	                              e.Cap(), e.view.rope_tables, view, superslm::OptionGKLandingMode::kLegacy, {}, token_index,
	                              nullptr, e.QWidth(), GemmThreading{});
}

// The chunk view loop, as sslm_prefill calls the flat one.
inline SslmForwardStatus StepChunk(const EngineFixture& e, int8_t* codes, CarriedScale* scales, size_t n,
                                   int64_t start, const KvPageView& view, SequenceLayerState& counters) {
	return superslm::RunLayerLoopChunkBatched(
	    codes, scales, n, e.layers.data(), e.L(), e.Hidden(), e.D(), e.Hkv(), e.C().intermediate_size, e.Cap(), start,
	    e.view.rope_tables, view, false, &counters.kv_saturation_count, {}, nullptr, e.QWidth(),
	    &counters.kv_landing_saturation_count, &counters.k_channel_landing_saturation_count,
	    &counters.rope_q_saturation_count, &counters.rope_k_saturation_count, GemmThreading{});
}

inline SslmForwardStatus Embed(const EngineFixture& e, int32_t token, int8_t* out, CarriedScale* scale) {
	return superslm::EmbedEntry(token, static_cast<int32_t>(e.C().vocab_size), e.embed, e.Hidden(), e.embed_sc, out,
	                            scale);
}

// sslm_decode_step's finish for a greedy, schema-free sequence: final_norm, logits, the lowest-index
// argmax. -1 on a rejection (in *st).
inline int32_t Finish(const EngineFixture& e, const int8_t* hidden, CarriedScale scale, SslmForwardStatus* st) {
	std::vector<int8_t> final_codes(e.Hidden());
	CarriedScale fs{};
	*st = superslm::RmsNormSite(hidden, e.final_gain.data(), e.Hidden(), scale, e.final_sc, final_codes.data(), &fs,
	                            "final_norm");
	if (*st != SslmForwardStatus::Ok) return -1;
	const size_t vocab = e.C().vocab_size;
	std::vector<int64_t> wide(vocab);
	std::vector<int32_t> logits(vocab);
	*st = superslm::LogitsSite(final_codes.data(), e.Hidden(), e.head, vocab, wide.data(), logits.data());
	if (*st != SslmForwardStatus::Ok) return -1;
	return superslm::ArgmaxLowestIndexTieBreak(logits.data(), vocab);
}

// A live sequence over the test's pages: its own hidden row, and the counters both loops feed.
struct Seq {
	std::vector<int8_t> hidden;
	SequenceLayerState st;
	explicit Seq(const EngineFixture& e) : hidden(e.Hidden(), 0) { st.hidden_codes = hidden.data(); }
};

// ---- the widths matrix (4.1 / 4.6 / 6.1) --------------------------------------------------------

enum class Drive { kChunked, kPerToken };
inline const char* DriveName(Drive d) { return d == Drive::kChunked ? "chunked" : "per-token"; }

// 4.1's widths, from the fixture's own B and cap: 1, B-1, B, B+1, 2B, prefix + 64 (the 1,000-token
// prefix of 4.2), cap-1, cap. These are exactly the widths scenario's (pkv_scenarios.h).
inline std::vector<int64_t> Widths(const pkv::Fixture& fx) {
	const int64_t B = fx.B(), cap = fx.geo.context_cap;
	return {1, B - 1, B, B + 1, 2 * B, 1000 + 64, cap - 1, cap};
}

// The views every engine cell runs at (§9, C2's exit: "engine cells green at B = 4 and 16"), plus the
// one-page view (B = cap, table {0}), which is the flat wrappers' own view (§3.1's identity).
inline std::vector<int64_t> ViewPositions(const EngineFixture& e) { return {4, 16, e.Cap()}; }

// One width, one view, one drive: what the reference's `widths` records hold.
struct WidthRun {
	bool ran = false;
	std::string fail;            // the first unexpected status, with where
	int32_t ready = -1;          // the token after the prefill (writes no row)
	int32_t decode1 = -1;        // the token after one more step (writes row W); W < cap only
	std::string rows_prefill;    // digest of rows [0, W)
	std::string rows_decode1;    // digest of rows [0, W + 1); W < cap only
	int64_t sat_prefill = -1;
	int64_t sat_decode1 = -1;
	int64_t final_length = -1;   // context length after the run
	long long stray = -1;        // first byte outside rows [0, final_length) that is not kFill
	bool accessors_ok = true;    // the view accessors address what §3.1's formula addresses
	std::string accessor_fail;
	// W == cap only: the write past the cap is refused KvCapacityExhausted with nothing changed.
	bool cap_refused = true;
	std::string cap_fail;
};

inline void Note(std::string* s, const std::string& what) {
	if (s->empty()) *s = what;
}

// The view accessors against the formula, at the first, page-edge and last mapped positions.
inline void CheckAccessors(const EngineFixture& e, const PagedKv& kv, const KvPageView& view, int64_t n, WidthRun* r) {
	std::vector<int64_t> positions = {0, kv.B - 1, kv.B, n - 1};
	for (int64_t pos : positions) {
		if (pos < 0 || pos >= n) continue;
		for (uint32_t l = 0; l < e.L(); ++l)
			for (size_t h = 0; h < e.Hkv(); ++h) {
				const int8_t* k = superslm::KeyRow(view, l, e.Hkv(), e.D(), h, pos);
				const int8_t* v = superslm::ValueRow(view, l, e.Hkv(), e.D(), h, pos);
				int8_t* mk = superslm::MutableKeyRow(view, l, e.Hkv(), e.D(), h, pos);
				int8_t* mv = superslm::MutableValueRow(view, l, e.Hkv(), e.D(), h, pos);
				const int8_t* fk = reinterpret_cast<const int8_t*>(kv.Row(l, 0, h, pos));
				const int8_t* fv = reinterpret_cast<const int8_t*>(kv.Row(l, 1, h, pos));
				if (k != fk || v != fv || mk != fk || mv != fv) {
					r->accessors_ok = false;
					Note(&r->accessor_fail, "B=" + std::to_string(kv.B) + " l=" + std::to_string(l) + " h=" +
					                            std::to_string(h) + " pos=" + std::to_string(pos));
				}
			}
	}
}

// Prefill W tokens of the widths scenario's stream, finish the ready token, then step it (writing
// row W) and finish again; at W == cap, the step is refused instead. Each call sees exactly the
// pages its writes need mapped (ceil((start + n) / B)), so the coverage guard runs at its boundary
// on every call.
inline WidthRun RunWidthUncached(const EngineFixture& e, int64_t B, Drive drive, int64_t W) {
	WidthRun r;
	r.ran = true;
	PagedKv kv(e, B);
	Seq s(e);
	const int64_t cap = e.Cap();
	const std::vector<int32_t> tokens =
	    pkv::Stream(kWidthsSeed, static_cast<int32_t>(W), static_cast<int32_t>(e.C().vocab_size));
	const size_t hid = e.Hidden();
	SslmForwardStatus st = SslmForwardStatus::Ok;
	auto where = [&](const char* what, int64_t at) {
		return std::string(what) + " at " + std::to_string(at) + ": " + StatusName(st);
	};
	if (drive == Drive::kChunked) {
		std::vector<int8_t> codes;
		std::vector<CarriedScale> scales;
		for (int64_t at = 0; at < W && r.fail.empty(); at += kChunk) {
			const size_t n = static_cast<size_t>(std::min<int64_t>(kChunk, W - at));
			codes.assign(n * hid, 0);
			scales.assign(n, CarriedScale{});
			for (size_t i = 0; i < n && st == SslmForwardStatus::Ok; ++i)
				st = Embed(e, tokens[static_cast<size_t>(at) + i], codes.data() + i * hid, &scales[i]);
			if (st == SslmForwardStatus::Ok)
				st = StepChunk(e, codes.data(), scales.data(), n, at, kv.View(kv.PagesFor(at + static_cast<int64_t>(n))), s.st);
			if (st != SslmForwardStatus::Ok) {
				Note(&r.fail, where("prefill chunk", at));
				break;
			}
			// sslm_prefill: the sequence rests at the last token's final hidden state.
			std::memcpy(s.hidden.data(), codes.data() + (n - 1) * hid, hid);
			s.st.hidden_scale = scales[n - 1];
			s.st.layer_index = 0;
			s.st.context_length = at + static_cast<int64_t>(n);
		}
	} else {
		for (int64_t i = 0; i < W && r.fail.empty(); ++i) {
			st = Embed(e, tokens[static_cast<size_t>(i)], s.hidden.data(), &s.st.hidden_scale);
			s.st.layer_index = 0;
			if (st == SslmForwardStatus::Ok) st = StepToken(e, s.st, kv.View(kv.PagesFor(i + 1)), static_cast<size_t>(i));
			if (st != SslmForwardStatus::Ok) Note(&r.fail, where("token", i));
			else if (s.st.context_length != i + 1)
				Note(&r.fail, "token " + std::to_string(i) + " left context_length " + std::to_string(s.st.context_length));
		}
	}
	if (!r.fail.empty()) return r;
	{
		const std::vector<uint8_t> rows = kv.RowsPositionMajor(W);
		r.rows_prefill = pkv::Sha(rows.data(), rows.size());
		r.sat_prefill = static_cast<int64_t>(s.st.kv_saturation_count);
	}
	r.ready = Finish(e, s.hidden.data(), s.st.hidden_scale, &st);
	if (r.ready < 0) {
		Note(&r.fail, where("finish (ready)", W));
		return r;
	}
	st = Embed(e, r.ready, s.hidden.data(), &s.st.hidden_scale);
	s.st.layer_index = 0;
	if (st != SslmForwardStatus::Ok) {
		Note(&r.fail, where("embed (ready)", W));
		return r;
	}
	if (W < cap) {
		st = StepToken(e, s.st, kv.View(kv.PagesFor(W + 1)), static_cast<size_t>(W));
		if (st != SslmForwardStatus::Ok) {
			Note(&r.fail, where("decode step", W));
			return r;
		}
		r.decode1 = Finish(e, s.hidden.data(), s.st.hidden_scale, &st);
		if (r.decode1 < 0) Note(&r.fail, where("finish (decode1)", W + 1));
		const std::vector<uint8_t> rows = kv.RowsPositionMajor(W + 1);
		r.rows_decode1 = pkv::Sha(rows.data(), rows.size());
		r.sat_decode1 = static_cast<int64_t>(s.st.kv_saturation_count);
	} else {
		// The last admissible write was row cap - 1. Every page mapped: the step past the cap is
		// refused by KvCapacityExhausted (not the coverage guard), nothing written, the sequence
		// as it was; and the same for a one-token chunk at the cap.
		const std::vector<uint8_t> before = kv.Snapshot();
		const std::vector<int8_t> hidden_before = s.hidden;
		st = StepToken(e, s.st, kv.View(kv.logical), static_cast<size_t>(W));
		if (st != SslmForwardStatus::KvCapacityExhausted || s.st.context_length != cap || s.st.layer_index != 0 ||
		    s.hidden != hidden_before || !kv.Equals(before)) {
			r.cap_refused = false;
			Note(&r.cap_fail, std::string("single-token step at the cap: ") + StatusName(st));
		}
		std::vector<int8_t> one(s.hidden);
		CarriedScale one_scale = s.st.hidden_scale;
		SequenceLayerState counters;
		st = StepChunk(e, one.data(), &one_scale, 1, cap, kv.View(kv.logical), counters);
		if (st != SslmForwardStatus::KvCapacityExhausted || !kv.Equals(before)) {
			r.cap_refused = false;
			Note(&r.cap_fail, std::string("one-token chunk at the cap: ") + StatusName(st));
		}
	}
	r.final_length = s.st.context_length;
	r.stray = kv.FirstStrayByte(r.final_length);
	CheckAccessors(e, kv, kv.View(kv.PagesFor(r.final_length)), r.final_length, &r);
	return r;
}

// Runs are shared by the cells that grade them (4.1 tokens and statuses, 4.6 both score branches,
// 6.1 bytes), so the matrix is computed once per process.
inline const WidthRun& RunWidth(const EngineFixture& e, int64_t B, Drive drive, int64_t W) {
	static std::map<std::tuple<std::string, int64_t, int, int64_t>, WidthRun> cache;
	const auto key = std::make_tuple(e.fx->stem, B, static_cast<int>(drive), W);
	auto it = cache.find(key);
	if (it == cache.end()) it = cache.emplace(key, RunWidthUncached(e, B, drive, W)).first;
	return it->second;
}

// The reference's three records for width W ("w<W>_prefill", "w<W>_ready", "w<W>_decode1").
struct WidthRef {
	const pkv::RefRecord* prefill = nullptr;
	const pkv::RefRecord* ready = nullptr;
	const pkv::RefRecord* decode1 = nullptr;  // absent at W == cap
};
inline WidthRef LookupWidth(const pkv::Fixture& fx, int64_t W) {
	const pkv::RefFile& ref = pkv::Reference("v1.11.0", fx);
	const std::string tag = "w" + std::to_string(W);
	WidthRef w;
	w.prefill = pkv::RefLookup(ref, "widths", tag + "_prefill");
	w.ready = pkv::RefLookup(ref, "widths", tag + "_ready");
	if (W < fx.geo.context_cap) w.decode1 = pkv::RefLookup(ref, "widths", tag + "_decode1");
	return w;
}

}  // namespace pkv_engine

#endif  // SUPERSLM_TESTS_PKV_ENGINE_HELPERS_H
