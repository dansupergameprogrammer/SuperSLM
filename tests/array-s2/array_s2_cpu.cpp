// array_s2_cpu.cpp -- the CPU side of the S2 array harness, on the public C ABI only
// (superslm/sslm_abi.h). See array_s2_shared.h for the interface.
#include <chrono>
#include <memory>

#include "superslm/model.h"  // SslmModel::Load, for the layer count only
#include "superslm/sslm_abi.h"
#include "array_s2_shared.h"

#ifdef SUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS
// The schema fast-decode test seam (src/sslm_abi.cpp): a hidden-size bound of 0 clears the
// model's masked-finish flag at map time, so every bound decode runs the full finish.
namespace superslm::test::sfd {
void SetExactnessHiddenBound(size_t bound) noexcept;
void ResetExactnessHiddenBound() noexcept;
}  // namespace superslm::test::sfd
#endif

namespace {

constexpr uint32_t kBlocks = 2;
constexpr int32_t kJumpBudget = 64;

struct Aligned {
	std::vector<uint8_t> raw;
	void* p = nullptr;
	void Reset(size_t bytes) {
		raw.assign(bytes + SSLM_ABI_ALIGNMENT_BYTES, 0);
		p = raw.data();
		size_t space = raw.size();
		std::align(SSLM_ABI_ALIGNMENT_BYTES, bytes, p, space);
	}
};

double MsSince(std::chrono::steady_clock::time_point t0) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string Err(const char* call, sslm_status st) { return std::string("error:") + call + " " + std::to_string(st); }

}  // namespace

struct CpuSession {
	sslm_model model = nullptr;
	sslm_schema schema = nullptr;
	sslm_kv_pool pool = nullptr;
	sslm_workspace ws = nullptr;
	Aligned pool_buf, ws_buf;
	int32_t layers = 0;
};

bool CpuSeamAvailable() {
#ifdef SUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS
	return true;
#else
	return false;
#endif
}

CpuSession* CpuOpen(const uint8_t* bytes, size_t n, const std::string& schema, bool full_finish,
                    int32_t max_chunk, MapResult* map) {
	auto s = std::make_unique<CpuSession>();
#ifdef SUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS
	if (full_finish) superslm::test::sfd::SetExactnessHiddenBound(0);
#else
	if (full_finish) {
		map->error = "full finish needs the seam build";
		return nullptr;
	}
#endif
	const auto t0 = std::chrono::steady_clock::now();
	sslm_status st = sslm_model_map(bytes, n, &s->model);
	map->map_ms = MsSince(t0);
#ifdef SUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS
	superslm::test::sfd::ResetExactnessHiddenBound();
#endif
	if (st != SSLM_OK) {
		map->error = Err("sslm_model_map", st);
		return nullptr;
	}
	if (!schema.empty()) {
		st = sslm_schema_lookup(s->model, schema.c_str(), &s->schema);
		if (st != SSLM_OK) {
			map->error = Err("sslm_schema_lookup", st);
			sslm_model_unmap(s->model);
			return nullptr;
		}
	}
	const size_t pool_bytes = sslm_kv_block_size(s->model) * kBlocks + sslm_kv_pool_overhead_size(s->model, kBlocks);
	s->pool_buf.Reset(pool_bytes);
	st = sslm_kv_pool_create(s->model, s->pool_buf.p, pool_bytes, kBlocks, &s->pool);
	if (st != SSLM_OK) {
		map->error = Err("sslm_kv_pool_create", st);
		sslm_model_unmap(s->model);
		return nullptr;
	}
	{
		// The C ABI has no layer-count query; the artifact's own Config carries it.
		superslm::SslmModelView view;
		std::string load_err;
		if (superslm::SslmModel::Load(bytes, n, view, &load_err) != superslm::SslmModelStatus::Ok) {
			map->error = "SslmModel::Load: " + load_err;
			CpuClose(s.release());
			return nullptr;
		}
		s->layers = static_cast<int32_t>(view.config.num_hidden_layers);
	}
	sslm_config config{};
	config.max_batch = 1;
	config.max_chunk_budget = max_chunk;
	config.max_layer_budget = s->layers;
	const size_t ws_bytes = sslm_workspace_size(s->model, &config);
	s->ws_buf.Reset(ws_bytes);
	st = sslm_workspace_create(s->model, &config, s->ws_buf.p, ws_bytes, &s->ws);
	if (st != SSLM_OK) {
		map->error = Err("sslm_workspace_create", st);
		CpuClose(s.release());
		return nullptr;
	}
	map->ok = true;
	return s.release();
}

void CpuClose(CpuSession* s) {
	if (!s) return;
	if (s->ws) sslm_workspace_destroy(s->ws);
	if (s->pool) sslm_kv_pool_destroy(s->pool);
	if (s->model) sslm_model_unmap(s->model);
	delete s;
}

namespace {

int32_t Accepting(CpuSession* s, sslm_seq seq) {
	sslm_stats_out out{};
	return sslm_stats(s->model, seq, &out) == SSLM_OK ? out.schema_accepting : -1;
}

sslm_status NewBoundSeq(CpuSession* s, sslm_seq* seq) {
	sslm_status st = sslm_seq_create(s->model, &s->pool, seq);
	if (st != SSLM_OK) return st;
	st = sslm_seq_set_schema(*seq, s->schema);
	if (st != SSLM_OK) {
		sslm_seq_release(*seq);
		*seq = nullptr;
	}
	return st;
}

sslm_status DecodeOne(CpuSession* s, sslm_seq seq, int32_t* token) {
	sslm_decode_params params{};
	params.struct_size = sizeof(params);
	params.layer_budget = s->layers;
	sslm_seq seqs[1] = {seq};
	*token = -1;
	sslm_status st = SSLM_OK;
	for (int guard = 0; *token == -1 && guard < 64; ++guard) {
		st = sslm_decode_step(s->model, seqs, 1, &params, s->ws, token);
		if (st != SSLM_OK) break;
	}
	return st;
}

// Decodes from a sequence ready to decode until the dead end (-2) or until `out` holds max_tokens,
// recording the accepting flag after each token.
void DecodeTrace(CpuSession* s, sslm_seq seq, int32_t max_tokens, Trace* out) {
	while (static_cast<int32_t>(out->ids.size()) < max_tokens) {
		int32_t t = -1;
		const sslm_status st = DecodeOne(s, seq, &t);
		if (st != SSLM_OK) {
			out->end = Err("sslm_decode_step", st);
			return;
		}
		if (t == -2) {
			out->end = "dead_end";
			return;
		}
		out->ids.push_back(t);
		out->accepting.push_back(Accepting(s, seq));
	}
	out->end = "max_tokens";
}

// Teacher-forces `ids` one token per schema-content prefill call, recording each admitted token.
void ForceTrace(CpuSession* s, sslm_seq seq, const int32_t* ids, size_t n, Trace* out) {
	for (size_t i = 0; i < n; ++i) {
		int32_t consumed = 0;
		const sslm_status st = sslm_prefill(s->model, seq, &ids[i], 1, 1, SSLM_SPAN_SCHEMA_CONTENT, s->ws, &consumed);
		if (st != SSLM_OK || consumed != 1) {
			out->end = st == SSLM_SCHEMA_SPAN_UNREACHABLE ? "refused" : Err("sslm_prefill", st);
			return;
		}
		out->ids.push_back(ids[i]);
		out->accepting.push_back(Accepting(s, seq));
	}
	out->end = "forced";
}

sslm_status PrefillPrompt(CpuSession* s, sslm_seq seq, const std::vector<int32_t>& prompt) {
	int32_t consumed = 0;
	const sslm_status st = sslm_prefill(s->model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                                    static_cast<int32_t>(prompt.size()), SSLM_SPAN_PROMPT, s->ws, &consumed);
	return st == SSLM_OK && consumed != static_cast<int32_t>(prompt.size()) ? SSLM_INVALID_ARGUMENT : st;
}

bool SaveBlob(sslm_seq seq, std::vector<uint8_t>* blob, std::string* error) {
	size_t n = 0;
	sslm_seq_save(seq, nullptr, &n);
	blob->assign(n, 0);
	const sslm_status st = sslm_seq_save(seq, blob->data(), &n);
	if (st != SSLM_OK) {
		*error = Err("sslm_seq_save", st);
		return false;
	}
	blob->resize(n);
	return true;
}

}  // namespace

WalkResult CpuWalk(CpuSession* s, const std::vector<int32_t>& ids) {
	WalkResult r;
	sslm_seq seq = nullptr;
	sslm_status st = NewBoundSeq(s, &seq);
	if (st != SSLM_OK) {
		r.refusal = Err("bind", st);
		return r;
	}
	for (size_t i = 0; i < ids.size(); ++i) {
		int32_t consumed = 0;
		st = sslm_prefill(s->model, seq, &ids[i], 1, 1, SSLM_SPAN_SCHEMA_CONTENT, s->ws, &consumed);
		if (st != SSLM_OK || consumed != 1) {
			r.refused_at = static_cast<int32_t>(i);
			r.refusal = st == SSLM_SCHEMA_SPAN_UNREACHABLE ? "unreachable" : "status " + std::to_string(st);
			break;
		}
		++r.admitted;
		r.accepting.push_back(Accepting(s, seq));
	}
	r.accepting_end = Accepting(s, seq);
	sslm_seq_release(seq);
	return r;
}

GenResult CpuGen(CpuSession* s, const std::vector<int32_t>& prompt, int32_t max_tokens, GenMode mode) {
	GenResult r;
	sslm_seq seq = nullptr;
	sslm_status st = NewBoundSeq(s, &seq);
	if (st != SSLM_OK) {
		r.end = Err("bind", st);
		return r;
	}
	int32_t consumed = 0;
	st = sslm_prefill(s->model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                  static_cast<int32_t>(prompt.size()), SSLM_SPAN_PROMPT, s->ws, &consumed);
	if (st != SSLM_OK) {
		r.end = Err("sslm_prefill", st);
		sslm_seq_release(seq);
		return r;
	}
	sslm_jump_options exact{};
	exact.struct_size = sizeof(exact);
	exact.mode = SSLM_JUMP_EXACT;
	const auto t0 = std::chrono::steady_clock::now();
	std::vector<int32_t> forced(kJumpBudget);
	auto forced_count = [&]() {
		int32_t count = 0, accepting = 0;
		return sslm_seq_forced_tokens(s->model, seq, nullptr, 0, &count, &accepting) == SSLM_OK ? count : -1;
	};
	auto jump = [&](const sslm_jump_options* opts) -> bool {
		int32_t got = 0;
		const sslm_status js = sslm_seq_jump_forward(s->model, seq, kJumpBudget, nullptr, opts, s->ws,
		                                             forced.data(), kJumpBudget, &got);
		if (js != SSLM_OK) {
			r.end = Err("sslm_seq_jump_forward", js);
			return false;
		}
		if (got > 0) ++r.jump_calls;
		r.ids.insert(r.ids.end(), forced.begin(), forced.begin() + got);
		return got > 0;
	};
	while (static_cast<int32_t>(r.ids.size()) < max_tokens) {
		if (mode == GenMode::JumpCanonical) {
			// The host loop: jump until no forced token is left, then decode.
			while (forced_count() > 0) {
				if (!jump(nullptr)) break;
			}
		} else if (mode == GenMode::JumpExact && !r.ids.empty() && forced_count() > 0) {
			jump(&exact);
		}
		if (r.end.rfind("error:", 0) == 0) break;
		int32_t t = -1;
		st = DecodeOne(s, seq, &t);
		if (st != SSLM_OK) {
			r.end = Err("sslm_decode_step", st);
			break;
		}
		if (t == -2) {
			r.end = "dead_end";
			break;
		}
		r.ids.push_back(t);
	}
	if (r.end.empty()) r.end = "max_tokens";
	r.ms = MsSince(t0);
	r.accepting = Accepting(s, seq);
	sslm_stats_out stats{};
	if (sslm_stats(s->model, seq, &stats) == SSLM_OK) r.forced = stats.forced_token_count;
	sslm_seq_release(seq);
	return r;
}

std::vector<RestoreResult> CpuRestores(CpuSession* s, const std::vector<int32_t>& prompt,
                                       const std::vector<int32_t>& ref, const std::vector<int32_t>& after,
                                       const RestoreOptions& opt, Trace* unbroken, std::string* error) {
	std::vector<RestoreResult> out;
	std::vector<std::vector<uint8_t>> blobs;
	std::vector<int32_t> saved_accepting;
	sslm_seq seq = nullptr;
	sslm_status st = NewBoundSeq(s, &seq);
	if (st != SSLM_OK) {
		*error = Err("bind", st);
		return out;
	}
	st = PrefillPrompt(s, seq, prompt);
	if (st != SSLM_OK) {
		*error = Err("sslm_prefill", st);
		sslm_seq_release(seq);
		return out;
	}
	// The unbroken run, through the whole reference and its dead end, saving at each point.
	size_t next = 0;
	for (size_t i = 0; i <= ref.size(); ++i) {
		int32_t t = -1;
		st = DecodeOne(s, seq, &t);
		if (st != SSLM_OK || (i < ref.size() && t != ref[i]) || (i == ref.size() && t != -2)) {
			*error = "unbroken run left the reference at " + std::to_string(i) + ": " + std::to_string(t) +
			         " != " + std::to_string(i < ref.size() ? ref[i] : -2) + " (status " + std::to_string(st) + ")";
			sslm_seq_release(seq);
			return out;
		}
		if (i == ref.size()) break;
		unbroken->ids.push_back(t);
		unbroken->accepting.push_back(Accepting(s, seq));
		if (next < after.size() && static_cast<int32_t>(i) == after[next]) {
			std::vector<uint8_t> blob;
			if (!SaveBlob(seq, &blob, error)) {
				sslm_seq_release(seq);
				return out;
			}
			blobs.push_back(std::move(blob));
			saved_accepting.push_back(unbroken->accepting.back());
			++next;
		}
	}
	unbroken->end = "dead_end";
	sslm_seq_release(seq);
	for (size_t k = 0; k < blobs.size(); ++k) {
		RestoreResult rr;
		rr.after = after[k];
		rr.blob_bytes = blobs[k].size();
		rr.saved_accepting = saved_accepting[k];
		if (opt.state_zero && !ZeroBlobWalkState(&blobs[k], false)) rr.error = "state_zero: unexpected blob magic";
		sslm_seq restored = nullptr;
		st = rr.error.empty() ? sslm_seq_restore(s->model, &s->pool, blobs[k].data(), blobs[k].size(), &restored)
		                      : SSLM_INVALID_ARGUMENT;
		if (st != SSLM_OK) {
			if (rr.error.empty()) rr.error = Err("sslm_seq_restore", st);
		} else {
			rr.restored_accepting = Accepting(s, restored);
			DecodeTrace(s, restored, opt.max_tokens - rr.after - 1, &rr.continuation);
			sslm_seq_release(restored);
		}
		std::vector<uint8_t>().swap(blobs[k]);
		out.push_back(std::move(rr));
	}
	return out;
}

std::vector<RestoreResult> CpuTfRestores(CpuSession* s, const std::vector<int32_t>& prompt,
                                         const std::vector<int32_t>& doc, const std::vector<int32_t>& after,
                                         const RestoreOptions& opt, Trace* walk, std::string* error) {
	std::vector<RestoreResult> out;
	sslm_seq seq = nullptr;
	sslm_status st = NewBoundSeq(s, &seq);
	if (st == SSLM_OK) st = PrefillPrompt(s, seq, prompt);
	if (st != SSLM_OK) {
		*error = Err("walk setup", st);
		if (seq) sslm_seq_release(seq);
		return out;
	}
	ForceTrace(s, seq, doc.data(), doc.size(), walk);
	sslm_seq_release(seq);
	for (const int32_t a : after) {
		RestoreResult rr;
		rr.after = a;
		out.push_back(rr);
		RestoreResult& r = out.back();
		if (a < 0 || static_cast<size_t>(a) + 1 >= doc.size()) {
			r.error = "save index outside the document";
			continue;
		}
		seq = nullptr;
		st = NewBoundSeq(s, &seq);
		if (st == SSLM_OK) st = PrefillPrompt(s, seq, prompt);
		int32_t consumed = 0;
		if (st == SSLM_OK) {
			st = sslm_prefill(s->model, seq, doc.data(), a + 1, a + 1, SSLM_SPAN_SCHEMA_CONTENT, s->ws, &consumed);
			if (st == SSLM_OK && consumed != a + 1) st = SSLM_SCHEMA_SPAN_UNREACHABLE;
		}
		if (st != SSLM_OK) {
			r.error = Err("teacher-forced prefix", st);
			if (seq) sslm_seq_release(seq);
			continue;
		}
		r.saved_accepting = Accepting(s, seq);
		std::vector<uint8_t> blob;
		if (!SaveBlob(seq, &blob, &r.error)) {
			sslm_seq_release(seq);
			continue;
		}
		r.blob_bytes = blob.size();
		DecodeTrace(s, seq, opt.max_tokens, &r.unbroken);
		sslm_seq_release(seq);
		if (opt.state_zero && !ZeroBlobWalkState(&blob, false)) {
			r.error = "state_zero: unexpected blob magic";
			continue;
		}
		sslm_seq restored = nullptr;
		st = sslm_seq_restore(s->model, &s->pool, blob.data(), blob.size(), &restored);
		if (st != SSLM_OK) {
			r.error = Err("sslm_seq_restore", st);
			continue;
		}
		r.restored_accepting = Accepting(s, restored);
		DecodeTrace(s, restored, opt.max_tokens, &r.continuation);
		sslm_seq_release(restored);
		restored = nullptr;
		st = sslm_seq_restore(s->model, &s->pool, blob.data(), blob.size(), &restored);
		if (st != SSLM_OK) {
			r.error = Err("sslm_seq_restore (second)", st);
			continue;
		}
		ForceTrace(s, restored, doc.data() + a + 1, doc.size() - static_cast<size_t>(a) - 1, &r.forced);
		sslm_seq_release(restored);
	}
	return out;
}
