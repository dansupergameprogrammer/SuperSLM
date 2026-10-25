// array_s2_cpu.cpp -- the CPU side of the S2 array harness, on the public C ABI only
// (superslm/sslm_abi.h). See array_s2_shared.h for the interface.
#include <chrono>
#include <memory>

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
	s->layers = 24;  // Qwen2.5-0.5B
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

// Decodes from a sequence ready to decode until the dead end (-2) or max_tokens.
void DecodeToEnd(CpuSession* s, sslm_seq seq, int32_t max_tokens, std::vector<int32_t>* ids, std::string* end) {
	while (static_cast<int32_t>(ids->size()) < max_tokens) {
		int32_t t = -1;
		const sslm_status st = DecodeOne(s, seq, &t);
		if (st != SSLM_OK) {
			*end = Err("sslm_decode_step", st);
			return;
		}
		if (t == -2) {
			*end = "dead_end";
			return;
		}
		ids->push_back(t);
	}
	*end = "max_tokens";
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
                                       int32_t max_tokens, std::string* error) {
	std::vector<RestoreResult> out;
	std::vector<std::vector<uint8_t>> blobs;
	sslm_seq seq = nullptr;
	sslm_status st = NewBoundSeq(s, &seq);
	if (st != SSLM_OK) {
		*error = Err("bind", st);
		return out;
	}
	int32_t consumed = 0;
	st = sslm_prefill(s->model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                  static_cast<int32_t>(prompt.size()), SSLM_SPAN_PROMPT, s->ws, &consumed);
	if (st != SSLM_OK) {
		*error = Err("sslm_prefill", st);
		sslm_seq_release(seq);
		return out;
	}
	size_t next = 0;
	for (size_t i = 0; i < ref.size() && next < after.size(); ++i) {
		int32_t t = -1;
		st = DecodeOne(s, seq, &t);
		if (st != SSLM_OK || t != ref[i]) {
			*error = "unbroken run left the reference at " + std::to_string(i) + ": " + std::to_string(t) +
			         " != " + std::to_string(ref[i]);
			sslm_seq_release(seq);
			return out;
		}
		if (static_cast<int32_t>(i) == after[next]) {
			size_t n = 0;
			sslm_seq_save(seq, nullptr, &n);
			std::vector<uint8_t> blob(n);
			st = sslm_seq_save(seq, blob.data(), &n);
			if (st != SSLM_OK) {
				*error = Err("sslm_seq_save", st);
				sslm_seq_release(seq);
				return out;
			}
			blob.resize(n);
			blobs.push_back(std::move(blob));
			++next;
		}
	}
	sslm_seq_release(seq);
	for (size_t k = 0; k < blobs.size(); ++k) {
		RestoreResult rr;
		rr.after = after[k];
		rr.blob_bytes = blobs[k].size();
		sslm_seq restored = nullptr;
		st = sslm_seq_restore(s->model, &s->pool, blobs[k].data(), blobs[k].size(), &restored);
		if (st != SSLM_OK) {
			rr.end = Err("sslm_seq_restore", st);
		} else {
			DecodeToEnd(s, restored, max_tokens - rr.after - 1, &rr.continuation, &rr.end);
			sslm_seq_release(restored);
		}
		out.push_back(std::move(rr));
	}
	return out;
}
