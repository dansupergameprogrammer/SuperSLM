// array_s2_gpu.cpp -- the GPU side of the S2 array harness, on the shipped GPU surface
// (superslm/gpu_1p0.h and its G5 schema bridge). See array_s2_shared.h for the interface.
#include <chrono>
#include <cstdlib>
#include <memory>

#include <dxgi1_4.h>
#include <wrl/client.h>

#include "superslm/gpu_1p0.h"
using enum SslmGpuStatus;
#include "superslm/gpu_1p0_g5_bridge.h"
#include "superslm/gpu_port.h"  // superslm_gpu::kDispatchesPerLayer
#include "superslm/model.h"
#include "array_s2_shared.h"

namespace {

// The decode drive submits and waits once per `dispatch_budget` dispatches, so the budget sets
// the slices per token: ARRAY_S2_GPU_LAYERS_PER_SLICE layers per submission (default 1, so a
// 24-layer model decodes in 24 slices; 24 gives one slice per token).
uint32_t DispatchBudget() {
	static const uint32_t budget = [] {
		const char* env = std::getenv("ARRAY_S2_GPU_LAYERS_PER_SLICE");
		const int layers = env ? std::atoi(env) : 1;
		return static_cast<uint32_t>(layers > 0 ? layers : 1) * superslm_gpu::kDispatchesPerLayer;
	}();
	return budget;
}

double MsSince(std::chrono::steady_clock::time_point t0) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string Err(const char* call, SslmGpuStatus st) {
	return std::string("error:") + call + " " + std::to_string(static_cast<int>(st));
}

// Local-segment (VRAM) usage of this process on adapter 0, or -1.
int64_t LocalVideoMemoryUsage() {
	Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
	if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return -1;
	Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
	if (FAILED(factory->EnumAdapters1(0, &adapter))) return -1;
	Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter3;
	if (FAILED(adapter.As(&adapter3))) return -1;
	DXGI_QUERY_VIDEO_MEMORY_INFO info{};
	if (FAILED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) return -1;
	return static_cast<int64_t>(info.CurrentUsage);
}

}  // namespace

struct GpuSession {
	superslm::SslmModelView view;
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	int32_t schema_index = -1;
	int64_t context_cap = 0;
};

GpuSession* GpuOpen(const uint8_t* bytes, size_t n, const std::string& schema, MapResult* map) {
	auto s = std::make_unique<GpuSession>();
	std::string load_err;
	if (superslm::SslmModel::Load(bytes, n, s->view, &load_err) != superslm::SslmModelStatus::Ok) {
		map->error = "SslmModel::Load: " + load_err;
		return nullptr;
	}
	GpuContextConfig gcfg{};
	GpuResidencyConfig rcfg{};
	SslmGpuStatus st = sslm_gpu_context_create(gcfg, &s->ctx);
	if (st != SSLM_OK) {
		map->error = Err("sslm_gpu_context_create", st);
		return nullptr;
	}
	const int64_t before = LocalVideoMemoryUsage();
	const auto t0 = std::chrono::steady_clock::now();
	st = sslm_gpu_model_map(s->ctx, &s->view, rcfg, &s->model);
	map->map_ms = MsSince(t0);
	const int64_t after = LocalVideoMemoryUsage();
	if (before >= 0 && after >= 0) map->vram_delta_bytes = after - before;
	if (st != SSLM_OK) {
		map->error = Err("sslm_gpu_model_map", st);
		sslm_gpu_context_destroy(s->ctx);
		return nullptr;
	}
	if (!schema.empty()) {
		s->schema_index = SslmGpuSchemaLookupForG5Bridge(s->model, schema.c_str());
		if (s->schema_index < 0) {
			map->error = "SslmGpuSchemaLookupForG5Bridge failed";
			GpuClose(s.release());
			return nullptr;
		}
	}
	s->context_cap = static_cast<int64_t>(s->view.config.context_cap);
	map->ok = true;
	return s.release();
}

void GpuClose(GpuSession* s) {
	if (!s) return;
	if (s->model) sslm_gpu_model_unmap(s->ctx, s->model);
	if (s->ctx) sslm_gpu_context_destroy(s->ctx);
	delete s;
}

namespace {

SslmGpuStatus NewBoundSeq(GpuSession* s, SslmGpuSequenceHandle** seq) {
	SslmGpuStatus st = sslm_gpu_seq_create(s->ctx, s->model, s->context_cap, seq);
	if (st != SSLM_OK) return st;
	st = SslmGpuSeqSetSchemaForG5Bridge(s->ctx, *seq, s->schema_index);
	if (st != SSLM_OK) {
		sslm_gpu_seq_release(s->ctx, *seq);
		*seq = nullptr;
	}
	return st;
}

int32_t Accepting(GpuSession* s, SslmGpuSequenceHandle* seq) {
	int32_t a = -1;
	return SslmGpuSeqSchemaAcceptingForG5Bridge(s->ctx, seq, &a) == SSLM_OK ? a : -1;
}

// Decodes until the dead end (-2) or until ids holds max_tokens; `last` is the token to embed.
void DecodeToEnd(GpuSession* s, SslmGpuSequenceHandle* seq, int32_t last, int32_t max_tokens,
                 std::vector<int32_t>* ids, std::string* end) {
	while (static_cast<int32_t>(ids->size()) < max_tokens) {
		int32_t t = -1;
		const SslmGpuStatus st = SslmGpuSeqDecodeStepForG5Bridge(s->ctx, seq, last, DispatchBudget(), &t);
		if (st != SSLM_OK) {
			*end = Err("SslmGpuSeqDecodeStepForG5Bridge", st);
			return;
		}
		if (t == -2) {
			*end = "dead_end";
			return;
		}
		ids->push_back(t);
		last = t;
	}
	*end = "max_tokens";
}

}  // namespace

WalkResult GpuWalk(GpuSession* s, const std::vector<int32_t>& ids) {
	WalkResult r;
	SslmGpuSequenceHandle* seq = nullptr;
	SslmGpuStatus st = NewBoundSeq(s, &seq);
	if (st != SSLM_OK) {
		r.refusal = Err("bind", st);
		return r;
	}
	for (size_t i = 0; i < ids.size(); ++i) {
		int32_t consumed = 0;
		st = SslmGpuSeqPrefillSchemaContentForG5Bridge(s->ctx, seq, &ids[i], 1, DispatchBudget(), &consumed);
		if (st != SSLM_OK || consumed != 1) {
			r.refused_at = static_cast<int32_t>(i);
			r.refusal = st == SSLM_SEQUENCE_REJECTED ? "unreachable" : "status " + std::to_string(static_cast<int>(st));
			break;
		}
		++r.admitted;
		r.states.push_back(SslmGpuSeqWalkStateForG5Bridge(seq));
	}
	r.accepting_end = Accepting(s, seq);
	sslm_gpu_seq_release(s->ctx, seq);
	return r;
}

GenResult GpuGen(GpuSession* s, const std::vector<int32_t>& prompt, int32_t max_tokens) {
	GenResult r;
	SslmGpuSequenceHandle* seq = nullptr;
	SslmGpuStatus st = NewBoundSeq(s, &seq);
	if (st != SSLM_OK) {
		r.end = Err("bind", st);
		return r;
	}
	st = SslmGpuSeqPrefillPromptForG5Bridge(s->ctx, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                                        DispatchBudget());
	if (st != SSLM_OK) {
		r.end = Err("SslmGpuSeqPrefillPromptForG5Bridge", st);
		sslm_gpu_seq_release(s->ctx, seq);
		return r;
	}
	const auto t0 = std::chrono::steady_clock::now();
	DecodeToEnd(s, seq, prompt.back(), max_tokens, &r.ids, &r.end);
	r.ms = MsSince(t0);
	r.accepting = Accepting(s, seq);
	sslm_gpu_seq_release(s->ctx, seq);
	return r;
}

std::vector<RestoreResult> GpuRestores(GpuSession* s, const std::vector<int32_t>& prompt,
                                       const std::vector<int32_t>& ref, const std::vector<int32_t>& after,
                                       int32_t max_tokens, std::string* error) {
	std::vector<RestoreResult> out;
	std::vector<std::vector<uint8_t>> blobs;
	SslmGpuSequenceHandle* seq = nullptr;
	SslmGpuStatus st = NewBoundSeq(s, &seq);
	if (st != SSLM_OK) {
		*error = Err("bind", st);
		return out;
	}
	st = SslmGpuSeqPrefillPromptForG5Bridge(s->ctx, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                                        DispatchBudget());
	if (st != SSLM_OK) {
		*error = Err("SslmGpuSeqPrefillPromptForG5Bridge", st);
		sslm_gpu_seq_release(s->ctx, seq);
		return out;
	}
	int32_t last = prompt.back();
	size_t next = 0;
	for (size_t i = 0; i < ref.size() && next < after.size(); ++i) {
		int32_t t = -1;
		st = SslmGpuSeqDecodeStepForG5Bridge(s->ctx, seq, last, DispatchBudget(), &t);
		if (st != SSLM_OK || t != ref[i]) {
			*error = "unbroken run left the reference at " + std::to_string(i) + ": " + std::to_string(t) +
			         " != " + std::to_string(ref[i]);
			sslm_gpu_seq_release(s->ctx, seq);
			return out;
		}
		last = t;
		if (static_cast<int32_t>(i) == after[next]) {
			size_t n = 0;
			sslm_gpu_seq_save(s->ctx, seq, nullptr, &n);
			std::vector<uint8_t> blob(n);
			st = sslm_gpu_seq_save(s->ctx, seq, blob.data(), &n);
			if (st != SSLM_OK || n == 0) {
				*error = Err("sslm_gpu_seq_save", st);
				sslm_gpu_seq_release(s->ctx, seq);
				return out;
			}
			blob.resize(n);
			blobs.push_back(std::move(blob));
			++next;
		}
	}
	sslm_gpu_seq_release(s->ctx, seq);
	for (size_t k = 0; k < blobs.size(); ++k) {
		RestoreResult rr;
		rr.after = after[k];
		rr.blob_bytes = blobs[k].size();
		SslmGpuSequenceHandle* restored = nullptr;
		st = sslm_gpu_seq_restore(s->ctx, s->model, blobs[k].data(), blobs[k].size(), &restored);
		if (st != SSLM_OK) {
			rr.end = Err("sslm_gpu_seq_restore", st);
		} else {
			DecodeToEnd(s, restored, ref[static_cast<size_t>(rr.after)], max_tokens - rr.after - 1,
			            &rr.continuation, &rr.end);
			sslm_gpu_seq_release(s->ctx, restored);
		}
		out.push_back(std::move(rr));
	}
	return out;
}
