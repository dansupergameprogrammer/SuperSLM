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
	uint32_t budget = DispatchBudget();  // dispatches per decode-step call
};

GpuSession* GpuOpen(const uint8_t* bytes, size_t n, const std::string& schema, MapResult* map,
                    const GpuOptions& opt) {
	auto s = std::make_unique<GpuSession>();
	std::string load_err;
	if (superslm::SslmModel::Load(bytes, n, s->view, &load_err) != superslm::SslmModelStatus::Ok) {
		map->error = "SslmModel::Load: " + load_err;
		return nullptr;
	}
	GpuContextConfig gcfg{};
	GpuResidencyConfig rcfg{};
	if (opt.head_on_device) rcfg.flags |= SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE;
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

// Decodes until the dead end (-2) or until `out` holds max_tokens, recording the accepting flag
// and the walk state after each token; `last` is the token to embed (ignored while the sequence's
// ready-for-logits flag is set, as after a prefill or a restore of a post-prefill save).
void DecodeTrace(GpuSession* s, SslmGpuSequenceHandle* seq, int32_t last, int32_t max_tokens, Trace* out) {
	while (static_cast<int32_t>(out->ids.size()) < max_tokens) {
		int32_t t = -1;
		const SslmGpuStatus st = SslmGpuSeqDecodeStepForG5Bridge(s->ctx, seq, last, s->budget, &t);
		if (st != SSLM_OK) {
			out->end = Err("SslmGpuSeqDecodeStepForG5Bridge", st);
			return;
		}
		if (t == -2) {
			out->end = "dead_end";
			return;
		}
		out->ids.push_back(t);
		out->accepting.push_back(Accepting(s, seq));
		out->states.push_back(SslmGpuSeqWalkStateForG5Bridge(seq));
		last = t;
	}
	out->end = "max_tokens";
}

// Teacher-forces `ids` one token per schema-content prefill call, recording each admitted token.
void ForceTrace(GpuSession* s, SslmGpuSequenceHandle* seq, const int32_t* ids, size_t n, Trace* out) {
	for (size_t i = 0; i < n; ++i) {
		int32_t consumed = 0;
		const SslmGpuStatus st =
		    SslmGpuSeqPrefillSchemaContentForG5Bridge(s->ctx, seq, &ids[i], 1, DispatchBudget(), &consumed);
		if (st != SSLM_OK || consumed != 1) {
			out->end = st == SSLM_SEQUENCE_REJECTED ? "refused" : Err("SslmGpuSeqPrefillSchemaContentForG5Bridge", st);
			return;
		}
		out->ids.push_back(ids[i]);
		out->accepting.push_back(Accepting(s, seq));
		out->states.push_back(SslmGpuSeqWalkStateForG5Bridge(seq));
	}
	out->end = "forced";
}

SslmGpuStatus PrefillPrompt(GpuSession* s, SslmGpuSequenceHandle* seq, const std::vector<int32_t>& prompt) {
	return SslmGpuSeqPrefillPromptForG5Bridge(s->ctx, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                                          DispatchBudget());
}

bool SaveBlob(GpuSession* s, SslmGpuSequenceHandle* seq, std::vector<uint8_t>* blob, std::string* error) {
	size_t n = 0;
	sslm_gpu_seq_save(s->ctx, seq, nullptr, &n);
	blob->assign(n, 0);
	const SslmGpuStatus st = sslm_gpu_seq_save(s->ctx, seq, blob->data(), &n);
	if (st != SSLM_OK || n == 0) {
		*error = Err("sslm_gpu_seq_save", st);
		return false;
	}
	blob->resize(n);
	return true;
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
		r.accepting.push_back(Accepting(s, seq));
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
	// No per-token queries here: `gputime` times this loop.
	const auto t0 = std::chrono::steady_clock::now();
	int32_t last = prompt.back();
	r.end = "max_tokens";
	while (static_cast<int32_t>(r.ids.size()) < max_tokens) {
		int32_t t = -1;
		const auto ts = std::chrono::steady_clock::now();
		st = SslmGpuSeqDecodeStepForG5Bridge(s->ctx, seq, last, s->budget, &t);
		r.step_ms.push_back(MsSince(ts));
		r.step_gpu_busy_ms.push_back(superslm_gpu::LastCallTiming().gpu_busy_ms);
		if (st != SSLM_OK) {
			r.end = Err("SslmGpuSeqDecodeStepForG5Bridge", st);
			break;
		}
		if (t == -2) {
			r.end = "dead_end";
			break;
		}
		r.ids.push_back(t);
		last = t;
	}
	r.ms = MsSince(t0);
	r.accepting = Accepting(s, seq);
	sslm_gpu_seq_release(s->ctx, seq);
	return r;
}

std::vector<RestoreResult> GpuRestores(GpuSession* s, const std::vector<int32_t>& prompt,
                                       const std::vector<int32_t>& ref, const std::vector<int32_t>& after,
                                       const RestoreOptions& opt, Trace* unbroken, std::string* error) {
	std::vector<RestoreResult> out;
	std::vector<std::vector<uint8_t>> blobs;
	SslmGpuSequenceHandle* seq = nullptr;
	SslmGpuStatus st = NewBoundSeq(s, &seq);
	if (st != SSLM_OK) {
		*error = Err("bind", st);
		return out;
	}
	st = PrefillPrompt(s, seq, prompt);
	if (st != SSLM_OK) {
		*error = Err("SslmGpuSeqPrefillPromptForG5Bridge", st);
		sslm_gpu_seq_release(s->ctx, seq);
		return out;
	}
	// The unbroken run, through the whole reference and its dead end, saving at each point.
	int32_t last = prompt.back();
	size_t next = 0;
	for (size_t i = 0; i <= ref.size(); ++i) {
		int32_t t = -1;
		st = SslmGpuSeqDecodeStepForG5Bridge(s->ctx, seq, last, s->budget, &t);
		if (st != SSLM_OK || (i < ref.size() && t != ref[i]) || (i == ref.size() && t != -2)) {
			*error = "unbroken run left the reference at " + std::to_string(i) + ": " + std::to_string(t) +
			         " != " + std::to_string(i < ref.size() ? ref[i] : -2) + " (status " +
			         std::to_string(static_cast<int>(st)) + ")";
			sslm_gpu_seq_release(s->ctx, seq);
			return out;
		}
		if (i == ref.size()) break;
		last = t;
		unbroken->ids.push_back(t);
		unbroken->accepting.push_back(Accepting(s, seq));
		unbroken->states.push_back(SslmGpuSeqWalkStateForG5Bridge(seq));
		if (next < after.size() && static_cast<int32_t>(i) == after[next]) {
			std::vector<uint8_t> blob;
			if (!SaveBlob(s, seq, &blob, error)) {
				sslm_gpu_seq_release(s->ctx, seq);
				return out;
			}
			blobs.push_back(std::move(blob));
			++next;
		}
	}
	unbroken->end = "dead_end";
	sslm_gpu_seq_release(s->ctx, seq);
	for (size_t k = 0; k < blobs.size(); ++k) {
		RestoreResult rr;
		rr.after = after[k];
		rr.blob_bytes = blobs[k].size();
		rr.saved_accepting = unbroken->accepting[static_cast<size_t>(rr.after)];
		rr.saved_state = unbroken->states[static_cast<size_t>(rr.after)];
		if (opt.state_zero && !ZeroBlobWalkState(&blobs[k], true)) rr.error = "state_zero: unexpected blob magic";
		SslmGpuSequenceHandle* restored = nullptr;
		if (rr.error.empty()) {
			st = sslm_gpu_seq_restore(s->ctx, s->model, blobs[k].data(), blobs[k].size(), &restored);
			if (st != SSLM_OK) rr.error = Err("sslm_gpu_seq_restore", st);
		}
		if (rr.error.empty()) {
			rr.restored_accepting = Accepting(s, restored);
			rr.restored_state = SslmGpuSeqWalkStateForG5Bridge(restored);
			DecodeTrace(s, restored, ref[static_cast<size_t>(rr.after)], opt.max_tokens - rr.after - 1,
			            &rr.continuation);
			sslm_gpu_seq_release(s->ctx, restored);
		}
		std::vector<uint8_t>().swap(blobs[k]);
		out.push_back(std::move(rr));
	}
	return out;
}

std::vector<RestoreResult> GpuTfRestores(GpuSession* s, const std::vector<int32_t>& prompt,
                                         const std::vector<int32_t>& doc, const std::vector<int32_t>& after,
                                         const RestoreOptions& opt, Trace* walk, std::string* error) {
	std::vector<RestoreResult> out;
	SslmGpuSequenceHandle* seq = nullptr;
	SslmGpuStatus st = NewBoundSeq(s, &seq);
	if (st == SSLM_OK) st = PrefillPrompt(s, seq, prompt);
	if (st != SSLM_OK) {
		*error = Err("walk setup", st);
		if (seq) sslm_gpu_seq_release(s->ctx, seq);
		return out;
	}
	ForceTrace(s, seq, doc.data(), doc.size(), walk);
	sslm_gpu_seq_release(s->ctx, seq);
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
			st = SslmGpuSeqPrefillSchemaContentForG5Bridge(s->ctx, seq, doc.data(), a + 1, DispatchBudget(), &consumed);
			if (st == SSLM_OK && consumed != a + 1) st = SSLM_SEQUENCE_REJECTED;
		}
		if (st != SSLM_OK) {
			r.error = Err("teacher-forced prefix", st);
			if (seq) sslm_gpu_seq_release(s->ctx, seq);
			continue;
		}
		r.saved_accepting = Accepting(s, seq);
		r.saved_state = SslmGpuSeqWalkStateForG5Bridge(seq);
		std::vector<uint8_t> blob;
		if (!SaveBlob(s, seq, &blob, &r.error)) {
			sslm_gpu_seq_release(s->ctx, seq);
			continue;
		}
		r.blob_bytes = blob.size();
		const int32_t last = doc[static_cast<size_t>(a)];
		DecodeTrace(s, seq, last, opt.max_tokens, &r.unbroken);
		sslm_gpu_seq_release(s->ctx, seq);
		if (opt.state_zero && !ZeroBlobWalkState(&blob, true)) {
			r.error = "state_zero: unexpected blob magic";
			continue;
		}
		SslmGpuSequenceHandle* restored = nullptr;
		st = sslm_gpu_seq_restore(s->ctx, s->model, blob.data(), blob.size(), &restored);
		if (st != SSLM_OK) {
			r.error = Err("sslm_gpu_seq_restore", st);
			continue;
		}
		r.restored_accepting = Accepting(s, restored);
		r.restored_state = SslmGpuSeqWalkStateForG5Bridge(restored);
		DecodeTrace(s, restored, last, opt.max_tokens, &r.continuation);
		sslm_gpu_seq_release(s->ctx, restored);
		restored = nullptr;
		st = sslm_gpu_seq_restore(s->ctx, s->model, blob.data(), blob.size(), &restored);
		if (st != SSLM_OK) {
			r.error = Err("sslm_gpu_seq_restore (second)", st);
			continue;
		}
		ForceTrace(s, restored, doc.data() + a + 1, doc.size() - static_cast<size_t>(a) - 1, &r.forced);
		sslm_gpu_seq_release(s->ctx, restored);
	}
	return out;
}
