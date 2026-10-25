// Multi-row GPU plan (rev 1.3.2), MR1 gate "Wine one-layer smoke equal on pkv_def and pkv_qk": the one-row
// twin's side, captured from public verbs only (plan §7 grading rule: the twin is the same stream through
// sslm_gpu_seq_embed_token and sslm_decode_step_gpu, in the same process).
//
// For a fixed token stream it feeds each token one row at a time, one layer per sslm_decode_step_gpu call
// (dispatch_budget = DispatchesPerLayer: 24, or 25 on pkv_qk), drains with sslm_gpu_ready, and saves the
// dedicated handle (SLM5) after every layer of every row. A one-row save mid-token is admitted (plan E10).
// Blob (row r, layer l) is the state one-row leaves after row r's layer l: the reference any widened leg of the
// smoke compares against (a one-layer widened pass against l = 0 per row; a full chunk against the last row's
// last layer). The capture runs twice, on fresh contexts, and the two runs must be byte-identical: the reference
// is deterministic on this device before anything is graded against it.
//
// Usage: mr1_twin_capture.exe --model=PATH [--rows=8] [--start=0] [--out=DIR] [--shaders=DIR]
//   --start  one-row tokens fed (to full depth) before the captured rows, so p0 = start (cell 6.1's offset s)
//   --out    writes r<row>_l<layer>.slm5 for each captured blob and index.txt (row layer bytes fnv1a64)
// Exit 0 when every call returned SSLM_OK and the two runs agree.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"

namespace {

using S = ::SslmGpuStatus;

struct Args {
	std::string model, out, shaders;
	int rows = 8, start = 0;
};

bool ReadFileBytes(const std::string& path, std::vector<uint8_t>* out) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	const size_t n = sz > 0 ? std::fread(out->data(), 1, static_cast<size_t>(sz), f) : 0;
	std::fclose(f);
	return n == out->size();
}

uint64_t Fnv1a64(const std::vector<uint8_t>& b) {
	uint64_t h = 1469598103934665603ull;
	for (uint8_t c : b) h = (h ^ c) * 1099511628211ull;
	return h;
}

// A fixed stream inside the fixture's vocabulary (deterministic; no RNG state to carry).
int32_t TokenAt(int i, int32_t vocab) { return static_cast<int32_t>((1103515245u * static_cast<uint32_t>(i + 7) + 12345u) % static_cast<uint32_t>(vocab)); }

struct Blob {
	int row, layer;
	std::vector<uint8_t> bytes;
};

bool Fail(const char* what, S st) {
	std::printf("FAIL %s -> status %d\n", what, static_cast<int>(st));
	return false;
}

bool Drain(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
	int32_t ready = 0;
	S out = S::SSLM_OK;
	const S st = sslm_gpu_ready(ctx, seq, 1, &ready, &out);
	if (st != S::SSLM_OK) return Fail("sslm_gpu_ready", st);
	if (!ready) return Fail("sslm_gpu_ready (block=1) returned not ready", st);
	if (out != S::SSLM_OK) return Fail("slice status", out);
	return true;
}

bool Save(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, std::vector<uint8_t>* b) {
	size_t need = 0;
	sslm_gpu_seq_save(ctx, seq, nullptr, &need);
	if (need == 0) return Fail("sslm_gpu_seq_save (size query) returned 0 bytes", S::SSLM_OK);
	b->resize(need);
	size_t sz = need;
	const S st = sslm_gpu_seq_save(ctx, seq, b->data(), &sz);
	if (st != S::SSLM_OK) return Fail("sslm_gpu_seq_save", st);
	b->resize(sz);
	return true;
}

bool Capture(const Args& a, const superslm::SslmModelView& view, std::vector<Blob>* blobs) {
	const auto& c = view.config;
	bool qk = false;
	for (uint32_t l = 0; l < c.num_hidden_layers; ++l)
		if (view.weights.Tensor("layer" + std::to_string(l) + ".q_norm.gain") != nullptr) qk = true;
	const uint32_t dpl = superslm_gpu::DispatchesPerLayer(qk);
	const uint32_t L = c.num_hidden_layers;
	const int32_t vocab = static_cast<int32_t>(c.vocab_size);

	GpuContextConfig cfg{};
	cfg.shader_dir = a.shaders.empty() ? nullptr : a.shaders.c_str();
	SslmGpuContext* ctx = nullptr;
	S st = sslm_gpu_context_create(cfg, &ctx);
	if (st != S::SSLM_OK || !ctx) return Fail("sslm_gpu_context_create", st);
	SslmGpuModelHandle* model = nullptr;
	st = sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model);
	if (st != S::SSLM_OK || !model) return Fail("sslm_gpu_model_map", st);
	SslmGpuSequenceHandle* seq = nullptr;
	st = sslm_gpu_seq_create(ctx, model, 64, &seq);
	if (st != S::SSLM_OK || !seq) return Fail("sslm_gpu_seq_create(cap 64)", st);

	bool ok = true;
	for (int i = 0; ok && i < a.start + a.rows; ++i) {
		st = sslm_gpu_seq_embed_token(ctx, seq, TokenAt(i, vocab));
		if (st != S::SSLM_OK) { ok = Fail("sslm_gpu_seq_embed_token", st); break; }
		const bool captured = i >= a.start;
		const uint32_t budget = captured ? dpl : dpl * L;
		for (uint32_t done = 0; ok && done < L; done += budget / dpl) {
			st = sslm_decode_step_gpu(ctx, seq, nullptr, budget);
			if (st != S::SSLM_OK) { ok = Fail("sslm_decode_step_gpu", st); break; }
			if (!(ok = Drain(ctx, seq))) break;
			if (captured) {
				Blob b{i - a.start, static_cast<int>(done), {}};
				if (!(ok = Save(ctx, seq, &b.bytes))) break;
				blobs->push_back(std::move(b));
			}
		}
	}
	sslm_gpu_seq_release(ctx, seq);
	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);
	if (ok) std::printf("captured %zu blobs: %d rows x %u layers, DispatchesPerLayer %u, p0 %d\n", blobs->size(), a.rows, L,
	                    dpl, a.start);
	return ok;
}

}  // namespace

int main(int argc, char** argv) {
	Args a;
	for (int i = 1; i < argc; ++i) {
		const std::string s = argv[i];
		auto take = [&](const char* f, std::string* o) {
			const size_t n = std::strlen(f);
			if (s.compare(0, n, f) != 0) return false;
			*o = s.substr(n);
			return true;
		};
		std::string v;
		if (take("--model=", &a.model) || take("--out=", &a.out) || take("--shaders=", &a.shaders)) continue;
		if (take("--rows=", &v)) { a.rows = std::atoi(v.c_str()); continue; }
		if (take("--start=", &v)) { a.start = std::atoi(v.c_str()); continue; }
		std::printf("usage: mr1_twin_capture --model=PATH [--rows=8] [--start=0] [--out=DIR] [--shaders=DIR]\n");
		return 2;
	}
	if (a.rows < 1 || a.rows > 8 || a.start < 0 || a.start > 7) {
		std::printf("FAIL --rows must be 1..8 and --start 0..7\n");
		return 2;
	}
	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	std::string err;
	if (a.model.empty() || !ReadFileBytes(a.model, &bytes) ||
	    superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) != superslm::SslmModelStatus::Ok) {
		std::printf("FAIL cannot load model '%s' %s\n", a.model.c_str(), err.c_str());
		return 1;
	}
	std::vector<Blob> run1, run2;
	if (!Capture(a, view, &run1) || !Capture(a, view, &run2)) return 1;
	int diff = 0;
	for (size_t i = 0; i < run1.size(); ++i)
		if (i >= run2.size() || run1[i].bytes != run2[i].bytes) ++diff;
	if (diff || run1.size() != run2.size()) {
		std::printf("FAIL the twin is not deterministic: %d of %zu blobs differ between two fresh runs\n", diff, run1.size());
		return 1;
	}
	// The capture changes state: consecutive layer blobs of one row differ (else the save is not observing the slice).
	int stuck = 0;
	for (size_t i = 1; i < run1.size(); ++i)
		if (run1[i].bytes == run1[i - 1].bytes) ++stuck;
	if (stuck) {
		std::printf("FAIL %d consecutive captures are identical: the save does not observe the layer slice\n", stuck);
		return 1;
	}
	if (!a.out.empty()) {
		CreateDirectoryA(a.out.c_str(), nullptr);
		std::FILE* idx = std::fopen((a.out + "/index.txt").c_str(), "w");
		for (const Blob& b : run1) {
			const std::string p = a.out + "/r" + std::to_string(b.row) + "_l" + std::to_string(b.layer) + ".slm5";
			if (std::FILE* f = std::fopen(p.c_str(), "wb")) {
				std::fwrite(b.bytes.data(), 1, b.bytes.size(), f);
				std::fclose(f);
			}
			if (idx) std::fprintf(idx, "%d %d %zu %016llx\n", b.row, b.layer, b.bytes.size(), static_cast<unsigned long long>(Fnv1a64(b.bytes)));
		}
		if (idx) std::fclose(idx);
	}
	std::printf("PASS twin capture: %zu blobs, deterministic across two fresh contexts\n", run1.size());
	return 0;
}
