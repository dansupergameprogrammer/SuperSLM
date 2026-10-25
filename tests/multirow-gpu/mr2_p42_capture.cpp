// Multi-row GPU plan (rev 1.3.2), cell P4.2 [MR2 cloud]: hashes of the one-row outputs on fixed streams, recorded
// from the BASE TAG's binary, must match the branch's (plan §7 P4: an anchor outside the branch).
//
// Public verbs that exist at the base tag only (this file compiles unchanged against v1.15.0 and the branch), so
// the same source builds the reference binary and the branch binary. For each fixture given and each holder kind:
//   dedicated  sslm_gpu_seq_create (cap 256)                         -> SLM5 bytes
//   pooled     sslm_gpu_seq_create_budgeted over a 32-page pool     -> SLM6 bytes
// it feeds kPrompt tokens one row at a time (embed, one full-token decode step, drain), saves the handle (blob 1),
// then runs a 64-token greedy continuation through SslmGpuSeqDecodeStepForG5Bridge (each produced token is the
// next one embedded) and saves again (blob 2). Every status must be SSLM_OK. It prints one line per item:
//   <fixture> <kind> <item> <bytes> <fnv1a64>
// with item in {prompt_blob, continuation_tokens, continuation_blob}. The runner (mr2_wine.sh p42) diffs the
// branch's lines against p42_base_<tag>.txt, written by the tag's binary.
//
// Usage: mr2_p42_capture.exe --model=PATH [--model=PATH ...]     exit 0 when every call returned SSLM_OK.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"

namespace {

using S = ::SslmGpuStatus;
constexpr int kPrompt = 16;
constexpr int kContinuation = 64;

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
uint64_t Fnv(const uint8_t* p, size_t n) {
	uint64_t h = 1469598103934665603ull;
	for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
	return h;
}
int32_t TokenAt(int i, int32_t vocab) {
	return static_cast<int32_t>((2654435761u * static_cast<uint32_t>(i + 3) + 97u) % static_cast<uint32_t>(vocab));
}
bool Fail(const std::string& what, S st) {
	std::printf("FAIL %s -> status %d\n", what.c_str(), static_cast<int>(st));
	return false;
}
bool Save(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, std::vector<uint8_t>* b) {
	size_t need = 0;
	sslm_gpu_seq_save(ctx, seq, nullptr, &need);
	if (need == 0) return Fail("save size query", S::SSLM_OK);
	b->resize(need);
	size_t sz = need;
	const S st = sslm_gpu_seq_save(ctx, seq, b->data(), &sz);
	if (st != S::SSLM_OK) return Fail("save", st);
	b->resize(sz);
	return true;
}

bool RunOne(const std::string& name, const superslm::SslmModelView& view, bool pooled) {
	const auto& c = view.config;
	bool qk = false;
	for (uint32_t l = 0; l < c.num_hidden_layers; ++l)
		if (view.weights.Tensor("layer" + std::to_string(l) + ".q_norm.gain") != nullptr) qk = true;
	const uint32_t full = superslm_gpu::DispatchesPerLayer(qk) * c.num_hidden_layers;
	const int32_t vocab = static_cast<int32_t>(c.vocab_size);
	const char* kind = pooled ? "pooled" : "dedicated";
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	SslmGpuKvPoolHandle* pool = nullptr;
	SslmGpuSequenceHandle* seq = nullptr;
	bool ok = true;
	S st = sslm_gpu_context_create(GpuContextConfig{}, &ctx);
	if (st != S::SSLM_OK) return Fail(name + " context", st);
	if ((st = sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model)) != S::SSLM_OK) ok = Fail(name + " map", st);
	if (ok && pooled) {
		if ((st = sslm_gpu_kv_pool_create(ctx, model, 32, &pool)) != S::SSLM_OK) ok = Fail(name + " pool", st);
		else if ((st = sslm_gpu_seq_create_budgeted(ctx, model, pool, kPrompt + kContinuation + 8, &seq)) != S::SSLM_OK)
			ok = Fail(name + " budgeted create", st);
	} else if (ok && (st = sslm_gpu_seq_create(ctx, model, 256, &seq)) != S::SSLM_OK) {
		ok = Fail(name + " create", st);
	}
	for (int i = 0; ok && i < kPrompt; ++i) {
		if ((st = sslm_gpu_seq_embed_token(ctx, seq, TokenAt(i, vocab))) != S::SSLM_OK) { ok = Fail(name + " embed", st); break; }
		if ((st = sslm_decode_step_gpu(ctx, seq, nullptr, full)) != S::SSLM_OK) { ok = Fail(name + " decode", st); break; }
		int32_t ready = 0;
		S out = S::SSLM_OK;
		st = sslm_gpu_ready(ctx, seq, 1, &ready, &out);
		if (st != S::SSLM_OK || !ready || out != S::SSLM_OK) { ok = Fail(name + " ready", st != S::SSLM_OK ? st : out); break; }
	}
	std::vector<uint8_t> blob;
	if (ok && (ok = Save(ctx, seq, &blob)))
		std::printf("%s %s prompt_blob %zu %016llx\n", name.c_str(), kind, blob.size(),
		            static_cast<unsigned long long>(Fnv(blob.data(), blob.size())));
	std::vector<int32_t> toks;
	int32_t next = TokenAt(kPrompt, vocab);
	for (int i = 0; ok && i < kContinuation; ++i) {
		int32_t out_tok = -1;
		if ((st = SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, next, full, &out_tok)) != S::SSLM_OK) {
			ok = Fail(name + " continuation step " + std::to_string(i), st);
			break;
		}
		toks.push_back(out_tok);
		next = out_tok >= 0 ? out_tok : TokenAt(i, vocab);
	}
	if (ok) {
		std::printf("%s %s continuation_tokens %zu %016llx\n", name.c_str(), kind, toks.size() * 4,
		            static_cast<unsigned long long>(Fnv(reinterpret_cast<const uint8_t*>(toks.data()), toks.size() * 4)));
		if ((ok = Save(ctx, seq, &blob)))
			std::printf("%s %s continuation_blob %zu %016llx\n", name.c_str(), kind, blob.size(),
			            static_cast<unsigned long long>(Fnv(blob.data(), blob.size())));
	}
	if (seq) sslm_gpu_seq_release(ctx, seq);
	if (pool) sslm_gpu_kv_pool_destroy(ctx, pool);
	if (model) sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);
	return ok;
}

}  // namespace

int main(int argc, char** argv) {
	std::vector<std::string> models;
	for (int i = 1; i < argc; ++i) {
		const std::string s = argv[i];
		if (s.rfind("--model=", 0) == 0) models.push_back(s.substr(8));
		else { std::printf("usage: mr2_p42_capture --model=PATH [--model=PATH ...]\n"); return 2; }
	}
	if (models.empty()) { std::printf("usage: mr2_p42_capture --model=PATH [--model=PATH ...]\n"); return 2; }
	bool ok = true;
	for (const std::string& m : models) {
		std::vector<uint8_t> bytes;
		superslm::SslmModelView view{};
		std::string err;
		if (!ReadFileBytes(m, &bytes) ||
		    superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) != superslm::SslmModelStatus::Ok) {
			std::printf("FAIL cannot load %s %s\n", m.c_str(), err.c_str());
			ok = false;
			continue;
		}
		std::string name = m.substr(m.find_last_of("/\\") + 1);
		name = name.substr(0, name.find('.'));
		ok = RunOne(name, view, false) && ok;
		ok = RunOne(name, view, true) && ok;
	}
	return ok ? 0 : 1;
}
