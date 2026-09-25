// TE-441 -- GPU cells: the per-site saturation census through GPU prefill, GPU decode, and
// sslm_gpu_seq_save/sslm_gpu_seq_restore, and a v1.8.1-format GPU blob restored at the fix. See
// te441_common.h for the claim and the test-design record. The GPU surface has no prefix
// adoption (include/superslm/gpu_1p0.h declares none), so there is no GPU adopt cell.
//
// Modes (one model per process; the process's shaders load from <exe dir>\shaders):
//   cells   --model=PATH [--mutant=zero-persite]
//                                        G1a, G1b, G2a, G2b, G2c, G4a, G4b (below); the mutant
//                                        is the vitality leg, expected RED
//   genblob --model=PATH --blob=OUT      writes the saved GPU sequence's blob. Run ONLY at v1.8.1
//                                        (84bed02); v181 refuses any blob not pinned below.
//   v181    --model=PATH --blob=PATH [--v181-persite=saved|zero]
//                                        G3: the pinned v1.8.1 GPU blob restored at the engine
//                                        under test: restore succeeds and the next greedy tokens
//                                        equal a never-saved GPU sequence's. Per-site reading as
//                                        in te441_cpu_cells.cpp's v181 mode.
//
// A GPU sequence's census is read through the bench bridge (gpu_1p0_bench_bridge.h): the total is
// the handle's own kv_saturation_count, the four per-site counts are its live SequenceLayerState's
// -- the state sslm_gpu_seq_save serializes. Oracles as in the CPU cells: every expected census is
// read from a live sequence that never went through save or restore.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_1p0_g5_bridge.h"
#include "superslm/gpu_port.h"  // superslm_gpu::kQkNormDispatchesPerLayer
#include "superslm/model.h"
#include "te441_common.h"

uint64_t* SslmGpuSeqHandleKvSaturationForBench(SslmGpuSequenceHandle* seq);

namespace {

struct PinnedBlob {
	const char* model_tag;
	const char* sha256;
	Census saved;
};
static const PinnedBlob kPinnedV181GpuBlobs[] = {
    // Written by `genblob` at 84bed02 into D:/_artifacts/superslm/te441-v181-blobs/.
    // qwen2.5-1.5b-instruct.gpu-slm5.blob, 919,172 bytes.
    {"qwen2.5-1.5b-instruct", "46983b1ca4562bd1e4c70f7b558291605ca480bc16d3f24e7bd40845ae30a6a3",
     Census{6u, 0u, 0u, 6u, 0u}},
    // qwen3-embedding-0.6b-1p5.gpu-slm5.blob, 3,671,172 bytes.
    {"qwen3-embedding-0.6b-1p5", "8adc49cbe6afeec66795a97cb512b0829751f3b4c2d2dd0b272e3103d9d2264b",
     Census{2076u, 5u, 48u, 2023u, 0u}},
};

constexpr int64_t kSeqCap = 64;  // prompt 16 + 12 before save + 8 after, with room

const char* St(SslmGpuStatus s) {
	static char buf[32];
	std::snprintf(buf, sizeof(buf), "%u", static_cast<unsigned>(s));
	return buf;
}

struct GpuModel {
	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	uint32_t budget = 0;

	void Open(const std::string& path) {
		SETUP(!path.empty(), "--model=PATH is required");
		SETUP(ReadFileBytes(path, &bytes), "cannot read %s", path.c_str());
		std::string err;
		SETUP(superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) == superslm::SslmModelStatus::Ok,
		      "SslmModel::Load(%s): %s", path.c_str(), err.c_str());
		// Every layer to full depth in one submission: the per-layer dispatch count is at most
		// kQkNormDispatchesPerLayer (gpu_port.h), so this budget covers the deepest layer shape.
		budget = superslm_gpu::kQkNormDispatchesPerLayer * view.config.num_hidden_layers;
		GpuContextConfig cfg{};
		const SslmGpuStatus cs = sslm_gpu_context_create(cfg, &ctx);
		SETUP(cs == SslmGpuStatus::SSLM_OK && ctx, "sslm_gpu_context_create: %s", St(cs));
		const SslmGpuStatus ms = sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model);
		SETUP(ms == SslmGpuStatus::SSLM_OK && model, "sslm_gpu_model_map(%s): %s", path.c_str(), St(ms));
		std::printf("model %s: layers=%u vocab=%u, GPU sequences at context_cap %lld\n", path.c_str(),
		            view.config.num_hidden_layers, view.config.vocab_size, static_cast<long long>(kSeqCap));
	}
	~GpuModel() {
		if (model) sslm_gpu_model_unmap(ctx, model);
		if (ctx) sslm_gpu_context_destroy(ctx);
	}
};

Census Read(SslmGpuSequenceHandle* s) {
	const superslm::SequenceLayerState* st = SslmGpuSeqHandleLiveStateForBench(s);
	const uint64_t* total = SslmGpuSeqHandleKvSaturationForBench(s);
	SETUP(st != nullptr && total != nullptr, "bench bridge returned null");
	Census c = CensusOf(*st);
	c.total = *total;
	return c;
}

// A GPU sequence with its greedy history: `last` is the most recent token it produced, the one
// the next decode step embeds.
struct GpuSeq {
	SslmGpuSequenceHandle* h = nullptr;
	int32_t last = -1;
};

std::vector<int32_t> Greedy(GpuModel& f, GpuSeq& s, int steps) {
	std::vector<int32_t> t;
	for (int i = 0; i < steps; ++i) {
		int32_t tok = -1;
		const SslmGpuStatus st = SslmGpuSeqDecodeStepForG5Bridge(f.ctx, s.h, s.last, f.budget, &tok);
		if (st != SslmGpuStatus::SSLM_OK) {
			std::printf("  SslmGpuSeqDecodeStepForG5Bridge status %s\n", St(st));
			t.push_back(-999);
			break;
		}
		t.push_back(tok);
		s.last = tok;
	}
	while (static_cast<int>(t.size()) < steps) t.push_back(-998);
	return t;
}

GpuSeq Create(GpuModel& f) {
	GpuSeq s;
	const SslmGpuStatus st = sslm_gpu_seq_create(f.ctx, f.model, kSeqCap, &s.h);
	SETUP(st == SslmGpuStatus::SSLM_OK && s.h, "sslm_gpu_seq_create: %s", St(st));
	return s;
}

void Prefill(GpuModel& f, GpuSeq& s, const int32_t* prompt, int32_t n) {
	const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(f.ctx, s.h, prompt, n, f.budget);
	SETUP(st == SslmGpuStatus::SSLM_OK, "GPU prefill: %s", St(st));
	s.last = prompt[n - 1];
}

void Decode(GpuModel& f, GpuSeq& s, int steps) {
	for (int32_t x : Greedy(f, s, steps)) SETUP(x >= 0, "GPU decode before save produced no token");
}

std::vector<uint8_t> Save(GpuModel& f, SslmGpuSequenceHandle* h) {
	size_t need = 0;
	(void)sslm_gpu_seq_save(f.ctx, h, nullptr, &need);
	SETUP(need > 0, "sslm_gpu_seq_save sizing");
	std::vector<uint8_t> blob(need);
	size_t wrote = blob.size();
	const SslmGpuStatus st = sslm_gpu_seq_save(f.ctx, h, blob.data(), &wrote);
	SETUP(st == SslmGpuStatus::SSLM_OK, "sslm_gpu_seq_save: %s", St(st));
	blob.resize(wrote);
	return blob;
}

const int32_t kOwnN = static_cast<int32_t>(sizeof(kOwnPrompt) / sizeof(kOwnPrompt[0]));
const int32_t kOtherN = static_cast<int32_t>(sizeof(kOtherPrompt) / sizeof(kOtherPrompt[0]));

struct SavedPair {
	GpuSeq s, n;
	Census at_save;
};

// The saved GPU sequence S and its never-saved twin N. `prefill_checks` runs G1a/G1b on S.
SavedPair MakeSavedPair(GpuModel& f, bool prefill_checks) {
	SavedPair p;
	p.s = Create(f);
	Prefill(f, p.s, kOwnPrompt, kOwnN);
	const Census after_prefill = Read(p.s.h);
	PrintCensus("GPU sequence after prefill", after_prefill);
	// G1a: GPU prefill keeps the census summing to the total.
	if (prefill_checks) CheckSumHolds("G1a GPU prefill", after_prefill);
	Decode(f, p.s, kDecodeBeforeSave);
	p.at_save = Read(p.s.h);
	PrintCensus("GPU sequence after decoding, at save", p.at_save);
	// G1b: GPU decode keeps the census summing to the total.
	if (prefill_checks) CheckSumHolds("G1b GPU prefill then decode", p.at_save);
	p.n = Create(f);
	Prefill(f, p.n, kOwnPrompt, kOwnN);
	Decode(f, p.n, kDecodeBeforeSave);
	PrintCensus("never-saved GPU twin", Read(p.n.h));
	SETUP(p.at_save.Sum() > 0 && p.at_save.total > 0,
	      "the saved GPU sequence's census is empty on this model and prompt; the cells cannot discriminate");
	SETUP(Read(p.n.h) == p.at_save, "the never-saved GPU twin's census differs from the saved sequence's");
	SETUP(p.n.last == p.s.last, "the never-saved GPU twin decoded different tokens before the save point");
	return p;
}

GpuSeq Restore(GpuModel& f, const std::vector<uint8_t>& blob, int32_t last, SslmGpuStatus* out) {
	GpuSeq r;
	*out = sslm_gpu_seq_restore(f.ctx, f.model, blob.data(), blob.size(), &r.h);
	r.last = last;
	return r;
}

int RunCells(const Args& a) {
	GpuModel f;
	f.Open(a.model);
	SavedPair p = MakeSavedPair(f, /*prefill_checks=*/true);
	std::vector<uint8_t> blob = Save(f, p.s.h);
	std::printf("GPU blob: %zu bytes, magic %.4s\n", blob.size(), reinterpret_cast<const char*>(blob.data()));
	// --mutant=zero-persite: the vitality leg. The saved blob's four per-site counters are zeroed
	// before every restore, which is the defect class these cells exist to catch (a save format
	// that does not carry them), produced through the real save path. Every G2 census check must
	// then fail; run_suite.ps1 scores this leg against an expected RED. The offsets are
	// GpuSeqBlobHeader's (src/gpu/superslm_gpu.cpp): 88 kv_landing, 96 k_channel_landing,
	// 104 rope_q, 112 rope_k, each a uint64_t, in the 'SLM5' header.
	if (a.mutant == "zero-persite") {
		SETUP(blob.size() >= 120 && std::memcmp(blob.data(), "SLM5", 4) == 0, "mutant needs an 'SLM5' blob");
		std::memset(blob.data() + 88, 0, 32);
		std::printf("MUTANT zero-persite: the blob's four per-site counters zeroed before restore\n");
	} else {
		SETUP(a.mutant.empty(), "--mutant=%s is not zero-persite", a.mutant.c_str());
	}

	// G2a: restored, the four per-site counts equal the saved ones and sum to the restored total.
	SslmGpuStatus st = SslmGpuStatus::SSLM_OK;
	GpuSeq r1 = Restore(f, blob, p.s.last, &st);
	CHECK_MSG(st == SslmGpuStatus::SSLM_OK && r1.h, "G2a GPU restore: %s", St(st));
	if (!r1.h) return FinishCells("te441_gpu cells");
	const Census c_r1 = Read(r1.h);
	PrintCensus("G2a restored GPU sequence", c_r1);
	CheckCensusEquals("G2a GPU restored", c_r1, p.at_save);

	// G4a: the next greedy tokens after the restore equal the never-saved twin's.
	const std::vector<int32_t> t_r1 = Greedy(f, r1, kDecodeAfterRestore);
	const std::vector<int32_t> t_n = Greedy(f, p.n, kDecodeAfterRestore);
	CheckTokensEqual("G4a GPU restored vs never-saved", t_r1, t_n);

	// G2c: after the same tokens, the restored sequence's census equals the twin's, site by site.
	const Census c_r1_after = Read(r1.h);
	const Census c_n_after = Read(p.n.h);
	PrintCensus("G2c restored GPU sequence, after decoding", c_r1_after);
	PrintCensus("G2c never-saved GPU twin, after decoding", c_n_after);
	CheckCensusEquals("G2c GPU restored then decoded vs never-saved", c_r1_after, c_n_after);

	// G2b: restored after the saved sequence and a sequence with its own, different census are both
	// released, so nothing live carries the saved counts.
	GpuSeq u = Create(f);
	Prefill(f, u, kOtherPrompt, kOtherN);
	Decode(f, u, kDecodeBeforeSave);
	const Census c_u = Read(u.h);
	PrintCensus("G2b released sequence", c_u);
	SETUP(!(c_u == p.at_save), "the released sequence's census equals the saved one; G2b cannot discriminate");
	sslm_gpu_seq_release(f.ctx, p.s.h);
	sslm_gpu_seq_release(f.ctx, u.h);
	GpuSeq r2 = Restore(f, blob, p.s.last, &st);
	CHECK_MSG(st == SslmGpuStatus::SSLM_OK && r2.h, "G2b GPU restore: %s", St(st));
	if (r2.h) {
		const Census c_r2 = Read(r2.h);
		PrintCensus("G2b restored GPU sequence", c_r2);
		CheckCensusEquals("G2b GPU restored after its source was released", c_r2, p.at_save);
		// G4b: its next greedy tokens equal the never-saved twin's at the same point.
		CheckTokensEqual("G4b GPU restored after release vs never-saved", Greedy(f, r2, kDecodeAfterRestore), t_n);
		sslm_gpu_seq_release(f.ctx, r2.h);
	}
	sslm_gpu_seq_release(f.ctx, r1.h);
	sslm_gpu_seq_release(f.ctx, p.n.h);
	return FinishCells("te441_gpu cells");
}

int RunGenBlob(const Args& a) {
	SETUP(!a.blob.empty(), "--blob=OUT is required");
	GpuModel f;
	f.Open(a.model);
	SavedPair p = MakeSavedPair(f, /*prefill_checks=*/false);
	const std::vector<uint8_t> blob = Save(f, p.s.h);
	SETUP(WriteFileBytes(a.blob, blob.data(), blob.size()), "cannot write %s", a.blob.c_str());
	const std::vector<int32_t> t = Greedy(f, p.n, kDecodeAfterRestore);
	std::printf("GENBLOB path=%s bytes=%zu magic=%.4s sha256=%s\n", a.blob.c_str(), blob.size(),
	            reinterpret_cast<const char*>(blob.data()), Sha256Hex(blob.data(), blob.size()).c_str());
	std::printf("GENBLOB saved census {%lluu, %lluu, %lluu, %lluu, %lluu} (total, kv_landing, k_channel_landing, rope_q, rope_k)\n",
	            static_cast<unsigned long long>(p.at_save.total), static_cast<unsigned long long>(p.at_save.kv),
	            static_cast<unsigned long long>(p.at_save.kch), static_cast<unsigned long long>(p.at_save.rq),
	            static_cast<unsigned long long>(p.at_save.rk));
	std::printf("GENBLOB never-saved next tokens:");
	for (int32_t x : t) std::printf(" %d", x);
	std::printf("\n");
	sslm_gpu_seq_release(f.ctx, p.s.h);
	sslm_gpu_seq_release(f.ctx, p.n.h);
	return 0;
}

int RunV181(const Args& a) {
	SETUP(!a.blob.empty(), "--blob=PATH is required");
	std::vector<uint8_t> blob;
	SETUP(ReadFileBytes(a.blob, &blob), "cannot read %s", a.blob.c_str());
	const std::string sha = Sha256Hex(blob.data(), blob.size());
	const PinnedBlob* pin = nullptr;
	for (const PinnedBlob& b : kPinnedV181GpuBlobs) {
		if (sha == b.sha256) pin = &b;
	}
	SETUP(pin != nullptr, "blob %s (sha256 %s) is not a pinned v1.8.1 GPU blob", a.blob.c_str(), sha.c_str());
	SETUP(blob.size() >= 4 && std::memcmp(blob.data(), "SLM5", 4) == 0, "pinned GPU blob is not 'SLM5'");
	std::printf("v1.8.1 GPU blob %s (%s): %zu bytes\n", a.blob.c_str(), pin->model_tag, blob.size());
	PrintCensus("census the saved GPU sequence held at 84bed02", pin->saved);

	GpuModel f;
	f.Open(a.model);
	GpuSeq n = Create(f);
	Prefill(f, n, kOwnPrompt, kOwnN);
	Decode(f, n, kDecodeBeforeSave);

	// G3: restore of the v1.8.1-format GPU blob succeeds at the engine under test.
	SslmGpuStatus st = SslmGpuStatus::SSLM_OK;
	GpuSeq r = Restore(f, blob, n.last, &st);
	CHECK_MSG(st == SslmGpuStatus::SSLM_OK && r.h, "G3 GPU restore of the v1.8.1 blob: %s", St(st));
	if (!r.h) return FinishCells("te441_gpu v181");
	const Census c_r = Read(r.h);
	PrintCensus("G3 restored v1.8.1 GPU blob", c_r);
	if (a.v181_persite == "saved") {
		CheckCensusEquals("G3 per-site counts of the restored v1.8.1 GPU blob (pinned: saved)", c_r, pin->saved);
	} else if (a.v181_persite == "zero") {
		CHECK_MSG(c_r.kv == 0 && c_r.kch == 0 && c_r.rq == 0 && c_r.rk == 0,
		          "G3 per-site counts of the restored v1.8.1 GPU blob (pinned: zero) are not all 0");
	} else {
		SETUP(a.v181_persite.empty(), "--v181-persite=%s is not saved|zero", a.v181_persite.c_str());
		std::printf("  G3 per-site counts not asserted: the builder has not named their reading for a v1.8.1 blob\n");
	}
	// G4c: the restored v1.8.1 GPU blob decodes the same next greedy tokens as a never-saved sequence.
	CheckTokensEqual("G4c GPU restored v1.8.1 blob vs never-saved", Greedy(f, r, kDecodeAfterRestore),
	                 Greedy(f, n, kDecodeAfterRestore));
	sslm_gpu_seq_release(f.ctx, r.h);
	sslm_gpu_seq_release(f.ctx, n.h);
	return FinishCells("te441_gpu v181");
}

}  // namespace

int main(int argc, char** argv) {
	const Args a = ParseArgs(argc, argv);
	if (a.mode == "cells") return RunCells(a);
	if (a.mode == "genblob") return RunGenBlob(a);
	if (a.mode == "v181") return RunV181(a);
	std::printf("usage: te441_gpu_cells.exe cells|genblob|v181 --model=PATH [--blob=PATH] [--v181-persite=saved|zero]\n");
	return 2;
}
