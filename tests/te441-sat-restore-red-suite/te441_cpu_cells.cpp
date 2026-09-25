// TE-441 -- CPU cells: the per-site saturation census through sslm_seq_save/sslm_seq_restore, and
// a v1.8.1-format ('SSB4') blob restored at the fix. See te441_common.h for the claim and the
// test-design record.
//
// Modes (one model per process):
//   cells   --model=PATH                 C0, C1a, C1b, C1c, C4a, C4b (below)
//   genblob --model=PATH --blob=OUT      writes the saved sequence's blob. Run ONLY at v1.8.1
//                                        (84bed02); the v181 mode refuses any blob whose SHA-256
//                                        is not one of the pinned v1.8.1 blobs below.
//   v181    --model=PATH --blob=PATH [--v181-persite=saved|zero]
//                                        C2: the pinned v1.8.1 blob restored at the engine under
//                                        test. Restore succeeds and the next greedy tokens equal
//                                        a never-saved sequence's. The per-site reading is the
//                                        builder's decision (TE-440); --v181-persite pins it once
//                                        named: `saved` = the census the saved sequence held,
//                                        `zero` = all four per-site counts 0. Absent, the per-site
//                                        counts are printed and not asserted.
//
// Oracles. Every expected census is read from a LIVE sequence that never went through save or
// restore: the saved sequence itself at the moment it is saved, and its never-saved twin (same
// prompt, same decode) afterwards. Neither the save format nor the restore path produces any
// expected value. The tokens oracle is the twin's own next greedy tokens.
//
// Built by build_suite.bat against superslm_test_injection (the SslmSeqLiveStateForTest accessor
// is compiled only there).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "superslm/sslm_abi.h"
#include "te441_common.h"

extern "C" superslm::SequenceLayerState* SslmSeqLiveStateForTest(sslm_seq seq);

namespace {

// The v1.8.1 blobs genblob wrote at 84bed02 (D:/_artifacts/superslm/te441-v181-blobs/), and the
// census each saved sequence held when it was saved -- read from the live sequence, not the blob.
struct PinnedBlob {
	const char* model_tag;
	const char* sha256;
	Census saved;
};
static const PinnedBlob kPinnedV181Blobs[] = {
    // qwen2.5-1.5b-instruct.cpu-ssb4.blob, 469,763,712 bytes, written by `genblob` at 84bed02.
    {"qwen2.5-1.5b-instruct", "463066b67d55448bf95c2e13da1c7f9a23b792e8a1f132df06c85bceb982a353",
     Census{6u, 0u, 0u, 6u, 0u}},
};

struct CpuModel {
	std::vector<uint8_t> bytes;
	sslm_model m = nullptr;
	void* pool_buf = nullptr;
	sslm_kv_pool pool = nullptr;
	uint32_t blocks = 0;

	void Open(const std::string& path, uint32_t block_count) {
		SETUP(!path.empty(), "--model=PATH is required");
		SETUP(ReadFileBytes(path, &bytes), "cannot read %s", path.c_str());
		SETUP(sslm_model_map(bytes.data(), bytes.size(), &m) == SSLM_OK, "sslm_model_map(%s)", path.c_str());
		blocks = block_count;
		const size_t bs = sslm_kv_block_size(m);
		const size_t size = block_count * bs + sslm_kv_pool_overhead_size(m, block_count);
		pool_buf = ::operator new(size, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		SETUP(sslm_kv_pool_create(m, pool_buf, size, block_count, &pool) == SSLM_OK, "sslm_kv_pool_create");
		std::printf("model %s: kv block %zu bytes, pool of %u blocks\n", path.c_str(), bs, block_count);
	}
	~CpuModel() {
		if (pool) sslm_kv_pool_destroy(pool);
		if (pool_buf) ::operator delete(pool_buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		if (m) sslm_model_unmap(m);
	}
};

Census Read(sslm_seq s) {
	const superslm::SequenceLayerState* st = SslmSeqLiveStateForTest(s);
	SETUP(st != nullptr, "SslmSeqLiveStateForTest returned null");
	return CensusOf(*st);
}

// `steps` greedy tokens. One layer per call: a call that finishes a token returns it (>= 0).
std::vector<int32_t> Greedy(sslm_model m, sslm_seq s, int steps) {
	sslm_decode_params p{};
	p.layer_budget = 1;
	sslm_seq b[1] = {s};
	std::vector<int32_t> t;
	for (int guard = 0; static_cast<int>(t.size()) < steps && guard < 20000; ++guard) {
		int32_t tok = -1;
		const sslm_status st = sslm_decode_step(m, b, 1, &p, nullptr, &tok);
		if (st != SSLM_OK) {
			std::printf("  sslm_decode_step status %d\n", static_cast<int>(st));
			t.push_back(-999);
			break;
		}
		if (tok >= 0) t.push_back(tok);
	}
	while (static_cast<int>(t.size()) < steps) t.push_back(-998);
	return t;
}

sslm_seq PrefillDecode(CpuModel& f, const int32_t* prompt, int32_t n, int decode) {
	sslm_seq s = nullptr;
	SETUP(sslm_seq_create(f.m, &f.pool, &s) == SSLM_OK && s, "sslm_seq_create");
	int32_t consumed = 0;
	SETUP(sslm_prefill(f.m, s, prompt, n, n, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK && consumed == n,
	      "sslm_prefill consumed %d of %d", consumed, n);
	const std::vector<int32_t> t = Greedy(f.m, s, decode);
	for (int32_t x : t) SETUP(x >= 0, "decode before save produced no token");
	return s;
}

std::vector<uint8_t> Save(sslm_seq s) {
	size_t need = 0;
	SETUP(sslm_seq_save(s, nullptr, &need) == SSLM_BUFFER_TOO_SMALL && need > 0, "sslm_seq_save sizing");
	std::vector<uint8_t> blob(need);
	size_t wrote = blob.size();
	SETUP(sslm_seq_save(s, blob.data(), &wrote) == SSLM_OK, "sslm_seq_save");
	blob.resize(wrote);
	return blob;
}

const int32_t kOwnN = static_cast<int32_t>(sizeof(kOwnPrompt) / sizeof(kOwnPrompt[0]));
const int32_t kOtherN = static_cast<int32_t>(sizeof(kOtherPrompt) / sizeof(kOtherPrompt[0]));

// The saved sequence S and its never-saved twin N: same prompt, same decode, never saved.
struct SavedPair {
	sslm_seq s = nullptr, n = nullptr;
	Census at_save;
};

SavedPair MakeSavedPair(CpuModel& f) {
	SavedPair p;
	p.s = PrefillDecode(f, kOwnPrompt, kOwnN, kDecodeBeforeSave);
	p.n = PrefillDecode(f, kOwnPrompt, kOwnN, kDecodeBeforeSave);
	p.at_save = Read(p.s);
	PrintCensus("saved sequence, at save", p.at_save);
	PrintCensus("never-saved twin", Read(p.n));
	// The cells below need a census that is really there: a sequence that never saturated would
	// let a restore that drops every count pass.
	SETUP(p.at_save.Sum() > 0 && p.at_save.total > 0,
	      "the saved sequence's census is empty on this model and prompt; the cells cannot discriminate");
	SETUP(Read(p.n) == p.at_save, "the never-saved twin's census differs from the saved sequence's before any save");
	return p;
}

int RunCells(const Args& a) {
	CpuModel f;
	// Four blocks: the saved sequence, its twin, the first restore and the C1b occupant; the
	// saved sequence's released block goes to a filler, so the occupant's is the only free one.
	f.Open(a.model, /*block_count=*/4);
	SavedPair p = MakeSavedPair(f);

	// C0 (precondition, green at 1.8.1): the live saved sequence's census sums to its total.
	CheckSumHolds("C0 live sequence before save", p.at_save);

	const std::vector<uint8_t> blob = Save(p.s);
	std::printf("blob: %zu bytes, magic %.4s\n", blob.size(), reinterpret_cast<const char*>(blob.data()));

	// C1a: restored on a fresh handle, the four per-site counts equal the saved ones and sum to
	// the restored total.
	sslm_seq r1 = nullptr;
	CHECK_MSG(sslm_seq_restore(f.m, &f.pool, blob.data(), blob.size(), &r1) == SSLM_OK && r1, "C1a restore");
	if (!r1) return FinishCells("te441_cpu cells");
	const Census c_r1 = Read(r1);
	PrintCensus("C1a restored, fresh handle", c_r1);
	CheckCensusEquals("C1a restored on a fresh handle", c_r1, p.at_save);

	// C4a: the next greedy tokens after the restore equal the never-saved twin's.
	const std::vector<int32_t> t_r1 = Greedy(f.m, r1, kDecodeAfterRestore);
	const std::vector<int32_t> t_n = Greedy(f.m, p.n, kDecodeAfterRestore);
	CheckTokensEqual("C4a restored vs never-saved", t_r1, t_n);

	// C1c: the restored handle keeps counting from the carried values: after the same tokens,
	// its census equals the never-saved twin's, site by site.
	const Census c_r1_after = Read(r1);
	const Census c_n_after = Read(p.n);
	PrintCensus("C1c restored, after decoding", c_r1_after);
	PrintCensus("C1c never-saved twin, after decoding", c_n_after);
	CheckCensusEquals("C1c restored then decoded vs never-saved", c_r1_after, c_n_after);

	// C1b: the blob restored into the pool slot a sequence with its own, different census just
	// released. The pool is filled so that slot is the only free one, and the saved sequence
	// itself is released first, so the restore has no live source to read from.
	sslm_seq u = PrefillDecode(f, kOtherPrompt, kOtherN, kDecodeBeforeSave);
	const Census c_u = Read(u);
	PrintCensus("C1b slot's previous occupant", c_u);
	SETUP(!(c_u == p.at_save), "the previous occupant's census equals the saved one; C1b cannot discriminate");
	CHECK_MSG(sslm_seq_release(p.s) == SSLM_OK, "release the saved sequence");
	p.s = nullptr;
	std::vector<sslm_seq> fillers;
	for (;;) {
		sslm_seq x = nullptr;
		if (sslm_seq_create(f.m, &f.pool, &x) != SSLM_OK || !x) break;
		fillers.push_back(x);
	}
	CHECK_MSG(sslm_seq_release(u) == SSLM_OK, "release the previous occupant");
	sslm_seq r2 = nullptr;
	CHECK_MSG(sslm_seq_restore(f.m, &f.pool, blob.data(), blob.size(), &r2) == SSLM_OK && r2,
	          "C1b restore into the released slot");
	if (r2) {
		const Census c_r2 = Read(r2);
		PrintCensus("C1b restored into the released slot", c_r2);
		CheckCensusEquals("C1b restored into a slot released by a sequence with its own census", c_r2, p.at_save);
		// C4b: its next greedy tokens equal the never-saved twin's at the same point.
		CheckTokensEqual("C4b restored into the released slot vs never-saved", Greedy(f.m, r2, kDecodeAfterRestore),
		                 t_n);
		sslm_seq_release(r2);
	}
	for (sslm_seq x : fillers) sslm_seq_release(x);
	sslm_seq_release(r1);
	sslm_seq_release(p.n);
	return FinishCells("te441_cpu cells");
}

int RunGenBlob(const Args& a) {
	SETUP(!a.blob.empty(), "--blob=OUT is required");
	CpuModel f;
	f.Open(a.model, /*block_count=*/3);
	SavedPair p = MakeSavedPair(f);
	const std::vector<uint8_t> blob = Save(p.s);
	SETUP(WriteFileBytes(a.blob, blob.data(), blob.size()), "cannot write %s", a.blob.c_str());
	const std::vector<int32_t> t = Greedy(f.m, p.n, kDecodeAfterRestore);
	std::printf("GENBLOB path=%s bytes=%zu magic=%.4s sha256=%s\n", a.blob.c_str(), blob.size(),
	            reinterpret_cast<const char*>(blob.data()), Sha256Hex(blob.data(), blob.size()).c_str());
	std::printf("GENBLOB saved census {%lluu, %lluu, %lluu, %lluu, %lluu} (total, kv_landing, k_channel_landing, rope_q, rope_k)\n",
	            static_cast<unsigned long long>(p.at_save.total), static_cast<unsigned long long>(p.at_save.kv),
	            static_cast<unsigned long long>(p.at_save.kch), static_cast<unsigned long long>(p.at_save.rq),
	            static_cast<unsigned long long>(p.at_save.rk));
	std::printf("GENBLOB never-saved next tokens:");
	for (int32_t x : t) std::printf(" %d", x);
	std::printf("\n");
	sslm_seq_release(p.s);
	sslm_seq_release(p.n);
	return 0;
}

int RunV181(const Args& a) {
	SETUP(!a.blob.empty(), "--blob=PATH is required");
	std::vector<uint8_t> blob;
	SETUP(ReadFileBytes(a.blob, &blob), "cannot read %s", a.blob.c_str());
	const std::string sha = Sha256Hex(blob.data(), blob.size());
	const PinnedBlob* pin = nullptr;
	for (const PinnedBlob& b : kPinnedV181Blobs) {
		if (sha == b.sha256) pin = &b;
	}
	SETUP(pin != nullptr, "blob %s (sha256 %s) is not a pinned v1.8.1 blob", a.blob.c_str(), sha.c_str());
	SETUP(blob.size() >= 4 && std::memcmp(blob.data(), "SSB4", 4) == 0, "pinned blob is not 'SSB4'");
	std::printf("v1.8.1 blob %s (%s): %zu bytes\n", a.blob.c_str(), pin->model_tag, blob.size());
	PrintCensus("census the saved sequence held at 84bed02", pin->saved);

	CpuModel f;
	f.Open(a.model, /*block_count=*/3);
	sslm_seq n = PrefillDecode(f, kOwnPrompt, kOwnN, kDecodeBeforeSave);

	// C2: restore of the v1.8.1-format blob succeeds at the engine under test.
	sslm_seq r = nullptr;
	const sslm_status st = sslm_seq_restore(f.m, &f.pool, blob.data(), blob.size(), &r);
	CHECK_MSG(st == SSLM_OK && r, "C2 restore of the v1.8.1 blob: status %d", static_cast<int>(st));
	if (!r) return FinishCells("te441_cpu v181");
	const Census c_r = Read(r);
	PrintCensus("C2 restored v1.8.1 blob", c_r);
	if (a.v181_persite == "saved") {
		CheckCensusEquals("C2 per-site counts of the restored v1.8.1 blob (pinned: saved)", c_r, pin->saved);
	} else if (a.v181_persite == "zero") {
		CHECK_MSG(c_r.kv == 0 && c_r.kch == 0 && c_r.rq == 0 && c_r.rk == 0,
		          "C2 per-site counts of the restored v1.8.1 blob (pinned: zero) are not all 0");
	} else {
		SETUP(a.v181_persite.empty(), "--v181-persite=%s is not saved|zero", a.v181_persite.c_str());
		std::printf("  C2 per-site counts not asserted: the builder has not named their reading for a v1.8.1 blob\n");
	}

	// C4c: the restored v1.8.1 blob decodes the same next greedy tokens as a sequence that was never saved.
	CheckTokensEqual("C4c restored v1.8.1 blob vs never-saved", Greedy(f.m, r, kDecodeAfterRestore),
	                 Greedy(f.m, n, kDecodeAfterRestore));
	sslm_seq_release(r);
	sslm_seq_release(n);
	return FinishCells("te441_cpu v181");
}

}  // namespace

int main(int argc, char** argv) {
	const Args a = ParseArgs(argc, argv);
	SETUP(a.mutant.empty(), "--mutant is a GPU-cells leg only");
	if (a.mode == "cells") return RunCells(a);
	if (a.mode == "genblob") return RunGenBlob(a);
	if (a.mode == "v181") return RunV181(a);
	std::printf("usage: te441_cpu_cells.exe cells|genblob|v181 --model=PATH [--blob=PATH] [--v181-persite=saved|zero]\n");
	return 2;
}
