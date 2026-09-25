// TE-441 -- shared harness for the SuperSLM 1.8.2 saturation-census red cells.
//
// The claim under test (include/superslm/forward_sites.h, SequenceLayerState): the four per-site
// saturation counts kv_landing, k_channel_landing, rope_q and rope_k sum to kv_saturation_count.
// 1.8.1 made that hold through CPU prefill and prefix adoption; its release note
// (docs/releases/1.8.1.md) states the counts do not survive sslm_seq_save/sslm_seq_restore. These
// cells hold the sum AND the carried values through save and restore, through a v1.8.1-format
// blob restored at the fix, and on the GPU path, on real models with real saturation: no count
// is seeded anywhere in this suite.
//
// Every cell is a documented-local executable (no GPU CI runner, D-SLM3432), built by
// build_suite.bat against build_engine.bat's output and driven by run_suite.ps1. The test-design
// record, with each cell's oracle and its reading at 84bed02, is
// Claude/Curie/te441-sat-restore-red-2026-09-25.md in the Wizard records tree.
#ifndef SSLM_TE441_COMMON_H
#define SSLM_TE441_COMMON_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/forward_sites.h"  // superslm::SequenceLayerState
#include "superslm/sha256.h"

static int GChecks = 0;
static int GFailures = 0;

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)
#define CHECK(cond) CHECK_MSG(cond, "%s", "")

// A setup failure is not a verdict on the claim: the cell prints INVALID and exits 2, which the
// runner scores apart from a red (1) or a green (0) reading.
#define SETUP(cond, ...) \
	do { \
		if (!(cond)) { \
			std::printf("INVALID %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
			std::exit(2); \
		} \
	} while (0)

inline int FinishCells(const char* name) {
	const bool pass = GFailures == 0 && GChecks > 0;
	std::printf("%s: checks=%d failures=%d -> %s\n", name, GChecks, GFailures, pass ? "PASS" : "FAIL");
	return pass ? 0 : 1;
}

// One sequence's saturation census, read from its live SequenceLayerState.
struct Census {
	uint64_t total = 0, kv = 0, kch = 0, rq = 0, rk = 0;
	uint64_t Sum() const { return kv + kch + rq + rk; }
	bool SitesEqual(const Census& o) const { return kv == o.kv && kch == o.kch && rq == o.rq && rk == o.rk; }
	bool operator==(const Census& o) const { return total == o.total && SitesEqual(o); }
};

inline Census CensusOf(const superslm::SequenceLayerState& s) {
	Census c;
	c.total = s.kv_saturation_count;
	c.kv = s.kv_landing_saturation_count;
	c.kch = s.k_channel_landing_saturation_count;
	c.rq = s.rope_q_saturation_count;
	c.rk = s.rope_k_saturation_count;
	return c;
}

inline void PrintCensus(const char* tag, const Census& c) {
	std::printf("  %-44s total=%llu kv_landing=%llu k_channel_landing=%llu rope_q=%llu rope_k=%llu sum=%llu%s\n",
	            tag, static_cast<unsigned long long>(c.total), static_cast<unsigned long long>(c.kv),
	            static_cast<unsigned long long>(c.kch), static_cast<unsigned long long>(c.rq),
	            static_cast<unsigned long long>(c.rk), static_cast<unsigned long long>(c.Sum()),
	            c.Sum() == c.total ? "" : "  (sum != total)");
}

// The census checks every cell makes: `got` carries `want`'s per-site values and total, and its
// sites sum to its total. `what` names the reading in the failure text.
inline void CheckCensusEquals(const char* what, const Census& got, const Census& want) {
	CHECK_MSG(got.total == want.total, "%s: total %llu, want %llu", what,
	          static_cast<unsigned long long>(got.total), static_cast<unsigned long long>(want.total));
	CHECK_MSG(got.kv == want.kv, "%s: kv_landing %llu, want %llu", what,
	          static_cast<unsigned long long>(got.kv), static_cast<unsigned long long>(want.kv));
	CHECK_MSG(got.kch == want.kch, "%s: k_channel_landing %llu, want %llu", what,
	          static_cast<unsigned long long>(got.kch), static_cast<unsigned long long>(want.kch));
	CHECK_MSG(got.rq == want.rq, "%s: rope_q %llu, want %llu", what,
	          static_cast<unsigned long long>(got.rq), static_cast<unsigned long long>(want.rq));
	CHECK_MSG(got.rk == want.rk, "%s: rope_k %llu, want %llu", what,
	          static_cast<unsigned long long>(got.rk), static_cast<unsigned long long>(want.rk));
	CHECK_MSG(got.Sum() == got.total, "%s: per-site counts sum to %llu, total is %llu", what,
	          static_cast<unsigned long long>(got.Sum()), static_cast<unsigned long long>(got.total));
}

inline void CheckSumHolds(const char* what, const Census& c) {
	CHECK_MSG(c.Sum() == c.total, "%s: per-site counts sum to %llu, total is %llu", what,
	          static_cast<unsigned long long>(c.Sum()), static_cast<unsigned long long>(c.total));
}

inline void CheckTokensEqual(const char* what, const std::vector<int32_t>& got, const std::vector<int32_t>& want) {
	bool same = got.size() == want.size();
	for (size_t i = 0; same && i < got.size(); ++i) same = got[i] == want[i];
	std::string g, w;
	for (int32_t t : got) g += std::to_string(t) + " ";
	for (int32_t t : want) w += std::to_string(t) + " ";
	CHECK_MSG(same, "%s: next greedy tokens [%s], never-saved sequence [%s]", what, g.c_str(), w.c_str());
	bool real = !want.empty();
	for (int32_t t : want) real = real && t >= 0;
	CHECK_MSG(real, "%s: the never-saved sequence did not produce real tokens [%s]", what, w.c_str());
}

inline bool ReadFileBytes(const std::string& path, std::vector<uint8_t>* out) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long long sz = _ftelli64(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	const size_t n = sz > 0 ? std::fread(out->data(), 1, static_cast<size_t>(sz), f) : 0;
	std::fclose(f);
	return n == out->size();
}

inline bool WriteFileBytes(const std::string& path, const void* data, size_t n) {
	std::FILE* f = std::fopen(path.c_str(), "wb");
	if (!f) return false;
	const bool ok = std::fwrite(data, 1, n, f) == n;
	return std::fclose(f) == 0 && ok;
}

inline std::string Sha256Hex(const void* data, size_t n) {
	uint8_t d[32];
	superslm::Sha256Hash(static_cast<const uint8_t*>(data), n, d);
	return superslm::ToHex(d);
}

// argv: <mode> --model=PATH [--blob=PATH] [--v181-persite=saved|zero] [--mutant=zero-persite]
struct Args {
	std::string mode, model, blob, v181_persite, mutant;
};
inline Args ParseArgs(int argc, char** argv) {
	Args a;
	if (argc > 1) a.mode = argv[1];
	for (int i = 2; i < argc; ++i) {
		const std::string s = argv[i];
		auto take = [&](const char* flag) -> const char* {
			const size_t n = std::strlen(flag);
			return s.compare(0, n, flag) == 0 ? s.c_str() + n : nullptr;
		};
		if (const char* m = take("--model=")) a.model = m;
		else if (const char* b = take("--blob=")) a.blob = b;
		else if (const char* p = take("--v181-persite=")) a.v181_persite = p;
		else if (const char* x = take("--mutant=")) a.mutant = x;
	}
	return a;
}

// The real prompts. Both are Qwen-family token ids, valid in both real artifacts' vocabularies
// (Qwen2.5-1.5B-Instruct and Qwen3-Embedding-0.6B share the 151k tokenizer). The saved sequence
// is prefilled with kOwnPrompt and then decodes kDecodeBeforeSave greedy tokens, so its census
// holds both prefill and decode saturation; kOtherPrompt gives a second sequence a different
// history of its own. These are the conductor's probe prompts (TE-430), on which Qwen2.5-1.5B
// saturates at 84bed02 (rope_q = 6 at save).
static const int32_t kOwnPrompt[16] = {9707, 11, 1246, 525, 498, 3351, 30, 358, 1079, 10480, 3645, 311, 1492, 697, 13, 576};
static const int32_t kOtherPrompt[12] = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13, 151645, 198};
constexpr int kDecodeBeforeSave = 12;
constexpr int kDecodeAfterRestore = 8;

#endif  // SSLM_TE441_COMMON_H
