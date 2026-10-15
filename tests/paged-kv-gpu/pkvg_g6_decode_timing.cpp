// Paged-KV plan step G6 and §10 R1 as the plan of record states it (rev 16.30's row, which carries rev 16.16 and rev
// 16.18): the GPU decode timing harness, paged against the legacy view, on two rows under cell 7.9's per-process rule.
//
// R1: "G6 measures paged vs legacy-view decode tok/s on the 0.5B cohort at width 1,712 on the RTX 2080 SUPER. Bar
// (planner default): no more than 5% slower, stated with resolving power. Over the bar -> a fold choosing a larger
// engine page (a larger run target in §3.1's rule) before G7." Rev 16.16 (T-3155 G11): "G6 also runs a 1.5B row at
// width 16,384 to measure GPU set aliasing." Rev 16.18 (T-3162 N-c): "G6's rows use 7.9's per-process rule, §20.4."
// R2 (the table-buffer barrier, once per B tokens in decode) is measured inside it. §8: the harness's readings are
// quarantined until its instrument is commissioned -- recorded, never acted on or headlined.
//
// This replaces the rev 16.2 harness (one process, 508 pairs a repeat, repeats pooled, 0.5B only; §32 records the
// gap). Its readings stay where they were written (tools\run_paged_kv_g6_box.ps1 kept them under <Scratch>\timing);
// none of them is reused here, because the per-process rule's unit of verdict is a process and every one of those
// pairs came from one process.
//
// THE PER-PROCESS RULE (cell 7.9, §20.4), applied to each row:
//   - A row is five processes on one boot. The caller reads the boot (LastBootUpTime) before and after each process
//     and passes it as --boot; a process whose boot changed is discarded and rerun (the runner deletes its record),
//     and a record on another boot than the current one is never mixed: the next process invocation discards every
//     record of the row that is on another boot, so the row restarts on this boot.
//   - Each process loads the model once, builds or restores both arms, decodes 4 untimed warm-up tokens per arm, then
//     times 104 pairs: each pair's order (paged first or legacy first) drawn from mt19937(seed + process), each token
//     timed alone (one SslmGpuSeqDecodeStepForG5Bridge call, which waits for the token; steady_clock). The two arms'
//     tokens must be equal at every step (ready token, untimed steps, warm-up, pairs); a mismatch is RED, never a
//     timing.
//   - The bar is s_i = tau_paged / tau_legacy - 1 <= 0.05 per pair (time-ratio form, T-3155 M2). The process statistic
//     is the median s_i; its exact sign interval of ranks (42, 63) at n = 104, coverage 96.10%, is reported and enters
//     no verdict.
//   - The row rule: PASS when the largest of the five process medians is <= 0.05; FAIL when the median of the five is
//     > 0.05; NO RESULT otherwise. A NO RESULT earns one rerun of five fresh processes (the runner's attempt 2), whose
//     verdict is its own and never pooled; a second NO RESULT goes to the planner.
//   - Resolving power: the between-process spread, printed with every row (the SD of the five medians).
//
// THE ROWS. In both, the legacy arm is a dedicated cap-size sequence (the one-page view, §4 item 2) and the paged arm a
// pooled sequence decoding through its page table at the GPU pool's page, kv_block_size (B = 16 on both artifacts).
//   --row=0.5b  the 0.5B cohort at width 1,712 (g1e_cohort.h's construction, rev 16.30's R1 note): a page pool of
//               exactly the cohort's pages, the world of 1,000, 10 personas by begin_from (each frozen as built), the
//               world released, 50 budget-512 sequences each adopting its persona; the timed member is --timed-seq
//               (default 0). The legacy arm is sslm_gpu_seq_create at the cap, prefilled with the same 1,200 tokens.
//               1,712 is the cohort's limit, so the row is timed at the last decodable widths, as 7.9's cap-width row
//               is: after the ready token, 404 untimed steps per arm (positions 1,200-1,603), the warm-up (1,604-1,607)
//               and the 104 pairs at widths 1,608-1,711 -- the last step attends to 1,712 positions. Built fresh in
//               every process (about 3,600 prefill tokens: no blob is needed).
//   --row=1.5b  the shipped 1.5B at its cap (32,768) at width 16,384: the pairs at widths 16,384-16,487, after the
//               warm-up at 16,380-16,383, from a prompt of 16,380 tokens (g1e Stream(97), the cohort's vocabulary).
//               Prefill near 16k costs minutes per arm on the GPU (batched prefill runs at about the decode rate), so,
//               as 7.9 does, the row's blobs are made once (`prepare`) and each process restores them: the paged arm
//               from the pooled holder's 'SLM6' (sslm_gpu_seq_restore_into_pool, private, into a pool of R(budget)
//               pages; budget 16,488 = width + pairs), the legacy arm from the dedicated holder's 'SLM5'
//               (sslm_gpu_seq_restore, which sizes the fresh handle to the blob's cap: the cap-size one-page view).
//               A manifest binds both blobs' SHA-256, the model's SHA-256, the stream and the ready token; each
//               process checks the blobs against it before restoring (a mismatch is INFRASTRUCTURE, nothing timed).
//
// MODES (one OS process each; tools\run_paged_kv_g6_box.ps1 drives them):
//   identity                                   print the identity line
//   adapter                                    print the DXGI description of SSLM_GPU_ADAPTER_INDEX's adapter (the
//                                              runner checks it before a timing leg: R1 is read on the 2080 SUPER)
//   prepare  --row=1.5b --model=M --blobs=DIR [--width=16384] [--pairs=104] [--warmup=4]
//            the 1.5B row's two blobs. Resumable per blob: each blob is written to .partial and renamed, with a
//            sidecar naming its parameters; a blob whose sidecar matches is kept, so a crash after the first prefill
//            reruns only the second. manifest.txt is written last.
//   process  --row=R --model=M --out=DIR --process=K --boot=STR [--blobs=DIR] [--width=W] [--pairs=104] [--warmup=4]
//            [--seed=N] [--timed-seq=0] [--fresh] [--allow-busy] [--commissioned]
//            one process of the row: process_K.log (written to .partial, renamed after its END line). A complete
//            record of this run on this boot is kept and the process is not rerun (exit 0, nothing opened).
//   grade    --out=DIR      the row rule over process_1..5.log; result.txt
//   verdict  --rows=DIR[,DIR] --result=FILE
//            R1 over the rows' graded attempts: PASS when every row reads PASS, FAIL when any row reads FAIL, NO RESULT
//            otherwise; its status is STANDING only when every row's is.
//
// Run identity. DIR/run.txt holds this executable's and the model's SHA-256, the row, the adapter selection and its
// DXGI description, the commissioning state (so readings taken quarantined are never regraded as standing), the
// width, pairs, warm-up, timed sequence, hook, the blob manifest's SHA-256 (1.5B) and the seed, chosen once and kept.
// A process whose identity differs is refused (exit 40) unless --fresh, which deletes DIR's records first.
//
// Quiet guard (the CPU timing runners' idle check, run_commissioning.py's BUSY_PROCESSES and the box lock, plus the
// GPU's own contenders): before each process the process list and D:\_ssu_build_lock are read; any offender refuses
// the process (exit 40, completed processes kept). They are read again at its end, and a process that ended on a busy
// box is discarded the same way. --allow-busy records the offenders instead and stamps the result NOT_QUIET.
//
// Status of a graded row: MUTANT_CONSTRUCTION when the slow-arm hook is set (SUPERSLM_PKVG_G6_SLOW_ARM=A or B: that
// arm busy-waits 17.65% of its own call time after each timed call, the CPU 7.9 hook's figure, so a correct instrument
// reads s near +0.1765 and must report FAIL for A at either row -- the commissioning must-reject -- and PASS for B);
// MECHANISM_ONLY when the pairs, the warm-up or the 1.5B width are not the plan's (104, 4, 16,384); NOT_QUIET under
// --allow-busy; STANDING only with --commissioned (passed by the runner only after instrument-commission.ps1 -Verify
// lists the receipt); otherwise QUARANTINED.
//
// Exit: process and prepare: 0 complete (or kept); 1 a token mismatch between the arms ("RED:"); 40 infrastructure
// (arguments, artifact, a blob or manifest mismatch, the quiet guard, an identity mismatch, a construction that would
// not build). grade and verdict: 40 when a record is missing or torn; otherwise 0, except 1 for a STANDING or a
// MUTANT_CONSTRUCTION reading of FAIL or NO RESULT (a mutant's grade is the commissioning verdict, so the must-reject
// exits non-zero and the must-accept zero); a QUARANTINED, MECHANISM_ONLY or NOT_QUIET reading exits 0 whatever its
// bar, since it is a reading, not a verdict.
//
// Run once per GPU (SSLM_GPU_ADAPTER_INDEX), on a quiet box, debug layer OFF (a timing run is not a debug-layer run).
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION) || defined(SUPERSLM_GPU_TEST_SEAMS) || \
    defined(SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION) || defined(SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING) || \
    defined(SUPERSLM_ENABLE_GPU_API_FAILURE_INJECTION) || defined(SUPERSLM_ENABLE_BAD_ALLOC_INJECTION) || \
    defined(SUPERSLM_ENABLE_MATVEC_TEST_SEAMS) || defined(SUPERSLM_FORCE_SCALAR_MATMUL)
#error The G6 decode timing harness must be built in the plain configuration: no seam or injection definition.
#endif
#include "g1e_cohort.h"

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <tlhelp32.h>
#include <dxgi1_6.h>
#endif

using namespace g1e;
using namespace g1e::cohort;

namespace {

constexpr const char* kIdentity = "PKVG_G6_DECODE_TIMING_V2 per-process defs=none";
// Cell 7.9's per-process rule (§20.4).
constexpr int kPlanPairs = 104;
constexpr int kPlanWarmup = 4;
constexpr int kProcesses = 5;
constexpr double kBar = 0.05;
// The 1.5B row (rev 16.16).
constexpr int64_t kPlanWidth15 = 16384;
constexpr uint32_t kStreamSeed15 = 97;
using clock_type = std::chrono::steady_clock;
namespace fs = std::filesystem;

[[noreturn]] void Infra(const std::string& why) {
	std::printf("INFRASTRUCTURE: %s\n", why.c_str());
	std::fflush(stdout);
	std::exit(40);
}

std::string FileSha(const std::string& path) {
	std::vector<uint8_t> b;
	if (!ReadFileBytes(path, &b)) return "missing";
	return Sha256Hex(b.data(), b.size());
}
std::string SelfPath() {
#if defined(_WIN32)
	char p[4096]{};
	const DWORD n = GetModuleFileNameA(nullptr, p, sizeof p);
	return n && n < sizeof p ? std::string(p, n) : std::string();
#else
	return fs::read_symlink("/proc/self/exe").string();
#endif
}

// The adapter SSLM_GPU_ADAPTER_INDEX names: d3d12_harness.h selects by the raw EnumAdapters1 index when the variable
// is set, so the same enumeration names it here. Recorded in the run's identity, so a resume on another device (indices
// swapped by a driver update or a new card) is refused, and the result says which GPU R1 was read on.
std::string AdapterDescription(const char* index_env) {
	if (!index_env || !*index_env) return "default selection (SSLM_GPU_ADAPTER_INDEX unset)";
#if defined(_WIN32)
	char* end = nullptr;
	const long idx = std::strtol(index_env, &end, 10);
	if (!end || *end || idx < 0) return std::string("unparsed index ") + index_env;
	IDXGIFactory1* factory = nullptr;
	if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) || !factory) return "no DXGI factory";
	IDXGIAdapter1* a = nullptr;
	std::string out = "no adapter at index " + std::to_string(idx);
	if (SUCCEEDED(factory->EnumAdapters1(static_cast<UINT>(idx), &a)) && a) {
		DXGI_ADAPTER_DESC1 d{};
		if (SUCCEEDED(a->GetDesc1(&d))) {
			out.clear();
			for (const wchar_t* w = d.Description; *w; ++w) out += (*w >= 32 && *w < 127) ? static_cast<char>(*w) : '?';
			char ids[64];
			std::snprintf(ids, sizeof ids, " (vendor %04x device %04x)", d.VendorId, d.DeviceId);
			out += ids;
		}
		a->Release();
	}
	factory->Release();
	return out;
#else
	return std::string("index ") + index_env;
#endif
}

// ---- the quiet guard -----------------------------------------------------------------------------

std::string Lower(std::string s) {
	for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return s;
}
// The CPU runners' offenders (run_commissioning.py BUSY_PROCESSES, run_paged_kv_c6_box.ps1 Get-IdleReport) and the
// GPU's own: another suite, harness or engine process would share the device.
std::vector<std::string> Offenders() {
	std::vector<std::string> out;
	if (fs::exists("D:\\_ssu_build_lock")) out.push_back("D:\\_ssu_build_lock present");
#if defined(_WIN32)
	static const char* const kBusy[] = {"unrealeditor", "unrealeditor-cmd", "cl", "link", "dotnet", "unrealbuildtool", "ninja",
	                                    "cmake", "msbuild", "cc1plus", "smoke_adapter_driver"};
	static const char* const kBusyPrefix[] = {"superslm_", "pkvg_", "g1d_cell_", "g1e_cell_"};
	const DWORD self_pid = GetCurrentProcessId();
	HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	if (snap == INVALID_HANDLE_VALUE) {
		out.push_back("process list unreadable");
		return out;
	}
	PROCESSENTRY32 pe{};
	pe.dwSize = sizeof pe;
	for (BOOL more = Process32First(snap, &pe); more; more = Process32Next(snap, &pe)) {
		if (pe.th32ProcessID == self_pid) continue;
		std::string name = Lower(pe.szExeFile);
		const std::string full = name;
		if (name.size() > 4 && name.compare(name.size() - 4, 4, ".exe") == 0) name.resize(name.size() - 4);
		bool busy = false;
		for (const char* b : kBusy) busy = busy || name == b;
		for (const char* b : kBusyPrefix) busy = busy || name.rfind(b, 0) == 0;
		if (busy) out.push_back("process " + full + " (pid " + std::to_string(pe.th32ProcessID) + ")");
	}
	CloseHandle(snap);
#endif
	return out;
}
std::string Join(const std::vector<std::string>& v) {
	std::string s;
	for (const std::string& x : v) s += (s.empty() ? "" : "; ") + x;
	return s.empty() ? "none" : s;
}

// ---- grading -------------------------------------------------------------------------------------

double Median(std::vector<double> v) {
	if (v.empty()) return 0;
	std::sort(v.begin(), v.end());
	const size_t n = v.size();
	return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}
// The exact sign interval of the median: the 1-based ranks (l, n + 1 - l), l the largest rank with
// P(Bin(n, 1/2) <= l - 1) <= 0.025. At n = 104: (42, 63), coverage 96.10% (§20.4, executed by the plan reviewer).
// l = 0 (n < 6): no interval.
struct SignInterval {
	long lo_rank = 0, hi_rank = 0;
	double coverage = 0;
};
SignInterval SignRanks(size_t n) {
	SignInterval s;
	std::vector<double> pmf(n + 1);
	for (size_t k = 0; k <= n; ++k)
		pmf[k] = std::exp(std::lgamma(double(n) + 1) - std::lgamma(double(k) + 1) - std::lgamma(double(n - k) + 1) - double(n) * std::log(2.0));
	double cdf = 0;
	for (size_t k = 0; k <= n; ++k) {
		cdf += pmf[k];
		if (cdf <= 0.025 + 1e-12) s.lo_rank = static_cast<long>(k) + 1;
		else break;
	}
	if (s.lo_rank < 1) return s;
	s.hi_rank = static_cast<long>(n) + 1 - s.lo_rank;
	for (long k = s.lo_rank; k < s.hi_rank; ++k) s.coverage += pmf[static_cast<size_t>(k)];
	return s;
}
std::string Fmt(double v) {
	char b[64];
	std::snprintf(b, sizeof b, "%.5f", v);
	return b;
}
// PASS when the largest process median is <= 0.05; FAIL when the median of the medians is > 0.05; NO RESULT otherwise.
std::string RowRule(const std::vector<double>& medians) {
	if (medians.size() != static_cast<size_t>(kProcesses)) return "NO RESULT";
	if (*std::max_element(medians.begin(), medians.end()) <= kBar) return "PASS";
	if (Median(medians) > kBar) return "FAIL";
	return "NO RESULT";
}

// ---- the arms ------------------------------------------------------------------------------------

struct Pair {
	char first = 'P';
	double paged_ns = 0, legacy_ns = 0;
	int32_t paged_tok = -1, legacy_tok = -1;
};

// One timed whole token on `s`; the slow-arm hook busy-waits 17.65% of the call's own time (CPU 7.9's figure).
std::pair<double, int32_t> TimedToken(const Env& e, GSeq& s, bool slow) {
	GS st = GS::SSLM_OK;
	const auto t0 = clock_type::now();
	const int32_t tok = Token(e, s, &st);
	if (slow) {
		const double own = std::chrono::duration<double, std::nano>(clock_type::now() - t0).count();
		const auto until = clock_type::now() + std::chrono::nanoseconds(static_cast<int64_t>(own * 0.1765));
		while (clock_type::now() < until) {
		}
	}
	const double ns = std::chrono::duration<double, std::nano>(clock_type::now() - t0).count();
	if (st != GS::SSLM_OK || tok < 0) Infra(std::string("a decode returned ") + StatusName(st) + ", token " + std::to_string(tok));
	return {ns, tok};
}

struct Options {
	std::string mode, row, model, out, blobs, boot, rows, result;
	int process = 0, pairs = kPlanPairs, warmup = kPlanWarmup, timed_seq = 0;
	int64_t width = 0;  // 0: the row's own (1,712 is fixed by the cohort; 16,384 for the 1.5B)
	uint64_t seed = 0;
	bool seed_given = false, fresh = false, allow_busy = false, commissioned = false;
};

// The 1.5B row's arithmetic: the prompt, the pooled holder's budget (origin 0, so it covers the prompt), and the pool.
struct Row15 {
	int64_t width = 0, prompt = 0, budget = 0;
	std::vector<int32_t> Tokens() const { return Stream(kStreamSeed15, static_cast<int32_t>(prompt), kVocab); }
};
Row15 Row15Of(const Options& o) {
	Row15 r;
	r.width = o.width ? o.width : kPlanWidth15;
	r.prompt = r.width - o.warmup;
	r.budget = r.width + o.pairs;  // positions [0, width + pairs): prompt, warm-up, pairs
	return r;
}
uint32_t Pool15Pages(const Geo& g, const Row15& r) { return static_cast<uint32_t>(g.R(r.budget) + 2); }

// ---- the blobs of the 1.5B row ---------------------------------------------------------------------

std::vector<std::string> ReadLines(const fs::path& p) {
	std::vector<std::string> out;
	std::ifstream in(p);
	std::string l;
	while (std::getline(in, l)) {
		if (!l.empty() && l.back() == '\r') l.pop_back();
		out.push_back(l);
	}
	return out;
}
std::map<std::string, std::string> ReadKv(const fs::path& p) {
	std::map<std::string, std::string> kv;
	for (const std::string& l : ReadLines(p)) {
		const size_t eq = l.find('=');
		if (eq != std::string::npos) kv[l.substr(0, eq)] = l.substr(eq + 1);
	}
	return kv;
}
void WriteAtomic(const fs::path& p, const std::string& text) {
	const fs::path tmp = p.string() + ".partial";
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		out << text;
		if (!out) Infra("cannot write " + tmp.string());
	}
	std::error_code ec;
	fs::rename(tmp, p, ec);
	if (ec) Infra("cannot rename " + tmp.string() + " to " + p.string() + ": " + ec.message());
}
void WriteBytesAtomic(const fs::path& p, const std::vector<uint8_t>& b) {
	const fs::path tmp = p.string() + ".partial";
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		out.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
		if (!out) Infra("cannot write " + tmp.string());
	}
	std::error_code ec;
	fs::rename(tmp, p, ec);
	if (ec) Infra("cannot rename " + tmp.string() + " to " + p.string() + ": " + ec.message());
}
std::string Kv(const std::vector<std::pair<std::string, std::string>>& kv) {
	std::string t;
	for (const auto& x : kv) t += x.first + "=" + x.second + "\n";
	return t;
}

// What a blob is a blob of: the model's bytes, the row's arithmetic and the stream. A blob whose sidecar reads this is
// kept by a resumed prepare.
std::vector<std::pair<std::string, std::string>> BlobParams(const std::string& model_sha, const Row15& r, const char* arm) {
	return {{"arm", arm},
	        {"model_sha256", model_sha},
	        {"width", std::to_string(r.width)},
	        {"prompt", std::to_string(r.prompt)},
	        {"budget", std::to_string(r.budget)},
	        {"stream", "g1e::Stream(" + std::to_string(kStreamSeed15) + ", prompt, " + std::to_string(kVocab) + ")"}};
}
struct Manifest {
	bool ok = false;
	std::string sha;  // the manifest file's own SHA-256
	std::map<std::string, std::string> kv;
};
// The manifest, checked against the model and the row's arithmetic; with `blobs_too`, both blobs' SHA-256 against it.
Manifest CheckManifest(const fs::path& dir, const std::string& model_sha, const Row15& r, bool blobs_too, std::string* why) {
	Manifest m;
	const fs::path mf = dir / "manifest.txt";
	if (!fs::exists(mf)) {
		*why = mf.string() + " is missing (run prepare first)";
		return m;
	}
	m.kv = ReadKv(mf);
	m.sha = FileSha(mf.string());
	for (const char* arm : {"paged", "legacy"})
		for (const auto& kv : BlobParams(model_sha, r, arm)) {
			if (kv.first == "arm") continue;
			if (m.kv[kv.first] != kv.second) {
				*why = mf.string() + "'s " + kv.first + " (" + m.kv[kv.first] + ") is not this row's (" + kv.second + ")";
				return m;
			}
		}
	if (m.kv["ready_token"].empty() || m.kv["ready_token"].find_first_not_of("0123456789") != std::string::npos) {
		*why = mf.string() + " has no ready_token";
		return m;
	}
	if (blobs_too)
		for (const char* f : {"paged.slm6", "legacy.slm5"}) {
			const std::string have = FileSha((dir / f).string());
			if (have != m.kv[std::string(f) + "_sha256"]) {
				*why = (dir / f).string() + "'s SHA-256 " + have + " is not the manifest's " + m.kv[std::string(f) + "_sha256"];
				return m;
			}
		}
	m.ok = true;
	return m;
}

int Prepare(const Options& o) {
	if (o.row != "1.5b") Infra("prepare is the 1.5B row's (--row=1.5b): the 0.5B row builds its cohort in every process");
	if (o.blobs.empty()) Infra("prepare needs --blobs=DIR");
	const fs::path dir(o.blobs);
	std::error_code ec;
	fs::create_directories(dir, ec);
	if (!fs::is_directory(dir)) Infra("cannot create " + o.blobs);
	const std::string model_sha = FileSha(o.model);
	if (model_sha == "missing") Infra("cannot read the model " + o.model);
	const Row15 r = Row15Of(o);
	std::string why;
	if (CheckManifest(dir, model_sha, r, true, &why).ok) {
		std::printf("prepare: kept %s (manifest and both blobs match)\n", (dir / "manifest.txt").string().c_str());
		return 0;
	}
	Env e;
	if (!e.Open(o.model, false)) Infra("the model did not open on the GPU");
	const Geo& g = e.geo;
	std::printf("prepare: model %s: B = %lld, cap %lld; prompt %lld, budget %lld, pool %u pages\n", o.model.c_str(),
	            static_cast<long long>(g.B), static_cast<long long>(g.cap), static_cast<long long>(r.prompt),
	            static_cast<long long>(r.budget), Pool15Pages(g, r));
	if (r.prompt < 1 || r.budget > g.cap) Infra("width " + std::to_string(r.width) + " + pairs does not fit the cap " + std::to_string(g.cap));
	const std::vector<int32_t> prompt = r.Tokens();
	std::map<std::string, int32_t> ready;
	for (const char* arm : {"paged", "legacy"}) {
		const bool paged = std::string(arm) == "paged";
		const fs::path blob = dir / (paged ? "paged.slm6" : "legacy.slm5");
		const fs::path side = blob.string() + ".txt";
		const std::vector<std::pair<std::string, std::string>> params = BlobParams(model_sha, r, arm);
		if (fs::exists(blob) && fs::exists(side)) {
			std::map<std::string, std::string> have = ReadKv(side);
			bool same = have["sha256"] == FileSha(blob.string());
			for (const auto& kv : params) same = same && have[kv.first] == kv.second;
			if (same && !have["ready_token"].empty()) {
				ready[arm] = static_cast<int32_t>(std::stol(have["ready_token"]));
				std::printf("prepare: kept %s\n", blob.string().c_str());
				continue;
			}
		}
		fs::remove(side, ec);
		const auto t0 = clock_type::now();
		Rig rig(e, paged ? Pool15Pages(g, r) : 2u);  // the legacy holder draws nothing from its rig's pool
		if (!rig.pool) Infra("the pool did not create");
		GSeq s;
		if (paged) {
			s = rig.Budgeted(static_cast<int32_t>(r.budget));
		} else {
			SslmGpuSequenceHandle* h = nullptr;
			if (sslm_gpu_seq_create(e.ctx, e.model(), g.cap, &h) != GS::SSLM_OK || !h) Infra("sslm_gpu_seq_create at the cap failed");
			s.h = rig.Seq(h);
		}
		if (!s.h) Infra(std::string("the ") + arm + " holder did not create");
		const GS ps = Prefill(e, s, prompt);
		if (ps != GS::SSLM_OK) Infra(std::string("the ") + arm + " prefill of " + std::to_string(prompt.size()) + " tokens: " + StatusName(ps));
		const std::vector<uint8_t> bytes = Save(e, s.h);
		if (bytes.empty()) Infra(std::string("the ") + arm + " save failed");
		if (paged ? !Slm6SelfCheck(bytes, g, static_cast<int32_t>(r.budget), 0, r.prompt, "the paged blob") : !IsMagic(bytes, "SLM5"))
			Infra(std::string("the ") + arm + " blob is not the format its holder saves");
		// The ready token, recorded so every process checks its restored arms against the prefill's own.
		GS st = GS::SSLM_OK;
		const int32_t tok = Token(e, s, &st);
		if (st != GS::SSLM_OK || tok < 0) Infra(std::string("the ") + arm + " ready token: " + StatusName(st));
		ready[arm] = tok;
		WriteBytesAtomic(blob, bytes);
		std::vector<std::pair<std::string, std::string>> sidecar = params;
		sidecar.push_back({"sha256", Sha256Hex(bytes.data(), bytes.size())});
		sidecar.push_back({"bytes", std::to_string(bytes.size())});
		sidecar.push_back({"ready_token", std::to_string(tok)});
		sidecar.push_back({"prepared_by", FileSha(SelfPath())});
		WriteAtomic(side, Kv(sidecar));
		std::printf("prepare: wrote %s (%zu bytes, ready token %d) in %.0f s\n", blob.string().c_str(), bytes.size(), tok,
		            std::chrono::duration<double>(clock_type::now() - t0).count());
		std::fflush(stdout);
	}
	if (GFailures) Infra("a check failed while preparing (see FAIL above)");
	if (ready["paged"] != ready["legacy"]) {
		std::printf("RED: the prepared arms' ready tokens differ (paged %d, legacy %d)\n", ready["paged"], ready["legacy"]);
		return 1;
	}
	std::vector<std::pair<std::string, std::string>> m = BlobParams(model_sha, r, "both");
	m.erase(m.begin());
	m.push_back({"model", Stem(o.model)});
	m.push_back({"paged.slm6_sha256", FileSha((dir / "paged.slm6").string())});
	m.push_back({"legacy.slm5_sha256", FileSha((dir / "legacy.slm5").string())});
	m.push_back({"ready_token", std::to_string(ready["paged"])});
	m.push_back({"page_positions", std::to_string(g.B)});
	m.push_back({"cap", std::to_string(g.cap)});
	WriteAtomic(dir / "manifest.txt", Kv(m));
	std::printf("prepare: wrote %s\n", (dir / "manifest.txt").string().c_str());
	return 0;
}

// ---- one process of a row ----------------------------------------------------------------------------

struct ProcessRun {
	std::vector<Pair> pairs;
	std::string mismatch;  // empty, or where the arms' tokens first differed
};

// Both arms' remaining steps after their construction: the ready token, `skip` untimed steps, the warm-up, the pairs.
ProcessRun DriveArms(const Env& e, GSeq& paged, GSeq& legacy, int skip, const Options& o, uint32_t seed, const std::string& hook,
                     int32_t expect_ready) {
	ProcessRun run;
	const int32_t ra = TimedToken(e, paged, false).second, rb = TimedToken(e, legacy, false).second;
	if (ra != rb || (expect_ready >= 0 && ra != expect_ready)) {
		run.mismatch = "the ready token (paged " + std::to_string(ra) + ", legacy " + std::to_string(rb) +
		               (expect_ready >= 0 ? ", manifest " + std::to_string(expect_ready) : std::string()) + ")";
		return run;
	}
	for (int i = 0; i < skip + o.warmup; ++i)
		if (TimedToken(e, paged, false).second != TimedToken(e, legacy, false).second) {
			run.mismatch = (i < skip ? "untimed step " : "warm-up step ") + std::to_string(i < skip ? i : i - skip);
			return run;
		}
	const bool slow_p = hook == "A", slow_l = hook == "B";
	std::mt19937 random(seed);
	for (int i = 0; i < o.pairs; ++i) {
		Pair p;
		std::pair<double, int32_t> x, y;
		if ((random() & 1u) != 0) {
			p.first = 'P';
			x = TimedToken(e, paged, slow_p);
			y = TimedToken(e, legacy, slow_l);
		} else {
			p.first = 'L';
			y = TimedToken(e, legacy, slow_l);
			x = TimedToken(e, paged, slow_p);
		}
		p.paged_ns = x.first;
		p.paged_tok = x.second;
		p.legacy_ns = y.first;
		p.legacy_tok = y.second;
		run.pairs.push_back(p);
		if (x.second != y.second) {
			run.mismatch = "pair " + std::to_string(i);
			return run;
		}
	}
	return run;
}

// The 0.5B row: the cohort in its exact pool and a dedicated sequence holding the same prompt, built in this process.
ProcessRun Run05(const Env& e, const Options& o, uint32_t seed, const std::string& hook) {
	const Geo& g = e.geo;
	Rig rig(e, static_cast<uint32_t>(CohortPages(g)));
	if (!rig.pool) Infra("the paged arm's pool did not create");
	const GWorld w = BuildWorld(rig);
	if (!w.ok) Infra("the paged arm's world and personas did not build");
	std::vector<GSeq> cohort;
	for (int i = 0; i < kSeqs; ++i) {
		GSeq s = rig.Budgeted(kBudget);
		if (!s.h) Infra("cohort create " + std::to_string(i) + " was refused");
		if (Adopt(rig, s, w.personas[static_cast<size_t>(PersonaOf(i))], kOrigin) != GS::SSLM_OK)
			Infra("cohort adopt " + std::to_string(i) + " was refused");
		cohort.push_back(s);
	}
	GSeq& paged = cohort[static_cast<size_t>(o.timed_seq)];
	SslmGpuSequenceHandle* h = nullptr;
	if (sslm_gpu_seq_create(e.ctx, e.model(), g.cap, &h) != GS::SSLM_OK || !h) Infra("the legacy arm's sslm_gpu_seq_create failed");
	GSeq legacy;
	legacy.h = rig.Seq(h);
	std::vector<int32_t> prompt = WorldTokens();
	const std::vector<int32_t> persona = PersonaTokens(PersonaOf(o.timed_seq));
	prompt.insert(prompt.end(), persona.begin(), persona.end());
	const GS ps = Prefill(e, legacy, prompt);
	if (ps != GS::SSLM_OK) Infra(std::string("the legacy arm's prefill: ") + StatusName(ps));
	// The pairs end at the cohort's limit: positions kLimit - pairs .. kLimit - 1.
	const int skip = kBudget - o.warmup - o.pairs;
	return DriveArms(e, paged, legacy, skip, o, seed, hook, -1);
}

// The 1.5B row: both arms restored from the row's blobs, checked against the manifest first.
ProcessRun Run15(const Env& e, const Options& o, uint32_t seed, const std::string& hook, const std::string& model_sha) {
	const Geo& g = e.geo;
	const Row15 r = Row15Of(o);
	std::string why;
	const Manifest m = CheckManifest(o.blobs, model_sha, r, true, &why);
	if (!m.ok) Infra("the 1.5B row's blobs: " + why);
	std::vector<uint8_t> pb, lb;
	if (!ReadFileBytes((fs::path(o.blobs) / "paged.slm6").string(), &pb) || !ReadFileBytes((fs::path(o.blobs) / "legacy.slm5").string(), &lb))
		Infra("cannot read the row's blobs under " + o.blobs);
	if (Sha256Hex(pb.data(), pb.size()) != m.kv.at("paged.slm6_sha256") || Sha256Hex(lb.data(), lb.size()) != m.kv.at("legacy.slm5_sha256"))
		Infra("a blob changed between its check and its read");
	Rig rig(e, Pool15Pages(g, r));
	if (!rig.pool) Infra("the paged arm's pool did not create");
	const std::vector<int32_t> prompt = r.Tokens();
	GSeq like;
	like.last = prompt.back();
	like.ready = like.primed = true;
	const Restored got = RestoreIntoPool(rig, pb, nullptr, like);
	if (got.status != GS::SSLM_OK || !got.seq.h) Infra(std::string("the paged arm's restore into the pool: ") + StatusName(got.status));
	GSeq paged = got.seq;
	SslmGpuSequenceHandle* h = nullptr;
	const GS ls = sslm_gpu_seq_restore(e.ctx, e.model(), lb.data(), lb.size(), &h);
	if (ls != GS::SSLM_OK || !h) Infra(std::string("the legacy arm's restore: ") + StatusName(ls));
	GSeq legacy = like;
	legacy.h = rig.Seq(h);
	return DriveArms(e, paged, legacy, 0, o, seed, hook, static_cast<int32_t>(std::stol(m.kv.at("ready_token"))));
}

struct ProcessRecord {
	bool complete = false;
	std::string boot;
	std::vector<double> slowdown, paged_ns, legacy_ns;
};
std::string ProcessHeader(const std::string& run_sha, int k, uint32_t seed, const std::string& boot) {
	return "PROCESS run=" + run_sha + " k=" + std::to_string(k) + " seed=" + std::to_string(seed) + " boot=" + boot;
}
// A complete process record: its header names this run's identity digest, the process, its seed and its boot; one PAIR
// line per pair, in order, tokens equal; an END line with the pair count, last.
ProcessRecord ReadProcess(const fs::path& p, const std::string& run_sha, int k, uint32_t seed, int pairs) {
	ProcessRecord r;
	const std::vector<std::string> lines = ReadLines(p);
	const std::string head = "PROCESS run=" + run_sha + " k=" + std::to_string(k) + " seed=" + std::to_string(seed) + " boot=";
	if (lines.empty() || lines[0].rfind(head, 0) != 0) return r;
	r.boot = lines[0].substr(head.size());
	int count = 0;
	for (size_t i = 1; i < lines.size(); ++i) {
		std::stringstream ss(lines[i]);
		std::string tag;
		ss >> tag;
		if (tag == "PAIR") {
			int idx = -1;
			std::string first;
			double a = 0, b = 0;
			int32_t ta = -1, tb = -1;
			ss >> idx >> first >> a >> b >> ta >> tb;
			if (!ss || idx != count || ta != tb || !(a > 0) || !(b > 0) || !std::isfinite(a) || !std::isfinite(b)) return ProcessRecord{};
			r.paged_ns.push_back(a);
			r.legacy_ns.push_back(b);
			r.slowdown.push_back(a / b - 1.0);  // s_i = tau_paged / tau_legacy - 1
			++count;
		} else if (tag == "END") {
			std::string field;
			ss >> field;
			const int n = field.rfind("pairs=", 0) == 0 ? std::atoi(field.c_str() + 6) : -1;
			r.complete = n == pairs && count == pairs && i + 1 == lines.size();
		} else {
			return ProcessRecord{};
		}
	}
	if (!r.complete) return ProcessRecord{};
	return r;
}
fs::path ProcessPath(const fs::path& out, int k) { return out / ("process_" + std::to_string(k) + ".log"); }

// The run's identity: everything a process record depends on. The seed is chosen once and kept by run.txt.
std::vector<std::pair<std::string, std::string>> Identity(const Options& o, const std::string& model_sha, const std::string& hook,
                                                          const std::string& manifest_sha) {
	const char* adapter_env = std::getenv("SSLM_GPU_ADAPTER_INDEX");
	const char* debug_env = std::getenv("SSLM_GPU_ENABLE_DEBUG_LAYER");
	return {
	    {"identity", kIdentity},
	    {"exe_sha256", FileSha(SelfPath())},
	    {"row", o.row},
	    {"model", Stem(o.model)},
	    {"model_sha256", model_sha},
	    {"adapter_index", adapter_env ? adapter_env : "default"},
	    {"adapter", AdapterDescription(adapter_env)},
	    {"commissioned", o.commissioned ? "1" : "0"},
	    {"debug_layer", debug_env ? debug_env : "unset"},
	    {"width", o.row == "1.5b" ? std::to_string(Row15Of(o).width) : std::to_string(kLimit)},
	    {"pairs", std::to_string(o.pairs)},
	    {"warmup", std::to_string(o.warmup)},
	    {"timed_seq", o.row == "0.5b" ? std::to_string(o.timed_seq) : "n/a"},
	    {"hook", hook.empty() ? "none" : hook},
	    {"allow_busy", o.allow_busy ? "1" : "0"},
	    {"blob_manifest_sha256", manifest_sha.empty() ? "n/a" : manifest_sha},
	};
}

int Process(const Options& o) {
	if (o.out.empty() || o.process < 1 || o.process > kProcesses || o.boot.empty())
		Infra("process needs --out=DIR, --process=1..5 and --boot=STR");
	if (o.boot.find_first_of(" \t\r\n") != std::string::npos) Infra("--boot must not contain white space");
	if (o.row == "1.5b" && o.blobs.empty()) Infra("the 1.5B row's process needs --blobs=DIR (prepare writes it)");
	if (o.row == "0.5b" && o.width && o.width != kLimit) Infra("the 0.5B row's width is the cohort's limit, 1,712");
	if (o.row == "0.5b" && o.warmup + o.pairs > kBudget) Infra("the 0.5B row: warm-up + pairs <= 512, the cohort's budget");
	const fs::path out(o.out);
	std::error_code ec;
	fs::create_directories(out, ec);
	if (!fs::is_directory(out)) Infra("cannot create " + o.out);
	if (o.fresh)
		for (const auto& entry : fs::directory_iterator(out)) {
			const std::string n = entry.path().filename().string();
			if (n.rfind("run.txt", 0) == 0 || n.rfind("result.txt", 0) == 0 || n.rfind("process_", 0) == 0) fs::remove(entry.path(), ec);
		}
	const char* hook_env = std::getenv("SUPERSLM_PKVG_G6_SLOW_ARM");
	const std::string hook = hook_env ? hook_env : "";
	if (!hook.empty() && hook != "A" && hook != "B") Infra("SUPERSLM_PKVG_G6_SLOW_ARM must be A or B");
	const char* debug_env = std::getenv("SSLM_GPU_ENABLE_DEBUG_LAYER");
	if (debug_env && std::string(debug_env) == "1")
		std::printf("NOTE: SSLM_GPU_ENABLE_DEBUG_LAYER=1 is set: the debug layer costs time; the identity records it\n");
	const std::string model_sha = FileSha(o.model);
	if (model_sha == "missing") Infra("cannot read the model " + o.model);
	std::string manifest_sha;
	if (o.row == "1.5b") {
		std::string why;
		const Manifest m = CheckManifest(o.blobs, model_sha, Row15Of(o), false, &why);
		if (!m.ok) Infra("the 1.5B row's blobs: " + why);
		manifest_sha = m.sha;
	}
	const std::vector<std::pair<std::string, std::string>> id = Identity(o, model_sha, hook, manifest_sha);
	const fs::path run_txt = out / "run.txt";
	uint64_t seed = o.seed;
	if (fs::exists(run_txt)) {
		std::map<std::string, std::string> have = ReadKv(run_txt);
		for (const auto& kv : id)
			if (have[kv.first] != kv.second)
				Infra("this run's " + kv.first + " (" + kv.second + ") differs from " + run_txt.string() + "'s (" + have[kv.first] +
				      "): pass --fresh to discard those records, or another --out");
		const uint64_t kept = std::stoull(have.count("seed") ? have["seed"] : "0");
		if (o.seed_given && kept != o.seed) Infra("--seed differs from the kept seed " + std::to_string(kept) + " (pass --fresh)");
		seed = kept;
	} else {
		if (!o.seed_given) seed = std::random_device{}() & 0xffffffffu;
		WriteAtomic(run_txt, Kv(id) + "seed=" + std::to_string(seed) + "\n");
	}
	std::vector<uint8_t> run_bytes;
	ReadFileBytes(run_txt.string(), &run_bytes);
	const std::string run_sha = Sha256Hex(run_bytes.data(), run_bytes.size());

	// Never mixed across boots: a complete record on another boot is discarded with every record of the row.
	bool other_boot = false;
	for (int k = 1; k <= kProcesses; ++k) {
		const ProcessRecord r = ReadProcess(ProcessPath(out, k), run_sha, k, static_cast<uint32_t>(seed + k), o.pairs);
		if (r.complete && r.boot != o.boot) other_boot = true;
	}
	if (other_boot) {
		std::printf("a process record of %s is on another boot than %s: the row's records are discarded and it restarts on this boot\n",
		            o.out.c_str(), o.boot.c_str());
		for (int k = 1; k <= kProcesses; ++k) fs::remove(ProcessPath(out, k), ec);
		fs::remove(out / "result.txt", ec);
	}
	const uint32_t ps = static_cast<uint32_t>(seed + static_cast<uint64_t>(o.process));
	const fs::path rec = ProcessPath(out, o.process);
	if (fs::exists(rec)) {
		if (ReadProcess(rec, run_sha, o.process, ps, o.pairs).complete) {
			std::printf("process %d: kept (%s)\n", o.process, rec.string().c_str());
			return 0;
		}
		std::printf("process %d: %s is not a complete record of this run; rerunning it\n", o.process, rec.string().c_str());
		fs::remove(rec, ec);
	}
	fs::remove(out / "result.txt", ec);  // a new process makes any earlier grade stale

	const std::vector<std::string> before = Offenders();
	if (!before.empty() && !o.allow_busy) Infra("process " + std::to_string(o.process) + " refused: the box is not quiet: " + Join(before));
	const auto t0 = clock_type::now();
	Env e;
	if (!e.Open(o.model, o.row == "0.5b")) Infra("the model did not open on the GPU");
	std::printf("row %s, process %d: model %s: B = %lld, cap %lld; seed %u, boot %s, quiet before: %s\n", o.row.c_str(), o.process,
	            o.model.c_str(), static_cast<long long>(e.geo.B), static_cast<long long>(e.geo.cap), ps, o.boot.c_str(),
	            Join(before).c_str());
	std::fflush(stdout);
	const ProcessRun run = o.row == "0.5b" ? Run05(e, o, ps, hook) : Run15(e, o, ps, hook, model_sha);
	if (GFailures) Infra("a check failed while building the arms (see FAIL above)");
	if (!run.mismatch.empty()) {
		std::printf("RED: row %s, process %d: the arms' tokens differ at %s\n", o.row.c_str(), o.process, run.mismatch.c_str());
		return 1;
	}
	const std::vector<std::string> after = Offenders();
	if (!after.empty() && !o.allow_busy)
		Infra("process " + std::to_string(o.process) + " discarded: the box was not quiet at its end: " + Join(after));
	std::string text = ProcessHeader(run_sha, o.process, ps, o.boot) + "\n";
	for (size_t i = 0; i < run.pairs.size(); ++i) {
		char b[200];
		std::snprintf(b, sizeof b, "PAIR %zu %c %.1f %.1f %d %d\n", i, run.pairs[i].first, run.pairs[i].paged_ns, run.pairs[i].legacy_ns,
		              run.pairs[i].paged_tok, run.pairs[i].legacy_tok);
		text += b;
	}
	text += "END pairs=" + std::to_string(run.pairs.size()) + " quiet_before=" + Join(before) + " quiet_after=" + Join(after) + "\n";
	WriteAtomic(rec, text);
	const ProcessRecord back = ReadProcess(rec, run_sha, o.process, ps, o.pairs);
	if (!back.complete) Infra("process " + std::to_string(o.process) + "'s record does not read back");
	std::printf("process %d: %zu pairs, median s = %s, in %.0f s\n", o.process, run.pairs.size(), Fmt(Median(back.slowdown)).c_str(),
	            std::chrono::duration<double>(clock_type::now() - t0).count());
	return 0;
}

// ---- grading a row, and R1 ---------------------------------------------------------------------------

struct RowGrade {
	std::string status, verdict, report;
	std::map<std::string, std::string> id;
};
std::string StatusOf(const std::map<std::string, std::string>& id) {
	const bool mechanism = id.at("pairs") != std::to_string(kPlanPairs) || id.at("warmup") != std::to_string(kPlanWarmup) ||
	                       (id.at("row") == "1.5b" && id.at("width") != std::to_string(kPlanWidth15));
	return id.at("hook") != "none"      ? "MUTANT_CONSTRUCTION"
	       : mechanism                  ? "MECHANISM_ONLY"
	       : id.at("allow_busy") == "1" ? "NOT_QUIET"
	       : id.at("commissioned") == "1" ? "STANDING"
	                                      : "QUARANTINED";
}
int ExitFor(const std::string& status, const std::string& verdict) {
	if (status == "STANDING" || status == "MUTANT_CONSTRUCTION") return verdict == "PASS" ? 0 : 1;
	return 0;
}

int Grade(const Options& o) {
	if (o.out.empty()) Infra("grade needs --out=DIR");
	const fs::path out(o.out), run_txt = out / "run.txt";
	if (!fs::exists(run_txt)) Infra(run_txt.string() + " is missing: no process of this row has run");
	std::map<std::string, std::string> id = ReadKv(run_txt);
	for (const char* k : {"row", "pairs", "warmup", "hook", "allow_busy", "commissioned", "seed", "width"})
		if (!id.count(k)) Infra(run_txt.string() + " has no " + k);
	std::vector<uint8_t> run_bytes;
	ReadFileBytes(run_txt.string(), &run_bytes);
	const std::string run_sha = Sha256Hex(run_bytes.data(), run_bytes.size());
	const uint64_t seed = std::stoull(id["seed"]);
	const int pairs = std::stoi(id["pairs"]);
	std::vector<ProcessRecord> rec;
	std::string missing;
	for (int k = 1; k <= kProcesses; ++k) {
		rec.push_back(ReadProcess(ProcessPath(out, k), run_sha, k, static_cast<uint32_t>(seed + k), pairs));
		if (!rec.back().complete) missing += (missing.empty() ? "" : ",") + std::to_string(k);
	}
	if (!missing.empty()) Infra("the row is not complete: process record(s) " + missing + " missing or torn under " + o.out);
	for (const ProcessRecord& r : rec)
		if (r.boot != rec[0].boot) Infra("the row's processes are on more than one boot (" + rec[0].boot + ", " + r.boot + "): never mixed");
	const SignInterval si = SignRanks(static_cast<size_t>(pairs));
	std::vector<double> medians;
	std::string report;
	for (int k = 1; k <= kProcesses; ++k) {
		const ProcessRecord& r = rec[static_cast<size_t>(k - 1)];
		std::vector<double> s = r.slowdown;
		std::sort(s.begin(), s.end());
		const double med = Median(s);
		medians.push_back(med);
		report += "G6 R1 row " + id["row"] + " process " + std::to_string(k) + ": n=" + std::to_string(s.size()) + " median_s=" + Fmt(med);
		if (si.lo_rank >= 1)
			report += " sign_interval=[" + Fmt(s[static_cast<size_t>(si.lo_rank - 1)]) + "," + Fmt(s[static_cast<size_t>(si.hi_rank - 1)]) +
			          "] ranks=(" + std::to_string(si.lo_rank) + "," + std::to_string(si.hi_rank) + ") coverage=" + Fmt(si.coverage) +
			          " (reported; no verdict)";
		else
			report += " sign_interval=none";
		report += " paged_tok_s=" + Fmt(1e9 / Median(r.paged_ns)) + " legacy_tok_s=" + Fmt(1e9 / Median(r.legacy_ns)) + "\n";
	}
	double mean = 0, var = 0;
	for (double m : medians) mean += m / kProcesses;
	for (double m : medians) var += (m - mean) * (m - mean) / (kProcesses - 1);
	const std::string verdict = RowRule(medians);
	const std::string status = StatusOf(id);
	report += "G6 R1 row " + id["row"] + ": process_medians=";
	for (size_t i = 0; i < medians.size(); ++i) report += (i ? "," : "") + Fmt(medians[i]);
	report += " max=" + Fmt(*std::max_element(medians.begin(), medians.end())) + " median_of_medians=" + Fmt(Median(medians)) +
	          " between_process_sd=" + Fmt(std::sqrt(var)) + " boot=" + rec[0].boot + " rule=" + verdict + " status=" + status + "\n";
	report += status == "STANDING" ? "G6 R1 row " + id["row"] + " verdict: " + verdict + " (STANDING)\n"
	          : status == "MUTANT_CONSTRUCTION"
	              ? "G6 R1 row " + id["row"] + " verdict: " + verdict + " (MUTANT_CONSTRUCTION, hook " + id["hook"] + ": the commissioning reading)\n"
	              : "G6 R1 row " + id["row"] + " verdict: none (" + status + "; the row rule reads " + verdict + ", not a verdict)\n";
	std::printf("%s", report.c_str());
	std::string result;
	for (const std::string& l : ReadLines(run_txt)) result += l + "\n";
	result += "run_sha256=" + run_sha + "\nrow_rule=" + verdict + "\nstatus=" + status + "\n" + report;
	WriteAtomic(out / "result.txt", result);
	return ExitFor(status, verdict);
}

int Verdict(const Options& o) {
	if (o.rows.empty() || o.result.empty()) Infra("verdict needs --rows=DIR[,DIR] and --result=FILE");
	std::vector<std::string> dirs;
	std::stringstream ss(o.rows);
	for (std::string d; std::getline(ss, d, ',');)
		if (!d.empty()) dirs.push_back(d);
	std::string report, rows;
	bool all_pass = true, any_fail = false, all_standing = true, all_mutant = true;
	std::string hooks;
	for (const std::string& d : dirs) {
		const fs::path res = fs::path(d) / "result.txt";
		if (!fs::exists(res)) Infra(res.string() + " is missing: grade the row first");
		std::map<std::string, std::string> kv = ReadKv(res);
		if (!kv.count("row_rule") || !kv.count("status") || !kv.count("row")) Infra(res.string() + " is not a graded row");
		rows += (rows.empty() ? "" : ",") + kv["row"];
		all_pass = all_pass && kv["row_rule"] == "PASS";
		any_fail = any_fail || kv["row_rule"] == "FAIL";
		all_standing = all_standing && kv["status"] == "STANDING";
		all_mutant = all_mutant && kv["status"] == "MUTANT_CONSTRUCTION";
		hooks += (hooks.empty() ? "" : ",") + kv["hook"];
		report += "G6 R1 row " + kv["row"] + " (" + d + "): rule=" + kv["row_rule"] + " status=" + kv["status"] + " adapter=" + kv["adapter"] +
		          "\n";
	}
	const std::string verdict = any_fail ? "FAIL" : all_pass ? "PASS" : "NO RESULT";
	const std::string status = all_standing ? "STANDING" : all_mutant ? "MUTANT_CONSTRUCTION" : "NOT STANDING";
	const bool both = rows.find("0.5b") != std::string::npos && rows.find("1.5b") != std::string::npos;
	report += "G6 R1: rows=" + rows + (both ? "" : " (not both rows: R1 as the plan states it needs the 0.5B and the 1.5B rows)") +
	          " rule=" + verdict + " status=" + status + "\n";
	report += status == "STANDING" && both ? "G6 R1 verdict: " + verdict + " (STANDING; bar: paged no more than 5% slower than the legacy view)\n"
	                                       : "G6 R1 verdict: none (" + status + (both ? "" : ", rows missing") + "; the rows read " + verdict + ")\n";
	std::printf("%s", report.c_str());
	WriteAtomic(o.result, report);
	if (status == "STANDING" && both) return verdict == "PASS" ? 0 : 1;
	if (status == "MUTANT_CONSTRUCTION") return verdict == "PASS" ? 0 : 1;
	return 0;
}

Options Parse(int argc, char** argv) {
	Options o;
	if (argc < 2) Infra("usage: pkvg_g6_decode_timing identity|prepare|process|grade|verdict [--flags] (see the header)");
	o.mode = argv[1];
	for (int i = 2; i < argc; ++i) {
		const std::string a = argv[i];
		auto val = [&](const char* f, std::string* out) {
			const size_t n = std::strlen(f);
			if (a.compare(0, n, f) != 0) return false;
			*out = a.substr(n);
			return true;
		};
		std::string v;
		try {
			if (val("--row=", &o.row) || val("--model=", &o.model) || val("--out=", &o.out) || val("--blobs=", &o.blobs) ||
			    val("--boot=", &o.boot) || val("--rows=", &o.rows) || val("--result=", &o.result))
				continue;
			if (val("--process=", &v)) o.process = std::stoi(v);
			else if (val("--pairs=", &v)) o.pairs = std::stoi(v);
			else if (val("--warmup=", &v)) o.warmup = std::stoi(v);
			else if (val("--width=", &v)) o.width = std::stoll(v);
			else if (val("--timed-seq=", &v)) o.timed_seq = std::stoi(v);
			else if (val("--seed=", &v)) {
				o.seed = std::stoull(v);
				o.seed_given = true;
			} else if (a == "--fresh") o.fresh = true;
			else if (a == "--allow-busy") o.allow_busy = true;
			else if (a == "--commissioned") o.commissioned = true;
			else Infra("unknown argument " + a);
		} catch (const std::exception&) {
			Infra("invalid value in " + a);
		}
	}
	if (o.mode == "prepare" || o.mode == "process") {
		if (o.row != "0.5b" && o.row != "1.5b") Infra("--row=0.5b or --row=1.5b is required");
		if (o.model.empty()) Infra("--model=PATH is required");
		if (o.pairs < 1 || o.warmup < 0) Infra("--pairs >= 1, --warmup >= 0");
		if (o.width < 0 || (o.row == "1.5b" && o.width && o.width <= o.warmup)) Infra("--width must exceed the warm-up");
		if (o.timed_seq < 0 || o.timed_seq >= kSeqs) Infra("--timed-seq must be 0..49");
		if (o.seed > 0xffffffffull) Infra("--seed must fit 32 bits");
	}
	return o;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc == 2 && std::string(argv[1]) == "identity") {
		std::printf("%s\n", kIdentity);
		return 0;
	}
	if (argc == 2 && std::string(argv[1]) == "adapter") {
		std::printf("adapter: %s\n", AdapterDescription(std::getenv("SSLM_GPU_ADAPTER_INDEX")).c_str());
		return 0;
	}
	const Options o = Parse(argc, argv);
	std::printf("%s\n", kIdentity);
	if (o.mode == "prepare") return Prepare(o);
	if (o.mode == "process") return Process(o);
	if (o.mode == "grade") return Grade(o);
	if (o.mode == "verdict") return Verdict(o);
	Infra("unknown mode " + o.mode + " (identity, prepare, process, grade, verdict)");
}
