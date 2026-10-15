// Paged-KV plan (rev 16.2) step G1: the GPU red suite's process runner. One command runs the whole suite
// at a step, one process per job, on the GPU the environment selects (SSLM_GPU_ADAPTER_INDEX):
//   - the in-process runner of the step (pkvg_g2 .. pkvg_g5; pkvg_main.cpp), which carries every
//     g1a/g1b/g1c cell of the default engine at or before the step;
//   - the executables of the other engine configurations (pkvg_alloc_g3, pkvg_bench_g3,
//     pkvg_firsttouch_g3) and the cells that own their main() (g1d: per-cell engine configurations and a
//     fresh process per stranded leg; g1e: per-cell executables, and 9.8's second leg, which is the
//     v1.12.0 reader built from the other tree).
// It replaces g1d's draft job runner (g1d_main_draft.cpp), whose job table and rules it keeps: selection
// by job-id prefix or '=' exact id, a selector that selects nothing fails the run, a job whose artifact
// was not given is RED (a skip fails the run). Every job runs with SSLM_GPU_ENABLE_DEBUG_LAYER=1 except the
// te425 hook jobs, which run with it =0 (DebugMode, below); each job's line and the summary say which.
//
// Usage: pkvg_suite --bin=DIR [--step=G2|G3|G4|G5|G6] [--list] [--no-debug-layer] [artifact flags] [JOB ...]
//   --bin=DIR           where build_paged_kv_gpu.bat (or _mingw.sh) put the executables
//   --step=GN           run the jobs whose bracket is at or before GN, and the in-process runner pkvg_gN
//                       (G6 adds no in-process cell: at G6 the runner is pkvg_g5) (default G6: everything)
//   --pkv-fixtures=DIR  tools/gen_paged_kv_fixture.py's output (pkv_def.sslm, pkv_qk.sslm, ...)
//   --pkv-reference=DIR tests/paged-kv/reference (default: relative to the working directory, the repo root)
//   --gpu-ref=DIR       6.5a's pre-G2 GPU records (gpu_pre_g2_<fx>_adapter<N>.ref)
//   --model05=PATH --model15=PATH   the real Qwen2.5-0.5B / 1.5B artifacts (g1a 4.8, g1b 11.4, the G6 box legs)
//   --qwen3=PATH --r15=PATH --g5=PATH  g1d's artifacts (te425's --qwen3, --r15, --g5)
//   --adapter=PATH --adapter-other-base=PATH  g1c's LoRA adapters (3.6d adapter_map, adapter_other_base)
//   --out98=DIR         9.8's exchange directory (SUPERSLM_PKV_GPU_98_OUT)
//   --old-bin=DIR       where the v1.12.0 g1e_old_reader_9_8.exe was built (9.8 leg 2)
//   --no-debug-layer    run every job with SSLM_GPU_ENABLE_DEBUG_LAYER=0, the layer-on jobs included
//                       (software-GPU smoke runs only; a run without it is not an acceptance run, and 3.6
//                       fails by design)
// Exit 0 only when at least one job ran, every selector matched and every job exited 0.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

// The D3D12 debug layer a job runs under (ruling 2026-10-13). Every job runs with it ON, except the jobs
// that install te425's API-failure hooks (te425::InstallHooks): under the layer, context creation fails
// or a hooked Signal reports a failure while the call reads OK, so a hook leg cannot run there. Those run
// with it OFF. A job is a hook job when its cell source calls te425::InstallHooks on its path:
// g1c_cell_3_6d_first_touch.cpp (3.6d first touch), g1d_cell_3_6f.cpp (--run=stranded and
// --run=waitnoarm; --run=first installs none and asserts the layer), and every leg of
// g1d_cell_3_6g_stranded.cpp (3.6g stranded, per-context, and the nine stranded pins).
enum DebugMode { kOn, kHook };

struct Job {
	const char* id;    // "pkvg/g3", "3.6f/first", "9.8/leg2"
	int bracket;       // the step at which it can first link (2..6)
	int only_at;       // nonzero: the in-process runner of exactly that step (pkvg_gN)
	const char* exe;   // under --bin ("old:" prefix: under --old-bin)
	const char* args;  // {key} expands to the flag's value; a "--x={key}" word whose value is empty is dropped
	const char* need;  // the artifact keys whose absence makes the job RED before it runs ("" = none)
	DebugMode debug;   // kOn: SSLM_GPU_ENABLE_DEBUG_LAYER=1; kHook: =0 (a te425 hook job)
};

// Common flags of the in-process runners.
#define PKVG_COMMON "--pkv-fixtures={fx} --pkv-reference={ref} --gpu-ref={gpuref} --model05={m05} --model15={m15}"
#define PKVG_G1C "--adapter={adapter} --adapter-other-base={adapter_other}"

const Job kJobs[] = {
    // In-process runners: every g1a/g1b/g1c cell of the default engine at or before the step.
    {"pkvg/g2", 2, 2, "pkvg_g2.exe", PKVG_COMMON, "fx", kOn},
    {"pkvg/g3", 3, 3, "pkvg_g3.exe", PKVG_COMMON " " PKVG_G1C, "fx", kOn},
    {"pkvg/g4", 4, 4, "pkvg_g4.exe", PKVG_COMMON " " PKVG_G1C, "fx", kOn},
    {"pkvg/g5", 5, 5, "pkvg_g5.exe", PKVG_COMMON " " PKVG_G1C, "fx", kOn},
    // Allocation-fault engine: 11.4/G3, and its zero-fill leg alone in a fresh process (terminal context).
    // Not PKVG_COMMON: pkvg_alloc_g3 links group g1b only, which takes no --gpu-ref (an untaken flag fails the run).
    {"11.4/G3", 3, 0, "pkvg_alloc_g3.exe", "--pkv-fixtures={fx} --pkv-reference={ref} --model05={m05} --model15={m15} 11.4/G3", "fx", kOn},
    {"11.4/G3:zerofill", 3, 0, "pkvg_alloc_g3.exe", "--pkv-fixtures={fx} =11.4/G3:zerofill", "fx", kOn},
    // review-g23 F2: the pooled release that meets a refcount underflow still finishes.
    {"11.3/G3:release", 3, 0, "pkvg_alloc_g3.exe", "--pkv-fixtures={fx} =11.3/G3:release", "fx", kOn},
    // Bench engine: 3.6d's bench prefill and its count==0 precedence.
    {"3.6d/G3:bench", 3, 0, "pkvg_bench_g3.exe", "--pkv-fixtures={fx}", "fx", kOn},
    // Allocation-fault engine + te425 hooks: 3.6d's first touch, alone in a fresh process.
    {"3.6d/G3:first_touch", 3, 0, "pkvg_firsttouch_g3.exe", "--pkv-fixtures={fx} =3.6d/G3:first_touch", "fx", kHook},
    // g1d (its job table, unchanged): 3.6e (allocation-fault build; the G3-only binary links at G3).
    {"3.6e/g3", 3, 0, "g1d_cell_3_6e_g3.exe", "--part=g3 --model={r15}", "r15", kOn},
    {"3.6e/g4", 4, 0, "g1d_cell_3_6e.exe", "--part=g4 --model={r15}", "r15", kOn},
    // 3.6f (allocation-fault build + te425 hooks), each run its own process.
    {"3.6f/first", 3, 0, "g1d_cell_3_6f.exe", "--run=first --model={r15}", "r15", kOn},
    {"3.6f/stranded", 3, 0, "g1d_cell_3_6f.exe", "--run=stranded --model={r15}", "r15", kHook},
    {"3.6f/waitnoarm", 3, 0, "g1d_cell_3_6f.exe", "--run=waitnoarm --model={r15}", "r15", kHook},
    // 3.6g recovered (T2169 build), stranded and per-context (allocation-fault build + te425 hooks).
    {"3.6g/recovered", 3, 0, "g1d_cell_3_6g_recovered.exe", "--model={qwen3}", "qwen3", kOn},
    {"3.6g/stranded.signal", 3, 0, "g1d_cell_3_6g_stranded.exe", "--leg=3.6g.signal --model={r15}", "r15", kHook},
    {"3.6g/stranded.setevent", 3, 0, "g1d_cell_3_6g_stranded.exe", "--leg=3.6g.setevent --model={r15}", "r15", kHook},
    {"3.6g/percontext", 3, 0, "g1d_cell_3_6g_stranded.exe", "--leg=3.6g.percontext --model={qwen3}", "qwen3", kHook},
    // §4.4's nine stranded pins (G3's gate), each alone in a fresh process.
    {"pin/P.setevent1.oom", 3, 0, "g1d_cell_3_6g_stranded.exe", "--leg=P.setevent1.oom --model={qwen3}", "qwen3", kHook},
    {"pin/P.setevent1.fail", 3, 0, "g1d_cell_3_6g_stranded.exe", "--leg=P.setevent1.fail --model={qwen3}", "qwen3", kHook},
    {"pin/P.setevent2.fail", 3, 0, "g1d_cell_3_6g_stranded.exe", "--leg=P.setevent2.fail --model={qwen3}", "qwen3", kHook},
    {"pin/D.setevent1.fail", 3, 0, "g1d_cell_3_6g_stranded.exe", "--leg=D.setevent1.fail --model={qwen3}", "qwen3", kHook},
    {"pin/R.signal2.oom", 3, 0, "g1d_cell_3_6g_stranded.exe", "--leg=R.signal2.oom --model={qwen3}", "qwen3", kHook},
    {"pin/S.setevent1.fail", 3, 0, "g1d_cell_3_6g_stranded.exe", "--leg=S.setevent1.fail --model={g5}", "g5", kHook},
    {"pin/C.signal1.oom", 3, 0, "g1d_cell_3_6g_stranded.exe", "--leg=C.signal1.oom --model={qwen3}", "qwen3", kHook},
    {"pin/M.signal1.oom", 3, 0, "g1d_cell_3_6g_stranded.exe", "--leg=M.signal1.oom --model={qwen3}", "qwen3", kHook},
    {"pin/L.setevent1.oom", 3, 0, "g1d_cell_3_6g_stranded.exe", "--leg=L.setevent1.oom --model={r15}", "r15", kHook},
    // 3.12 (default build). The G4-only binary links at G4; the full binary needs G5's restore-into-pool.
    {"3.12/g4-set", 4, 0, "g1d_cell_3_12_g4.exe", "--run=g4-set --model={r15}", "r15", kOn},
    {"3.12/g4-cleared", 4, 0, "g1d_cell_3_12_g4.exe", "--run=g4-cleared --model={r15}", "r15", kOn},
    {"3.12/g5-set", 5, 0, "g1d_cell_3_12.exe", "--run=g5-set --model={r15}", "r15", kOn},
    {"3.12/g5-cleared", 5, 0, "g1d_cell_3_12.exe", "--run=g5-cleared --model={r15}", "r15", kOn},
    {"3.12/g5-clear", 5, 0, "g1d_cell_3_12.exe", "--run=g5-clear --model={r15}", "r15", kOn},
    // g1e: one executable per cell (default build).
    {"9.6", 5, 0, "g1e_cell_9_6.exe", "--fixture-dir={fx}", "fx", kOn},
    {"9.12", 5, 0, "g1e_cell_9_12.exe", "--fixture-dir={fx}", "fx", kOn},
    {"9.13", 5, 0, "g1e_cell_9_13.exe", "--fixture-dir={fx}", "fx", kOn},
    {"8.17", 5, 0, "g1e_cell_8_17.exe", "--fixture-dir={fx}", "fx", kOn},
    // G5 review F1/F2: a pooled holder at its cap saves and restores; GPU SLM6 K/V == CPU SSB6 K/V.
    {"G5/slm6-cap", 5, 0, "g1e_cell_slm6_cap.exe", "--fixture-dir={fx}", "fx", kOn},
    // 1.13.0 contract fold: the same-handle decode race, every decode entry point, dedicated and pooled.
    {"G5/same-handle-race", 5, 0, "pkvg_same_handle_race.exe", "--fixture-dir={fx}", "fx", kOn},
    // 9.8: leg 1 writes the pooled SLM6 and dedicated SLM5 files into --out98; leg 2 is the v1.12.0
    // reader (built from the other tree, --old-bin) refusing SLM6 with 7. Leg 2 runs after leg 1.
    {"9.8/leg1", 5, 0, "g1e_cell_9_8.exe", "--fixture-dir={fx}", "fx out98", kOn},
    {"9.8/leg2", 5, 0, "old:g1e_old_reader_9_8.exe", "--model={fx}/pkv_def.sslm --in={out98}", "fx out98 oldbin", kOn},
    // G6 (§7 10.3, 10.5, 10.7): the GPU cohort and admission cells (g1e_cohort.h). Each has a fixture leg on pkv_def
    // (B = 16 at cap 4,096, the geometry the plan's figures were executed at; it runs on the software GPU) and a box
    // leg on the real 0.5B. Both read the R0 records under --pkv-reference. The G6 decode timing (§10 R1) is not a
    // suite job: tools\run_paged_kv_g6_box.ps1 runs pkvg_g6_decode_timing.exe on a quiet box.
    {"10.3/fixture", 6, 0, "g1e_cell_10_3.exe", "--leg=fixture --fixture-dir={fx} --pkv-reference={ref}", "fx", kOn},
    {"10.3/box", 6, 0, "g1e_cell_10_3.exe", "--leg=box --model05={m05} --pkv-reference={ref}", "m05", kOn},
    {"10.5/fixture", 6, 0, "g1e_cell_10_5.exe", "--leg=fixture --fixture-dir={fx}", "fx", kOn},
    {"10.5/box", 6, 0, "g1e_cell_10_5.exe", "--leg=box --model05={m05}", "m05", kOn},
    {"10.7/fixture", 6, 0, "g1e_cell_10_7.exe", "--leg=fixture --fixture-dir={fx} --pkv-reference={ref}", "fx", kOn},
    {"10.7/box", 6, 0, "g1e_cell_10_7.exe", "--leg=box --model05={m05} --pkv-reference={ref}", "m05", kOn},
};

bool Matches(const char* id, const std::string& arg) {
	return arg[0] == '=' ? arg.compare(1, std::string::npos, id) == 0 : std::strncmp(id, arg.c_str(), arg.size()) == 0;
}

std::vector<std::string> Words(const std::string& s) {
	std::vector<std::string> w;
	size_t i = 0;
	while (i < s.size()) {
		while (i < s.size() && s[i] == ' ') ++i;
		size_t j = i;
		while (j < s.size() && s[j] != ' ') ++j;
		if (j > i) w.push_back(s.substr(i, j - i));
		i = j;
	}
	return w;
}

// Expands {key}; a word that names an empty value is dropped (the cell then reports the artifact it lacks).
std::string Expand(const std::string& tmpl, const std::map<std::string, std::string>& v) {
	std::string out;
	for (const std::string& w : Words(tmpl)) {
		std::string e;
		bool empty_value = false;
		for (size_t i = 0; i < w.size(); ++i) {
			if (w[i] == '{') {
				const size_t close = w.find('}', i);
				const std::string key = w.substr(i + 1, close - i - 1);
				const auto it = v.find(key);
				const std::string val = it == v.end() ? std::string() : it->second;
				if (val.empty()) empty_value = true;
				e += val;
				i = close;
			} else {
				e += w[i];
			}
		}
		if (empty_value) continue;
		out += (out.empty() ? "" : " ") + std::string("\"") + e + "\"";
	}
	return out;
}

bool FileExists(const std::string& p) {
	if (FILE* f = std::fopen(p.c_str(), "rb")) {
		std::fclose(f);
		return true;
	}
	return false;
}

void SetEnv(const char* k, const std::string& v) {
#if defined(_WIN32)
	_putenv_s(k, v.c_str());
#else
	setenv(k, v.c_str(), 1);
#endif
}

}  // namespace

int main(int argc, char** argv) {
	std::map<std::string, std::string> v;
	v["ref"] = "tests/paged-kv/reference";
	std::string bin = ".", step_s = "G6";
	bool list = false, debug_layer = true;
	std::vector<std::string> sel;
	const std::pair<const char*, const char*> kFlags[] = {
	    {"--pkv-fixtures=", "fx"},  {"--pkv-reference=", "ref"}, {"--gpu-ref=", "gpuref"},
	    {"--model05=", "m05"},      {"--model15=", "m15"},       {"--qwen3=", "qwen3"},
	    {"--r15=", "r15"},          {"--g5=", "g5"},             {"--adapter=", "adapter"},
	    {"--adapter-other-base=", "adapter_other"},              {"--out98=", "out98"},
	    {"--old-bin=", "oldbin"}};
	int bad = 0;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		bool taken = false;
		for (const auto& f : kFlags) {
			const size_t n = std::strlen(f.first);
			if (a.compare(0, n, f.first) == 0) {
				v[f.second] = a.substr(n);
				taken = true;
				break;
			}
		}
		if (taken) continue;
		if (a.rfind("--bin=", 0) == 0) bin = a.substr(6);
		else if (a.rfind("--step=", 0) == 0) step_s = a.substr(7);
		else if (a == "--list") list = true;
		else if (a == "--no-debug-layer") debug_layer = false;
		else if (a.rfind("--", 0) == 0) {
			std::fprintf(stderr, "unknown flag %s\n", a.c_str());
			++bad;
		} else sel.push_back(a);
	}
	const int step = (step_s.size() == 2 && (step_s[0] == 'G' || step_s[0] == 'g')) ? step_s[1] - '0' : 0;
	if (step < 2 || step > 6) {
		std::fprintf(stderr, "--step must be G2, G3, G4, G5 or G6\n");
		return 2;
	}
	// The in-process runner of the step: pkvg_g2 .. pkvg_g5. G6 brings process cells only, so at G6 the runner is
	// pkvg_g5's (every in-process cell at or before G6).
	const int inproc = step < 5 ? step : 5;
	if (v.count("fx") == 0) {
		if (const char* d = std::getenv("SUPERSLM_PAGED_KV_FIXTURE_DIR")) v["fx"] = d;
	}
	if (!v["out98"].empty()) SetEnv("SUPERSLM_PKV_GPU_98_OUT", v["out98"]);
#if defined(_WIN32)
	const char sep = '\\';
#else
	const char sep = '/';
#endif

	int unmatched = 0;
	for (const std::string& a : sel) {
		bool any = false;
		for (const Job& j : kJobs) any = any || Matches(j.id, a);
		if (!any) {
			std::fprintf(stderr, "argument '%s' matches no job\n", a.c_str());
			++unmatched;
		}
	}
	int ran = 0, failed = 0, ran_on = 0, ran_off = 0;
	for (const Job& j : kJobs) {
		if (j.bracket > step || (j.only_at && j.only_at != inproc)) continue;
		bool selected = sel.empty();
		for (size_t i = 0; i < sel.size() && !selected; ++i) selected = Matches(j.id, sel[i]);
		if (!selected) continue;
		const bool old = std::strncmp(j.exe, "old:", 4) == 0;
		const std::string dir = old ? v["oldbin"] : bin;
		const std::string exe = dir + sep + (old ? j.exe + 4 : j.exe);
		const std::string cmd = "\"" + exe + "\" " + Expand(j.args, v);
		// The layer this job runs under: ON unless it is a hook job (or --no-debug-layer turned every job off).
		const bool layer_on = debug_layer && j.debug == kOn;
		const char* mode = layer_on ? "debug layer ON"
		                   : j.debug == kHook ? "debug layer OFF (te425 hook job)"
		                                      : "debug layer OFF (--no-debug-layer)";
		if (list) {
			std::printf("[G%d] %s [%s]: %s\n", j.bracket, j.id, mode, cmd.c_str());
			continue;
		}
		++ran;
		std::string missing;
		for (const std::string& k : Words(j.need))
			if (v[k].empty()) missing += (missing.empty() ? "" : ", ") + k;
		if (!missing.empty()) {
			std::printf("[G%d] %s RED (not given: %s; a skip fails the run)\n", j.bracket, j.id, missing.c_str());
			++failed;
			continue;
		}
		if (!FileExists(exe)) {
			std::printf("[G%d] %s RED (%s not built: red by link or compile at this step, see the build log)\n", j.bracket,
			            j.id, exe.c_str());
			++failed;
			continue;
		}
#if defined(_WIN32)
		const std::string full = "\"" + cmd + "\"";  // cmd /c strips one pair of outer quotes
#else
		const std::string full = cmd;
#endif
		// Set per child: the child inherits this process's environment at std::system.
		SetEnv("SSLM_GPU_ENABLE_DEBUG_LAYER", layer_on ? "1" : "0");
		++(layer_on ? ran_on : ran_off);
		std::printf("[G%d] %s ... [%s]\n  %s\n", j.bracket, j.id, mode, cmd.c_str());
		std::fflush(stdout);
		const int rc = std::system(full.c_str());
		if (rc != 0) ++failed;
		std::printf("[G%d] %s %s (exit %d) [%s]\n", j.bracket, j.id, rc == 0 ? "green" : "RED", rc, mode);
		std::fflush(stdout);
	}
	if (list) return bad ? 2 : 0;
	// ran_on + ran_off counts the jobs that started; a job RED before it ran (not given, not built) is in neither.
	std::printf("pkvg_suite at G%d: %d jobs, %d red; %d ran with the debug layer ON, %d OFF%s\n", step, ran, failed,
	            ran_on, ran_off,
	            debug_layer ? " (OFF = the te425 hook jobs, by ruling)" : " (--no-debug-layer: not an acceptance run)");
	if (unmatched) std::printf("%d argument(s) matched no job (named on stderr)\n", unmatched);
	return ran > 0 && unmatched == 0 && failed == 0 && bad == 0 ? 0 : 1;
}
