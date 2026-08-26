// T-2296 (Curie) -- shared CHECK harness and process-isolation helpers for the red-first suite
// realizing Claude/Vitruvius/t2265-superslm-fp-free-open-design-2026-08-24.md's Coverage Model
// (design of record, PROPOSED, nothing built as of this suite's authoring).
//
// CHECK/CHECK_MSG/SKIP_MSG follow this repo's own established convention verbatim
// (tests/t2138-abi-red-suite/fixture_common.h, tests/t2130-g5-red-suite/fixture_common.h,
// tests/t2112-gpu-1p0-red-suite/fixture_common.h, tests/test_main.cpp).
//
// PROCESS-ISOLATED TRAP DETECTION: several of this design's own cells (dimension 6 Cell A,
// dimension 7's fifth cell-group) assert that a construction TRAPS (an SEH/masks-cleared
// exception exit) or ABORTS (SIGABRT/0xC0000409) under a specific input. Neither can be caught
// in-process without corrupting the state of every later cell in the same binary -- a trapped or
// aborted process is gone. Every such cell therefore re-execs this same binary as a CHILD process
// in one of the "modes" main() dispatches on below (the identical parent/child shape
// Claude/Loki/t2284-probe/threshold.cpp uses to sweep the float-representability boundary, and
// Claude/Vitruvius/t2265-fold21-probe/te19_isolate.cpp/tensites.cpp use for the same reason --
// attribution stated per StandardsDocument.md Sec4/Sec7, this file's own construction, not a
// copy of either probe's text) and inspects the CHILD's exit behaviour from the parent, which
// survives regardless of what the child does.
//
// TIMEOUT, not std::system(): the mutation-proof requirement dimension 7 Cell B carries names an
// infinite loop as a NAMED possible reversion shape ("silent return false, or an infinite loop
// for the pre-fold-1 shape") -- a plain std::system() call blocks forever on a hung child and
// would hang this whole suite's own CI job. RunChildMode below uses CreateProcessA +
// WaitForSingleObject with a bounded timeout, TerminateProcess-ing and reporting kTimedOut on
// expiry, so a reverted implementation that hangs FAILS this suite's own assertion (a hang is
// never mistaken for a pass) instead of wedging the runner.
#ifndef SSLM_T2296_FIXTURE_COMMON_H
#define SSLM_T2296_FIXTURE_COMMON_H

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <xmmintrin.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

static int GChecks = 0;
static int GFailures = 0;
static int GSkips = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		} \
	} while (0)

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

#define SKIP_MSG(...) \
	do { \
		++GSkips; \
		std::printf("SKIP %s:%d -- ", __FILE__, __LINE__); \
		std::printf(__VA_ARGS__); \
		std::printf("\n"); \
	} while (0)

// Same three-prefix search tests/sslm_tokenizer_fixtures.h's own ResolveFixturePath uses,
// reproduced here (not included from that header) so this directory's own build does not pull
// in that file's unrelated golden-pack machinery.
inline std::string ResolveFixturePath(const std::string& filename) {
	static const char* kPrefixes[] = {"tests/fixtures/", "../tests/fixtures/",
	                                   "../../tests/fixtures/"};
	for (const char* prefix : kPrefixes) {
		std::string candidate = std::string(prefix) + filename;
		std::error_code ec;
		if (std::filesystem::exists(candidate, ec)) return candidate;
	}
	return {};
}

// Clears all six IEEE-754 exception masks in MXCSR (IM/DM/ZM/OM/UM/PM), the identical
// construction TE-19/TE-32/T-2284/fold round 21's own probes use (design Sec1, Sec4.1) --
// leaves rounding mode and denormal-handling bits untouched. Call this ONLY inside a child
// process (RunChildMode below): a masked floating-point operation on this path is expected to
// raise SIGFPE-class hardware exception once masks are cleared, which is exactly the trap this
// design's own Sec4.2 cell measures.
inline void ClearAllMxcsrExceptionMasks() {
	_mm_setcsr(_mm_getcsr() & ~0x1F80u);
}

enum class ChildResult { kExitedZero, kExitedNonzero, kCrashed, kTimedOut, kSpawnFailed };

// Re-execs this same binary (argv[0], read by main() and stashed in g_self_path below) with
// ["--child-mode=" + mode] appended, and reports how it ended -- WITHOUT letting a trap or an
// abort() in the child touch this (the parent) process's own state. kTimedOut fires the
// mutation-proof's own "does not hang" requirement (see this file's header comment); every
// other outcome maps to a real child process exit.
inline std::string g_self_path;

// Resolves g_self_path from the OS (GetModuleFileNameA), never from argv[0] verbatim -- argv[0]
// can be a bare relative name ("dim6_determinism_red.exe") the CHILD's own CreateProcessA call
// (run from the same working directory, so this is not merely a theoretical gap) would fail to
// resolve if the working directory changes between spawn and exec.
inline void SetSelfPathFromModule() {
	char buf[MAX_PATH] = {};
	DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
	if (n > 0 && n < MAX_PATH) g_self_path.assign(buf, n);
}

// Spawns the child and returns its RAW exit code via *out_code (undefined if the return value
// is kSpawnFailed or kTimedOut). This is the single source of truth every classification below
// is derived from -- a cell that needs the actual exit-code VALUE (fp_free_sticky_readback.cpp's
// own sticky-bitmask-in-the-exit-code convention) calls this directly instead of spawning the
// child a second time to recover what the classified-only overload below already discarded.
inline ChildResult RunChildModeRaw(const std::string& mode, DWORD* out_code,
                                    DWORD timeout_ms = 15000) {
	if (g_self_path.empty()) return ChildResult::kSpawnFailed;
	std::string cmdline = "\"" + g_self_path + "\" --child-mode=" + mode;

	STARTUPINFOA si{};
	si.cb = sizeof(si);
	PROCESS_INFORMATION pi{};
	std::vector<char> mutable_cmdline(cmdline.begin(), cmdline.end());
	mutable_cmdline.push_back('\0');

	BOOL ok = CreateProcessA(nullptr, mutable_cmdline.data(), nullptr, nullptr, FALSE,
	                          CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
	if (!ok) return ChildResult::kSpawnFailed;

	DWORD wait = WaitForSingleObject(pi.hProcess, timeout_ms);
	if (wait == WAIT_TIMEOUT) {
		TerminateProcess(pi.hProcess, 0xDEAD);
		CloseHandle(pi.hThread);
		CloseHandle(pi.hProcess);
		return ChildResult::kTimedOut;
	}
	DWORD code = 0;
	GetExitCodeProcess(pi.hProcess, &code);
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);
	if (out_code) *out_code = code;
	if (code == 0) return ChildResult::kExitedZero;
	// SEH/hardware-trap exits (0xC000008F STATUS_FLOAT_MULTIPLE_TRAPS-class,
	// 0xC0000409 STATUS_STACK_BUFFER_OVERRUN/fastfail from abort()) and any other
	// nonzero-exception-coded exit are both "the child did not return normally" --
	// distinguished in per-cell assertions below by which the cell expects.
	if (code >= 0xC0000000u) return ChildResult::kCrashed;
	return ChildResult::kExitedNonzero;
}

inline ChildResult RunChildMode(const std::string& mode, DWORD timeout_ms = 15000) {
	DWORD code = 0;
	return RunChildModeRaw(mode, &code, timeout_ms);
}

inline const char* ChildResultName(ChildResult r) {
	switch (r) {
		case ChildResult::kExitedZero: return "exited 0";
		case ChildResult::kExitedNonzero: return "exited nonzero";
		case ChildResult::kCrashed: return "crashed (SEH/abort-class exit code)";
		case ChildResult::kTimedOut: return "TIMED OUT (killed)";
		case ChildResult::kSpawnFailed: return "failed to spawn";
	}
	return "?";
}

// Parses --child-mode=X off argv into *out_mode; returns true if this process is a child
// dispatch (main() should run the mode and return, never falling into the parent's own cells).
inline bool ParseChildMode(int argc, char** argv, std::string* out_mode) {
	for (int i = 1; i < argc; ++i) {
		const char* prefix = "--child-mode=";
		if (std::strncmp(argv[i], prefix, std::strlen(prefix)) == 0) {
			*out_mode = argv[i] + std::strlen(prefix);
			return true;
		}
	}
	return false;
}

#endif  // SSLM_T2296_FIXTURE_COMMON_H
