#include "crash_probe.h"

#include "test_harness.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <process.h>  // _getpid
#else
#include <unistd.h>  // getpid
#endif

namespace superslm_test_registry {

std::string GSelfPath;  // argv[0], captured in main()

std::string CrashProbeBeganMarker(const std::string& probe_name) {
	return "CRASH_PROBE_BEGAN:" + probe_name;
}

const char* const kCrashProbeChildEnvVar = "SUPERSLM_CRASH_PROBE_CHILD";

// std::getenv is the portable read; MSVC's /W4 flags it as deprecated in favor of
// _dupenv_s. This process only checks presence, never reads a value into a fixed
// buffer, so the deprecation does not apply -- silenced locally rather than
// project-wide.
bool EnvVarIsSet(const char* name) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
	return std::getenv(name) != nullptr;
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
}

namespace {

// Portable current-process-id read, used only to keep each crash-probe child's capture
// file distinct from every other configuration's: two configurations of this binary
// (e.g. a debug and an NDEBUG build) run concurrently on one machine and, before this,
// wrote and read the same fixed path in the shared system temp directory -- a race that
// turned a lost capture file from a degraded diagnostic message into an outright cell
// failure once the began-marker check started depending on that file's contents.
long CurrentProcessId() {
#ifdef _WIN32
	return static_cast<long>(_getpid());
#else
	return static_cast<long>(getpid());
#endif
}

}  // namespace

const char* CrashProbeOutcomeName(CrashProbeOutcome outcome) {
	switch (outcome) {
		case CrashProbeOutcome::kDidNotRun:
			return "did-not-run";
		case CrashProbeOutcome::kRanNoCrash:
			return "ran-no-crash";
		case CrashProbeOutcome::kRanAndCrashed:
			return "ran-and-crashed";
	}
	return "(unknown CrashProbeOutcome)";
}

// Runs the named crash probe in a child process and returns a single verdict that already
// accounts for whether the probe genuinely ran -- a caller cannot obtain kRanNoCrash or
// kRanAndCrashed without the began-marker check below having passed first, because the
// CHECK_MSG that enforces it, and the early return past it, both live here rather than at
// each call site.
CrashProbeOutcome RunsCrashProbeAndCrashes(const char* probe_name, std::string* out_tail) {
	std::filesystem::path out_path =
	    std::filesystem::temp_directory_path() /
	    (std::string("superslm_crash_probe_") + probe_name + "_" +
	     std::to_string(CurrentProcessId()) + ".txt");
	std::error_code rm_ec;
	std::filesystem::remove(out_path, rm_ec);

	std::string cmd = "\"" + GSelfPath + "\" --crash-probe=" + probe_name +
	                   " > \"" + out_path.string() + "\" 2>&1";
#ifdef _WIN32
	// system() on Windows invokes `cmd.exe /c <cmd>`; when <cmd> itself begins with a
	// quoted executable path, cmd.exe's first/last-quote-stripping parser misreads the
	// nested quotes (a well-known cmd.exe quirk) unless the WHOLE string is wrapped in
	// one more outer quote pair -- that outer pair is what cmd strips, leaving the
	// interior correctly quoted. This wrap is a cmd.exe-only workaround: std::system on
	// POSIX invokes `/bin/sh -c <cmd>`, which has no equivalent quote-stripping step, so
	// the same outer wrap there collapses the whole command into one (nonexistent)
	// command word -- verified by execution: sh exits 127 and the child never runs, which
	// a naive `rc != 0` read misreports as "crashed" (Claude/Brunel/
	// superslm-s2.5-finding-for-curie-crash-probe-2026-07-20.md).
	std::string wrapped_cmd = "\"" + cmd + "\"";
	_putenv_s(kCrashProbeChildEnvVar, "1");
#else
	const std::string& wrapped_cmd = cmd;
	setenv(kCrashProbeChildEnvVar, "1", /*overwrite=*/1);
#endif
	int rc = std::system(wrapped_cmd.c_str());
#ifdef _WIN32
	// _putenv_s with an empty value removes the variable (documented behavior). Set only
	// for the duration of spawning this one child; a variable left set in the parent's own
	// environment would be inherited by any later child this process spawns, which would
	// misread as "I am a crash-probe child" and refuse to run (Claude/Poirot/
	// 7511117-s2.5-golden-crash-probe-reverify-2026-07-20.md, finding 4).
	_putenv_s(kCrashProbeChildEnvVar, "");
#else
	unsetenv(kCrashProbeChildEnvVar);
#endif

	std::string content;
	{
		std::ifstream f(out_path, std::ios::binary);
		content.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
	}
	if (out_tail) *out_tail = content;
	std::filesystem::remove(out_path, rm_ec);

	// The began-marker check that used to live in the caller (Claude/Poirot/
	// 7511117-s2.5-golden-crash-probe-reverify-2026-07-20.md, finding 3): the contract-
	// violating call must have been genuinely dispatched before "crashed" or "did not
	// crash" means anything. Folded in here, this CHECK_MSG fires for every caller of this
	// helper, present and future, not only for the one call site that remembers to repeat
	// it.
	bool began = content.find(CrashProbeBeganMarker(probe_name)) != std::string::npos;
	CHECK_MSG(began,
	          "the child's captured output never contains the began-marker for probe "
	          "'%s' -- the contract-violating call was never dispatched (probe name "
	          "mismatch, broken dispatch, or the child exited before reaching it), so no "
	          "crashed/did-not-crash outcome can be trusted for it -- child output was: %s",
	          probe_name, content.c_str());
	if (!began) return CrashProbeOutcome::kDidNotRun;

	// Reachable only once the marker has proven the probe genuinely ran. RunCrashProbe's
	// unrecognized-name sentinel (return code 2) cannot occur on this path: it returns 2
	// only on the branch that never prints the marker, so `began` would have been false
	// and the function would already have returned above. rc's only remaining meanings are
	// therefore "completed normally" (0) or "terminated abnormally" (nonzero) -- the
	// residual the reviewer named (RunCrashProbe's code 2 satisfying `rc != 0` for a probe
	// that never ran) closes by construction, not by exclusion, because the sentinel and a
	// present marker cannot occur together.
	return (rc != 0) ? CrashProbeOutcome::kRanAndCrashed : CrashProbeOutcome::kRanNoCrash;
}

namespace {

std::unordered_map<std::string, CrashProbeFn>& CrashProbeRegistry() {
	static std::unordered_map<std::string, CrashProbeFn> registry;
	return registry;
}

}  // namespace

void RegisterCrashProbe(const char* name, CrashProbeFn fn) {
	auto& reg = CrashProbeRegistry();
	auto it = reg.find(name);
	if (it != reg.end()) {
		std::fprintf(stderr, "RegisterCrashProbe: duplicate probe name '%s'\n", name);
		std::fflush(stderr);
		std::abort();
	}
	reg.emplace(name, fn);
}

int RunCrashProbe(const std::string& name) {
	auto& reg = CrashProbeRegistry();
	auto it = reg.find(name);
	if (it == reg.end()) {
		std::printf("PROBE DID NOT CRASH (unknown probe name: %s)\n", name.c_str());
		return 2;
	}
	return it->second();
}

}  // namespace superslm_test_registry
