// Death-test / crash-probe machinery (T-1574 test suite split, Stage 0).
// Moved verbatim out of tests/test_main.cpp: GSelfPath, the began-marker
// helper, the child-process env-var sentinel, the process-id helper, the
// CrashProbeOutcome verdict type, and RunsCrashProbeAndCrashes -- which
// spawns the current binary as a child process (via GSelfPath and
// --crash-probe=<name>) and reports whether the named probe genuinely ran
// and, if so, whether it crashed.
//
// New in this migration: a second self-registering registry,
// RegisterCrashProbe(name, fn)/RunCrashProbe(name), so a probe is registered
// where it is defined (today, still tests/test_main.cpp; from the stage that
// extracts each probe's owning area onward, that area's own .cpp) instead of
// being one more branch in a single shared if/else chain.
#pragma once

#include <string>

namespace superslm_test_registry {

// argv[0], captured in main() for the death-test probe machinery below.
extern std::string GSelfPath;

// Printed by a crash-probe function to the child's stdout, immediately
// before the contract-violating call, so the parent can prove the named
// probe was actually dispatched and reached the call -- not merely that the
// child exited some way.
std::string CrashProbeBeganMarker(const std::string& probe_name);

// Environment variable set by RunsCrashProbeAndCrashes before spawning the
// child, and inherited by it, so main() can recognize "this process is a
// crash-probe child" independently of argv parsing.
extern const char* const kCrashProbeChildEnvVar;

bool EnvVarIsSet(const char* name);

// The single verdict RunsCrashProbeAndCrashes returns. kDidNotRun is a third
// state a caller must handle explicitly -- there is no bool-shaped shortcut
// back to "crashed" or "did not crash" for a probe that never ran.
enum class CrashProbeOutcome {
	kDidNotRun,      // the began-marker for the requested probe never appeared in the
	                 // child's captured output -- the contract-violating call was never
	                 // dispatched.
	kRanNoCrash,     // the marker is present (the probe genuinely ran) and the child
	                 // completed without abnormal termination.
	kRanAndCrashed,  // the marker is present (the probe genuinely ran) and the child
	                 // terminated abnormally.
};

const char* CrashProbeOutcomeName(CrashProbeOutcome outcome);

// Runs the named crash probe in a child process (GSelfPath re-invoked with
// --crash-probe=<probe_name>) and returns a single verdict that already
// accounts for whether the probe genuinely ran -- enforced here via a
// CHECK_MSG on the began-marker, not repeated at each call site.
CrashProbeOutcome RunsCrashProbeAndCrashes(const char* probe_name, std::string* out_tail);

// A registered crash probe. Prints its own began-marker, makes its
// contract-violating (or fault-inducing) call, and returns 0 if the call ran
// to completion without crashing -- a suite defect the caller exists to
// catch, not the expected outcome for most probes. May return a nonzero
// non-crash sentinel (e.g. 3) if the probe's own setup could not reach the
// state it requires; that sentinel is never confused with "crashed" because
// RunsCrashProbeAndCrashes's began-marker check already requires the marker
// to have been printed before any outcome besides kDidNotRun is possible.
using CrashProbeFn = int (*)();

// Registers `fn` under `name`. Aborts (via the same duplicate-name diagnostic
// shape as RegisterTest) if `name` is already registered -- two probes must
// never silently shadow each other under the same --crash-probe= key.
void RegisterCrashProbe(const char* name, CrashProbeFn fn);

// Dispatched from main() when argv[1] == "--crash-probe=<name>": looks up
// `name` in the registry and runs it. Returns 2, without printing any
// began-marker, if `name` is not registered -- this must read as "the probe
// did not run" to every caller, never as "the probe ran and did not crash".
int RunCrashProbe(const std::string& name);

}  // namespace superslm_test_registry
