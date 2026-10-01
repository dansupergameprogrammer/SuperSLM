// Paged-KV plan (rev 16.1) step C1: running one case of a cell in a child process, for the cells
// whose correct outcome or whose mutant ends the process (7.13's trap is std::abort; 11.3's mutant
// is an assert; 11.9's guard-removed mutants write through a null or wrapped address).
//
// POSIX: the case runs in a fork()ed child, which _exit()s 0 when every check of the case held
// and 1 otherwise; the parent grades the child's end (a normal exit with its code, or the signal).
// A case's own PKV_CHECK failures print in the child as usual.
//
// The box's MSVC leg: there is no fork. The parent re-runs its own executable on the same cell id
// with PKV_CHILD_CASE=<case> in the environment; the cell, seeing that variable name the case, runs
// only it and exits. std::abort ends an MSVC process with exit code 3, which ChildEnd reports as
// `aborted` (the child first turns off the CRT's fail-fast abort report, which would otherwise end it
// with 0xC0000409). The box leg's first run confirmed this. In a re-run child the cases ahead of
// its own report "the child did not start"; only the child's exit code is graded.

#ifndef SUPERSLM_TESTS_PKV_ENGINE_CHILD_H
#define SUPERSLM_TESTS_PKV_ENGINE_CHILD_H

#include "pkv_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace pkv_engine {

struct ChildEnd {
	bool started = false;
	bool exited = false;   // a normal exit, with `code`
	int code = -1;
	bool aborted = false;  // SIGABRT (POSIX) or exit code 3 (MSVC's std::abort)
	int signal = 0;        // POSIX: the terminating signal, 0 if none
	std::string Describe() const {
		if (!started) return "the child did not start";
		if (aborted) return "aborted";
		if (exited) return "exited " + std::to_string(code);
		return "killed by signal " + std::to_string(signal);
	}
};

// Runs `body` (true = every check of the case held) in a child process and reports how it ended.
inline ChildEnd RunInChild(const char* case_name, const std::function<bool()>& body) {
	ChildEnd end;
	const char* want = std::getenv("PKV_CHILD_CASE");
	if (want && *want) {  // the re-run child (MSVC leg): run only the named case, spawn nothing
		if (std::strcmp(want, case_name) != 0) return end;
#ifdef _MSC_VER
		// A Release CRT's std::abort otherwise ends with 0xC0000409 (fail-fast), not exit code 3.
		_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
		std::fflush(stdout);
		std::exit(body() ? 0 : 1);
	}
	std::fflush(stdout);
	std::fflush(stderr);
#ifdef _WIN32
	char self[MAX_PATH] = {0};
	if (GetModuleFileNameA(nullptr, self, MAX_PATH) == 0) return end;
	_putenv_s("PKV_CHILD_CASE", case_name);
	const std::string cmd = std::string("\"\"") + self + "\" \"" + pkv::State().cell + "\"\"";
	const int rc = std::system(cmd.c_str());
	_putenv_s("PKV_CHILD_CASE", "");
	end.started = true;
	end.exited = rc != 3;
	end.aborted = rc == 3;
	end.code = rc;
#else
	const pid_t pid = fork();
	if (pid < 0) return end;
	if (pid == 0) {
		const bool ok = body();
		std::fflush(stdout);
		std::fflush(stderr);
		_exit(ok ? 0 : 1);
	}
	end.started = true;
	int status = 0;
	if (waitpid(pid, &status, 0) != pid) {
		end.started = false;
		return end;
	}
	if (WIFEXITED(status)) {
		end.exited = true;
		end.code = WEXITSTATUS(status);
	} else if (WIFSIGNALED(status)) {
		end.signal = WTERMSIG(status);
		end.aborted = end.signal == SIGABRT;
	}
#endif
	return end;
}

// The child ended normally with every check held.
inline bool Passed(const ChildEnd& c) { return c.started && c.exited && c.code == 0; }

}  // namespace pkv_engine

#endif  // SUPERSLM_TESTS_PKV_ENGINE_CHILD_H
