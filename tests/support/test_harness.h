// Test harness primitives: the check/failure counters and the CHECK/CHECK_MSG
// assertion macros. Moved verbatim out of tests/test_main.cpp (T-1574 test
// suite split, Stage 0) -- no behavior change, same counters, same macro
// bodies. Every tests/areas/*.cpp translation unit includes this header.
#pragma once

extern int GChecks;
extern int GFailures;

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
			std::printf("FAIL %s:%d: %s — ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)
