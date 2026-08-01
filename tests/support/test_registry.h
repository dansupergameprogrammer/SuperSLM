// Self-registering test registry (T-1574 test suite split, Stage 0). Every
// SSLM_TEST-wrapped test function registers itself, by name and by an
// explicit integer order key, via a namespace-scope object whose constructor
// runs during static initialization -- before main(). RegisterTest aborts on
// a duplicate name or a duplicate order key; RunAllRegisteredTests asserts
// the registered order keys are a dense permutation of [0, N) before running
// anything, then runs every registered test in ascending order-key order.
//
// The registry itself is a function-local static container
// (construct-on-first-use), read/written only during the pre-main()
// registration window and the single subsequent RunAllRegisteredTests()
// pass -- see Claude/Plans/SuperSLM_TestSuiteSplit_Plan.md §6 dimension 1.
#pragma once

#include <cstddef>

namespace superslm_test_registry {

using TestFn = void (*)();

// Registers `fn` under `name` at position `order`. Aborts the process (via
// std::abort, after printing a diagnostic naming both colliding entries) if
// `name` has already been registered, or if `order` has already been claimed
// by a different name.
void RegisterTest(const char* name, TestFn fn, int order);

// Returns the number of currently registered tests.
size_t RegisteredTestCount();

// Prints one "name,order" line per registered test, in ascending order-key
// order. Backs the --list-tests argv dispatch.
void ListRegisteredTests();

// Asserts (aborts, with a diagnostic) that the registered order keys are a
// dense permutation of [0, RegisteredTestCount()) -- no gaps, no duplicates
// past what RegisterTest already rejected -- then runs every registered test
// in ascending order-key order.
void RunAllRegisteredTests();

}  // namespace superslm_test_registry

// SSLM_TEST(Name, OrderKey) declares and opens the definition of a niladic
// void test function named `Name`, and registers it under that exact name at
// position `OrderKey` as a side effect of static initialization (a
// namespace-scope registrar object constructed before main() runs). Used as:
//
//   SSLM_TEST(TestFoo, 3) {
//       CHECK(...);
//   }
//
// which is the drop-in replacement for the pre-migration
// `static void TestFoo() { ... }` plus its own explicit call from main().
#define SSLM_TEST(Name, OrderKey)                                              \
	static void Name();                                                        \
	namespace {                                                                \
	struct Name##_Registrar {                                                  \
		Name##_Registrar() {                                                   \
			::superslm_test_registry::RegisterTest(#Name, &Name, (OrderKey));  \
		}                                                                       \
	} Name##_registrar_instance;                                               \
	}                                                                           \
	static void Name()
