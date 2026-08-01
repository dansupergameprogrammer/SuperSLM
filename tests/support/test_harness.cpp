// Definitions for the extern counters test_harness.h declares. Moved verbatim
// out of tests/test_main.cpp (T-1574 test suite split, Stage 0): same two
// counters, same initial values, now with external linkage so every
// tests/areas/*.cpp translation unit shares one pair.
#include "test_harness.h"

int GChecks = 0;
int GFailures = 0;
