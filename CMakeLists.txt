cmake_minimum_required(VERSION 3.16)
project(superslm CXX)

# C++20: the S-LoRA delta path is specified at -std=c++20 (SuperSLM_Plan.md §19),
# and the core carries wide-integer (int64) composition arithmetic throughout. One
# standard across the project avoids a later per-target split.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The core library. Layer 1 is independently embeddable — no Unreal, no third-party
# runtime dependency in its graph (DecisionLog D-SLM13). Standard library only.
set(SUPERSLM_CORE_SOURCES
	src/artifact.cpp
	src/sha256.cpp
	src/tokenizer.cpp
	src/model.cpp
	src/intmath.cpp
	src/silu_lut.cpp
	src/matmul.cpp
	src/proof_manifest.cpp
)

add_library(superslm STATIC ${SUPERSLM_CORE_SOURCES})
target_include_directories(superslm PUBLIC include)

# Determinism: forbid FP contraction. Implicit fused multiply-add would change
# rounding and break the scalar/SIMD bit-equality the whole project is built on
# (SuperSLM_Plan.md §6, §13). The kernel set (S2) will add explicit, mirrored FMA;
# until then the rule is simply "no implicit contraction anywhere."
if(MSVC)
	target_compile_options(superslm PRIVATE /W4 /fp:precise)
else()
	target_compile_options(superslm PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

# S-HARDEN-7's test-only fault-injection seam (tests/support/bad_alloc_
# injection.h) lets the test suite prove the "throws only std::bad_alloc"
# contract without any production consumer of this library ever compiling a
# reference to tests/support/*. A second compile of the SAME sources, built
# only with SUPERSLM_ENABLE_BAD_ALLOC_INJECTION defined, backs the test
# binary; every other consumer (sslm_verify, and every tool built outside
# CMake by build_inspect.bat) links the clean `superslm` above, which never
# defines the macro and so never even includes the seam header
# (src/bad_alloc_wrap.h #ifdef-gates it out entirely -- Layer 1 stays
# independently embeddable, D-SLM13).
add_library(superslm_test_injection STATIC ${SUPERSLM_CORE_SOURCES})
target_include_directories(superslm_test_injection PUBLIC include)
target_include_directories(superslm_test_injection PRIVATE tests)
target_compile_definitions(superslm_test_injection PRIVATE SUPERSLM_ENABLE_BAD_ALLOC_INJECTION)
if(MSVC)
	target_compile_options(superslm_test_injection PRIVATE /W4 /fp:precise)
else()
	target_compile_options(superslm_test_injection PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

add_executable(superslm_tests tests/test_main.cpp)
target_link_libraries(superslm_tests PRIVATE superslm_test_injection)
if(MSVC)
	target_compile_options(superslm_tests PRIVATE /W4 /fp:precise)
else()
	target_compile_options(superslm_tests PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

enable_testing()
add_test(NAME superslm_tests COMMAND superslm_tests)

# The §13 item 7 independent converter verifier (S-HARDEN-3, F13): loads a
# `.sslm` through the same SslmModel::Load every other consumer uses, then
# writes a proof manifest (include/superslm/proof_manifest.h) from what the
# loader actually accepted -- never from the Python writer's own arrays.
# convert_model.py invokes this binary after every write; §17.3 cell 2's
# integration test invokes it directly as the independent oracle.
add_executable(sslm_verify tools/sslm_verify.cpp)
target_link_libraries(sslm_verify PRIVATE superslm)
if(MSVC)
	target_compile_options(sslm_verify PRIVATE /W4 /fp:precise)
else()
	target_compile_options(sslm_verify PRIVATE -Wall -Wextra -ffp-contract=off)
endif()
