# GPU paged-KV plan (rev 16), step G3: the D3D12-free core's host-only cells.
#
# superslm_g3_core_tests runs every cell of tests/gpu-g3-core (3.6h, the ledger's rules, the
# tokens, the registry cell with REGRACE, churn and the depth-0 grid, and the TAIL snapshot) against
# the library's own D3D12-free sources: src/gpu/submission_ledger.h, gpu_inflight_token.h,
# gpu_inflight_registry.{h,cpp} and gpu_call_timing_snapshot.h. It builds on every platform.
# The sanitizer arms and the mutants are run by tests/gpu-g3-core/run_g3_core.py (g++ 13.3, the
# plan's ThreadSanitizer route), which builds each arm itself.
#
# superslm_tests also runs the ledger's cells (3.6h and G3/ledger) through
# g3_superslm_tests_adapter.cpp, with src/gpu on its include path on every platform (outside the
# WIN32 block, where it was added before G3) and SUPERSLM_LEDGER_YIELD_HOOK defined on the whole
# target, so every translation unit of that binary sees one definition of the ledger class.
set(G3_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/gpu-g3-core)
add_executable(superslm_g3_core_tests
	${G3_CORE_DIR}/g3_main.cpp
	${G3_CORE_DIR}/g3_cells_ledger.cpp
	${G3_CORE_DIR}/g3_cells_token.cpp
	${G3_CORE_DIR}/g3_cells_registry.cpp
	${G3_CORE_DIR}/g3_cells_tail.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/gpu_inflight_registry.cpp)
target_include_directories(superslm_g3_core_tests PRIVATE ${G3_CORE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu)
target_compile_definitions(superslm_g3_core_tests PRIVATE SUPERSLM_LEDGER_YIELD_HOOK)
target_compile_features(superslm_g3_core_tests PRIVATE cxx_std_20)
if(MSVC)
	target_compile_options(superslm_g3_core_tests PRIVATE /W4)
else()
	target_compile_options(superslm_g3_core_tests PRIVATE -Wall -Wextra)
	find_package(Threads REQUIRED)
	target_link_libraries(superslm_g3_core_tests PRIVATE Threads::Threads)
endif()
add_test(NAME superslm_g3_core_tests COMMAND superslm_g3_core_tests)

# gpu_inflight_registry.cpp is a core library source (SUPERSLM_CORE_SOURCES) since the task
# trampoline at parallel_split.cpp's `run` call site needs it, so superslm_tests links it from there.
target_sources(superslm_tests PRIVATE
	${G3_CORE_DIR}/g3_superslm_tests_adapter.cpp
	${G3_CORE_DIR}/g3_cells_ledger.cpp)
target_include_directories(superslm_tests PRIVATE ${G3_CORE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu)
target_compile_definitions(superslm_tests PRIVATE SUPERSLM_LEDGER_YIELD_HOOK)
