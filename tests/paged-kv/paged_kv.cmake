# Paged-KV plan (rev 16.1): the red suite's per-step executables (step C1).
#
# One executable per owning step, so each links once its own step's builder has defined every
# symbol its cells call, and not before (red by link, tests/paged-kv/pkv_abi.h):
#   superslm_pkv_r0   the harness's own checks; links and runs on the base
#   superslm_pkv_c2   engine cells (the page view, per-page attention, GemmProbQ15AccumulateInto)
#   superslm_pkv_c3   page-module cells
#   superslm_pkv_c4   legacy-holder ABI cells and the C4 seams
#   superslm_pkv_c5   budget-mode ABI cells
#   superslm_pkv_c6   the box's achievement and timing cells (real artifacts)
# Every one is EXCLUDE_FROM_ALL until its step turns it green, so the default build stays green;
# tools/check_paged_kv_red.sh builds each and confirms its link fails on exactly the symbols its
# step owes. Cells read SUPERSLM_PAGED_KV_FIXTURE_DIR (tools/gen_paged_kv_fixture.py).

set(PKV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/paged-kv)

function(superslm_pkv_target name)
	cmake_parse_arguments(PKV "" "" "SOURCES" ${ARGN})
	add_executable(${name} EXCLUDE_FROM_ALL ${PKV_DIR}/pkv_main.cpp ${PKV_SOURCES})
	target_link_libraries(${name} PRIVATE superslm_test_injection)
	target_include_directories(${name} PRIVATE ${PKV_DIR} include src tests)
	target_compile_definitions(${name} PRIVATE PKV_REFERENCE_DIR="${PKV_DIR}/reference")
	if(MSVC)
		target_compile_options(${name} PRIVATE /W4 /fp:precise)
	else()
		target_compile_options(${name} PRIVATE -Wall -Wextra -ffp-contract=off)
		find_package(Threads REQUIRED)
		target_link_libraries(${name} PRIVATE Threads::Threads)
	endif()
endfunction()

superslm_pkv_target(superslm_pkv_r0 SOURCES ${PKV_DIR}/r0_harness_selftest.cpp)

file(GLOB PKV_C2_SOURCES CONFIGURE_DEPENDS ${PKV_DIR}/c2_*.cpp)
file(GLOB PKV_C3_SOURCES CONFIGURE_DEPENDS ${PKV_DIR}/c3_*.cpp)
file(GLOB PKV_C4_SOURCES CONFIGURE_DEPENDS ${PKV_DIR}/c4_*.cpp)
file(GLOB PKV_C5_SOURCES CONFIGURE_DEPENDS ${PKV_DIR}/c5_*.cpp)
file(GLOB PKV_C6_SOURCES CONFIGURE_DEPENDS ${PKV_DIR}/c6_*.cpp)
foreach(step c2 c3 c4 c5 c6)
	string(TOUPPER ${step} STEP)
	if(PKV_${STEP}_SOURCES)
		superslm_pkv_target(superslm_pkv_${step} SOURCES ${PKV_${STEP}_SOURCES})
	endif()
endforeach()
