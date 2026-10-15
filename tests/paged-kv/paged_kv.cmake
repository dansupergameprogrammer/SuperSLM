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

# Paged-KV plan rev 16.24: the two test-side switches the R9 builder flips, each in the commit that lands the
# behaviour it follows. Both default to 2ebe5e3's behaviour, so every C2 to C6 and R0 suite stays green there.
#   SUPERSLM_PKV_T1_ROTATED_LAYOUT (D-SLM8449): RETIRED (T-3202 F-7). The engine rotates unconditionally since T1,
#     and tests/paged-kv/pkv_common.h's in-page formula is rotated unconditionally. Its text was: every test-side in-page
#     address (pkv_common.h, PageRunOffset) becomes §3.1's rotated slot (r + j) mod runs, and the C2 engine view
#     carries the run count.
#   SUPERSLM_PKV_T2_FALLBACK_RETIRED (D-SLM8448): RETIRED with the fallback (T-3189 §13). Its eight C4, C5 and R0
#     pkv_odd fallback assertions (plan §26.5's table) are deleted, and Fixture::B() reads the rule's 256 for pkv_odd.
# Neither option exists any more; a build directory whose cache still holds either value builds the same tests (CMake
# ignores a cache entry nothing reads).

function(superslm_pkv_target name)
	cmake_parse_arguments(PKV "" "LIBRARY" "SOURCES" ${ARGN})
	if(NOT PKV_LIBRARY)
		set(PKV_LIBRARY superslm_test_injection)
	endif()
	add_executable(${name} EXCLUDE_FROM_ALL ${PKV_DIR}/pkv_main.cpp ${PKV_SOURCES})
	target_link_libraries(${name} PRIVATE ${PKV_LIBRARY})
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

# Red-first timer and CPUID decision-path constructions. Neither target is a
# production timing leg or eligible to publish a C6 reading.
superslm_pkv_target(superslm_pkv_74_no_tsc SOURCES ${PKV_DIR}/c6_dim7_reset_adopt_timing.cpp)
target_compile_definitions(superslm_pkv_74_no_tsc PRIVATE PKV_74_FORCE_NO_TSC=1)
superslm_pkv_target(superslm_pkv_74_no_subtraction SOURCES ${PKV_DIR}/c6_dim7_reset_adopt_timing.cpp)
target_compile_definitions(superslm_pkv_74_no_subtraction PRIVATE PKV_74_NO_SUBTRACTION_MUTANT=1)

# C6's measured 7.9/7.9b arms and piece-wise SSB5 prefill use the production
# library. This target has no test-injection compile definitions or link input.
add_executable(superslm_pkv_pair EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/tools/paged_kv_pair.cpp)
target_link_libraries(superslm_pkv_pair PRIVATE superslm)
target_include_directories(superslm_pkv_pair PRIVATE ${PKV_DIR} include src tests)
target_compile_definitions(superslm_pkv_pair PRIVATE PKV_REFERENCE_DIR="${PKV_DIR}/reference")
if(MSVC)
	target_compile_options(superslm_pkv_pair PRIVATE /W4 /fp:precise)
else()
	target_compile_options(superslm_pkv_pair PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

# Cell 7.9's must-reject, 7.9-MR-OPT (plan rev 16.21, D-SLM8423; slot R79MR): a mutant build of the production sources
# in which ComputePageGeometry yields B = 16 and the slot ignores the logical page (SUPERSLM_MUTANT_PKV_MR_OPT, read by
# src/sslm_abi.cpp and include/superslm/forward_sites.h), and the production pair executable built from the same source
# against it. The pair is built with PKV_PAIR_MR_OPT, so it prints 7.9-MR-OPT's own identity line,
# "PKV_PAIR_MR_OPT_V1 defs=none" (D-SLM8466 (2)), and runs through the C6 driver as
# `tools/run_paged_kv_rows.py --construction 7.9-MR-OPT --page-positions 16`, which refuses any other identity (79-7).
# Box-owed; EXCLUDE_FROM_ALL.
add_library(superslm_mr_opt STATIC EXCLUDE_FROM_ALL ${SUPERSLM_CORE_SOURCES})
target_include_directories(superslm_mr_opt PUBLIC include)
# SUPERSLM_MUTANT_PKV_MR_OPT_TARGET acknowledges the mutant: the sources #error on the mutant definition without it
# (T-3202 F-4), so a mutant definition that leaks to any other compile is refused.
target_compile_definitions(superslm_mr_opt PRIVATE SUPERSLM_MUTANT_PKV_MR_OPT SUPERSLM_MUTANT_PKV_MR_OPT_TARGET)
add_executable(superslm_pkv_pair_mr_opt EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/tools/paged_kv_pair.cpp)
target_link_libraries(superslm_pkv_pair_mr_opt PRIVATE superslm_mr_opt)
target_include_directories(superslm_pkv_pair_mr_opt PRIVATE ${PKV_DIR} include src tests)
# PKV_PAIR_MR_OPT: this pair prints 7.9-MR-OPT's own identity line (D-SLM8466 (2), tools/paged_kv_pair.cpp BuildIdentity).
target_compile_definitions(superslm_pkv_pair_mr_opt PRIVATE PKV_REFERENCE_DIR="${PKV_DIR}/reference" PKV_PAIR_MR_OPT)
if(MSVC)
	target_compile_options(superslm_mr_opt PRIVATE /W4 /fp:precise)
	target_compile_options(superslm_pkv_pair_mr_opt PRIVATE /W4 /fp:precise)
else()
	target_compile_options(superslm_mr_opt PRIVATE -Wall -Wextra -ffp-contract=off)
	target_compile_options(superslm_pkv_pair_mr_opt PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

# Lifecycle cost has a frozen v1.11.0 baseline and a paged leg linked to the
# production library. Its result cannot carry a seam or injection definition.
superslm_pkv_target(superslm_pkv_lifecycle_prod LIBRARY superslm SOURCES ${PKV_DIR}/c6_lifecycle_timing.cpp)
target_compile_definitions(superslm_pkv_lifecycle_prod PRIVATE PKV_LIFECYCLE_PRODUCTION_LEG=1)

# Link-time source-set receipts are build artifacts. The box runners re-hash
# these files and the invoked binaries/scripts before accepting commissioning.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
function(pkv_source_manifest target library)
	set(args --repo ${CMAKE_CURRENT_SOURCE_DIR} --exe $<TARGET_FILE:${target}>
		--library $<TARGET_FILE:${library}>)
	foreach(source IN LISTS ARGN)
		list(APPEND args --source ${source})
	endforeach()
	if(target STREQUAL "superslm_pkv_pair")
		# plan §30.1 (D-SLM8476): the 7.9 driver is invoked beside the 7.9b runner, so the binding verifier re-hashes it
		list(APPEND args --invoked ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_paged_kv_pairs.py
			--invoked ${CMAKE_CURRENT_SOURCE_DIR}/tools/paged_kv_load.py
			--invoked ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_paged_kv_rows.py)
	endif()
	add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/write_paged_kv_source_manifest.py ${args}
		VERBATIM)
endfunction()
set(PKV_DECIDING_COMMON ${PKV_DIR}/pkv_main.cpp ${PKV_DIR}/pkv_common.h
	${PKV_DIR}/pkv_budget_c_helpers.h ${PKV_DIR}/pkv_engine_api.h)
pkv_source_manifest(superslm_pkv_c6 superslm_test_injection ${PKV_C6_SOURCES}
	${PKV_DECIDING_COMMON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_paged_kv_c6_box.ps1
	${CMAKE_CURRENT_SOURCE_DIR}/tools/paged_kv_c6_provenance.ps1)
pkv_source_manifest(superslm_pkv_lifecycle_prod superslm ${PKV_DIR}/c6_lifecycle_timing.cpp
	${PKV_DECIDING_COMMON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_paged_kv_c6_box.ps1
	${CMAKE_CURRENT_SOURCE_DIR}/tools/paged_kv_c6_provenance.ps1)
# The 7.9 driver (tools/run_paged_kv_rows.py) is a source too: the timing successor's instrument, whose hash the binding
# verifier reads from this manifest (plan §30.1, D-SLM8476).
pkv_source_manifest(superslm_pkv_pair superslm ${CMAKE_CURRENT_SOURCE_DIR}/tools/paged_kv_pair.cpp
	${PKV_DIR}/pkv_common.h ${PKV_DIR}/pkv_budget_c_helpers.h
	${CMAKE_CURRENT_SOURCE_DIR}/tools/run_paged_kv_pairs.py ${CMAKE_CURRENT_SOURCE_DIR}/tools/paged_kv_load.py
	${CMAKE_CURRENT_SOURCE_DIR}/tools/run_paged_kv_rows.py)
get_target_property(PKV_PROD_DEFS superslm COMPILE_DEFINITIONS)
if("${PKV_PROD_DEFS}" MATCHES "SUPERSLM_.*(SEAM|INJECTION|INSTRUMENT|BENCH|MUTANT)")
	message(FATAL_ERROR "production superslm library carries a seam, injection or mutant definition; lifecycle grading refused")
endif()

# Cell 7.9's must-accepts MA-1 and MA-2 (slot R79A; plan rev 16.29 §29.3, §29.4, §30.5; D-SLM8471, D-SLM8472, D-SLM8479):
# the commissioning seat's construction, tools/paged_kv_pair_ma.cpp, which prints "PKV_PAIR_MA1_V1 defs=none" or
# "PKV_PAIR_MA2_V1 defs=none" and runs through the C6 driver as `--construction MA-1` or `MA-2`. Linked against the
# production library superslm and nothing else from the engine; one object on its link step, compiled from that source
# (tools/check_paged_kv_pair_link.py checks 4 and 8). Box-owed; EXCLUDE_FROM_ALL.
add_executable(superslm_pkv_pair_ma EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/tools/paged_kv_pair_ma.cpp)
target_link_libraries(superslm_pkv_pair_ma PRIVATE superslm)
target_include_directories(superslm_pkv_pair_ma PRIVATE ${PKV_DIR} include src tests)
target_compile_definitions(superslm_pkv_pair_ma PRIVATE PKV_REFERENCE_DIR="${PKV_DIR}/reference")
if(MSVC)
	target_compile_options(superslm_pkv_pair_ma PRIVATE /W4 /fp:precise)
else()
	target_compile_options(superslm_pkv_pair_ma PRIVATE -Wall -Wextra -ffp-contract=off)
endif()
pkv_source_manifest(superslm_pkv_pair_ma superslm ${CMAKE_CURRENT_SOURCE_DIR}/tools/paged_kv_pair_ma.cpp
	${PKV_DIR}/pkv_common.h ${PKV_DIR}/pkv_budget_c_helpers.h)

# Cells 4.7/C5 and 6.3/C5: the DotRow tiers. superslm_pkv_c5's sources linked against each forced-tier
# library of CMakeLists.txt in place of superslm_test_injection (x86-64 only, like those libraries;
# EXCLUDE_FROM_ALL both). Each forced library carries the bad-alloc and matvec seams, so the whole C5
# suite links; a tier leg runs the two tier cells by id, e.g.
#   superslm_pkv_c5_avx2_forced 4.7/C5 6.3/C5
# which runs exactly those two cells; an argument that matches no registered id fails the run
# (pkv_main.cpp), so a renamed cell cannot drop out of a tier leg unnoticed.
# The table-sentinel trap (7.13) is compiled only into superslm_test_injection, so 7.13 belongs to the
# default binary, not to these.
if(PKV_C5_SOURCES AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64|X64)$")
	foreach(tier sse2 avx2 avx512)
		superslm_pkv_target(superslm_pkv_c5_${tier}_forced LIBRARY superslm_${tier}_forced SOURCES ${PKV_C5_SOURCES})
	endforeach()
endif()

# Paged-KV plan rev 16.21 §23.8, step T0: the R9 red suite for steps T1 to T3 (the page-size rule and
# rotation, the partial last page, the contiguous pool option). Cells are tagged with their R9 step
# (T1, T2, T3) and their §7 id. Two executables:
#   superslm_pkv_r9      every bench cell: generated and CFG1-patched fixtures, no real artifact
#   superslm_pkv_r9_box  the real-artifact cells (SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR); authored and
#                        compiled at T0, run on the box only
# Both are EXCLUDE_FROM_ALL. The R9 verbs and the pin seam are declared in tests/paged-kv/pkv_r9.h.
# Until the builder defines them, pkv_r9_stubs.cpp defines each as an unimplemented stub (a refusal,
# nothing written), so the cells link and fail at their assertions. The builder turns each option
# OFF in the commit that defines the real symbol; a stub left ON beside a real definition fails the
# link with a duplicate symbol, so the two cannot coexist.
option(SUPERSLM_PKV_R9_STUB_CONTIGUOUS "T0 stubs for sslm_kv_pool_contiguous_size and sslm_kv_pool_create_contiguous" OFF)
option(SUPERSLM_PKV_R9_STUB_PIN "T0 stub for the test-only pin KvPagesTestPinPagePositions" OFF)
file(GLOB PKV_R9_SOURCES CONFIGURE_DEPENDS ${PKV_DIR}/r9_*.cpp)
file(GLOB PKV_R9_BOX_SOURCES CONFIGURE_DEPENDS ${PKV_DIR}/r9box_*.cpp)
foreach(r9 superslm_pkv_r9 superslm_pkv_r9_box)
	if(r9 STREQUAL "superslm_pkv_r9")
		superslm_pkv_target(${r9} SOURCES ${PKV_R9_SOURCES} ${PKV_DIR}/pkv_r9_stubs.cpp)
	else()
		superslm_pkv_target(${r9} SOURCES ${PKV_R9_BOX_SOURCES} ${PKV_DIR}/pkv_r9_stubs.cpp)
	endif()
	if(SUPERSLM_PKV_R9_STUB_CONTIGUOUS)
		target_compile_definitions(${r9} PRIVATE PKV_R9_STUB_CONTIGUOUS=1)
	endif()
	if(SUPERSLM_PKV_R9_STUB_PIN)
		target_compile_definitions(${r9} PRIVATE PKV_R9_STUB_PIN=1)
	endif()
endforeach()

# Cell 4.12's link half: a unit that references the pin, linked to the production `superslm`
# library, must FAIL to link (tests/paged-kv/r9_pin_reach.py builds this target and expects a link
# error naming the pin). It never carries the stub.
add_executable(superslm_pkv_r9_pin_link_probe EXCLUDE_FROM_ALL ${PKV_DIR}/pkv_r9_pin_link_probe.cpp)
target_link_libraries(superslm_pkv_r9_pin_link_probe PRIVATE superslm)
target_include_directories(superslm_pkv_r9_pin_link_probe PRIVATE ${PKV_DIR} include)

# Cell 4.13's forced-SIMD half: the derived-B crossing (tests/paged-kv/r9_t1_crossing.cpp, T1) and the cap-4,100
# member at the partial last page (tests/paged-kv/r9_t2_partial.cpp, 4.13/T2:cap4100; T-3196 F-13) linked against each
# forced-tier library, as the C5 tier legs are; run by id, e.g.
#   superslm_pkv_r9_avx2_forced 4.13/T1 4.6/T1 4.13/T2:cap4100
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64|X64)$")
	foreach(tier sse2 avx2 avx512)
		superslm_pkv_target(superslm_pkv_r9_${tier}_forced LIBRARY superslm_${tier}_forced
			SOURCES ${PKV_DIR}/r9_t1_crossing.cpp ${PKV_DIR}/r9_t2_partial.cpp ${PKV_DIR}/pkv_r9_stubs.cpp)
		if(SUPERSLM_PKV_R9_STUB_CONTIGUOUS)
			target_compile_definitions(superslm_pkv_r9_${tier}_forced PRIVATE PKV_R9_STUB_CONTIGUOUS=1)
		endif()
		if(SUPERSLM_PKV_R9_STUB_PIN)
			target_compile_definitions(superslm_pkv_r9_${tier}_forced PRIVATE PKV_R9_STUB_PIN=1)
		endif()
	endforeach()
endif()

# Paged-KV plan rev 16.23, D-SLM8439: the C4 and C5 suites' fixture loader pins the generated fixtures to
# B = 16 (pkv_common.h, PKV_PIN_GENERATED_FIXTURES). Until the builder defines the pin (T1), these executables
# link the T0 stub, so they stay linkable and green on 2ebe5e3; the builder turns SUPERSLM_PKV_R9_STUB_PIN OFF
# in the commit that defines it, which drops the stub here too.
foreach(t superslm_pkv_c4 superslm_pkv_c5 superslm_pkv_c5_sse2_forced superslm_pkv_c5_avx2_forced superslm_pkv_c5_avx512_forced)
	if(TARGET ${t})
		target_compile_definitions(${t} PRIVATE PKV_PIN_GENERATED_FIXTURES=1)
		if(SUPERSLM_PKV_R9_STUB_PIN)
			target_sources(${t} PRIVATE ${PKV_DIR}/pkv_r9_stubs.cpp)
			target_compile_definitions(${t} PRIVATE PKV_R9_STUB_PIN=1)
		endif()
	endif()
endforeach()

# Paged-KV plan rev 16.24, D-SLM8447: the C6 cells read B from §3.1's rule (PKV_FIXTURE_B_FROM_RULE), never the C4
# and C5 pinned 16, so their cloud twins run the box cells' code at the box's B (red on 2ebe5e3, green from T1). 7.4
# alone pins (64 on the box, 16 on the twin, D-SLM8298) on its own mapping, so the 7.4 executables link the pin stub
# until T1, as C4 and C5 do. The lifecycle production leg links `superslm`, which has no pin, and never pins.
foreach(t superslm_pkv_c6 superslm_pkv_74_no_tsc superslm_pkv_74_no_subtraction superslm_pkv_lifecycle_prod)
	if(TARGET ${t})
		target_compile_definitions(${t} PRIVATE PKV_FIXTURE_B_FROM_RULE=1)
	endif()
endforeach()
foreach(t superslm_pkv_c6 superslm_pkv_74_no_tsc superslm_pkv_74_no_subtraction)
	if(TARGET ${t} AND SUPERSLM_PKV_R9_STUB_PIN)
		target_sources(${t} PRIVATE ${PKV_DIR}/pkv_r9_stubs.cpp)
		target_compile_definitions(${t} PRIVATE PKV_R9_STUB_PIN=1)
	endif()
endforeach()
