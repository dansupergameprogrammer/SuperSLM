set(PC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/precache)
file(GLOB PC_SOURCES CONFIGURE_DEPENDS ${PC_DIR}/e*.cpp)
add_executable(superslm_precache_e1 EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/paged-kv/pkv_main.cpp
    ${PC_DIR}/precache_stubs.cpp ${PC_SOURCES})
# Invoked separately at E2. It deliberately fails to compile against the E1
# header and must compile only when the public macro and ordinal are published.
add_executable(superslm_precache_macro_vitality EXCLUDE_FROM_ALL
    ${PC_DIR}/prefix_macro_vitality.cpp)
add_executable(superslm_precache_reference_header EXCLUDE_FROM_ALL
    ${PC_DIR}/reference_header_identity.cpp)
add_executable(superslm_precache_compiler_job_identity EXCLUDE_FROM_ALL
    ${PC_DIR}/compiler_job_identity.cpp)
target_include_directories(superslm_precache_reference_header PRIVATE include src tests tests/paged-kv ${PC_DIR})
target_include_directories(superslm_precache_compiler_job_identity PRIVATE include src tests tests/paged-kv ${PC_DIR})
target_link_libraries(superslm_precache_compiler_job_identity PRIVATE superslm_test_injection)
target_compile_definitions(superslm_precache_reference_header PRIVATE
    PKV_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/paged-kv/reference")
target_compile_definitions(superslm_precache_compiler_job_identity PRIVATE
    PKV_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/paged-kv/reference"
    PC_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    PC_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    PC_PROJECT_VERSION="${PROJECT_VERSION}")
target_include_directories(superslm_precache_macro_vitality PRIVATE include)
target_link_libraries(superslm_precache_e1 PRIVATE superslm_test_injection)
target_include_directories(superslm_precache_e1 PRIVATE include src tests tests/paged-kv ${PC_DIR})
target_compile_definitions(superslm_precache_e1 PRIVATE
    PKV_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/paged-kv/reference")
# SuperSLM 1.13.0: the page-size rule, not CFG1, selects B. The cells were written at B = 16 (and the
# B = 4 twin), so the generated fixtures are pinned as the paged-KV C4/C5 suites pin theirs
# (pkv_common.h, D-SLM8439), and precache_common.h's PinFixtureB pins the synthetic shapes.
target_compile_definitions(superslm_precache_e1 PRIVATE PKV_PIN_GENERATED_FIXTURES=1)
# Plan revs 1.13-1.17 (§7 preamble): the unpinned E2 cells (4.7, 8.13, 9.8, 7.12's pool-kind leg) and E3's nested legs
# of 4.7 and 9.8 run at each model's rule B. They build in a target of their own, with PKV_FIXTURE_B_FROM_RULE and without PKV_PIN_GENERATED_FIXTURES, so
# Fixture::B() and every helper built on it read the rule. The define never goes on superslm_precache_e1, whole or per
# file (rev 1.15, R1): the sources sit in unpinned/, outside the e*.cpp glob above.
set(PC_UNPINNED_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tests/paged-kv/pkv_main.cpp
    ${PC_DIR}/precache_stubs.cpp ${PC_DIR}/unpinned/e2_unpinned.cpp ${PC_DIR}/unpinned/e3_unpinned.cpp)
add_executable(superslm_precache_unpinned EXCLUDE_FROM_ALL ${PC_UNPINNED_SOURCES})
# The plan's receipt for that rule: the same cells with the paged-KV fixtures taken through GetFixture under the e1
# target's pin. Expected red: 9.8 on pkv_def reads B = 16, not the rule's 128, at its first assertion.
add_executable(superslm_precache_unpinned_getfixture_receipt EXCLUDE_FROM_ALL ${PC_UNPINNED_SOURCES})
foreach(t superslm_precache_unpinned superslm_precache_unpinned_getfixture_receipt)
    target_link_libraries(${t} PRIVATE superslm_test_injection)
    target_include_directories(${t} PRIVATE include src tests tests/paged-kv ${PC_DIR})
    target_compile_definitions(${t} PRIVATE
        PKV_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/paged-kv/reference")
    if(MSVC)
        target_compile_options(${t} PRIVATE /W4 /fp:precise)
    else()
        target_compile_options(${t} PRIVATE -Wall -Wextra -ffp-contract=off)
    endif()
endforeach()
target_compile_definitions(superslm_precache_unpinned PRIVATE PKV_FIXTURE_B_FROM_RULE=1)
# Plan 9.9 [E5] (rev 1.18.1): 1.13.1's sequence saves on the release tree, at pkv_def's production rule B (the
# capture's). Only in the unpinned target: the receipt target's pinned GetFixture would restore at B = 16.
target_sources(superslm_precache_unpinned PRIVATE ${PC_DIR}/unpinned/e5_unpinned.cpp)
target_compile_definitions(superslm_precache_unpinned PRIVATE PC_REFERENCE_DIR="${PC_DIR}/reference")
target_compile_definitions(superslm_precache_unpinned_getfixture_receipt PRIVATE
    PKV_PIN_GENERATED_FIXTURES=1 PC_UNPINNED_RECEIPT_VIA_GETFIXTURE=1)
# Plan rev 1.18.1 (§8, R13): after M1 this plan's work runs on a post-M1 branch cut from main at the squash, and the
# line is origin/main. git merge-base HEAD origin/main then moves exactly when a main commit is merged into the
# branch, so every foreign commit taken in forces the population to be regenerated and licensed before grading.
# Set here, before the generator block reads it (code review 2026-10-22, finding 1: with it set later, a fresh cache
# read it empty and the first configure wrote "unanchored-no-merge-base").
set(PRECACHE_REFERENCE_LINE "origin/main" CACHE STRING
    "Line whose merge-base anchors licensed precache references")
# §8, R13: the licensed reference population is regenerated at every new `git merge-base HEAD <line>`. Its generator is
# T-3035's prefill-only harness, carried from origin/curie/t3035-precache-refs (972d5b7b; t3035_reference.cpp
# last changed there at abe60753, t3022_precache_reference.cpp at c9d003c8) with one change, plan rev 1.18.13 §15.33:
# the discrimination record each header binds is a required per-stem path argument, not a fixed v1 name. So the box builds the same capture code beside
# the engine it anchors to. It links the shipping library (no test seams) and calls no save or restore verb. Its
# header's anchor is the merge-base read when this tree is configured; when the engine sources at HEAD or in the
# working tree differ from that anchor's, or the history is shallow, the anchor is written as "unanchored-...", which
# every grading refuses (precache_reference.h). Reconfigure before regenerating.
add_executable(superslm_precache_reference_gen EXCLUDE_FROM_ALL ${PC_DIR}/reference/t3035_reference.cpp)
target_include_directories(superslm_precache_reference_gen PRIVATE include src)
target_link_libraries(superslm_precache_reference_gen PRIVATE superslm)
set(PC_GEN_ANCHOR "unanchored-no-git")
set(PC_GEN_HARNESS "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --is-shallow-repository
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE pc_shallow OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE PC_GEN_HARNESS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${GIT_EXECUTABLE} merge-base HEAD ${PRECACHE_REFERENCE_LINE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} RESULT_VARIABLE pc_mb_rc
        OUTPUT_VARIABLE pc_mb OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT pc_shallow STREQUAL "false")
        set(PC_GEN_ANCHOR "unanchored-shallow-history")
    elseif(NOT pc_mb_rc EQUAL 0 OR NOT pc_mb MATCHES "^[0-9a-f]+$")
        set(PC_GEN_ANCHOR "unanchored-no-merge-base")
    else()
        execute_process(COMMAND ${GIT_EXECUTABLE} diff --quiet ${pc_mb} HEAD -- src include CMakeLists.txt
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} RESULT_VARIABLE pc_engine_diff)
        # Uncommitted engine edits count too (code review finding 8): the working tree must equal HEAD there.
        execute_process(COMMAND ${GIT_EXECUTABLE} diff --quiet HEAD -- src include CMakeLists.txt
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} RESULT_VARIABLE pc_engine_dirty)
        if(pc_engine_diff EQUAL 0 AND pc_engine_dirty EQUAL 0)
            set(PC_GEN_ANCHOR "${pc_mb}")
        else()
            set(PC_GEN_ANCHOR "unanchored-engine-differs-from-${pc_mb}-or-dirty")
        endif()
    endif()
endif()
if(NOT PC_GEN_ANCHOR MATCHES "^[0-9a-f]+$")
    message(STATUS "superslm_precache_reference_gen: references would be written ${PC_GEN_ANCHOR}")
endif()
target_compile_definitions(superslm_precache_reference_gen PRIVATE
    T3035_ENGINE_ANCHOR="${PC_GEN_ANCHOR}" T3035_HARNESS_COMMIT="${PC_GEN_HARNESS}")
find_package(Python3 COMPONENTS Interpreter)
set(PC_MERGE_BASE_FILE "${CMAKE_CURRENT_BINARY_DIR}/precache-reference-merge-base.txt")
if(Python3_Interpreter_FOUND)
    set(PC_PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
    add_test(NAME superslm_precache_version_header
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/ci/check_version_header.py)
    add_test(NAME superslm_precache_version_header_commission
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/ci/test_check_version_header.py)
    add_test(NAME superslm_precache_converter_hash_guards
        COMMAND ${Python3_EXECUTABLE} ${PC_DIR}/check_scm1_serializer.py)
    add_test(NAME superslm_precache_spx1_oracle
        COMMAND ${Python3_EXECUTABLE} ${PC_DIR}/test_check_spx1_oracle.py)
    add_test(NAME superslm_precache_status_mirrors
        COMMAND ${Python3_EXECUTABLE} ${PC_DIR}/check_prefix_status_mirrors.py)
    # Plan 7.17 (E5): every docs/api.md status ordinal is that status's ordinal; the new status at 30.
    add_test(NAME superslm_precache_api_ordinals
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/ci/check_api_ordinals.py)
    # Rev 1.18.4 (suite audit G11): the widened locator's must-accept and must-rejects.
    add_test(NAME superslm_precache_api_ordinals_locator
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/ci/test_check_api_ordinals.py)
    # Plan 7.22 (M1, E5, the 1.15.0 release): the GPU twins labelled unreleased until 1.16.0.
    add_test(NAME superslm_precache_gpu_prefix_labels
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/ci/check_gpu_prefix_labels.py)
    add_test(NAME superslm_precache_reference_header_identity
        COMMAND $<TARGET_FILE:superslm_precache_reference_header>)
    add_test(NAME superslm_precache_compiler_job_identity
        COMMAND $<TARGET_FILE:superslm_precache_compiler_job_identity>)
    set_tests_properties(superslm_precache_reference_header_identity PROPERTIES
        FIXTURES_SETUP precache_header_identity)
    add_test(NAME superslm_precache_reference_anchor
        COMMAND ${Python3_EXECUTABLE} ${PC_DIR}/write_merge_base.py
            ${CMAKE_CURRENT_SOURCE_DIR} ${PRECACHE_REFERENCE_LINE} ${PC_MERGE_BASE_FILE})
    set_tests_properties(superslm_precache_reference_anchor PROPERTIES FIXTURES_SETUP precache_reference)
    # Plan rev 1.18.9 §15.28 item 2 (§8): the anchor fixture refuses a shallow history, on a repository of the cell's own.
    add_test(NAME superslm_precache_merge_base_shallow
        COMMAND ${Python3_EXECUTABLE} ${PC_DIR}/test_merge_base_shallow.py)
    add_test(NAME superslm_precache_real_two_process
        COMMAND ${Python3_EXECUTABLE} ${PC_DIR}/run_real_two_process.py
            $<TARGET_FILE:superslm_precache_e1>)
    set_tests_properties(superslm_precache_real_two_process PROPERTIES
        FIXTURES_REQUIRED "precache_reference;precache_header_identity"
        ENVIRONMENT "SUPERSLM_PRECACHE_MERGE_BASE_FILE=${PC_MERGE_BASE_FILE}")
    add_test(NAME superslm_precache_e2_reference
        COMMAND $<TARGET_FILE:superslm_precache_e1> "=6.2/E2:reference")
    add_test(NAME superslm_precache_e3_reference
        COMMAND $<TARGET_FILE:superslm_precache_e1> "=6.3/E3:reference")
    set_tests_properties(superslm_precache_e2_reference superslm_precache_e3_reference PROPERTIES
        FIXTURES_REQUIRED "precache_reference;precache_header_identity"
        ENVIRONMENT "SUPERSLM_PRECACHE_MERGE_BASE_FILE=${PC_MERGE_BASE_FILE}")
else()
    set(PC_PYTHON_EXECUTABLE python3)
endif()
target_compile_definitions(superslm_precache_e1 PRIVATE
    PC_PYTHON_PATH="${PC_PYTHON_EXECUTABLE}"
    PC_CMAKE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
    PC_FORMAT_SCRIPT_PATH="${PC_DIR}/check_spx1.py"
    PC_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    PC_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    PC_PROJECT_VERSION="${PROJECT_VERSION}")
if(MSVC)
    target_compile_options(superslm_precache_e1 PRIVATE /W4 /fp:precise)
else()
    target_compile_options(superslm_precache_e1 PRIVATE -Wall -Wextra -ffp-contract=off)
endif()
