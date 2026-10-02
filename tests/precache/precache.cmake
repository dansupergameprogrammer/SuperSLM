set(PC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/precache)
file(GLOB PC_SOURCES CONFIGURE_DEPENDS ${PC_DIR}/e*.cpp)
add_executable(superslm_precache_e1 EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/paged-kv/pkv_main.cpp
    ${PC_DIR}/precache_stubs.cpp ${PC_SOURCES})
# Invoked separately at E2. It deliberately fails to compile against the E1
# header and must compile only when the public macro and ordinal are published.
add_executable(superslm_precache_macro_vitality EXCLUDE_FROM_ALL
    ${PC_DIR}/prefix_macro_vitality.cpp)
target_include_directories(superslm_precache_macro_vitality PRIVATE include)
target_link_libraries(superslm_precache_e1 PRIVATE superslm_test_injection)
target_include_directories(superslm_precache_e1 PRIVATE include src tests tests/paged-kv ${PC_DIR})
target_compile_definitions(superslm_precache_e1 PRIVATE
    PKV_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/paged-kv/reference")
find_package(Python3 COMPONENTS Interpreter)
set(PRECACHE_REFERENCE_LINE "origin/claude/shared-prefix-kv-build-wj8q9c" CACHE STRING
    "Line whose merge-base anchors licensed precache references")
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
    add_test(NAME superslm_precache_reference_anchor
        COMMAND ${Python3_EXECUTABLE} ${PC_DIR}/write_merge_base.py
            ${CMAKE_CURRENT_SOURCE_DIR} ${PRECACHE_REFERENCE_LINE} ${PC_MERGE_BASE_FILE})
    set_tests_properties(superslm_precache_reference_anchor PROPERTIES FIXTURES_SETUP precache_reference)
    add_test(NAME superslm_precache_real_two_process
        COMMAND ${Python3_EXECUTABLE} ${PC_DIR}/run_real_two_process.py
            $<TARGET_FILE:superslm_precache_e1>)
    set_tests_properties(superslm_precache_real_two_process PROPERTIES
        FIXTURES_REQUIRED precache_reference
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
