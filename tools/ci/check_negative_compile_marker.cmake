# T-2139 fifth confirmation review (Claude/Poirot/ce5aff2-t2139-fifth-confirmation-review.md S3):
# ctest's WILL_FAIL only inverts the build's own EXIT CODE -- it cannot distinguish a must-reject
# construction's own assertion firing from any OTHER reason the same target might fail to compile
# (a stray syntax error, a missing header, an unrelated regression upstream of the assertion). The
# sentinel must-reject construction (tools/t2139_gate_c_sentinel_negative.cpp) demonstrated this
# live: one governed append to both real headers desynchronizes the construction's own extra
# enumerator from the real sentinels, so the file still fails to compile -- for an unrelated
# C2039 name-lookup reason -- while WILL_FAIL keeps reporting green with nothing anywhere showing
# the sentinel mechanism itself stopped firing.
#
# This script runs the named target's build and requires BOTH: the build failed, AND its own
# captured output contains the target's own marker text (the literal static_assert message that
# proves the INTENDED assertion is what fired). The script's contract is exit 0 exactly when the
# construction failed for its own reason -- a failure for the wrong reason is a hard script
# failure (non-zero exit), which ctest reports as a plain, correctly-attributed test failure.
#
# WILL_FAIL MUST NEVER BE SET on a ctest add_test entry that invokes this script (T-2139 sixth
# confirmation review, Claude/Poirot/5fbd04d-t2139-sixth-confirmation-review.md S2). This script's
# own exit code is already correct -- 0 only when both conditions hold. Adding WILL_FAIL on top
# inverts the check completely: a wrong-reason compile failure (this script exits 1) would report
# PASS, and a healthy tree (this script exits 0) would report FAIL. See CMakeLists.txt's own
# add_test entries for this script -- none of them set WILL_FAIL, and none of them ever should.
#
# SOURCE_FILE (T-2142, Claude/Poirot/aea6116-t2139-seventh-confirmation-review.md M1, the
# residual half): "a marker must match its construction's own full static_assert text, not any
# substring of it." MARKER_TEXT alone, checked only against the compiler's OWN output, cannot
# tell a complete message from a truncated fragment of it -- a truncated fragment IS a substring
# of the real message, so it passes the compiler-output check vacuously too. This is not
# hypothetical: two of this file's own four callers carried exactly this mangling in HEAD before
# this fix (CMakeLists.txt's xmacro/sentinel entries were missing their construction's own
# macro-generated suffix and literal prefix respectively). SOURCE_FILE names the .cpp the target
# actually compiles; tools/ci/verify_negative_compile_marker_source.py reads it and requires
# MARKER_TEXT to equal one of the COMPLETE static_assert messages that file can produce, source-
# derived rather than compiler-output-derived, closing the truncation class the empty-string
# guard below does not reach.
#
# Invocation (see CMakeLists.txt's own add_test entries for the exact arguments):
#   cmake -DBUILD_DIR=<binary dir> -DBUILD_TARGET=<target> -DBUILD_CONFIG=<config>
#         -DSOURCE_FILE=<path to the target's own .cpp, relative to CMAKE_SOURCE_DIR>
#         -DMARKER_TEXT=<the target's own COMPLETE static_assert message text> -P
#         tools/ci/check_negative_compile_marker.cmake
if(NOT DEFINED BUILD_DIR OR NOT DEFINED BUILD_TARGET OR NOT DEFINED SOURCE_FILE OR NOT DEFINED MARKER_TEXT)
	message(FATAL_ERROR "check_negative_compile_marker.cmake: BUILD_DIR, BUILD_TARGET, SOURCE_FILE, and MARKER_TEXT are all required -DVAR=... arguments")
endif()
# T-2139 sixth confirmation review M1 (Claude/Poirot/5fbd04d-t2139-sixth-confirmation-review.md):
# -DMARKER_TEXT= (empty string) satisfies NOT DEFINED above (it IS defined, just empty), and
# string(FIND "${output}" "" pos) returns 0 -- found -- so an empty marker passed vacuously.
# Reject it explicitly rather than let a mis-escaped or dropped -DMARKER_TEXT= argument silently
# degrade this check to exit-code-only, which is the exact defect this script exists to close.
if(MARKER_TEXT STREQUAL "")
	message(FATAL_ERROR "check_negative_compile_marker.cmake: MARKER_TEXT must not be empty (an empty marker matches vacuously and defeats this script's whole purpose)")
endif()
if(NOT DEFINED BUILD_CONFIG OR BUILD_CONFIG STREQUAL "")
	set(BUILD_CONFIG "Debug")
endif()

# Source-derived completeness check (M1's residual half, see the SOURCE_FILE comment above):
# runs BEFORE the build, cheap, and independent of the compiler's own output -- a mangled marker
# is rejected here even if it would also happen to appear as a substring of whatever the compiler
# prints, which is exactly the failure mode a compiler-output-only check cannot see.
#
# CMAKE_SOURCE_DIR is NOT set in `cmake -P` script mode (this script's own invocation shape) --
# it silently expands empty, which previously resolved SOURCE_FILE relative to ctest's own
# working directory (BUILD_DIR) instead of the repo root, "file not found" on every caller.
# CMAKE_CURRENT_LIST_DIR IS set in script mode (this listfile's own directory, tools/ci/) and is
# the anchor every path below is computed from instead.
get_filename_component(SSLM_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
find_program(PYTHON_EXECUTABLE NAMES python python3)
if(NOT PYTHON_EXECUTABLE)
	message(FATAL_ERROR "check_negative_compile_marker.cmake: no python/python3 on PATH -- cannot run the SOURCE_FILE marker-completeness check (tools/ci/verify_negative_compile_marker_source.py). This is a hard requirement, not a skip: the class this check closes (a truncated marker passing vacuously) is exactly what happened in HEAD before it existed.")
endif()
execute_process(
	COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/verify_negative_compile_marker_source.py" "${SSLM_REPO_ROOT}/${SOURCE_FILE}" "${MARKER_TEXT}"
	RESULT_VARIABLE MARKER_SOURCE_RESULT
	OUTPUT_VARIABLE MARKER_SOURCE_STDOUT
	ERROR_VARIABLE MARKER_SOURCE_STDERR
)
if(NOT MARKER_SOURCE_RESULT EQUAL 0)
	message(FATAL_ERROR
		"${BUILD_TARGET}: MARKER_TEXT failed the source-completeness check against ${SOURCE_FILE}:\n${MARKER_SOURCE_STDOUT}${MARKER_SOURCE_STDERR}")
endif()

execute_process(
	COMMAND ${CMAKE_COMMAND} --build "${BUILD_DIR}" --target "${BUILD_TARGET}" --config "${BUILD_CONFIG}"
	RESULT_VARIABLE BUILD_RESULT
	OUTPUT_VARIABLE BUILD_STDOUT
	ERROR_VARIABLE BUILD_STDERR
)
set(FULL_OUTPUT "${BUILD_STDOUT}${BUILD_STDERR}")

if(BUILD_RESULT EQUAL 0)
	message(FATAL_ERROR
		"${BUILD_TARGET} COMPILED CLEAN -- this must-reject construction has regressed (its own mechanism no longer catches the divergence it names). Build output:\n${FULL_OUTPUT}")
endif()

string(FIND "${FULL_OUTPUT}" "${MARKER_TEXT}" MARKER_POS)
if(MARKER_POS EQUAL -1)
	message(FATAL_ERROR
		"${BUILD_TARGET} failed to compile, but NOT for its own assertion -- marker text \"${MARKER_TEXT}\" was not found in the build output, so this failure is for an unrelated reason (a governed append desynchronizing a hand-transcribed construction, a stray syntax error, a missing header, ...). Build output:\n${FULL_OUTPUT}")
endif()

message(STATUS "${BUILD_TARGET}: failed to compile for its own reason -- marker text confirmed present in the build output")
