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
# proves the INTENDED assertion is what fired). A failure for the wrong reason is reported as a
# hard script failure -- ctest sees this script exit non-zero and (via its own WILL_FAIL) still
# reports the test correctly, but the diagnostic in the log names the real problem instead of
# reading as a routine, expected compile failure.
#
# Invocation (see CMakeLists.txt's own add_test entries for the exact arguments):
#   cmake -DBUILD_DIR=<binary dir> -DBUILD_TARGET=<target> -DBUILD_CONFIG=<config>
#         -DMARKER_TEXT=<substring of the target's own static_assert message> -P
#         tools/ci/check_negative_compile_marker.cmake
if(NOT DEFINED BUILD_DIR OR NOT DEFINED BUILD_TARGET OR NOT DEFINED MARKER_TEXT)
	message(FATAL_ERROR "check_negative_compile_marker.cmake: BUILD_DIR, BUILD_TARGET, and MARKER_TEXT are all required -DVAR=... arguments")
endif()
if(NOT DEFINED BUILD_CONFIG OR BUILD_CONFIG STREQUAL "")
	set(BUILD_CONFIG "Debug")
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
