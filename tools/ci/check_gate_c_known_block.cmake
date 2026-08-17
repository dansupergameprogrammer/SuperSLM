# check_gate_c_known_block.cmake -- P2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.4,
# third confirmation pass). CTest's own WILL_FAIL property is a static flag with no way to check
# WHY a build failed -- exactly the shape that made N1 a real finding (a flag written for one
# known condition, never revisited when that condition changed). Gate C's own must-accept target
# is currently, correctly expected to fail to compile for a SPECIFIC, disclosed reason (the
# ordinal interleaving P2 names, tools/t2139_gate_c_type_identity_check.cpp's own assertion
# message) -- this script runs the build and passes ONLY if that specific message is present,
# so a compile that starts succeeding (the reconciliation landing) OR one that fails for a
# DIFFERENT, unexpected reason both correctly report as a test failure here, rather than either
# silently masquerading as the other.
#
# Invoked as: cmake -DBUILD_COMMAND=... -DBUILD_DIR=... -P check_gate_c_known_block.cmake
execute_process(
	COMMAND ${CMAKE_COMMAND} --build ${BUILD_DIR} --target t2139_gate_c_must_accept --config ${CONFIG}
	RESULT_VARIABLE build_result
	OUTPUT_VARIABLE build_output
	ERROR_VARIABLE build_error
)

string(FIND "${build_output}${build_error}" "ordinal collision" found_marker)

if(build_result EQUAL 0)
	message(FATAL_ERROR "Gate C must-accept COMPILED CLEAN -- P2's own known-interleaving block is "
		"no longer present. This is either the enum-governance ruling landing (update this "
		"script and CMakeLists.txt to a plain must-pass check, matching Gate A's own shape) or a "
		"real regression -- investigate before assuming either.")
elseif(found_marker EQUAL -1)
	message(FATAL_ERROR "Gate C must-accept FAILED TO COMPILE for an UNEXPECTED reason (the known "
		"'ordinal collision' marker is absent from the compiler's own output) -- see the output "
		"below.\n${build_output}${build_error}")
else()
	message(STATUS "Gate C must-accept: KNOWN, DISCLOSED ordinal-interleaving block (P2, "
		"Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.4) -- enum-governance ruling "
		"pending at the planner.")
endif()
