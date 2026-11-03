# Engine 1.19, the clean stop at the schema's end: CTest wiring for the GPU cells G1-G10 (sfdg_cleanstop.h,
# sfdg_cell_cs_g1.cpp .. sfdg_cell_cs_g10.cpp). Included from CMakeLists.txt's WIN32 GPU block when BUILD_TESTING is on
# and the compiler is MSVC (the flags below are MSVC's). The CPU cells E1-E8 run inside superslm_tests
# (tests/schema_fast_decode/sfd_cells_cleanstop.inc).
#
# The engine is compiled once with the schema fast-decode GPU suite's seam definitions (build_sfd_gpu.bat's), and each
# cell links against it. The cells need a hardware D3D12 device, the shaders (superslm_gpu_shaders, staged beside the
# executables) and the F-SFD fixture in $SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR (python
# tools/gen_schema_fast_decode_fixture.py DIR); a missing fixture fails the cell. Hosted CI has no D3D12 device and
# does not select them; the box runs them with `ctest -C Release -L clean-stop-gpu`.
add_library(superslm_sfdg_cs_engine STATIC ${SUPERSLM_CORE_SOURCES} src/gpu/superslm_gpu.cpp src/gpu/gpu_1p0.cpp)
target_include_directories(superslm_sfdg_cs_engine PUBLIC include src src/gpu)
target_compile_definitions(superslm_sfdg_cs_engine PUBLIC SUPERSLM_GPU_TEST_SEAMS SUPERSLM_GPU_ALLOC_FAULT_INJECTION
	SUPERSLM_ENABLE_BAD_ALLOC_INJECTION SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION)
target_include_directories(superslm_sfdg_cs_engine PRIVATE tests)
target_compile_options(superslm_sfdg_cs_engine PRIVATE /W4 /fp:precise)
target_link_libraries(superslm_sfdg_cs_engine PUBLIC d3d12 dxgi dxguid shlwapi psapi)
add_dependencies(superslm_sfdg_cs_engine superslm_gpu_shaders)
foreach(_sslm_cs_cell 1 2 3 4 5 6 7 8 9 10)
	set(_sslm_cs_target sfdg_cell_cs_g${_sslm_cs_cell})
	add_executable(${_sslm_cs_target} tests/schema-fast-decode-gpu/${_sslm_cs_target}.cpp)
	target_link_libraries(${_sslm_cs_target} PRIVATE superslm_sfdg_cs_engine)
	target_include_directories(${_sslm_cs_target} PRIVATE tests tests/paged-kv tests/paged-kv-gpu
		tests/t2791-gpu-prefill-read-red-suite tests/schema-fast-decode-gpu)
	target_compile_options(${_sslm_cs_target} PRIVATE /W4 /fp:precise)
	add_test(NAME ${_sslm_cs_target} COMMAND ${_sslm_cs_target})
	set_tests_properties(${_sslm_cs_target} PROPERTIES LABELS "clean-stop-gpu;schema-fast-decode-gpu")
endforeach()
unset(_sslm_cs_cell)
unset(_sslm_cs_target)
