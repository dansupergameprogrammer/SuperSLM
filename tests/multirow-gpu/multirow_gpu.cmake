# Multi-row GPU plan (rev 1.3.2): the red suite's CTest wiring. Included from CMakeLists.txt only when
# SUPERSLM_MULTIROW_GPU_TESTS is ON (default OFF), so no other build sees these tests.
#
# Registered tests (all host-only; the GPU legs run through the scripts named in each file's header):
#   mr1_cells            cells 4.3, 4.4, 4.5, 4.7 and mutants MX14/MX15 (tests/multirow-gpu/test_mr1_cells.py);
#                        red until the MR1 port lands (no gen_wide_shaders.py under src/gpu)
#   mr1_lint_population  the 4.7 lint's own must-accept / must-reject population (green before the port)
# Needs python3 and a Linux or Windows dxc (DXC, PATH, or /opt/dxc/bin/dxc); 4.3c also needs mingw-w64 g++
# (MR_MINGW_CXX) on Linux.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME mr1_cells
	COMMAND ${Python3_EXECUTABLE} -m unittest -v tests/multirow-gpu/test_mr1_cells.py
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME mr1_lint_population
	COMMAND ${Python3_EXECUTABLE} -m unittest -v tests/multirow-gpu/test_mr1_lint_population.py
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(mr1_cells mr1_lint_population PROPERTIES LABELS "multirow-gpu;mr1")
