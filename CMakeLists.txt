cmake_minimum_required(VERSION 3.16)
project(superslm VERSION 1.0.0 LANGUAGES CXX)

# C++20: the S-LoRA delta path is specified at -std=c++20 (SuperSLM_Plan.md §19),
# and the core carries wide-integer (int64) composition arithmetic throughout. One
# standard across the project avoids a later per-target split.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# T-2115 (D-SLM3432): the versioned install surface (GNUInstallDirs for the
# standard destination layout, CMakePackageConfigHelpers for the generated
# find_package() config) needed by every install() rule below.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# T-2115 fix round (S1, Claude/Poirot/6f60491-t2115-cmake-product-surface.md):
# the fixed, non-config-dependent install destination for the compiled GPU
# shaders -- defined unconditionally (not only inside `if(TARGET
# superslm_gpu)` below) so configure_package_config_file's PATH_VARS entry
# for it always resolves, whether or not this configure actually built the
# GPU target. Under CMAKE_INSTALL_DATADIR (GNUInstallDirs' own data-files
# convention), not LIBDIR/BINDIR: the .cso files are neither a library nor an
# executable, and this destination is never assumed to sit beside a
# consumer's own binary (see cmake/superslmConfig.cmake.in's own comment).
set(SUPERSLM_INSTALL_GPU_SHADER_DIR "${CMAKE_INSTALL_DATADIR}/superslm/gpu-shaders")

# The core library. Layer 1 is independently embeddable — no Unreal, no third-party
# runtime dependency in its graph (DecisionLog D-SLM13). Standard library only.
set(SUPERSLM_CORE_SOURCES
	src/artifact.cpp
	src/sha256.cpp
	src/tokenizer.cpp
	src/model.cpp
	src/intmath.cpp
	src/silu_lut.cpp
	src/matmul.cpp
	src/proof_manifest.cpp
	src/trace_hook.cpp
	src/forward/checked_chain_funnel.cpp
	src/forward/forward_sites.cpp
	src/decode_digest.cpp
)

add_library(superslm STATIC ${SUPERSLM_CORE_SOURCES})
# BUILD_INTERFACE/INSTALL_INTERFACE: consumers building against this tree see
# the source-tree include/ path; consumers of an installed package see the
# installed header destination (CMAKE_INSTALL_INCLUDEDIR) instead — the same
# target property serves both without a second target definition.
target_include_directories(superslm PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Determinism: forbid FP contraction. Implicit fused multiply-add would change
# rounding and break the scalar/SIMD bit-equality the whole project is built on
# (SuperSLM_Plan.md §6, §13). The kernel set (S2) will add explicit, mirrored FMA;
# until then the rule is simply "no implicit contraction anywhere."
if(MSVC)
	target_compile_options(superslm PRIVATE /W4 /fp:precise)
else()
	target_compile_options(superslm PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

# S-HARDEN-7's test-only fault-injection seam (tests/support/bad_alloc_
# injection.h) lets the test suite prove the "throws only std::bad_alloc"
# contract without any production consumer of this library ever compiling a
# reference to tests/support/*. A second compile of the SAME sources, built
# only with SUPERSLM_ENABLE_BAD_ALLOC_INJECTION defined, backs the test
# binary; every other consumer (sslm_verify, and every tool built outside
# CMake by build_inspect.bat) links the clean `superslm` above, which never
# defines the macro and so never even includes the seam header
# (src/bad_alloc_wrap.h #ifdef-gates it out entirely -- Layer 1 stays
# independently embeddable, D-SLM13).
add_library(superslm_test_injection STATIC ${SUPERSLM_CORE_SOURCES})
target_include_directories(superslm_test_injection PUBLIC include)
target_include_directories(superslm_test_injection PRIVATE tests)
target_compile_definitions(superslm_test_injection PRIVATE SUPERSLM_ENABLE_BAD_ALLOC_INJECTION)
if(MSVC)
	target_compile_options(superslm_test_injection PRIVATE /W4 /fp:precise)
else()
	target_compile_options(superslm_test_injection PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

add_executable(superslm_tests tests/test_main.cpp)
target_link_libraries(superslm_tests PRIVATE superslm_test_injection)
if(MSVC)
	target_compile_options(superslm_tests PRIVATE /W4 /fp:precise)
else()
	target_compile_options(superslm_tests PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

enable_testing()
add_test(NAME superslm_tests COMMAND superslm_tests)

# The §13 item 7 independent converter verifier (S-HARDEN-3, F13): loads a
# `.sslm` through the same SslmModel::Load every other consumer uses, then
# writes a proof manifest (include/superslm/proof_manifest.h) from what the
# loader actually accepted -- never from the Python writer's own arrays.
# convert_model.py invokes this binary after every write; §17.3 cell 2's
# integration test invokes it directly as the independent oracle.
add_executable(sslm_verify tools/sslm_verify.cpp)
target_link_libraries(sslm_verify PRIVATE superslm)
if(MSVC)
	target_compile_options(sslm_verify PRIVATE /W4 /fp:precise)
else()
	target_compile_options(sslm_verify PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

# T-2046 (design §24.2 D-SLM3162, closing Poirot c81e48c review Significant 6): dumps an adapter
# artifact's DeltaFoldScales/UFoldScales/Weights sections, loaded through the REAL B0b typed path,
# as JSON -- the write->load round-trip proof's own read side (tools/sslm_adapter_dump.cpp).
add_executable(sslm_adapter_dump tools/sslm_adapter_dump.cpp)
target_link_libraries(sslm_adapter_dump PRIVATE superslm)
if(MSVC)
	target_compile_options(sslm_adapter_dump PRIVATE /W4 /fp:precise)
else()
	target_compile_options(sslm_adapter_dump PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

# S-HARDEN-6 (design Sec2.1 component 2/3, T-410): the cross-toolchain /
# cross-optimization axis-digest harness, promoted from a disposable
# commission instrument into committed CI tooling
# (tools/ci/sslm_axis_digest.cpp). sslm_axis_digest links the clean
# `superslm` (the SIMD-enabled path on x64, matching production); a second
# executable, sslm_axis_digest_scalar_forced, links a third compile of the
# same core sources with SUPERSLM_FORCE_SCALAR_MATMUL defined -- the
# component-3 axis the digest-compare job needs a scalar-only build to
# compare against.
add_executable(sslm_axis_digest tools/ci/sslm_axis_digest.cpp)
target_link_libraries(sslm_axis_digest PRIVATE superslm)
if(MSVC)
	target_compile_options(sslm_axis_digest PRIVATE /W4 /fp:precise)
else()
	target_compile_options(sslm_axis_digest PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

add_library(superslm_scalar_forced STATIC ${SUPERSLM_CORE_SOURCES})
target_include_directories(superslm_scalar_forced PUBLIC include)
target_compile_definitions(superslm_scalar_forced PRIVATE SUPERSLM_FORCE_SCALAR_MATMUL)
if(MSVC)
	target_compile_options(superslm_scalar_forced PRIVATE /W4 /fp:precise)
else()
	target_compile_options(superslm_scalar_forced PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

add_executable(sslm_axis_digest_scalar_forced tools/ci/sslm_axis_digest.cpp)
target_link_libraries(sslm_axis_digest_scalar_forced PRIVATE superslm_scalar_forced)
if(MSVC)
	target_compile_options(sslm_axis_digest_scalar_forced PRIVATE /W4 /fp:precise)
else()
	target_compile_options(sslm_axis_digest_scalar_forced PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

# T-2115 (D-SLM3432): the GPU acceleration target (src/gpu/gpu_1p0.cpp,
# src/gpu/superslm_gpu.cpp) is D3D12/HLSL and Windows-only -- build.bat is the
# only build recipe that has ever compiled it. Off by default and gated behind
# an explicit option, so every CI leg in .github/workflows/tests.yml (including
# windows-x64, which runs a bare `cmake --build build` with no target named and
# therefore builds every default target in the project) keeps building exactly
# what it builds today; the GPU library only enters the build when a caller
# opts in with -DSUPERSLM_BUILD_GPU=ON, and it errors immediately if that
# happens on a non-Windows configure rather than failing later on a missing
# header.
option(SUPERSLM_BUILD_GPU "Build the D3D12/HLSL GPU acceleration library (Windows/MSVC only)" OFF)

if(SUPERSLM_BUILD_GPU)
	if(NOT WIN32)
		message(FATAL_ERROR "SUPERSLM_BUILD_GPU requires Windows: src/gpu/*.cpp is a D3D12/HLSL implementation with no non-Windows backend.")
	endif()

	# dxc.exe (the DirectX Shader Compiler) is not reliably on PATH; build.bat
	# hardcodes one Windows Kits 10 version. Search PATH first, then fall back
	# to a glob over every installed Windows Kits 10 SDK version's x64 dxc.exe,
	# preferring the highest (last, after a lexical sort -- SDK version
	# directories are dot-separated zero-padded numerals, e.g. 10.0.19041.0,
	# so a lexical sort is also a numeric sort here).
	find_program(SUPERSLM_DXC_EXECUTABLE NAMES dxc.exe dxc)
	if(NOT SUPERSLM_DXC_EXECUTABLE)
		file(GLOB SUPERSLM_DXC_CANDIDATES
			"$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin/*/x64/dxc.exe"
			"C:/Program Files (x86)/Windows Kits/10/bin/*/x64/dxc.exe"
		)
		if(SUPERSLM_DXC_CANDIDATES)
			list(SORT SUPERSLM_DXC_CANDIDATES)
			list(REVERSE SUPERSLM_DXC_CANDIDATES)
			list(GET SUPERSLM_DXC_CANDIDATES 0 SUPERSLM_DXC_EXECUTABLE)
		endif()
	endif()
	if(NOT SUPERSLM_DXC_EXECUTABLE)
		message(FATAL_ERROR "SUPERSLM_BUILD_GPU=ON but dxc.exe (DirectX Shader Compiler) was not found on PATH or under any installed Windows Kits 10 SDK. Install the Windows SDK, or pass -DSUPERSLM_DXC_EXECUTABLE=<path to dxc.exe>.")
	endif()
	message(STATUS "SuperSLM GPU: compiling shaders with ${SUPERSLM_DXC_EXECUTABLE}")

	# Every compute shader compiles to cs_6_2, -O3, -HV 2018, -WX -- the same
	# pinned target and flags build.bat uses (T-1986 GPU-serial port, Sec5.7).
	# superslm_gpu::harness::ShaderPath() (src/gpu/superslm_gpu.cpp) resolves
	# "<exe_dir>/shaders/<name>.cso" from GetModuleFileNameA at runtime -- the
	# directory beside whatever executable is actually running, not a fixed
	# build-tree path. On the Windows generator (multi-config: Visual Studio)
	# an executable's own default output directory is
	# ${CMAKE_BINARY_DIR}/$<CONFIG>/, not ${CMAKE_BINARY_DIR}/ itself, so the
	# shader output directory below carries the same $<CONFIG> generator
	# expression -- on a single-config generator (Ninja, Makefiles)
	# GENERATOR_IS_MULTI_CONFIG is false and $<CONFIG> is never part of an
	# executable's own output path, so the shaders land directly under
	# ${CMAKE_BINARY_DIR}/shaders there instead, matching build.bat's own
	# out\shaders-beside-out\*.exe layout in both cases. This unit adds no
	# in-tree GPU-dispatching executable of its own -- this only makes the
	# location agree with where one would land if added.
	file(GLOB SUPERSLM_GPU_SHADER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/shaders/*.hlsl")
	file(GLOB SUPERSLM_GPU_SHADER_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/shaders/*.hlsli")

	get_property(SUPERSLM_GPU_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
	if(SUPERSLM_GPU_IS_MULTI_CONFIG)
		set(SUPERSLM_GPU_SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/$<CONFIG>/shaders")
	else()
		set(SUPERSLM_GPU_SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")
	endif()

	# T-2115 fix round 3 (NEW-1, Claude/Poirot/6f60491-t2115-cmake-product-
	# surface.md): a second, FIXED copy of every compiled .cso, independent of
	# $<CONFIG>, staged here purely so install(FILES ...) below never needs a
	# generator expression in its file list -- install(FILES ...) resolving a
	# $<CONFIG>-bearing path against `cmake --install --config <cfg>` is not,
	# on inspection of CMake's own documentation, gated on a cmake_minimum_
	# required floor above 3.16 the way an earlier round of this fix
	# incorrectly justified (that round cited install(FILES RENAME ...)'s own
	# 3.20 addition, a feature this file does not use); raising the floor
	# anyway would have cost CMake 3.16-3.19 users -- Ubuntu 20.04, Debian
	# 11 -- the ability to configure this project AT ALL, for a Windows-only,
	# off-by-default target's install detail. Avoided entirely by never
	# putting a generator expression in an install(FILES ...) list: this
	# staging copy is a plain, configure-time-known path, unconditionally the
	# same string on every generator and every configuration. A single
	# file(MAKE_DIRECTORY ...) below (not a build-time step) is correct here,
	# unlike SUPERSLM_GPU_SHADER_OUT_DIR above -- this path carries no
	# generator expression for file(MAKE_DIRECTORY ...) to fail to evaluate.
	set(SUPERSLM_GPU_SHADER_STAGE_DIR "${CMAKE_BINARY_DIR}/gpu-shaders-staged")
	file(MAKE_DIRECTORY "${SUPERSLM_GPU_SHADER_STAGE_DIR}")

	set(SUPERSLM_GPU_SHADER_OUTPUTS "")
	set(SUPERSLM_GPU_SHADER_STAGED "")
	foreach(_superslm_shader ${SUPERSLM_GPU_SHADER_SOURCES})
		get_filename_component(_superslm_shader_name "${_superslm_shader}" NAME_WE)
		set(_superslm_shader_cso "${SUPERSLM_GPU_SHADER_OUT_DIR}/${_superslm_shader_name}.cso")
		set(_superslm_shader_staged "${SUPERSLM_GPU_SHADER_STAGE_DIR}/${_superslm_shader_name}.cso")
		add_custom_command(
			OUTPUT "${_superslm_shader_cso}" "${_superslm_shader_staged}"
			# The output directory is created here, at build time, rather than via a
			# configure-time file(MAKE_DIRECTORY ...): SUPERSLM_GPU_SHADER_OUT_DIR
			# carries a $<CONFIG> generator expression on multi-config generators,
			# which file(MAKE_DIRECTORY ...) cannot evaluate (it only ever sees the
			# literal, un-expanded string at configure time).
			COMMAND "${CMAKE_COMMAND}" -E make_directory "${SUPERSLM_GPU_SHADER_OUT_DIR}"
			COMMAND "${SUPERSLM_DXC_EXECUTABLE}" -T cs_6_2 -E main -Fo "${_superslm_shader_cso}" "${_superslm_shader}" -O3 -HV 2018 -WX
			COMMAND "${CMAKE_COMMAND}" -E copy "${_superslm_shader_cso}" "${_superslm_shader_staged}"
			DEPENDS "${_superslm_shader}" ${SUPERSLM_GPU_SHADER_HEADERS}
			COMMENT "dxc: ${_superslm_shader_name}.hlsl -> ${_superslm_shader_name}.cso"
			VERBATIM
		)
		list(APPEND SUPERSLM_GPU_SHADER_OUTPUTS "${_superslm_shader_cso}")
		list(APPEND SUPERSLM_GPU_SHADER_STAGED "${_superslm_shader_staged}")
	endforeach()
	unset(_superslm_shader)
	unset(_superslm_shader_name)
	unset(_superslm_shader_cso)
	unset(_superslm_shader_staged)

	add_custom_target(superslm_gpu_shaders DEPENDS ${SUPERSLM_GPU_SHADER_OUTPUTS} ${SUPERSLM_GPU_SHADER_STAGED})

	add_library(superslm_gpu STATIC
		src/gpu/gpu_1p0.cpp
		src/gpu/superslm_gpu.cpp
	)
	# PUBLIC link to superslm: the GPU sources use SslmModelView/SslmModel::Load
	# (model.h) and the checked-chain-funnel/forward-sites declarations
	# (gpu_port.h), and this also carries superslm's own PUBLIC include/
	# interface (BUILD_INTERFACE/INSTALL_INTERFACE above) through transitively,
	# so superslm_gpu needs no separate include_directories(... PUBLIC include)
	# of its own.
	target_link_libraries(superslm_gpu PUBLIC superslm)
	# d3d12_harness.h (src/gpu/, not installed -- GPU-internal) is a private
	# implementation include, matching build.bat's own /Iinclude (public) with
	# no separate GPU include path: the two GPU .cpp files #include
	# "d3d12_harness.h" directly, which only resolves because src/gpu is each
	# translation unit's own containing directory -- adding it explicitly here
	# makes that dependency visible rather than relying on the implicit
	# same-directory search every compiler also does.
	target_include_directories(superslm_gpu PRIVATE src/gpu)
	target_link_libraries(superslm_gpu PRIVATE d3d12 dxgi dxguid)
	# MSVC-guarded, not WIN32-guarded: WIN32 is also true under a MinGW or
	# GNU-frontend-clang toolchain, where MSVC is false and /W4/-fp:precise
	# are not valid flags (the compiler chokes on /W4 as a filename) --
	# matching the if(MSVC)/else() convention every other target in this file
	# already uses, rather than assuming WIN32 implies MSVC.
	if(MSVC)
		target_compile_options(superslm_gpu PRIVATE /W4 /fp:precise)
	else()
		target_compile_options(superslm_gpu PRIVATE -Wall -Wextra -ffp-contract=off)
	endif()
	add_dependencies(superslm_gpu superslm_gpu_shaders)
endif()

# T-2115 (D-SLM3432): the versioned install surface. Only the product library
# (or libraries, when SUPERSLM_BUILD_GPU=ON) and the public headers are
# installed -- superslm_test_injection, superslm_tests, and every tool/
# executable above are internal to this repository's own build and test loop,
# not part of what an external consumer links against, so none of them gets
# an install() rule.
install(TARGETS superslm
	EXPORT superslmTargets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(TARGET superslm_gpu)
	install(TARGETS superslm_gpu
		EXPORT superslmTargets
		ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
		RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
	)

	# S1 (Claude/Poirot/6f60491-t2115-cmake-product-surface.md): superslm_gpu
	# is unusable without its 33 compiled .cso shaders -- every real GPU
	# dispatch reaches superslm_gpu::harness::GetOrBuildPipeline/
	# GetOrBuildComposedPipeline (src/gpu/superslm_gpu.cpp), which throws if
	# ShaderPath()'s file is missing, and every call site along that path
	# catches std::exception and returns a generic failure status -- an
	# installed package with the library but not the shaders links and runs
	# sslm_gpu_context_create cleanly, then fails opaquely at first dispatch.
	# Installs from SUPERSLM_GPU_SHADER_STAGED (NEW-1 fix, round 3), the fixed,
	# non-config-dependent copy staged above -- no generator expression
	# anywhere in this FILES list. The .cso bytecode itself does not vary by
	# Debug/Release -- only the build-tree path used to reach it does -- so a
	# single staged copy, produced once regardless of which configuration was
	# actually built, is correct for every `cmake --install` regardless of
	# `--config`.
	install(FILES ${SUPERSLM_GPU_SHADER_STAGED}
		DESTINATION ${SUPERSLM_INSTALL_GPU_SHADER_DIR}
	)
endif()

install(DIRECTORY include/superslm/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/superslm
)

install(EXPORT superslmTargets
	FILE superslmTargets.cmake
	NAMESPACE superslm::
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/superslm
)

write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/superslmConfigVersion.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/superslmConfig.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/superslmConfig.cmake"
	INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/superslm
	PATH_VARS SUPERSLM_INSTALL_GPU_SHADER_DIR
)

install(FILES
	"${CMAKE_CURRENT_BINARY_DIR}/superslmConfig.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/superslmConfigVersion.cmake"
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/superslm
)
