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
	src/sslm_abi.cpp
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

# T-2152 (outside strike item 1): tests/test_main.cpp's own T-2019/T-2024 GPU-serial-port
# section calls `superslm_gpu::*` directly (#ifdef _WIN32-guarded in that file so the same
# test_main.cpp still compiles on the Linux/macOS CI legs, which take neither branch below) --
# on WIN32 this target must also compile src/gpu/superslm_gpu.cpp and link d3d12/dxgi/dxguid,
# unconditionally, matching build.bat's own superslm_tests.exe recipe (build.bat, the "quick
# MSVC build + test" path, has never gated this behind an opt-in flag: it is the only build
# recipe that has ever produced a superslm_tests.exe that actually links, so its own shape is
# the ONE test_main truth this target now matches). This is deliberately independent of the
# SUPERSLM_BUILD_GPU option above: that option controls whether the public superslm_gpu
# library and its install() rule exist; superslm_tests needs the same GPU sources compiled in
# to link *at all* on Windows regardless of whether a consumer opts into the installed GPU
# target. Only src/gpu/superslm_gpu.cpp is added here, not src/gpu/gpu_1p0.cpp -- every symbol
# tests/test_main.cpp calls (SaveGpuSequenceState, RestoreGpuSequenceState,
# LastWeightUploadWasSkipped, RunLayerLoopGpu, etc.) is defined in superslm_gpu.cpp, matching
# build.bat's own main test recipe, which also omits gpu_1p0.cpp from this exact binary.
if(WIN32)
	target_sources(superslm_tests PRIVATE src/gpu/superslm_gpu.cpp)
	target_include_directories(superslm_tests PRIVATE src/gpu)
	target_link_libraries(superslm_tests PRIVATE d3d12 dxgi dxguid)
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

add_executable(sslm_generate tools/sslm_generate.cpp)
target_link_libraries(sslm_generate PRIVATE superslm)
if(MSVC)
	target_compile_options(sslm_generate PRIVATE /W4 /fp:precise)
else()
	target_compile_options(sslm_generate PRIVATE -Wall -Wextra -ffp-contract=off)
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

# T-2149 (Claude/Vitruvius/t2149-avx-kernel-design-2026-08-18.md) / T-2158 red suite
# (Claude/Curie/t2158-t2149-avx-red-suite-2026-08-18.md): the AVX2/AVX-512 matmul tiers,
# the cached CPUID dispatch (DetectBestDotRowTier), and the four-macro force selection
# (SUPERSLM_FORCE_{SCALAR,SSE2,AVX2,AVX512}_MATMUL, design §6.1) do not exist yet -- this
# option is OFF by default so the red suite's tier-dependent cells (dispatch-cache
# lifetime, first-call-race concurrency; design §10 dimensions 1/3) stay excluded from
# compilation and the existing 34184/0 battery (D-SLM3496) is unaffected. Whoever builds
# the design's §11 step 1 turns this ON (and, for step 5/7's new forced test executables,
# additionally propagates the corresponding SUPERSLM_FORCE_*_MATMUL definition to each
# forced `superslm_tests_*_forced` executable's own compile of tests/test_main.cpp --
# PUBLIC on the forced library, or an explicit target_compile_definitions on the forced
# executable target itself -- so the dispatch-cache cell's "== 0 in each forced binary"
# half, gated the same way, can see which tier it was built for; see the red-suite
# casebook §"Dimension 1" for the exact contract).
option(SUPERSLM_T2149_AVX_TIERS_BUILT
	"T-2149: the AVX2/AVX-512 matmul tiers and cached dispatch exist in src/matmul.cpp. OFF (default) keeps T-2158's tier-dependent red-suite cells out of the build so main stays green with the feature unbuilt."
	OFF)
if(SUPERSLM_T2149_AVX_TIERS_BUILT)
	target_compile_definitions(superslm_test_injection PUBLIC
		SUPERSLM_T2149_AVX_TIERS_BUILT
		SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT)
endif()

# T-2158 dimension 7 item (d) (design §10, D-SLM3510): a persisted CI guard for the
# AVX-isolation claim (design §6.4) -- no target_compile_options call anywhere in this
# file may set a translation-unit-wide AVX2/AVX-512 flag (see
# tools/ci/check_matmul_avx_isolation.py's own header for the full argument and its
# mutation-vitality proof, executed and recorded in the T-2158 casebook). Runs today,
# unconditionally, independent of SUPERSLM_T2149_AVX_TIERS_BUILT: the claim it protects
# ("only two function-attributed sites may use these instructions") is already true
# (vacuously, since neither attributed site exists yet) and stays true once they land --
# this is exactly the guard design §10 dimension 7 commissions as a standing check, not a
# one-time build-time probe like its dimension-7c/7g siblings (§11 steps 4b/4d).
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
	add_test(NAME check_matmul_avx_isolation
		COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/ci/check_matmul_avx_isolation.py
			--cmakelists ${CMAKE_SOURCE_DIR}/CMakeLists.txt)
else()
	message(STATUS "T-2158: Python3 interpreter not found -- check_matmul_avx_isolation ctest entry not added (the script itself is still committed and runnable directly).")
endif()

# C1 fix round (S9/S8 residual, Claude/Poirot/2c18dab-t2139-abi-build-review.md): Gate A and Gate
# C's own must-accept/must-reject constructions (tools/t2139_gate_a_header_parity_check*.cpp,
# tools/t2139_gate_c_type_identity_check*.cpp) were previously compiled ONLY by build.bat (MSVC
# x64) -- the design's own stated CI-matrix closure mechanism ("re-run as part of every real
# slot's CI build across that matrix") did not exist for either gate. Wired here as CMake targets
# so `cmake --build` on ANY configured toolchain (not only build.bat's MSVC x64) exercises them.
#
# EXCLUDE_FROM_ALL: none of these four is meant to LINK into anything or run as a program (Gate A
# is compile-only by design Sec9's own construction; Gate C the same) -- they are pure
# compilation checks, so they are built on demand by the ctest entries below rather than by every
# plain `cmake --build`. The must-reject targets are EXPECTED TO FAIL TO COMPILE; the must-accept
# targets are expected to succeed, both of them -- Gate C's own sslm_g5.h ordinal-17 collision
# (SSLM_TOKEN_ID_UNMAPPED vs SSLM_SCHEMA_NOT_FOUND) reconciled at curie/t2130-g5-red-suite@59e26ff,
# closing S8 for real.
#
# N1 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3): the WILL_FAIL TRUE below on Gate C's
# own must-accept entry was written when the reconciliation above was still pending, with a comment
# saying to flip it once reconciliation landed -- reconciliation landed two rounds later and the
# flag was never revisited (the exact rule-enforced-by-memory class this arc has been fighting
# elsewhere). Left in place, this ctest entry inverts: it reports FAILURE (ctest exit 8) on a
# healthy tree, and would report SUCCESS the day Gate C genuinely regresses -- the one entry whose
# whole job is catching header drift silently saying the opposite of what it finds. Fixed here:
# WILL_FAIL removed (the target now must actually compile, matching its Gate A sibling), and the
# entry renamed off its own now-false "known_pending" name.
add_executable(t2139_gate_a_must_accept EXCLUDE_FROM_ALL
	tools/t2139_gate_a_header_parity_check.cpp)
target_include_directories(t2139_gate_a_must_accept PRIVATE include tests)

add_executable(t2139_gate_a_must_reject EXCLUDE_FROM_ALL
	tools/t2139_gate_a_header_parity_check_negative.cpp)
target_include_directories(t2139_gate_a_must_reject PRIVATE include tools tests)

add_executable(t2139_gate_c_must_accept EXCLUDE_FROM_ALL
	tools/t2139_gate_c_type_identity_check.cpp)
target_include_directories(t2139_gate_c_must_accept PRIVATE include)

add_executable(t2139_gate_c_must_reject EXCLUDE_FROM_ALL
	tools/t2139_gate_c_type_identity_check_negative.cpp)
target_include_directories(t2139_gate_c_must_reject PRIVATE include)

# S1's structural remedy (Claude/Poirot/3bcbe43-t2139-fourth-confirmation-review.md, owed remedy
# 5): the THIRD Gate C TU -- includes the REAL tests/t2130-g5-red-suite/sslm_g5.h directly (not a
# transcription) alongside the library's own SSLM_STATUS_ENUM_LIST expansion, so a divergence in
# the real suite header itself (not just a stale hand-transcription of it) fails here. Needs
# `tests` on its include path to resolve the real suite header (matching Gate A's own convention).
add_executable(t2139_gate_c_real_suite_side_check EXCLUDE_FROM_ALL
	tools/t2139_gate_c_real_suite_side_check.cpp)
target_include_directories(t2139_gate_c_real_suite_side_check PRIVATE include tests)

# S4's owed must-reject constructions (same casebook): one per NEW mechanism this round's own Gate
# C remedy added, each exercising the REAL SSLM_STATUS_ENUM_LIST/SSLM_STATUS_NEXT_FREE machinery
# rather than a toy hand-written enum (tools/t2139_gate_c_type_identity_check_negative.cpp's own
# sibling, unchanged, still covers the generic "does static_assert work" case).
add_executable(t2139_gate_c_xmacro_must_reject EXCLUDE_FROM_ALL
	tools/t2139_gate_c_xmacro_check_negative.cpp)
target_include_directories(t2139_gate_c_xmacro_must_reject PRIVATE include)

add_executable(t2139_gate_c_sentinel_must_reject EXCLUDE_FROM_ALL
	tools/t2139_gate_c_sentinel_negative.cpp)
target_include_directories(t2139_gate_c_sentinel_must_reject PRIVATE include)

# T-2141 (Claude/Poirot/3bcbe43-t2139-fourth-confirmation-review.md O3; Claude/Zelda/Board.md
# T-2141 row): the t2138 red suite's own sslm_abi.h enum copy (tests/t2138-abi-red-suite/
# sslm_abi.h) was guarded only by the 564-cell link, never by a compile-time parity check --
# that file's own header comment names the gap. This must-accept/must-reject pair closes it for
# sslm_status, same generated-per-name-plus-sentinel shape as t2139_gate_c_type_identity_check
# above, but reading the REAL suite header (namespaced, to avoid the extern "C" function-name
# collision the real library header's own SUPERSLM_ABI_ENUM_ONLY escape hatch exists for --
# executed and confirmed, see tools/t2141_gate_c_t2138_suite_side_check.cpp's own header
# comment for the exact collision this construction hit and routed around).
add_executable(t2141_gate_c_t2138_suite_side_must_accept EXCLUDE_FROM_ALL
	tools/t2141_gate_c_t2138_suite_side_check.cpp)
target_include_directories(t2141_gate_c_t2138_suite_side_must_accept PRIVATE include tests)

add_executable(t2141_gate_c_t2138_suite_side_must_reject EXCLUDE_FROM_ALL
	tools/t2141_gate_c_t2138_suite_side_check_negative.cpp)
target_include_directories(t2141_gate_c_t2138_suite_side_must_reject PRIVATE include)

enable_testing()
add_test(NAME t2139_gate_a_must_accept_compiles
	COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target t2139_gate_a_must_accept --config $<CONFIG>)
# T-2139 sixth confirmation review (Claude/Poirot/5fbd04d-t2139-sixth-confirmation-review.md S1):
# these two were still WILL_FAIL-only (exit-code-only) after S3 gave the X-macro/sentinel pair
# below the marker-text discrimination -- the same class the fifth casebook's S3 named in class
# terms, not instance terms. Both now go through tools/ci/check_negative_compile_marker.cmake with
# their own static_assert marker text, matching their two siblings' convention.
add_test(NAME t2139_gate_a_must_reject_fails_to_compile
	COMMAND ${CMAKE_COMMAND}
		-DBUILD_DIR=${CMAKE_BINARY_DIR} -DBUILD_TARGET=t2139_gate_a_must_reject -DBUILD_CONFIG=$<CONFIG>
		-DSOURCE_FILE=tools/t2139_gate_a_header_parity_check_negative.cpp
		-DMARKER_TEXT=sslm_model_map:\ library\ signature\ diverges\ from\ tests/t2130-g5-red-suite/sslm_g5.h
		-P ${CMAKE_SOURCE_DIR}/tools/ci/check_negative_compile_marker.cmake)
add_test(NAME t2139_gate_c_must_reject_fails_to_compile
	COMMAND ${CMAKE_COMMAND}
		-DBUILD_DIR=${CMAKE_BINARY_DIR} -DBUILD_TARGET=t2139_gate_c_must_reject -DBUILD_CONFIG=$<CONFIG>
		-DSOURCE_FILE=tools/t2139_gate_c_type_identity_check_negative.cpp
		-DMARKER_TEXT=SSLM_ARTIFACT_REJECTED\ diverges
		-P ${CMAKE_SOURCE_DIR}/tools/ci/check_negative_compile_marker.cmake)
# Gate C must-accept: a plain must-pass compile check, matching its Gate A sibling. The known
# ordinal-interleaving block this entry previously disclosed (P2, Claude/Poirot/
# 2c18dab-t2139-abi-build-review.md Sec7.4) closed when sslm_g5.h took the T-2133 Sec6 registry
# mirror (curie/t2130-g5-red-suite@52dc6cd): sslm_g5.h now mirrors sslm_status's full 26-entry
# ordinal registry verbatim, so the type-identity construction compiles and this entry must pass.
add_test(NAME t2139_gate_c_must_accept_compiles
	COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target t2139_gate_c_must_accept --config $<CONFIG>)

# S1's structural remedy: the third TU is a plain must-pass compile check, matching its two
# siblings above.
add_test(NAME t2139_gate_c_real_suite_side_check_compiles
	COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target t2139_gate_c_real_suite_side_check --config $<CONFIG>)

# S4's owed must-reject constructions: both MUST fail to compile, same convention as
# t2139_gate_a_must_reject_fails_to_compile and t2139_gate_c_must_reject_fails_to_compile above
# (all four negative-compile constructions now run through the marker script; see S1 comment
# above them for the finding that made this true of all four rather than just these two).
#
# T-2139 fifth confirmation review (Claude/Poirot/ce5aff2-t2139-fifth-confirmation-review.md S3):
# a plain WILL_FAIL entry only inverts the build's exit code -- it cannot tell the construction's
# own assertion firing apart from any other compile failure, and the sentinel construction proved
# this live (one governed append to both real headers desynchronizes its own extra enumerator from
# the real sentinels, so it still fails to compile, but for an unrelated C2039 name-lookup reason,
# with WILL_FAIL still reporting green). tools/ci/check_negative_compile_marker.cmake requires BOTH
# the build failing AND the target's own static_assert marker text appearing in the captured
# output, so a degeneration of the underlying mechanism is a hard ctest failure instead of a silent
# pass. WILL_FAIL must never be set on an entry that invokes this script -- see the script's own
# header comment for why (T-2139 sixth confirmation review S2): the script's exit code is already
# correct (0 only when both conditions hold), and WILL_FAIL on top of it inverts the check.
# T-2142 (Claude/Poirot/aea6116-t2139-seventh-confirmation-review.md M1, the residual half): both
# MARKER_TEXT values below were TRUNCATED fragments of their construction's own full
# static_assert message until this fix -- reproduced and confirmed before fixing
# (tools/ci/verify_negative_compile_marker_source.py against each source file with the OLD
# truncated text: FAILED, correctly). The xmacro entry was missing its macro-generated suffix
# " (deliberate corruption, must-reject construction)"; the sentinel entry was missing its
# literal prefix "registry-top divergence (deliberate one-sided append, must-reject
# construction): the two ". Both passed the compiler-output substring check vacuously the whole
# time -- a truncated fragment of a real message IS a substring of that message -- which is
# exactly the class check_negative_compile_marker.cmake's own new SOURCE_FILE argument now closes
# structurally rather than by a fourth careful transcription.
add_test(NAME t2139_gate_c_xmacro_must_reject_fails_to_compile
	COMMAND ${CMAKE_COMMAND}
		-DBUILD_DIR=${CMAKE_BINARY_DIR} -DBUILD_TARGET=t2139_gate_c_xmacro_must_reject -DBUILD_CONFIG=$<CONFIG>
		-DSOURCE_FILE=tools/t2139_gate_c_xmacro_check_negative.cpp
		-DMARKER_TEXT=SSLM_ARTIFACT_REJECTED\ diverges\ \(deliberate\ corruption,\ must-reject\ construction\)
		-P ${CMAKE_SOURCE_DIR}/tools/ci/check_negative_compile_marker.cmake)
add_test(NAME t2139_gate_c_sentinel_must_reject_fails_to_compile
	COMMAND ${CMAKE_COMMAND}
		-DBUILD_DIR=${CMAKE_BINARY_DIR} -DBUILD_TARGET=t2139_gate_c_sentinel_must_reject -DBUILD_CONFIG=$<CONFIG>
		-DSOURCE_FILE=tools/t2139_gate_c_sentinel_negative.cpp
		"-DMARKER_TEXT=registry-top divergence (deliberate one-sided append, must-reject construction): the two SSLM_STATUS_NEXT_FREE sentinels no longer agree"
		-P ${CMAKE_SOURCE_DIR}/tools/ci/check_negative_compile_marker.cmake)

# T-2141's must-accept/must-reject pair (see the add_executable block above for the construction
# itself): must-accept is a plain must-pass compile check, matching every Gate A/C sibling's own
# convention; must-reject goes through the same marker script as every other must-reject entry.
add_test(NAME t2141_gate_c_t2138_suite_side_must_accept_compiles
	COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target t2141_gate_c_t2138_suite_side_must_accept --config $<CONFIG>)
add_test(NAME t2141_gate_c_t2138_suite_side_must_reject_fails_to_compile
	COMMAND ${CMAKE_COMMAND}
		-DBUILD_DIR=${CMAKE_BINARY_DIR} -DBUILD_TARGET=t2141_gate_c_t2138_suite_side_must_reject -DBUILD_CONFIG=$<CONFIG>
		-DSOURCE_FILE=tools/t2141_gate_c_t2138_suite_side_check_negative.cpp
		-DMARKER_TEXT=SSLM_ARTIFACT_REJECTED\ diverges\ \(deliberate\ corruption,\ must-reject\ construction\)
		-P ${CMAKE_SOURCE_DIR}/tools/ci/check_negative_compile_marker.cmake)

# T-2115 (D-SLM3432)/T-2152 (outside strike item 1): the GPU acceleration sources
# (src/gpu/gpu_1p0.cpp, src/gpu/superslm_gpu.cpp) and their compiled shaders are
# D3D12/HLSL and Windows-only -- build.bat is the build recipe that has always
# compiled and shaded them, UNCONDITIONALLY (dxc.exe is build.bat's own first,
# non-optional step). T-2115 originally gated shader compilation itself behind
# SUPERSLM_BUILD_GPU, matching a superslm_gpu STATIC library that was ALSO fully
# opt-in -- correct for that library (a consumer who never wants GPU acceleration
# should not need dxc.exe installed to link the CPU-only product), wrong for
# tests/test_main.cpp, which calls `superslm_gpu::*` directly and unconditionally
# on WIN32 (see the `if(WIN32)` block on the superslm_tests target above) and so
# needs the SAME compiled shaders at test-run time on every Windows configure,
# not only an opted-in one. Split here: shader compilation (this block) now runs
# on every WIN32 configure, matching build.bat's own unconditional posture and
# giving superslm_tests real shaders to dispatch against; the PUBLIC, installable
# superslm_gpu library stays under SUPERSLM_BUILD_GPU below (a consumer who wants
# only the CPU-only product library still never links GPU code, only now compiles
# its shaders as a byproduct of being on Windows -- see the version/dxc checks
# just below for what that costs, and why it is honest cost rather than a papered
# gap: it is exactly what build.bat has always required).
option(SUPERSLM_BUILD_GPU "Build and install the D3D12/HLSL GPU acceleration library (Windows/MSVC only)" OFF)

if(SUPERSLM_BUILD_GPU AND NOT WIN32)
	message(FATAL_ERROR "SUPERSLM_BUILD_GPU requires Windows: src/gpu/*.cpp is a D3D12/HLSL implementation with no non-Windows backend.")
endif()

if(WIN32)
	# T-2115 round 4 (NEW-3, Claude/Poirot/6f60491-t2115-cmake-product-
	# surface.md, second confirmation), narrowed by T-2152: the project's own
	# declared floor is 3.16 for Linux/macOS, which never reach this block at
	# all (CPU-only Layer 1 embeddability, D-SLM13, stays reachable there on
	# CMake 3.16-3.19). On WIN32 specifically, shader compilation below needs
	# 3.20: SUPERSLM_GPU_SHADER_OUT_DIR carries a $<CONFIG> generator
	# expression on multi-config generators, and that string reaches
	# add_custom_command(OUTPUT ...) -- CMake's own Help/command/
	# add_custom_command.rst documents generator expressions in OUTPUT as
	# `.. versionadded:: 3.20`. Below 3.20 the genex is not expanded, so the
	# build would attempt to create and write into a directory literally
	# named "$<CONFIG>" -- an illegible path error (`<`/`>` are invalid
	# Windows path characters) instead of the legible one-line refusal
	# cmake_minimum_required exists to give. This is now the real floor for
	# ANY Windows configure (T-2152: shader compilation is no longer gated
	# behind SUPERSLM_BUILD_GPU, because tests/test_main.cpp needs it
	# unconditionally on WIN32) -- Linux/macOS are unaffected.
	if(CMAKE_VERSION VERSION_LESS "3.20")
		message(FATAL_ERROR "A Windows configure of this project requires CMake 3.20 or higher: tests/test_main.cpp calls superslm_gpu::* directly and unconditionally on WIN32 (T-2152), so the GPU shader compilation below -- which needs CMake 3.20's own generator-expression support in add_custom_command(OUTPUT ...), CMake Help/command/add_custom_command.rst -- now runs on every Windows configure, not only an opted-in SUPERSLM_BUILD_GPU=ON one. Linux/macOS configures are unaffected and still work on CMake 3.16+.")
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
		message(FATAL_ERROR "dxc.exe (DirectX Shader Compiler) was not found on PATH or under any installed Windows Kits 10 SDK. A Windows configure of this project needs it unconditionally (T-2152: tests/test_main.cpp calls superslm_gpu::* directly on WIN32, so its shaders must exist to run the test suite, matching build.bat's own always-on requirement) -- install the Windows SDK, or pass -DSUPERSLM_DXC_EXECUTABLE=<path to dxc.exe>.")
	endif()
	message(STATUS "SuperSLM GPU: compiling shaders with ${SUPERSLM_DXC_EXECUTABLE}")

	# Every compute shader compiles to cs_6_2, -O3, -HV 2018, -WX -- the same
	# pinned target and flags build.bat uses (T-1986 GPU-serial port, Sec5.7).
	# superslm_gpu::harness::ShaderPath() (src/gpu/superslm_gpu.cpp) resolves
	# "<exe_dir>/shaders/<name>.cso" from GetModuleFileNameA at runtime -- the
	# directory beside whatever executable is actually running, not a fixed
	# build-tree path. superslm_tests.exe lands directly in ${CMAKE_BINARY_DIR}
	# on a single-config generator (Ninja, Makefiles) and in
	# ${CMAKE_BINARY_DIR}/$<CONFIG>/ on a multi-config one (Visual Studio), so
	# the shader output directory below carries the same $<CONFIG> generator
	# expression on multi-config generators and resolves to
	# ${CMAKE_BINARY_DIR}/shaders on a single-config one -- matching
	# build.bat's own out\shaders-beside-out\*.exe layout in both cases, and
	# now the directory superslm_tests.exe itself looks in at runtime, not
	# only a hypothetical future in-tree GPU-dispatching executable's.
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
	# 3.20 addition, a feature this file does not use). Avoided entirely by
	# never putting a generator expression in an install(FILES ...) list: this
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

	# T-2152: superslm_tests dispatches real GPU shaders on WIN32 (the T-2019
	# GPU-serial-port suite) and must not start until they exist -- matching
	# build.bat's own ordering (dxc runs before the test binary compiles).
	add_dependencies(superslm_tests superslm_gpu_shaders)

	if(SUPERSLM_BUILD_GPU)
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

# T-2150 (D-SLM3423 scrub gate, P2-8): gpu_1p0_bench_bridge.h is a build-seat-only bench
# accessor header (its own banner: "NOT part of the shipped 1.0 API surface"; its only
# intended includers are tools/t2113_b3_sequence_smoke.cpp and the other in-tree diagnostic/
# bench tools, none of which are installed) -- it stays in the source tree for those tools to
# include, but is excluded from the public install surface.
install(DIRECTORY include/superslm/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/superslm
	PATTERN "gpu_1p0_bench_bridge.h" EXCLUDE
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
