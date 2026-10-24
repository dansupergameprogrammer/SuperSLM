# Streamed LoRA adapters (plan: superslm-lora-residency rev 1.2): the engine red suite (step L1) and
# its cells (green at L2). Cells read SUPERSLM_PAGED_KV_FIXTURE_DIR (tools/gen_paged_kv_fixture.py).
# EXCLUDE_FROM_ALL, as the paged-KV and precache suites are. lr_alloc_seam.cpp replaces the global
# operator new and delete for this executable only (the allocation seam, plan §8).
set(LR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/lora-residency)
add_executable(superslm_lora_residency_l1 EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/paged-kv/pkv_main.cpp
    ${LR_DIR}/lr_alloc_seam.cpp ${LR_DIR}/l1_engine.cpp)
target_link_libraries(superslm_lora_residency_l1 PRIVATE superslm)
target_include_directories(superslm_lora_residency_l1 PRIVATE include src tests tests/paged-kv ${LR_DIR})
target_compile_definitions(superslm_lora_residency_l1 PRIVATE
    PKV_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/paged-kv/reference"
    LR_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
if(MSVC)
    target_compile_options(superslm_lora_residency_l1 PRIVATE /W4 /fp:precise)
else()
    target_compile_options(superslm_lora_residency_l1 PRIVATE -Wall -Wextra -ffp-contract=off)
    find_package(Threads REQUIRED)
    target_link_libraries(superslm_lora_residency_l1 PRIVATE Threads::Threads)
endif()
