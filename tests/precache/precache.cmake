set(PC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/precache)
file(GLOB PC_SOURCES CONFIGURE_DEPENDS ${PC_DIR}/e*.cpp)
add_executable(superslm_precache_e1 EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/paged-kv/pkv_main.cpp
    ${PC_DIR}/precache_stubs.cpp ${PC_SOURCES})
target_link_libraries(superslm_precache_e1 PRIVATE superslm_test_injection)
target_include_directories(superslm_precache_e1 PRIVATE include src tests tests/paged-kv ${PC_DIR})
target_compile_definitions(superslm_precache_e1 PRIVATE
    PKV_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/paged-kv/reference")
if(MSVC)
    target_compile_options(superslm_precache_e1 PRIVATE /W4 /fp:precise)
else()
    target_compile_options(superslm_precache_e1 PRIVATE -Wall -Wextra -ffp-contract=off)
endif()
