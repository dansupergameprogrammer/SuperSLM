cmake_minimum_required(VERSION 3.16)
project(superslm_install_consumer LANGUAGES CXX)

# Deliberately do not set CMAKE_CXX_STANDARD or a target CXX_STANDARD. The
# installed product target must carry its own public C++20 requirement.
find_package(superslm 1.4.0 REQUIRED CONFIG)

get_target_property(_superslm_features superslm::superslm INTERFACE_COMPILE_FEATURES)
if(NOT "cxx_std_20" IN_LIST _superslm_features)
	message(FATAL_ERROR "installed superslm::superslm does not propagate cxx_std_20")
endif()

add_executable(headers_cpu_first headers_cpu_first.cpp)
target_link_libraries(headers_cpu_first PRIVATE superslm::superslm)

add_executable(headers_gpu_first headers_gpu_first.cpp)
target_link_libraries(headers_gpu_first PRIVATE superslm::superslm)

if(TARGET superslm::superslm_gpu)
	target_link_libraries(headers_cpu_first PRIVATE superslm::superslm_gpu)
	superslm_deploy_gpu_shaders(headers_cpu_first)
endif()
