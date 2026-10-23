cmake_minimum_required(VERSION 3.18)
project(prefix_sum_gpu CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(prefix_sum_gpu prefix_sum_gpu.cu)

target_link_libraries(prefix_sum_gpu PRIVATE m)

set_target_properties(prefix_sum_gpu PROPERTIES
    CUDA_ARCHITECTURES 75
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)