cmake_minimum_required(VERSION 3.18)
project(prefix_sum_hetero CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(prefix_sum_hetero
    prefix_sum_hetero.cu
    prefix_part.cpp
)

target_link_libraries(prefix_sum_hetero PRIVATE
    OpenMP::OpenMP_CXX
    pthread
    m
)

target_compile_options(prefix_sum_hetero PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
)

set_target_properties(prefix_sum_hetero PROPERTIES
    CUDA_ARCHITECTURES 75
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)