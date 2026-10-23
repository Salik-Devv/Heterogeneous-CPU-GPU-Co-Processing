cmake_minimum_required(VERSION 3.18)
project(gemm_hetero LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(OpenMP REQUIRED)

add_executable(gemm_hetero
    gemm_hetero.cu
    cpu_part.cpp
)

target_compile_options(gemm_hetero PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_link_libraries(gemm_hetero PRIVATE
    OpenMP::OpenMP_CXX
    pthread
)

set_target_properties(gemm_hetero PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)