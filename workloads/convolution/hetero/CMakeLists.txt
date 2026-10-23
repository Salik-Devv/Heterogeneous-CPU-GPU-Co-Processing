cmake_minimum_required(VERSION 3.18)
project(convolution_hetero CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(conv_hetero
    conv_hetero.cu
    conv_part.cpp
)

target_link_libraries(conv_hetero PRIVATE
    OpenMP::OpenMP_CXX
    pthread
    m
)

target_compile_options(conv_hetero PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=${OpenMP_CXX_FLAGS}
        -Xcudafe "--diag_suppress=550"
    >
)

set_target_properties(conv_hetero PROPERTIES
    CUDA_ARCHITECTURES 75
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)