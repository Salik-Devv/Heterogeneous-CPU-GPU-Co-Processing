cmake_minimum_required(VERSION 3.18)
project(convolution_gpu CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(conv_gpu conv_gpu.cu)

target_link_libraries(conv_gpu PRIVATE m)

set_target_properties(conv_gpu PROPERTIES
    CUDA_ARCHITECTURES 75
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ── Suppress stb_image.h internal warnings only ──
target_compile_options(conv_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcudafe "--diag_suppress=550"
    >
)