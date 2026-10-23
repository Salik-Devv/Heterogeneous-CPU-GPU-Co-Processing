cmake_minimum_required(VERSION 3.18)

# ── Must set CUDA compiler BEFORE project() so CMake registers the language ──
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(CMAKE_CUDA_COMPILER nvcc
        PATHS /usr/local/cuda/bin /usr/cuda/bin
        REQUIRED
    )
endif()

project(gemm_gpu LANGUAGES CXX CUDA)

# ── Standards ─────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── Output binary into build/bin/ (matches other workloads) ──────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── Find CUDA toolkit for cuBLAS ──────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Target ────────────────────────────────────────────────────────────
add_executable(gemm_gpu gemm_gpu.cu)

target_include_directories(gemm_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}       # local headers (cpu_part.h etc.)
    ${CMAKE_CURRENT_SOURCE_DIR}/..    # gemm/ root: init_matrix.h, gemm_csv.h
)

target_link_libraries(gemm_gpu PRIVATE
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(gemm_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)