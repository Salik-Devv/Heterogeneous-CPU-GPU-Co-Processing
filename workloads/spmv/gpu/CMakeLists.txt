cmake_minimum_required(VERSION 3.18)

project(spmv_gpu LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -------------------------
# GPU Naive CSR SpMV
# -------------------------
add_executable(spmv_gpu
    spmv_gpu.cu
)

set_target_properties(spmv_gpu PROPERTIES
    CUDA_ARCHITECTURES 89
)

target_compile_options(spmv_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_link_libraries(spmv_gpu PRIVATE m)

# -------------------------
# Output binary
# -------------------------
set_target_properties(spmv_gpu
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)