cmake_minimum_required(VERSION 3.18)

project(spmv_hetero LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# -------------------------
# CPU+GPU Heterogeneous SpMV
# -------------------------
add_executable(spmv_hetero
    spmv_hetero.cu
    spmv_part.cpp
)

set_target_properties(spmv_hetero PROPERTIES
    CUDA_ARCHITECTURES 89
)

target_compile_options(spmv_hetero PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

target_link_libraries(spmv_hetero PRIVATE
    OpenMP::OpenMP_CXX
    Threads::Threads
    m
)

# -------------------------
# Output binary
# -------------------------
set_target_properties(spmv_hetero
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)