cmake_minimum_required(VERSION 3.18)

project(gemm_cpu LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

# -------------------------
# Serial CPU GEMM
# -------------------------
add_executable(gemm_cpu
    gemm_cpu.cpp
)

target_compile_options(gemm_cpu PRIVATE
    -O3
    -march=native
)

# -------------------------
# OpenMP CPU GEMM
# -------------------------
add_executable(gemm_cpu_omp
    gemm_cpu_omp.cpp
)

target_compile_options(gemm_cpu_omp PRIVATE
    -O3
    -march=native
)

target_link_libraries(gemm_cpu_omp PRIVATE OpenMP::OpenMP_CXX)

# -------------------------
# Output binaries
# -------------------------
set_target_properties(gemm_cpu gemm_cpu_omp
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)