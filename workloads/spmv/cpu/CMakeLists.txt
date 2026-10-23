cmake_minimum_required(VERSION 3.18)

project(spmv_cpu LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

# -------------------------
# Serial CPU SpMV
# -------------------------
add_executable(spmv_cpu_serial
    spmv_cpu_serial.cpp
)

target_compile_options(spmv_cpu_serial PRIVATE
    -O3
    -march=native
)

# -------------------------
# OpenMP CPU SpMV
# -------------------------
add_executable(spmv_cpu_omp
    spmv_cpu_omp.cpp
)

target_compile_options(spmv_cpu_omp PRIVATE
    -O3
    -march=native
)

target_link_libraries(spmv_cpu_omp PRIVATE OpenMP::OpenMP_CXX)

# -------------------------
# Output binaries
# -------------------------
set_target_properties(spmv_cpu_serial spmv_cpu_omp
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)