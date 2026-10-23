cmake_minimum_required(VERSION 3.18)
project(prefix_sum_cpu CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ── Explicitly name each target's source — no ambiguity ──
add_executable(prefix_sum_cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/prefix_sum_cpu.cpp
)

add_executable(prefix_sum_cpu_omp
    ${CMAKE_CURRENT_SOURCE_DIR}/prefix_sum_cpu_omp.cpp
)

target_link_libraries(prefix_sum_cpu
    PRIVATE m
)

target_link_libraries(prefix_sum_cpu_omp
    PRIVATE
    OpenMP::OpenMP_CXX
    m
)

set_target_properties(prefix_sum_cpu prefix_sum_cpu_omp
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)