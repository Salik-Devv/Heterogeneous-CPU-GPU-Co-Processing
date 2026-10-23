cmake_minimum_required(VERSION 3.18)
project(convolution_cpu CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

# stb is header-only — just needs to be in the include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(conv_cpu     conv_cpu.cpp)
add_executable(conv_cpu_omp conv_cpu_omp.cpp)

target_link_libraries(conv_cpu     PRIVATE m)
target_link_libraries(conv_cpu_omp PRIVATE OpenMP::OpenMP_CXX m)

set_target_properties(conv_cpu conv_cpu_omp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)