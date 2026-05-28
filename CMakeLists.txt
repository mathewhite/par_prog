cmake_minimum_required(VERSION 3.10)
project(MatrixMultOpenMP CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPENMP_PREFIX "/opt/homebrew/opt/libomp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I${OPENMP_PREFIX}/include")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${OPENMP_PREFIX}/lib -lomp")

add_executable(matrix_mult_omp matrix_mult_omp.cpp)
add_executable(perf_test_omp perf_test_omp.cpp)