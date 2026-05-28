cmake_minimum_required(VERSION 3.10)
project(MatrixMultMPI CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(matrix_mult_mpi matrix_mult_mpi.cpp)
target_link_libraries(matrix_mult_mpi MPI::MPI_CXX)

add_executable(perf_test_mpi perf_test_mpi.cpp)
target_link_libraries(perf_test_mpi MPI::MPI_CXX)