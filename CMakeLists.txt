cmake_minimum_required(VERSION 3.10)
project(MatrixMultCUDA CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

cuda_add_executable(matrix_mult_cuda matrix_mult_cuda.cu)
cuda_add_executable(perf_test_cuda perf_test_cuda.cu)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CUDA_NVCC_FLAGS "-O3 --use_fast_math")