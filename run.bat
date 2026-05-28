@echo off
echo === OpenMP Matrix Multiplication Benchmark ===
echo.

if not exist build mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

echo.
echo Running benchmark...
Release\benchmark_omp.exe

echo.
echo Verification for 100x100 matrix...
Release\matrix_mult_omp.exe -t 100
python ..\verify_omp.py verify_A.txt verify_B.txt verify_C.txt

echo.
echo Results:
type benchmark_results.csv