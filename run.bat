@echo off
echo === MPI Benchmark ===
echo.

if not exist build mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

cd ..
if exist mpi_results.csv del mpi_results.csv

for %%p in (1 2 4 8) do (
    echo Running with %%p processes...
    mpiexec -n %%p build\Release\benchmark_mpi.exe
    echo.
)

echo Results saved to mpi_results.csv
type mpi_results.csv