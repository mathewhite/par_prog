@echo off
echo ========================================
echo    CUDA Matrix Multiplication Build & Run
echo ========================================
echo.

REM Check if NVCC is available
nvcc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] NVCC compiler not found in PATH. Please install CUDA Toolkit.
    pause
    exit /b 1
)

REM Create build directory
if not exist build mkdir build
cd build

echo [1/3] Compiling main application...
nvcc -O3 -o matrix_mult_cuda.exe ..\matrix_mult_cuda.cu
if %errorlevel% neq 0 (
    echo [ERROR] Compilation of matrix_mult_cuda.cu failed.
    cd ..
    pause
    exit /b 1
)

echo [2/3] Compiling performance test...
nvcc -O3 -o perf_test_cuda.exe ..\perf_test_cuda.cu
if %errorlevel% neq 0 (
    echo [ERROR] Compilation of perf_test_cuda.cu failed.
    cd ..
    pause
    exit /b 1
)

cd ..

echo.
echo [3/3] Running benchmarks for sizes 200, 400, 800...
echo.

for %%s in (200 400 800) do (
    echo --- Testing N=%%s ---
    build\matrix_mult_cuda.exe --bench %%s
    echo.
)

echo.
echo ========================================
echo    Benchmarks Complete.
echo ========================================
pause