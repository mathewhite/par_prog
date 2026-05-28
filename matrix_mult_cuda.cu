#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdexcept>

// Конфигурация блоков по умолчанию
#define DEFAULT_TILE_SIZE 16

using Matrix = std::vector<std::vector<double>>;

// --- CUDA Kernel ---

__global__ void matmul_kernel(const double* __restrict__ A, 
                              const double* __restrict__ B, 
                              double* __restrict__ C, 
                              int n) {
    // Вычисление глобальных индексов строки и столбца
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            // Row-major access: A[row][k] -> A[row * n + k]
            // Col-major access for B would be better for coalescing, but keeping simple O(N^3) logic
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// --- Helper Functions ---

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

Matrix load_matrix_from_file(const std::string& path, int& dim) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    file >> dim;
    Matrix mat(dim, std::vector<double>(dim));
    
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (!(file >> mat[i][j])) {
                throw std::runtime_error("Invalid data format in " + path);
            }
        }
    }
    return mat;
}

void save_matrix_to_file(const std::string& path, const Matrix& mat) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + path);
    }

    int n = static_cast<int>(mat.size());
    file << n << "\n";
    file << std::fixed << std::setprecision(6);

    for (const auto& row : mat) {
        for (double val : row) {
            file << val << " ";
        }
        file << "\n";
    }
}

Matrix generate_random_matrix(int n) {
    Matrix mat(n, std::vector<double>(n));
    for (auto& row : mat) {
        for (double& val : row) {
            val = static_cast<double>(std::rand() % 10) + 1.0;
        }
    }
    return mat;
}

Matrix cpu_multiply(const Matrix& A, const Matrix& B) {
    int n = static_cast<int>(A.size());
    Matrix C(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double a_val = A[i][k];
            for (int j = 0; j < n; ++j) {
                C[i][j] += a_val * B[k][j];
            }
        }
    }
    return C;
}

// --- Main Execution Logic ---

void run_benchmark(int N) {
    std::cout << "\n>>> CUDA Benchmark: N=" << N << " <<<" << std::endl;
    
    // Генерация данных
    Matrix A = generate_random_matrix(N);
    Matrix B = generate_random_matrix(N);
    
    // CPU Baseline
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    Matrix C_cpu = cpu_multiply(A, B);
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(t_cpu_end - t_cpu_start).count();
    
    std::cout << "CPU Time: " << std::fixed << std::setprecision(2) << cpu_time_ms << " ms" << std::endl;

    // Подготовка данных для GPU
    size_t bytes = N * N * sizeof(double);
    double *d_A, *d_B, *d_C;
    
    check_cuda_error(cudaMalloc(&d_A, bytes), "Malloc A");
    check_cuda_error(cudaMalloc(&d_B, bytes), "Malloc B");
    check_cuda_error(cudaMalloc(&d_C, bytes), "Malloc C");

    // Flatten matrices for transfer
    std::vector<double> h_A_flat(N*N), h_B_flat(N*N);
    for(int i=0; i<N; ++i)
        for(int j=0; j<N; ++j) {
            h_A_flat[i*N + j] = A[i][j];
            h_B_flat[i*N + j] = B[i][j];
        }

    check_cuda_error(cudaMemcpy(d_A, h_A_flat.data(), bytes, cudaMemcpyHostToDevice), "Memcpy H->D A");
    check_cuda_error(cudaMemcpy(d_B, h_B_flat.data(), bytes, cudaMemcpyHostToDevice), "Memcpy H->D B");

    // Тестирование разных размеров блоков
    std::vector<int> tile_sizes = {8, 16, 32};
    
    std::cout << "\nTile Size | GPU Time (ms) | Speedup   | GFLOPS" << std::endl;
    std::cout << "----------|---------------|-----------|-------" << std::endl;

    for (int tile : tile_sizes) {
        dim3 block(tile, tile);
        dim3 grid((N + tile - 1) / tile, (N + tile - 1) / tile);

        // CUDA Events for precise timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Запуск ядра
        cudaEventRecord(start);
        matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        
        // Проверка ошибок запуска
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Расчет метрик
        long long ops = 2LL * N * N * N;
        double speedup = cpu_time_ms / milliseconds;
        double gflops = (ops / 1e9) / (milliseconds / 1000.0);

        std::cout << std::setw(9) << tile << "x" << std::setw(1) << tile 
                  << " | " << std::setw(13) << std::fixed << std::setprecision(3) << milliseconds 
                  << " | " << std::setw(9) << std::setprecision(1) << speedup << "x"
                  << " | " << std::setw(5) << std::setprecision(1) << gflops << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Очистка памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char* argv[]) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    try {
        if (argc == 3 && std::string(argv[1]) == "--bench") {
            int N = std::stoi(argv[2]);
            run_benchmark(N);
        } 
        else if (argc == 5) {
            // Режим: файл_А файл_В файл_результат размер_блока
            std::string fileA = argv[1];
            std::string fileB = argv[2];
            std::string fileOut = argv[3];
            int tile_size = std::stoi(argv[4]);

            int N;
            Matrix A = load_matrix_from_file(fileA, N);
            Matrix B = load_matrix_from_file(fileB, N); // Предполагаем квадратные матрицы одинакового размера

            size_t bytes = N * N * sizeof(double);
            double *d_A, *d_B, *d_C;
            
            check_cuda_error(cudaMalloc(&d_A, bytes), "Malloc A");
            check_cuda_error(cudaMalloc(&d_B, bytes), "Malloc B");
            check_cuda_error(cudaMalloc(&d_C, bytes), "Malloc C");

            std::vector<double> h_A_flat(N*N), h_B_flat(N*N);
            for(int i=0; i<N; ++i)
                for(int j=0; j<N; ++j) {
                    h_A_flat[i*N + j] = A[i][j];
                    h_B_flat[i*N + j] = B[i][j];
                }

            check_cuda_error(cudaMemcpy(d_A, h_A_flat.data(), bytes, cudaMemcpyHostToDevice), "Memcpy H->D");
            check_cuda_error(cudaMemcpy(d_B, h_B_flat.data(), bytes, cudaMemcpyHostToDevice), "Memcpy H->D");

            dim3 block(tile_size, tile_size);
            dim3 grid((N + tile_size - 1) / tile_size, (N + tile_size - 1) / tile_size);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);

            std::cout << "GPU Execution Time: " << ms << " ms" << std::endl;

            // Копирование результата обратно
            std::vector<double> h_C_flat(N*N);
            check_cuda_error(cudaMemcpy(h_C_flat.data(), d_C, bytes, cudaMemcpyDeviceToHost), "Memcpy D->H");

            Matrix C(N, std::vector<double>(N));
            for(int i=0; i<N; ++i)
                for(int j=0; j<N; ++j)
                    C[i][j] = h_C_flat[i*N + j];

            save_matrix_to_file(fileOut, C);
            std::cout << "Result saved to: " << fileOut << std::endl;

            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            cudaEventDestroy(start); cudaEventDestroy(stop);
        } 
        else {
            std::cerr << "Usage:\n"
                      << "  " << argv[0] << " --bench N          # Run benchmark for size N\n"
                      << "  " << argv[0] << " A.txt B.txt out.txt TILE_SIZE\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}