#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA error in " << #call << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

using Matrix = std::vector<std::vector<double>>;

__global__ void simple_matmul(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        double val = 0.0;
        for (int k = 0; k < n; ++k) {
            val += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = val;
    }
}

Matrix gen_rand(int n) {
    Matrix m(n, std::vector<double>(n));
    for(auto& r : m) for(double& v : r) v = (rand() % 10) + 1;
    return m;
}

double cpu_matmul_time(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double a_val = A[i][k];
            for (int j = 0; j < n; ++j) {
                C[i][j] += a_val * B[k][j];
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::vector<int> dims = {200, 400, 800, 1200, 1600, 2000};
    std::vector<int> tiles = {8, 16, 32};
    
    std::ofstream csv("cuda_perf_data.csv");
    csv << "Size,Tile,CPU_Time_ms,GPU_Time_ms,Speedup,GFLOPS\n";
    
    std::cout << "Starting CUDA Performance Test...\n";
    std::cout << "Size\tTile\tCPU(ms)\tGPU(ms)\tSpeedup\tGFLOPS\n";

    for (int N : dims) {
        srand(42);
        Matrix A = gen_rand(N);
        Matrix B = gen_rand(N);
        
        double cpu_ms = cpu_matmul_time(A, B);
        long long ops = 2LL * N * N * N;
        
        // Flatten host data
        std::vector<double> h_A(N*N), h_B(N*N);
        for(int i=0; i<N; ++i)
            for(int j=0; j<N; ++j) {
                h_A[i*N+j] = A[i][j];
                h_B[i*N+j] = B[i][j];
            }

        // Device buffers
        double *d_A, *d_B, *d_C;
        size_t bytes = N * N * sizeof(double);
        CHECK_CUDA(cudaMalloc(&d_A, bytes));
        CHECK_CUDA(cudaMalloc(&d_B, bytes));
        CHECK_CUDA(cudaMalloc(&d_C, bytes));
        
        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

        for (int T : tiles) {
            dim3 block(T, T);
            dim3 grid((N+T-1)/T, (N+T-1)/T);
            
            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            
            CHECK_CUDA(cudaEventRecord(start));
            simple_matmul<<<grid, block>>>(d_A, d_B, d_C, N);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            
            float gpu_ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
            
            double speedup = cpu_ms / gpu_ms;
            double gflops = (ops / 1e9) / (gpu_ms / 1000.0);
            
            std::cout << N << "\t" << T << "\t" 
                      << std::fixed << std::setprecision(2) << cpu_ms << "\t"
                      << gpu_ms << "\t"
                      << speedup << "\t"
                      << gflops << "\n";
                      
            csv << N << "," << T << "," << cpu_ms << "," << gpu_ms << "," << speedup << "," << gflops << "\n";
            
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
        }
        
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        std::cout << "---\n";
    }
    
    csv.close();
    std::cout << "Data saved to cuda_perf_data.csv\n";
    return 0;
}