#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using Matrix = std::vector<std::vector<double>>;
using Clock = std::chrono::high_resolution_clock;

Matrix gen_rand(int n) {
    Matrix m(n, std::vector<double>(n));
    for(auto& r : m) for(double& v : r) v = (rand() % 10) + 1;
    return m;
}

double run_test(const Matrix& A, const Matrix& B, int threads) {
    int n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));
    
#ifdef _OPENMP
    omp_set_num_threads(threads);
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0;
            for (int k = 0; k < n; ++k) s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
    }
    // Возвращаем время только для примера, в реальности замеряем снаружи
    return 0; 
}

int main() {
    std::vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    std::vector<int> threads_list = {1, 2, 4, 8};
    
    std::ofstream csv("perf_data.csv");
    csv << "Size,Threads,Time_ms,GFLOPS,Speedup,Efficiency\n";

    std::cout << "Starting Performance Benchmark...\n";
    std::cout << "Size\tThreads\tTime(ms)\tGFLOPS\tSpeedup\tEff(%)\n";

    for (int N : sizes) {
        // Генерируем данные один раз для размера N
        srand(42); // Фиксируем сид для воспроизводимости
        Matrix A = gen_rand(N);
        Matrix B = gen_rand(N);
        
        long long ops = 2LL * N * N * N;
        double base_time = 0.0;

        for (int T : threads_list) {
#ifdef _OPENMP
            if (T > omp_get_max_threads()) continue;
#endif

            // Замер времени
            auto start = Clock::now();
            // Вызываем функцию умножения (код встроен для простоты или вызов функции)
            // Здесь используем упрощенный вызов для замера
            Matrix C(N, std::vector<double>(N, 0.0));
            
#ifdef _OPENMP
            omp_set_num_threads(T);
            #pragma omp parallel for collapse(2) schedule(static)
#endif
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < N; ++k) {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] = sum;
                }
            }
            
            auto end = Clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            // Для первого потока (база) запоминаем время
            if (T == 1) base_time = time_ms;

            double gflops = (ops / 1e9) / (time_ms / 1000.0);
            double speedup = base_time / time_ms;
            double efficiency = (speedup / T) * 100.0;

            std::cout << N << "\t" << T << "\t" 
                      << std::fixed << std::setprecision(2) << time_ms << "\t"
                      << gflops << "\t"
                      << speedup << "\t"
                      << efficiency << "\n";
            
            csv << N << "," << T << "," << time_ms << "," << gflops << "," << speedup << "," << efficiency << "\n";
        }
        std::cout << "---\n";
    }
    csv.close();
    std::cout << "Data saved to perf_data.csv\n";
    return 0;
}