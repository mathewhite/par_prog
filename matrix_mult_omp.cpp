#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// Алиасы для удобства
using Matrix = std::vector<std::vector<double>>;
using Clock = std::chrono::high_resolution_clock;

// --- I/O Functions ---

Matrix load_matrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int n;
    file >> n;
    
    Matrix mat(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file >> mat[i][j];
        }
    }
    return mat;
}

void save_matrix(const std::string& filename, const Matrix& mat) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + filename);
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
            val = static_cast<double>(std::rand() % 10);
        }
    }
    return mat;
}

// --- Core Algorithm ---

Matrix multiply_parallel(const Matrix& A, const Matrix& B, int threads) {
    int n = static_cast<int>(A.size());
    Matrix C(n, std::vector<double>(n, 0.0));

#ifdef _OPENMP
    omp_set_num_threads(threads);
    
    // Параллелизация по внешним циклам
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
#else
    // Fallback если OpenMP не подключен
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
#endif
    return C;
}

// --- Main ---

int main(int argc, char* argv[]) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    try {
        if (argc < 2) {
            std::cerr << "Usage:\n"
                      << "  " << argv[0] << " -b N          # Benchmark mode (random NxN)\n"
                      << "  " << argv[0] << " A.txt B.txt out.txt T  # File mode with T threads\n";
            return 1;
        }

        std::string mode = argv[1];

        if (mode == "-b") {
            // Режим бенчмарка
            if (argc != 3) {
                std::cerr << "Error: Missing size for benchmark.\n";
                return 1;
            }
            int N = std::stoi(argv[2]);
            
            std::cout << "[Benchmark] Generating random matrices " << N << "x" << N << "...\n";
            Matrix A = generate_random_matrix(N);
            Matrix B = generate_random_matrix(N);

            // Сохраняем для верификации
            save_matrix("bench_A.txt", A);
            save_matrix("bench_B.txt", B);

            int max_threads = 1;
#ifdef _OPENMP
            max_threads = omp_get_max_threads();
#endif
            
            std::cout << "[Info] Max available threads: " << max_threads << "\n";
            std::cout << "----------------------------------------\n";
            std::cout << "Threads | Time (ms) | GFLOPS\n";
            std::cout << "----------------------------------------\n";

            // Тестируем разные количества потоков
            std::vector<int> test_threads = {1, 2, 4, 8};
            for (int t : test_threads) {
                if (t > max_threads) continue;

                auto start = Clock::now();
                Matrix C = multiply_parallel(A, B, t);
                auto end = Clock::now();

                long long ops = 2LL * N * N * N;
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                double gflops = (ops / 1e9) / (time_ms / 1000.0);

                std::cout << std::setw(7) << t << " | " 
                          << std::fixed << std::setprecision(2) << std::setw(9) << time_ms 
                          << " | " << std::setw(6) << gflops << "\n";
                
                // Сохраняем результат последнего прогона для проверки
                if (t == test_threads.back() || t == max_threads) {
                     save_matrix("bench_C.txt", C);
                }
            }
            std::cout << "----------------------------------------\n";
        } 
        else {
            // Режим работы с файлами
            if (argc != 5) {
                std::cerr << "Error: Invalid arguments for file mode.\n";
                return 1;
            }
            
            std::string fileA = argv[1];
            std::string fileB = argv[2];
            std::string fileOut = argv[3];
            int threads = std::stoi(argv[4]);

            std::cout << "[File Mode] Loading matrices...\n";
            Matrix A = load_matrix(fileA);
            Matrix B = load_matrix(fileB);

            if (A.size() != B.size()) {
                throw std::invalid_argument("Matrix dimensions mismatch");
            }

            std::cout << "[Process] Multiplying with " << threads << " threads...\n";
            auto start = Clock::now();
            Matrix C = multiply_parallel(A, B, threads);
            auto end = Clock::now();

            save_matrix(fileOut, C);

            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            long long ops = 2LL * A.size() * A.size() * A.size();
            double gflops = (ops / 1e9) / (time_ms / 1000.0);

            std::cout << "[Result] Saved to " << fileOut << "\n";
            std::cout << "Time: " << std::fixed << std::setprecision(3) << time_ms << " ms\n";
            std::cout << "Perf: " << std::setprecision(2) << gflops << " GFLOPS\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}