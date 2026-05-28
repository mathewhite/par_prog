#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <string>
#include <locale>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

using Matrix = std::vector<std::vector<double>>;
using Clock = std::chrono::high_resolution_clock;

// --- File I/O Operations ---

Matrix load_matrix_from_file(const std::string& filename, int& dimension) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[Error] Cannot open file: " << filename << std::endl;
        exit(1);
    }

    file >> dimension;
    Matrix mat(dimension, std::vector<double>(dimension));

    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            if (!(file >> mat[i][j])) {
                std::cerr << "[Error] Invalid data format in " << filename << std::endl;
                exit(1);
            }
        }
    }
    return mat;
}

void save_matrix_to_file(const std::string& filename, const Matrix& mat) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[Error] Cannot create file: " << filename << std::endl;
        exit(1);
    }

    int n = static_cast<int>(mat.size());
    file << n << "\n";
    
    // Настройка точности для единообразия с NumPy (обычно double)
    file << std::fixed << std::setprecision(6);

    for (const auto& row : mat) {
        for (double val : row) {
            file << val << " ";
        }
        file << "\n";
    }
}

// --- Core Logic ---

Matrix generate_random_matrix(int dimension) {
    Matrix mat(dimension, std::vector<double>(dimension));
    for (auto& row : mat) {
        for (double& val : row) {
            // Генерация чисел от 0 до 9
            val = static_cast<double>(rand() % 10);
        }
    }
    return mat;
}

Matrix multiply_matrices(const Matrix& A, const Matrix& B) {
    int n = static_cast<int>(A.size());
    Matrix C(n, std::vector<double>(n, 0.0));

    // Классическое умножение O(N^3)
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double a_ik = A[i][k];
            for (int j = 0; j < n; ++j) {
                C[i][j] += a_ik * B[k][j];
            }
        }
    }
    return C;
}

// --- Reporting ---

void write_report(const std::string& filename, int n, long long duration_us) {
    std::ofstream report(filename);
    if (!report.is_open()) {
        std::cerr << "[Error] Failed to write report to " << filename << std::endl;
        return;
    }

    double duration_ms = duration_us / 1000.0;
    double duration_s = duration_ms / 1000.0;
    
    // 2 операции (умножение + сложение) на каждую ячейку результата N*N, всего N итераций
    long long total_ops = 2LL * n * n * n;
    double mflops = (total_ops / 1e6) / duration_s;

    report << "========================================\n";
    report << "           PERFORMANCE REPORT           \n";
    report << "========================================\n";
    report << "Matrix Size:     " << n << " x " << n << "\n";
    report << "Total Elements:  " << 3 * n * n << " (A+B+C)\n";
    report << "----------------------------------------\n";
    report << "Time (us):       " << duration_us << "\n";
    report << "Time (ms):       " << std::fixed << std::setprecision(3) << duration_ms << "\n";
    report << "Time (s):        " << std::fixed << std::setprecision(6) << duration_s << "\n";
    report << "----------------------------------------\n";
    report << "Operations:      " << total_ops << "\n";
    report << "Performance:     " << std::fixed << std::setprecision(2) << mflops << " MFLOPS\n";
    report << "========================================\n";
}

// --- Main Entry Point ---

int main(int argc, char* argv[]) {
    // Локализация для корректного вывода в консоли Windows/Linux
#ifdef _WIN32
    SetConsoleOutputCP(1251);
    setlocale(LC_ALL, "Russian");
#else
    setlocale(LC_ALL, "ru_RU.UTF-8");
#endif

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Режимы работы
    bool is_random_mode = false;
    int matrix_size = 0;
    std::string file_a, file_b, file_c, report_file;

    // Парсинг аргументов
    if (argc == 1) {
        // Дефолтный тестовый запуск
        is_random_mode = true;
        matrix_size = 5;
        report_file = "results.txt";
        std::cout << "[Mode] Default test run (N=5)" << std::endl;
    } 
    else if (argc == 4 && std::string(argv[1]) == "-r") {
        // Генерация случайных матриц заданного размера
        is_random_mode = true;
        matrix_size = std::stoi(argv[2]);
        report_file = argv[3];
        std::cout << "[Mode] Random generation (N=" << matrix_size << ")" << std::endl;
    } 
    else if (argc == 5) {
        // Работа с файлами
        file_a = argv[1];
        file_b = argv[2];
        file_c = argv[3];
        report_file = argv[4];
        std::cout << "[Mode] File processing" << std::endl;
    } 
    else {
        std::cerr << "\nUsage:\n";
        std::cerr << "  " << argv[0] << "                          # Test run (5x5)\n";
        std::cerr << "  " << argv[0] << " -r <size> <report.txt>   # Random matrices\n";
        std::cerr << "  " << argv[0] << " <A.txt> <B.txt> <C.txt> <report.txt>\n";
        return 1;
    }

    Matrix A, B, C;

    if (is_random_mode) {
        A = generate_random_matrix(matrix_size);
        B = generate_random_matrix(matrix_size);
        
        // Сохраняем входные данные для возможной верификации
        save_matrix_to_file("matrix_A.txt", A);
        save_matrix_to_file("matrix_B.txt", B);
        std::cout << "[Info] Generated input files: matrix_A.txt, matrix_B.txt" << std::endl;
    } else {
        int dim_a, dim_b;
        A = load_matrix_from_file(file_a, dim_a);
        B = load_matrix_from_file(file_b, dim_b);
        
        if (dim_a != dim_b) {
            std::cerr << "[Error] Matrix dimensions mismatch (" << dim_a << " vs " << dim_b << ")" << std::endl;
            return 1;
        }
        matrix_size = dim_a;
    }

    std::cout << "[Process] Multiplying matrices... " << std::flush;
    
    auto start_time = Clock::now();
    C = multiply_matrices(A, B);
    auto end_time = Clock::now();
    
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    std::cout << "Done." << std::endl;

    // Сохранение результата
    if (is_random_mode) {
        save_matrix_to_file("matrix_C.txt", C);
        std::cout << "[Info] Result saved to: matrix_C.txt" << std::endl;
    } else {
        save_matrix_to_file(file_c, C);
        std::cout << "[Info] Result saved to: " << file_c << std::endl;
    }

    // Генерация отчета
    write_report(report_file, matrix_size, duration_us);
    std::cout << "[Info] Report saved to: " << report_file << std::endl;
    
    std::cout << "\n----------------------------------------\n";
    std::cout << "Execution Time: " << duration_us << " us (" 
              << std::fixed << std::setprecision(3) << duration_us / 1000.0 << " ms)\n";
    std::cout << "----------------------------------------\n";

    return 0;
}