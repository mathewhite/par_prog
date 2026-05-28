#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <mpi.h>
#include <stdexcept>

// Типы данных
using Matrix = std::vector<std::vector<double>>;

// --- I/O Operations ---

Matrix load_matrix_from_file(const std::string& filename, int& dimension) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    file >> dimension;
    Matrix mat(dimension, std::vector<double>(dimension));

    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            if (!(file >> mat[i][j])) {
                throw std::runtime_error("Invalid data format in " + filename);
            }
        }
    }
    return mat;
}

void save_matrix_to_file(const std::string& filename, const Matrix& mat) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create file: " + filename);
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

// --- MPI Logic ---

/**
 * Распределяет строки матрицы между процессами.
 * Используется циклическое или блочное распределение (здесь блочное для простоты).
 */
std::pair<Matrix, int> distribute_rows(const Matrix& global_mat, int rank, int size) {
    int n = global_mat.size();
    int base_rows = n / size;
    int remainder = n % size;
    
    // Сколько строк получает этот процесс
    int local_rows = base_rows + (rank < remainder ? 1 : 0);
    
    // Смещение начала блока для этого процесса
    int offset = rank * base_rows + std::min(rank, remainder);
    
    Matrix local_mat(local_rows, std::vector<double>(n));
    for (int i = 0; i < local_rows; ++i) {
        local_mat[i] = global_mat[offset + i];
    }
    
    return {local_mat, local_rows};
}

/**
 * Собирает результаты от всех процессов в итоговую матрицу на процессе 0.
 */
Matrix gather_results(const Matrix& local_result, int local_rows, int rank, int size, int global_n) {
    Matrix global_result;
    
    if (rank == 0) {
        global_result.resize(global_n, std::vector<double>(global_n, 0.0));
        // Копируем свои данные
        for (int i = 0; i < local_rows; ++i) {
            global_result[i] = local_result[i];
        }
        
        // Принимаем данные от других процессов
        int current_offset = local_rows;
        for (int p = 1; p < size; ++p) {
            int p_rows = global_n / size + (p < (global_n % size) ? 1 : 0);
            std::vector<double> buffer(p_rows * global_n);
            
            MPI_Recv(buffer.data(), p_rows * global_n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < p_rows; ++i) {
                for (int j = 0; j < global_n; ++j) {
                    global_result[current_offset + i][j] = buffer[i * global_n + j];
                }
            }
            current_offset += p_rows;
        }
    } else {
        // Отправляем свои данные процессу 0
        std::vector<double> buffer(local_rows * global_n);
        for (int i = 0; i < local_rows; ++i) {
            for (int j = 0; j < global_n; ++j) {
                buffer[i * global_n + j] = local_result[i][j];
            }
        }
        MPI_Send(buffer.data(), local_rows * global_n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    return global_result;
}

Matrix multiply_local(const Matrix& local_A, const Matrix& B) {
    int local_n = local_A.size();
    int n = B.size();
    Matrix C(local_n, std::vector<double>(n, 0.0));
    
    for (int i = 0; i < local_n; ++i) {
        for (int k = 0; k < n; ++k) {
            double a_val = local_A[i][k];
            for (int j = 0; j < n; ++j) {
                C[i][j] += a_val * B[k][j];
            }
        }
    }
    return C;
}

// --- Main ---

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    try {
        bool is_test_mode = false;
        int matrix_size = 0;
        std::string file_a, file_b, file_out;
        
        // Парсинг аргументов
        if (argc == 3 && std::string(argv[1]) == "--test") {
            is_test_mode = true;
            matrix_size = std::stoi(argv[2]);
        } else if (argc == 4) {
            file_a = argv[1];
            file_b = argv[2];
            file_out = argv[3];
        } else {
            if (rank == 0) {
                std::cerr << "Usage:\n"
                          << "  mpirun -np N " << argv[0] << " --test SIZE\n"
                          << "  mpirun -np N " << argv[0] << " A.txt B.txt out.txt\n";
            }
            MPI_Finalize();
            return 1;
        }
        
        Matrix A, B;
        
        if (is_test_mode) {
            if (rank == 0) {
                std::srand(static_cast<unsigned int>(std::time(nullptr)));
                A = generate_random_matrix(matrix_size);
                B = generate_random_matrix(matrix_size);
                
                // Сохраняем для верификации
                save_matrix_to_file("mpi_A.txt", A);
                save_matrix_to_file("mpi_B.txt", B);
                std::cout << "[Test Mode] Generated matrices " << matrix_size << "x" << matrix_size << "\n";
            }
        } else {
            if (rank == 0) {
                int dim_a, dim_b;
                A = load_matrix_from_file(file_a, dim_a);
                B = load_matrix_from_file(file_b, dim_b);
                if (dim_a != dim_b) {
                    throw std::invalid_argument("Matrix dimensions mismatch");
                }
                matrix_size = dim_a;
                std::cout << "[File Mode] Loaded matrices from files\n";
            }
        }
        
        // Рассылка размера матрицы всем процессам
        MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Если режим файлов, нужно разослать матрицы B всем (A распределяется)
        // Для упрощения в этом примере мы предполагаем, что B помещается в память каждого процесса
        // Или рассылаем B через Broadcast если он мал, или используем ScatterV для больших
        
        if (!is_test_mode || rank == 0) {
             // В реальном приложении здесь был бы MPI_Bcast для матрицы B, 
             // но для простоты лабы часто загружают B локально или шлют целиком.
             // Здесь сделаем упрощение: если rank != 0, то A и B пустые, их надо получить.
             
             // Для корректности в рамках этой лабы, давайте просто загрузим B на всех процессах из файла,
             // если это не тестовый режим. В тестовом режиме B есть только на 0.
             
             if (!is_test_mode) {
                 int dummy_dim;
                 B = load_matrix_from_file(file_b, dummy_dim);
             } else {
                 // В тестовом режиме рассылаем B от 0 ко всем
                 // Сначала создадим буфер для B на всех процессах
                 if (rank != 0) {
                     B.resize(matrix_size, std::vector<double>(matrix_size));
                 }
                 
                 // Flatten B for sending
                 std::vector<double> flat_B(matrix_size * matrix_size);
                 if (rank == 0) {
                     for(int i=0; i<matrix_size; ++i)
                         for(int j=0; j<matrix_size; ++j)
                             flat_B[i*matrix_size + j] = B[i][j];
                 }
                 
                 MPI_Bcast(flat_B.data(), matrix_size * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                 
                 if (rank != 0) {
                     for(int i=0; i<matrix_size; ++i)
                         for(int j=0; j<matrix_size; ++j)
                             B[i][j] = flat_B[i*matrix_size + j];
                 }
                 
                 // Также нужно разослать A, если это тест, но мы будем распределять его частями.
                 // Для теста проще сгенерировать A на всех процессах одинаково (seed по рангу?) 
                 // Нет, лучше разослать A целиком как B, а потом нарезать.
                 
                 std::vector<double> flat_A(matrix_size * matrix_size);
                 if (rank == 0) {
                     for(int i=0; i<matrix_size; ++i)
                         for(int j=0; j<matrix_size; ++j)
                             flat_A[i*matrix_size + j] = A[i][j];
                 }
                 MPI_Bcast(flat_A.data(), matrix_size * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                 
                 if (rank != 0) {
                     A.resize(matrix_size, std::vector<double>(matrix_size));
                     for(int i=0; i<matrix_size; ++i)
                         for(int j=0; j<matrix_size; ++j)
                             A[i][j] = flat_A[i*matrix_size + j];
                 }
             }
        } else {
             // Если не тест и не 0 процесс, мы уже загрузили B выше.
             // Нужно загрузить A? Нет, A мы будем получать через распределение.
             // Но в текущей логике distribute_rows требует полный A.
             // Значит, в файловом режиме тоже нужно разослать A целиком или читать файл на каждом процессе.
             // Чтение файла на каждом процессе - самое простое для MPI lab.
             
             int dummy_dim;
             A = load_matrix_from_file(file_a, dummy_dim);
             // B уже загружен выше
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        
        // Распределение A
        auto [local_A, local_rows] = distribute_rows(A, rank, size);
        
        // Локальное умножение
        Matrix local_C = multiply_local(local_A, B);
        
        // Сбор результатов
        Matrix final_C = gather_results(local_C, local_rows, rank, size, matrix_size);
        
        double end_time = MPI_Wtime();
        
        if (rank == 0) {
            double elapsed_ms = (end_time - start_time) * 1000.0;
            long long ops = 2LL * matrix_size * matrix_size * matrix_size;
            double gflops = (ops / 1e9) / (elapsed_ms / 1000.0);
            
            std::cout << "----------------------------------------\n";
            std::cout << "Size: " << matrix_size << "x" << matrix_size << "\n";
            std::cout << "Processes: " << size << "\n";
            std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n";
            std::cout << "GFLOPS: " << std::setprecision(2) << gflops << "\n";
            std::cout << "----------------------------------------\n";
            
            if (is_test_mode) {
                save_matrix_to_file("mpi_C.txt", final_C);
                std::cout << "[Info] Result saved to mpi_C.txt\n";
            } else {
                save_matrix_to_file(file_out, final_C);
                std::cout << "[Info] Result saved to " << file_out << "\n";
            }
        }
        
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "[Error] " << e.what() << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Finalize();
    return 0;
}