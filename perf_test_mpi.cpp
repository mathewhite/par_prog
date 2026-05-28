#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <mpi.h>

using Matrix = std::vector<std::vector<double>>;

Matrix build_random_matrix(int n) {
    Matrix mat(n, std::vector<double>(n));
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            mat[r][c] = (rand() % 10) + 1;
    return mat;
}

Matrix serial_multiply(const Matrix& U, const Matrix& V) {
    int n = U.size();
    Matrix W(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                W[i][j] += U[i][k] * V[k][j];
    return W;
}

Matrix mpi_multiply(const Matrix& U, const Matrix& V, int my_rank, int world_sz) {
    int n = U.size();
    int base = n / world_sz;
    int extra = n % world_sz;
    int my_rows = base + (my_rank < extra ? 1 : 0);
    
    int start_row = 0;
    for (int p = 0; p < my_rank; ++p)
        start_row += base + (p < extra ? 1 : 0);
    
    Matrix local_U(my_rows, std::vector<double>(n));
    for (int r = 0; r < my_rows; ++r)
        for (int c = 0; c < n; ++c)
            local_U[r][c] = U[start_row + r][c];
    
    Matrix local_W(my_rows, std::vector<double>(n, 0.0));
    for (int r = 0; r < my_rows; ++r)
        for (int c = 0; c < n; ++c)
            for (int k = 0; k < n; ++k)
                local_W[r][c] += local_U[r][k] * V[k][c];
    
    Matrix result(n, std::vector<double>(n));
    
    if (my_rank == 0) {
        for (int r = 0; r < my_rows; ++r)
            for (int c = 0; c < n; ++c)
                result[start_row + r][c] = local_W[r][c];
        
        int next_row = start_row + my_rows;
        for (int p = 1; p < world_sz; ++p) {
            int p_rows = base + (p < extra ? 1 : 0);
            std::vector<double> chunk(p_rows * n);
            MPI_Recv(chunk.data(), p_rows * n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int r = 0; r < p_rows; ++r)
                for (int c = 0; c < n; ++c)
                    result[next_row + r][c] = chunk[r * n + c];
            
            next_row += p_rows;
        }
    } else {
        std::vector<double> out_buffer(my_rows * n);
        for (int r = 0; r < my_rows; ++r)
            for (int c = 0; c < n; ++c)
                out_buffer[r * n + c] = local_W[r][c];
        
        MPI_Send(out_buffer.data(), my_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    return result;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int my_id, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    std::vector<int> dimensions = {200, 400, 800, 1200, 1600, 2000};
    
    std::ofstream log;
    if (my_id == 0) {
        log.open("mpi_perf.csv");
        log << "Procs,Size,Time_ms,GFLOPS,Speedup,Efficiency_pct\n";
        std::cout << "========== MPI Performance Report ==========\n\n";
    }
    
    for (int N : dimensions) {
        srand(42 + my_id);
        Matrix A = build_random_matrix(N);
        Matrix B = build_random_matrix(N);
        
        double serial_duration = 0.0;
        if (my_id == 0) {
            std::cout << "Dimension: " << N << "x" << N << " ...\n";
            double t0 = MPI_Wtime();
            Matrix C_serial = serial_multiply(A, B);
            double t1 = MPI_Wtime();
            serial_duration = t1 - t0;
        }
        MPI_Bcast(&serial_duration, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        double t_start = MPI_Wtime();
        Matrix C_parallel = mpi_multiply(A, B, my_id, num_procs);
        double t_end = MPI_Wtime();
        double local_dur = t_end - t_start;
        
        double max_duration = local_dur;
        MPI_Reduce(&local_dur, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        if (my_id == 0) {
            long long flops = 2LL * N * N * N;
            double spdup = serial_duration / max_duration;
            double eff = (spdup / num_procs) * 100.0;
            double gigaflops = flops / max_duration / 1e9;
            
            std::cout << "  Processors: " << num_procs << "\n"
                      << "  Wall time: " << max_duration * 1000.0 << " ms\n"
                      << "  Speedup: " << std::fixed << std::setprecision(2) << spdup << "x\n"
                      << "  Efficiency: " << eff << "%\n"
                      << "  GFLOPS: " << gigaflops << "\n\n";
            
            log << num_procs << "," << N << "," << max_duration * 1000.0 << ","
                << std::fixed << std::setprecision(2) << gigaflops << ","
                << spdup << "," << eff << "\n";
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (my_id == 0) {
        log.close();
        std::cout << "Saved: mpi_perf.csv\n";
    }
    
    MPI_Finalize();
    return 0;
}