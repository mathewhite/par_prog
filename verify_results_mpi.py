import numpy as np
import sys

def load_matrix(path):
    """Загрузка матрицы из файла формата: размерность, затем элементы."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        raise ValueError(f"Empty file: {path}")
        
    size = int(lines[0])
    data = []
    for i in range(size):
        row = list(map(float, lines[i + 1].split()))
        if len(row) != size:
            raise ValueError(f"Row length mismatch in {path}")
        data.append(row)
        
    return np.array(data)

def verify_mpi_results(a_path, b_path, c_path, tolerance=1e-9):
    print("\n" + "="*50)
    print("       MPI VERIFICATION REPORT")
    print("="*50)
    
    try:
        A = load_matrix(a_path)
        B = load_matrix(b_path)
        C_mpi = load_matrix(c_path)
    except Exception as e:
        print(f"[Error] Loading failed: {e}")
        return False
        
    print(f"\nDimensions:")
    print(f"  A: {A.shape[0]}x{A.shape[1]}")
    print(f"  B: {B.shape[0]}x{B.shape[1]}")
    print(f"  C: {C_mpi.shape[0]}x{C_mpi.shape[1]}")
    
    # Эталонный расчет
    C_numpy = np.dot(A, B)
    
    # Анализ ошибок
    diff = np.abs(C_mpi - C_numpy)
    max_err = np.max(diff)
    
    print(f"\nError Analysis:")
    print(f"  Max Absolute Error: {max_err:.2e}")
    
    if max_err < tolerance:
        print(f"\n[SUCCESS] Results match within tolerance ({tolerance}).")
        return True
    else:
        print(f"\n[FAILURE] Results differ significantly.")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python verify_results_mpi.py <matrix_A.txt> <matrix_B.txt> <matrix_C.txt>")
        sys.exit(1)
        
    success = verify_mpi_results(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(0 if success else 1)