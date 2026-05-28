import numpy as np
import sys

def load_matrix(path):
    """Load square matrix from text file."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        raise ValueError(f"Empty file: {path}")
        
    n = int(lines[0])
    data = []
    for i in range(n):
        row = list(map(float, lines[i + 1].split()))
        if len(row) != n:
            raise ValueError(f"Row length mismatch in {path}")
        data.append(row)
        
    return np.array(data)

def verify_cuda_result(a_path, b_path, c_path, tol=1e-6):
    print("\n" + "="*50)
    print("       CUDA VERIFICATION REPORT")
    print("="*50)
    
    try:
        A = load_matrix(a_path)
        B = load_matrix(b_path)
        C_gpu = load_matrix(c_path)
    except Exception as e:
        print(f"[Error] Loading failed: {e}")
        return False
        
    print(f"\nDimensions: {A.shape[0]}x{A.shape[1]}")
    
    # Reference calculation
    C_cpu = np.dot(A, B)
    
    # Check closeness
    if np.allclose(C_gpu, C_cpu, atol=tol):
        max_err = np.max(np.abs(C_gpu - C_cpu))
        print(f"\n[SUCCESS] Results match within tolerance ({tol}).")
        print(f"Max absolute error: {max_err:.2e}")
        return True
    else:
        max_err = np.max(np.abs(C_gpu - C_cpu))
        print(f"\n[FAILURE] Results differ significantly.")
        print(f"Max absolute error: {max_err:.2e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python verify_results_cuda.py <A.txt> <B.txt> <C.txt>")
        sys.exit(1)
        
    success = verify_cuda_result(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(0 if success else 1)