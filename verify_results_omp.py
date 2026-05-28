import numpy as np
import sys

def load_mat(path):
    with open(path, 'r') as f:
        n = int(f.readline())
        data = [list(map(float, f.readline().split())) for _ in range(n)]
        return np.array(data)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python verify_results_omp.py A.txt B.txt C.txt")
        sys.exit(1)

    A = load_mat(sys.argv[1])
    B = load_mat(sys.argv[2])
    C_res = load_mat(sys.argv[3])
    
    C_exp = A @ B
    
    diff = np.max(np.abs(C_res - C_exp))
    
    print(f"Max deviation: {diff:.2e}")
    if diff < 1e-6:
        print("Status: OK")
        sys.exit(0)
    else:
        print("Status: FAIL")
        sys.exit(1)