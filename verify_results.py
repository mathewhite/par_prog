import numpy as np
import sys

def load_matrix(filepath):
    """Чтение матрицы из файла формата: размерность, затем элементы."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        size = int(lines[0])
        data = []
        for i in range(size):
            # Разбиваем строку по пробелам и конвертируем в float
            row = list(map(float, lines[i + 1].split()))
            data.append(row)
            
        return np.array(data)
    except Exception as e:
        print(f"[Error] Failed to load {filepath}: {e}")
        sys.exit(1)

def verify_computation(mat_a_path, mat_b_path, mat_res_path, tolerance=1e-9):
    """Сравнение результата C++ программы с numpy.dot"""
    
    print("\n" + "="*50)
    print("       VERIFICATION REPORT (NumPy vs C++)")
    print("="*50)

    # Загрузка матриц
    A = load_matrix(mat_a_path)
    B = load_matrix(mat_b_path)
    C_cpp = load_matrix(mat_res_path)

    print(f"\nDimensions:")
    print(f"  A: {A.shape[0]}x{A.shape[1]}")
    print(f"  B: {B.shape[0]}x{B.shape[1]}")
    print(f"  C (result): {C_cpp.shape[0]}x{C_cpp.shape[1]}")

    # Эталонный расчет
    C_numpy = np.dot(A, B)

    # Поиск максимального расхождения
    diff = np.abs(C_cpp - C_numpy)
    max_err = np.max(diff)
    mean_err = np.mean(diff)

    print(f"\nError Analysis:")
    print(f"  Max Absolute Error: {max_err:.2e}")
    print(f"  Mean Absolute Error: {mean_err:.2e}")

    # Вердикт
    if max_err < tolerance:
        print(f"\n[SUCCESS] Results match within tolerance ({tolerance}).")
        return True
    else:
        print(f"\n[FAILURE] Results differ significantly.")
        print("\nSample (top-left 3x3):")
        print("C++ Result:\n", C_cpp[:3, :3])
        print("NumPy Expected:\n", C_numpy[:3, :3])
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python verify_results.py <matrix_A.txt> <matrix_B.txt> <matrix_C.txt>")
        sys.exit(1)

    success = verify_computation(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(0 if success else 1)