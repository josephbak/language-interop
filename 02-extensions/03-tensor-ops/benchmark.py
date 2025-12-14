import tensor
import time

def benchmark(name, func, runs=5):
    # Warmup
    func()
    
    start = time.perf_counter()
    for _ in range(runs):
        func()
    elapsed = (time.perf_counter() - start) / runs
    print(f"{name}: {elapsed*1000:.3f}ms")
    return elapsed

print("=== Matrix Multiplication Benchmark ===\n")

for size in [64, 128, 256, 512]:
    print(f"--- {size}x{size} matrices ---")
    
    # Create test data
    data_a = [[float(i * size + j) for j in range(size)] for i in range(size)]
    data_b = [[float(i * size + j) for j in range(size)] for i in range(size)]
    
    # C++ tensor
    a = tensor.from_list(data_a)
    b = tensor.from_list(data_b)
    
    # Benchmark C++
    cpp_time = benchmark("C++ matmul", lambda: tensor.matmul(a, b))
    
    # Benchmark pure Python
    def python_matmul():
        m, k, n = size, size, size
        result = [[0.0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                for kk in range(k):
                    result[i][j] += data_a[i][kk] * data_b[kk][j]
        return result
    
    py_time = benchmark("Python matmul", python_matmul, runs=1)
    
    print(f"Speedup: {py_time/cpp_time:.1f}x\n")