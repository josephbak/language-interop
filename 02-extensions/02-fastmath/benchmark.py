import fastmath
import time
import math

def benchmark(name, c_func, py_func, runs=3):
    print(f"=== {name} ===")
    
    # C++ timing (run multiple times for accuracy)
    start = time.perf_counter()
    for _ in range(runs):
        result_c = c_func()
    c_time = (time.perf_counter() - start) / runs
    
    # Python timing
    start = time.perf_counter()
    for _ in range(runs):
        result_py = py_func()
    py_time = (time.perf_counter() - start) / runs
    
    print(f"C++ time:    {c_time:.6f}s")
    print(f"Python time: {py_time:.6f}s")
    if c_time > 0:
        print(f"Speedup:     {py_time / c_time:.1f}x")
    else:
        print(f"Speedup:     C++ too fast to measure")
    print()

# --- sum_of_squares ---
n = 10_000_000
benchmark(
    "sum_of_squares",
    lambda: fastmath.sum_of_squares(n),
    lambda: sum(i * i for i in range(n + 1))
)

# --- dot_product ---
size = 1_000_000
a = [float(i) for i in range(size)]
b = [float(i) for i in range(size)]

benchmark(
    "dot_product",
    lambda: fastmath.dot_product(a, b),
    lambda: sum(x * y for x, y in zip(a, b))
)

# --- norm ---
benchmark(
    "norm",
    lambda: fastmath.norm(a),
    lambda: math.sqrt(sum(x * x for x in a))
)