import autodiff as ad
import math

print("=== Forward-Mode Automatic Differentiation ===\n")

# Example 1: f(x) = x² + 3x
# f'(x) = 2x + 3
print("--- f(x) = x² + 3x ---")
x = ad.var(2.0)  # x = 2, seeded with grad = 1
f = x * x + 3 * x
print(f"At x = 2:")
print(f"  f(x)  = {f.val}")   # 4 + 6 = 10
print(f"  f'(x) = {f.grad}")  # 2*2 + 3 = 7
print()

# Example 2: f(x) = sin(x) * cos(x)
# f'(x) = cos(x)*cos(x) - sin(x)*sin(x) = cos(2x)
print("--- f(x) = sin(x) * cos(x) ---")
x = ad.var(0.0)
f = ad.sin(x) * ad.cos(x)
print(f"At x = 0:")
print(f"  f(x)  = {f.val}")   # 0 * 1 = 0
print(f"  f'(x) = {f.grad}")  # cos(0) = 1
print(f"  (expected: cos(2*0) = 1)")
print()

# Example 3: f(x) = exp(x) / x
# f'(x) = (exp(x) * x - exp(x)) / x² = exp(x) * (x - 1) / x²
print("--- f(x) = exp(x) / x ---")
x = ad.var(2.0)
f = ad.exp(x) / x
expected_grad = math.exp(2.0) * (2.0 - 1.0) / (2.0 * 2.0)
print(f"At x = 2:")
print(f"  f(x)  = {f.val:.6f}")
print(f"  f'(x) = {f.grad:.6f}")
print(f"  (expected: {expected_grad:.6f})")
print()

# Example 4: f(x) = sqrt(x² + 1)
# f'(x) = x / sqrt(x² + 1)
print("--- f(x) = sqrt(x² + 1) ---")
x = ad.var(3.0)
f = ad.sqrt(x * x + 1)
expected_grad = 3.0 / math.sqrt(3.0 * 3.0 + 1)
print(f"At x = 3:")
print(f"  f(x)  = {f.val:.6f}")
print(f"  f'(x) = {f.grad:.6f}")
print(f"  (expected: {expected_grad:.6f})")
print()

# Example 5: Composition - f(x) = log(sin(x) + 2)
print("--- f(x) = log(sin(x) + 2) ---")
x = ad.var(1.0)
f = ad.log(ad.sin(x) + 2)
# f'(x) = cos(x) / (sin(x) + 2)
expected_grad = math.cos(1.0) / (math.sin(1.0) + 2)
print(f"At x = 1:")
print(f"  f(x)  = {f.val:.6f}")
print(f"  f'(x) = {f.grad:.6f}")
print(f"  (expected: {expected_grad:.6f})")
print()

# Example 6: Using operators with Python numbers
print("--- f(x) = 5*x³ - 2*x + 7 ---")
x = ad.var(2.0)
f = 5 * ad.pow(x, 3) - 2 * x + 7
# f'(x) = 15x² - 2
expected_grad = 15 * 2.0 * 2.0 - 2
print(f"At x = 2:")
print(f"  f(x)  = {f.val:.6f}")   # 5*8 - 4 + 7 = 43
print(f"  f'(x) = {f.grad:.6f}")  # 15*4 - 2 = 58
print(f"  (expected: {expected_grad:.6f})")