import autodiff as ad

print("=== Reverse-Mode Automatic Differentiation ===\n")

# Example 1: f(x, y) = (x + y) * x
# Manual: f = x² + xy
#         ∂f/∂x = 2x + y
#         ∂f/∂y = x
print("--- f(x, y) = (x + y) * x ---")
x = ad.Var(2.0)
y = ad.Var(3.0)
a = x + y
f = a * x
f.backward()
print(f"At x=2, y=3:")
print(f"  f = {f.val}")        # (2+3)*2 = 10
print(f"  ∂f/∂x = {x.grad}")   # 2*2 + 3 = 7
print(f"  ∂f/∂y = {y.grad}")   # 2
print()

# Example 2: Shared variable usage
# f(x) = x² + x³
# f'(x) = 2x + 3x²
print("--- f(x) = x² + x³ ---")
x = ad.Var(2.0)
x2 = x * x
x3 = x2 * x
f = x2 + x3
f.backward()
print(f"At x=2:")
print(f"  f = {f.val}")        # 4 + 8 = 12
print(f"  f'(x) = {x.grad}")   # 2*2 + 3*4 = 16
print(f"  (expected: 16)")
print()

# Example 3: With math functions
# f(x) = sin(x) * exp(x)
# f'(x) = cos(x)*exp(x) + sin(x)*exp(x) = exp(x)*(cos(x) + sin(x))
print("--- f(x) = sin(x) * exp(x) ---")
x = ad.Var(1.0)
f = ad.sin(x) * ad.exp(x)
f.backward()
import math
expected = math.exp(1.0) * (math.cos(1.0) + math.sin(1.0))
print(f"At x=1:")
print(f"  f = {f.val:.6f}")
print(f"  f'(x) = {x.grad:.6f}")
print(f"  (expected: {expected:.6f})")
print()

# Example 4: Composing operations
# f(x, y) = log(x * y) + x / y
print("--- f(x, y) = log(x * y) + x / y ---")
x = ad.Var(4.0)
y = ad.Var(2.0)
product = x * y
f = ad.log(product) + x / y
f.backward()
# ∂f/∂x = 1/(x*y) * y + 1/y = 1/x + 1/y
# ∂f/∂y = 1/(x*y) * x - x/y²
expected_dx = 1/4.0 + 1/2.0
expected_dy = 1/(4.0*2.0) * 4.0 - 4.0/(2.0*2.0)
print(f"At x=4, y=2:")
print(f"  f = {f.val:.6f}")
print(f"  ∂f/∂x = {x.grad:.6f} (expected: {expected_dx:.6f})")
print(f"  ∂f/∂y = {y.grad:.6f} (expected: {expected_dy:.6f})")
print()

# Example 5: zero_grad and multiple backward passes
print("--- Multiple backward passes ---")
x = ad.Var(3.0)
f = x * x
f.backward()
print(f"First backward: x.grad = {x.grad}")
f.zero_grad()
print(f"After zero_grad: x.grad = {x.grad}")
f.backward()
print(f"Second backward: x.grad = {x.grad}")
print()

# Example 6: Neural network-like scenario
# f(x, w, b) = (x * w + b)²
print("--- f(x, w, b) = (x * w + b)² ---")
x = ad.Var(2.0)   # input
w = ad.Var(3.0)   # weight
b = ad.Var(1.0)   # bias
z = x * w + b
f = z * z
f.backward()
print(f"At x=2, w=3, b=1:")
print(f"  f = {f.val}")           # (2*3+1)² = 49
print(f"  ∂f/∂x = {x.grad}")      # 2*(x*w+b)*w = 2*7*3 = 42
print(f"  ∂f/∂w = {w.grad}")      # 2*(x*w+b)*x = 2*7*2 = 28
print(f"  ∂f/∂b = {b.grad}")      # 2*(x*w+b)*1 = 2*7 = 14