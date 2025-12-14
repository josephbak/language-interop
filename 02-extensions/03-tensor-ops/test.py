import tensor

print("=== Creating Tensors ===")
a = tensor.zeros(5)
print(f"zeros(5): {a}")
print(f"shape: {a.shape}")

b = tensor.zeros((2, 3))
print(f"zeros((2,3)): {b}")
print(f"shape: {b.shape}")

print("\n=== From List ===")
v1 = tensor.from_list([1.0, 2.0, 3.0, 4.0])
v2 = tensor.from_list([5.0, 6.0, 7.0, 8.0])
print(f"v1: {v1}")
print(f"v2: {v2}")

print("\n=== Element-wise Operations ===")
print(f"add(v1, v2): {tensor.add(v1, v2)}")
print(f"mul(v1, v2): {tensor.mul(v1, v2)}")
print(f"sum(v1): {tensor.sum(v1)}")

print("\n=== Matrix Multiplication ===")
m1 = tensor.from_list([[1.0, 2.0], [3.0, 4.0]])
m2 = tensor.from_list([[5.0, 6.0], [7.0, 8.0]])
print(f"m1: {m1}")
print(f"m2: {m2}")
result = tensor.matmul(m1, m2)
print(f"matmul(m1, m2): {result}")
print(f"as list: {result.tolist()}")