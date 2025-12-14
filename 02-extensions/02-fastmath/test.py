import fastmath

# Test sum_of_squares
print("sum_of_squares(10):", fastmath.sum_of_squares(10))  # 385
print("sum_of_squares(100):", fastmath.sum_of_squares(100))  # 338350

# Test dot_product
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]
print(f"dot_product({a}, {b}):", fastmath.dot_product(a, b))  # 32.0

# Test norm
v = [3.0, 4.0]
print(f"norm({v}):", fastmath.norm(v))  # 5.0