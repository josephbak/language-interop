import layout

print("=== Memory Layout Performance ===\n")

# Create large tensors
size = 256
data = [[float(i * size + j) for j in range(size)] for i in range(size)]

rm = layout.from_list(data, layout="row_major")
cm = layout.from_list(data, layout="col_major")
tiled = layout.from_list(data, layout="tiled", tile_size=16)

print(f"Matrix size: {size}x{size}")
print(f"1000 iterations each\n")

# Row-major tensor
print("--- Row-major layout ---")
row_result = layout.benchmark_row_sum(rm)
col_result = layout.benchmark_col_sum(rm)
raw_result = layout.benchmark_raw_sequential(rm)
print(f"  Row iteration:   {row_result['time_ms']:.2f}ms")
print(f"  Col iteration:   {col_result['time_ms']:.2f}ms")
print(f"  Raw sequential:  {raw_result['time_ms']:.2f}ms")
print()

# Column-major tensor
print("--- Column-major layout ---")
row_result = layout.benchmark_row_sum(cm)
col_result = layout.benchmark_col_sum(cm)
raw_result = layout.benchmark_raw_sequential(cm)
print(f"  Row iteration:   {row_result['time_ms']:.2f}ms")
print(f"  Col iteration:   {col_result['time_ms']:.2f}ms")
print(f"  Raw sequential:  {raw_result['time_ms']:.2f}ms")
print()

# Tiled tensor
print("--- Tiled layout (16x16 tiles) ---")
row_result = layout.benchmark_row_sum(tiled)
col_result = layout.benchmark_col_sum(tiled)
raw_result = layout.benchmark_raw_sequential(tiled)
print(f"  Row iteration:   {row_result['time_ms']:.2f}ms")
print(f"  Col iteration:   {col_result['time_ms']:.2f}ms")
print(f"  Raw sequential:  {raw_result['time_ms']:.2f}ms")
print()

print("=== Analysis ===")
print("- Row-major is fastest when iterating row-by-row (cache-friendly)")
print("- Column-major is fastest when iterating column-by-column")
print("- Raw sequential is always fastest (direct memory access)")
print("- Tiled balances row/column access for block operations")