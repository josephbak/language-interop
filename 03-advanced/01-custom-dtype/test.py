import layout

print("=== Memory Layout Demonstration ===\n")

data = [[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]]

# Row-major
rm = layout.from_list(data, layout="row_major")
print(f"Row-major: {rm}")
print(f"  Logical: {rm.tolist()}")
print(f"  Memory:  {rm.memory_view()}")
print()

# Column-major
cm = layout.from_list(data, layout="col_major")
print(f"Column-major: {cm}")
print(f"  Logical: {cm.tolist()}")
print(f"  Memory:  {cm.memory_view()}")
print()

# Tiled (4x4 for clean tiles)
data_4x4 = [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]]

tiled = layout.from_list(data_4x4, layout="tiled", tile_size=2)
print(f"Tiled: {tiled}")
print(f"  Logical: {tiled.tolist()}")
print(f"  Memory:  {tiled.memory_view()}")
print()

# Verify all layouts give same values
print("=== Verification ===")
print(f"rm.get(1, 2) = {rm.get(1, 2)}")  # should be 7
print(f"cm.get(1, 2) = {cm.get(1, 2)}")  # should be 7
print(f"tiled.get(1, 2) = {tiled.get(1, 2)}")  # should be 7