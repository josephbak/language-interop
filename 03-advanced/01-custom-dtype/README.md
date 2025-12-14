# 01-custom-dtype: Memory Layouts

Demonstrates how the same logical tensor can have different physical memory layouts.

## Layouts

### Row-major (C/C++/NumPy default)
```
Logical:           Memory:
[[1, 2, 3],        [1, 2, 3, 4, 5, 6, 7, 8, 9]
 [4, 5, 6],
 [7, 8, 9]]
```

Index formula: `i * cols + j`

### Column-major (Fortran/MATLAB)
```
Logical:           Memory:
[[1, 2, 3],        [1, 4, 7, 2, 5, 8, 3, 6, 9]
 [4, 5, 6],
 [7, 8, 9]]
```

Index formula: `j * rows + i`

### Tiled (Accelerators/MLIR)
```
Logical:                    Memory (2x2 tiles):
[[1,  2,  3,  4],           [1, 2, 5, 6], [3, 4, 7, 8], ...
 [5,  6,  7,  8],             tile 0        tile 1
 [9,  10, 11, 12],
 [13, 14, 15, 16]]
```

Keeps related data together for block operations.

## MLIR Connection

MLIR represents layouts via affine maps:
```mlir
// Row-major
memref<4x4xf32, affine_map<(i,j) -> (i * 4 + j)>>

// Column-major
memref<4x4xf32, affine_map<(i,j) -> (j * 4 + i)>>
```

Understanding layouts is fundamental for:
- Tensor compiler optimizations
- Hardware-specific code generation
- Efficient data movement

## Build and Run

```bash
make run    # demonstrate layouts
make bench  # benchmark different access patterns
```