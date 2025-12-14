# 03-tensor-ops: Mini Tensor Library

A minimal NumPy-like tensor library in C++.

## Features

| Function | Description |
|----------|-------------|
| `zeros(shape)` | Create tensor filled with zeros |
| `from_list(data)` | Create tensor from Python list |
| `add(a, b)` | Element-wise addition |
| `mul(a, b)` | Element-wise multiplication |
| `matmul(a, b)` | Matrix multiplication: C = AB |
| `sum(a)` | Sum all elements |

## What This Demonstrates

- Custom Python type (`Tensor` class)
- Memory layout: row-major 2D arrays in contiguous memory
- Matrix multiplication: O(n³) naive implementation

## Memory Layout

2D tensor stored as 1D array (row-major):

```
Logical:          Physical (data[]):
[[1, 2, 3],       [1, 2, 3, 4, 5, 6]
 [4, 5, 6]]

Index [i][j] = data[i * num_cols + j]
```

## Build and Run

```bash
make run    # basic tests
make bench  # benchmark vs pure Python
```

## Relevance to ML Infrastructure

This is exactly how NumPy, PyTorch, and TensorFlow work internally:
- Python interface for ease of use
- C/C++ backend for performance
- Contiguous memory for cache efficiency