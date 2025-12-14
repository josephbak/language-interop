# 02-fastmath: Fast Math Operations

C++ extension with multiple math functions and benchmarks.

## Functions

| Function | Description |
|----------|-------------|
| `sum_of_squares(n)` | Σᵢ₌₀ⁿ i² |
| `dot_product(a, b)` | Σᵢ aᵢbᵢ |
| `norm(v)` | √(Σᵢ vᵢ²) |

## What This Demonstrates

- Error handling (`PyErr_SetString`)
- Working with Python lists (`PyList_Check`, `PyList_Size`, `PyList_GetItem`)
- Type conversions (`PyFloat_AsDouble`, `PyFloat_FromDouble`)
- Multiple functions in one module

## Build and Run

```bash
make run    # basic tests
make bench  # benchmarks vs pure Python
```