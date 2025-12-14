# language-interop

Exploring how compiled (C++) and interpreted (Python) code work together.

## Topics Covered

- **Embedding**: Running Python interpreter inside C++ programs
- **Extensions**: Calling C++ from Python for performance
- **Advanced**: Custom data types, automatic differentiation

## Why This Matters

This is how NumPy, PyTorch, and TensorFlow work internally—Python interface, native code underneath.

## Building

macOS:
```bash
# Uses clang++ and python3-config
make all
```

Linux:
```bash
make all
```

## Prerequisites

- clang++
- Python 3.x with development headers