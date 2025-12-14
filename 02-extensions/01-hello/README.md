# 01-hello: Minimal C++ Extension

The simplest possible Python extension in C++.

## What This Demonstrates

- `PyArg_ParseTuple()` — extract Python arguments
- `PyMethodDef` — declare functions to expose
- `PyModuleDef` — define the module
- `PyInit_<name>()` — entry point when Python imports

## Build and Run

```bash
make run
```

## Expected Output

```
Hello, World! (from C++)
Hello, Python! (from C++)
Hello, C++! (from C++)
```