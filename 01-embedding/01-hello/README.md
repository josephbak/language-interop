# 01-hello: Minimal Python Embedding

The simplest possible example of embedding Python in C++.

## What This Demonstrates

- `Py_Initialize()` — starts the interpreter
- `PyRun_SimpleString()` — executes Python code
- `Py_Finalize()` — shuts down the interpreter

## Build and Run

```bash
make
make run
```

## Expected Output

```
Python interpreter initialized
Python version: 3.14.x (...)
Hello from Python inside C++!
The answer is 42
Python interpreter finalized
```