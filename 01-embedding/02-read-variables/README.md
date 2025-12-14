# 02-read-variables: Reading Python Variables from C++

C++ loads a Python config file and extracts values.

## What This Demonstrates

- `PyRun_SimpleFile()` — execute external .py file
- `PyImport_AddModule("__main__")` — access the module namespace
- `PyDict_GetItemString()` — retrieve variables by name
- Type conversions: `PyUnicode_AsUTF8`, `PyLong_AsLong`, `PyFloat_AsDouble`, `PyTuple_GetItem`

## Build and Run

```bash
make run
```

## Try It

Edit `config.py`, run again—no recompilation needed.