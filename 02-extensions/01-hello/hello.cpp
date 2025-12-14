#include <Python.h>

// The function Python will call
static PyObject* say_hello(PyObject* self, PyObject* args) {
    const char* name;
    
    // Parse the Python string argument
    if (!PyArg_ParseTuple(args, "s", &name)) {
        return NULL;
    }
    
    // Print from C++
    printf("Hello, %s! (from C++)\n", name);
    
    // Return None
    Py_RETURN_NONE;
}

// List of functions this module exposes
static PyMethodDef HelloMethods[] = {
    {"say_hello", say_hello, METH_VARARGS, "Say hello to someone"},
    {NULL, NULL, 0, NULL}  // sentinel
};

// Module definition
static struct PyModuleDef hellomodule = {
    PyModuleDef_HEAD_INIT,
    "hello",        // module name (import hello)
    "A minimal C++ extension",  // docstring
    -1,                         // state size (-1 = global state)
    HelloMethods                // the function table
};

// Module initialization (called when Python imports)
PyMODINIT_FUNC PyInit_hello(void) {
    return PyModule_Create(&hellomodule);
}