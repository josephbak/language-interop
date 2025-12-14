#include <Python.h>
#include <cmath>

// Sum of squares: 0² + 1² + 2² + ... + n²
static PyObject* sum_of_squares(PyObject* self, PyObject* args) {
    long n;
    if (!PyArg_ParseTuple(args, "l", &n)) {
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }

    long long result = 0;
    for (long i = 0; i <= n; i++) {
        result += i * i;
    }

    return PyLong_FromLongLong(result);
}

// Dot product of two lists
static PyObject* dot_product(PyObject* self, PyObject* args) {
    PyObject* list_a;
    PyObject* list_b;

    if (!PyArg_ParseTuple(args, "OO", &list_a, &list_b)) {
        return NULL;
    }

    // Verify both are lists
    if (!PyList_Check(list_a) || !PyList_Check(list_b)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be lists");
        return NULL;
    }

    Py_ssize_t len_a = PyList_Size(list_a);
    Py_ssize_t len_b = PyList_Size(list_b);

    if (len_a != len_b) {
        PyErr_SetString(PyExc_ValueError, "Lists must have same length");
        return NULL;
    }

    double result = 0.0;
    for (Py_ssize_t i = 0; i < len_a; i++) {
        PyObject* item_a = PyList_GetItem(list_a, i);
        PyObject* item_b = PyList_GetItem(list_b, i);

        double val_a = PyFloat_AsDouble(item_a);
        double val_b = PyFloat_AsDouble(item_b);

        if (PyErr_Occurred()) {
            return NULL;  // conversion failed
        }

        result += val_a * val_b;
    }

    return PyFloat_FromDouble(result);
}

// Euclidean norm (length) of a vector
static PyObject* norm(PyObject* self, PyObject* args) {
    PyObject* list;

    if (!PyArg_ParseTuple(args, "O", &list)) {
        return NULL;
    }

    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list");
        return NULL;
    }

    Py_ssize_t len = PyList_Size(list);
    double sum_sq = 0.0;

    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject* item = PyList_GetItem(list, i);
        double val = PyFloat_AsDouble(item);

        if (PyErr_Occurred()) {
            return NULL;
        }

        sum_sq += val * val;
    }

    return PyFloat_FromDouble(std::sqrt(sum_sq));
}

static PyMethodDef FastMathMethods[] = {
    {"sum_of_squares", sum_of_squares, METH_VARARGS, "Sum of squares from 0 to n"},
    {"dot_product", dot_product, METH_VARARGS, "Dot product of two vectors"},
    {"norm", norm, METH_VARARGS, "Euclidean norm of a vector"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastmathmodule = {
    PyModuleDef_HEAD_INIT,
    "fastmath",
    "Fast math operations in C++",
    -1,
    FastMathMethods
};

PyMODINIT_FUNC PyInit_fastmath(void) {
    return PyModule_Create(&fastmathmodule);
}