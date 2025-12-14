#include <Python.h>
#include <vector>
#include <sstream>
#include <cmath>

// ============================================================
// Tensor class
// ============================================================
struct Tensor {
    std::vector<double> data;
    std::vector<size_t> shape;

    size_t size() const {
        size_t s = 1;
        for (auto dim : shape) s *= dim;
        return s;
    }

    bool same_shape(const Tensor& other) const {
        return shape == other.shape;
    }
};

// ============================================================
// Python object wrapping Tensor
// ============================================================
typedef struct {
    PyObject_HEAD
    Tensor* tensor;
} PyTensor;

// ============================================================
// Forward declarations of type methods
// ============================================================
static PyObject* Tensor_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);
static void Tensor_dealloc(PyTensor* self);
static PyObject* Tensor_repr(PyTensor* self);
static PyObject* Tensor_tolist(PyTensor* self, PyObject* args);
static PyObject* Tensor_shape(PyTensor* self, void* closure);

// ============================================================
// Method and getset tables
// ============================================================
static PyMethodDef Tensor_methods[] = {
    {"tolist", (PyCFunction)Tensor_tolist, METH_NOARGS, "Convert to Python list"},
    {NULL}
};

static PyGetSetDef Tensor_getset[] = {
    {"shape", (getter)Tensor_shape, NULL, "Shape of tensor", NULL},
    {NULL}
};

// ============================================================
// PyTensorType definition (before functions that use it)
// ============================================================
static PyTypeObject PyTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "tensor.Tensor",            // tp_name
    sizeof(PyTensor),           // tp_basicsize
    0,                          // tp_itemsize
    (destructor)Tensor_dealloc, // tp_dealloc
    0,                          // tp_vectorcall_offset
    0,                          // tp_getattr
    0,                          // tp_setattr
    0,                          // tp_as_async
    (reprfunc)Tensor_repr,      // tp_repr
    0,                          // tp_as_number
    0,                          // tp_as_sequence
    0,                          // tp_as_mapping
    0,                          // tp_hash
    0,                          // tp_call
    0,                          // tp_str
    0,                          // tp_getattro
    0,                          // tp_setattro
    0,                          // tp_as_buffer
    Py_TPFLAGS_DEFAULT,         // tp_flags
    "Tensor object",            // tp_doc
    0,                          // tp_traverse
    0,                          // tp_clear
    0,                          // tp_richcompare
    0,                          // tp_weaklistoffset
    0,                          // tp_iter
    0,                          // tp_iternext
    Tensor_methods,             // tp_methods
    0,                          // tp_members
    Tensor_getset,              // tp_getset
    0,                          // tp_base
    0,                          // tp_dict
    0,                          // tp_descr_get
    0,                          // tp_descr_set
    0,                          // tp_dictoffset
    0,                          // tp_init
    0,                          // tp_alloc
    Tensor_new,                 // tp_new
};

// ============================================================
// Helper functions (now PyTensorType is defined)
// ============================================================
static Tensor* get_tensor(PyObject* obj) {
    if (!PyObject_TypeCheck(obj, &PyTensorType)) {
        PyErr_SetString(PyExc_TypeError, "Expected Tensor");
        return nullptr;
    }
    return ((PyTensor*)obj)->tensor;
}

static PyObject* make_pytensor(Tensor* t) {
    PyTensor* self = PyObject_New(PyTensor, &PyTensorType);
    if (!self) {
        delete t;
        return NULL;
    }
    self->tensor = t;
    return (PyObject*)self;
}

// ============================================================
// Type method implementations
// ============================================================
static PyObject* Tensor_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    PyTensor* self = (PyTensor*)type->tp_alloc(type, 0);
    if (self) {
        self->tensor = new Tensor();
    }
    return (PyObject*)self;
}

static void Tensor_dealloc(PyTensor* self) {
    delete self->tensor;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Tensor_shape(PyTensor* self, void* closure) {
    PyObject* shape_tuple = PyTuple_New(self->tensor->shape.size());
    for (size_t i = 0; i < self->tensor->shape.size(); i++) {
        PyTuple_SetItem(shape_tuple, i, PyLong_FromSize_t(self->tensor->shape[i]));
    }
    return shape_tuple;
}

static PyObject* Tensor_tolist(PyTensor* self, PyObject* args) {
    Tensor* t = self->tensor;
    
    if (t->shape.size() == 1) {
        PyObject* list = PyList_New(t->shape[0]);
        for (size_t i = 0; i < t->shape[0]; i++) {
            PyList_SetItem(list, i, PyFloat_FromDouble(t->data[i]));
        }
        return list;
    } else if (t->shape.size() == 2) {
        PyObject* outer = PyList_New(t->shape[0]);
        for (size_t i = 0; i < t->shape[0]; i++) {
            PyObject* inner = PyList_New(t->shape[1]);
            for (size_t j = 0; j < t->shape[1]; j++) {
                PyList_SetItem(inner, j, PyFloat_FromDouble(t->data[i * t->shape[1] + j]));
            }
            PyList_SetItem(outer, i, inner);
        }
        return outer;
    }
    
    PyErr_SetString(PyExc_NotImplementedError, "Only 1D and 2D tensors supported");
    return NULL;
}

static PyObject* Tensor_repr(PyTensor* self) {
    std::ostringstream oss;
    oss << "Tensor(shape=(";
    for (size_t i = 0; i < self->tensor->shape.size(); i++) {
        if (i > 0) oss << ", ";
        oss << self->tensor->shape[i];
    }
    oss << "), data=[";
    size_t n = std::min(self->tensor->data.size(), (size_t)6);
    for (size_t i = 0; i < n; i++) {
        if (i > 0) oss << ", ";
        oss << self->tensor->data[i];
    }
    if (self->tensor->data.size() > 6) oss << ", ...";
    oss << "])";
    return PyUnicode_FromString(oss.str().c_str());
}

// ============================================================
// Module-level functions
// ============================================================
static PyObject* tensor_zeros(PyObject* self, PyObject* args) {
    PyObject* shape_obj;
    if (!PyArg_ParseTuple(args, "O", &shape_obj)) {
        return NULL;
    }

    std::vector<size_t> shape;
    if (PyLong_Check(shape_obj)) {
        shape.push_back(PyLong_AsSize_t(shape_obj));
    } else if (PyTuple_Check(shape_obj)) {
        for (Py_ssize_t i = 0; i < PyTuple_Size(shape_obj); i++) {
            shape.push_back(PyLong_AsSize_t(PyTuple_GetItem(shape_obj, i)));
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "shape must be int or tuple");
        return NULL;
    }

    Tensor* t = new Tensor();
    t->shape = shape;
    t->data.resize(t->size(), 0.0);
    return make_pytensor(t);
}

static PyObject* tensor_from_list(PyObject* self, PyObject* args) {
    PyObject* list_obj;
    if (!PyArg_ParseTuple(args, "O", &list_obj)) {
        return NULL;
    }

    if (!PyList_Check(list_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected list");
        return NULL;
    }

    Tensor* t = new Tensor();
    Py_ssize_t size0 = PyList_Size(list_obj);
    
    PyObject* first = PyList_GetItem(list_obj, 0);
    if (PyList_Check(first)) {
        Py_ssize_t size1 = PyList_Size(first);
        t->shape = {(size_t)size0, (size_t)size1};
        t->data.resize(size0 * size1);
        
        for (Py_ssize_t i = 0; i < size0; i++) {
            PyObject* row = PyList_GetItem(list_obj, i);
            for (Py_ssize_t j = 0; j < size1; j++) {
                t->data[i * size1 + j] = PyFloat_AsDouble(PyList_GetItem(row, j));
            }
        }
    } else {
        t->shape = {(size_t)size0};
        t->data.resize(size0);
        for (Py_ssize_t i = 0; i < size0; i++) {
            t->data[i] = PyFloat_AsDouble(PyList_GetItem(list_obj, i));
        }
    }

    if (PyErr_Occurred()) {
        delete t;
        return NULL;
    }

    return make_pytensor(t);
}

static PyObject* tensor_add(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return NULL;
    }

    Tensor* a = get_tensor(a_obj);
    Tensor* b = get_tensor(b_obj);
    if (!a || !b) return NULL;

    if (!a->same_shape(*b)) {
        PyErr_SetString(PyExc_ValueError, "Shape mismatch");
        return NULL;
    }

    Tensor* result = new Tensor();
    result->shape = a->shape;
    result->data.resize(a->size());

    for (size_t i = 0; i < a->size(); i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return make_pytensor(result);
}

static PyObject* tensor_mul(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return NULL;
    }

    Tensor* a = get_tensor(a_obj);
    Tensor* b = get_tensor(b_obj);
    if (!a || !b) return NULL;

    if (!a->same_shape(*b)) {
        PyErr_SetString(PyExc_ValueError, "Shape mismatch");
        return NULL;
    }

    Tensor* result = new Tensor();
    result->shape = a->shape;
    result->data.resize(a->size());

    for (size_t i = 0; i < a->size(); i++) {
        result->data[i] = a->data[i] * b->data[i];
    }

    return make_pytensor(result);
}

static PyObject* tensor_matmul(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return NULL;
    }

    Tensor* a = get_tensor(a_obj);
    Tensor* b = get_tensor(b_obj);
    if (!a || !b) return NULL;

    if (a->shape.size() != 2 || b->shape.size() != 2) {
        PyErr_SetString(PyExc_ValueError, "matmul requires 2D tensors");
        return NULL;
    }

    size_t m = a->shape[0];
    size_t k = a->shape[1];
    size_t n = b->shape[1];

    if (k != b->shape[0]) {
        PyErr_SetString(PyExc_ValueError, "Inner dimensions must match");
        return NULL;
    }

    Tensor* result = new Tensor();
    result->shape = {m, n};
    result->data.resize(m * n, 0.0);

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t kk = 0; kk < k; kk++) {
                sum += a->data[i * k + kk] * b->data[kk * n + j];
            }
            result->data[i * n + j] = sum;
        }
    }

    return make_pytensor(result);
}

static PyObject* tensor_sum(PyObject* self, PyObject* args) {
    PyObject* a_obj;
    if (!PyArg_ParseTuple(args, "O", &a_obj)) {
        return NULL;
    }

    Tensor* a = get_tensor(a_obj);
    if (!a) return NULL;

    double sum = 0.0;
    for (double val : a->data) {
        sum += val;
    }

    return PyFloat_FromDouble(sum);
}

// ============================================================
// Module definition
// ============================================================
static PyMethodDef TensorMethods[] = {
    {"zeros", tensor_zeros, METH_VARARGS, "Create tensor of zeros"},
    {"from_list", tensor_from_list, METH_VARARGS, "Create tensor from list"},
    {"add", tensor_add, METH_VARARGS, "Element-wise addition"},
    {"mul", tensor_mul, METH_VARARGS, "Element-wise multiplication"},
    {"matmul", tensor_matmul, METH_VARARGS, "Matrix multiplication"},
    {"sum", tensor_sum, METH_VARARGS, "Sum all elements"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef tensormodule = {
    PyModuleDef_HEAD_INIT,
    "tensor",
    "Mini tensor library",
    -1,
    TensorMethods
};

PyMODINIT_FUNC PyInit_tensor(void) {
    if (PyType_Ready(&PyTensorType) < 0) {
        return NULL;
    }

    PyObject* m = PyModule_Create(&tensormodule);
    if (!m) return NULL;

    Py_INCREF(&PyTensorType);
    if (PyModule_AddObject(m, "Tensor", (PyObject*)&PyTensorType) < 0) {
        Py_DECREF(&PyTensorType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}