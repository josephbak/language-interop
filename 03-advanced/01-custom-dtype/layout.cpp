#include <Python.h>
#include <vector>
#include <sstream>
#include <cmath>
#include <chrono>
#include <string>

// ============================================================
// Layout types
// ============================================================
enum class Layout {
    RowMajor,
    ColMajor,
    Tiled
};

// ============================================================
// LayoutTensor class
// ============================================================
struct LayoutTensor {
    std::vector<double> data;
    size_t rows;
    size_t cols;
    Layout layout;
    size_t tile_size;  // only used for Tiled layout

    // Convert logical (i, j) to physical index
    size_t index(size_t i, size_t j) const {
        switch (layout) {
            case Layout::RowMajor:
                return i * cols + j;
            
            case Layout::ColMajor:
                return j * rows + i;
            
            case Layout::Tiled: {
                // Which tile?
                size_t tile_row = i / tile_size;
                size_t tile_col = j / tile_size;
                size_t tiles_per_row = (cols + tile_size - 1) / tile_size;
                size_t tile_idx = tile_row * tiles_per_row + tile_col;
                
                // Offset within tile
                size_t local_i = i % tile_size;
                size_t local_j = j % tile_size;
                size_t local_offset = local_i * tile_size + local_j;
                
                // Final index
                return tile_idx * (tile_size * tile_size) + local_offset;
            }
        }
        return 0;
    }

    double get(size_t i, size_t j) const {
        return data[index(i, j)];
    }

    void set(size_t i, size_t j, double val) {
        data[index(i, j)] = val;
    }

    size_t size() const {
        return rows * cols;
    }
};

// ============================================================
// Python wrapper
// ============================================================
typedef struct {
    PyObject_HEAD
    LayoutTensor* tensor;
} PyLayoutTensor;

// Forward declarations
static PyObject* LayoutTensor_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);
static void LayoutTensor_dealloc(PyLayoutTensor* self);
static PyObject* LayoutTensor_repr(PyLayoutTensor* self);
static PyObject* LayoutTensor_memory_view(PyLayoutTensor* self, PyObject* args);
static PyObject* LayoutTensor_tolist(PyLayoutTensor* self, PyObject* args);
static PyObject* LayoutTensor_get(PyLayoutTensor* self, PyObject* args);
static PyObject* LayoutTensor_shape(PyLayoutTensor* self, void* closure);
static PyObject* LayoutTensor_layout_name(PyLayoutTensor* self, void* closure);

static PyMethodDef LayoutTensor_methods[] = {
    {"memory_view", (PyCFunction)LayoutTensor_memory_view, METH_NOARGS, 
     "View raw memory layout"},
    {"tolist", (PyCFunction)LayoutTensor_tolist, METH_NOARGS,
     "Convert to nested Python list"},
    {"get", (PyCFunction)LayoutTensor_get, METH_VARARGS,
     "Get element at (i, j)"},
    {NULL}
};

static PyGetSetDef LayoutTensor_getset[] = {
    {"shape", (getter)LayoutTensor_shape, NULL, "Shape (rows, cols)", NULL},
    {"layout_name", (getter)LayoutTensor_layout_name, NULL, "Layout type", NULL},
    {NULL}
};

static PyTypeObject PyLayoutTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "layout.LayoutTensor",       // tp_name
    sizeof(PyLayoutTensor),      // tp_basicsize
    0,                           // tp_itemsize
    (destructor)LayoutTensor_dealloc,
    0, 0, 0, 0,
    (reprfunc)LayoutTensor_repr,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT,
    "Tensor with configurable memory layout",
    0, 0, 0, 0, 0, 0,
    LayoutTensor_methods,
    0,
    LayoutTensor_getset,
    0, 0, 0, 0, 0, 0, 0,
    LayoutTensor_new,
};

// ============================================================
// Helper functions
// ============================================================
static LayoutTensor* get_tensor(PyObject* obj) {
    if (!PyObject_TypeCheck(obj, &PyLayoutTensorType)) {
        PyErr_SetString(PyExc_TypeError, "Expected LayoutTensor");
        return nullptr;
    }
    return ((PyLayoutTensor*)obj)->tensor;
}

static PyObject* make_pytensor(LayoutTensor* t) {
    PyLayoutTensor* self = PyObject_New(PyLayoutTensor, &PyLayoutTensorType);
    if (!self) {
        delete t;
        return NULL;
    }
    self->tensor = t;
    return (PyObject*)self;
}

static Layout parse_layout(const char* name) {
    std::string s(name);
    if (s == "row_major") return Layout::RowMajor;
    if (s == "col_major") return Layout::ColMajor;
    if (s == "tiled") return Layout::Tiled;
    return Layout::RowMajor;  // default
}

static const char* layout_to_string(Layout l) {
    switch (l) {
        case Layout::RowMajor: return "row_major";
        case Layout::ColMajor: return "col_major";
        case Layout::Tiled: return "tiled";
    }
    return "unknown";
}

// ============================================================
// Type method implementations
// ============================================================
static PyObject* LayoutTensor_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    PyLayoutTensor* self = (PyLayoutTensor*)type->tp_alloc(type, 0);
    if (self) {
        self->tensor = new LayoutTensor();
    }
    return (PyObject*)self;
}

static void LayoutTensor_dealloc(PyLayoutTensor* self) {
    delete self->tensor;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* LayoutTensor_repr(PyLayoutTensor* self) {
    LayoutTensor* t = self->tensor;
    std::ostringstream oss;
    oss << "LayoutTensor(shape=(" << t->rows << ", " << t->cols << "), "
        << "layout=" << layout_to_string(t->layout);
    if (t->layout == Layout::Tiled) {
        oss << ", tile_size=" << t->tile_size;
    }
    oss << ")";
    return PyUnicode_FromString(oss.str().c_str());
}

static PyObject* LayoutTensor_memory_view(PyLayoutTensor* self, PyObject* args) {
    LayoutTensor* t = self->tensor;
    PyObject* list = PyList_New(t->data.size());
    for (size_t i = 0; i < t->data.size(); i++) {
        PyList_SetItem(list, i, PyFloat_FromDouble(t->data[i]));
    }
    return list;
}

static PyObject* LayoutTensor_tolist(PyLayoutTensor* self, PyObject* args) {
    LayoutTensor* t = self->tensor;
    PyObject* outer = PyList_New(t->rows);
    for (size_t i = 0; i < t->rows; i++) {
        PyObject* inner = PyList_New(t->cols);
        for (size_t j = 0; j < t->cols; j++) {
            PyList_SetItem(inner, j, PyFloat_FromDouble(t->get(i, j)));
        }
        PyList_SetItem(outer, i, inner);
    }
    return outer;
}

static PyObject* LayoutTensor_get(PyLayoutTensor* self, PyObject* args) {
    int i, j;
    if (!PyArg_ParseTuple(args, "ii", &i, &j)) {
        return NULL;
    }
    return PyFloat_FromDouble(self->tensor->get(i, j));
}

static PyObject* LayoutTensor_shape(PyLayoutTensor* self, void* closure) {
    return Py_BuildValue("(nn)", self->tensor->rows, self->tensor->cols);
}

static PyObject* LayoutTensor_layout_name(PyLayoutTensor* self, void* closure) {
    return PyUnicode_FromString(layout_to_string(self->tensor->layout));
}

// ============================================================
// Module-level functions
// ============================================================

// from_list(data, layout="row_major", tile_size=2)
static PyObject* layout_from_list(PyObject* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"data", "layout", "tile_size", NULL};
    PyObject* list_obj;
    const char* layout_name = "row_major";
    int tile_size = 2;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|si", 
            const_cast<char**>(kwlist), &list_obj, &layout_name, &tile_size)) {
        return NULL;
    }

    if (!PyList_Check(list_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected list");
        return NULL;
    }

    // Parse 2D list
    Py_ssize_t rows = PyList_Size(list_obj);
    PyObject* first_row = PyList_GetItem(list_obj, 0);
    Py_ssize_t cols = PyList_Check(first_row) ? PyList_Size(first_row) : 1;

    LayoutTensor* t = new LayoutTensor();
    t->rows = rows;
    t->cols = cols;
    t->layout = parse_layout(layout_name);
    t->tile_size = tile_size;

    // For tiled layout, may need padding
    size_t data_size = rows * cols;
    if (t->layout == Layout::Tiled) {
        size_t padded_rows = ((rows + tile_size - 1) / tile_size) * tile_size;
        size_t padded_cols = ((cols + tile_size - 1) / tile_size) * tile_size;
        data_size = padded_rows * padded_cols;
    }
    t->data.resize(data_size, 0.0);

    // Fill data using layout-aware indexing
    for (Py_ssize_t i = 0; i < rows; i++) {
        PyObject* row = PyList_GetItem(list_obj, i);
        for (Py_ssize_t j = 0; j < cols; j++) {
            double val = PyFloat_AsDouble(PyList_GetItem(row, j));
            t->set(i, j, val);
        }
    }

    if (PyErr_Occurred()) {
        delete t;
        return NULL;
    }

    return make_pytensor(t);
}

// zeros(rows, cols, layout="row_major", tile_size=2)
static PyObject* layout_zeros(PyObject* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"rows", "cols", "layout", "tile_size", NULL};
    int rows, cols;
    const char* layout_name = "row_major";
    int tile_size = 2;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|si",
            const_cast<char**>(kwlist), &rows, &cols, &layout_name, &tile_size)) {
        return NULL;
    }

    LayoutTensor* t = new LayoutTensor();
    t->rows = rows;
    t->cols = cols;
    t->layout = parse_layout(layout_name);
    t->tile_size = tile_size;

    size_t data_size = rows * cols;
    if (t->layout == Layout::Tiled) {
        size_t padded_rows = ((rows + tile_size - 1) / tile_size) * tile_size;
        size_t padded_cols = ((cols + tile_size - 1) / tile_size) * tile_size;
        data_size = padded_rows * padded_cols;
    }
    t->data.resize(data_size, 0.0);

    return make_pytensor(t);
}

// benchmark_row_sum(tensor) - sum by iterating rows first
static PyObject* layout_benchmark_row_sum(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }

    LayoutTensor* t = get_tensor(obj);
    if (!t) return NULL;

    auto start = std::chrono::high_resolution_clock::now();
    
    double sum = 0.0;
    for (int iter = 0; iter < 1000; iter++) {
        sum = 0.0;
        for (size_t i = 0; i < t->rows; i++) {
            for (size_t j = 0; j < t->cols; j++) {
                sum += t->get(i, j);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    return Py_BuildValue("{s:d,s:d}", "sum", sum, "time_ms", ms);
}

// benchmark_col_sum(tensor) - sum by iterating columns first
static PyObject* layout_benchmark_col_sum(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }

    LayoutTensor* t = get_tensor(obj);
    if (!t) return NULL;

    auto start = std::chrono::high_resolution_clock::now();
    
    double sum = 0.0;
    for (int iter = 0; iter < 1000; iter++) {
        sum = 0.0;
        for (size_t j = 0; j < t->cols; j++) {
            for (size_t i = 0; i < t->rows; i++) {
                sum += t->get(i, j);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    return Py_BuildValue("{s:d,s:d}", "sum", sum, "time_ms", ms);
}

// benchmark_raw_sequential(tensor) - sum raw memory order
static PyObject* layout_benchmark_raw_sequential(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }

    LayoutTensor* t = get_tensor(obj);
    if (!t) return NULL;

    auto start = std::chrono::high_resolution_clock::now();
    
    double sum = 0.0;
    for (int iter = 0; iter < 1000; iter++) {
        sum = 0.0;
        for (size_t i = 0; i < t->data.size(); i++) {
            sum += t->data[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    return Py_BuildValue("{s:d,s:d}", "sum", sum, "time_ms", ms);
}

// ============================================================
// Module definition
// ============================================================
static PyMethodDef LayoutMethods[] = {
    {"from_list", (PyCFunction)layout_from_list, METH_VARARGS | METH_KEYWORDS,
     "Create tensor from list with specified layout"},
    {"zeros", (PyCFunction)layout_zeros, METH_VARARGS | METH_KEYWORDS,
     "Create zero tensor with specified layout"},
    {"benchmark_row_sum", layout_benchmark_row_sum, METH_VARARGS,
     "Benchmark summing by rows"},
    {"benchmark_col_sum", layout_benchmark_col_sum, METH_VARARGS,
     "Benchmark summing by columns"},
    {"benchmark_raw_sequential", layout_benchmark_raw_sequential, METH_VARARGS,
     "Benchmark summing in raw memory order"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef layoutmodule = {
    PyModuleDef_HEAD_INIT,
    "layout",
    "Tensor with configurable memory layouts",
    -1,
    LayoutMethods
};

PyMODINIT_FUNC PyInit_layout(void) {
    if (PyType_Ready(&PyLayoutTensorType) < 0) {
        return NULL;
    }

    PyObject* m = PyModule_Create(&layoutmodule);
    if (!m) return NULL;

    Py_INCREF(&PyLayoutTensorType);
    if (PyModule_AddObject(m, "LayoutTensor", (PyObject*)&PyLayoutTensorType) < 0) {
        Py_DECREF(&PyLayoutTensorType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}