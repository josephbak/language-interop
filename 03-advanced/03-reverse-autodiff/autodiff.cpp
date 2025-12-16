#include <Python.h>
#include <vector>
#include <functional>
#include <memory>
#include <cmath>
#include <sstream>
#include <set>
#include <algorithm>

// Forward declaration
struct Var;

// ============================================================
// BackwardEdge: stores how to propagate gradient to an input
// ============================================================
struct BackwardEdge {
    Var* input;
    std::function<double()> grad_fn;  // computes gradient contribution
    
    BackwardEdge(Var* inp, std::function<double()> fn) 
        : input(inp), grad_fn(fn) {}
};

// ============================================================
// Var: computation graph node
// ============================================================
struct Var {
    double val;
    double grad;
    std::vector<BackwardEdge> backward_edges;
    
    Var(double v) : val(v), grad(0.0) {}
    
    // Add edge: "when I backprop, call this function and add to input's grad"
    void add_edge(Var* input, std::function<double()> grad_fn) {
        backward_edges.emplace_back(input, grad_fn);
    }
    
    // Topological sort helper
    void topo_sort(std::vector<Var*>& order, std::set<Var*>& visited) {
        if (visited.count(this)) return;
        visited.insert(this);
        
        for (auto& edge : backward_edges) {
            edge.input->topo_sort(order, visited);
        }
        
        order.push_back(this);
    }
    
    // Backpropagation: propagate gradients through the graph
    void backward() {
        // Topological sort (reverse of execution order)
        std::vector<Var*> order;
        std::set<Var*> visited;
        topo_sort(order, visited);
        
        // Seed this node with grad = 1
        this->grad = 1.0;
        
        // Propagate gradients in reverse topological order
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            Var* node = *it;
            for (auto& edge : node->backward_edges) {
                double grad_contribution = edge.grad_fn();
                edge.input->grad += grad_contribution;
            }
        }
    }
    
    // Zero all gradients in the graph
    void zero_grad() {
        std::set<Var*> visited;
        zero_grad_helper(visited);
    }
    
    void zero_grad_helper(std::set<Var*>& visited) {
        if (visited.count(this)) return;
        visited.insert(this);
        grad = 0.0;
        for (auto& edge : backward_edges) {
            edge.input->zero_grad_helper(visited);
        }
    }
};

// ============================================================
// Operations that build the computation graph
// ============================================================

Var* op_add(Var* a, Var* b) {
    Var* result = new Var(a->val + b->val);
    
    // da = dresult * 1
    result->add_edge(a, [result]() { return result->grad * 1.0; });
    
    // db = dresult * 1
    result->add_edge(b, [result]() { return result->grad * 1.0; });
    
    return result;
}

Var* op_sub(Var* a, Var* b) {
    Var* result = new Var(a->val - b->val);
    
    result->add_edge(a, [result]() { return result->grad * 1.0; });
    result->add_edge(b, [result]() { return result->grad * -1.0; });
    
    return result;
}

Var* op_mul(Var* a, Var* b) {
    Var* result = new Var(a->val * b->val);
    
    // Product rule: d(a*b)/da = b
    result->add_edge(a, [result, b]() { return result->grad * b->val; });
    
    // d(a*b)/db = a
    result->add_edge(b, [result, a]() { return result->grad * a->val; });
    
    return result;
}

Var* op_div(Var* a, Var* b) {
    Var* result = new Var(a->val / b->val);
    
    // d(a/b)/da = 1/b
    result->add_edge(a, [result, b]() { return result->grad / b->val; });
    
    // d(a/b)/db = -a/b²
    result->add_edge(b, [result, a, b]() { 
        return result->grad * (-a->val / (b->val * b->val)); 
    });
    
    return result;
}

Var* op_neg(Var* a) {
    Var* result = new Var(-a->val);
    result->add_edge(a, [result]() { return result->grad * -1.0; });
    return result;
}

Var* op_pow(Var* a, double n) {
    Var* result = new Var(std::pow(a->val, n));
    
    // d(x^n)/dx = n * x^(n-1)
    result->add_edge(a, [result, a, n]() {
        return result->grad * n * std::pow(a->val, n - 1);
    });
    
    return result;
}

Var* op_sin(Var* a) {
    Var* result = new Var(std::sin(a->val));
    
    // d(sin(x))/dx = cos(x)
    result->add_edge(a, [result, a]() {
        return result->grad * std::cos(a->val);
    });
    
    return result;
}

Var* op_cos(Var* a) {
    Var* result = new Var(std::cos(a->val));
    
    // d(cos(x))/dx = -sin(x)
    result->add_edge(a, [result, a]() {
        return result->grad * (-std::sin(a->val));
    });
    
    return result;
}

Var* op_exp(Var* a) {
    Var* result = new Var(std::exp(a->val));
    
    // d(exp(x))/dx = exp(x)
    result->add_edge(a, [result]() {
        return result->grad * result->val;  // exp(x) is already computed
    });
    
    return result;
}

Var* op_log(Var* a) {
    Var* result = new Var(std::log(a->val));
    
    // d(log(x))/dx = 1/x
    result->add_edge(a, [result, a]() {
        return result->grad / a->val;
    });
    
    return result;
}

// ============================================================
// Python wrapper
// ============================================================
typedef struct {
    PyObject_HEAD
    Var* var;
} PyVar;

// Forward declarations
static PyObject* Var_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);
static int Var_init(PyVar* self, PyObject* args, PyObject* kwargs);
static void Var_dealloc(PyVar* self);
static PyObject* Var_repr(PyVar* self);
static PyObject* Var_val(PyVar* self, void* closure);
static PyObject* Var_grad(PyVar* self, void* closure);
static PyObject* Var_backward(PyVar* self, PyObject* args);
static PyObject* Var_zero_grad(PyVar* self, PyObject* args);

// Arithmetic
static PyObject* Var_add(PyObject* a, PyObject* b);
static PyObject* Var_sub(PyObject* a, PyObject* b);
static PyObject* Var_mul(PyObject* a, PyObject* b);
static PyObject* Var_div(PyObject* a, PyObject* b);
static PyObject* Var_neg(PyObject* a);

static PyMethodDef Var_methods[] = {
    {"backward", (PyCFunction)Var_backward, METH_NOARGS, 
     "Compute gradients via backpropagation"},
    {"zero_grad", (PyCFunction)Var_zero_grad, METH_NOARGS,
     "Zero all gradients in computation graph"},
    {NULL}
};

static PyGetSetDef Var_getset[] = {
    {"val", (getter)Var_val, NULL, "Value", NULL},
    {"grad", (getter)Var_grad, NULL, "Gradient", NULL},
    {NULL}
};

static PyNumberMethods Var_as_number = {
    .nb_add = Var_add,
    .nb_subtract = Var_sub,
    .nb_multiply = Var_mul,
    .nb_negative = Var_neg,
    .nb_true_divide = Var_div,
};

static PyTypeObject PyVarType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "autodiff.Var",              // tp_name
    sizeof(PyVar),               // tp_basicsize
    0,                           // tp_itemsize
    (destructor)Var_dealloc,     // tp_dealloc
    0, 0, 0, 0,
    (reprfunc)Var_repr,          // tp_repr
    &Var_as_number,              // tp_as_number
    0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT,          // tp_flags
    "Variable for reverse-mode autodiff",
    0, 0, 0, 0, 0, 0,
    Var_methods,
    0,
    Var_getset,
    0, 0, 0, 0, 0,
    (initproc)Var_init,
    0,
    Var_new,
};

// ============================================================
// Helpers
// ============================================================
static PyObject* make_pyvar(Var* v) {
    PyVar* self = PyObject_New(PyVar, &PyVarType);
    if (!self) {
        delete v;
        return NULL;
    }
    self->var = v;
    return (PyObject*)self;
}

static bool get_var(PyObject* obj, Var*& out) {
    if (PyObject_TypeCheck(obj, &PyVarType)) {
        out = ((PyVar*)obj)->var;
        return true;
    } else if (PyFloat_Check(obj) || PyLong_Check(obj)) {
        double val = PyFloat_Check(obj) ? PyFloat_AsDouble(obj) : (double)PyLong_AsLong(obj);
        out = new Var(val);  // constant (no edges)
        return true;
    }
    return false;
}

// ============================================================
// Type method implementations
// ============================================================
static PyObject* Var_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    PyVar* self = (PyVar*)type->tp_alloc(type, 0);
    if (self) {
        self->var = nullptr;  // will be set in init
    }
    return (PyObject*)self;
}

static int Var_init(PyVar* self, PyObject* args, PyObject* kwargs) {
    double val;
    if (!PyArg_ParseTuple(args, "d", &val)) {
        return -1;
    }
    self->var = new Var(val);
    return 0;
}

static void Var_dealloc(PyVar* self) {
    // Note: This is simplified. In production, need proper graph cleanup
    delete self->var;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Var_repr(PyVar* self) {
    std::ostringstream oss;
    oss << "Var(val=" << self->var->val << ", grad=" << self->var->grad << ")";
    return PyUnicode_FromString(oss.str().c_str());
}

static PyObject* Var_val(PyVar* self, void* closure) {
    return PyFloat_FromDouble(self->var->val);
}

static PyObject* Var_grad(PyVar* self, void* closure) {
    return PyFloat_FromDouble(self->var->grad);
}

static PyObject* Var_backward(PyVar* self, PyObject* args) {
    self->var->backward();
    Py_RETURN_NONE;
}

static PyObject* Var_zero_grad(PyVar* self, PyObject* args) {
    self->var->zero_grad();
    Py_RETURN_NONE;
}

// ============================================================
// Arithmetic operations
// ============================================================
static PyObject* Var_add(PyObject* a, PyObject* b) {
    Var *va, *vb;
    if (!get_var(a, va) || !get_var(b, vb)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return make_pyvar(op_add(va, vb));
}

static PyObject* Var_sub(PyObject* a, PyObject* b) {
    Var *va, *vb;
    if (!get_var(a, va) || !get_var(b, vb)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return make_pyvar(op_sub(va, vb));
}

static PyObject* Var_mul(PyObject* a, PyObject* b) {
    Var *va, *vb;
    if (!get_var(a, va) || !get_var(b, vb)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return make_pyvar(op_mul(va, vb));
}

static PyObject* Var_div(PyObject* a, PyObject* b) {
    Var *va, *vb;
    if (!get_var(a, va) || !get_var(b, vb)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return make_pyvar(op_div(va, vb));
}

static PyObject* Var_neg(PyObject* a) {
    Var* va;
    if (!get_var(a, va)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return make_pyvar(op_neg(va));
}

// ============================================================
// Module-level functions
// ============================================================
static PyObject* autodiff_sin(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;

    Var* v;
    if (!get_var(obj, v)) {
        PyErr_SetString(PyExc_TypeError, "Expected Var or number");
        return NULL;
    }
    return make_pyvar(op_sin(v));
}

static PyObject* autodiff_cos(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;

    Var* v;
    if (!get_var(obj, v)) {
        PyErr_SetString(PyExc_TypeError, "Expected Var or number");
        return NULL;
    }
    return make_pyvar(op_cos(v));
}

static PyObject* autodiff_exp(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;

    Var* v;
    if (!get_var(obj, v)) {
        PyErr_SetString(PyExc_TypeError, "Expected Var or number");
        return NULL;
    }
    return make_pyvar(op_exp(v));
}

static PyObject* autodiff_log(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;

    Var* v;
    if (!get_var(obj, v)) {
        PyErr_SetString(PyExc_TypeError, "Expected Var or number");
        return NULL;
    }
    return make_pyvar(op_log(v));
}

static PyObject* autodiff_pow(PyObject* self, PyObject* args) {
    PyObject* obj;
    double n;
    if (!PyArg_ParseTuple(args, "Od", &obj, &n)) return NULL;

    Var* v;
    if (!get_var(obj, v)) {
        PyErr_SetString(PyExc_TypeError, "Expected Var or number");
        return NULL;
    }
    return make_pyvar(op_pow(v, n));
}

// ============================================================
// Module definition
// ============================================================
static PyMethodDef AutodiffMethods[] = {
    {"sin", autodiff_sin, METH_VARARGS, "Sine"},
    {"cos", autodiff_cos, METH_VARARGS, "Cosine"},
    {"exp", autodiff_exp, METH_VARARGS, "Exponential"},
    {"log", autodiff_log, METH_VARARGS, "Natural log"},
    {"pow", autodiff_pow, METH_VARARGS, "Power"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef autodiffmodule = {
    PyModuleDef_HEAD_INIT,
    "autodiff",
    "Reverse-mode automatic differentiation (backpropagation)",
    -1,
    AutodiffMethods
};

PyMODINIT_FUNC PyInit_autodiff(void) {
    if (PyType_Ready(&PyVarType) < 0) {
        return NULL;
    }

    PyObject* m = PyModule_Create(&autodiffmodule);
    if (!m) return NULL;

    Py_INCREF(&PyVarType);
    if (PyModule_AddObject(m, "Var", (PyObject*)&PyVarType) < 0) {
        Py_DECREF(&PyVarType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}