#include <Python.h>
#include <cmath>
#include <sstream>

// ============================================================
// Dual number: (value, derivative)
// ============================================================
struct Dual {
    double val;   // the value
    double grad;  // derivative with respect to the seeded variable

    Dual(double v = 0.0, double g = 0.0) : val(v), grad(g) {}

    // Addition: (a, a') + (b, b') = (a + b, a' + b')
    Dual operator+(const Dual& other) const {
        return Dual(val + other.val, grad + other.grad);
    }

    // Subtraction: (a, a') - (b, b') = (a - b, a' - b')
    Dual operator-(const Dual& other) const {
        return Dual(val - other.val, grad - other.grad);
    }

    // Multiplication (product rule): (a, a') * (b, b') = (a*b, a'*b + a*b')
    Dual operator*(const Dual& other) const {
        return Dual(val * other.val, grad * other.val + val * other.grad);
    }

    // Division (quotient rule): (a, a') / (b, b') = (a/b, (a'*b - a*b') / b²)
    Dual operator/(const Dual& other) const {
        double denom = other.val * other.val;
        return Dual(val / other.val, (grad * other.val - val * other.grad) / denom);
    }

    // Negation
    Dual operator-() const {
        return Dual(-val, -grad);
    }
};

// Math functions with chain rule
Dual dual_sin(const Dual& x) {
    // sin(x), derivative: cos(x) * x'
    return Dual(std::sin(x.val), std::cos(x.val) * x.grad);
}

Dual dual_cos(const Dual& x) {
    // cos(x), derivative: -sin(x) * x'
    return Dual(std::cos(x.val), -std::sin(x.val) * x.grad);
}

Dual dual_exp(const Dual& x) {
    // exp(x), derivative: exp(x) * x'
    double e = std::exp(x.val);
    return Dual(e, e * x.grad);
}

Dual dual_log(const Dual& x) {
    // log(x), derivative: (1/x) * x'
    return Dual(std::log(x.val), x.grad / x.val);
}

Dual dual_pow(const Dual& x, double n) {
    // x^n, derivative: n * x^(n-1) * x'
    return Dual(std::pow(x.val, n), n * std::pow(x.val, n - 1) * x.grad);
}

Dual dual_sqrt(const Dual& x) {
    // sqrt(x) = x^0.5, derivative: 0.5 * x^(-0.5) * x'
    double s = std::sqrt(x.val);
    return Dual(s, x.grad / (2.0 * s));
}

// ============================================================
// Python wrapper
// ============================================================
typedef struct {
    PyObject_HEAD
    Dual* dual;
} PyDual;

// Forward declarations
static PyObject* Dual_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);
static int Dual_init(PyDual* self, PyObject* args, PyObject* kwargs);
static void Dual_dealloc(PyDual* self);
static PyObject* Dual_repr(PyDual* self);
static PyObject* Dual_val(PyDual* self, void* closure);
static PyObject* Dual_grad(PyDual* self, void* closure);

// Arithmetic methods
static PyObject* Dual_add(PyObject* a, PyObject* b);
static PyObject* Dual_sub(PyObject* a, PyObject* b);
static PyObject* Dual_mul(PyObject* a, PyObject* b);
static PyObject* Dual_div(PyObject* a, PyObject* b);
static PyObject* Dual_neg(PyObject* a);

static PyNumberMethods Dual_as_number = {
    .nb_add = Dual_add,
    .nb_subtract = Dual_sub,
    .nb_multiply = Dual_mul,
    .nb_negative = Dual_neg,
    .nb_true_divide = Dual_div,
};

static PyGetSetDef Dual_getset[] = {
    {"val", (getter)Dual_val, NULL, "Value component", NULL},
    {"grad", (getter)Dual_grad, NULL, "Gradient component", NULL},
    {NULL}
};

static PyTypeObject PyDualType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "autodiff.Dual",             // tp_name
    sizeof(PyDual),              // tp_basicsize
    0,                           // tp_itemsize
    (destructor)Dual_dealloc,    // tp_dealloc
    0, 0, 0, 0,
    (reprfunc)Dual_repr,         // tp_repr
    &Dual_as_number,             // tp_as_number
    0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT,          // tp_flags
    "Dual number for autodiff",  // tp_doc
    0, 0, 0, 0, 0, 0,
    0,                           // tp_methods
    0,                           // tp_members
    Dual_getset,                 // tp_getset
    0, 0, 0, 0, 0,
    (initproc)Dual_init,         // tp_init
    0,
    Dual_new,                    // tp_new
};

// ============================================================
// Helpers
// ============================================================
static PyObject* make_pydual(const Dual& d) {
    PyDual* self = PyObject_New(PyDual, &PyDualType);
    if (!self) return NULL;
    self->dual = new Dual(d);
    return (PyObject*)self;
}

static bool get_dual(PyObject* obj, Dual& out) {
    if (PyObject_TypeCheck(obj, &PyDualType)) {
        out = *((PyDual*)obj)->dual;
        return true;
    } else if (PyFloat_Check(obj)) {
        out = Dual(PyFloat_AsDouble(obj), 0.0);  // constant: grad = 0
        return true;
    } else if (PyLong_Check(obj)) {
        out = Dual((double)PyLong_AsLong(obj), 0.0);
        return true;
    }
    return false;
}

// ============================================================
// Type method implementations
// ============================================================
static PyObject* Dual_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    PyDual* self = (PyDual*)type->tp_alloc(type, 0);
    if (self) {
        self->dual = new Dual();
    }
    return (PyObject*)self;
}

static int Dual_init(PyDual* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"val", "grad", NULL};
    double val = 0.0, grad = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|dd",
            const_cast<char**>(kwlist), &val, &grad)) {
        return -1;
    }

    self->dual->val = val;
    self->dual->grad = grad;
    return 0;
}

static void Dual_dealloc(PyDual* self) {
    delete self->dual;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Dual_repr(PyDual* self) {
    std::ostringstream oss;
    oss << "Dual(val=" << self->dual->val << ", grad=" << self->dual->grad << ")";
    return PyUnicode_FromString(oss.str().c_str());
}

static PyObject* Dual_val(PyDual* self, void* closure) {
    return PyFloat_FromDouble(self->dual->val);
}

static PyObject* Dual_grad(PyDual* self, void* closure) {
    return PyFloat_FromDouble(self->dual->grad);
}

// ============================================================
// Arithmetic operations
// ============================================================
static PyObject* Dual_add(PyObject* a, PyObject* b) {
    Dual da, db;
    if (!get_dual(a, da) || !get_dual(b, db)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return make_pydual(da + db);
}

static PyObject* Dual_sub(PyObject* a, PyObject* b) {
    Dual da, db;
    if (!get_dual(a, da) || !get_dual(b, db)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return make_pydual(da - db);
}

static PyObject* Dual_mul(PyObject* a, PyObject* b) {
    Dual da, db;
    if (!get_dual(a, da) || !get_dual(b, db)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return make_pydual(da * db);
}

static PyObject* Dual_div(PyObject* a, PyObject* b) {
    Dual da, db;
    if (!get_dual(a, da) || !get_dual(b, db)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return make_pydual(da / db);
}

static PyObject* Dual_neg(PyObject* a) {
    Dual da;
    if (!get_dual(a, da)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return make_pydual(-da);
}

// ============================================================
// Module-level math functions
// ============================================================
static PyObject* autodiff_sin(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;

    Dual d;
    if (!get_dual(obj, d)) {
        PyErr_SetString(PyExc_TypeError, "Expected Dual or number");
        return NULL;
    }
    return make_pydual(dual_sin(d));
}

static PyObject* autodiff_cos(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;

    Dual d;
    if (!get_dual(obj, d)) {
        PyErr_SetString(PyExc_TypeError, "Expected Dual or number");
        return NULL;
    }
    return make_pydual(dual_cos(d));
}

static PyObject* autodiff_exp(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;

    Dual d;
    if (!get_dual(obj, d)) {
        PyErr_SetString(PyExc_TypeError, "Expected Dual or number");
        return NULL;
    }
    return make_pydual(dual_exp(d));
}

static PyObject* autodiff_log(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;

    Dual d;
    if (!get_dual(obj, d)) {
        PyErr_SetString(PyExc_TypeError, "Expected Dual or number");
        return NULL;
    }
    return make_pydual(dual_log(d));
}

static PyObject* autodiff_pow(PyObject* self, PyObject* args) {
    PyObject* obj;
    double n;
    if (!PyArg_ParseTuple(args, "Od", &obj, &n)) return NULL;

    Dual d;
    if (!get_dual(obj, d)) {
        PyErr_SetString(PyExc_TypeError, "Expected Dual or number");
        return NULL;
    }
    return make_pydual(dual_pow(d, n));
}

static PyObject* autodiff_sqrt(PyObject* self, PyObject* args) {
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;

    Dual d;
    if (!get_dual(obj, d)) {
        PyErr_SetString(PyExc_TypeError, "Expected Dual or number");
        return NULL;
    }
    return make_pydual(dual_sqrt(d));
}

// Helper: create a variable (seeded with grad=1)
static PyObject* autodiff_var(PyObject* self, PyObject* args) {
    double val;
    if (!PyArg_ParseTuple(args, "d", &val)) return NULL;
    return make_pydual(Dual(val, 1.0));  // seed derivative = 1
}

// Helper: create a constant (grad=0)
static PyObject* autodiff_const(PyObject* self, PyObject* args) {
    double val;
    if (!PyArg_ParseTuple(args, "d", &val)) return NULL;
    return make_pydual(Dual(val, 0.0));  // derivative = 0
}

// ============================================================
// Module definition
// ============================================================
static PyMethodDef AutodiffMethods[] = {
    {"var", autodiff_var, METH_VARARGS, "Create variable (grad=1)"},
    {"const", autodiff_const, METH_VARARGS, "Create constant (grad=0)"},
    {"sin", autodiff_sin, METH_VARARGS, "Sine with autodiff"},
    {"cos", autodiff_cos, METH_VARARGS, "Cosine with autodiff"},
    {"exp", autodiff_exp, METH_VARARGS, "Exponential with autodiff"},
    {"log", autodiff_log, METH_VARARGS, "Natural log with autodiff"},
    {"pow", autodiff_pow, METH_VARARGS, "Power with autodiff"},
    {"sqrt", autodiff_sqrt, METH_VARARGS, "Square root with autodiff"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef autodiffmodule = {
    PyModuleDef_HEAD_INIT,
    "autodiff",
    "Forward-mode automatic differentiation using dual numbers",
    -1,
    AutodiffMethods
};

PyMODINIT_FUNC PyInit_autodiff(void) {
    if (PyType_Ready(&PyDualType) < 0) {
        return NULL;
    }

    PyObject* m = PyModule_Create(&autodiffmodule);
    if (!m) return NULL;

    Py_INCREF(&PyDualType);
    if (PyModule_AddObject(m, "Dual", (PyObject*)&PyDualType) < 0) {
        Py_DECREF(&PyDualType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}