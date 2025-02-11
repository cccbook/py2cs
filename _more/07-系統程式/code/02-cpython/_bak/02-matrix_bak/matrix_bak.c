#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
    PyObject_HEAD
    int rows;
    int cols;
    double* data;
} Matrix;

// 釋放矩陣的內存
static void Matrix_dealloc(Matrix* self) {
    free(self->data);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// 創建矩陣
static PyObject* Matrix_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    Matrix* self;
    self = (Matrix*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

// 初始化矩陣
static int Matrix_init(Matrix* self, PyObject* args) {
    if (!PyArg_ParseTuple(args, "ii", &self->rows, &self->cols)) {
        return -1;
    }
    self->data = (double*)malloc(self->rows * self->cols * sizeof(double));
    for (int i = 0; i < self->rows * self->cols; ++i) {
        self->data[i] = 0.0; // 初始化為0
    }
    return 0;
}

// 矩陣加法
static PyObject* Matrix_add(PyObject* a, PyObject* b) {
    Matrix* mat1 = (Matrix*)a;
    Matrix* mat2 = (Matrix*)b;

    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        PyErr_SetString(PyExc_ValueError, "Matrices must have the same dimensions");
        return NULL;
    }

    Matrix* result = (Matrix*)Matrix_new(&PyType_Type, NULL, NULL);
    result->rows = mat1->rows;
    result->cols = mat1->cols;
    result->data = (double*)malloc(result->rows * result->cols * sizeof(double));

    for (int i = 0; i < result->rows * result->cols; ++i) {
        result->data[i] = mat1->data[i] + mat2->data[i];
    }

    return (PyObject*)result;
}

// 矩陣乘法
static PyObject* Matrix_mul(PyObject* a, PyObject* b) {
    Matrix* mat1 = (Matrix*)a;
    Matrix* mat2 = (Matrix*)b;

    if (mat1->cols != mat2->rows) {
        PyErr_SetString(PyExc_ValueError, "Invalid matrix dimensions for multiplication");
        return NULL;
    }

    Matrix* result = (Matrix*)Matrix_new(&PyType_Type, NULL, NULL);
    result->rows = mat1->rows;
    result->cols = mat2->cols;
    result->data = (double*)malloc(result->rows * result->cols * sizeof(double));

    for (int i = 0; i < result->rows; ++i) {
        for (int j = 0; j < result->cols; ++j) {
            result->data[i * result->cols + j] = 0.0;
            for (int k = 0; k < mat1->cols; ++k) {
                result->data[i * result->cols + j] += mat1->data[i * mat1->cols + k] * mat2->data[k * mat2->cols + j];
            }
        }
    }

    return (PyObject*)result;
}

// 定義類型方法
static PyMethodDef Matrix_methods[] = {
    {NULL, NULL, 0, NULL} // 結束標記
};

// 定義矩陣類型
static PyTypeObject MatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "matrix.Matrix",             // tp_name
    sizeof(Matrix),              // tp_basicsize
    0,                           // tp_itemsize
    (destructor)Matrix_dealloc, // tp_dealloc
    0,                           // tp_print
    0,                           // tp_getattr
    0,                           // tp_setattr
    0,                           // tp_reserved
    0,                           // tp_repr
    0,                           // tp_as_number
    0,                           // tp_as_sequence
    0,                           // tp_as_mapping
    0,                           // tp_hash
    0,                           // tp_call
    0,                           // tp_str
    0,                           // tp_getattro
    0,                           // tp_setattro
    0,                           // tp_as_buffer
    Py_TPFLAGS_DEFAULT,         // tp_flags
    "Matrix objects",           // tp_doc
    0,                           // tp_traverse
    0,                           // tp_clear
    0,                           // tp_richcompare
    0,                           // tp_weaklistoffset
    0,                           // tp_iter
    0,                           // tp_iternext
    Matrix_methods,             // tp_methods
    0,                           // tp_members
    0,                           // tp_getset
    0,                           // tp_base
    0,                           // tp_dict
    0,                           // tp_descr_get
    0,                           // tp_descr_set
    0,                           // tp_dictoffset
    (initproc)Matrix_init,      // tp_init
    0,                           // tp_alloc
    Matrix_new,                 // tp_new
};

// 定義模組方法
static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL} // 結束標記
};

// 模組初始化函數
static struct PyModuleDef matrixmodule = {
    PyModuleDef_HEAD_INIT,
    "matrix", // 模組名稱
    NULL,     // 模組文檔
    -1,       // 允許多次加載
    module_methods
};

// 初始化模組
PyMODINIT_FUNC PyInit_matrix(void) {
    PyObject* m;

    if (PyType_Ready(&MatrixType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&matrixmodule);
    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&MatrixType);
    PyModule_AddObject(m, "Matrix", (PyObject*)&MatrixType);
    return m;
}
