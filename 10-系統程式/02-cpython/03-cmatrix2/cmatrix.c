#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject* matrix_add(PyObject* self, PyObject* args) {
    PyObject *matrix_a, *matrix_b;
    Py_ssize_t rows_a, cols_a, rows_b, cols_b;

    // 解析 Python 的輸入參數
    if (!PyArg_ParseTuple(args, "OO", &matrix_a, &matrix_b)) {
        return NULL;
    }

    // 獲取矩陣的大小
    rows_a = PyList_Size(matrix_a);
    cols_a = PyList_Size(PyList_GetItem(matrix_a, 0));
    rows_b = PyList_Size(matrix_b);
    cols_b = PyList_Size(PyList_GetItem(matrix_b, 0));

    // 檢查矩陣大小是否相同
    if (rows_a != rows_b || cols_a != cols_b) {
        PyErr_SetString(PyExc_ValueError, "Matrices must have the same dimensions");
        return NULL;
    }

    // 創建返回結果矩陣
    PyObject *result = PyList_New(rows_a);
    for (Py_ssize_t i = 0; i < rows_a; i++) {
        PyObject *row = PyList_New(cols_a);
        for (Py_ssize_t j = 0; j < cols_a; j++) {
            PyObject *item_a = PyList_GetItem(PyList_GetItem(matrix_a, i), j);
            PyObject *item_b = PyList_GetItem(PyList_GetItem(matrix_b, i), j);
            double sum = PyFloat_AsDouble(item_a) + PyFloat_AsDouble(item_b);
            PyList_SetItem(row, j, PyFloat_FromDouble(sum));
        }
        PyList_SetItem(result, i, row);
    }

    return result;
}

static PyObject* matrix_sub(PyObject* self, PyObject* args) {
    PyObject *matrix_a, *matrix_b;
    Py_ssize_t rows_a, cols_a, rows_b, cols_b;

    // 解析 Python 的輸入參數
    if (!PyArg_ParseTuple(args, "OO", &matrix_a, &matrix_b)) {
        return NULL;
    }

    // 獲取矩陣的大小
    rows_a = PyList_Size(matrix_a);
    cols_a = PyList_Size(PyList_GetItem(matrix_a, 0));
    rows_b = PyList_Size(matrix_b);
    cols_b = PyList_Size(PyList_GetItem(matrix_b, 0));

    // 檢查矩陣大小是否相同
    if (rows_a != rows_b || cols_a != cols_b) {
        PyErr_SetString(PyExc_ValueError, "Matrices must have the same dimensions");
        return NULL;
    }

    // 創建返回結果矩陣
    PyObject *result = PyList_New(rows_a);
    for (Py_ssize_t i = 0; i < rows_a; i++) {
        PyObject *row = PyList_New(cols_a);
        for (Py_ssize_t j = 0; j < cols_a; j++) {
            PyObject *item_a = PyList_GetItem(PyList_GetItem(matrix_a, i), j);
            PyObject *item_b = PyList_GetItem(PyList_GetItem(matrix_b, i), j);
            double sum = PyFloat_AsDouble(item_a) - PyFloat_AsDouble(item_b);
            PyList_SetItem(row, j, PyFloat_FromDouble(sum));
        }
        PyList_SetItem(result, i, row);
    }

    return result;
}

static PyMethodDef MatrixMethods[] = {
    {"add", matrix_add, METH_VARARGS, "Add two matrices."},
    {"sub", matrix_sub, METH_VARARGS, "Substract two matrices."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef matrixmodule = {
    PyModuleDef_HEAD_INIT,
    "cmatrix",// 模組名稱
    NULL, // 模組文檔
    -1,// 允許多次加載
    MatrixMethods // 方法列表
};

PyMODINIT_FUNC PyInit_cmatrix(void) {
    return PyModule_Create(&matrixmodule);
}
