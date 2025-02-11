#define PY_SSIZE_T_CLEAN
#include <Python.h>

// 定義一個返回 "Hello, World!" 的函數
static PyObject* hello_world(PyObject* self, PyObject* args) {
    return Py_BuildValue("s", "Hello, World!");
}

// 定義模組的方法列表
static PyMethodDef HelloMethods[] = {
    {"hello", hello_world, METH_VARARGS, "Return Hello, World!"},
    {NULL, NULL, 0, NULL} // 結束標記
};

// 定義模組結構
static struct PyModuleDef hellomodule = {
    PyModuleDef_HEAD_INIT,
    "hello", // 模組名稱
    NULL, // 模組文檔
    -1, // 允許多次加載
    HelloMethods // 方法列表
};

// 模組初始化函數
PyMODINIT_FUNC PyInit_hello(void) {
    return PyModule_Create(&hellomodule);
}
