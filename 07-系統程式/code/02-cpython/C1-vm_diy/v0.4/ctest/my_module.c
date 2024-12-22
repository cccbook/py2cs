#define PY_SSIZE_T_CLEAN
#include <Python.h>

// 定義函數的實現
static PyObject* my_function(PyObject* self, PyObject* args) {
    int a, b;

    // 解析傳入的參數
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL; // 返回 NULL 以指示錯誤
    }

    return PyLong_FromLong(a + b); // 返回兩個整數的和
}

// 創建函數的 PyFunctionObject
static PyObject* create_function() {
    // 使用 PyCode_NewEmpty 創建一個空的 PyCodeObject
    PyCodeObject* code_obj = PyCode_NewEmpty("my_module", "my_function", 1);
    if (code_obj == NULL) {
        return NULL; // 代碼創建失敗
    }

    // 創建一個函數物件，使用空的代碼對象
    PyObject* func = PyFunction_New((PyObject*)code_obj, NULL);
    Py_DECREF(code_obj); // 釋放代碼對象的引用

    return func; // 返回新的 PyFunctionObject
}

// 模組的初始化
static PyModuleDef my_module = {
    PyModuleDef_HEAD_INIT,
    "my_module", // 模組名稱
    NULL, // 模組文檔
    -1, // 模組狀態
    NULL, // 方法表
    NULL, // 清理函數
    NULL, // 對象
    NULL, // 不支持多線程
    NULL // 不支持多線程
};

// 模組初始化函數
PyMODINIT_FUNC PyInit_my_module(void) {
    // 創建函數並將其添加到模組中
    PyObject* func = create_function();
    PyObject* module = PyModule_Create(&my_module);

    if (module != NULL && func != NULL) {
        PyModule_AddObject(module, "my_function", func);
    }

    return module; // 返回模組對象
}

// 主函數
int main(int argc, char* argv[]) {
    // 初始化 Python 解釋器
    Py_Initialize();

    // 加載模組
    PyInit_my_module();

    // 測試函數
    PyObject* pModule = PyImport_ImportModule("my_module");
    PyObject* pFunc = PyObject_GetAttrString(pModule, "my_function");

    // 調用函數
    PyObject* pArgs = PyTuple_Pack(2, PyLong_FromLong(2), PyLong_FromLong(3));
    PyObject* pValue = PyObject_CallObject(pFunc, pArgs);

    // 打印結果
    if (pValue != NULL) {
        printf("Result: %ld\n", PyLong_AsLong(pValue));
        Py_DECREF(pValue);
    }

    // 清理
    Py_DECREF(pArgs);
    Py_DECREF(pFunc);
    Py_DECREF(pModule);
    Py_Finalize();

    return 0;
}
