#include <Python.h>
#include <marshal.h>

#define HEADER_SIZE 16

PyCodeObject* load_code_object(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Error opening file");
        return NULL;
    }

    // Read header
    unsigned char header[HEADER_SIZE];
    fread(header, sizeof(unsigned char), HEADER_SIZE, f);

    // Read the code object using marshal
    PyObject *code_obj = PyMarshal_ReadObjectFromFile(f);
    fclose(f);
    
    if (!code_obj || !PyCode_Check(code_obj)) {
        fprintf(stderr, "Failed to read a code object.\n");
        return NULL;
    }

    return (PyCodeObject *)code_obj;
}

void execute_code_object(PyCodeObject *code_obj, PyObject *globals) {
    // 確保 globals 是一個字典
    if (!globals || !PyDict_Check(globals)) {
        fprintf(stderr, "Provided globals is not a valid dictionary.\n");
        return;
    }

    // 執行代碼物件
    PyObject *result = PyEval_EvalCode((PyObject *)code_obj, globals, NULL);
    if (result) {
        // 處理結果
        Py_DECREF(result);  // 不再需要結果，釋放引用
    } else {
        PyErr_Print();  // 打印錯誤信息
    }
}

int main(int argc, char *argv[]) {
    Py_Initialize();

    // 假設 globals 是一個新的字典，並添加全域變數
    PyObject *globals = PyDict_New();
    // 在這裡可以添加全域變數，例如：
    PyDict_SetItemString(globals, "my_var", PyLong_FromLong(42));

    // 載入並執行代碼物件
    PyCodeObject *code_obj = load_code_object(argv[1]);
    if (code_obj) {
        execute_code_object(code_obj, globals);
        Py_DECREF(code_obj); // 釋放代碼物件
    }

    Py_DECREF(globals); // 釋放全域變數字典
    Py_Finalize();

    return 0;
}
