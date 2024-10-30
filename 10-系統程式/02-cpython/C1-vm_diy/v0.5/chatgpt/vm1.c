#include <Python.h>
#include <marshal.h>
#include <stdio.h>

void execute_bytecode(PyCodeObject *code) {
    Py_ssize_t i, n;
    PyObject *bytecode = PyCode_GetCode(code); // (PyObject *) code_obj->co_code_adaptive;
    Py_ssize_t bytecode_size = PyBytes_Size(bytecode);
    unsigned char *instructions = (unsigned char *)PyBytes_AsString(bytecode);

    for (i = 0; i < bytecode_size; i += 2) {  // 每個指令由 2 個字節組成
        unsigned char opcode = instructions[i];
        unsigned char oparg = instructions[i + 1];

        switch (opcode) {
            case 100: {  // LOAD_CONST
                PyObject *constant = PyTuple_GetItem(code->co_consts, oparg);
                printf("LOAD_CONST: ");
                PyObject_Print(constant, stdout, 0);
                printf("\n");
                break;
            }
            case 83:  // RETURN_VALUE
                printf("RETURN_VALUE\n");
                return;
            default:
                printf("Unknown opcode: %d\n", opcode);
                break;
        }
    }
}

int main(int argc, char *argv[]) {
    printf("main...\n");
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <pyc file>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        perror("Error opening file");
        return 1;
    }

    // Skip header (16 bytes in Python 3.12)
    fseek(file, 16, SEEK_SET);

    printf("try readObjectFromFile...\n");
    printf("try initialize...\n");
    Py_Initialize();
    printf("inintialize...\n");
    // Read the code object from .pyc file
    PyObject *code_object = PyMarshal_ReadObjectFromFile(file);
    fclose(file);

    printf("fclose():finish\n");
    if (!code_object || !PyCode_Check(code_object)) {
        fprintf(stderr, "Error: Not a valid code object\n");
        Py_XDECREF(code_object);
        return 1;
    }

    execute_bytecode((PyCodeObject*) code_object);
    Py_DECREF(code_object);
    Py_Finalize();

    return 0;
}
