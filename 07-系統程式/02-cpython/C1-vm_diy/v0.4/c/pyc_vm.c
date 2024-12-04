#include <Python.h>
#include <marshal.h>
#include <opcode.h>  // 用於操作碼
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "op.h"

#define HEADER_SIZE 16  // Python 3.12 header size

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

void dump_code_object(PyCodeObject *code_obj) {
    printf("Disassembling code object: %s\n", PyUnicode_AsUTF8(code_obj->co_name));
    printf("Filename: %s\n", PyUnicode_AsUTF8(code_obj->co_filename));
    printf("First Line Number: %d\n", code_obj->co_firstlineno);

    Py_ssize_t size;
    size = PyTuple_Size(code_obj->co_consts);
    PyObject **constant_objs = malloc(sizeof(PyObject*)*size);
    // Print constants
    printf("Constants:\n");
    for (Py_ssize_t i = 0; i < PyTuple_Size(code_obj->co_consts); i++) {
        PyObject *const_obj = PyTuple_GetItem(code_obj->co_consts, i);
        constant_objs[i] = const_obj;
        // printf("  [%zd] %s", i, PyUnicode_Check(const_obj) ? 
        //      PyUnicode_AsUTF8(const_obj) : "Non-string constant");
        PyTypeObject *type = Py_TYPE(const_obj);
        printf("%4zd: %s:", i, type->tp_name);
        if (strcmp(type->tp_name, "int")==0) {
            long ivalue = PyLong_AsLong(const_obj);
            printf("%ld ", ivalue);
        } else if (strcmp(type->tp_name, "float")==0) {
            double fvalue = PyFloat_AsDouble(const_obj);
            printf("%f ", fvalue);
        } else if (strcmp(type->tp_name, "str")==0) {
            const char *svalue = PyUnicode_AsUTF8(const_obj);
            printf("%s ", svalue);
        } else if (strcmp(type->tp_name, "code")==0) {
            PyObject_Print(const_obj, stdout, 0);
        } else if (strcmp(type->tp_name, "NoneType")==0) {
            printf("None ");
        }
        printf("\n");
    }
    
    size = PyTuple_Size(code_obj->co_names);
    const char **names = malloc(sizeof(char*)*size);
    // Print names
    printf("Names:\n");
    for (Py_ssize_t i = 0; i < PyTuple_Size(code_obj->co_names); i++) {
        PyObject *name_item = PyTuple_GetItem(code_obj->co_names, i);
        printf("%4zd: %s\n", i, PyUnicode_AsUTF8(name_item));
        names[i] = PyUnicode_AsUTF8(name_item);
    }

    // Print the bytecode
    printf("Bytecode:\n");
    PyObject *bytecode = PyCode_GetCode(code_obj); // (PyObject *) code_obj->co_code_adaptive;
    Py_ssize_t bytecode_size = PyBytes_Size(bytecode);
    unsigned char *code = (unsigned char *)PyBytes_AsString(bytecode);

    for (Py_ssize_t i = 0; i < bytecode_size; ) {
        Py_ssize_t addr = i;
        int opcode = code[i++];
        int oparg = code[i++];
        printf("%4zd: %s %d \t # ", addr, op_names[opcode], oparg);  // 使用 Python 的操作碼名稱 (這個在 internal/pycore_opcode_metadata 引用不到)
        PyObject *arg_obj = NULL;
        switch (opcode) {
            case LOAD_CONST:
            case RETURN_CONST:
                arg_obj = PyTuple_GetItem(code_obj->co_consts, oparg);
                break;
            case LOAD_NAME:
            case STORE_NAME:
            case DELETE_NAME:
            case IMPORT_NAME:
                arg_obj = PyTuple_GetItem(code_obj->co_names, oparg);
                break;
            default:
                printf("");
        }
        if (opcode == CALL) i+=6; // CALL 指令占 8byte (不知為何？)

        if (arg_obj)
            PyObject_Print(arg_obj, stdout, 0);
        printf("\n");
        // i++;
    }
}

#define NSTACK 10000

PyObject *stack[NSTACK];
Py_ssize_t stack_top = 0;
PyObject *globals = NULL;

void run_code_object(PyCodeObject *code_obj) {
    printf("run_code_object()...\n");
    globals = PyDict_New();    

    Py_ssize_t const_size = PyTuple_Size(code_obj->co_consts);
    Py_ssize_t name_size = PyTuple_Size(code_obj->co_names);
    // PyObject *name_item = PyTuple_GetItem(code_obj->co_names, i);
    PyObject *bytecode = PyCode_GetCode(code_obj); // (PyObject *) code_obj->co_code_adaptive;
    Py_ssize_t bytecode_size = PyBytes_Size(bytecode);
    unsigned char *code = (unsigned char *)PyBytes_AsString(bytecode);
    PyObject *co_consts = code_obj->co_consts;
    PyObject *co_names = code_obj->co_names;

    printf("vm start...\n");
    stack_top = 0;
    Py_ssize_t pc = 0;
    while (pc < bytecode_size) {
        // printf("looping...\n");
        Py_ssize_t addr = pc;
        int opcode = code[pc++];
        int oparg = code[pc++];
        // printf("opcode=%d op_name=%s oparg=%d\n", opcode, op_names[opcode], oparg);
        printf("%4zd: %s %d \t # ", addr, op_names[opcode], oparg);  // 使用 Python 的操作碼名稱 (這個在 internal/pycore_opcode_metadata 引用不到)
        PyObject *arg_obj = NULL;
        switch (opcode) {
            case LOAD_CONST:
                stack[stack_top++] = arg_obj = PyTuple_GetItem(co_consts, oparg);
                break;
            case RETURN_CONST:
                arg_obj = PyTuple_GetItem(co_consts, oparg);
                // ???
                break;
            case LOAD_NAME:
                stack[stack_top++] = arg_obj = PyTuple_GetItem(co_names, oparg);
                break;
            case STORE_NAME:
                arg_obj = stack[--stack_top];
                PyTuple_SetItem(co_names, oparg, arg_obj); 
                break;
            case DELETE_NAME:
                Py_DECREF(PyTuple_GetItem(co_names, oparg)); 
                PyTuple_SetItem(co_names, oparg, Py_None);
                Py_INCREF(Py_None);
                break;
            case PUSH_NULL:
                stack[stack_top++] = NULL;
                break;
            case POP_TOP:
                stack_top--;
                break;
            case IMPORT_NAME:
                arg_obj = PyTuple_GetItem(co_names, oparg);
                // ???
                break;
            case MAKE_FUNCTION:
                stack[stack_top++] = PyFunction_New((PyObject*)code_obj, globals);
                break;
            case RESUME:
                printf(" 待處理 ...");
                break;
            case CALL:
                printf(" 待處理 ...");
                break;
            default:
                printf("未支援的 opcode: %d %s", opcode, op_names[opcode]);
                exit(1);
        }
        if (opcode == CALL) pc+=6; // CALL 指令占 8byte (不知為何？)

        if (arg_obj)
            PyObject_Print(arg_obj, stdout, 0);
        printf("\n");
    }
}

// 呼叫物件的成員函數，例如假設我們呼叫 "method_name" 並傳遞一個參數,  "(O)"  代表傳遞一個參數，是物件 (Object)
//    PyObject *result = PyObject_CallMethod(obj, "method_name", "(O)", PyLong_FromLong(5));

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <pyc file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Initialize Python interpreter
    Py_Initialize();
    printf("global=%p\n", PyEval_GetGlobals());
    PyCodeObject *code_obj = load_code_object(argv[1]);
    if (code_obj) {
        dump_code_object(code_obj);
        run_code_object(code_obj);
        Py_DECREF(code_obj);  // Decrease reference count
    }

    // Finalize Python interpreter
    Py_Finalize();
    return EXIT_SUCCESS;
}

