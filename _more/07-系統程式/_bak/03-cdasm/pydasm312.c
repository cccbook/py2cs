#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define HEADER_SIZE 16

// Python 3.12 的指令對應表
const char* opcodes[] = {
    "CACHE", "POP_TOP", "PUSH_NULL", "INTERPRETER_EXIT", "END_FOR", "END_SEND", 
    "?", "?", "?", "NOP", "?", "UNARY_NEGATIVE", 
    "UNARY_NOT", "?", "?", "UNARY_INVERT", "?", "RESERVED",
    "?", "?", "?", "?", "?", "?", 
    "?", "BINARY_SUBSCR", "BINARY_SLICE", "STORE_SLICE", "?", 
    "?", "GET_LEN", "MATCH_MAPPING", "MATCH_SEQUENCE", "MATCH_KEYS", 
    "?", "PUSH_EXC_INFO", "CHECK_EXC_MATCH", "CHECK_EG_MATCH", "?", 
    "?", "?", "?", "?", "?", "?", 
    "?", "?", "WITH_EXCEPT_START", "GET_AITER", "GET_ANEXT", 
    "BEFORE_ASYNC_WITH", "BEFORE_WITH", "END_ASYNC_FOR", "CLEANUP_THROW", "?", 
    "?", "?", "?", "STORE_SUBSCR", "DELETE_SUBSCR", "?", 
    "?", "?", "?", "?", "GET_ITER", "GET_YIELD_FROM_ITER", 
    "?", "LOAD_BUILD_CLASS", "?", "?", "LOAD_ASSERTION_ERROR", 
    "RETURN_GENERATOR", "?", "?", "?", "?", "?", 
    "?", "?", "RETURN_VALUE", "?", "SETUP_ANNOTATIONS", "?", 
    "LOAD_LOCALS", "?", "POP_EXCEPT", "STORE_NAME", "DELETE_NAME", "UNPACK_SEQUENCE", 
    "FOR_ITER", "UNPACK_EX", "STORE_ATTR", "DELETE_ATTR", "STORE_GLOBAL", 
    "DELETE_GLOBAL", "SWAP", "LOAD_CONST", "LOAD_NAME", "BUILD_TUPLE", "BUILD_LIST", 
    "BUILD_SET", "BUILD_MAP", "LOAD_ATTR", "COMPARE_OP", "IMPORT_NAME", "IMPORT_FROM", 
    "JUMP_FORWARD", "?", "?", "?", "POP_JUMP_IF_FALSE", 
    "POP_JUMP_IF_TRUE", "LOAD_GLOBAL", "IS_OP", "CONTAINS_OP", "RERAISE", "COPY", 
    "RETURN_CONST", "BINARY_OP", "SEND", "LOAD_FAST", "STORE_FAST", "DELETE_FAST", 
    "LOAD_FAST_CHECK", "POP_JUMP_IF_NOT_NONE", "POP_JUMP_IF_NONE", "RAISE_VARARGS", 
    "GET_AWAITABLE", "MAKE_FUNCTION", "BUILD_SLICE", "JUMP_BACKWARD_NO_INTERRUPT", 
    "MAKE_CELL", "LOAD_CLOSURE", "LOAD_DEREF", "STORE_DEREF", "DELETE_DEREF", 
    "JUMP_BACKWARD", "LOAD_SUPER_ATTR", "CALL_FUNCTION_EX", "LOAD_FAST_AND_CLEAR", 
    "EXTENDED_ARG", "LIST_APPEND", "SET_ADD", "MAP_ADD", "?", "COPY_FREE_VARS", 
    "YIELD_VALUE", "RESUME", "MATCH_CLASS", "?", "?", "FORMAT_VALUE", 
    "BUILD_CONST_KEY_MAP", "BUILD_STRING", "?", "?", "?", 
    "?", "LIST_EXTEND", "SET_UPDATE", "DICT_MERGE", "DICT_UPDATE", 
    "?", "?", "?", "?", "?", "CALL", "KW_NAMES", 
    "CALL_INTRINSIC_1", "CALL_INTRINSIC_2", "LOAD_FROM_DICT_OR_GLOBALS", 
    "LOAD_FROM_DICT_OR_DEREF", "?", "?", "?", "?", 
    "?", "?", "?", "?", "?", "?", 
    "?", "?", "?", "?", "?", "?", 
    "?", "?", "?", "?", "?", "?", 
    "?", "?", "?", "?", "?", "?", 
    "?", "?", "?", "?", "?", "?", 
    "?", "?", "?", "?", "?", "?", 
    "?", "?", "?", "?", "?", "?", 
    "?", "?", "INSTRUMENTED_LOAD_SUPER_ATTR", "INSTRUMENTED_POP_JUMP_IF_NONE", 
    "INSTRUMENTED_POP_JUMP_IF_NOT_NONE", "INSTRUMENTED_RESUME", "INSTRUMENTED_CALL", 
    "INSTRUMENTED_RETURN_VALUE", "INSTRUMENTED_YIELD_VALUE", "INSTRUMENTED_CALL_FUNCTION_EX", 
    "INSTRUMENTED_JUMP_FORWARD", "INSTRUMENTED_JUMP_BACKWARD", "INSTRUMENTED_RETURN_CONST", 
    "INSTRUMENTED_FOR_ITER", "INSTRUMENTED_POP_JUMP_IF_FALSE", "INSTRUMENTED_POP_JUMP_IF_TRUE", 
    "INSTRUMENTED_END_FOR", "INSTRUMENTED_END_SEND", "INSTRUMENTED_INSTRUCTION", 
    "INSTRUMENTED_LINE", "?", "SETUP_FINALLY", "SETUP_CLEANUP", "SETUP_WITH", 
    "POP_BLOCK", "JUMP", "JUMP_NO_INTERRUPT", "LOAD_METHOD", "LOAD_SUPER_METHOD", 
    "LOAD_ZERO_SUPER_METHOD", "LOAD_ZERO_SUPER_ATTR", "STORE_FAST_MAYBE_NULL"
};

// 反組譯字節碼
void disassemble(uint8_t *bytecode, size_t size) {
    for (size_t i = 0; i < size; ) {
        uint8_t opcode = bytecode[i];
        printf("%3zu: %s\n", i, opcodes[opcode]);
        i++;
    }
}

// 讀取 .pyc 檔案
void read_pyc(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("無法打開檔案");
        return;
    }

    // 讀取表頭
    uint8_t header[HEADER_SIZE];
    fread(header, 1, HEADER_SIZE, file);

    // 顯示表頭資訊
    printf("Magic Number: %x\n", *(uint32_t*)header);
    printf("Python Version: %x\n", *(uint16_t*)(header + 4));

    // 跳過檔案大小與其他表頭資料
    fseek(file, HEADER_SIZE, SEEK_SET);

    // 讀取字節碼
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, HEADER_SIZE, SEEK_SET); // 返回字節碼開始處

    size_t bytecode_size = file_size - HEADER_SIZE;
    uint8_t *bytecode = malloc(bytecode_size);
    fread(bytecode, 1, bytecode_size, file);

    // 反組譯字節碼
    disassemble(bytecode, bytecode_size);

    // 釋放資源
    free(bytecode);
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("用法: %s <pyc檔案>\n", argv[0]);
        return 1;
    }

    read_pyc(argv[1]);

    return 0;
}
