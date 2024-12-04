#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

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


// 讀取 16 位元標頭並返回 bytecode 開始的指標
uint8_t* read_header(FILE* file) {
    uint8_t* buffer = malloc(16);
    if (fread(buffer, 1, 16, file) != 16) {
        fprintf(stderr, "無法讀取文件標頭\n");
        free(buffer);
        return NULL;
    }
    return buffer;
}

// 顯示 opcode 指令
void disassemble(uint8_t* bytecode, size_t length) {
    size_t i = 0;
    while (i < length) {
        uint8_t opcode = bytecode[i++];
        
        if (opcode < sizeof(opcodes) / sizeof(opcodes[0])) {
            printf("%s\n", opcodes[opcode]);
        } else {
            printf("未知指令: %d\n", opcode);
        }

        // 檢查有無參數 (指令 90 以上通常有參數)
        if (opcode >= 82) {
            if (i >= length) break;  // 防止越界
            uint8_t arg = bytecode[i++];
            printf("參數: %d\n", arg);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "用法: %s <pyc_file>\n", argv[0]);
        return 1;
    }

    FILE* file = fopen(argv[1], "rb");
    if (!file) {
        perror("無法開啟檔案");
        return 1;
    }

    // 讀取標頭
    uint8_t* header = read_header(file);
    if (!header) {
        fclose(file);
        return 1;
    }
    free(header);

    // 讀取剩餘的 bytecode
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 16, SEEK_SET);  // 略過標頭

    size_t bytecode_size = file_size - 16;
    uint8_t* bytecode = malloc(bytecode_size);
    if (fread(bytecode, 1, bytecode_size, file) != bytecode_size) {
        fprintf(stderr, "無法讀取 bytecode\n");
        free(bytecode);
        fclose(file);
        return 1;
    }

    // 反組譯 bytecode
    disassemble(bytecode, bytecode_size);

    free(bytecode);
    fclose(file);
    return 0;
}
