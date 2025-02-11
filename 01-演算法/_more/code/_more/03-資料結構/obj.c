#include <stdio.h>

enum {
    TFLOAT,    // 雙精度浮點數
    TSTRING,   // 字串
    TARRAY,    // 陣列
};

// The object type
struct obj {
    int type; // 物件型態
    int size; // 大小 (a->size or string length ...)
    union {
        double f;    // 浮點數值
        char   *str; // 字串常數
        struct obj **a;  // 陣列
    };
};

void o_print(struct obj *o) {
    if (o == NULL) printf("(obj:null)");
    switch (o->type) {
        case TFLOAT:
            printf("%g", o->f);
            break;
        case TSTRING:
            printf("%s", o->str);
            break;
        case TARRAY:
            printf("[");
            for (int i=0; i<o->size; i++) {
                o_print(o->a[i]);
                if (i != o->size-1) printf(",");
            }
            printf("]");
            break;
        default:
            printf("(unknown:type=%d)", o->type);
    }
}

int main() {
    struct obj o1 = { .type=TFLOAT, .f=3.14159};
    struct obj o2 = { .type=TSTRING, .str="hello"};
    struct obj *array[] = { &o1, &o2 };
    struct obj o3 = { .type=TARRAY, .a=array, .size=2 };

    o_print(&o1); printf("\n");
    o_print(&o2); printf("\n");
    o_print(&o3); printf("\n");
}