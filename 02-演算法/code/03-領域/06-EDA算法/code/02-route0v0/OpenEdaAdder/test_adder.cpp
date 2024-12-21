// test_adder.cpp
#include <verilated.h>
#include "Vadder.h"

int main() {
    Vadder* adder = new Vadder;

    adder->a = 5;  // 0101
    adder->b = 3;  // 0011
    adder->eval();

    printf("Sum: %d\n", adder->sum);  // 應輸出 8

    delete adder;
    return 0;
}
