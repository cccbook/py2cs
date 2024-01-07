#include <iostream>
#include "pylib.cpp"

int fib(int n) {
    if (n==0 || n==1) return 1;
    return fib(n-1)+fib(n-2);
}

int main() {
    // std::cout << "Hello World!";
    print("fib(5)=", fib(5));
    return 0;
}