#include <stdio.h>

double power3(double x) {
    return x*x*x;
}

#define h 0.0001

typedef double (*F1)(double);

double df(F1 f, double x) {
    double dy = f(x+h)-f(x);
    return dy/h;
}

int main() {
    printf("df(power3, 2.0)=%f\n", df(power3, 2.0));
}
