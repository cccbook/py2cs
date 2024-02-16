#include <iostream>

using namespace std;

template<class... Args>
void print(Args... args)
{
    (cout << ... << args) << "\n";
}

/*
int main() {
    cout << "hello" << 5 << endl;
    print("Hello", 5);
    return 0;
}
*/