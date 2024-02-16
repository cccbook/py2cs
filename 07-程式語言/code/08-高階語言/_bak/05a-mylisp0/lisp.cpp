#include <iostream>
#include <unordered_map>
#include <vector>
#include <functional>

// 定義一些基本的運算符
std::unordered_map<std::string, std::function<double(double, double)>> ENV = {
    {"+", [](double a, double b) { return a + b; }},
    {"-", [](double a, double b) { return a - b; }},
    {"*", [](double a, double b) { return a * b; }},
    {"/", [](double a, double b) { return a / b; }}
};

// Scheme 解釋器
std::string evaluate(const std::vector<std::string> &exp, const std::unordered_map<std::string, double> &env = {}) {
    if (exp.size() > 0) {
        // 函數調用
        auto op = ENV.at(exp[0]);
        std::vector<double> args;
        for (std::size_t i = 1; i < exp.size(); ++i) {
            args.push_back(evaluate(exp[i], env));
        }
        return op(args[0], args[1]);
    } else if (env.find(exp[0]) != env.end()) {
        // 變數引用
        return env.at(exp[0]);
    } else {
        // 常數
        return exp[0];
    }
}

int main() {
    // 定義一個簡單的 Scheme 表達式
    std::vector<std::string> code = {"+", "2", {"*", "3", "4"}};
    // 解釋和計算結果
    double result = evaluate(code, {});
    std::cout << "Result: " << result << std::endl;

    return 0;
}
