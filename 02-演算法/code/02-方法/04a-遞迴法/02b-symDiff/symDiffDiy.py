# 本程式用 chatGPT 撰寫 -- https://chatgpt.com/share/66f20d18-d02c-8012-b862-ee07ca35cdae
import re

# 定義常見函數的導數規則
derivatives = {
    'x': '1',      # dx/dx = 1
    'sin': 'cos',  # d(sin(x))/dx = cos(x)
    'cos': '-sin', # d(cos(x))/dx = -sin(x)
    'tan': 'sec^2',# d(tan(x))/dx = sec^2(x)
    'exp': 'exp',  # d(exp(x))/dx = exp(x)
    'ln': '1/x',   # d(ln(x))/dx = 1/x
    'x^n': lambda n: f'{n}*x^{n-1}'  # d(x^n)/dx = n*x^(n-1)
}

# 主函數：計算 f(x) 的導數
def differentiate(expr):
    # 處理多項式，如 x^n
    if re.match(r"x\^\d+", expr):
        n = int(expr.split('^')[1])
        return derivatives['x^n'](n)

    # 處理簡單變量 x
    if expr == 'x':
        return derivatives['x']

    # 處理函數，如 sin(x), cos(x), exp(x)
    for func in derivatives:
        if expr.startswith(f"{func}(") and expr.endswith(")"):
            inner_expr = expr[len(func)+1:-1]
            inner_deriv = differentiate(inner_expr)
            return f"{derivatives[func]}({inner_expr}) * {inner_deriv}"

    # 若無法識別則回傳未知
    return f"未知的函數: {expr}"

# 測試例子
print('diff(x)=', differentiate('x'))         # 1
print('diff(x^3)=', differentiate('x^3'))       # 3*x^2
print('diff(sin(x))=', differentiate('sin(x)'))    # cos(x) * 1
print('diff(sin(x^2))=', differentiate('sin(x^2)'))  # cos(x^2) * 2x
print('diff(sin(cos(x^2)))=', differentiate('sin(cos(x^2))'))  # cos(cos(x^2)) * -sin(x^2) * 2*x^1
# print(differentiate('sin(cos(x^2)^2)'))  # cos(cos(x^2)) * -sin(x^2) * 2*x^1
print('diff(cos(x))=', differentiate('cos(x)'))    # -sin(x) * 1
print('diff(exp(x))=', differentiate('exp(x)'))    # exp(x) * 1
print('diff(exp(x^3))=', differentiate('exp(x^3)'))  # exp(x^3) * 3*x^2