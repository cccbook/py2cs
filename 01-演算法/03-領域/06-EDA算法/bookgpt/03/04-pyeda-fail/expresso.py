# https://chatgpt.com/c/674d3e82-0288-8012-b784-9b2c82714db5
# 
from pyeda.inter import espresso_exprs
from pyeda.boolalg.expr import exprvars, Or

def espresso_simplify(minterms, dont_cares, num_vars):
    """
    使用 Espresso 演算法簡化布林代數式

    :param minterms: 最小項列表 (以數字表示)
    :param dont_cares: 不關心條件列表 (以數字表示)
    :param num_vars: 變數的數量
    :return: 簡化後的布林運算式
    """
    # 生成變數
    variables = exprvars('x', num_vars)

    # 將最小項與不關心條件轉為 PyEDA 的表達式格式
    def terms_to_expr(terms):
        return Or(*[tuple((variables[i] if (term >> i) & 1 else ~variables[i]) 
                         for i in range(num_vars)) for term in terms])

    minterm_expr = terms_to_expr(minterms)
    dont_care_expr = terms_to_expr(dont_cares)

    # 使用 Espresso 簡化
    simplified_exprs = espresso_exprs(minterm_expr, dont_care_expr)

    # 簡化結果以列表形式返回
    return simplified_exprs

def main():
    # 定義輸入
    minterms = [1, 2, 5, 6, 7]  # 最小項
    dont_cares = [3]            # 不關心條件
    num_vars = 3                # 變數數量

    # 簡化運算
    simplified = espresso_simplify(minterms, dont_cares, num_vars)

    # 輸出結果
    print("簡化後的布林代數式:")
    for expr in simplified:
        print(expr)

# 執行測試
main()
