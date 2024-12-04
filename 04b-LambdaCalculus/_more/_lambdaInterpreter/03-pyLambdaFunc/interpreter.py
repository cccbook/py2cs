import re
from parser import tokenize, Parser
# Tokenizer 和 Parser 保持不變

# 解譯器，支持運算符定義和執行
class Interpreter:
    def __init__(self, builtins=None):
        self.env = builtins or {}

    def evaluate(self, node):
        """
        遞迴地解釋 AST 節點
        """
        if node["type"] == "CALL":
            name = node["name"]
            if name not in self.env:
                raise ValueError(f"未知的運算符: {name}")
            func = self.env[name]
            args = [self.evaluate(arg) for arg in node["args"]]
            result = func
            for arg in args:
                result = result(arg)
            return result
        else:
            raise ValueError(f"無效的 AST 節點: {node}")

    def execute(self, code):
        """
        支持運算符定義和執行混合的多行程式
        """
        lines = code.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "=" in line:  # 定義運算符
                self.define_operator(line)
            else:  # 執行表達式
                tokens = list(tokenize(line))
                parser = Parser(tokens)
                ast = parser.parse()
                result = self.evaluate(ast)
                print(f"結果: {result}")  # 輸出執行結果

    def define_operator(self, line):
        """
        動態解析運算符定義
        """
        match = re.match(r"(\w+)\s*=\s*(.+)", line)
        if not match:
            raise SyntaxError(f"無效的運算符定義: {line}")
        name, expr = match.groups()
        try:
            # 將字串形式的 lambda 表達式轉為 Python 函數
            func = eval(expr)
            if not callable(func):
                raise ValueError(f"定義的運算符必須是可呼叫的: {expr}")
            self.env[name] = func
            print(f"已定義運算符: {name}")
        except Exception as e:
            raise ValueError(f"解析運算符時出錯: {e}")

"""
ASSERT = lambda truth: (IF(truth)
    (lambda description:f'[✓] ${description}')
    (lambda description:f'[✗] ${description}')
)

REFUTE = lambda truth:ASSERT(NOT(truth))

TEST   = lambda description:lambda assertion:\
    print(assertion(description))

ENV = {
    'ASSERT': ASSERT,
    'REFUTE': REFUTE,
    'TEST': TEST,
}
"""

# ENV = {'print':print}
ENV = {}

# 測試程式
if __name__ == "__main__":
    code = """
    IF = lambda c: lambda x: lambda y: c(x)(y)
    TRUE = lambda x: lambda y: x
    FALSE = lambda x: lambda y: y
    OR = lambda p: lambda q: p(p)(q)
    NOT = lambda c: c(FALSE)(TRUE)

    TRUE
    """
    interpreter = Interpreter(ENV)
    r = interpreter.execute(code)
    import inspect
    print(inspect.getsource(r))
