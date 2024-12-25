import ast

# 將 Python 程式碼轉換為 Mojo 程式碼的函數
def python_to_mojo(python_code: str) -> str:
    # 解析 Python 程式碼為 AST
    tree = ast.parse(python_code)
    
    # 用於保存生成的 Mojo 程式碼
    mojo_code = []

    # 定義一個遞迴函數來遍歷 AST 並生成 Mojo 程式碼
    def visit(node):
        # 處理函數定義
        if isinstance(node, ast.FunctionDef):
            mojo_code.append(f"func {node.name}(")
            for arg in node.args.args:
                mojo_code.append(arg.arg + ", ")
            mojo_code.append("):\n")
            for body_node in node.body:
                visit(body_node)

        # 處理函數內部語句
        elif isinstance(node, ast.Assign):
            # 使用 let 來聲明變數
            for target in node.targets:
                mojo_code.append(f"let {target.id} = ")
            value = visit(node.value)
            mojo_code.append(f"{value}\n")

        # 處理表達式
        elif isinstance(node, ast.Expr):
            return visit(node.value)
        
        # 處理函數調用
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            args = ", ".join([visit(arg) for arg in node.args])
            return f"{func_name}({args})"
        
        # 處理基本數據類型
        elif isinstance(node, ast.Constant):
            return str(node.value)

        # 處理運算符
        elif isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            op = type(node.op).__name__.lower()
            if op == 'add':
                return f"{left} + {right}"
            elif op == 'sub':
                return f"{left} - {right}"
            elif op == 'mult':
                return f"{left} * {right}"
            elif op == 'div':
                return f"{left} / {right}"

        # 其他情況
        return ""

    # 遍歷 AST 節點並生成 Mojo 程式碼
    for node in tree.body:
        visit(node)

    return "".join(mojo_code)

# 測試
python_code = """
def add(a, b):
    result = a + b
    return result

x = 10
y = 20
print(add(x, y))
"""

# 將 Python 程式碼轉換為 Mojo 程式碼
mojo_code = python_to_mojo(python_code)
print("轉換後的 Mojo 程式碼:")
print(mojo_code)
