import ast
from typing import Any, Optional

class PythonToJavaScriptTransformer(ast.NodeVisitor):
    def __init__(self):
        self.indent_level = 0
        self.output = []
        
    def visit(self, node: ast.AST) -> Any:
        """訪問 AST 節點並轉換為 JavaScript 代碼"""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)
    
    def indent(self) -> str:
        """產生縮排"""
        return "  " * self.indent_level
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """轉換函數定義"""
        # 忽略型別註解，JavaScript 不需要
        args = [arg.arg for arg in node.args.args]
        
        # 生成函數定義
        self.output.append(f"{self.indent()}function {node.name}({', '.join(args)}) {{")
        
        # 訪問函數體
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.output.append(f"{self.indent()}}}")
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """轉換賦值語句"""
        target = self.visit(node.targets[0])
        value = self.visit(node.value)
        # JavaScript 使用 let 作為變數宣告
        self.output.append(f"{self.indent()}let {target} = {value};")
    
    def visit_Return(self, node: ast.Return) -> None:
        """轉換return語句"""
        if node.value:
            value = self.visit(node.value)
            self.output.append(f"{self.indent()}return {value};")
        else:
            self.output.append(f"{self.indent()}return;")
    
    def visit_Name(self, node: ast.Name) -> str:
        """轉換變數名稱"""
        # Python 的 True/False/None 轉換為 JavaScript 對應值
        py_to_js = {
            'True': 'true',
            'False': 'false',
            'None': 'null'
        }
        return py_to_js.get(node.id, node.id)
    
    def visit_Constant(self, node: ast.Constant) -> str:
        """轉換常量"""
        if node.value is None:
            return 'null'
        elif isinstance(node.value, bool):
            return str(node.value).lower()
        elif isinstance(node.value, str):
            return f'"{node.value}"'
        return str(node.value)
    
    def visit_If(self, node: ast.If) -> None:
        """轉換if語句"""
        test = self.visit(node.test)
        self.output.append(f"{self.indent()}if ({test}) {{")
        
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        
        if node.orelse:
            self.output.append(f"{self.indent()}}} else {{")
            self.indent_level += 1
            for stmt in node.orelse:
                self.visit(stmt)
            self.indent_level -= 1
        self.output.append(f"{self.indent()}}}")
    
    def visit_Compare(self, node: ast.Compare) -> str:
        """轉換比較運算"""
        ops = {
            ast.Eq: "===",
            ast.NotEq: "!==",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">="
        }
        left = self.visit(node.left)
        op = ops[type(node.ops[0])]
        right = self.visit(node.comparators[0])
        return f"{left} {op} {right}"

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """轉換二元運算"""
        ops = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "Math.floor(/)",  # JavaScript 需要使用 Math.floor
            ast.Mod: "%",
            ast.Pow: "**"
        }
        left = self.visit(node.left)
        op = ops[type(node.op)]
        right = self.visit(node.right)
        
        if isinstance(node.op, ast.FloorDiv):
            return f"Math.floor({left} / {right})"
        return f"{left} {op} {right}"

    def visit_Call(self, node: ast.Call) -> str:
        """轉換函數呼叫"""
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        
        # 特殊處理某些 Python 內建函數
        if func == 'print':
            func = 'console.log'
        elif func == 'len':
            if args:
                return f"{args[0]}.length"
        
        return f"{func}({', '.join(args)})"

    def visit_Expr(self, node: ast.Expr) -> None:
        """轉換表達式語句"""
        expr = self.visit(node.value)
        self.output.append(f"{self.indent()}{expr};")

    def visit_Lambda(self, node: ast.Lambda) -> str:
        """轉換 lambda 表達式"""
        args = [arg.arg for arg in node.args.args]
        body = self.visit(node.body)
        return f"({', '.join(args)}) => {body}"

def py2js(python_code: str) -> str:
    """將Python代碼轉換為JavaScript代碼"""
    tree = ast.parse(python_code)
    transformer = PythonToJavaScriptTransformer()
    transformer.visit(tree)
    return "\n".join(transformer.output)


python_code = """
def calculate_sum(a: int, b: int) -> int:
    if a > b:
        return a + b
    else:
        return b - a

def main():
    x = 3
    y = 2*x
    print(calculate_sum(x,y))
    power2 = lambda n:2**n
    print(power2(3))

"""

if __name__=="__main__":
    mojo_code = py2js(python_code)
    print(mojo_code)
