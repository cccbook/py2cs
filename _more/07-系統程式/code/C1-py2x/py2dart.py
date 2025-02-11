import ast
from typing import Any, Optional, Dict

class PythonToDartTransformer(ast.NodeVisitor):
    def __init__(self):
        self.indent_level = 0
        self.output = []
        # 追蹤變數型別
        self.var_types: Dict[str, str] = {}
        
    def visit(self, node: ast.AST) -> Any:
        """訪問 AST 節點並轉換為 Dart 代碼"""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)
    
    def indent(self) -> str:
        """產生縮排"""
        return "  " * self.indent_level
    
    def get_dart_type(self, python_type: str) -> str:
        """將 Python 型別轉換為 Dart 型別"""
        type_mapping = {
            'int': 'int',
            'float': 'double',
            'str': 'String',
            'bool': 'bool',
            'list': 'List',
            'dict': 'Map',
            'None': 'void'
        }
        return type_mapping.get(python_type, 'dynamic')
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """轉換函數定義"""
        # 處理返回型別
        return_type = 'dynamic'
        if node.returns:
            return_type = self.get_dart_type(self.visit(node.returns))
        
        # 處理函數參數
        args = []
        for arg in node.args.args:
            if arg.annotation:
                arg_type = self.get_dart_type(self.visit(arg.annotation))
                args.append(f"{arg_type} {arg.arg}")
            else:
                args.append(f"dynamic {arg.arg}")
        
        # 生成函數定義
        self.output.append(f"{self.indent()}{return_type} {node.name}({', '.join(args)}) {{")
        
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
        
        # 嘗試從值推斷型別
        var_type = 'var'
        if isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, int):
                var_type = 'int'
            elif isinstance(node.value.value, float):
                var_type = 'double'
            elif isinstance(node.value.value, str):
                var_type = 'String'
            elif isinstance(node.value.value, bool):
                var_type = 'bool'
        
        # 記錄變數型別
        if isinstance(node.targets[0], ast.Name):
            self.var_types[node.targets[0].id] = var_type
        
        self.output.append(f"{self.indent()}{var_type} {target} = {value};")
    
    def visit_Return(self, node: ast.Return) -> None:
        """轉換return語句"""
        if node.value:
            value = self.visit(node.value)
            self.output.append(f"{self.indent()}return {value};")
        else:
            self.output.append(f"{self.indent()}return;")
    
    def visit_Name(self, node: ast.Name) -> str:
        """轉換變數名稱"""
        py_to_dart = {
            'True': 'true',
            'False': 'false',
            'None': 'null'
        }
        return py_to_dart.get(node.id, node.id)
    
    def visit_Constant(self, node: ast.Constant) -> str:
        """轉換常量"""
        if node.value is None:
            return 'null'
        elif isinstance(node.value, bool):
            return str(node.value).lower()
        elif isinstance(node.value, str):
            return f"'{node.value}'"
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
            self.output.append(f"{self.indent()}}}\nelse {{")
            self.indent_level += 1
            for stmt in node.orelse:
                self.visit(stmt)
            self.indent_level -= 1
        self.output.append(f"{self.indent()}}}")
    
    def visit_Compare(self, node: ast.Compare) -> str:
        """轉換比較運算"""
        ops = {
            ast.Eq: "==",
            ast.NotEq: "!=",
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
            ast.FloorDiv: "~/",  # Dart 的整數除法
            ast.Mod: "%",
            ast.Pow: "pow"  # 需要 import 'dart:math'
        }
        left = self.visit(node.left)
        op = ops[type(node.op)]
        right = self.visit(node.right)
        
        if isinstance(node.op, ast.Pow):
            return f"pow({left}, {right})"
        return f"{left} {op} {right}"

    def visit_Call(self, node: ast.Call) -> str:
        """轉換函數呼叫"""
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        
        # 處理常見的內建函數
        if func == 'print':
            return f"print({', '.join(args)})"
        elif func == 'len':
            if args:
                return f"{args[0]}.length"
        elif func == 'str':
            if args:
                return f"{args[0]}.toString()"
        elif func == 'int':
            if args:
                return f"int.parse({args[0]})"
        
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

def py2dart(python_code: str) -> str:
    """將Python代碼轉換為Dart代碼"""
    tree = ast.parse(python_code)
    transformer = PythonToDartTransformer()
    
    # 添加必要的 import
    output = ["import 'dart:math';", ""]
    
    # 添加 main 函數包裝
    output.append("void main() {")
    
    # 轉換代碼
    transformer.indent_level = 1
    transformer.visit(tree)
    output.extend(transformer.output)
    
    output.append("}")
    
    return "\n".join(output)



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
    mojo_code = py2dart(python_code)
    print(mojo_code)
