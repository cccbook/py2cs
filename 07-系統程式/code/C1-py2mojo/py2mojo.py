import ast
from typing import Any, Optional

class PythonToMojoTransformer(ast.NodeVisitor):
    def __init__(self):
        self.indent_level = 0
        self.output = []
        
    def visit(self, node: ast.AST) -> Any:
        """訪問 AST 節點並轉換為 Mojo 代碼"""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)
    
    def indent(self) -> str:
        """產生縮排"""
        return "    " * self.indent_level
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """轉換函數定義"""
        returns = ""
        if node.returns:
            returns = f" -> {self.visit(node.returns)}"
        
        args = []
        for arg in node.args.args:
            if arg.annotation:
                args.append(f"{arg.arg}: {self.visit(arg.annotation)}")
            else:
                args.append(f"{arg.arg}: Any")
        
        self.output.append(f"{self.indent()}fn {node.name}({', '.join(args)}){returns}:")
        
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """轉換賦值語句"""
        target = self.visit(node.targets[0])
        value = self.visit(node.value)
        self.output.append(f"{self.indent()}let {target} = {value}")
    
    def visit_Return(self, node: ast.Return) -> None:
        """轉換return語句"""
        if node.value:
            value = self.visit(node.value)
            self.output.append(f"{self.indent()}return {value}")
        else:
            self.output.append(f"{self.indent()}return")
    
    def visit_Name(self, node: ast.Name) -> str:
        """轉換變數名稱"""
        type_mapping = {
            'int': 'Int',
            'float': 'Float64',
            'str': 'String',
            'bool': 'Bool'
        }
        return type_mapping.get(node.id, node.id)
    
    def visit_Constant(self, node: ast.Constant) -> str:
        """轉換常量"""
        if isinstance(node.value, str):
            return f'"{node.value}"'
        return str(node.value)
    
    def visit_If(self, node: ast.If) -> None:
        """轉換if語句"""
        test = self.visit(node.test)
        self.output.append(f"{self.indent()}if {test}:")
        
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        
        if node.orelse:
            self.output.append(f"{self.indent()}else:")
            self.indent_level += 1
            for stmt in node.orelse:
                self.visit(stmt)
            self.indent_level -= 1
    
    def visit_Compare(self, node: ast.Compare) -> str:
        """轉換比較運算"""
        ops = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
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
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.Pow: "**"
        }
        left = self.visit(node.left)
        op = ops[type(node.op)]
        right = self.visit(node.right)
        return f"{left} {op} {right}"

    def visit_Call(self, node: ast.Call) -> str:
        """轉換函數呼叫"""
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        return f"{func}({', '.join(args)})"

    def visit_Expr(self, node: ast.Expr) -> None:
        """轉換表達式語句（如 print 呼叫）"""
        expr = self.visit(node.value)
        self.output.append(f"{self.indent()}{expr}")

    def visit_Lambda(self, node: ast.Lambda) -> str:
        """轉換 lambda 表達式為 Mojo 閉包函數"""
        # 處理參數
        args = []
        for arg in node.args.args:
            if arg.annotation:
                args.append(f"{arg.arg}: {self.visit(arg.annotation)}")
            else:
                # 在 Mojo 中，我們需要指定類型，這裡預設使用 Int
                args.append(f"{arg.arg}: Int")
        
        # 處理函數體
        body = self.visit(node.body)
        
        # 返回 Mojo 格式的閉包函數
        return f"|{', '.join(args)}| -> Int: {body}"

def py2mojo(python_code: str) -> str:
    """將Python代碼轉換為Mojo代碼"""
    tree = ast.parse(python_code)
    transformer = PythonToMojoTransformer()
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
    mojo_code = py2mojo(python_code)
    print(mojo_code)
