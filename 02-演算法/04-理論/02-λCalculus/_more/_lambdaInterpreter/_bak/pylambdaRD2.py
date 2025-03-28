class LambdaParser:
    def __init__(self, code):
        self.tokens = self.tokenize(code)
        self.current = 0
    
    def tokenize(self, code):
        """
        將輸入字串分解為標記
        """
        tokens = []
        i = 0
        while i < len(code):
            char = code[i]
            
            # 跳過空白
            if char.isspace():
                i += 1
                continue
            
            # 識別運算子
            if char in ['+', '-', '*', '/', '(', ')', ':', ',', '=']:
                tokens.append(char)
                i += 1
                continue
            
            # 識別標識符和數字
            if char.isalpha() or char.isdigit():
                word = char
                while i + 1 < len(code) and (code[i+1].isalnum() or code[i+1] == '_'):
                    i += 1
                    word += code[i]
                tokens.append(word)
                i += 1
                continue
            
            i += 1
        
        return tokens
    
    def parse(self):
        """
        主解析函數
        """
        return self.parse_assignment()
    
    def parse_assignment(self):
        """
        解析賦值語句
        """
        func_name = self.match_identifier()
        self.consume('=')
        func_body = self.parse_lambda()
        return (func_name, func_body)
    
    def parse_lambda(self):
        """
        解析 lambda 函數
        """
        self.consume_keyword('lambda')
        params = self.parse_parameters()
        self.consume(':')
        body = self.parse_expression()
        return LambdaFunction(params, body)
    
    def parse_parameters(self):
        """
        解析函數參數
        """
        params = []
        while self.current < len(self.tokens) and self.tokens[self.current] != ':':
            param = self.match_identifier()
            params.append(param)
            
            # 處理多個參數的情況
            if self.current < len(self.tokens) and self.tokens[self.current] == ',':
                self.consume(',')
        
        return params
    
    def parse_expression(self):
        """
        解析表達式
        """
        return self.parse_additive_expression()
    
    def parse_additive_expression(self):
        """
        解析加法表達式
        """
        left = self.parse_multiplicative_expression()
        
        while self.current < len(self.tokens) and self.tokens[self.current] in ['+', '-']:
            op = self.tokens[self.current]
            self.consume(op)
            right = self.parse_multiplicative_expression()
            left = BinaryOperation(op, left, right)
        
        return left
    
    def parse_multiplicative_expression(self):
        """
        解析乘法表達式
        """
        left = self.parse_primary()
        
        while self.current < len(self.tokens) and self.tokens[self.current] in ['*', '/']:
            op = self.tokens[self.current]
            self.consume(op)
            right = self.parse_primary()
            left = BinaryOperation(op, left, right)
        
        return left
    
    def parse_primary(self):
        """
        解析基本表達式（數字、標識符、括號）
        """
        if self.tokens[self.current].isdigit():
            return self.parse_number()
        elif self.tokens[self.current].isalpha():
            return self.parse_identifier()
        elif self.tokens[self.current] == '(':
            self.consume('(')
            expr = self.parse_expression()
            self.consume(')')
            return expr
        else:
            raise SyntaxError(f"意外的標記: {self.tokens[self.current]}")
    
    def parse_number(self):
        """
        解析數字
        """
        value = int(self.tokens[self.current])
        self.consume(self.tokens[self.current])
        return Literal(value)
    
    def parse_identifier(self):
        """
        解析標識符
        """
        name = self.tokens[self.current]
        self.consume(name)
        return Variable(name)
    
    def match_identifier(self):
        """
        匹配標識符
        """
        if not self.tokens[self.current].isalpha():
            raise SyntaxError(f"預期是標識符，得到: {self.tokens[self.current]}")
        identifier = self.tokens[self.current]
        self.current += 1
        return identifier
    
    def consume_keyword(self, keyword):
        """
        消耗特定關鍵字
        """
        if self.tokens[self.current] != keyword:
            raise SyntaxError(f"預期是 {keyword}，得到: {self.tokens[self.current]}")
        self.current += 1
    
    def consume(self, expected_token):
        """
        消耗匹配的標記
        """
        if self.current >= len(self.tokens) or self.tokens[self.current] != expected_token:
            raise SyntaxError(f"預期是 {expected_token}，得到: {self.tokens[self.current] if self.current < len(self.tokens) else '結束'}")
        self.current += 1
        
    # 縮排輔助函數
    def print_indent(level):
        return "│   " * level
    
    # 遞迴輸出 AST 節點
    def dump_node(node, level):
        if isinstance(node, Literal):
            print(f"{print_indent(level)}├── Literal: {node.value}")
        
        elif isinstance(node, Variable):
            print(f"{print_indent(level)}├── Variable: {node.name}")
        
        elif isinstance(node, BinaryOperation):
            print(f"{print_indent(level)}├── BinaryOperation: '{node.op}'")
            print(f"{print_indent(level)}│   ├── Left:")
            dump_node(node.left, level + 1)
            print(f"{print_indent(level)}│   └── Right:")
            dump_node(node.right, level + 1)

    def dump_exp(self, ast_node=None, indent=0):
        """
        將 AST 節點以樹狀結構輸出
        
        Args:
            ast_node: 要輸出的 AST 節點，預設為解析的完整表達式
            indent: 縮排層級
        """
        # 如果沒有提供節點，則從頭開始解析
        if ast_node is None:
            try:
                # 重置解析器
                self.current = 0
                ast_node = self.parse_expression()
            except Exception as e:
                print(f"解析錯誤: {e}")
                return

        print("AST 表達式樹:")
        dump_node(ast_node, indent)

# AST 節點類別
class Literal:
    def __init__(self, value):
        self.value = value
    
    def evaluate(self, context=None):
        return self.value

class Variable:
    def __init__(self, name):
        self.name = name
    
    def evaluate(self, context=None):
        if context and self.name in context:
            return context[self.name]
        raise NameError(f"未定義的變量: {self.name}")

class BinaryOperation:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
    
    def evaluate(self, context=None):
        left_val = self.left.evaluate(context)
        right_val = self.right.evaluate(context)
        
        if self.op == '+':
            return left_val + right_val
        elif self.op == '-':
            return left_val - right_val
        elif self.op == '*':
            return left_val * right_val
        elif self.op == '/':
            return left_val / right_val

class LambdaFunction:
    def __init__(self, params, body):
        self.params = params
        self.body = body
    
    def __call__(self, *args, context=None):
        if len(args) != len(self.params):
            raise TypeError(f"期望 {len(self.params)} 個參數，得到 {len(args)} 個")
        
        # 創建函數調用的新上下文
        new_context = context.copy() if context else {}
        for param, value in zip(self.params, args):
            new_context[param] = value
        
        return self.body.evaluate(new_context)

class LambdaInterpreter:
    def __init__(self):
        # self.functions = {}
        self.environment = {}
    
    def parse_lambda(self, code):
        """
        解析 lambda 函數定義
        """
        try:
            parser = LambdaParser(code)
            func_name, lambda_func = parser.parse()
            # self.functions[func_name] = lambda_func
            self.environment[func_name] = lambda_func
            return lambda_func
        except Exception as e:
            print(f"解析錯誤: {e}")
            return None
    
    def execute(self, code):
        """
        執行 lambda 表達式
        """
        code = code.strip()
        
        if '=' in code and 'lambda' in code:
            return self.parse_lambda(code)
        
        try:
            print(f'code={code}')
            parser = LambdaParser(code)
            print(f'parser.tokens={parser.tokens}')
            exp = parser.parse_expression()
            print(f'exp={exp}')
            print(f'exp.dump_exp()={exp.dump_exp()}')
            result = exp.evaluate(self.environment)
            print(f'result={result}')
            # print(f'result()={result()}')
            return result
        except Exception as e:
            print(f"env={self.environment}")
            print(f"執行錯誤: {e}")
            return None
    
    def call_function(self, func_name, *args):
        """
        調用已定義的函數
        """
        # if func_name in self.functions:
        #    return self.functions[func_name](*args, context=self.environment)
        if func_name in self.environment:
            return self.environment[func_name](*args, context=self.environment)
        else:
            raise ValueError(f"未定義的函數: {func_name}")
    
    def list_env(self):
        """
        列出所有已定義的函數
        """
        return list(self.environment.keys())
    
    def set_variable(self, name, value):
        """
        設置環境變數
        """
        self.environment[name] = value

# 使用範例和測試
def test_lambda_interpreter():
    # 創建解釋器實例
    interpreter = LambdaInterpreter()
    
    # 定義簡單函數
    interpreter.execute("f = lambda x, y: x + y")
    interpreter.execute("g = lambda x: x * 2")
    
    # 設置環境變數
    interpreter.set_variable('z', 10)
    
    # 調用函數並打印結果
    print("f(3, 4) =", interpreter.call_function("f", 3, 4))  # 應輸出 7
    print("g(5) =", interpreter.call_function("g", 5))        # 應輸出 10
    
    # 嘗試使用環境變數的複雜表達式
    result = interpreter.execute("f(g(3), z)")
    print("f(g(3), z) =", result)
    
    # 列出已定義函數
    print("已定義函數:", interpreter.list_env())

# 執行測試
test_lambda_interpreter()
