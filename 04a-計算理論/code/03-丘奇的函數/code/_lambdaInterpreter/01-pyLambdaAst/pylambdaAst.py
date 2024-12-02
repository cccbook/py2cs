import ast
import inspect

class LambdaCalculusInterpreter:
    def __init__(self):
        # 儲存已定義的函數
        self.functions = {}
    
    def parse_lambda(self, code):
        """
        解析 lambda 表達式
        """
        try:
            # 使用 ast 模組解析代碼
            parsed = ast.parse(code)
            
            # 檢查是否是賦值語句 f = lambda x,y: x+y
            if isinstance(parsed.body[0], ast.Assign): # 如果運算子為 assign(=)
                target = parsed.body[0].targets[0] # 取出 變數名稱 f
                value = parsed.body[0].value # 取出函數內容 lambda x,y: x+y
                
                # 檢查是否是 lambda 函數
                if isinstance(value, ast.Lambda): # value 是 lambda 函數
                    # 提取參數名稱
                    args = [arg.arg for arg in value.args.args] # 取得所有參數
                    
                    # 提取函數體
                    func_body = ast.unparse(value.body) # 取得 body 的語法樹
                    
                    # 創建可執行函數
                    lambda_func = eval(f"lambda {','.join(args)}: {func_body}") # 創建該 lambda 函數
                    
                    # 儲存函數
                    self.functions[target.id] = lambda_func # 儲存到 functions[] 字典裏
                    return lambda_func
        except Exception as e:
            print(f"解析錯誤: {e}")
            return None
    
    def execute(self, code):
        """
        執行 lambda 表達式
        """
        result = self.parse_lambda(code)
        return result
    
    def call_function(self, func_name, *args):
        """
        調用已定義的函數
        """
        if func_name in self.functions:
            return self.functions[func_name](*args)
        else:
            raise ValueError(f"未定義的函數: {func_name}")
    
    def list_functions(self):
        """
        列出所有已定義的函數
        """
        return list(self.functions.keys())

# 使用範例
interpreter = LambdaCalculusInterpreter()

# 測試函數定義和調用
def test_interpreter():
    # 定義簡單的函數
    interpreter.execute("f = lambda x, y: x + y")
    interpreter.execute("g = lambda x: x * 2")
    
    # 調用函數
    print("f(3, 4) =", interpreter.call_function("f", 3, 4))  # 應輸出 7
    print("g(5) =", interpreter.call_function("g", 5))        # 應輸出 10
    
    # 列出已定義函數
    print("已定義函數:", interpreter.list_functions())

# 執行測試
test_interpreter()
