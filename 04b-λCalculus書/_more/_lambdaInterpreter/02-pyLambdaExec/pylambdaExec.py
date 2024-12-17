import traceback

class LambdaInterpreter:
    def __init__(self):
        # 存儲函數的字典
        self.functions = {}
        # 存儲環境變數
        self.environment = {}
    
    def parse_lambda(self, code):
        """
        解析 lambda 函數定義
        """
        
        try:
            # 分割賦值語句
            parts = code.split('=')
            func_name = parts[0].strip()
            lambda_def = parts[1].strip()
            
            # 構建完整的可執行函數定義
            full_def = f"{func_name}={lambda_def}"
            # 使用 exec 來執行函數定義
            local_env = {}
            exec(full_def, self.environment, local_env)
            # 將函數存入 functions 字典
            self.functions[func_name] = local_env[func_name]
            
            return self.functions[func_name]
        
        except Exception as e:
            print(f"解析錯誤: {e}")
            print(traceback.format_exc())
            return None
    
    def execute(self, code):
        """
        執行 lambda 表達式
        """
        # 去除前後空白
        code = code.strip()
        
        # 處理賦值語句
        if '=' in code and ('lambda' in code):
            return self.parse_lambda(code)
        
        # 處理直接的函數調用或表達式
        try:
            return eval(code, self.environment, self.functions)
        except Exception as e:
            print(f"執行錯誤: {e}")
            return None
    
    def call_function(self, func_name, *args):
        """
        調用已定義的函數
        """
        if func_name in self.functions:
            return self.functions[func_name](*args)
        elif func_name in self.environment:
            return self.environment[func_name](*args)
        else:
            raise ValueError(f"未定義的函數: {func_name}")
    
    def list_functions(self):
        """
        列出所有已定義的函數
        """
        return list(self.functions.keys())
    
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
    print("已定義函數:", interpreter.list_functions())

# 執行測試
test_lambda_interpreter()
