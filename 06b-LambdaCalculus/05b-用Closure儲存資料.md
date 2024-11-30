### 第 5 章：用 Closure (閉包) 儲存資料

本章將探討如何使用 **閉包 (Closure)** 在函數內部儲存資料，並解析閉包的本質、特性以及其在函數式程式設計中的應用。閉包是一種強大的工具，能讓我們在無需使用類別的情況下創建狀態保持的函數。

---

#### **5.1 什麼是閉包？**

1. **定義**  
   - 閉包是一個函數，該函數「記住」了它所在的作用域中的變數，即使該作用域已經結束。
   - 簡單來說，閉包讓函數具備「攜帶環境」的能力。

2. **範例：普通函數 vs 閉包**
   ```python
   # 普通函數
   def add(x, y):
       return x + y

   print(add(2, 3))  # 輸出 5

   # 閉包範例
   def make_adder(x):
       def adder(y):
           return x + y
       return adder

   add_5 = make_adder(5)
   print(add_5(10))  # 輸出 15
   ```

3. **閉包的條件**
   - 必須有內嵌函數。
   - 內嵌函數必須使用外部作用域中的變數。
   - 外部函數的執行結束後，內嵌函數仍然可存取外部變數。

---

#### **5.2 為什麼需要閉包？**

1. **模擬物件的行為**
   - 閉包提供了類似物件的狀態儲存方式，但更加輕量化。
     ```python
     def counter():
         count = 0
         def increment():
             nonlocal count
             count += 1
             return count
         return increment

     c = counter()
     print(c())  # 輸出 1
     print(c())  # 輸出 2
     ```

2. **避免全域變數污染**
   - 閉包允許我們在函數中封裝狀態，避免使用全域變數。
     ```python
     def multiplier(factor):
         def multiply(number):
             return number * factor
         return multiply

     double = multiplier(2)
     triple = multiplier(3)
     print(double(5))  # 輸出 10
     print(triple(5))  # 輸出 15
     ```

3. **延遲求值 (Lazy Evaluation)**
   - 使用閉包儲存運算的中間狀態，推遲計算以提升效能。

---

#### **5.3 閉包的運作原理**

1. **作用域鏈與自由變數**
   - 自由變數 (Free Variable)：不是在本地作用域中定義的變數，但可被函數存取。
   - Python 中閉包會將自由變數儲存在 `__closure__` 屬性中。

   範例：
   ```python
   def outer_function(x):
       def inner_function(y):
           return x + y
       return inner_function

   closure_fn = outer_function(10)
   print(closure_fn(5))  # 輸出 15
   print(closure_fn.__closure__[0].cell_contents)  # 輸出 10
   ```

2. **`nonlocal` 關鍵字**
   - 當內部函數需要修改外部作用域的變數時，需使用 `nonlocal` 聲明。
     ```python
     def counter():
         count = 0
         def increment():
             nonlocal count
             count += 1
             return count
         return increment

     c = counter()
     print(c())  # 輸出 1
     print(c())  # 輸出 2
     ```

3. **閉包的效能注意事項**
   - 過多的閉包可能導致記憶體使用效率低下，因為閉包會保留其作用域的變數。

---

#### **5.4 閉包的實務應用**

1. **記錄器 (Logger)**
   - 使用閉包封裝日誌設定。
     ```python
     def logger(level):
         def log(message):
             print(f"[{level}] {message}")
         return log

     info_logger = logger("INFO")
     error_logger = logger("ERROR")

     info_logger("This is an info message.")  # 輸出 [INFO] This is an info message.
     error_logger("This is an error message.")  # 輸出 [ERROR] This is an error message.
     ```

2. **函數裝飾器**
   - 閉包是實現裝飾器的核心技術。
     ```python
     def decorator(func):
         def wrapper(*args, **kwargs):
             print("Before calling the function")
             result = func(*args, **kwargs)
             print("After calling the function")
             return result
         return wrapper

     @decorator
     def say_hello(name):
         print(f"Hello, {name}!")

     say_hello("Alice")
     ```

3. **快取機制 (Memoization)**
   - 閉包可用於儲存運算結果，避免重複計算。
     ```python
     def memoize(func):
         cache = {}
         def wrapper(x):
             if x not in cache:
                 cache[x] = func(x)
             return cache[x]
         return wrapper

     @memoize
     def fibonacci(n):
         if n <= 1:
             return n
         return fibonacci(n - 1) + fibonacci(n - 2)

     print(fibonacci(10))  # 輸出 55
     ```

---

#### **5.5 閉包的限制**

1. **複雜性**
   - 過度使用閉包可能導致程式碼可讀性降低。

2. **記憶體開銷**
   - 閉包會持續引用其外部作用域變數，可能導致不必要的記憶體佔用。

3. **調試困難**
   - 閉包的作用域鏈可能難以追蹤，特別是在多層嵌套的情況下。

---

#### **5.6 小結**

閉包是 Python 中非常靈活且實用的工具，能夠封裝狀態並提供延遲計算、模組化與輕量級物件的替代方案。透過閉包，我們可以避免全域變數污染、簡化高階函數應用，並實現如裝飾器、快取等實用模式。然而，在實際應用中需注意閉包的效能問題與過度複雜性，確保程式碼保持簡潔與高效。