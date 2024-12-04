### 第 4 章：Curry (柯里化)

本章將介紹 **柯里化 (Currying)**，這是一種將多參數函數分解為一系列單參數函數的技術。此概念得名於數學家 **Haskell Curry**，是函數式程式設計中的核心技巧之一，能夠提升程式的靈活性與重用性，特別是在高階函數與延遲計算中。

---

#### **4.1 柯里化的基本概念**

1. **什麼是柯里化？**
   - **未柯里化函數**：接收多個參數並同時處理它們。
     ```python
     def add(x, y):
         return x + y
     print(add(3, 5))  # 輸出 8
     ```
   - **柯里化函數**：將上述函數轉換為嵌套的單參數函數。
     ```python
     def curried_add(x):
         return lambda y: x + y
     add_3 = curried_add(3)
     print(add_3(5))  # 輸出 8
     ```

2. **數學背景**
   - 在數學中，柯里化將多元函數 \( f(x, y, z) \) 拆解為 \( f(x)(y)(z) \)。
   - 這一技術允許函數以部分參數進行應用，並逐步構建完整運算。

3. **特性**
   - 每次函數呼叫僅處理單一參數。
   - 柯里化的核心是**將複雜邏輯分解為簡單的函數鏈條**。

---

#### **4.2 Python 中的柯里化**

1. **手動實現柯里化**
   - 使用嵌套函數定義。
     ```python
     def multiply(x):
         return lambda y: x * y

     double = multiply(2)
     print(double(10))  # 輸出 20
     ```

2. **使用高階函數工具**
   - 利用 `functools.partial` 進行部分應用（類似柯里化）。
     ```python
     from functools import partial

     def power(base, exponent):
         return base ** exponent

     square = partial(power, exponent=2)
     print(square(4))  # 輸出 16
     ```

3. **通用的柯里化函數**
   - 撰寫一個通用工具，將多參數函數自動柯里化。
     ```python
     def curry(func):
         def curried(*args):
             if len(args) >= func.__code__.co_argcount:
                 return func(*args)
             return lambda *more_args: curried(*(args + more_args))
         return curried

     # 使用範例
     def add_three(a, b, c):
         return a + b + c

     curried_add = curry(add_three)
     print(curried_add(1)(2)(3))  # 輸出 6
     ```

---

#### **4.3 為什麼使用柯里化？**

1. **部分應用的便利性**
   - 柯里化允許預先固定部分參數，簡化後續操作。
     ```python
     def greet(greeting):
         return lambda name: f"{greeting}, {name}!"

     say_hello = greet("Hello")
     print(say_hello("Alice"))  # 輸出 "Hello, Alice!"
     ```

2. **提升程式的模組化與重用性**
   - 複雜運算可以被分解為更小的單位函數。
     ```python
     def is_divisible_by(divisor):
         return lambda number: number % divisor == 0

     is_even = is_divisible_by(2)
     print(is_even(10))  # 輸出 True
     ```

3. **簡化高階函數操作**
   - 與 `map`、`filter` 等高階函數結合時，柯里化能提升程式可讀性。
     ```python
     numbers = [1, 2, 3, 4]
     increment = curry(lambda x, y: x + y)(1)
     print(list(map(increment, numbers)))  # [2, 3, 4, 5]
     ```

---

#### **4.4 柯里化的限制與挑戰**

1. **語法冗長**
   - 在非純函數式語言（如 Python）中，柯里化需要額外的函數嵌套，可能導致程式碼不夠直觀。

2. **效能影響**
   - 多層次函數的嵌套會增加額外的調用成本，對性能敏感的程式需謹慎使用。

3. **應用場景有限**
   - 柯里化更適用於純函數場景，對於需要處理狀態或副作用的程式未必合適。

---

#### **4.5 實務應用中的柯里化**

1. **Web 開發中的 URL 生成**
   - 使用部分應用來動態生成帶參數的 URL。
     ```python
     def create_url(base_url):
         return lambda endpoint: lambda **params: f"{base_url}/{endpoint}?" + "&".join(f"{k}={v}" for k, v in params.items())

     api = create_url("https://api.example.com")
     user_endpoint = api("users")
     print(user_endpoint(id=123))  # https://api.example.com/users?id=123
     ```

2. **數據處理中的邏輯分解**
   - 將複雜的數據處理邏輯拆分為小型柯里化函數。
     ```python
     def transform(add, multiply):
         return lambda x: (x + add) * multiply

     curried_transform = curry(transform)
     add_5 = curried_transform(5)
     multiply_by_2 = add_5(2)
     print(multiply_by_2(10))  # 輸出 30
     ```

3. **機器學習中的管線化**
   - 在特徵工程與數據清理過程中，柯里化可簡化參數的傳遞與組合。

---

#### **4.6 小結**

柯里化是一種強大的函數式程式設計技術，特別適合於構建靈活、模組化與易於重用的程式邏輯。透過柯里化，我們可以逐步應用參數、簡化高階函數操作，並提升程式的抽象能力。儘管在 Python 中語法相對繁瑣，但透過高階函數與工具的結合，我們可以在實務中充分發揮柯里化的潛力。