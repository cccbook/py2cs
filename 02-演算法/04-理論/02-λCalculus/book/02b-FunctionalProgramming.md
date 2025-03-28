### 第 2 章：Functional Programming

在本章中，我們將深入探討**函數式程式設計 (Functional Programming)** 的概念。這是一種以數學函數為基礎的編程範式，強調不變性 (Immutability)、純函數 (Pure Functions) 和高階函數 (Higher-Order Functions)。函數式程式設計的核心思想是避免副作用 (Side Effects)，從而提升程式的可預測性和可測試性。

---

#### **2.1 函數式程式設計的基礎概念**
1. **純函數 (Pure Functions)**  
   - 輸出僅依賴輸入，無外部副作用。
   - 範例：
     ```python
     def add(x, y):
         return x + y
     ```
     此函數不依賴全域變數，並且不改變任何外部狀態。

2. **不變性 (Immutability)**  
   - 資料結構一旦創建便無法改變。
   - 範例：
     ```python
     original_list = [1, 2, 3]
     new_list = original_list + [4]
     ```

3. **高階函數 (Higher-Order Functions)**  
   - 函數可以作為參數或回傳值。
   - 範例：
     ```python
     def apply_function(func, value):
         return func(value)

     apply_function(lambda x: x * 2, 10)  # 回傳 20
     ```

---

#### **2.2 Lambda 演算 (Lambda Calculus)**
Lambda 演算是函數式程式設計的數學基礎。它由以下基本概念構成：
1. **變數 (Variable)**  
   符號代表輸入，範例：`x`

2. **函數 (Function)**  
   定義輸入與輸出關係，範例：`λx.x+1`

3. **應用 (Application)**  
   將函數應用於輸入，範例：`(λx.x+1) 2`，結果為 `3`。

---

#### **2.3 Python 中的函數式工具**
Python 雖然不是純函數式語言，但它提供了支援函數式程式設計的工具。

1. **`map` 函數**  
   - 將函數應用於每個元素。
   - 範例：
     ```python
     numbers = [1, 2, 3]
     doubled = map(lambda x: x * 2, numbers)
     print(list(doubled))  # [2, 4, 6]
     ```

2. **`filter` 函數**  
   - 篩選滿足條件的元素。
   - 範例：
     ```python
     numbers = [1, 2, 3, 4]
     evens = filter(lambda x: x % 2 == 0, numbers)
     print(list(evens))  # [2, 4]
     ```

3. **`reduce` 函數**  
   - 累積地應用函數。
   - 範例：
     ```python
     from functools import reduce
     numbers = [1, 2, 3, 4]
     product = reduce(lambda x, y: x * y, numbers)
     print(product)  # 24
     ```

---

#### **2.4 遞歸 (Recursion)**
函數式程式設計中的遞歸是一個核心技巧，透過讓函數調用自身來解決問題。

範例：計算階乘
```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # 120
```

---

#### **2.5 函數式程式設計的優勢**
1. **可測試性高**：純函數的行為是可預測的，易於測試。
2. **易於並行化**：無副作用使並行計算更安全。
3. **表達力強**：高階函數和不變性可以簡化邏輯。

---

#### **2.6 小結**
函數式程式設計是一種強大而優雅的編程方式，特別適合數據處理、並行計算和需要高穩定性的應用。在後續章節中，我們將深入探討如何使用 Python 中的函數式工具來解決實際問題，並進一步理解 Lambda 演算的應用。