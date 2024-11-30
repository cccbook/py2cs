### 第 3 章：Lambda 與 Lazy (延遲求值)

本章將介紹 **Lambda** 和 **Lazy (延遲求值)** 的核心概念與實踐應用。這兩個特性是函數式程式設計的重要基石，能提升程式的效率與靈活性。Lambda 為輕量化的匿名函數，適用於快速定義簡單運算；而 Lazy 則是透過延後計算需求來提升效能，避免不必要的運算。

---

#### **3.1 Lambda 表達式**
Lambda 是一種匿名函數，允許在不定義完整函數的情況下，快速撰寫小型邏輯。

1. **語法**  
   ```python
   lambda 參數1, 參數2, ...: 表達式
   ```

   範例：
   ```python
   square = lambda x: x * x
   print(square(5))  # 輸出 25
   ```

2. **使用場景**
   - **作為高階函數的參數**  
     ```python
     numbers = [1, 2, 3, 4]
     doubled = map(lambda x: x * 2, numbers)
     print(list(doubled))  # [2, 4, 6, 8]
     ```
   - **臨時定義簡單邏輯**  
     ```python
     sort_by_second = lambda pair: pair[1]
     data = [(1, 3), (2, 1), (4, 2)]
     print(sorted(data, key=sort_by_second))  # [(2, 1), (4, 2), (1, 3)]
     ```

3. **與普通函數的比較**
   - Lambda 適合用於單行的簡單運算。
   - 若邏輯複雜，應使用 `def` 明確定義函數。

---

#### **3.2 延遲求值 (Lazy Evaluation)**
延遲求值是一種策略，在需要時才計算值，而非立即執行。

1. **核心概念**
   - **即時求值 (Eager Evaluation)**：在程式執行時，立刻計算所有表達式。
   - **延遲求值 (Lazy Evaluation)**：僅在結果需要時才執行表達式。

2. **Python 中的實現方式**
   - **生成器 (Generators)**  
     Python 的生成器是 Lazy Evaluation 的典型例子。它透過 `yield` 暫存結果，並在需要時產生下一個值。
     ```python
     def lazy_range(n):
         i = 0
         while i < n:
             yield i
             i += 1

     for num in lazy_range(5):
         print(num)  # 依序輸出 0, 1, 2, 3, 4
     ```

   - **`itertools` 模組**  
     `itertools` 提供了多種延遲求值工具。
     ```python
     import itertools

     # 無窮序列
     counter = itertools.count(start=1, step=2)
     print(next(counter))  # 1
     print(next(counter))  # 3
     ```

3. **延遲求值的優勢**
   - **效能提升**：避免不必要的運算，節省記憶體。
   - **處理大數據**：能處理無窮序列或超大型數據集。

---

#### **3.3 Lambda 與 Lazy 的結合**
Lambda 表達式與 Lazy Evaluation 結合可創造靈活而高效的運算。

1. **延遲定義運算邏輯**
   ```python
   lazy_add = lambda x, y: lambda: x + y
   computation = lazy_add(5, 10)
   print(computation())  # 輸出 15
   ```

2. **與生成器配合**
   ```python
   lazy_filter = lambda func, iterable: (x for x in iterable if func(x))
   numbers = range(10)
   even_numbers = lazy_filter(lambda x: x % 2 == 0, numbers)
   print(list(even_numbers))  # [0, 2, 4, 6, 8]
   ```

---

#### **3.4 延遲求值在實務中的應用**
1. **文件處理**
   - 延遲讀取大文件，避免一次載入大量資料。
     ```python
     def read_large_file(file_path):
         with open(file_path, 'r') as file:
             for line in file:
                 yield line.strip()

     for line in read_large_file('large_file.txt'):
         print(line)
     ```

2. **數據流處理**
   - 實時處理無窮序列或動態輸入。
     ```python
     def live_data_stream():
         data = 0
         while True:
             yield data
             data += 1

     stream = live_data_stream()
     for _ in range(5):
         print(next(stream))  # 輸出 0, 1, 2, 3, 4
     ```

---

#### **3.5 小結**
Lambda 和 Lazy 是函數式程式設計中不可或缺的工具。Lambda 簡化了匿名函數的定義，使代碼更具表現力；Lazy 則透過延後運算提高效能，尤其在處理大型或無窮數據時表現卓越。本章提供了這兩者的基礎與應用，為進一步的程式設計實踐奠定基礎。