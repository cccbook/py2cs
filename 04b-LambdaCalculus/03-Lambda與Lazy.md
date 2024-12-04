### 第 3 章：Lambda 與 Lazy (延遲求值)

本章討論了 Python 中延遲求值 (Lazy Evaluation) 的概念，並展示了它如何解決在遞迴計算中可能導致的問題。為此，我們將比較兩個範例程式，分別為未使用延遲求值的 `no_lazy.py` 和使用延遲求值的 `lazy.py`。

## 未使用延遲求值

在 `no_lazy.py` 中，我們定義了一個簡單的條件函數 `IF`，並用它來實現階層函數 `FACTORIAL`。  

檔案內容：`no_lazy.py`

```py
def IF(cond, job_true, job_false):
    if cond:
        return job_true
    else:
        return job_false

print(f'IF(True, "Yes", "No") = {IF(True, "Yes", "No")}')

# 階層 FACTORIAL(n) = n!
def FACTORIAL(n):
    return IF(n == 0, 1, n * FACTORIAL(n - 1))

print(f'FACTORIAL(3) = {FACTORIAL(3)}')
```

---

### 問題分析

1. **`IF` 的工作原理**：  
   `IF(cond, job_true, job_false)` 是一個簡單的條件函數，直接返回 `job_true` 或 `job_false` 的值。
   
2. **問題出現**：  
   當 `FACTORIAL(n)` 遞迴到 `n == 0` 時，`IF` 的第三個參數 `n * FACTORIAL(n - 1)` 會被**立即計算**，即使條件 `cond` 不為真。這導致了不必要的遞迴執行，進而導致程式崩潰 (stack overflow)。

---

### 執行結果

```sh
$ python no_lazy.py
IF(True, "Yes", "No") = Yes
Traceback (most recent call last):
  ...
RecursionError: maximum recursion depth exceeded
```

---

## 使用延遲求值

在 `lazy.py` 中，我們對 `IF` 的邏輯進行改進，改用 **Lambda 函數** 將參數包裝起來，使其在條件成立時才執行相關計算。

檔案內容：`lazy.py`

```py
def IF(cond, job_true, job_false):
    if cond:
        return job_true()
    else:
        return job_false()

# 階層 FACTORIAL(n) = n!
def FACTORIAL(n):
    return IF(n == 0, lambda: 1, lambda: n * FACTORIAL(n - 1))

print(f'FACTORIAL(3) = {FACTORIAL(3)}')
print(f'FACTORIAL(5) = {FACTORIAL(5)}')
```

---

### 解決方案分析

1. **延遲求值的實現**：  
   使用 Lambda 函數 (`lambda`) 將 `job_true` 和 `job_false` 包裝起來，僅當對應條件成立時才執行計算。
   
2. **如何解決問題**：  
   當條件 `cond` 為 `True` 時，`job_true()` 才會執行；若為 `False`，則執行 `job_false()`。因此，避免了遞迴過程中不必要的計算，從而成功執行。

---

### 執行結果

```sh
$ python lazy.py
FACTORIAL(3) = 6
FACTORIAL(5) = 120
```

---

## 延遲求值的優勢

1. **提升效能**：  
   延遲求值避免了條件分支中不必要的計算，尤其在深度遞迴或計算密集的情況下，顯著提升效能。

2. **解決遞迴問題**：  
   在 `FACTORIAL` 的實現中，未使用延遲求值會導致遞迴深度超過限制，而延遲求值則有效解決了這一問題。

---

### 小結

延遲求值 (Lazy Evaluation) 是函數式程式設計中的一個重要概念，通過將計算的執行推遲到必要時才進行，我們能夠避免不必要的開銷，並且在處理遞迴問題時特別有效。通過 Lambda 函數的靈活使用，我們能輕鬆實現延遲求值並解決實際問題。