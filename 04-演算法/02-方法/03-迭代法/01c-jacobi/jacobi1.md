

ChatGPT: https://chatgpt.com/c/67036a09-57fc-8012-b27e-ad9a88c64b5a

Jacobi 方法是一種數值方法，用來求解線性方程組 \(Ax = b\)，其中 \(A\) 是一個方陣，\(x\) 是未知向量，\(b\) 是已知向量。Jacobi 方法通過迭代更新來逼近解。其基本思想是將方程組中的每個方程孤立出來，利用其餘變量的當前值來更新每個變量。

### Jacobi 方法的公式：
對於線性方程 \(Ax = b\)，Jacobi 方法的迭代公式為：
\[
x_i^{(k+1)} = \frac{1}{A_{ii}} \left( b_i - \sum_{j \neq i} A_{ij} x_j^{(k)} \right)
\]
每次迭代都會更新 \(x\) 的每個分量，直到滿足收斂條件。

這是一個不使用 `numpy` 的 Jacobi 方法實現，僅使用 Python 原生的資料結構來解線性方程組 \(Ax = b\)。

### Python Jacobi 方法（無 `numpy`）

```python
def jacobi(A, b, x0, tol=1e-10, max_iters=1000):
    """
    使用 Jacobi 方法解線性方程組 Ax = b (不使用 numpy)

    參數:
    A : 系數矩陣 (n x n), 以列表的列表形式傳入
    b : 常數向量 (n x 1), 以列表形式傳入
    x0 : 初始猜測向量
    tol : 容差（迭代的收斂條件）
    max_iters : 最大迭代次數

    返回:
    x : 解向量 (n x 1), 以列表形式返回
    """
    n = len(b)
    x = x0[:]
    x_new = [0] * n
    
    for iteration in range(max_iters):
        for i in range(n):
            sum_ = 0
            for j in range(n):
                if i != j:
                    sum_ += A[i][j] * x[j]
            x_new[i] = (b[i] - sum_) / A[i][i]

        # 計算 x_new 和 x 之間的最大差異
        diff = max(abs(x_new[i] - x[i]) for i in range(n))

        # 判斷是否收斂
        if diff < tol:
            print(f"Converged after {iteration + 1} iterations")
            return x_new
        
        # 更新 x 值
        x = x_new[:]
    
    print("Did not converge within the maximum number of iterations")
    return x_new

# 範例使用 Jacobi 方法
if __name__ == "__main__":
    # 定義系數矩陣 A 和常數向量 b
    A = [[4, -1, 0, 0],
         [-1, 4, -1, 0],
         [0, -1, 4, -1],
         [0, 0, -1, 3]]
    
    b = [15, 10, 10, 10]
    
    # 初始猜測向量
    x0 = [0, 0, 0, 0]
    
    # 使用 Jacobi 方法解 Ax = b
    x = jacobi(A, b, x0)
    
    print("Solution:", x)
```

### 程式說明：

1. **`jacobi` 函數**：
   - `A` 是系數矩陣，以列表的列表形式傳入。
   - `b` 是常數向量，為一個列表。
   - `x0` 是初始猜測解，為一個列表。
   - 每次迭代時，根據 Jacobi 公式更新解向量，並檢查與前一次迭代結果的最大差值。如果差值小於容差 `tol`，則認為收斂。

2. **範例**：
   - 系數矩陣 \( A \) 是 4x4 的列表列表，常數向量 \( b \) 為 [15, 10, 10, 10]。
   - 初始解 \( x0 \) 設為全零向量 [0, 0, 0, 0]。

3. **結果**：
   - 程式會輸出收斂後的解向量，以及顯示經過了多少次迭代。

### 程式輸出範例：
```
Converged after X iterations
Solution: [解1, 解2, 解3, 解4]
```

這段程式碼無需依賴 `numpy`，並且使用了 Python 原生的列表操作來進行數學運算。