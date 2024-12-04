### 第 4 章：Curry (柯里化)

**柯里化 (Currying)** 是函數式程式設計中的一個核心概念，指的是將接受多個參數的函數轉化為一系列嵌套的單參數函數。每次函數調用都返回一個新函數，直到所有參數都被提供後，函數才真正執行計算。以下程式碼通過不同方式實現了函數的柯里化。

### 範例程式

#### **直接使用 Lambda 實現柯里化**

1. **一般函數**  
   `add_xyz_lambda` 是一個普通的 Lambda 函數，接受三個參數並返回它們的和。

   ```py
   add_xyz_lambda = lambda x, y, z: x + y + z
   print(f'add_xyz_lambda(1,2,3) = {add_xyz_lambda(1,2,3)}')
   ```

   執行結果：  
   ```sh
   add_xyz_lambda(1,2,3) = 6
   ```

2. **柯里化函數**  
   `add_xyz_curry` 是一個柯里化的 Lambda 函數，它將 `x`, `y`, `z` 三個參數分別放入三層嵌套函數中。

   ```py
   add_xyz_curry = lambda x: lambda y: lambda z: x + y + z
   print(f'add_xyz_curry(1)(2)(3) = {add_xyz_curry(1)(2)(3)}')
   ```

   執行結果：  
   ```sh
   add_xyz_curry(1)(2)(3) = 6
   ```

---

#### **用普通函數模擬柯里化**

`add_xyz` 是一個用普通函數定義的柯里化版本。它通過多層嵌套的函數來模擬 Lambda 的柯里化結構。

```py
def add_xyz(x, y, z):
    def addx_yz(y, z):
        def addxy_z(z):
            return x + y + z
        return addxy_z(z)
    return addx_yz(y, z)

print(f'add_xyz(1,2,3) = {add_xyz(1, 2, 3)}')
```

執行結果：  
```sh
add_xyz(1,2,3) = 6
```

---

### 詳細解析

1. **柯里化與非柯里化的區別**  
   - **非柯里化**：`add_xyz_lambda(1, 2, 3)` 一次性傳入所有參數並計算結果。
   - **柯里化**：`add_xyz_curry(1)(2)(3)` 每次只傳入一個參數，生成一個新函數，直到所有參數都被傳入後執行計算。

2. **柯里化的優勢**  
   - **延遲計算**：部分參數可以在需要時才提供，適合搭配延遲求值的場景。
   - **重用性高**：在某些應用中，可以為固定的參數建立新的函數，減少重複編寫。
     ```py
     add1 = add_xyz_curry(1)  # 固定 x = 1
     add1_2 = add1(2)        # 固定 x = 1, y = 2
     print(add1_2(3))        # 輸出: 6
     ```

3. **模擬柯里化的普通函數結構**  
   通過多層嵌套函數的方式，`add_xyz` 成功模擬了 Lambda 函數中的柯里化過程。雖然寫法稍顯冗長，但邏輯清晰，便於理解。

---

### 執行結果總結

```sh
add_xyz_lambda(1,2,3) = 6
add_xyz_curry(1)(2)(3) = 6
add_xyz(1,2,3) = 6
```

## 柯里化的 if

在上一章中，我們透過 lazy 的方式，讓 IF 函數在執行 FACTORIAL(n) 這類的遞迴程式時，不至於因為無窮遞迴兒當掉。

如果我們把 if 也用 curry 的方式改寫，就可以得到下列程式。

檔案： if_curry.py

```py
IF = lambda cond:lambda job_true:lambda job_false:\
    job_true() if cond else job_false()

# 階層 FACTORIAL(n) = n!
def FACTORIAL(n): 
  return IF(n==0)(lambda:1)(lambda:n*FACTORIAL(n-1))

print(f'FACTORIAL(3)={FACTORIAL(3)}')
print(f'FACTORIAL(5)={FACTORIAL(5)}')

```

執行結果

```
$ python if_curry.py
FACTORIAL(3)=6
FACTORIAL(5)=120
```

這樣的 if，就不再是有三個參數的函數，而是經過柯里化後的單參數函數。

只是當 cond 的布林條件確定之後，就能透過 job_true() 或 job_false() 的呼叫，進行真正的動作。

這樣的柯里化寫法，雖然有點奇怪，但是習慣之後，可能反而覺得非常好用。

不過本書中 Church 版本的 Lambda Calculus 裡的 IF 定義如下

```py
IF = lambda c:lambda x:lambda y:c(x)(y) #  if: λ c x y. c x y # if c then x else y.
```

因為 Church 的 Lambda Calculus 世界裡，一切皆函數，連資料也是用函數表達的，例如 TRUE, FALSE, 0, 1, 2, 3, .... 

所以直接把函數傳回即可，而不需要再次地進行求值動作，所以就不需要在 x, y 後面再加上 x(), y() 了。

### 小結

柯里化是一種強大的設計模式，特別適合用於多參數函數的分步計算。通過將參數拆分為多層嵌套結構，我們可以靈活地對參數進行部分應用或延遲計算，從而提高程式的模組化和可重用性。