### 第 7 章：Church Numeral (丘奇的數值系統)

本章將介紹 **Church Numeral 系統**，這是數學家 **Alonzo Church** 在 lambda 演算中提出的一種表示數字的方法。與傳統的數值表示不同，Church Numeral 以函數的形式來表達自然數，並且通過遞迴與組合來進行數值運算。了解 Church Numeral 系統能夠加深我們對 lambda 演算的理解，並展現函數式程式設計中數值處理的優雅性。

---

#### **7.1 Church Numeral 的基本概念**

1. **Church Numeral 的定義**
   - 在 Church Numeral 系統中，數字並不是直接用常規的數字來表示，而是使用一個函數，該函數重複執行某個操作的次數來表示數字。
   - Church Numeral 的基本思想是使用一個函數來模擬自然數的行為。這個函數接收一個參數（通常是函數本身）並返回一個新的函數。

2. **基本數字的表示**
   - **零（0）** 是一個接受兩個參數的函數，並直接返回第二個參數：
     ```python
     def ZERO(f):
         return lambda x: x
     ```
     這裡，`ZERO` 函數不對 `f` 做任何操作，直接返回 `x`，相當於 "不進行任何操作"。

   - **一（1）** 是一個接受兩個參數的函數，並將第一個參數 `f` 應用一次於第二個參數 `x`：
     ```python
     def ONE(f):
         return lambda x: f(x)
     ```

   - **二（2）** 是一個接受兩個參數的函數，並將 `f` 應用兩次於 `x`：
     ```python
     def TWO(f):
         return lambda x: f(f(x))
     ```

   - 以此類推，Church Numeral 系統中的數字是基於這種遞迴結構來定義的。數字 `n` 就是將 `f` 應用 `n` 次於 `x`。

   範例：
   - `ZERO` 表示 0：`λf.λx.x`
   - `ONE` 表示 1：`λf.λx.f(x)`
   - `TWO` 表示 2：`λf.λx.f(f(x))`
   - `THREE` 表示 3：`λf.λx.f(f(f(x)))`

---

#### **7.2 使用 Church Numeral 進行加法**

1. **加法操作**
   - Church Numeral 中的加法是通過將兩個數字表示的函數合併來實現的。
   - 具體來說，兩個數字 `m` 和 `n` 的和是將函數 `f` 應用 `m` 次後，再將函數 `f` 應用 `n` 次。

   **加法的定義**：
   ```python
   def ADD(m):
       return lambda n: lambda f: lambda x: m(f)(n(f)(x))
   ```

   這個定義的含義是：首先應用 `m(f)`，再應用 `n(f)`，以此方式累加兩個數字。

   範例：
   ```python
   def THREE(f):
       return lambda x: f(f(f(x)))

   def FOUR(f):
       return lambda x: f(f(f(f(x))))

   add_three_and_four = ADD(THREE)(FOUR)
   print(add_three_and_four(lambda x: x + 1)(0))  # 輸出 7
   ```

2. **加法解釋**
   - 在這個範例中，`THREE` 和 `FOUR` 是分別表示 3 和 4 的 Church Numerals。加法函數將 `f` 應用 3 次，再將 `f` 應用 4 次，最終得到 7。

---

#### **7.3 使用 Church Numeral 進行乘法**

1. **乘法操作**
   - Church Numeral 中的乘法是將兩個數字表示的函數進行組合，將第一個數字的函數應用於第二個數字的結果上。
   - 具體來說，兩個數字 `m` 和 `n` 的乘積是將 `m(f)` 應用 `n` 次，即將 `n(f)` 的結果重複 `m` 次。

   **乘法的定義**：
   ```python
   def MULT(m):
       return lambda n: lambda f: m(n(f))
   ```

   範例：
   ```python
   def FIVE(f):
       return lambda x: f(f(f(f(f(x)))))

   def SIX(f):
       return lambda x: f(f(f(f(f(f(x))))))

   multiply_five_and_six = MULT(FIVE)(SIX)
   print(multiply_five_and_six(lambda x: x + 1)(0))  # 輸出 30
   ```

2. **乘法解釋**
   - `FIVE` 和 `SIX` 分別表示 5 和 6。乘法函數會將 `SIX(f)` 重複應用 5 次，最終得到 30。

---

#### **7.4 Church Numeral 系統的其他操作**

1. **減法操作**
   - 在 Church Numeral 中，減法並不像加法或乘法那樣簡單。為了實現減法，通常需要引入一種稱為 "先驗 (Predecessor)" 的方法，該方法用來計算一個數字的前一個數字。

2. **比較操作**
   - 在 Church Numeral 系統中，我們可以定義比較兩個數字的操作（例如：大於、等於）。這些操作會根據函數的應用次數來進行判斷。

---

#### **7.5 Church Numeral 系統的優點與局限性**

1. **優點**
   - Church Numeral 系統完全基於函數式程式設計的原則，展現了 lambda 演算如何以純函數的方式表示數字及其運算。
   - 該系統不依賴數字本身，而是將數字的概念與函數操作綁定在一起，這對於理解計算理論和函數式程式設計非常重要。

2. **局限性**
   - Church Numeral 系統的運算相對較為繁瑣，並且對於較大的數字計算效率不高。在實際的應用中，我們更傾向於使用傳統的數值類型。
   - 由於其本質是基於遞迴，使用 Church Numerals 進行大規模的數值計算可能會遇到性能瓶頸。

---

#### **7.6 小結**

Church Numeral 系統展示了如何用純函數的方式來表示數字及其運算。這一系統不僅有助於理解 lambda 演算的運作方式，也提供了一個簡潔且富有理論意義的數字表示方法。雖然在現代程式設計中，這種表示方式並不常見，但它為理解函數式程式設計和計算理論提供了寶貴的視角。