### **第 2 章：Lambda Calculus 的簡介**

#### **2.1 什麼是 Lambda Calculus？**

Lambda Calculus 是數學和計算機科學中的一種形式系統，由 Alonzo Church 在 1930 年代提出。它是研究函數抽象與應用的理論基礎，並被認為是計算理論的核心之一。Lambda Calculus 使用一種簡單的語法來描述所有的計算邏輯，其基本單位是函數。

Lambda Calculus 的核心目標：
- 使用簡單的語法來描述計算。
- 定義函數與它們的應用。
- 提供計算的形式化基礎。

---

#### **2.2 為什麼學習 Lambda Calculus？**

1. **理解計算的本質**  
   Lambda Calculus 是所有現代程式語言的理論基礎，特別是函數式程式設計語言如 Haskell、Lisp 和 Clojure。  
   
2. **強化邏輯思維能力**  
   學習 Lambda Calculus 需要理解純函數的抽象表示，這對程式設計邏輯的提升大有幫助。  

3. **應用於現代技術**  
   Lambda Calculus 的概念廣泛應用於編譯器、型別系統和人工智慧（如 Lambda 表達式在深度學習中的應用）。

---

#### **2.3 Lambda Calculus 的語法**

Lambda Calculus 的語法只有三個基本部分：
1. **變數 (Variable)**：表示一個函數的參數或值。  
   範例：`x`、`y`。  

2. **函數抽象 (Abstraction)**：用於定義一個函數，語法為 `λx.E`，表示一個變數 `x` 與表達式 `E`。  
   範例：`λx.x+1` 表示一個接收參數 `x` 並返回 `x+1` 的函數。

3. **函數應用 (Application)**：將一個函數應用於一個參數，語法為 `(F X)`，表示將函數 `F` 應用到參數 `X` 上。  
   範例：`(λx.x+1 2)` 表示將數字 `2` 傳入函數 `λx.x+1`，結果為 `3`。

---

#### **2.4 Lambda Calculus 的核心概念**

1. **α-轉換 (Alpha Conversion)**  
   改變函數參數的名稱，而不影響函數的邏輯。  
   範例：`λx.x` 可以改寫為 `λy.y`。

2. **β-簡化 (Beta Reduction)**  
   將函數應用於實際參數，並計算出結果。  
   範例：`(λx.x+1 2)` 簡化為 `2+1 = 3`。

3. **η-轉換 (Eta Conversion)**  
   簡化表達式，使其等價於一個更簡潔的形式。  
   範例：`λx.(F x)` 等價於 `F`，若 `F` 在 `x` 無其他依賴。

---

#### **2.5 Lambda Calculus 的基本應用**

1. **邏輯運算的表示**  
   - 定義布林值：`TRUE = λx.λy.x`，`FALSE = λx.λy.y`。
   - 定義 IF 條件：`IF = λc.λx.λy.c x y`。

2. **數字與算術運算的表示（Church Numerals）**  
   - 數字 `0`：`λf.λx.x`  
   - 數字 `1`：`λf.λx.f x`  
   - 加法運算：`ADD = λm.λn.λf.λx.m f (n f x)`。

3. **遞迴與 Y-Combinator**  
   遞迴是一種重複應用函數的技術，Y-Combinator 是 Lambda Calculus 中的遞迴工具，形式為：  
   ```
   Y = λf.(λx.f (x x)) (λx.f (x x))
   ```


#### **2.7 小結**

本章介紹了 Lambda Calculus 的歷史、語法、核心概念和基本應用，並簡單說明如何用 Python 模擬其語法。從下一章開始，我們將深入探索如何用 Python 實現邏輯運算、數字運算與資料結構，並逐步搭建 Lambda Calculus 的運算模型。