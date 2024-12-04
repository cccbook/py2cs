### 第 6 章：Church Boolean (丘奇的布林系統)

本章介紹了 Church Boolean 的設計邏輯，這是一種完全基於 λ 演算的布林邏輯系統，可以模擬布林值（`True` 和 `False`）以及邏輯運算（如 `AND`、`OR`、`NOT` 等）。

---

#### **1. 定義基礎布林邏輯**

在 Church Boolean 中，我們將邏輯值和操作定義為純 λ 函數：

- **TRUE** 和 **FALSE** 是兩個 λ 函數，分別返回其兩個參數中的第一個或第二個。
- **IF** 是條件選擇操作，根據條件執行對應的邏輯分支。

```py
IF    = lambda c: lambda x: lambda y: c(x)(y) 
TRUE  = lambda x: lambda y: x
FALSE = lambda x: lambda y: y

print(IF(TRUE)("Yes")("No"))   # 輸出 "Yes"
print(IF(FALSE)("Yes")("No"))  # 輸出 "No"
```

執行結果：

```sh
Yes
No
```

---

#### **2. 定義邏輯操作**

基於布林值的邏輯運算如下：

- **AND**：當兩個條件都為 TRUE 時，結果為 TRUE。
- **OR**：只要任一條件為 TRUE，結果即為 TRUE。
- **XOR**：當且僅當兩個條件不相同，結果為 TRUE。
- **NOT**：對布林值取反。

```py
AND   = lambda p: lambda q: p(q)(p)
OR    = lambda p: lambda q: p(p)(q)
XOR   = lambda p: lambda q: p(NOT(q))(q)
NOT   = lambda c: c(FALSE)(TRUE)

print(IF(AND(TRUE)(TRUE))("Yes")("No"))  # 輸出 "Yes"
print(IF(OR(TRUE)(FALSE))("Yes")("No"))  # 輸出 "Yes"
print(IF(XOR(TRUE)(FALSE))("Yes")("No"))  # 輸出 "Yes"
print(IF(NOT(TRUE))("Yes")("No"))  # 輸出 "No"
```

#### 完整程式碼

檔案： churchBool.py

```py
# Church Booleans : Logic
IF    = lambda c:lambda x:lambda y:c(x)(y) #  if: λ c x y. c x y # if c then x else y.
TRUE  = lambda x:lambda y:x # if true then x # 兩個參數執行第一個
FALSE = lambda x:lambda y:y # if false then y # 兩個參數執行第二個
AND   = lambda p:lambda q:p(q)(p) # if p then q else p
OR    = lambda p:lambda q:p(p)(q) # if p then p else q
XOR   = lambda p:lambda q:p(NOT(q))(q) #  if p then not q else q
NOT   = lambda c:c(FALSE)(TRUE) # if c then false else true

ASSERT = lambda truth: (IF(truth)
    (lambda description:f'[✓] ${description}')
    (lambda description:f'[✗] ${description}')
)

REFUTE = lambda truth:ASSERT(NOT(truth))

TEST   = lambda description:lambda assertion:\
    print(assertion(description))

TEST('TRUE')\
    (ASSERT(TRUE))

TEST('FALSE')\
    (REFUTE(FALSE))

TEST('AND')\
  (ASSERT(AND(TRUE)(TRUE)))

TEST('OR')(ASSERT(AND\
  (AND(OR(TRUE)(FALSE))(OR(FALSE)(TRUE)))\
  (NOT(OR(FALSE)(FALSE)))))

TEST('XOR')(ASSERT(AND\
  (AND(XOR(TRUE)(FALSE))(XOR(FALSE)(TRUE)))\
  (NOT(XOR(TRUE)(TRUE)))))

TEST('NOT')\
  (REFUTE(NOT(TRUE)))
```

執行結果

```
$ python churchBool.py   
[✓] $TRUE
[✓] $FALSE
[✓] $AND
[✓] $OR
[✓] $XOR
[✓] $NOT
```

#### **4. 分析與總結**

Church Boolean 的核心在於將邏輯值和操作完全基於 λ 演算構建。以下是其特性與應用：

1. **純函數特性**：布林值和邏輯運算都是 λ 函數，具有極高的抽象性。
2. **可擴展性**：可以基於這套系統構建更複雜的邏輯，如條件表達式、多重選擇等。
3. **驗證清晰**：使用自定義的測試框架驗證邏輯正確性，增加可讀性和可靠性。

---

#### **下一步探索**

1. 擴展邏輯運算，例如 `NAND`、`NOR` 等。
2. 與其他 Church Encoding 系統結合，如數字與操作。
3. 用 λ 演算設計簡單的邏輯電路模擬器。

