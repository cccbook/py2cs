### 第 5 章：用 Closure (閉包) 儲存資料

本章介紹如何使用閉包 (Closure) 實現簡單的資料結構，並基於閉包設計操作方法。我們以 **PAIR (對偶)** 的實現為基礎，進一步構建範圍 (RANGE)、遍歷 (EACH) 以及映射 (MAP) 等功能，模擬出類似鏈表的行為。

#### 閉包 (Closure) 範例

在 Python， JavaScript 這類的程式語言裡，支援所謂的閉包 (Closure) 機制，這種機制會在函數執行時，將使用到的外部變數，抓進函數內部來存放，以便後續使用。

例如以下函數，當我們執行 add_5 = make_adder(5) 的時候， x=5 已經被抓進來綁定了，於是當我們執行 add_5(10) 的時候， 就會得到 x:5 + y:10 = 15 的結果。

```py
def make_adder(x):
    def adder(y):
        return x + y
    return adder

add_5 = make_adder(5)
print(add_5(10))  # 輸出 15
```

這種將外部函數綁定後，給內部使用的機制，就稱為閉包 Closure。

#### **基本資料結構：配對 (PAIR)**

**PAIR** 是用閉包構建的一個簡單的二元組。它能儲存兩個值，並通過提供的選擇參數 (`sel`) 來取得對應的值。

```py
PAIR = lambda x: lambda y: lambda sel: x if sel == 0 else y
HEAD = lambda p: p(0)
TAIL = lambda p: p(1)
CONS = PAIR
CAR = HEAD
CDR = TAIL
```

---

#### **PAIR 的基本操作**

利用 `PAIR` 來儲存並操作數據：

```py
p = PAIR(3)(5)
print(f'p(0) = {p(0)}')  # 取得第一個值
print(f'p(1) = {p(1)}')  # 取得第二個值
print(f'HEAD(p) = {HEAD(p)}')  # 等價於 p(0)
print(f'TAIL(p) = {TAIL(p)}')  # 等價於 p(1)
```

執行結果：

```sh
p(0) = 3
p(1) = 5
HEAD(p) = 3
TAIL(p) = 5
```

---

#### **PAIR 的嵌套結構**

`PAIR` 可以嵌套構建，形成樹狀或鏈式結構。

```py
p2 = PAIR(p)(p)
print(f'HEAD(HEAD(p2)) = {HEAD(HEAD(p2))}')  # 最內層 HEAD(p) 的值
print(f'TAIL(HEAD(p2)) = {TAIL(HEAD(p2))}')  # 最內層 TAIL(p) 的值
```

執行結果：

```sh
HEAD(HEAD(p2)) = 3
TAIL(HEAD(p2)) = 5
```

---

### **RANGE：範圍生成器**

利用 `PAIR`，實現一個類似遞迴列表的結構，用於生成從 `m` 到 `n` 的範圍。

```py
RANGE = lambda m: lambda n: PAIR(m)(None) if m == n else PAIR(m)(RANGE(m + 1)(n))
```

- `PAIR(m)(None)`：當 `m == n` 時，結束範圍生成。
- `PAIR(m)(RANGE(m + 1)(n))`：繼續生成範圍。

#### **操作範例**

```py
r = RANGE(3)(5)
print(f'HEAD(r) = {HEAD(r)}')                  # 範圍的第一個值
print(f'TAIL(r) = {TAIL(r)}')                  # 剩餘部分
print(f'HEAD(TAIL(r)) = {HEAD(TAIL(r))}')      # 第二個值
print(f'HEAD(TAIL(TAIL(r))) = {HEAD(TAIL(TAIL(r)))}')  # 第三個值
```

執行結果：

```sh
HEAD(r) = 3
TAIL(r) = <closure>
HEAD(TAIL(r)) = 4
HEAD(TAIL(TAIL(r))) = 5
```

---

### **EACH：遍歷**

定義 `EACH` 用於遍歷結構中的每個元素並執行指定函數。

```py
EACH = lambda x: lambda f: \
    f(x) if x == None or isinstance(x, int) \
    else (f(HEAD(x)), EACH(TAIL(x))(f))[-1]
```

- 當 `x` 是數值或 `None` 時，直接應用函數 `f`。
- 當 `x` 是結構時，遞迴處理 `HEAD` 和 `TAIL`。

#### **操作範例**

```py
EACH(r)(lambda x: print(x))  # 遍歷並打印範圍 [3, 4, 5]
```

執行結果：

```sh
3
4
5
```

---

### **MAP：映射**

定義 `MAP` 將結構中的每個元素應用函數轉換，並返回新的結構。

```py
MAP = lambda x: lambda f: \
    None if x == None \
    else f(x) if isinstance(x, int) \
    else PAIR(MAP(HEAD(x))(f))(MAP(TAIL(x))(f))
```

- 當 `x` 是 `None` 時返回 `None`，表示結束。
- 當 `x` 是數值時應用函數 `f`。
- 否則，對 `HEAD` 和 `TAIL` 遞迴映射。

#### **操作範例**

```py
m = MAP(r)(lambda x: x * 2)  # 將範圍 [3, 4, 5] 映射為 [6, 8, 10]
EACH(m)(print)
```

執行結果：

```sh
6
8
10
```

---

### **總結**

本章利用閉包構建了簡單的資料結構，並實現了基本操作：

- **PAIR**：二元組的存取方法。
- **RANGE**：生成遞迴結構的範圍。
- **EACH**：遍歷範圍中的每個元素。
- **MAP**：對範圍內的元素應用映射。

這些閉包的使用展示了函數式設計的靈活性，並為更多進階功能打下基礎。