### 第 2 章：Functional Programming

在這一章中，我們將學習函數式程式設計的基礎概念，以及如何用 Python 實現常見的函數式編程操作，例如 `map`、`filter`、`reduce` 等。以下程式碼展示了這些操作的具體實現。

---

#### 定義範圍函數 `RANGE`

`RANGE(m, n)` 函數的功能是產生一個從 `m` 到 `n` 的整數列表，類似於內建的 `range` 函數，但返回的是列表。

```py
def RANGE(m, n):
    r = []
    for i in range(m, n + 1):
        r.append(i)
    return r
```

範例：  
```py
print(RANGE(1, 5))
# 輸出: [1, 2, 3, 4, 5]
```

---

#### 遍歷函數 `EACH`

`EACH(a, f)` 接受一個列表 `a` 和一個函數 `f`，對 `a` 中的每個元素執行 `f`。

```py
def EACH(a, f):
    for x in a:
        f(x)
```

範例：  
```py
a = RANGE(1, 5)
EACH(a, lambda x: print(f"Element: {x}"))
# 輸出:
# Element: 1
# Element: 2
# Element: 3
# Element: 4
# Element: 5
```

---

#### 映射函數 `MAP`

`MAP(a, f)` 將函數 `f` 應用於列表 `a` 的每個元素，並返回結果列表。

```py
def MAP(a, f):
    r = []
    for x in a:
        r.append(f(x))
    return r
```

範例：  
```py
a = RANGE(1, 5)
print(MAP(a, lambda x: x * x))
# 輸出: [1, 4, 9, 16, 25]
```

---

#### 過濾函數 `FILTER`

`FILTER(a, f)` 根據條件函數 `f` 過濾列表 `a` 的元素，並返回符合條件的結果列表。

```py
def FILTER(a, f):
    r = []
    for x in a:
        if f(x):
            r.append(x)
    return r
```

範例：  
```py
a = RANGE(1, 5)
print(FILTER(a, lambda x: x % 2 == 1))
# 輸出: [1, 3, 5]
```

---

#### 累積函數 `REDUCE`

`REDUCE(a, f, init)` 使用函數 `f` 對列表 `a` 進行累積操作，初始值為 `init`。

```py
def REDUCE(a, f, init):
    r = init
    for x in a:
        r = f(r, x)
    return r
```

範例：  
```py
a = RANGE(1, 5)
print(REDUCE(a, lambda x, y: x + y, 0))
# 輸出: 15
```

---

#### 主程式執行

以下是完整的程式碼範例，展示了如何使用上述函數：

```py
if __name__ == "__main__":
    a = RANGE(1, 5)
    # 遍歷並打印每個元素
    EACH(a, lambda x: print(x))
    
    # 將列表中的每個元素平方
    print(MAP(a, lambda x: x * x))
    
    # 過濾出奇數
    print(FILTER(a, lambda x: x % 2 == 1))
    
    # 將列表中的元素累加
    print(REDUCE(a, lambda x, y: x + y, 0))
```

執行結果：  
```sh
$ python fp.py
1
2
3
4
5
[1, 4, 9, 16, 25]
[1, 3, 5]
15
```

---

### 小結

函數式程式設計使我們能以簡潔且抽象的方式處理列表操作。透過設計這些高階函數，我們可以大幅減少重複的程式碼，並提高程式的可讀性和維護性。