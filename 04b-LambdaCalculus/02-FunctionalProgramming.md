### 第 2 章：Functional Programming

在這一章中，我們將學習函數式程式設計的基礎概念，以及如何用 Python 實現常見的函數式編程操作，例如 `map`、`filter`、`reduce` 等。以下程式碼展示了這些操作的具體實現。

#### 遍歷函數 `EACH`

`EACH(a, f)` 接受一個列表 `a` 和一個函數 `f`，對 `a` 中的每個元素執行 `f`。

```py
def EACH(a, f):
    for x in a:
        f(x)
```

範例：  
```py
a = [1,2,3,4,5]
EACH(a, lambda x: print(f"Element: {x}"))
# 輸出:
# Element: 1
# Element: 2
# Element: 3
# Element: 4
# Element: 5
```

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
a = [1,2,3,4,5]
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
a = [1,2,3,4,5]
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
a = [1,2,3,4,5]
print(REDUCE(a, lambda x, y: x + y, 0))
# 輸出: 15
```


#### 完整的程式碼

以下是完整的程式碼範例，展示了如何使用上述函數：

檔案： fp.py

```py
def EACH(a, f):
    for x in a:
        f(x)

def MAP(a, f):
    r = []
    for x in a:
        r.append(f(x))
    return r

def FILTER(a, f):
    r = []
    for x in a:
        if f(x): r.append(x)
    return r

def REDUCE(a, f, init):
    r = init
    for x in a:
        r = f(r, x)
    return r

if __name__=="__main__":
    a = [1,2,3,4,5]
    EACH(a, lambda x:print(x))
    print(MAP(a, lambda x:x*x))
    print(FILTER(a, lambda x:x%2==1))
    print(REDUCE(a, lambda x,y:x+y, 0))
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

## 不倚賴迴圈的函數式編程

上述的版本有使用到 for 迴圈，但是很多函數式編程語言裡面，其實是沒有 for, while 這類的回圈的，像是 list, haskell 等等。

不使用 for, while 回圈，怎麼完成 EACH, MAP, FILTER, REDUCE 這些函數呢？

以下是我們將上述程式用遞迴取代回圈的結果。

檔案: fp_noloop.py

```py
def _each(a, f, i):
    if i==len(a):
        return
    else:
        f(a[i])
        _each(a, f, i+1)

def EACH(a, f):
    _each(a,f,0)

def _map(a, f, i, r):
    if i==len(a):
        return
    else:
        r.append(f(a[i]))
        _map(a, f, i+1, r)

def MAP(a, f):
    r = []
    _map(a, f, 0, r)
    return r

def _filter(a, f, i, r):
    if i == len(a):
        return
    else:
        if f(a[i]): r.append(a[i])
        _filter(a, f, i+1, r)

def FILTER(a, f):
    r = []
    _filter(a, f, 0, r)
    return r

def _reduce(a, f, i, r):
    if i == len(a):
        return r
    else:
        r = f(r, a[i])
        return _reduce(a, f, i+1, r)

def REDUCE(a, f, init):
    return _reduce(a, f, 0, init)

if __name__=="__main__":
    a = [1,2,3,4,5]
    EACH(a, lambda x:print(x))
    print(MAP(a, lambda x:x*x))
    print(FILTER(a, lambda x:x%2==1))
    print(REDUCE(a, lambda x,y:x+y, 0))
```

執行結果

```
$ python fp_noloop.py
1
2
3
4
5
[1, 4, 9, 16, 25]
[1, 3, 5]
15
```

### 小結

函數式程式設計使我們能以簡潔且抽象的方式處理列表操作。透過設計這些高階函數，我們可以大幅減少重複的程式碼，並提高程式的可讀性和維護性。