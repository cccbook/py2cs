# 貝氏網路

## ccc: 請問甚麼是貝氏網路

貝氏網路（Bayesian network）是一種用來表示隨機变量之間關係的統計學模型。它通過圖形來表示這些關係，其中包含許多用來表示變量之間依存性的節點和邊。貝氏網路可以用來預測變量的值，並且可以用來求解複雜的概率問題。

## ccc: 可以給個簡單的貝氏網路之 python 程式嗎?

您好，這是一個簡單的貝氏網路 python 程式：

```py
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

# 定義網路結構
model = BayesianModel([('A', 'C'), ('B', 'C')])

# 用資料來源更新網路模型
data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
model.fit(data, estimator=MaximumLikelihoodEstimator)

# 顯示網路的 CPDs
for cpd in model.get_cpds():
    print(cpd)

# 查詢某個節點的概率分布
query = model.predict_prob({'A': 1, 'B': 0})
print(query)
```

在這個程式中，我們定義了一個網路結構，並且用資料來源來更新網路模型。然後，我們顯示網路的概率表（也稱為条件概率分布（CPD）），最後我們查詢了某個節點的概率分布。

## ccc: 甚麼是網路的 CPDs ?

CPD 指的是条件概率分布（conditional probability distribution），它用來描述網路中某個節點的概率分布，並且是在特定的條件下求出的。在貝氏網路中，每個節點都有一個對應的 CPD，表示該節點的概率分布受其他節點的影響。通過網路的 CPDs，我們可以求出網路中任意一個節點的概率分布，這對於做出決策或進行預測非常有用。

## ccc: 上述程式的執行結果為何?

這是一個模擬的例子，所以執行結果可能會有所不同。

假設我們定義的網路結構如下：

```py
model = BayesianModel([('A', 'C'), ('B', 'C')])
```

這表示我們有三個節點（A、B、C），並且它們之間的關係如下：

A 是 C 的父節點

B 是 C 的父節點

接下來，我們用資料來源來更新網路模型：

```py
data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
model.fit(data, estimator=MaximumLikelihoodEstimator)
```

這表示我們用三筆資料來更新網路模型。每筆資料中包含了 A、B 和 C 三個變量的值。

接下來，我們可以顯示網路的 CPDs：

```py
for cpd in model.get_cpds():
    print(cpd)
```

這可能會顯示出如下的資訊：

```
+-----+-----+------+
| A   | 0   | 0.67 |
|     | 1   | 0.33 |
+-----+-----+------+
| B   | 0   | 0.67 |
|     | 1   | 0.33 |
+-----+-----+------+
| A   | 0   | 0.5  |
|     | 1   | 0.5  |
+-----+-----+------+
| B   | 0   | 0.5  |
|     | 1   | 0.5  |
+-----+-----+------+
| C   | 0   | 0.33 |
|     | 1   | 0.67 |
+-----+-----+------+
```

這些數字表示每個節點在特定條件下的概率分布。

最後，我們可以通過查詢來查詢某個節點的概率分布：

```py
query = model.predict_prob({'A': 1, 'B': 0})
print(query)
```

這可能會顯示出如下的資訊：

```
+-----+-----+
| A   | 0   |
|     | 1   |
+-----+-----+
| B   | 0   |
|     | 1   |
+-----+-----+
| C   | 0   |
|     | 1   |
+-----+-----+
```

這些數字表示在給定 A 為 1 和 B 為 0 的條件下，C 的概率分布。


## ccc: 有辦法自動學習貝氏網路的 CPD 嗎?

是的，通常我們不會手動輸入 CPDs，而是通過訓練資料來學習 CPDs。在 Python 中，可以使用 pgmpy 套件來自動學習貝氏網路的 CPDs。

這是一個簡單的例子：

```py
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

# 定義網路結構
model = BayesianModel([('A', 'C'), ('B', 'C')])

# 用資料來源來學習網路的 CPDs
data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
model.fit(data, estimator=MaximumLikelihoodEstimator)

# 顯示網路的 CPDs
for cpd in model.get_cpds():
    print(cpd)
```

在這個例子中，我們首先定義了網路結構，然後用訓練資料來學習 CPDs。接下來，我們可以顯示網路的 CPDs，這些 CPDs 已經被學習得到。

## ccc: 學習貝氏網路的 CPD 該用甚麼演算法?

在學習貝氏網路的 CPDs 時，可以使用多種演算法。例如，可以使用最大似然法（maximum likelihood）或最大后驗概率（maximum a posteriori）法。

在 pgmpy 套件中，可以使用 MaximumLikelihoodEstimator 或 BayesianEstimator 類來學習 CPDs。這些類接受訓練資料並自動學習 CPDs。

例如，可以這麼寫：

```py
# 使用最大似然法學習 CPDs
model.fit(data, estimator=MaximumLikelihoodEstimator)

# 使用最大后驗概率法學習 CPDs
model.fit(data, estimator=BayesianEstimator)
```

在這兩種情況下，CPDs 都會被自動學習得到。但是，這兩種方法可能會得到不同的 CPDs，因為它們使用了不同的假設和方法。選擇哪種方法取決於實際情況。

