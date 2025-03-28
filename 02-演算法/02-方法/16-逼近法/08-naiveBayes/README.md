# Naive Bayes

* [單純貝氏分類器](https://zh.wikipedia.org/wiki/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%88%86%E7%B1%BB%E5%99%A8)
* [AI - Ch15 機器學習(3), 樸素貝葉斯分類器 Naive Bayes classifier](https://mropengate.blogspot.com/2015/06/ai-ch14-3-naive-bayes-classifier.html)
* [醫學統計學](https://wangcc.me/LSHTMlearningnote/)
    * [第 84 章 馬爾可夫鏈蒙特卡羅MCMC，圖形模型，BUGS語言](https://wangcc.me/LSHTMlearningnote/MCMC-methods.html)

## Naive Bayes Algorithm

Naive Bayes Algorithm : P(x,y,z) = P(x) P(y) P(z)

## 參考專案

* https://github.com/kwichmann/bayes-classifier-js

## naiveProb.py

P(x,y,z) = P(x) * P(y) * P(z)

機率

```
prob = {
  'x': 0.5,
  'y': 0.2,
  'z': 0.3
}
```

執行結果

```
$ python3 naiveProb.py
P(x,y,z) = P(x)P(y)P(z)= 0.03
```

## naiveBayesProb.py

機率

```
prob = {
  'c1': 0.6, 'c2': 0.4,
  'c1=>f1': 0.5, 'c1=>f2': 0.8, 'c1=>f3': 0.6,
  'c2=>f1': 0.7, 'c2=>f2': 0.6, 'c2=>f3': 0.2,
}
```

執行結果

```
$ python3 naiveBayesProb.py
P(c1|f1,f2) = 0.24
P(c2|f1,f2) = 0.16799999999999998
```

## naiveBayesClassifier.py

```
$ python3 naiveBayesClassifier.py
P(c1|f1,f2) = 0.24
P(c2|f1,f2) = 0.16799999999999998
p= [0.24, 0.16799999999999998]
P(c1|f1,f2) = 0.24
P(c2|f1,f2) = 0.16799999999999998
c1 的機率最大
```


