# 1972 年 -- P


## 用數學語言寫程式？

對於數學家而言，如果有個程式語言可以模仿《一階邏輯》的寫法，那是否能幫助數學家發現更多更好的數學定理呢？

PROLOG 基本上就是這樣一個程式語言，

```prolog
father_child(john, mary).
father_child(john, sally).
 
sibling(X, Y)      :- parent_child(Z, X), parent_child(Z, Y).
 
parent_child(X, Y) :- father_child(X, Y).
parent_child(X, Y) :- mother_child(X, Y).
```

然後你就可以用邏輯語句問 Prolog，他會拿去推理得到答案後在回答體，像是這樣

```
?- sibling(sally, erica).
Yes
?- sibling(X, Y).
sibling(mary, sally).
```
