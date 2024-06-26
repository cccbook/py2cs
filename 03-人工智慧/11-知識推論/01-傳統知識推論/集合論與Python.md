# 集合論與Python

## 集合論公理系統

```math
\begin{align*}
& (1)\;  S \in F； \\
& (2)\;  if\; A \in F,\;then\; \bar{A} \in F;  \\
& (3)\;  if\; A \in F, B \in F,\;then\; A \cup B \in F;  \\
\end{align*}
```

## 凡氏圖 (Venn Diagram)

兩個集合的文氏圖

![](http://upload.wikimedia.org/wikipedia/commons/b/b7/Intersection_of_two_sets_A_and_B.svg)

三個集合的文氏圖

![](http://upload.wikimedia.org/wikipedia/commons/7/7a/Venn_diagram_cmyk.svg)

## Python 的集合運算示範

```py
from sets import Set
A = Set([1, 3, 5, 7, 9])
B = Set([2, 3, 6, 8, 9])
print "A=", A
print "B=", B
print "A|B=", A|B
print "A&B=", A&B
print "A-B=", A-B
print "B-A=", B-A
```

執行結果

```
A= Set([1, 3, 9, 5, 7])
B= Set([8, 9, 2, 3, 6])
A|B= Set([1, 2, 3, 5, 6, 7, 8, 9])
A&B= Set([9, 3])
A-B= Set([1, 5, 7])
B-A= Set([8, 2, 6])
```

# 測度論

參考文獻

* [測度](https://zh.wikipedia.org/wiki/%E6%B5%8B%E5%BA%A6)
* [機率](https://zh.wikipedia.org/wiki/%E6%A6%82%E7%8E%87%E8%AE%BA)
* [機率空間](https://zh.wikipedia.org/wiki/%E6%A6%82%E7%8E%87%E7%A9%BA%E9%96%93)
* [Probability axioms(機率公理)](https://en.wikipedia.org/wiki/Probability_axioms)
* [機率論](https://zh.wikipedia.org/wiki/%E6%A6%82%E7%8E%87%E8%AE%BA)
* [σ-代数](https://zh.wikipedia.org/wiki/%CE%A3-%E4%BB%A3%E6%95%B0)
* [Real analysis(實變函數論)](https://en.wikipedia.org/wiki/Real_analysis)
* [勒貝格積分 (Lebesgue integration)](https://zh.wikipedia.org/wiki/%E5%8B%92%E8%B2%9D%E6%A0%BC%E7%A9%8D%E5%88%86)
* [積分](https://zh.wikipedia.org/wiki/%E7%A7%AF%E5%88%86)
* [Borel measure（博雷爾測度）](https://en.wikipedia.org/wiki/Borel_measure)
* [Measurable function](https://en.wikipedia.org/wiki/Measurable_function) [可測函數](https://zh.wikipedia.org/wiki/%E5%8F%AF%E6%B5%8B%E5%87%BD%E6%95%B0)
* [指示函數(Indicator function)](https://zh.wikipedia.org/wiki/%E6%8C%87%E7%A4%BA%E5%87%BD%E6%95%B0) : 不可測
* [Lebesgue measure](https://en.wikipedia.org/wiki/Lebesgue_measure)
* [概率论和实变函数（测度论）有什么联系？](https://www.zhihu.com/question/29800166)

## 參考文獻

* 課堂錄影：集合論 -- http://www.youtube.com/embed/GXCMeGd09A4
* 2001年初等集合論
   * http://www.sftw.umac.mo/~fstitl/2002-geometry/2003-set-notation-index.html
* 集合論與數學教育：王九逵
   * http://episte.math.ntu.edu.tw/articles/sm/sm_31_03_1/index.html