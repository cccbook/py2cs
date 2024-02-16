# SymPy

本資料夾的程式主要來自

* [用Python学数学之Sympy代数符号运算](https://zhuanlan.zhihu.com/p/60509430)

## 執行

```
mac020:sympy mac020$ python3 sqrt1.py
2.8284271247461903
2*sqrt(2)

mac020:sympy mac020$ python3 sym1.py
3*x + y**3

mac020:sympy mac020$ python3 factor1.py
x*(x**3 + y + 8)
x**4 + x*y + 8*x

mac020:sympy mac020$ python3 simplify1.py
-40*x**4*y**2

mac020:sympy mac020$ python3 solve1.py
[13500]

mac020:sympy mac020$ python3 solve2.py
:x: 6, y: 4

mac020:sympy mac020$ python3 solveRoot2.py
[(-b + sqrt(-4*a*c + b**2))/(2*a), -(b + sqrt(-4*a*c + b**2))/(2*a)]

mac020:sympy mac020$ python3 limit1.py
1

mac020:sympy mac020$ python3 derive1.py
exp(x)*sin(x) + exp(x)*cos(x)
2*exp(x)*cos(x)

mac020:sympy mac020$ python3 integrate1.py
exp(x)*sin(x)

mac020:sympy mac020$ python3 dintegrate1.py
sqrt(2)*sqrt(pi)/2
```


## 參考文獻

* https://docs.sympy.org/latest/tutorial/intro.html
* http://scipy-lectures.org/packages/sympy.html

