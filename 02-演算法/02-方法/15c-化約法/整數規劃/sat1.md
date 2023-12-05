# sat1.py

## 原理

此題為 ccc 自行新增的範例

布林變數 0<=x<=1

## run

```
$ python sat1.py
Welcome to the CBC MILP Solver
Version: Trunk
Build Date: Oct 28 2021

Starting solution of the Linear programming relaxation problem using Primal Simplex

Coin0506I Presolve 0 (-4) rows, 0 (-3) columns and 0 (-6) elements
Clp0000I Optimal - objective value 1
Coin0511I After Postsolve, objective 1, infeasibilities - dual 0 (0), primal 0 (0)
Clp0032I Optimal objective 1 - 0 iterations time 0.012, Presolve 0.01

Starting MIP optimization
Cgl0004I processed model has 0 rows, 0 columns (0 integer (0 of which binary)) and 0 elements
Cgl0015I Clique Strengthening extended 0 cliques, 0 were dominated
Cbc3007W No integer variables
Total time (CPU seconds):       0.02   (Wallclock seconds):       0.02

optimal solution cost 1.0 found
solution:
x : 1.0
y : 1.0
z : 0.0
```


