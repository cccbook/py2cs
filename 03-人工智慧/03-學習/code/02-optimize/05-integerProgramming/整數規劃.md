# IntegerProgramming-使用mip套件

* https://docs.python-mip.com/
    * https://docs.python-mip.com/en/latest/examples.html#the-0-1-knapsack-problem

## integerProgramming1.py

* 執行範例 -- https://en.wikipedia.org/wiki/Integer_programming#Example

```sh
user@DESKTOP-96FRN6B MINGW64 /d/ccc109/se/python/alg/17-npcomplete/integerProgramming (master)
$ python integerProgramming1.py 
Welcome to the CBC MILP Solver 
Version: devel
Build Date: Nov 15 2020        

Starting solution of the Linear programming relaxation problem using Primal Simplex

Coin0506I Presolve 3 (0) rows, 2 (0) columns and 6 (0) elements
Clp1000I sum of infeasibilities 6.95088e-12 - average 2.31696e-12, 0 fixed columns
Coin0506I Presolve 3 (0) rows, 2 (0) columns and 6 (0) elements
Clp0029I End of values pass after 2 iterations
Clp0000I Optimal - objective value 2.8
Clp0000I Optimal - objective value 2.8
Coin0511I After Postsolve, objective 2.8, infeasibilities - dual 0 (0), primal 0 (0)
Clp0000I Optimal - objective value 2.8
Clp0000I Optimal - objective value 2.8
Clp0006I 0  Obj 2.8
Clp0000I Optimal - objective value 2.8
Clp0032I Optimal objective 2.8 - 0 iterations time 1.942, Idiot 0.75

Starting MIP optimization
Cgl0004I processed model has 3 rows, 2 columns (2 integer (0 of which binary)) and 6 elements
Coin3009W Conflict graph built in 0.043 seconds, density: 0.000%
Cgl0015I Clique Strengthening extended 0 cliques, 0 were dominated
Cbc0012I Integer solution of -2 found by DiveCoefficient after 0 iterations and 0 nodes (0.69 seconds)
Cbc0001I Search completed - best objective -2, took 0 iterations and 0 nodes (0.72 seconds)
Cbc0035I Maximum depth 0, 0 variables fixed on reduced cost
Total time (CPU seconds):       1.08   (Wallclock seconds):       1.08

optimal solution cost 2.0 found
solution:
x : 1.0
y : 2.0
```

## sat1.py

```sh
user@DESKTOP-96FRN6B MINGW64 /d/ccc109/se/python/alg/17-npcomplete/integerProgramming (master)
$ python sat1.py 
Welcome to the CBC MILP Solver 
Version: devel
Build Date: Nov 15 2020 

Starting solution of the Linear programming relaxation problem using Primal Simplex

Coin0506I Presolve 0 (-4) rows, 0 (-3) columns and 0 (-6) elements
Clp0000I Optimal - objective value 1
Coin0511I After Postsolve, objective 1, infeasibilities - dual 0 (0), primal 0 (0)
Clp0032I Optimal objective 1 - 0 iterations time 0.622, Presolve 0.50

Starting MIP optimization
Cgl0004I processed model has 0 rows, 0 columns (0 integer (0 of which binary)) and 0 elements
Cgl0015I Clique Strengthening extended 0 cliques, 0 were dominated
Cbc3007W No integer variables - nothing to do
Total time (CPU seconds):       0.58   (Wallclock seconds):       0.58

optimal solution cost 1.0 found
solution:
x : 1.0
y : 1.0
z : 0.0
```

## knapsack1.py

```sh
$ python knapsack1.py 
Welcome to the CBC MILP Solver 
Version: devel 
Build Date: Nov 15 2020

Starting solution of the Linear programming relaxation problem using Primal Simplex

Coin0506I Presolve 1 (0) rows, 6 (0) columns and 6 (0) elements
Clp1000I sum of infeasibilities 0 - average 0, 6 fixed columns
Coin0506I Presolve 0 (-1) rows, 0 (-6) columns and 0 (-6) elements
Clp0000I Optimal - objective value -0
Clp0000I Optimal - objective value -0
Coin0511I After Postsolve, objective 0, infeasibilities - dual 0 (0), primal 0 (0)
Clp0000I Optimal - objective value 42.171429
Clp0000I Optimal - objective value 42.171429
Clp0000I Optimal - objective value 42.171429
Clp0032I Optimal objective 42.17142857 - 0 iterations time 0.922, Idiot 0.67

Starting MIP optimization
Cgl0004I processed model has 1 rows, 6 columns (6 integer (6 of which binary)) and 6 elements
Coin3009W Conflict graph built in 0.001 seconds, density: 14.103%
Cgl0015I Clique Strengthening extended 0 cliques, 0 were dominated
Cbc0038I Initial state - 1 integers unsatisfied sum - 0.457143
Cbc0038I Solution found of -28
Cbc0038I Before mini branch and bound, 5 integers at bound fixed and 0 continuous
Cbc0038I Full problem 1 rows 6 columns, reduced to 0 rows 0 columns
Cbc0038I Mini branch and bound did not improve solution (0.79 seconds)
Cbc0038I Round again with cutoff of -30.3171
Cbc0038I Reduced cost fixing fixed 1 variables on major pass 2
Cbc0038I Pass   1: suminf.    0.07474 (1) obj. -30.3171 iterations 1
Cbc0038I Pass   2: suminf.    0.45714 (1) obj. -42.1714 iterations 1
Cbc0038I Pass   3: suminf.    0.07474 (1) obj. -30.3171 iterations 1
Cbc0038I Pass   4: suminf.    0.45714 (1) obj. -42.1714 iterations 1
Cbc0038I Pass   5: suminf.    0.34461 (1) obj. -30.3171 iterations 2
Cbc0038I Solution found of -41
Cbc0038I Before mini branch and bound, 4 integers at bound fixed and 0 continuous
Cbc0038I Full problem 1 rows 6 columns, reduced to 1 rows 2 columns
Cbc0038I Mini branch and bound did not improve solution (2.29 seconds)
Cbc0038I Round again with cutoff of -42.0342
Cbc0038I Reduced cost fixing fixed 5 variables on major pass 3
Cbc0038I After 2.64 seconds - Feasibility pump exiting with objective of -41 - took 2.07 seconds
Cbc0012I Integer solution of -41 found by feasibility pump after 0 iterations and 0 nodes (2.80 seconds)
Cbc0038I Full problem 1 rows 6 columns, reduced to 0 rows 0 columns
Cbc0001I Search completed - best objective -41, took 0 iterations and 0 nodes (3.00 seconds)
Cbc0035I Maximum depth 0, 5 variables fixed on reduced cost
Total time (CPU seconds):       3.50   (Wallclock seconds):       3.50

selected items: [0, 3]
```

## pack2d1.py

```sh
$ python pack2d1.py
Welcome to the CBC MILP Solver 
Version: devel 
Build Date: Nov 15 2020 

Starting solution of the Linear programming relaxation problem using Primal Simplex

Coin0506I Presolve 8 (-8) rows, 31 (-7) columns and 61 (-15) elements
Clp1000I sum of infeasibilities 2.34542e-09 - average 2.93178e-10, 2 fixed columns
Coin0506I Presolve 8 (0) rows, 29 (-2) columns and 57 (-4) elements
Clp0029I End of values pass after 28 iterations
Clp0000I Optimal - objective value 11.822222
Clp0000I Optimal - objective value 11.822222
Coin0511I After Postsolve, objective 11.822222, infeasibilities - dual 0 (0), primal 0 (0)
Clp0000I Optimal - objective value 11.822222
Clp0006I 0  Obj 11.822222
Clp0000I Optimal - objective value 11.822222
Clp0000I Optimal - objective value 11.822222
Coin0511I After Postsolve, objective 11.822222, infeasibilities - dual 0 (0), primal 0 (0)
Clp0032I Optimal objective 11.82222222 - 0 iterations time 1.322, Presolve 0.28, Idiot 0.92

Starting MIP optimization
Cgl0003I 0 fixed, 0 tightened bounds, 1 strengthened rows, 0 substitutions
Cgl0004I processed model has 12 rows, 32 columns (32 integer (32 of which binary)) and 63 elements
Coin3009W Conflict graph built in 0.000 seconds, density: 7.308%
Cgl0015I Clique Strengthening extended 0 cliques, 0 were dominated
Cbc0038I Initial state - 0 integers unsatisfied sum - 4.44089e-16
Cbc0038I Solution found of 12
Cbc0038I Before mini branch and bound, 32 integers at bound fixed and 0 continuous
Cbc0038I Mini branch and bound did not improve solution (0.94 seconds)
Cbc0038I After 1.23 seconds - Feasibility pump exiting with objective of 12 - took 0.60 seconds
Cbc0012I Integer solution of 12 found by feasibility pump after 0 iterations and 0 nodes (1.39 seconds)
Cbc0001I Search completed - best objective 12, took 0 iterations and 0 nodes (1.57 seconds)
Cbc0035I Maximum depth 0, 0 variables fixed on reduced cost
Total time (CPU seconds):       1.95   (Wallclock seconds):       1.95

Items grouped with 0 : [2]
Items grouped with 1 : [5, 7]
Items grouped with 4 : [3, 6]
```

## queen1.py

```sh
user@DESKTOP-96FRN6B MINGW64 /d/ccc109/se/python/alg/17-npcomplete/integerProgramming (master)
$ python queen1.py
Welcome to the CBC MILP Solver 
Version: devel 
Build Date: Nov 15 2020 

Starting solution of the Linear programming relaxation problem using Primal Simplex

Coin0506I Presolve 232 (-1) rows, 1600 (0) columns and 6391 (-1) elements
Clp1000I sum of infeasibilities 9.47898e-08 - average 4.06823e-10, 231 fixed columns
Coin0506I Presolve 226 (-7) rows, 1369 (-231) columns and 5470 (-922) elements
Clp0000I Optimal - objective value 0
Clp0000I Optimal - objective value 0
Coin0511I After Postsolve, objective 0, infeasibilities - dual 0 (0), primal 0 (0)
Clp0000I Optimal - objective value 0
Clp0000I Optimal - objective value 0
Clp0006I 0  Obj 0
Clp0000I Optimal - objective value 0
Clp0032I Optimal objective 0 - 0 iterations time 0.962, Idiot 0.71

Starting MIP optimization
Cgl0003I 0 fixed, 0 tightened bounds, 4 strengthened rows, 0 substitutions
Cgl0003I 0 fixed, 0 tightened bounds, 2 strengthened rows, 0 substitutions
Cgl0003I 0 fixed, 0 tightened bounds, 1 strengthened rows, 0 substitutions
Cgl0004I processed model has 232 rows, 1600 columns (1600 integer (1600 of which binary)) and 6398 elements
Coin3009W Conflict graph built in 0.039 seconds, density: 2.052%
Cgl0015I Clique Strengthening extended 4 cliques, 4 were dominated
Cbc0045I No integer variables out of 1600 objects (1600 integer) have costs
Cbc0045I branch on satisfied N create fake objective Y random cost Y
Cbc0038I Initial state - 114 integers unsatisfied sum - 27.5672
Cbc0038I Pass   1: suminf.    9.18750 (30) obj. 0 iterations 508
Cbc0038I Pass   2: suminf.    0.00000 (0) obj. 0 iterations 379
Cbc0038I Solution found of 0
Cbc0038I Before mini branch and bound, 1467 integers at bound fixed and 0 continuous
Cbc0038I Mini branch and bound did not improve solution (2.32 seconds)
Cbc0038I After 2.36 seconds - Feasibility pump exiting with objective of 0 - took 0.74 seconds
Cbc0012I Integer solution of 0 found by feasibility pump after 0 iterations and 0 nodes (2.71 seconds)
Cbc0001I Search completed - best objective 0, took 0 iterations and 0 nodes (2.92 seconds)
Cbc0035I Maximum depth 0, 0 variables fixed on reduced cost
Total time (CPU seconds):       3.21   (Wallclock seconds):       3.21


. . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . .
. . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . .
. . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . 
. . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . .
. . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . 
. . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . .
. . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . .
. O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . .
. . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . .
. . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . .
. . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O .
. . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . 
```

## tsp1.py

```sh
user@DESKTOP-96FRN6B MINGW64 /d/ccc109/se/python/alg/17-npcomplete/integerProgramming (master)
$ python tsp1.py
Welcome to the CBC MILP Solver 
Version: devel
Build Date: Nov 15 2020 

Starting solution of the Linear programming relaxation problem using Primal Simplex

Coin0506I Presolve 184 (0) rows, 195 (-15) columns and 832 (0) elements
Clp1000I sum of infeasibilities 8.22201e-05 - average 4.46848e-07, 124 fixed columns
Coin0506I Presolve 182 (-2) rows, 69 (-126) columns and 474 (-358) elements
Clp0029I End of values pass after 69 iterations
Clp0014I Perturbing problem by 0.001% of 5.362616 - largest nonzero change 3.4125692e-05 ( 0.00073948616%) - largest zero change 1.8240846e-05
Clp0000I Optimal - objective value 399.2
Clp0000I Optimal - objective value 399.2
Coin0511I After Postsolve, objective 399.2, infeasibilities - dual 0 (0), primal 0 (0)
Clp0000I Optimal - objective value 399.2
Clp0006I 0  Obj 399.2
Clp0000I Optimal - objective value 399.2
Clp0000I Optimal - objective value 399.2
Coin0511I After Postsolve, objective 399.2, infeasibilities - dual 0 (0), primal 0 (0)
Clp0032I Optimal objective 399.2 - 0 iterations time 1.602, Presolve 0.43, Idiot 0.89

Starting MIP optimization
Cgl0004I processed model has 184 rows, 195 columns (182 integer (182 of which binary)) and 832 elements
Coin3009W Conflict graph built in 0.001 seconds, density: 3.103%
Cgl0015I Clique Strengthening extended 0 cliques, 0 were dominated
Cbc0038I Initial state - 26 integers unsatisfied sum - 2.13333
Cbc0038I Pass   1: suminf.    1.60000 (12) obj. 429.067 iterations 37
Cbc0038I Pass   2: suminf.    1.60000 (6) obj. 762.6 iterations 43
Cbc0038I Pass   3: suminf.    1.60000 (10) obj. 770.4 iterations 19
Cbc0038I Pass   4: suminf.    1.60000 (6) obj. 783.6 iterations 35
Cbc0038I Pass   5: suminf.    1.60000 (10) obj. 783 iterations 33
Cbc0038I Pass   6: suminf.    1.60000 (12) obj. 948.333 iterations 66
Cbc0038I Pass   7: suminf.    1.33333 (7) obj. 929 iterations 32
Cbc0038I Pass   8: suminf.    1.33333 (4) obj. 878.333 iterations 21
Cbc0038I Pass   9: suminf.    1.33333 (6) obj. 882.222 iterations 5
Cbc0038I Pass  10: suminf.    1.33333 (10) obj. 1001.27 iterations 45
Cbc0038I Pass  11: suminf.    1.33333 (4) obj. 1043 iterations 33
Cbc0038I Pass  12: suminf.    1.33333 (10) obj. 1042.33 iterations 40
Cbc0038I Pass  13: suminf.    1.33333 (10) obj. 1027.67 iterations 19
Cbc0038I Pass  14: suminf.    1.77778 (10) obj. 909.756 iterations 67
Cbc0038I Pass  15: suminf.    1.06667 (6) obj. 881.667 iterations 46
Cbc0038I Pass  16: suminf.    1.06667 (6) obj. 829.6 iterations 22
Cbc0038I Pass  17: suminf.    1.06667 (7) obj. 876.333 iterations 26
Cbc0038I Pass  18: suminf.    1.06667 (4) obj. 885.733 iterations 22
Cbc0038I Pass  19: suminf.    1.06667 (6) obj. 883.867 iterations 16
Cbc0038I Pass  20: suminf.    0.00000 (0) obj. 934 iterations 51
Cbc0038I Solution found of 934
Cbc0038I Relaxing continuous gives 934
Cbc0038I Before mini branch and bound, 109 integers at bound fixed and 0 continuous
Cbc0038I Full problem 184 rows 195 columns, reduced to 183 rows 85 columns - 4 fixed gives 173, 50 - still too large
Cbc0038I Full problem 184 rows 195 columns, reduced to 173 rows 50 columns - too large
Cbc0038I Mini branch and bound did not improve solution (4.05 seconds)
Cbc0038I Round again with cutoff of 879.62
Cbc0038I Pass  21: suminf.    1.60000 (12) obj. 429.067 iterations 0
Cbc0038I Pass  22: suminf.    1.60000 (6) obj. 762.6 iterations 52
Cbc0038I Pass  23: suminf.    1.60000 (10) obj. 770.4 iterations 31
Cbc0038I Pass  24: suminf.    1.60000 (6) obj. 783.6 iterations 30
Cbc0038I Pass  25: suminf.    1.60000 (10) obj. 783 iterations 33
Cbc0038I Pass  26: suminf.    4.69275 (14) obj. 879.62 iterations 57
Cbc0038I Pass  27: suminf.    1.60000 (15) obj. 879.62 iterations 69
Cbc0038I Pass  28: suminf.    2.00000 (8) obj. 735.5 iterations 59
Cbc0038I Pass  29: suminf.    1.60000 (16) obj. 688.167 iterations 23
Cbc0038I Pass  30: suminf.    1.90524 (9) obj. 879.62 iterations 85
Cbc0038I Pass  31: suminf.    1.60000 (11) obj. 832.5 iterations 39
Cbc0038I Pass  32: suminf.    1.60000 (7) obj. 839.4 iterations 38
Cbc0038I Pass  33: suminf.    1.60000 (9) obj. 870.983 iterations 19
Cbc0038I Pass  34: suminf.    2.00000 (10) obj. 850.467 iterations 48
Cbc0038I Pass  35: suminf.    1.60000 (12) obj. 834.267 iterations 23
Cbc0038I Pass  36: suminf.    1.60000 (6) obj. 852.467 iterations 26
Cbc0038I Pass  37: suminf.    1.60000 (4) obj. 832.8 iterations 31
Cbc0038I Pass  38: suminf.    1.60000 (7) obj. 807.4 iterations 21
Cbc0038I Pass  39: suminf.    2.80000 (6) obj. 833.2 iterations 19
Cbc0038I Pass  40: suminf.    1.60000 (12) obj. 818.667 iterations 40
Cbc0038I Pass  41: suminf.    1.60000 (9) obj. 870.983 iterations 23
Cbc0038I Pass  42: suminf.    2.00000 (10) obj. 850.467 iterations 56
Cbc0038I Pass  43: suminf.    1.60000 (12) obj. 834.267 iterations 22
Cbc0038I Pass  44: suminf.    1.60000 (6) obj. 852.467 iterations 30
Cbc0038I Pass  45: suminf.    1.60000 (4) obj. 832.8 iterations 36
Cbc0038I Pass  46: suminf.    1.60000 (7) obj. 807.4 iterations 48
Cbc0038I Pass  47: suminf.    2.80000 (6) obj. 833.2 iterations 20
Cbc0038I Pass  48: suminf.    1.60000 (12) obj. 818.667 iterations 38
Cbc0038I Pass  49: suminf.    1.60000 (9) obj. 870.983 iterations 35
Cbc0038I Pass  50: suminf.    2.00000 (10) obj. 850.467 iterations 43
Cbc0038I No solution found this major pass
Cbc0038I Before mini branch and bound, 125 integers at bound fixed and 1 continuous
Cbc0038I Full problem 184 rows 195 columns, reduced to 79 rows 67 columns
Cbc0038I Mini branch and bound did not improve solution (9.43 seconds)
Cbc0038I After 9.45 seconds - Feasibility pump exiting with objective of 934 - took 8.78 seconds
Cbc0012I Integer solution of 934 found by feasibility pump after 0 iterations and 0 nodes (9.64 seconds)
Cbc0038I Full problem 184 rows 195 columns, reduced to 174 rows 42 columns - 14 fixed gives 156, 13 - still too 
large
Cbc0031I 11 added rows had average density of 78.363636
Cbc0013I At root node, 11 cuts changed objective from 399.2 to 545.01701 in 100 passes
Cbc0014I Cut generator 0 (Probing) - 228 row cuts average 86.2 elements, 0 column cuts (0 active)  in 0.903 seconds - new frequency is 1
Cbc0014I Cut generator 1 (Gomory) - 1022 row cuts average 151.0 elements, 0 column cuts (0 active)  in 0.274 seconds - new frequency is 1
Cbc0014I Cut generator 2 (Knapsack) - 2 row cuts average 5.5 elements, 0 column cuts (0 active)  in 0.080 seconds - new frequency is -100
Cbc0014I Cut generator 3 (Clique) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.025 seconds 
- new frequency is -100
Cbc0014I Cut generator 4 (OddWheel) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.023 seconds - new frequency is -100
Cbc0014I Cut generator 5 (MixedIntegerRounding2) - 307 row cuts average 26.1 elements, 0 column cuts (0 active) 
 in 0.175 seconds - new frequency is 1
Cbc0014I Cut generator 6 (FlowCover) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.094 seconds - new frequency is -100
Cbc0014I Cut generator 7 (TwoMirCuts) - 327 row cuts average 30.1 elements, 0 column cuts (0 active)  in 0.095 seconds - new frequency is 1
Cbc0010I After 0 nodes, 1 on tree, 934 best solution, best possible 545.01701 (14.83 seconds)
Cbc0012I Integer solution of 552 found by DiveCoefficient after 3754 iterations and 1 nodes (15.38 seconds)
Cbc0004I Integer solution of 551 found after 4169 iterations and 10 nodes (15.81 seconds)
Cbc0012I Integer solution of 547 found by DiveCoefficient after 4372 iterations and 14 nodes (15.87 seconds)
Cbc0010I After 14 nodes, 7 on tree, 547 best solution, best possible 545.01701 (16.05 seconds)
Cbc0001I Search completed - best objective 547, took 5148 iterations and 22 nodes (16.37 seconds)
Cbc0032I Strong branching done 384 times (8876 iterations), fathomed 4 nodes and fixed 2 variables
Cbc0035I Maximum depth 9, 464 variables fixed on reduced cost
Total time (CPU seconds):       16.93   (Wallclock seconds):       16.93

route with total distance 547 found: Antwerp -> Bruges -> Ghent -> Grand-Place de Bruxelles -> Waterloo -> Mons 
-> Namur -> Dinant -> Remouchamps -> Montagne de Bueren -> C-Mine -> Hasselt -> Leuven -> Mechelen -> Antwerp
```