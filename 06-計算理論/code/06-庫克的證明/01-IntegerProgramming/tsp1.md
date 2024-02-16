# tsp1.py

## 原理

* https://docs.python-mip.com/en/latest/examples.html#the-traveling-salesman-problem

## run

```
$ python tsp1.py
Welcome to the CBC MILP Solver
Version: Trunk
Build Date: Oct 28 2021

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
Clp0000I Optimal - objective value 399.2
Clp0000I Optimal - objective value 399.2
Coin0511I After Postsolve, objective 399.2, infeasibilities - dual 0 (0), primal 0 (0)
Clp0032I Optimal objective 399.2 - 0 iterations time 0.182, Presolve 0.01, Idiot 0.16

Starting MIP optimization
Cgl0004I processed model has 184 rows, 195 columns (182 integer (182 of which binary)) and 832 elements
Coin3009W Conflict graph built in 0.001 seconds, density: 3.103%
Cgl0015I Clique Strengthening extended 0 cliques, 0 were dominated
Cbc0045I Nauty did not find any useful orbits in time 0.007
Cbc0038I Initial state - 26 integers unsatisfied sum - 2.13333
Cbc0038I Pass   1: suminf.    1.60000 (12) obj. 429.067 iterations 35
Cbc0038I Pass   2: suminf.    1.60000 (6) obj. 762.6 iterations 32
Cbc0038I Pass   3: suminf.    1.60000 (10) obj. 770.4 iterations 20
Cbc0038I Pass   4: suminf.    1.60000 (6) obj. 783.6 iterations 22
Cbc0038I Pass   5: suminf.    1.60000 (10) obj. 783 iterations 25
Cbc0038I Pass   6: suminf.    1.60000 (12) obj. 948.333 iterations 68
Cbc0038I Pass   7: suminf.    1.33333 (7) obj. 929 iterations 31
Cbc0038I Pass   8: suminf.    1.33333 (4) obj. 878.333 iterations 44
Cbc0038I Pass   9: suminf.    1.33333 (6) obj. 882.222 iterations 12
Cbc0038I Pass  10: suminf.    1.33333 (10) obj. 1001.27 iterations 28
Cbc0038I Pass  11: suminf.    1.33333 (4) obj. 1043 iterations 33
Cbc0038I Pass  12: suminf.    1.33333 (10) obj. 1042.33 iterations 28
Cbc0038I Pass  13: suminf.    1.33333 (10) obj. 1027.67 iterations 25
Cbc0038I Pass  14: suminf.    1.77778 (10) obj. 909.756 iterations 62
Cbc0038I Pass  15: suminf.    1.06667 (6) obj. 881.667 iterations 31
Cbc0038I Pass  16: suminf.    1.06667 (6) obj. 829.6 iterations 21
Cbc0038I Pass  17: suminf.    1.06667 (7) obj. 876.333 iterations 31
Cbc0038I Pass  18: suminf.    1.06667 (4) obj. 885.733 iterations 23
Cbc0038I Pass  19: suminf.    1.06667 (6) obj. 883.867 iterations 18
Cbc0038I Pass  20: suminf.    0.00000 (0) obj. 934 iterations 55
Cbc0038I Solution found of 934
Cbc0038I Relaxing continuous gives 934
Cbc0038I Before mini branch and bound, 109 integers at bound fixed and 0 continuous
Cbc0038I Full problem 184 rows 195 columns, reduced to 183 rows 85 columns - 4 fixed gives 173, 50 - still too large
Cbc0038I Full problem 184 rows 195 columns, reduced to 173 rows 50 columns - too large
Cbc0038I Mini branch and bound did not improve solution (0.41 seconds)
Cbc0038I Round again with cutoff of 879.62
Cbc0038I Pass  21: suminf.    1.60000 (12) obj. 429.067 iterations 0
Cbc0038I Pass  22: suminf.    1.60000 (6) obj. 762.6 iterations 47
Cbc0038I Pass  23: suminf.    1.60000 (10) obj. 770.4 iterations 35
Cbc0038I Pass  24: suminf.    1.60000 (6) obj. 783.6 iterations 47
Cbc0038I Pass  25: suminf.    1.60000 (10) obj. 783 iterations 30
Cbc0038I Pass  26: suminf.    4.69275 (14) obj. 879.62 iterations 55
Cbc0038I Pass  27: suminf.    1.60000 (15) obj. 879.62 iterations 44
Cbc0038I Pass  28: suminf.    1.60000 (7) obj. 846.467 iterations 55
Cbc0038I Pass  29: suminf.    1.60000 (9) obj. 870.983 iterations 23
Cbc0038I Pass  30: suminf.    2.00000 (10) obj. 850.467 iterations 22
Cbc0038I Pass  31: suminf.    1.60000 (12) obj. 834.267 iterations 27
Cbc0038I Pass  32: suminf.    1.60000 (6) obj. 852.467 iterations 55
Cbc0038I Pass  33: suminf.    1.60000 (4) obj. 832.8 iterations 28
Cbc0038I Pass  34: suminf.    1.60000 (7) obj. 807.4 iterations 32
Cbc0038I Pass  35: suminf.    2.80000 (6) obj. 833.2 iterations 20
Cbc0038I Pass  36: suminf.    1.60000 (12) obj. 818.667 iterations 43
Cbc0038I Pass  37: suminf.    1.60000 (9) obj. 870.983 iterations 61
Cbc0038I Pass  38: suminf.    2.00000 (10) obj. 850.467 iterations 28
Cbc0038I Pass  39: suminf.    1.60000 (12) obj. 834.267 iterations 23
Cbc0038I Pass  40: suminf.    1.60000 (6) obj. 852.467 iterations 55
Cbc0038I Pass  41: suminf.    1.60000 (4) obj. 832.8 iterations 30
Cbc0038I Pass  42: suminf.    1.60000 (7) obj. 807.4 iterations 26
Cbc0038I Pass  43: suminf.    2.80000 (6) obj. 833.2 iterations 24
Cbc0038I Pass  44: suminf.    1.60000 (12) obj. 818.667 iterations 34
Cbc0038I Pass  45: suminf.    1.60000 (9) obj. 870.983 iterations 53
Cbc0038I Pass  46: suminf.    2.00000 (10) obj. 850.467 iterations 62
Cbc0038I Pass  47: suminf.    1.60000 (12) obj. 834.267 iterations 14
Cbc0038I Pass  48: suminf.    1.60000 (6) obj. 852.467 iterations 56
Cbc0038I Pass  49: suminf.    1.60000 (4) obj. 832.8 iterations 30
Cbc0038I Pass  50: suminf.    1.60000 (7) obj. 807.4 iterations 26
Cbc0038I No solution found this major pass
Cbc0038I Before mini branch and bound, 128 integers at bound fixed and 1 continuous
Cbc0038I Full problem 184 rows 195 columns, reduced to 74 rows 62 columns
Cbc0038I Mini branch and bound did not improve solution (1.21 seconds)
Cbc0038I After 1.21 seconds - Feasibility pump exiting with objective of 934 - took 1.12 seconds
Cbc0012I Integer solution of 934 found by feasibility pump after 0 iterations and 0 nodes (1.22 seconds)
Cbc0038I Full problem 184 rows 195 columns, reduced to 174 rows 42 columns - 14 fixed gives 156, 13 - still too large
Cbc0031I 10 added rows had average density of 90.5
Cbc0013I At root node, 10 cuts changed objective from 399.2 to 542 in 43 passes
Cbc0014I Cut generator 0 (Probing) - 20 row cuts average 12.9 elements, 0 column cuts (0 active)  in 0.445 seconds - new frequency is -100
Cbc0014I Cut generator 1 (Gomory) - 251 row cuts average 125.2 elements, 0 column cuts (0 active)  in 0.086 seconds - new frequency is 1
Cbc0014I Cut generator 2 (Knapsack) - 2 row cuts average 5.5 elements, 0 column cuts (0
active)  in 0.029 seconds - new frequency is -100
Cbc0014I Cut generator 3 (Clique) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.007 seconds - new frequency is -100
Cbc0014I Cut generator 4 (OddWheel) - 0 row cuts average 0.0 elements, 0 column cuts (0
active)  in 0.007 seconds - new frequency is -100
Cbc0014I Cut generator 5 (MixedIntegerRounding2) - 79 row cuts average 7.2 elements, 0 column cuts (0 active)  in 0.049 seconds - new frequency is 1
Cbc0014I Cut generator 6 (FlowCover) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.035 seconds - new frequency is -100
Cbc0014I Cut generator 7 (TwoMirCuts) - 174 row cuts average 46.3 elements, 0 column cuts (0 active)  in 0.054 seconds - new frequency is 1
Cbc0010I After 0 nodes, 1 on tree, 934 best solution, best possible 542 (2.46 seconds)
Cbc0016I Integer solution of 551 found by strong branching after 1093 iterations and 6 nodes (2.65 seconds)
Cbc0016I Integer solution of 547 found by strong branching after 1529 iterations and 17
nodes (3.00 seconds)
Cbc0010I After 24 nodes, 2 on tree, 547 best solution, best possible 542.66818 (3.17 seconds)
Cbc0001I Search completed - best objective 547, took 2075 iterations and 30 nodes (3.25
seconds)
Cbc0032I Strong branching done 516 times (8468 iterations), fathomed 4 nodes and fixed 4 variables
Cbc0035I Maximum depth 5, 147 variables fixed on reduced cost
Total time (CPU seconds):       3.33   (Wallclock seconds):       3.33

route with total distance 547 found: Antwerp -> Mechelen -> Leuven -> Hasselt -> C-Mine
-> Montagne de Bueren -> Remouchamps -> Dinant -> Namur -> Mons -> Waterloo -> Grand-Place de Bruxelles -> Ghent -> Bruges -> Antwerp
```
