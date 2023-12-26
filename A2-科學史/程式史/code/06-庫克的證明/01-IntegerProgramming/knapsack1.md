
# knapsack1.py

## 原理

* https://docs.python-mip.com/en/latest/examples.html#the-0-1-knapsack-problem

## run


```
$ python knapsack1.py
Welcome to the CBC MILP Solver
Version: Trunk
Build Date: Oct 28 2021

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
Clp0032I Optimal objective 42.17142857 - 0 iterations time 0.192, Idiot 0.14

Starting MIP optimization
Cgl0004I processed model has 1 rows, 6 columns (6 integer (6 of which binary)) and 6 elements
Coin3009W Conflict graph built in 0.000 seconds, density: 14.103%
Cgl0015I Clique Strengthening extended 0 cliques, 0 were dominated
Cbc0045I Nauty did not find any useful orbits in time 0.007
Cbc0038I Initial state - 1 integers unsatisfied sum - 0.457143
Cbc0038I Solution found of -28
Cbc0038I Before mini branch and bound, 5 integers at bound fixed and 0 continuous
Cbc0038I Full problem 1 rows 6 columns, reduced to 0 rows 0 columns
Cbc0038I Mini branch and bound did not improve solution (0.27 seconds)
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
Cbc0038I Mini branch and bound did not improve solution (0.40 seconds)
Cbc0038I Round again with cutoff of -42.0342
Cbc0038I Reduced cost fixing fixed 5 variables on major pass 3
Cbc0038I After 0.42 seconds - Feasibility pump exiting with objective of -41 - took 0.20 seconds
Cbc0012I Integer solution of -41 found by feasibility pump after 0 iterations and 0 nodes (0.45 seconds)
Cbc0038I Full problem 1 rows 6 columns, reduced to 0 rows 0 columns
Cbc0001I Search completed - best objective -41, took 0 iterations and 0 nodes (0.54 seconds)
Cbc0035I Maximum depth 0, 5 variables fixed on reduced cost
Total time (CPU seconds):       0.57   (Wallclock seconds):       0.57

selected items: [0, 3]
```
