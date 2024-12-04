# pack2d1.py

## 原理

* https://docs.python-mip.com/en/latest/examples.html#two-dimensional-level-packing

## run


```
$ python pack2d1.py
Welcome to the CBC MILP Solver
Version: Trunk
Build Date: Oct 28 2021

Starting solution of the Linear programming relaxation problem using Primal Simplex

Coin0506I Presolve 8 (-8) rows, 31 (-7) columns and 61 (-15) elements
Clp1000I sum of infeasibilities 2.34542e-09 - average 2.93178e-10, 2 fixed columns
Coin0506I Presolve 8 (0) rows, 29 (-2) columns and 57 (-4) elements
Clp0029I End of values pass after 28 iterations
Clp0000I Optimal - objective value 11.822222
Clp0000I Optimal - objective value 11.822222
Coin0511I After Postsolve, objective 11.822222, infeasibilities - dual 0 (0), primal 0 (0)
Clp0000I Optimal - objective value 11.822222
Clp0000I Optimal - objective value 11.822222
Clp0000I Optimal - objective value 11.822222
Coin0511I After Postsolve, objective 11.822222, infeasibilities - dual 0 (0), primal 0 (0)
Clp0032I Optimal objective 11.82222222 - 0 iterations time 0.052, Presolve 0.01, Idiot 0.03

Starting MIP optimization
Cgl0003I 0 fixed, 0 tightened bounds, 1 strengthened rows, 0 substitutions
Cgl0004I processed model has 12 rows, 32 columns (32 integer (32 of which binary)) and 63 elements
Coin3009W Conflict graph built in 0.000 seconds, density: 7.308%
Cgl0015I Clique Strengthening extended 0 cliques, 0 were dominated
Cbc0045I Nauty: 69 orbits (8 useful covering 16 variables), 1 generators, group size: 2
- sparse size 406 - took 0.001 seconds
Cbc0038I Initial state - 0 integers unsatisfied sum - 7.77156e-16
Cbc0038I Solution found of 12
Cbc0038I Before mini branch and bound, 32 integers at bound fixed and 0 continuous
Cbc0038I Mini branch and bound did not improve solution (0.10 seconds)
Cbc0038I After 0.11 seconds - Feasibility pump exiting with objective of 12 - took 0.02
seconds
Cbc0012I Integer solution of 12 found by feasibility pump after 0 iterations and 0 nodes (0.12 seconds)
Cbc0001I Search completed - best objective 12, took 0 iterations and 0 nodes (0.13 seconds)
Cbc0035I Maximum depth 0, 0 variables fixed on reduced cost
Total time (CPU seconds):       0.15   (Wallclock seconds):       0.15

Items grouped with 0 : [2]
Items grouped with 1 : [5, 7]
Items grouped with 4 : [3, 6]
```