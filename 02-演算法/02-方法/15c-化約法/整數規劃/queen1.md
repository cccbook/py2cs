# queen1.py

## 原理

* https://docs.python-mip.com/en/latest/examples.html#n-queens

## run

```
$ python queen1.py
Welcome to the CBC MILP Solver
Version: Trunk
Build Date: Oct 28 2021

Starting solution of the Linear programming relaxation problem using Primal Simplex

Coin0506I Presolve 232 (-1) rows, 1600 (0) columns and 6391 (-1) elements
Clp1000I sum of infeasibilities 9.47898e-08 - average 4.06823e-10, 231 fixed columns
Coin0506I Presolve 226 (-7) rows, 1369 (-231) columns and 5470 (-922) elements
Clp0029I End of values pass after 1369 iterations
Clp0000I Optimal - objective value 0
Clp0000I Optimal - objective value 0
Coin0511I After Postsolve, objective 0, infeasibilities - dual 0 (0), primal 0 (0)
Clp0000I Optimal - objective value 0
Clp0000I Optimal - objective value 0
Clp0000I Optimal - objective value 0
Clp0032I Optimal objective 0 - 0 iterations time 0.222, Idiot 0.21

Starting MIP optimization
Cgl0003I 0 fixed, 0 tightened bounds, 4 strengthened rows, 0 substitutions
Cgl0003I 0 fixed, 0 tightened bounds, 2 strengthened rows, 0 substitutions
Cgl0003I 0 fixed, 0 tightened bounds, 1 strengthened rows, 0 substitutions
Cgl0004I processed model has 232 rows, 1600 columns (1600 integer (1600 of which binary)) and 6398 elements
Coin3009W Conflict graph built in 0.037 seconds, density: 2.052%
Cgl0015I Clique Strengthening extended 4 cliques, 4 were dominated
Cbc0045I Nauty: 975 orbits (780 useful covering 1560 variables), 1 generators, group size: 2 - sparse size 16476 - took 0.015 seconds
Cbc0045I No integer variables out of 1600 objects (1600 integer) have costs
Cbc0045I branch on satisfied N create fake objective Y random cost Y
Cbc0038I Initial state - 136 integers unsatisfied sum - 35.5838
Cbc0038I Pass   1: suminf.    9.33333 (39) obj. 0 iterations 389
Cbc0038I Pass   2: suminf.    5.75610 (46) obj. 0 iterations 185
Cbc0038I Pass   3: suminf.    5.70231 (97) obj. 0 iterations 81
Cbc0038I Pass   4: suminf.    4.20000 (14) obj. 0 iterations 346
Cbc0038I Pass   5: suminf.    2.00000 (4) obj. 0 iterations 70
Cbc0038I Pass   6: suminf.    2.00000 (4) obj. 0 iterations 57
Cbc0038I Pass   7: suminf.    2.00000 (4) obj. 0 iterations 10
Cbc0038I Pass   8: suminf.   32.11332 (124) obj. 0 iterations 446
Cbc0038I Pass   9: suminf.    9.42857 (28) obj. 0 iterations 476
Cbc0038I Pass  10: suminf.    3.73069 (132) obj. 0 iterations 225
Cbc0038I Pass  11: suminf.    3.62273 (132) obj. 0 iterations 62
Cbc0038I Pass  12: suminf.    5.58333 (25) obj. 0 iterations 529
Cbc0038I Pass  13: suminf.    4.50000 (11) obj. 0 iterations 266
Cbc0038I Pass  14: suminf.    0.00000 (0) obj. 0 iterations 185
Cbc0038I Solution found of 0
Cbc0038I Before mini branch and bound, 1161 integers at bound fixed and 0 continuous
Cbc0038I Mini branch and bound did not improve solution (1.91 seconds)
Cbc0038I After 1.91 seconds - Feasibility pump exiting with objective of 0 - took 1.14 seconds
Cbc0012I Integer solution of 0 found by feasibility pump after 0 iterations and 0 nodes
(1.92 seconds)
Cbc0001I Search completed - best objective 0, took 0 iterations and 0 nodes (1.94 seconds)
Cbc0035I Maximum depth 0, 0 variables fixed on reduced cost
Total time (CPU seconds):       1.99   (Wallclock seconds):       1.99


. . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O
. . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . .
. . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . .
. . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . .
. . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . .
. . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O .
. . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . .
O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . .
. . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . .
. . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . .
. . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . .
. . . . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . . . . . .
. . . . . . . . . . . . . . . O . . . . . . . . . . . . . . . . . . . . . . . .
```
