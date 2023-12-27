
```
$ python queen8.py
Welcome to the CBC MILP Solver
Version: Trunk
Build Date: Oct 28 2021

Starting solution of the Linear programming relaxation problem using Primal Simplex

Coin0506I Presolve 40 (-1) rows, 64 (0) columns and 247 (-1) elements
Clp1000I sum of infeasibilities 3.57996e-11 - average 8.9499e-13, 2 fixed columns
Coin0506I Presolve 40 (0) rows, 62 (-2) columns and 240 (-7) elements
Clp0029I End of values pass after 62 iterations
Clp0000I Optimal - objective value 0
Clp0000I Optimal - objective value 0
Coin0511I After Postsolve, objective 0, infeasibilities - dual 0 (0), primal 0 (0)
Clp0000I Optimal - objective value 0
Clp0000I Optimal - objective value 0
Clp0000I Optimal - objective value 0
Coin0511I After Postsolve, objective 0, infeasibilities - dual 0 (0), primal 0 (0)
Clp0032I Optimal objective 0 - 0 iterations time 0.112, Presolve 0.05, Idiot 0.05

Starting MIP optimization
Cgl0003I 0 fixed, 0 tightened bounds, 6 strengthened rows, 0 substitutions
Cgl0003I 0 fixed, 0 tightened bounds, 6 strengthened rows, 0 substitutions
Cgl0003I 0 fixed, 0 tightened bounds, 1 strengthened rows, 0 substitutions
Cgl0004I processed model has 40 rows, 64 columns (64 integer (64 of which binary)) and 260 elements
Coin3009W Conflict graph built in 0.007 seconds, density: 9.545%
Cgl0015I Clique Strengthening extended 0 cliques, 0 were dominated
Cbc0045I Nauty did not find any useful orbits in time 0.001
Cbc0038I Initial state - 26 integers unsatisfied sum - 5.88889
Cbc0038I Pass   1: suminf.    3.28571 (18) obj. 0 iterations 20
Cbc0038I Pass   2: suminf.    3.00000 (22) obj. 0 iterations 8
Cbc0038I Pass   3: suminf.    6.04598 (22) obj. 0 iterations 28
Cbc0038I Pass   4: suminf.    4.14286 (18) obj. 0 iterations 17
Cbc0038I Pass   5: suminf.    4.20000 (17) obj. 0 iterations 22
Cbc0038I Pass   6: suminf.    3.40000 (17) obj. 0 iterations 15
Cbc0038I Pass   7: suminf.    5.00000 (10) obj. 0 iterations 17
Cbc0038I Pass   8: suminf.    0.00000 (0) obj. 0 iterations 23
Cbc0038I Solution found of 0
Cbc0038I Before mini branch and bound, 16 integers at bound fixed and 0 continuous
Cbc0038I Mini branch and bound did not improve solution (0.23 seconds)
Cbc0038I After 0.23 seconds - Feasibility pump exiting with objective of 0 - took 0.09 seconds
Cbc0012I Integer solution of 0 found by feasibility pump after 0 iterations and 0 nodes (0.24 seconds)
Cbc0001I Search completed - best objective 0, took 0 iterations and 0 nodes (0.25 seconds)
Cbc0035I Maximum depth 0, 0 variables fixed on reduced cost
Total time (CPU seconds):       0.30   (Wallclock seconds):       0.30


. . O . . . . .
. . . . . . . O
. . . O . . . .
. . . . . . O .
O . . . . . . .
. . . . . O . .
. O . . . . . .
. . . . O . . .
```