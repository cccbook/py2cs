# integerProgramming1.py

```
$ python integerProgramming1.py
Welcome to the CBC MILP Solver
Version: Trunk
Build Date: Oct 28 2021

Starting solution of the Linear programming relaxation problem using Primal Simplex

Coin0506I Presolve 3 (0) rows, 2 (0) columns and 6 (0) elements
Clp1000I sum of infeasibilities 6.95088e-12 - average 2.31696e-12, 0 fixed columns
Coin0506I Presolve 3 (0) rows, 2 (0) columns and 6 (0) elements
Clp0029I End of values pass after 2 iterations
Clp0000I Optimal - objective value 2.8
Clp0000I Optimal - objective value 2.8
Clp0000I Optimal - objective value 2.8
Clp0032I Optimal objective 2.8 - 0 iterations time 0.032, Idiot 0.01

Starting MIP optimization
Cgl0004I processed model has 3 rows, 2 columns (2 integer (0 of which binary)) and 6 elements
Coin3009W Conflict graph built in 0.000 seconds, density: 0.000%
Cgl0015I Clique Strengthening extended 0 cliques, 0 were dominated
Cbc0045I Nauty did not find any useful orbits in time 0
Cbc0012I Integer solution of -2 found by DiveCoefficient after 0 iterations and 0 nodes
(0.07 seconds)
Cbc0001I Search completed - best objective -2, took 0 iterations and 0 nodes (0.07 seconds)
Cbc0035I Maximum depth 0, 0 variables fixed on reduced cost
Total time (CPU seconds):       0.09   (Wallclock seconds):       0.10

optimal solution cost 2.0 found
solution:
x : 1.0
y : 2.0
```