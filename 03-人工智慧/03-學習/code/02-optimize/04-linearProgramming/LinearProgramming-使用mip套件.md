# LinearProgramming-使用mip套件

* [維基百科:線性規劃](https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E8%A7%84%E5%88%92)

使用套件: https://github.com/coin-or/python-mip

```
$ pip install mip
```

## LinearProgramming1

```sh
user@DESKTOP-96FRN6B MINGW64 /d/ccc109/se/python/alg/16-reduction/linearProgramming (master)
$ python linearProgramming1.py 
Welcome to the CBC MILP Solver 
Version: 2.9.0
Build Date: Feb 12 2015

command line - C:\Users\user\AppData\Local\Programs\Python\Python37\lib\site-packages\pulp\apis\..\solverdir\cbc\win\64\cbc.exe C:\Users\user\AppData\Local\Temp\dbf20d787e804b48973232e1fa39f796-pulp.mps ratio None allow None threads None presolve on strong None gomory on knapsack on probing on branch printingOptions all solution C:\Users\user\AppData\Local\Temp\dbf20d787e804b48973232e1fa39f796-pulp.sol (default strategy 1)
At line 2 NAME          MODEL
At line 3 ROWS
At line 8 COLUMNS
At line 18 RHS
At line 22 BOUNDS
At line 26 ENDATA
Problem MODEL has 3 rows, 3 columns and 6 elements
Coin0008I MODEL read with 0 errors
String of None is illegal for double parameter ratioGap value remains 0
String of None is illegal for double parameter allowableGap value remains 0
String of None is illegal for integer parameter threads value remains 0
String of None is illegal for integer parameter strongBranching value remains 5
Option for gomoryCuts changed from ifmove to on
Option for knapsackCuts changed from ifmove to on
Presolve 1 (-2) rows, 2 (-1) columns and 2 (-4) elements
0  Obj 51.9 Primal inf 2.099999 (1)
1  Obj 54
Optimal - objective value 54
After Postsolve, objective 54, infeasibilities - dual 0 (0), primal 0 (0)
Optimal objective 54 - 1 iterations time 0.932, Presolve 0.20
Option for printingOptions changed from normal to all
Total time (CPU seconds):       2.23   (Wallclock seconds):       2.23

Status: Optimal
x = 4.0
y = -1.0
z = 6.0
objective= 54.0
```

## AmericanSteel.py

```sh
user@DESKTOP-96FRN6B MINGW64 /d/ccc109/se/python/alg/16-reduction/linearProgramming (master)
$ python americanSteel.py 
C:\Users\user\AppData\Local\Programs\Python\Python37\lib\site-packages\pulp\pulp.py:1198: UserWarning: Spaces are not permitted in the name. Converted to '_'
  warnings.warn("Spaces are not permitted in the name. Converted to '_'")
Welcome to the CBC MILP Solver 
Version: 2.9.0
Build Date: Feb 12 2015        

command line - C:\Users\user\AppData\Local\Programs\Python\Python37\lib\site-packages\pulp\apis\..\solverdir\cbc\win\64\cbc.exe C:\Users\user\AppData\Local\Temp\09b7b7942393488bb2234f7c2a444da7-pulp.mps ratio None allow None threads None presolve on strong None gomory on knapsack on probing on branch printingOptions all solution C:\Users\user\AppData\Local\Temp\09b7b7942393488bb2234f7c2a444da7-pulp.sol (default strategy 1)
At line 2 NAME          MODEL
At line 3 ROWS
At line 14 COLUMNS
At line 85 RHS
At line 95 BOUNDS
At line 113 ENDATA
Problem MODEL has 9 rows, 14 columns and 28 elements
Coin0008I MODEL read with 0 errors
String of None is illegal for double parameter ratioGap value remains 0
String of None is illegal for double parameter allowableGap value remains 0
String of None is illegal for integer parameter threads value remains 0
String of None is illegal for integer parameter strongBranching value remains 5
Option for gomoryCuts changed from ifmove to on
Option for knapsackCuts changed from ifmove to on
Continuous objective value is 15005 - 0.09 seconds
Cgl0003I 0 fixed, 1 tightened bounds, 0 strengthened rows, 0 substitutions
Cgl0004I processed model has 4 rows, 6 columns (6 integer (0 of which binary)) and 10 elements
Cutoff increment increased from 1e-005 to 0.024975
Cbc0012I Integer solution of 15005 found by DiveCoefficient after 0 iterations and 0 nodes (0.42 seconds)
Cbc0001I Search completed - best objective 15005, took 0 iterations and 0 nodes (0.51 seconds)
Cbc0035I Maximum depth 0, 0 variables fixed on reduced cost
Cuts at root node changed objective from 15005 to 15005
Probing was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds) 
Gomory was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)  
Knapsack was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
Clique was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
MixedIntegerRounding2 was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
FlowCover was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
TwoMirCuts was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)

Result - Optimal solution found

Objective value:                15005.00000000
Enumerated nodes:               0
Total iterations:               0
Time (CPU seconds):             1.35
Time (Wallclock seconds):       1.35

Option for printingOptions changed from normal to all
Total time (CPU seconds):       4.81   (Wallclock seconds):       4.81

Status: Optimal
Route_('Chicago',_'Gary') = 4000.0
Route_('Chicago',_'Tempe') = 2000.0
Route_('Cincinatti',_'Albany') = 2000.0
Route_('Cincinatti',_'Houston') = 3000.0
Route_('Kansas_City',_'Houston') = 4000.0
Route_('Kansas_City',_'Tempe') = 2000.0
Route_('Pittsburgh',_'Chicago') = 3000.0
Route_('Pittsburgh',_'Cincinatti') = 2000.0
Route_('Pittsburgh',_'Gary') = 2000.0
Route_('Pittsburgh',_'Kansas_City') = 3000.0
Route_('Youngstown',_'Albany') = 1000.0
Route_('Youngstown',_'Chicago') = 3000.0
Route_('Youngstown',_'Cincinatti') = 3000.0
Route_('Youngstown',_'Kansas_City') = 3000.0
Total Cost of Transportation =  15005.0
```