# 解決問題 https://en.wikipedia.org/wiki/Integer_programming 
from mip import *
m = Model(sense=MAXIMIZE)
x = m.add_var(name='x', var_type=INTEGER, lb=0)
y = m.add_var(name='y', var_type=INTEGER, lb=0)
m += -x + y <= 1
m += 3*x + 2*y <= 12
m += 2*x + 3*y <= 12
m.objective = maximize(y)
m.write('integerProgramming1.lp')
m.max_gap = 0.05
status = m.optimize(max_seconds=300)
if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(m.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))
if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
    print('solution:')
    for v in m.vars:
       if abs(v.x) > 1e-6: # only printing non-zeros
          print('{} : {}'.format(v.name, v.x))