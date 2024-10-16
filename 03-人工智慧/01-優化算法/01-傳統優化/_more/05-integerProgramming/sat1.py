# 解決問題 SAT('(x or y) and (not x or not z) and (x) and (y)', ['x', 'y', 'z'])
from mip import *
m = Model(sense=MAXIMIZE)
x = m.add_var(name='x', var_type=BINARY)
y = m.add_var(name='y', var_type=BINARY)
z = m.add_var(name='z', var_type=BINARY)
m += x + y >= 1
m += (1-x) + (1-z) >= 1
m += x >=1
m += y >=1
m.objective = maximize(y)
m.write('sat1.lp')
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
         # if abs(v.x) > 1e-6: # only printing non-zeros
         print('{} : {}'.format(v.name, v.x))