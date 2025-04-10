# https://www.kindsonthegenius.com/data-science/an-mip-problem-with-python-with-constraints-define-with-arrays/
from ipData import *

# IMPORT THE SOLVER
from ortools.linear_solver import pywraplp
solver = pywraplp.Solver.CreateSolver('SCIP')

# DECLARE THE VARIABLES
variables = [[]] * len(unknowns)
for i in range(0, len(unknowns)):
    variables[i] = solver.IntVar(0, solver.infinity(), unknowns[i])

# CREATE THE CONSTRAINT 0 <= f(x,y) <= 17.5
constraints = [0] * len(coefs)

for i in range(0, len(coefs)):
    constraints[i] = solver.Constraint(0, maxs[i])
    for j in range(0, len(coefs[i])):
        constraints[i].SetCoefficient(variables[j], coefs[i][j])

# DEFINE THE OBJECTIVE FUNCTION 7x1 + 8x2 + 2x3 + 9x4 + 6x5
obj = solver.Objective()
for i in range(0, len(cost_fn)):
    obj.SetCoefficient(variables[i], cost_fn[i])

obj.SetMaximization() # set the problem goal as maximization

# CALL THE SOLVER AND SHOW THE RESULT
status = solver.Solve()
print('Objective value = ', obj.Value())

# PRINT THE RESULTS
for i in range(0, len(unknowns)):
    print('%s = %f' %(unknowns[i], variables[i].solution_value()))

