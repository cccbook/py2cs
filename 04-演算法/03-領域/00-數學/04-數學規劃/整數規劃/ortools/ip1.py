# https://www.kindsonthegenius.com/data-science/an-mip-problem-with-python-with-constraints-define-with-arrays/

coefs = [[5, 7, 9, 2, 1],
        [18, 4, -9, 10, 12],
        [4, 7, 3, 8, 5],
        [5, 13, 15, 3, -7]]

maxs = [250,  285, 211, 315] 

unknowns = ['x1','x2','x3','x4','x5']

cost_fn = [7, 8, 2, 9, 6]

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

