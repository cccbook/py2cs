# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html

import numpy as np
c = -np.array([0, 1])

A = np.array([[-1, 1], [3, 2], [2, 3]])
b_u = np.array([1, 12, 12])
b_l = np.full_like(b_u, -np.inf)

from scipy.optimize import LinearConstraint
constraints = LinearConstraint(A, b_l, b_u)

integrality = np.ones_like(c)

## 整數規劃
from scipy.optimize import milp
res = milp(c=c, constraints=constraints, integrality=integrality)
print('限制為整數：res=', res)

# 放寬限制，不限制為整數

res = milp(c=c, constraints=constraints) 
print('不限制為整數：res=', res)


