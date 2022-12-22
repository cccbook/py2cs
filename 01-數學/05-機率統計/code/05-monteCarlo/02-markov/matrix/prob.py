import numpy as np
# 狀態機率： P(a) = 0.2, P(b) = 0.8
# 轉移機率： P(x => y)
#    a   b
# a  0.7 0.3
# b  0.5 0.5

a, b = 0, 1

P = np.array([0.2,0.8])
T = np.array([[0.7, 0.3],[0.5, 0.5]])
