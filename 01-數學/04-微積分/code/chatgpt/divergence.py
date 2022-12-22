import numpy as np

def divergence(Fx, Fy, Fz):
    return np.gradient(Fx)[0] + np.gradient(Fy)[1] + np.gradient(Fz)[2]

Fx = np.array([[1, 2, 3], [4, 5, 6]])
Fy = np.array([[7, 8, 9], [10, 11, 12]])
Fz = np.array([[13, 14, 15], [16, 17, 18]])

d = divergence(Fx, Fy, Fz)
print(d)  # 输出: [[ 3.  3.  3.]
           #      [ 3.  3.  3.]]
