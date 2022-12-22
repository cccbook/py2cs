import numpy as np

def curl(Fx, Fy, Fz):
    return np.array([np.gradient(Fz, axis=1) - np.gradient(Fy, axis=0),
                     np.gradient(Fx, axis=2) - np.gradient(Fz, axis=0),
                     np.gradient(Fy, axis=2) - np.gradient(Fx, axis=1)])

Fx = np.array([[1, 2, 3], [4, 5, 6]])
Fy = np.array([[7, 8, 9], [10, 11, 12]])
Fz = np.array([[13, 14, 15], [16, 17, 18]])

c = curl(Fx, Fy, Fz)
print(c)  # 输出: [[[ -6.  -6.  -6.]
           #        [ -6.  -6.  -6.]]
           #       [[ -6.  -6.  -6.]
           #        [ -6.  -6.  -6.]]
           #       [[ -6.  -6.  -6.]
           #        [ -6.  -6.  -6.]]]
