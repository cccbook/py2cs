import numpy as np

# 定義貝氏網路的概率分布
P_X = np.array([0.1, 0.9])
P_Y_given_X = np.array([[0.1, 0.8],
                        [0.9, 0.2]])
P_Z_given_Y = np.array([[0.7, 0.3],
                        [0.4, 0.6]])

# 定義輸入
x = 0
y = 1
z = 1
# 計算輸出
p_x = P_X[x]
p_y_given_x = P_Y_given_X[x][y]
p_z_given_y = P_Z_given_Y[y][z]
p_xyz = p_x * p_y_given_x * p_z_given_y

# 輸出結果
print(p_xyz)
