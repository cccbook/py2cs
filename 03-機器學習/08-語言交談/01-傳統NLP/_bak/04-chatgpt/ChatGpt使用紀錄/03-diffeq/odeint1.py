import numpy as np
# 導入 odeint 函數
from scipy.integrate import odeint

# 定義一個微分方程，y 為變量，t 為時間
def diff_eq(y, t):
    return y

# 設定初始值和求解時間
y0 = 1
t = np.linspace(0, 10, 100)

# 求解微分方程
y = odeint(diff_eq, y0, t)

# 輸出答案
print(y)
