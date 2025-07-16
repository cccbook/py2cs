import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
    def __init__(self, Q, R, dt=0.02):
        self.Q = Q
        self.R = R
        self.dt = dt

        # 定義 A 和 B 矩陣 (需要根據你的問題設置)
        self.A = np.array([[0, 1, 0, 0],
                           [0, 0, -10, 0],
                           [0, 0, 0, 1],
                           [0, 0, 10, 0]])
        self.B = np.array([[0],
                           [1],
                           [0],
                           [-1]])
        
        # 計算 LQR 增益矩陣 K
        self._compute_lqr_gain()

    def _compute_lqr_gain(self):
        try:
            # 使用 scipy.linalg.solve_continuous_are 求解代數里卡提方程
            P = solve_continuous_are(self.A, self.B, self.Q, self.R)
            # 計算 LQR 增益矩陣 K
            self.K = np.linalg.inv(self.R) @ self.B.T @ P
        except Exception as e:
            print(f"Error solving Riccati equation: {e}")
            # 可以給出一個簡單的預設增益矩陣
            self.K = np.zeros_like(self.B.T)

    def reset(self):
        pass

    def compute(self, state):
        return -np.dot(self.K, state)


class LQRAgent:
    def __init__(self, env):
        self.env = env
        # 初始化 LQR 控制器，設置 Q 和 R 加權矩陣
        Q = np.diag([10, 1, 100, 10])  # 可以根據需要調整 Q 的值
        R = np.array([[1]])  # 控制輸入的加權矩陣
        self.controller = LQRController(Q, R)
        
    def reset(self, mode=None):
        self.controller.reset()

    def step(self, observation, reward, terminated):
        # 提取觀察值 (位置, 速度, 角度, 角速度)
        position, velocity, angle, angle_velocity = observation
        state = np.array([position, velocity, angle, angle_velocity])  # 系統狀態向量
        
        # 計算 LQR 控制輸出
        control = self.controller.compute(state)
        
        # 根據 LQR 控制信號決定動作
        action = 1 if control > 0 else 0  # 如果控制信號為正，則推動右邊，否則推動左邊
        return action

    def close(self):
        pass
