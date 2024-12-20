import numpy as np
from scipy import linalg

class LQRAgent:
    def __init__(self, env):
        # 系統參數
        self.g = 9.81     # 重力加速度
        self.mc = 1.0     # 小車質量
        self.mp = 0.1     # 擺桿質量
        self.l = 0.5      # 擺桿半長
        self.dt = 0.02    # 時間步長
        
        # LQR 代價矩陣
        self.Q = np.diag([1.0, 1.0, 10.0, 10.0])  # 狀態代價
        self.R = np.array([[0.1]])                 # 控制代價
        
        # 計算線性化系統矩陣
        self.A, self.B = self._linearize_system()
        
        # 求解 LQR 增益矩陣
        self.K = self._solve_lqr()
        
    def _linearize_system(self):
        """
        在平衡點處線性化系統
        x = [position, velocity, angle, angular_velocity]
        """
        # 系統總質量
        mt = self.mc + self.mp
        
        # 在平衡點(x=0, theta=0)處的線性化矩陣
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, -self.mp*self.g/mt, 0],
            [0, 0, 0, 1],
            [0, 0, (mt*self.g)/(self.l*mt), 0]
        ])
        
        B = np.array([
            [0],
            [1/mt],
            [0],
            [-1/(self.l*mt)]
        ])
        
        # 離散化系統
        Ad = np.eye(4) + self.dt * A
        Bd = self.dt * B
        
        return Ad, Bd
        
    def _solve_lqr(self):
        """
        求解離散時間 LQR 控制器
        使用 scipy 的 dare (discrete algebraic Riccati equation) 求解器
        """
        # 求解離散時間 Riccati 方程
        P = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        
        # 計算最優增益矩陣
        K = -np.linalg.inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        
        return K
        
    def reset(self, mode=None):
        return
        
    def step(self, observation, reward=0, terminated=False):
        """執行一步 LQR 控制"""
        # 將觀測值轉換為狀態向量
        state = np.array(observation).reshape(-1, 1)
        
        # 計算控制輸入
        u = self.K @ state
        
        # 轉換為離散動作
        action = 1 if u[0] > 0 else 0
        
        return action
        
    def close(self):
        pass