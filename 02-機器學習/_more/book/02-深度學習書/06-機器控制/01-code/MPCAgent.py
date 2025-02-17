import numpy as np
from scipy.optimize import minimize

class MPCAgent:
    def __init__(self, env):
        # MPC parameters
        self.horizon = 10  # 預測視界
        self.dt = 0.02    # 時間步長
        self.g = 9.81     # 重力加速度
        self.mc = 1.0     # 小車質量
        self.mp = 0.1     # 擺桿質量
        self.l = 0.5      # 擺桿半長
        
        # 代價函數權重
        self.Q = np.diag([1.0, 1.0, 10.0, 10.0])  # 狀態代價
        self.R = 0.1                               # 控制代價
        self.last_u = 0  # 保存上一步的控制輸入
        
    def reset(self, mode=None):
        self.last_u = 0
        return
        
    def dynamics(self, state, u):
        """CartPole 系統動力學模型"""
        x, v, theta, omega = state
        
        # 簡化的動力學方程
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # 計算加速度
        num = (self.g * sin_theta + cos_theta * 
               (-u - self.mp * self.l * omega**2 * sin_theta) / (self.mc + self.mp))
        den = self.l * (4/3 - self.mp * cos_theta**2 / (self.mc + self.mp))
        theta_acc = num / den
        
        x_acc = (u + self.mp * self.l * (omega**2 * sin_theta - theta_acc * cos_theta)) / (self.mc + self.mp)
        
        # 更新狀態
        next_x = x + v * self.dt
        next_v = v + x_acc * self.dt
        next_theta = theta + omega * self.dt
        next_omega = omega + theta_acc * self.dt
        
        return np.array([next_x, next_v, next_theta, next_omega])
        
    def cost_function(self, u_sequence, current_state):
        """計算預測視界內的總代價"""
        total_cost = 0
        state = current_state.copy()  # 確保不修改原始狀態
        
        for i in range(len(u_sequence)):
            # 狀態代價
            state_cost = state @ self.Q @ state
            # 控制代價
            control_cost = self.R * u_sequence[i]**2
            # 控制變化代價
            delta_u_cost = 0.1 * (u_sequence[i] - self.last_u)**2
            total_cost += state_cost + control_cost + delta_u_cost
            
            # 預測下一狀態
            state = self.dynamics(state, u_sequence[i])
            self.last_u = u_sequence[i]
            
        return total_cost
        
    def step(self, observation, reward=0, terminated=False):
        """執行一步MPC控制"""
        try:
            # 初始化控制序列
            u_init = np.zeros(self.horizon)
            
            # 求解最優控制序列
            bounds = [(-1, 1)] * self.horizon  # 將控制範圍限制在 [-1, 1]
            result = minimize(
                fun=self.cost_function,
                x0=u_init,
                args=(observation,),
                method='SLSQP',
                bounds=bounds
            )
            
            # 取最優控制序列的第一個動作
            optimal_u = result.x[0]
            self.last_u = optimal_u
            
            # 將連續控制轉換為離散動作 (0 或 1)
            action = 1 if optimal_u > 0 else 0
            
            return action
            
        except Exception as e:
            print(f"MPC optimization failed: {e}")
            # 發生錯誤時返回預設動作
            return 1 if observation[2] > 0 else 0
        
    def close(self):
        pass