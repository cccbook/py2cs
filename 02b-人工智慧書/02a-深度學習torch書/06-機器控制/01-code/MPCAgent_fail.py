import numpy as np
import cvxpy as cp
import gym
from math import sin, cos

class MPCController:
    def __init__(self, model, horizon=10, Q=1.0, R=0.1):
        """
        初始化模型預測控制（MPC）控制器。
        
        :param model: 系統模型，應該有一個step函數來計算下一個狀態
        :param horizon: 預測的時間步長
        :param Q: 狀態加權矩陣
        :param R: 控制輸入加權矩陣
        """
        self.model = model
        self.horizon = horizon
        self.Q = Q
        self.R = R

    def reset(self):
        pass

    def step(self, observation):
        """
        根據當前狀態計算MPC控制輸出。
        
        :param observation: 當前的觀察狀態 (position, velocity, angle, angle_velocity)
        :return: 控制輸出
        """
        # 進行預測的變數設置
        x = np.array(observation)  # 當前狀態 (position, velocity, angle, angle_velocity)
        
        # 定義控制輸入變數 (這些是我們的優化變數)
        u = cp.Variable((self.horizon, 1))
        
        # 定義預測的狀態變數 (每個時間步都有一個狀態)
        x_pred = cp.Variable((self.horizon + 1, 4))
        
        # 定義代價函數 (最小化狀態誤差和控制輸入)
        cost = 0
        constraints = []
        for t in range(self.horizon):
            # 計算下一時間步的狀態預測
            constraints.append(x_pred[t+1] == self.model.step(x_pred[t], u[t]))
            # 代價函數：狀態誤差 + 控制輸入
            cost += self.Q * cp.norm(x_pred[t] - np.zeros(4), "fro")**2 + self.R * cp.norm(u[t], "fro")**2
        
        # 初始狀態
        constraints.append(x_pred[0] == x)
        
        # 解決優化問題
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()
        
        # 返回第一個控制輸出
        return u.value[0, 0]

class CartPoleModel:
    def __init__(self):
        pass
    
    def step(self, state):
        """
        根據當前狀態和控制輸入，返回下一個狀態。
        這是系統的動力學模型，應該是倒立擺的數學模型。
        
        :param state: 當前狀態 (position, velocity, angle, angle_velocity)
        # :param control: 控制輸入
        :return: 下一狀態
        """
        print('state=', state)
        # 當前狀態
        position, velocity, angle, angle_velocity = state
        
        #print('control=', control)
        # 控制力（假設控制是加在推車上的力）
        #force = control
        
        # 系統常數
        m = 1.0  # 小車質量
        M = 1.0  # 擺杆質量
        l = 1.0  # 擺杆長度
        g = 9.8  # 重力加速度
        d = 0.1  # 摩擦阻力
        
        print('angle=', angle)
        # 使用 numpy 計算 sin 和 cos
        sin_theta = sin(angle)
        cos_theta = cos(angle)
        
        # 系統的運動方程（簡化的倒立擺模型）
        theta_acc = (g * sin_theta - cos_theta * (force + d * velocity)) / (l * (4/3 - M * cos_theta**2 / (m + M)))
        x_acc = (force + m * l * (theta_acc * cos_theta - g * sin_theta)) / (m + M)
        
        # 計算下一狀態
        new_position = position + velocity * 0.02
        new_velocity = velocity + x_acc * 0.02
        new_angle = angle + angle_velocity * 0.02
        new_angle_velocity = angle_velocity + theta_acc * 0.02
        
        # 返回結果作為 numpy 陣列，轉回 cvxpy 表達式
        return np.array([new_position, new_velocity, new_angle, new_angle_velocity])


class MPCAgent:
    def __init__(self, env):
        self.env = env
        self.model = CartPoleModel()
        self.controller = MPCController(self.model)

    def reset(self, mode=None):
        pass

    def step(self, observation, reward, terminated):
        # 使用 MPC 控制器計算控制輸出
        print('observation=', observation)
        action = self.controller.step(observation)
        # 根據計算的控制輸出決定動作
        return 1 if action > 0 else 0

    def close(self):
        pass

