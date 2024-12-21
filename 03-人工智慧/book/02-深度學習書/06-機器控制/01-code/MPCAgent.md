Claude:


我來幫你設計一個基於模型預測控制(MPC)的智能體來控制 CartPole 系統。MPC的核心思想是:
1. 使用系統模型預測未來狀態
2. 在預測範圍內最小化代價函數
3. 只執行最優控制序列的第一個動作



```python
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
        
    def reset(self, mode=None):
        pass
        
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
        state = current_state
        
        for i in range(len(u_sequence)):
            # 狀態代價
            state_cost = state @ self.Q @ state
            # 控制代價
            control_cost = self.R * u_sequence[i]**2
            total_cost += state_cost + control_cost
            
            # 預測下一狀態
            state = self.dynamics(state, u_sequence[i])
            
        return total_cost
        
    def step(self, observation, reward, terminated):
        """執行一步MPC控制"""
        # 初始化控制序列
        u_init = np.zeros(self.horizon)
        
        # 求解最優控制序列
        result = minimize(
            fun=self.cost_function,
            x0=u_init,
            args=(observation,),
            method='SLSQP',
            bounds=[(-10, 10)] * self.horizon  # 控制量限制
        )
        
        # 取最優控制序列的第一個動作
        optimal_u = result.x[0]
        
        # 將連續控制轉換為離散動作
        action = 1 if optimal_u > 0 else 0
        
        return action
        
    def close(self):
        pass

```

這個 MPCAgent 實現了以下核心功能:

1. **系統建模**: 
   - 實現了 CartPole 的簡化動力學模型
   - 考慮了小車質量、擺桿質量、重力等物理參數

2. **預測控制**:
   - 使用 10 步預測視界
   - 在每個時間步優化未來控制序列
   - 通過最小化代價函數求解最優控制

3. **代價函數設計**:
   - 包含狀態代價和控制代價
   - 狀態代價權重矩陣 Q 重點關注角度和角速度
   - 控制代價係數 R 平衡控制效果和能量消耗

4. **優化求解**:
   - 使用 SLSQP 方法求解非線性優化問題
   - 設置了控制量的上下界限制

相比於原始的 ClosedFormAgent，這個 MPC 智能體具有以下優勢:
1. 考慮了系統的動力學特性
2. 能夠預測和優化未來行為
3. 可以處理控制約束
4. 控制性能更好(理論上)

需要注意的是，這個實現使用了簡化的動力學模型，如果需要更精確的控制，可以進一步完善系統模型和調整控制參數。