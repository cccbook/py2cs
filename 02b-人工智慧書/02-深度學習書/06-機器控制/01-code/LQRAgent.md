Claude:

好的，我來實現一個基於 LQR (Linear Quadratic Regulator) 的控制器。LQR 是一種最佳控制方法，它通過求解李雅普諾夫方程來找到最佳的狀態反饋增益矩陣。



```python
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

```

這個 LQR 控制器的主要特點：

1. **系統線性化**：
   - 在平衡點(垂直位置)處進行線性化
   - 得到線性時不變系統的狀態空間表示
   - 考慮了完整的動力學模型

2. **LQR 設計**：
   - 使用狀態代價矩陣 Q 和控制代價矩陣 R
   - 求解離散時間代數 Riccati 方程
   - 計算最優狀態反饋增益矩陣

3. **控制實現**：
   - 使用線性狀態反饋控制律
   - 將連續控制轉換為離散動作

4. **特點**：
   - 提供了最優控制性能（在線性化模型下）
   - 計算量適中，實時性好
   - 具有良好的穩定性和魯棒性

5. **與其他方法比較**：
   - 比 PID 更系統化，考慮了完整的狀態
   - 比 MPC 計算量更小
   - 但只在平衡點附近有較好的效果

這三種控制器（MPC、PID、LQR）各有特點：
- MPC：考慮預測和約束，但計算量大
- PID：簡單直觀，但參數調節需要經驗
- LQR：理論最優，但依賴於線性化模型

你可以根據具體需求選擇適合的控制方法。