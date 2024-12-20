好的，我來幫你實現一個基於 PID 控制的版本。對於 CartPole 系統，我們主要關注角度的控制，使用 PID 控制器來保持平衡。



```python
import numpy as np

class PIDAgent:
    def __init__(self, env):
        # PID 控制器參數
        self.Kp = 10.0    # 比例增益
        self.Ki = 0.1     # 積分增益
        self.Kd = 2.0     # 微分增益
        
        # 初始化誤差相關變量
        self.prev_error = 0.0    # 上一步誤差
        self.integral = 0.0      # 積分項
        self.dt = 0.02          # 時間步長
        
        # 積分限幅
        self.integral_limit = 1.0
        
    def reset(self, mode=None):
        """重置 PID 控制器狀態"""
        self.prev_error = 0.0
        self.integral = 0.0
        return
        
    def step(self, observation, reward=0, terminated=False):
        """執行一步 PID 控制"""
        # 解析觀測值
        _, _, angle, angle_velocity = observation
        
        # 計算誤差 (目標是保持垂直，即角度為0)
        error = angle
        
        # 更新積分項
        self.integral += error * self.dt
        # 限制積分範圍，防止積分飽和
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        # 計算微分項 (使用角速度作為微分項)
        derivative = angle_velocity
        
        # PID 控制輸出
        control = (self.Kp * error + 
                  self.Ki * self.integral + 
                  self.Kd * derivative)
        
        # 保存當前誤差
        self.prev_error = error
        
        # 將連續控制轉換為離散動作 (0 或 1)
        action = 1 if control > 0 else 0
        
        return action
        
    def close(self):
        pass

```

這個 PID 控制器的特點：

1. **控制目標**：
   - 控制桿子保持垂直位置（角度為0）
   - 使用角度作為主要控制誤差
   - 使用角速度作為微分項

2. **PID 參數設置**：
   - `Kp = 10.0`：較大的比例增益，對誤差有強烈反應
   - `Ki = 0.1`：較小的積分增益，緩慢消除穩態誤差
   - `Kd = 2.0`：適中的微分增益，提供阻尼作用

3. **特殊處理**：
   - 積分限幅，防止積分項過大
   - 直接使用角速度作為微分項，避免數值微分的噪聲
   - 連續控制輸出轉換為離散動作

4. **優點**：
   - 實現簡單，計算負擔小
   - 不需要系統模型
   - 容易調整參數

5. **局限性**：
   - 控制效果依賴於參數調整
   - 不考慮系統約束
   - 沒有預測能力

相比 MPC 方法，PID 控制器更簡單且計算量更小，但控制效果可能不如 MPC 精確。你可以通過調整 Kp、Ki、Kd 參數來優化控制效果。