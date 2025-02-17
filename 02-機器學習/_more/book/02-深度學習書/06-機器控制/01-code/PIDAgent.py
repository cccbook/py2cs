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