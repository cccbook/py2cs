class PIDController:
    def __init__(self, Kp, Ki, Kd, dt=0.02):
        # 初始化 PID 控制器的增益和時間步長
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        
        # 初始化 PID 控制器的誤差積分和微分
        self.prev_error = 0
        self.integral = 0

    def reset(self):
        # 重置 PID 控制器的狀態
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        # 計算比例項、積分項和微分項
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        
        # 計算 PID 控制輸出
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # 更新上一個誤差
        self.prev_error = error
        
        return output


class PIDAgent:
    def __init__(self, env):
        # 初始化 PID 控制器
        self.controller = PIDController(Kp=1.0, Ki=0.1, Kd=0.1)
        self.env = env

    def reset(self, mode=None):
        # 重置 PID 控制器
        self.controller.reset()

    def step(self, observation, reward, terminated):
        # 提取觀察值 (角度和角速度)
        position, velocity, angle, angle_velocity = observation
        
        # 計算角度誤差
        error = angle
        
        # 計算控制輸出
        control = self.controller.compute(error)
        
        # 根據控制輸出決定動作
        action = 1 if control > 0 else 0  # 如果控制信號為正，則推動右邊，否則推動左邊
        return action

    def close(self):
        pass