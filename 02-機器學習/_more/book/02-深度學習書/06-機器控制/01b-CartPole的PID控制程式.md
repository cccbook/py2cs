## ccc: 請用 python 寫一個 PID 版的 gym CartPole 控制程式

以下是一個基於 OpenAI Gym 的 CartPole 環境的 PID 控制器實現。這個程式控制小車的輸入力 \(F\)，以穩定倒立擺的位置和角度。

```py
import gym
import numpy as np
import time

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# 初始化 CartPole 環境
env = gym.make('CartPole-v1')
env.reset()

# PID 參數
kp_theta = 50.0  # 角度比例增益
ki_theta = 1.0   # 角度積分增益
kd_theta = 10.0  # 角度微分增益

pid = PIDController(kp=kp_theta, ki=ki_theta, kd=kd_theta)

time_step = 0.02  # 假設每個步驟的時間間隔

episodes = 5
for episode in range(episodes):
    obs = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        env.render()
        # 提取角度 (theta)
        _, _, theta, theta_dot = obs
        
        # 計算控制輸入
        error = -theta  # 目標角度是 0，目標值 - 當前值
        control = pid.compute(error, time_step)

        # 將控制輸入轉化為動作 (0 或 1)
        action = 1 if control > 0 else 0

        # 與環境交互
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        # 模擬時間步
        time.sleep(time_step)

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()

```

### 說明
1. **PID 控制器**：
   - `kp_theta`、`ki_theta` 和 `kd_theta` 是控制擺桿角度的 PID 增益。
   - PID 控制器根據角度誤差計算作用力，試圖將擺桿角度穩定在 0（直立位置）。

2. **環境步驟**：
   - 每次步驟，從觀察值中提取 \(\theta\) 和 \(\dot{\theta}\)。
   - 使用 PID 控制器計算動作，然後將動作應用到環境。

3. **渲染與模擬**：
   - `env.render()` 會顯示 CartPole 環境的圖形界面。
   - `time.sleep(time_step)` 用於模擬實際時間間隔。

4. **結果**：
   - 每個回合輸出總獎勵，總獎勵越高，表示控制效果越好。

執行此程式時，CartPole 的穩定性取決於 PID 增益的選擇，可以根據需求調整 `kp_theta`, `ki_theta`, 和 `kd_theta` 的值來優化性能。