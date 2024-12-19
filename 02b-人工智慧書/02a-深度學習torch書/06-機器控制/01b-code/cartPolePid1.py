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
env = gym.make('CartPole-v1', render_mode="human")
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
        # print('obs=', obs)
        observation, info = obs
        # position, velocity, angle, angle_velocity
        _, _, theta, theta_dot = observation
        
        # 計算控制輸入
        error = -theta  # 目標角度是 0，目標值 - 當前值
        control = pid.compute(error, time_step)

        # 將控制輸入轉化為動作 (0 或 1)
        action = 1 if control > 0 else 0

        # 與環境交互
        # obs, reward, done, _ = env.step(action)
        observation, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
        total_reward += reward

        # 模擬時間步
        time.sleep(time_step)

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
