import sys
import logging
import numpy as np
import gym

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    stream=sys.stdout, datefmt='%H:%M:%S')

env = gym.make('CartPole-v0')
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])
for key in vars(env.unwrapped):
    logging.info('%s: %s', key, vars(env.unwrapped)[key])


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


# 訓練過程
def play_episode(env, agent, seed=None, mode=None, render=False):
    observation, _ = env.reset(seed=seed)
    reward, terminated, truncated = 0., False, False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        action = agent.step(observation, reward, terminated)
        if render:
            env.render()
        if terminated or truncated:
            break
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
    agent.close()
    return episode_reward, elapsed_steps


logging.info('==== test ====')
episode_rewards = []
agent = PIDAgent(env)  # 創建 PID 控制器代理
for episode in range(100):
    episode_reward, elapsed_steps = play_episode(env, agent)
    episode_rewards.append(episode_reward)
    logging.info('test episode %d: reward = %.2f, steps = %d',
                 episode, episode_reward, elapsed_steps)

logging.info('average episode reward = %.2f ± %.2f',
             np.mean(episode_rewards), np.std(episode_rewards))
env.close()

# 使用 render (for human) 動畫播放玩一次
env = gym.make('CartPole-v0', render_mode="human")
episode_reward, elapsed_steps = play_episode(env, agent, render=True)
env.close()
