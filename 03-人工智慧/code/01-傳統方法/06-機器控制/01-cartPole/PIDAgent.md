

## ccc: 請修改下列程式，用傳統 PID 控制器的方法改寫

...

要將這個程式改寫成使用傳統 PID 控制器來控制 CartPole 環境，我們需要根據當前的系統狀態（例如角度和角速度）來設計 PID 控制器。PID 控制器通常基於三個組成部分：

1. **比例項 (P)**：根據誤差（目標值和當前值之間的差異）進行反應。
2. **積分項 (I)**：根據誤差的累積進行反應，主要用來消除穩態誤差。
3. **微分項 (D)**：根據誤差的變化率進行反應，主要用來預測未來的誤差。

以下是修改後的程式，使用 PID 控制器來進行 CartPole 的控制：

```python
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
```

### 程式碼改動：
1. **PID 控制器類 (`PIDController`)**：
   - `Kp`, `Ki`, `Kd`：比例、積分和微分增益。
   - `reset()`：重置 PID 控制器的狀態。
   - `compute(error)`：計算 PID 控制器的輸出，根據當前的誤差（在這裡使用的是角度誤差）來計算控制信號。

2. **PID 代理類 (`PIDAgent`)**：
   - 使用 `PIDController` 類來計算基於角度誤差的控制輸入。
   - 根據控制信號的正負來決定應該推動小車的方向，進行左或右的控制。

3. **訓練過程**：
   - 代理根據 PID 控制器的輸出選擇動作，並通過 `play_episode()` 進行環境互動。

### PID 控制原理：
- **比例項 (P)**：對於 CartPole 問題，比例項主要針對角度誤差進行修正，使得倒立擺朝著直立狀態（角度 0）進行調整。
- **積分項 (I)**：積分項會根據誤差的積累進行反應，幫助消除穩態誤差。
- **微分項 (D)**：微分項則是對誤差變化率的反應，對於快速變化的誤差提供預測控制。

### 控制策略：
- 根據計算的控制信號，若為正則推動小車向右移動，若為負則推動小車向左移動。

### 使用說明：
- 該程式會先訓練 100 回合，然後顯示每回合的回報和步數，並計算平均回報。
- 使用 `render=True` 時，會播放訓練過程的動畫。

### 注意：
- PID 控制器在這個問題中雖然能夠提供一種基本的控制方法，但在較為複雜或非線性問題中，PID 可能無法提供最佳的性能，因此對於這類問題，可能需要更為高級的控制策略（如 MPC 或強化學習）。