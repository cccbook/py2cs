## ccc: 請仿造下列 PICAgent，用 LQR 方法改寫

了解，下面我們會使用 **LQR（線性二次調節器）** 來改寫控制程式。LQR 是一種最優控制策略，通常用於線性系統，其中控制器的目的是最小化加權誤差（狀態誤差和控制輸入）的二次形式。

### LQR 控制的基本原理
LQR 控制基於以下問題的解：
\[
J = \int_0^\infty \left( x^T Q x + u^T R u \right) dt
\]
其中：
- \( x \) 是系統的狀態向量（如 CartPole 的位置、速度、角度、角速度等）。
- \( u \) 是控制輸入（動作）。
- \( Q \) 和 \( R \) 是加權矩陣，分別對狀態誤差和控制輸入的權重進行調整。

LQR 的目標是選擇一組最優控制律，這樣可以最小化 \( J \)，其中控制律通常是：
\[
u = -Kx
\]
其中 \( K \) 是計算得到的最優增益矩陣，通過求解里卡提方程得到。

### 使用 LQR 控制 CartPole 的程式碼
下面的程式碼實現了使用 LQR 控制 CartPole 問題。

```python
import sys
import logging
import numpy as np
import gym
from scipy.linalg import inv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    stream=sys.stdout, datefmt='%H:%M:%S')

env = gym.make('CartPole-v0')
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])
for key in vars(env.unwrapped):
    logging.info('%s: %s', key, vars(env.unwrapped)[key])

class LQRController:
    def __init__(self, Q, R, dt=0.02):
        # LQR 控制器的加權矩陣 Q 和 R
        self.Q = Q
        self.R = R
        self.dt = dt
        
        # 初始化系統的狀態矩陣
        self.A = np.array([[0, 1, 0, 0],
                           [0, 0, -10, 0],
                           [0, 0, 0, 1],
                           [0, 0, 10, 0]])
        self.B = np.array([[0],
                           [1],
                           [0],
                           [-1]])
        
        # 計算 LQR 增益矩陣 K
        self._compute_lqr_gain()

    def _compute_lqr_gain(self):
        # 解決里卡提方程獲得增益矩陣 K
        P = np.linalg.solve(self.A.T @ self.Q @ self.A - self.A.T @ self.R @ self.B @ inv(self.B.T @ P @ self.B) @ self.B.T @ self.A, self.A)
        self.K = inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A  # LQR增益矩陣

    def reset(self):
        pass

    def compute(self, state):
        # 計算 LQR 控制信號
        return -np.dot(self.K, state)

class LQRAgent:
    def __init__(self, env):
        self.env = env
        # 初始化 LQR 控制器，設置 Q 和 R 加權矩陣
        Q = np.diag([10, 1, 100, 10])  # 可以根據需要調整 Q 的值
        R = np.array([[1]])  # 控制輸入的加權矩陣
        self.controller = LQRController(Q, R)
        
    def reset(self, mode=None):
        self.controller.reset()

    def step(self, observation, reward, terminated):
        # 提取觀察值 (位置, 速度, 角度, 角速度)
        position, velocity, angle, angle_velocity = observation
        state = np.array([position, velocity, angle, angle_velocity])  # 系統狀態向量
        
        # 計算 LQR 控制輸出
        control = self.controller.compute(state)
        
        # 根據 LQR 控制信號決定動作
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
agent = LQRAgent(env)  # 使用 LQR 控制器的代理
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

### 程式碼解析：

1. **LQR 控制器（LQRController）**：
   - `Q` 和 `R` 是加權矩陣，用來調整狀態誤差和控制輸入的權重。
   - `A` 和 `B` 是系統的狀態轉移矩陣和控制矩陣，用來描述 CartPole 的動力學。
   - `compute` 方法計算控制信號，這是根據 LQR 的最優增益矩陣 \( K \) 和系統當前的狀態向量來生成控制信號的。

2. **LQR 代理（LQRAgent）**：
   - 在每一步中，代理會根據當前的系統狀態（位置、速度、角度、角速度）計算控制輸出。
   - 根據 LQR 的控制信號選擇動作。

3. **LQR 控制律**：
   - LQR 控制器的目的是最小化狀態誤差和控制輸入的加權二次型成本函數。控制律為 \( u = -Kx \)，其中 \( K \) 是 LQR 增益矩陣。

### LQR 控制特點：
- **穩定性與最優性**：LQR 是一種最優控制方法，它可以在滿足線性系統模型的情況下，提供穩定且最優的控制策略。
- **需要精確的系統模型**：LQR 假設系統是線性的，並且需要準確的狀態轉移矩陣和控制矩陣。

### 注意：
- 您可以根據需要調整 Q 和 R 矩陣來改變控制的效果。Q 矩陣通常用來加大對某些狀態變量（如角度誤差）的懲罰，而 R 矩陣則對控制輸入進行懲罰。

這樣的 LQR 控制方法能夠提供高效且穩定的控制策略，尤其在已知系統動力學的情況下非常有效。