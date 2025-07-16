## ccc: 請用 python 寫一個 LQR 版的 gym CartPole 控制程式

下面是使用 Python 編寫的 LQR（線性二次調節器）控制器來控制 OpenAI Gym 的 CartPole 環境的程式。LQR 是一種經典的最優控制方法，旨在最小化控制系統的加權平方誤差和控制輸入的平方和。

以下是完整的程式碼：

```python
import gym
import numpy as np
from scipy.linalg import solve_continuous_are

# 定義 LQR 控制器
class LQRController:
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        # 計算 LQR 的增益矩陣 K
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R).dot(self.B.T.dot(P))

    def compute_control(self, state):
        # 控制輸入：u = -K * x
        return -self.K.dot(state)

def main():
    # 創建 CartPole 環境
    env = gym.make('CartPole-v1')
    env.reset()

    # 系統的線性化模型
    # A 和 B 矩陣來自於 CartPole 系統的線性化
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -10.0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 20.0, 0]])

    B = np.array([[0],
                  [1.0],
                  [0],
                  [-1.0]])

    # 設置 Q 和 R 矩陣，用於 LQR 的性能指標
    Q = np.diag([1, 1, 10, 100])  # 重視位置和角度誤差
    R = np.array([[1]])  # 控制成本

    # 創建 LQR 控制器
    lqr = LQRController(A, B, Q, R)

    # 訓練循環
    for episode in range(10):  # 執行 10 次試驗
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 根據當前狀態計算控制輸入
            control_input = lqr.compute_control(state)

            # 執行控制輸入並獲得新的狀態
            next_state, reward, done, _ = env.step([control_input[0]])

            # 更新狀態
            state = next_state
            total_reward += reward

            # 繪製 CartPole 環境
            env.render()

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    # 關閉環境
    env.close()

if __name__ == "__main__":
    main()
```

### 程式碼解釋：
1. **LQR 控制器類別 (`LQRController`)**：
   - 初始化過程中，計算 LQR 的增益矩陣 `K`，使用的是 `scipy.linalg.solve_continuous_are` 解線性二次調節器的連續時間代數 Riccati 方程（CARE）。
   - `compute_control` 方法根據當前狀態計算控制輸入 `u`，其公式為 `u = -K * x`，其中 `x` 是系統的狀態向量。

2. **主要程序 (`main`)**：
   - 創建 CartPole 環境並設置線性化的系統矩陣 `A` 和 `B`。
   - 設置 LQR 的權重矩陣 `Q` 和 `R`，控制系統中的誤差和控制輸入的加權。
   - 使用 LQR 控制器根據環境狀態計算控制輸入，並通過環境的 `step` 方法執行該控制輸入來進行下一步。

3. **CartPole 環境**：
   - 這是來自 OpenAI Gym 的標準環境，用於模擬倒立擺（CartPole）問題。在這個環境中，我們的目標是使倒立擺保持平衡。

### 注意：
- 這個程式碼是基於 CartPole 環境的線性化模型，LQR 假設系統是線性的，這在某些情況下可能對真實環境的行為有一定的偏差。
- 在較大時間範圍內，這樣的控制可能無法完美地控制非線性系統，但它對較小擾動（如短期內的控制）非常有效。