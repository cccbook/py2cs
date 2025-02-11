## ccc: 請用 python 寫一個 MPC 版的 gym CartPole 控制程式

以下是使用模型預測控制（MPC）來控制 OpenAI Gym 中的 CartPole 環境的範例程式。MPC 是一種基於優化的方法，透過解決一個優化問題來預測未來一段時間的控制行為。

MPC 的核心概念是基於當前狀態來預測未來的系統行為，並且在每個時間步中解決一個最優控制問題。每次控制步驟會在一個有限的時間範圍內進行優化，並且只執行該時間範圍的首個控制指令。

以下是使用 MPC 的完整程式碼：

```python
import gym
import numpy as np
import scipy.linalg

# 定義 MPC 控制器
class MPCController:
    def __init__(self, A, B, Q, R, horizon=10, dt=0.02):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.horizon = horizon  # 預測的時間範圍
        self.dt = dt  # 時間步長

    def compute_control(self, state):
        # 計算 MPEC 的控制輸入
        n = len(state)  # 狀態向量的維度
        m = self.B.shape[1]  # 控制輸入的維度

        # 預測未來的狀態
        X = np.zeros((self.horizon + 1, n))
        U = np.zeros((self.horizon, m))

        X[0] = state

        # 解決最優控制問題（Q、R 和 A、B 用於狀態與控制）
        for t in range(self.horizon):
            # 計算控制輸入
            U[t] = np.linalg.inv(self.R + self.B.T @ self.Q @ self.B) @ (self.B.T @ self.Q @ X[t])
            X[t + 1] = self.A @ X[t] + self.B @ U[t]

        # 返回計算得到的首個控制輸入
        return U[0]

def main():
    # 創建 CartPole 環境
    env = gym.make('CartPole-v1')
    env.reset()

    # 系統的線性化模型
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -10.0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 20.0, 0]])

    B = np.array([[0],
                  [1.0],
                  [0],
                  [-1.0]])

    # 設置 Q 和 R 矩陣，用於 MPEC 的性能指標
    Q = np.diag([1, 1, 10, 100])  # 重視位置和角度誤差
    R = np.array([[1]])  # 控制成本

    # 創建 MPC 控制器
    mpc = MPCController(A, B, Q, R)

    # 訓練循環
    for episode in range(10):  # 執行 10 次試驗
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 根據當前狀態計算控制輸入
            control_input = mpc.compute_control(state)

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

1. **MPC 控制器類別 (`MPCController`)**：
   - **初始化方法**：設定了系統的狀態和控制矩陣（`A` 和 `B`），以及預測控制的時間範圍 `horizon` 和時間步長 `dt`。
   - **`compute_control` 方法**：該方法基於當前狀態預測未來一段時間的狀態（根據給定的模型），並通過優化問題計算每個時間步的最優控制輸入。這裡使用了線性系統模型來進行預測。

2. **主要程序 (`main`)**：
   - 創建 CartPole 環境，並設置系統的線性化模型（`A` 和 `B` 矩陣）。
   - 設置 MPC 的權重矩陣 `Q` 和 `R`，它們決定了誤差和控制輸入的加權。
   - 在每個回合中，根據當前的系統狀態計算並應用 MPC 控制器的輸出，並且執行相應的控制命令。

3. **CartPole 環境**：
   - 這是來自 OpenAI Gym 的標準環境，目的是使倒立擺（CartPole）保持平衡。

### MPC 控制原理：
- **預測**：在每個控制步驟中，MPC 會根據當前狀態預測未來的狀態，並且解決一個優化問題以最小化一個性能指標。
- **優化問題**：該問題會考慮狀態誤差和控制輸入的代價。預測的未來狀態由線性系統模型（`A` 和 `B` 矩陣）生成。
- **解決**：每次控制步驟只執行預測的控制序列的第一個控制輸入，然後重複這一過程。

### 注意：
- 這個 MPC 實現是基於線性化模型，適用於相對簡單的情況。在非線性系統中，MPC 可以擴展為非線性控制方法，但會需要更復雜的優化算法（如序列二次規劃等）。
- MPC 在這種情況下的計算會隨著預測範圍 `horizon` 的增大而變得更為複雜，因此需要在實際使用中根據需要進行適當的調整。