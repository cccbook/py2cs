

## ccc: 請給出 PPO 的演算法

**Proximal Policy Optimization (PPO)** 是強化學習中的一種策略優化算法，具有良好的訓練穩定性和性能。PPO 透過限制策略更新的幅度，來避免策略過度更新（overfitting），從而提高訓練的穩定性。

以下是 PPO 的演算法步驟：

### PPO 演算法步驟：

1. **初始化策略參數** \(\theta\) 和價值函數參數 \(\phi\)。

2. **重複以下步驟直到收斂：**

   1. **收集樣本數據：**
      - 使用當前策略 \(\pi_\theta\) 進行多次環境交互，收集一批經驗數據 \((s_t, a_t, r_t, s_{t+1})\)，即狀態、動作、回報和下一個狀態。
      - 使用這些數據來估計每個時間步的優勢 \(A_t\)，這裡可以使用**廣義優勢估計（GAE）**來計算：
      \[
      A_t = \delta_t + (\gamma \lambda) \delta_{t+1} + \dots + (\gamma \lambda)^{T-t+1} \delta_{T-1}
      \]
      其中 \(\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)\) 是 TD 誤差。

   2. **計算策略更新：**
      - 計算每個動作的**舊策略概率** \( \pi_{\theta_{\text{old}}}(a_t | s_t) \)。
      - 使用現有策略 \(\pi_\theta\) 計算**新的策略概率** \( \pi_\theta(a_t | s_t) \)。
      - 計算**概率比率** \( r_t(\theta) \)：
      \[
      r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
      \]

   3. **更新策略參數：**
      - 對每個動作計算以下兩個損失：
        \[
        L^{\text{clip}}_t(\theta) = \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right)
        \]
      其中 \(\epsilon\) 是限制更新幅度的超參數，通常設置為 0.1 到 0.3 左右。這個公式的作用是限制策略變化太大，從而保持策略更新的穩定性。
   
   4. **更新價值函數參數：**
      - 使用均方誤差來更新價值函數：
        \[
        L^{\text{value}}(\phi) = \frac{1}{2} \sum_{t} (V_\phi(s_t) - \hat{R}_t)^2
        \]
      其中 \(\hat{R}_t\) 是估計的回報。
   
   5. **最終損失函數：**
      - PPO 的損失函數同時包含策略更新和價值函數更新，並且加入了熵懲罰來鼓勵策略的探索：
        \[
        L^{\text{PPO}}(\theta, \phi) = \mathbb{E} \left[ L^{\text{clip}}_t(\theta) - c_1 L^{\text{value}}(\phi) + c_2 \mathbb{H}[\pi_\theta](s_t) \right]
        \]
      其中 \(c_1, c_2\) 是權重超參數，\(\mathbb{H}[\pi_\theta](s_t)\) 是策略的熵，鼓勵多樣性。

   6. **使用梯度下降更新參數：**
      - 根據損失函數 \(L^{\text{PPO}}(\theta, \phi)\) 使用梯度下降法來更新策略和價值函數的參數。

### PPO 演算法示意圖：

```
初始化策略參數 \theta 和價值函數參數 \phi
重複以下步驟直到收斂：
    1. 使用當前策略 \pi_\theta 在環境中與之交互，收集狀態、動作、回報
    2. 計算每個時間步的優勢 A_t
    3. 計算舊的策略概率 \pi_{\theta_{\text{old}}}(a_t | s_t)
    4. 使用 PPO Clip 計算新的策略概率和損失
    5. 使用梯度下降法更新策略和價值函數參數 \theta 和 \phi
```

### PPO 的 Python 代碼範例（簡化版）

這裡是一個簡化的 PPO 演算法的代碼示例，基於 PyTorch，使用假設的環境來展示如何更新策略。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 假設的策略網絡和價值網絡
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.actor = nn.Linear(128, action_dim)  # 策略
        self.critic = nn.Linear(128, 1)  # 價值函數
    
    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.actor(x), self.critic(x)
    
    def act(self, state):
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value

# PPO 演算法
class PPO:
    def __init__(self, actor_critic, lr=3e-4, gamma=0.99, epsilon=0.2, ppo_epochs=4):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.ppo_epochs = ppo_epochs
    
    def compute_advantages(self, rewards, values, next_value):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * gae
            advantages.insert(0, gae)
            next_value = values[i]
        return advantages
    
    def update(self, states, actions, log_probs_old, rewards, values, next_value):
        advantages = self.compute_advantages(rewards, values, next_value)
        
        for _ in range(self.ppo_epochs):
            logits, value = self.actor_critic(states)
            dist = Categorical(torch.softmax(logits, dim=-1))
            new_log_probs = dist.log_prob(actions)
            
            # 計算比例 r_t(θ)
            ratios = torch.exp(new_log_probs - log_probs_old)
            
            # 損失計算
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()
            
            # 價值損失
            loss_critic = nn.MSELoss()(value, rewards)
            
            # 總損失
            loss = loss_actor + 0.5 * loss_critic
            
            # 反向傳播更新參數
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# 假設狀態和行為空間
input_dim = 4  # 假設的狀態維度
action_dim = 2  # 假設的動作空間大小
ppo = PPO(ActorCritic(input_dim, action_dim))

# 示例更新步驟
states = torch.rand((5, input_dim))  # 假設的一批狀態
actions = torch.randint(0, action_dim, (5,))  # 假設的動作
log_probs_old = torch.rand((5,))  # 假設的舊 log_probs
rewards = torch.rand((5,))  # 假設的回報
values = torch.rand((5,))  # 假設的價值
next_value = torch.rand(1)  # 下一狀態的價值

# 執行 PPO 更新
ppo.update(states, actions, log_probs_old, rewards, values, next_value)
```

### 總結：
PPO 透過限制每次策略更新的幅度，達到穩定性與有效性之間的平衡。該演算法在策略優化時，保證了不會過度更新（過大的變化），從而讓訓練更加穩定且高效。