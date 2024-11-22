# https://github.com/datawhalechina/easy-rl/blob/master/notebooks/DQN.ipynb
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_states,n_actions,hidden_dim=128): # 初始化q網絡，為全連接網絡
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim) # 輸入層
        self.fc2 = nn.Linear(hidden_dim,hidden_dim) # 隱藏層
        self.fc3 = nn.Linear(hidden_dim, n_actions) # 輸出層
        
    def forward(self, x): # 前向傳遞
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)

from collections import deque
import random
class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self,transitions): # 儲存 transition 到經驗回放中
        self.buffer.append(transitions)

    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer): # 如果批量大小大於經驗回放的容量，則取經驗回放的容量
            batch_size = len(self.buffer)
        if sequential: # 順序采樣
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else: # 隨機采樣
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
        
    def clear(self): # 清空經驗回放
        self.buffer.clear()

    def __len__(self): # 返回當前存儲的量
        return len(self.buffer)

import torch
import torch.optim as optim
import math
import numpy as np

class DQN:

    def __init__(self,model,memory,cfg):
        self.n_actions = cfg['n_actions'] # 最大動作數
        self.device = torch.device(cfg['device']) 
        self.gamma = cfg['gamma'] # 獎勵的折扣因子
        # e-greedy策略相關參數
        self.sample_count = 0  # 用於epsilon的衰減計數
        self.epsilon = cfg['epsilon_start']
        self.sample_count = 0  
        self.epsilon_start = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']
        self.batch_size = cfg['batch_size']
        # policy_net 和 target_net 一開始相同
        self.policy_net = model.to(self.device)
        self.target_net = model.to(self.device)
        # 複製參數到目標網絡
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): 
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg['lr']) # 優化器
        self.memory = memory # 經驗回放

    def sample_action(self, state): # 采樣動作
        self.sample_count += 1
        # epsilon指數衰減
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        else:
            action = random.randrange(self.n_actions)
        return action

    @torch.no_grad() # 不計算梯度，該裝飾器效果等同於with torch.no_grad()：
    def predict_action(self, state): # 預測動作
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        q_values = self.policy_net(state)
        action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        return action

    def update(self):
        if len(self.memory) < self.batch_size: # 當經驗回放中不滿足一個批量時，不更新策略
            return
        # 從經驗回放中隨機采樣一個批量的轉移(transition)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        # ccc: 所謂的 batch，是 (s[t], a[t], r[t], s[t+1], done[t+1]) 這樣的 sample ...
        # 將數據轉換為tensor
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        # ccc: 重點程式，DQN 用 q_values 與 next_q_values 的差異作為引導，讓 DQN 朝 next_q_values 方向前進。
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch) # 計算當前狀態的 Q(s[t], a)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach() # 計算下一狀態的 max(Q'(s[t+1],a))
        expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch) # 計算期望的Q值，對於終止狀態，此時done_batch[0]=1, 對應的expected_q_value等於reward
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 計算均方根損失
        # == 反傳遞並向逆梯度方向前進一小步 ==
        self.optimizer.zero_grad()  
        loss.backward()
        # clip防止梯度爆炸
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 

def train(cfg, env, agent): # 訓練
    print("開始訓練！")
    rewards = []  # 記錄所有回合的獎勵
    steps = []
    for i_ep in range(cfg['train_eps']):
        ep_reward = 0  # 記錄一回合內的獎勵
        ep_step = 0
        state, info = env.reset()  # 重置環境，返回初始狀態
        for _ in range(cfg['ep_max_steps']):
            ep_step += 1
            action = agent.sample_action(state)  # 選擇動作
            next_state, reward, done, _, info = env.step(action)  # 更新環境，返回transition
            agent.memory.push((state, action, reward,next_state, done))  # 保存transition
            state = next_state  # 更新下一個狀態
            agent.update()  # 更新智能體
            ep_reward += reward  # 累加獎勵
            if done:
                break
        if (i_ep + 1) % cfg['target_update'] == 0:  # 智能體目標網絡更新
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        steps.append(ep_step)
        rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print(f"回合：{i_ep+1}/{cfg['train_eps']}，獎勵：{ep_reward:.2f}，Epislon：{agent.epsilon:.3f}")
    print("完成訓練！")
    env.close()
    return {'rewards':rewards}

def test(cfg, env, agent):
    print("開始測試！")
    rewards = []  # 記錄所有回合的獎勵
    steps = []
    for i_ep in range(cfg['test_eps']):
        ep_reward = 0  # 記錄一回合內的獎勵
        ep_step = 0
        state, info = env.reset()  # 重置環境，返回初始狀態
        for _ in range(cfg['ep_max_steps']):
            ep_step+=1
            action = agent.predict_action(state)  # 選擇動作
            next_state, reward, done, _, info = env.step(action)  # 更新環境，返回transition
            state = next_state  # 更新下一個狀態
            ep_reward += reward  # 累加獎勵
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg['test_eps']}，獎勵：{ep_reward:.2f}")
    print("完成測試")
    env.close()
    return {'rewards':rewards}

def run(env, agent):
    print("開始展示 run！")
    rewards = []  # 記錄所有回合的獎勵
    steps = []
    for i_ep in range(5):
        ep_reward = 0  # 記錄一回合內的獎勵
        ep_step = 0
        state, info = env.reset()  # 重置環境，返回初始狀態
        for _ in range(100000):
            env.render()
            ep_step+=1
            action = agent.predict_action(state)  # 選擇動作
            next_state, reward, done, _, info = env.step(action)  # 更新環境，返回transition
            state = next_state  # 更新下一個狀態
            ep_reward += reward  # 累加獎勵
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg['test_eps']}，獎勵：{ep_reward:.2f}")
    print("完成執行")
    env.close()
    return {'rewards':rewards}

import gymnasium as gym
import os

def all_seed(env,seed = 1): # 設定亂數種子 seed
    env.action_space.seed(seed) # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def env_agent_config(cfg):
    env = gym.make(cfg['env_name']) # 創建環境
    if cfg['seed'] !=0:
        all_seed(env,seed=cfg['seed'])
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"狀態空間維度：{n_states}，動作空間維度：{n_actions}")
    cfg.update({"n_states":n_states,"n_actions":n_actions}) # 更新n_states和n_actions到cfg參數中
    model = MLP(n_states, n_actions, hidden_dim = cfg['hidden_dim']) # 創建模型
    memory = ReplayBuffer(cfg['memory_capacity'])
    agent = DQN(model,memory,cfg)
    return env,agent

import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def get_args(): # 超參數
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='DQN',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='CartPole-v1',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=200,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--ep_max_steps',default = 100000,type=int,help="steps per episode, much larger value can simulate infinite steps")
    parser.add_argument('--gamma',default=0.95,type=float,help="discounted factor")
    parser.add_argument('--epsilon_start',default=0.95,type=float,help="initial value of epsilon")
    parser.add_argument('--epsilon_end',default=0.01,type=float,help="final value of epsilon")
    parser.add_argument('--epsilon_decay',default=500,type=int,help="decay rate of epsilon, the higher value, the slower decay")
    parser.add_argument('--lr',default=0.0001,type=float,help="learning rate")
    parser.add_argument('--memory_capacity',default=100000,type=int,help="memory capacity")
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--target_update',default=4,type=int)
    parser.add_argument('--hidden_dim',default=256,type=int)
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda or mps") 
    parser.add_argument('--seed',default=10,type=int,help="seed")   
    args = parser.parse_args([])
    args = {**vars(args)}  # 轉換成字典類型    
    ## 打印超參數
    print("超參數")
    print(''.join(['=']*80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("Name", "Value", "Type"))
    for k,v in args.items():
        print(tplt.format(k,v,str(type(v))))   
    print(''.join(['=']*80))      
    return args

def smooth(data, weight=0.9): # 用於平滑曲線，類似於Tensorboard中的smooth曲線
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 計算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,cfg, tag='train'): # 畫折線圖
    sns.set()
    plt.figure()  # 創建一個圖形實例，方便同時多畫幾個圖
    plt.title(f"{tag}ing curve on {cfg['device']} of {cfg['algo_name']} for {cfg['env_name']}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.show()

# 獲取參數
cfg = get_args() 

# 訓練
env, agent = env_agent_config(cfg)
res_dic = train(cfg, env, agent) 
plot_rewards(res_dic['rewards'], cfg, tag="train")  

# 測試
res_dic = test(cfg, env, agent)
plot_rewards(res_dic['rewards'], cfg, tag="test")

# 展示
env = gym.make(cfg['env_name'], render_mode="human")
run(env, agent)
