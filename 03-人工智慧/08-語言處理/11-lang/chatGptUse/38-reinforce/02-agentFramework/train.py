import gym
from agent import Agent

# 創建「CartPole-v1」環境
environment = gym.make("CartPole-v1")

# 設置智能代理
agent = Agent()

# 迴圈，表示智能代理不斷對環境進行探索和測試
MAX_ROUND = 5
for rounds in range(MAX_ROUND):
  # 獲取當前環境狀態
  state, info = environment.reset()
  loops = 0
  # 迴圈，表示智能代理在當前狀態下不斷做出決策
  while True:
    # 智能代理做出決策
    action = agent.make_decision(state)
    
    # 採取行動並獲取新狀態、獎勵和終止信息
    next_state, reward, terminated, truncated, info = environment.step(action)
    done = (loops > 10000) or terminated or truncated
    print('loops:', loops, 'next_state=', next_state, 'reward=', reward, 'done=', done)
    loops += 1
    # 記錄智能代理的行動和結果
    agent.record_results(state, action, next_state, reward, done)
    
    # 更新智能代理的學習
    agent.update_learning()
    
    # 如果環境已終止，退出迴圈
    if done:
      print('==> done: break')
      break
      
    # 更新狀態
    state = next_state
