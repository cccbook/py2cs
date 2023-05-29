import gym

# 創建環境
env = gym.make('CartPole-v0')

# 重置環境，得到起始狀態
observation = env.reset()

# 環境運行，最多執行 100 步
for t in range(100):
    env.render()
    action = env.action_space.sample()  # 隨機選擇一個動作
    r = env.step(action)  # 執行動作並返回環境的下一個狀態和相應的回饋信號
    print('r=', r)
    observation, reward, terminated, truncated, info = env.step(action)  # 執行動作並返回環境的下一個狀態和相應的回饋信號
    if terminated or truncated:
        print("Episode finished after {} timesteps".format(t+1))
        break
