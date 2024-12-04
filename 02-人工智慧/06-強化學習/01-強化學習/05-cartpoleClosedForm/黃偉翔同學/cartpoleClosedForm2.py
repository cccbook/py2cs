# https://github.com/weixiang0470/ai112b/blob/master/Homework/hw08/cartpole1.py

import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human") # 若改用這個，會畫圖
# env = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset(seed=42)
steps = 0
#action = 0
for _ in range(2000):
   env.render()

   #action = env.action_space.sample()  # 把這裡改成你的公式，看看能撐多久
   if observation[2] > 0 : 
    if observation[3] > 0.01 :
        action = 1 
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
    else : 
       action = 0
       observation, reward, terminated, truncated, info = env.step(action)
       steps += 1
   elif observation[2] < 0 : 
    if observation[3] < -0.01 :
        action = 0 
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
    else : 
       action = 1
       observation, reward, terminated, truncated, info = env.step(action)
       steps += 1
   
   if terminated or truncated: # 這裡要加入程式，紀錄你每次撐多久
      observation, info = env.reset()
      print('steps:',steps)
      steps = 0

env.close()