import gymnasium as gym
import time
env = gym.make('MountainCar-v0', render_mode="human")
env.reset()
env.render()
time.sleep(2)
env.close()