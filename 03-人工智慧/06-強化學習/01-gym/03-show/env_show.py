import gymnasium as gym
import time
import sys
env = gym.make(sys.argv[1], render_mode="human")
env.reset()
env.render()
time.sleep(2)
env.close()