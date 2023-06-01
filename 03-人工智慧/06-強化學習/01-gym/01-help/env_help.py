import gymnasium as gym
import sys
env = gym.make(sys.argv[1], render_mode="rgb_array")
help(env.unwrapped)
