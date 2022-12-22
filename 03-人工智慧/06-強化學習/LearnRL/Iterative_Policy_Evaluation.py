# packages
import numpy as np
import time, os

# environment setting
prob_action = np.full(4,0.25)
func_value = np.zeros(16)
func_reward = np.full(16,-1)
func_reward[0] = 0
func_reward[15] = 0
num_actions = 4
num_states = 16
T = np.load('./gridworld/T.npy')

# parameters
gamma = 0.99
theta = 0.05
delta = theta + 0.001
counter = 1

# iterativa policy evaluation
while delta > theta:
    func_value_now = func_value.copy()
    for state in range(1,15):
        prob_next_state = prob_action*T[:, state, :]
        future_reward = func_reward + func_value_now*gamma
        func_value[state] = np.sum(np.matmul(np.transpose(prob_next_state), future_reward))
    delta = np.max(np.abs(func_value - func_value_now))
    os.system('cls' if os.name == 'nt' else 'clear')
    print('='*60)
    print('[Parameters]')
    print('Gamma = ' + str(gamma))
    print('Threshold = ' + str(theta) + '\n')
    print('[Variables]')
    print('No.' + str(counter) + ' iteration')
    print('Delta = ' + str(delta) + '\n')
    print('[State-Value]')
    print(func_value.reshape(4,4))
    print('='*60)
    counter += 1
    time.sleep(1)

