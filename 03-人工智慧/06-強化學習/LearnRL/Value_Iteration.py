## 價值迭代
## from -- https://github.com/ChampDBG/LearnRL/blob/master/Value_Iteration.py
## Note -- https://ithelp.ithome.com.tw/articles/10203369
## package
import numpy as np
import time, os

## define function
def ValueIteration(func_value, func_reward, trans_mat, gamma):
    best_action = np.zeros(16)
    func_value_now = func_value.copy()
    for state in range(1,15):
        next_state = trans_mat[:,state,:]
        future_reward = func_reward + func_value*gamma
        func_value[state] = np.max(np.matmul(np.transpose(next_state), future_reward))
        best_action[state] = np.argmax(np.matmul(np.transpose(next_state), future_reward))
    delta = np.max(np.abs(func_value - func_value_now))
    return func_value, delta, best_action

def ShowValue(delta, theta, gamma, counter_total, func_value):
    print('='*60)
    print('No. ' + str(counter_total) + ' Value Iteration')
    print('='*60)
    print('[Parameters]')
    print('Gamma = ' + str(gamma))
    print('Threshold = ' + str(theta) + '\n')
    print('[Variables]')
    print('Delta = ' +str(delta) + '\n')
    print('[State-Value]')
    print(func_value.reshape(4,4))
    print('='*60)

def ShowPolicy(counter_total, best_action):
    policy_string = []
    policy_string.append('*')
    for i in range(1,15):
        if best_action[i] == 0:
            policy_string.append('^')
        elif best_action[i] == 1:
            policy_string.append('<')
        elif best_action[i] == 2:
            policy_string.append('v')
        elif best_action[i] == 3:
            policy_string.append('>')
    policy_string.append('*')
    policy_string = np.array(policy_string)
    print('[Policy]')
    print(policy_string.reshape(4,4))
    print('='*60)
    return policy_string

# main function
def main():
    ## environment setting
    # action
    ProbAction = np.zeros([16,4])
    ProbAction[1:15,:] = 0.25
    # value function
    FuncValue = np.zeros(16)
    # reward function
    FuncReward = np.full(16,-1)
    FuncReward[0] = 0
    FuncReward[15] = 0
    # transition matrix
    T = np.load('./gridworld/T.npy')

    # parameters
    gamma = 0.99
    theta = 0.05
    delta = delta = theta + 0.001
    counter_total = 0

    # iteration
    while delta > theta:
        counter_total += 1
        os.system('cls' if os.name == 'nt' else 'clear')
        ValueFunc, delta, BestAction = ValueIteration(FuncValue, FuncReward, T, gamma)
        ShowValue(delta, theta, gamma, counter_total, FuncValue)
        PolicyString = ShowPolicy(counter_total, BestAction)
        time.sleep(2)

    os.system('cls' if os.name == 'nt' else 'clear')
    print('='*60)
    print('Final Result')
    print('='*60)
    print('[State-value]')
    print(FuncValue.reshape(4,4))
    print('='*60)
    print('[Policy]')
    print(PolicyString.reshape(4,4))
    print('='*60)

## execute
if __name__ == '__main__':
    main()