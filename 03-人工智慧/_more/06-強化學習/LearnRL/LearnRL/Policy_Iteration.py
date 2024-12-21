# from -- https://github.com/ChampDBG/LearnRL/blob/master/Policy_Iteration.py
# note -- 
## package
import numpy as np
import time, os

## define function
def PolicyEvalution(func_value, best_action, func_reward, trans_mat, gamma):
    func_value_now = func_value.copy()
    for state in range(1,15):
        next_state = trans_mat[:, state, best_action[state]]
        future_reward = func_reward + func_value_now*gamma
        func_value[state] = np.sum(next_state*future_reward)
    delta = np.max(np.abs(func_value - func_value_now))
    return func_value, delta

def ShowValue(delta, theta, gamma, counter_total, counter, func_value):
    print('='*60)
    print('No. ' + str(counter_total) + ' Policy Evaluation')
    print('='*60)
    print('[Parameters]')
    print('Gamma = ' + str(gamma))
    print('Threshold = ' + str(theta) + '\n')
    print('[Variables]')
    print('No.' + str(counter) + ' iteration')
    print('Delta = ' +str(delta) + '\n')
    print('[State-Value]')
    print(func_value.reshape(4,4))
    print('='*60)

def PolicyImprovement(func_value, best_action, prob_action, func_reward, trans_mat, gamma):
    policy_stable = False
    best_action_now = best_action.copy()
    for state in range(1,15):
        prob_next_state = prob_action[state]*trans_mat[:,state,:]
        future_reward = func_reward + func_value*gamma
        best_action[state] = np.argmax(np.matmul(np.transpose(prob_next_state), future_reward))
    if np.all(best_action == best_action_now):
        policy_stable = True
    return best_action, policy_stable

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
    print('='*60)
    print('No. ' + str(counter_total) + ' Policy Improvement')
    print('='*60)
    print('[Policy]')
    print(policy_string.reshape(4,4))
    print('='*60)
    return policy_string

# main function
def main():
    ## environment setting
    # action
    BestAction = np.random.randint(0,4,16)
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
    counter_total = 0
    PolicyStable = False

    # iteration
    while not PolicyStable:
        delta = theta + 0.001
        counter = 1
        counter_total += 1
        while delta > theta:
            FuncValue, delta = PolicyEvalution(FuncValue, BestAction, FuncReward, T, gamma)
            counter += 1
        os.system('cls' if os.name == 'nt' else 'clear')
        ShowValue(delta, theta, gamma, counter_total, counter, FuncValue)
        time.sleep(2)
        BestAction, PolicyStable = PolicyImprovement(FuncValue, BestAction, ProbAction, FuncReward, T, gamma)
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


