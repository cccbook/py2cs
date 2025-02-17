## ref -- https://ithelp.ithome.com.tw/articles/10207744
## packages
import numpy as np
import os, time

## define function
def SimProc(action_value, reward, trans_mat, steps, gamma, alpha, epsilon):
    # initialize setting
    record = []
    state = np.random.randint(1,15)
    for step in range(steps):
        # get next infromation
        action = GetAction(action_value, epsilon, state)
        next_state = np.argmax(trans_mat[:,state,action])
        record.append([state, action, reward[next_state], next_state])
        # update action value
        action_value[state, action] = ValueUpdate(action_value, record[step], alpha, gamma)
        # update for next step
        state = next_state
        if state == 0 or state == 15:
            break
    return action_value

def GetAction(action_value, epsilon, next_state):
    if np.random.rand(1) >= epsilon:
        policy = np.argmax(action_value, axis = 1)
        action = policy[next_state]
    else:
        action = np.random.randint(0,4,1)
    return action

def ValueUpdate(action_value, record, alpha, gamma):
    state = record[0]
    action = record[1]
    reward = record[2]
    next_state = record[3]
    now_value = action_value[state, action]
    update_value = alpha*(reward + gamma*np.max(action_value[next_state,:]) - now_value)
    value = now_value + update_value
    return value

def PrintGreedyPolicy(now_episode, action_value):
    policy = np.argmax(action_value, axis = 1)
    policy_string = []
    policy_string.append('*')
    for i in range(1,15):
        if policy[i] == 0:
            policy_string.append('^')
        elif policy[i] == 1:
            policy_string.append('<')
        elif policy[i] == 2:
            policy_string.append('v')
        elif policy[i] == 3:
            policy_string.append('>')
    policy_string.append('*')
    policy_string = np.array(policy_string)
    os.system('cls' if os.name == 'nt' else 'clear')
    print('='*60)
    print('[Greedy Policy]')
    print('Episode: ' + str(now_episode+1))
    print(policy_string.reshape(4,4))
    print(np.max(action_value, axis = 1).reshape(4,4))
    print('='*60)

## main function
def main(Episodes):
    # Environment setting
    ActionValue = np.zeros([16,4])
    Reward = np.full(16, -1)
    Reward[0] = 0
    Reward[-1] = 0
    TransMat = np.load('./gridworld/T.npy')
    # parameters setting
    Gamma = 0.99
    Steps = 50
    Alpha = 0.05

    # Execute
    for episode in range(Episodes):
        Epsilon = 1/(episode+1)
        ActionValue = SimProc(ActionValue, Reward, TransMat, Steps, Gamma, Alpha, Epsilon)
        PrintGreedyPolicy(episode, ActionValue)
        #time.sleep(1)

if __name__ == '__main__':
    main(1000)
