## packages
import numpy as np
import os, time

## define function
def SimProc(action_value, action_trace, reward, trans_mat, steps, gamma, alpha, epsilon, trace_lambda):
    # initialize setting
    record = []
    state = np.random.randint(1,15)
    action = GetAction(action_value, epsilon, state)
    for step in range(steps):
        # get next information
        next_state = np.argmax(trans_mat[:, state, action])
        next_action = GetAction(action_value, epsilon, next_state)
        record.append([state, action, reward[next_state], next_state, next_action])
        # update action value
        action_value, action_trace = Update(action_value, action_trace, record[step], alpha, gamma, trace_lambda)
        state = next_state
        action = next_action
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

def Update(action_value, action_trace, records, alpha, gamma, trace_lambda):
    # information
    state = records[0]
    action = records[1]
    reward = records[2]
    next_state = records[3]
    next_action = records[4]
    # update parameters
    delta = reward + gamma*action_value[next_state, next_action] - action_value[state, action]
    action_trace[state, action] += 1
    # update action-value and decay trace
    action_value += alpha*delta*action_trace
    action_trace = gamma*trace_lambda*action_trace
    return action_value, action_trace

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
    Trace_Lambda = 0.5

    # Execute
    for episode in range(Episodes):
        Epsilon = 1/(episode+1)
        ActionTrace = np.zeros([16,4])
        ActionValue = SimProc(ActionValue, ActionTrace, Reward, TransMat, Steps, Gamma, Alpha, Epsilon, Trace_Lambda)
        PrintGreedyPolicy(episode, ActionValue)
        #time.sleep(1)

if __name__ == '__main__':
    main(1000)