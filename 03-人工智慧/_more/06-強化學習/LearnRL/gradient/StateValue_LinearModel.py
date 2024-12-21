## ref -- https://ithelp.ithome.com.tw/articles/10208788
import numpy as np
import os, time

## define function
def ValueFunction(state, weights):
    state_array = np.zeros([1,16])
    state_array[:, state] = 1
    value = np.matmul(state_array, np.transpose(weights))
    return value

def SimProc(reward, weights, steps, gamma, alpha, trace_lambda, trans_mat):
    state = np.random.randint(1,15)
    trace = 0
    for step in range(steps):
        action = np.random.randint(0,4)
        next_state = np.argmax(trans_mat[:, state, action])
        record = [state, action, reward[next_state], next_state]
        # update
        trace, weights = Update(weights, trace, record, alpha, gamma, trace_lambda)
        state = next_state
    return weights

def Update(weights, trace, records, alpha, gamma, trace_lambda):
    state = records[0]
    aciton = records[1]
    reward = records[2]
    next_state = records[3]
    # update value, trace, and state
    delta = reward + gamma*ValueFunction(next_state, weights) - ValueFunction(state, weights)
    trace = gamma*trace_lambda*trace + Gradient(state)
    weights += alpha*delta*trace
    return trace, weights

def Gradient(state):
    # gradient of linear model is state
    state_array = np.zeros([1,16])
    state_array[:, state] = 1
    return state_array

def PrintValue(weights, now_episode):
    value = np.zeros((16,1))
    for i in range(1,15):
        value[i] = ValueFunction(i, weights)
    os.system('cls' if os.name == 'nt' else 'clear')
    print('='*60)
    print('[State-Value]')
    print('Episode: ' + str(now_episode + 1))
    print(value.reshape(4,4))
    print('='*60)

# main
def main(Epidoses):
    # environment setting
    Weights = np.zeros([1,16])
    Reward = np.full(16, -1)
    Reward[0] = 0
    Reward[-1] = 0
    TransMat = np.load('./gridworld/T.npy')
    # parameters
    Steps = 50
    Gamma = 0.99
    Alpha = 0.05
    TraceLambda = 0.5

    # execute
    for episode in range(Epidoses):
        Weights = SimProc(Reward, Weights, Steps, Gamma, Alpha, TraceLambda, TransMat)
        PrintValue(Weights, episode)

if __name__ == '__main__':
    main(200)