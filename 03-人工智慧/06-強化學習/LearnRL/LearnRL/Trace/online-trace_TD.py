import numpy as np
import os, time

## define function
def SimProc(state_value, state_trace, steps, gamma, alpha, trace_lambda, reward, trans_mat):
    state = np.random.randint(1,15)
    action = np.random.randint(0,4)
    for step in range(steps):
        # get next infromation
        next_state = np.argmax(trans_mat[:, state, action])
        next_action = np.random.randint(0,4)
        record = [state, action, reward[next_state], next_state]
        # update value, infromation and decay trace
        state_value, state_trace = Update(state_value, state_trace, record, alpha, gamma, trace_lambda)
        state = next_state
        action = next_action
        if state == 0 or state == 15:
            break
    return state_value

def Update(state_value, state_trace, records, alpha, gamma, trace_lambda):
    # calculate value update
    state = records[0]
    action = records[1]
    reward = records[2]
    next_state = records[3]
    delta = reward + gamma*state_value[next_state] - state_value[state]
    # update trace matrix
    state_trace[state] += 1
    # update state-value and decay trace
    state_value += alpha*delta*state_trace
    state_trace = gamma*trace_lambda*state_trace
    return state_value, state_trace

def PrintValue(state_value, now_episode):
    os.system('cls' if os.name == 'nt' else 'clear')
    print('='*60)
    print('[State-Value]')
    print('Episode: ' + str(now_episode + 1))
    print(state_value.reshape(4,4))
    print('='*60)

def main(Episodes):
    # environment setting
    StateValue = np.zeros([16])
    Reward = np.full(16, -1)
    Reward[0] = 0
    Reward[-1] = 0
    TransMat = np.load('./gridworld/T.npy')
    # parameters
    GAMMA = 0.99
    ALPHA = 0.05
    STEPS = 50
    TRACE_LAMBDA = 0.5

    # execute
    for episode in range(Episodes):
        StateTrace = np.zeros([16])
        StateValue = SimProc(StateValue, StateTrace, STEPS, GAMMA, ALPHA, TRACE_LAMBDA, Reward, TransMat)
        PrintValue(StateValue, episode)
        #time.sleep(1)

if __name__ == '__main__':
    main(500)