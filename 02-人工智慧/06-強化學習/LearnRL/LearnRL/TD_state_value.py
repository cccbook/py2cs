# TD learning
# ref -- https://ithelp.ithome.com.tw/articles/10206921
import numpy as np
import os, time

## define function
def SimProc(init_state, init_action, state_value, steps, gamma, alpha, reward, trans_mat):
    state = init_state
    action = init_action
    for step in range(steps):
        # get next infromation
        next_state = np.argmax(trans_mat[:, state, action])
        next_action = np.random.randint(0,4)
        record = [state, next_state, action, reward[next_state]]
        # update state value
        state_value[state] = GetValue(state_value, record, gamma, alpha)
        # update infromation
        state = next_state
        action = next_action
        if state == 0 or state == 15:
            break
    return state_value

def GetValue(state_value, records, gamma, alpha):
    state = records[0]
    next_state = records[1]
    action = records[2]
    reward = records[3]
    value = state_value[state] + alpha*(reward + gamma*state_value[next_state] - state_value[state])
    return value

def PrintValue(state_value, now_episode):
    os.system('cls' if os.name == 'nt' else 'clear')
    print('='*60)
    print('[State-Value]')
    print('Episode: ' + str(now_episode + 1))
    print(state_value.reshape(4,4))
    print('='*60)

def main(InitState, InitAction, Episodes):
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

    # execute
    for episode in range(Episodes):
        StateValue = SimProc(InitState, InitAction, StateValue, STEPS, GAMMA, ALPHA, Reward, TransMat)
        PrintValue(StateValue, episode)
        #time.sleep(1)

if __name__ == '__main__':
    main(np.random.randint(1,15), np.random.randint(0,4), 1000)

