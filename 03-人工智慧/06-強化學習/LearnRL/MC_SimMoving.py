import numpy as np
import time, os

def Simulating(state, gamma, reward, trans_mat):
    value = 0
    counter = 0
    terminal = False
    ShowPath(state)
    time.sleep(1)
    while not terminal:
        action = np.random.randint(0,4)
        next_state = np.argmax(trans_mat[:, state, action])
        value += reward[next_state]*pow(gamma, counter)
        counter += 1
        state = next_state
        ShowPath(next_state)
        print('step: ' + str(counter) + ',action: ' + str(action) +', state: ' +str(state) + ', value: ' + str(value))
        if state == 0 or state == 15:
            terminal = True
        if counter > 50:
            return counter, terminal, value
        time.sleep(1)
    return counter, terminal, value

def ShowPath(state):
    os.system('cls' if os.name == 'nt' else 'clear')
    position = np.full(16,'_')
    position[state] = '*'
    print('='*20)
    print('[Now State]')
    position[state] = '*'
    print(position.reshape(4,4))
    print('='*20)

def main():
    ## environment setting
    # initial state
    InitState = 1
    # reward function
    FuncReward = np.full(16,-1)
    FuncReward[0] = 0
    FuncReward[15] = 0
    # transition matrix
    T = np.load('./gridworld/T.npy')
    # parameters
    gamma = 0.99
    # Run
    Simulating(InitState, gamma, FuncReward, T)

## execute
if __name__ == '__main__':
    main()
