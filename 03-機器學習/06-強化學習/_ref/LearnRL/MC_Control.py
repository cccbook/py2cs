## 蒙地卡羅控制
## ref -- https://ithelp.ithome.com.tw/articles/10205995
## packages
import numpy as np
import os, time

## define function
def SimProc(init_state, init_action, policy, steps, gamma, reward, trans_mat):
    record = []
    state = init_state
    action = init_action
    for step in range(steps):
        # get next information
        next_state = np.argmax(trans_mat[:, state, action])
        next_action = policy[next_state]
        record.append([state, action, reward[next_state]])
        # update information
        state = next_state
        action = next_action
        if state == 0 or state == 15:
            break
        if step > 50:
            print('Over 50 steps')
    return record

def GetValue(records, gamma):
    counter = 0
    value = 0
    for record in records:
        reward = record[-1]
        value += reward*pow(gamma, counter)
        counter += 1
    return value

def EpisodeValue(records, gamma):
    episode_visited = np.zeros([16, 4])
    episode_value = np.zeros([16, 4])
    for counter in range(len(records)):
        state = records[counter][0]
        action = records[counter][1]
        if episode_visited[state, action] == 1:
            continue
        episode_value[state, action] = GetValue(records[counter:], gamma)
        episode_visited[state, action] = 1
    return episode_value, episode_visited

def UpdateActionValue(episode_value, episode_visited, action_value, total_visited):
    total_value = (action_value*total_visited) + episode_value
    total_visited += episode_visited
    rst = total_value / total_visited
    action_value = np.nan_to_num(rst)
    return action_value, total_visited

def NextPolicy(action_value, epsilon):
    if np.random.rand(1) < epsilon:
        policy = np.random.randint(0,4,16)
        explore = True
    else:
        policy = np.argmax(action_value, axis = 1)
        explore = False
    return policy, explore

def PrintPolicy(now_episode, policy, explore):
    if explore:
        PolicyType = 'Explore'
    else:
        PolicyType = 'Greedy'
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
    print('[Policy]')
    print('Policy Type: ' + PolicyType)
    print('Episode: ' + str(now_episode+1))
    print(policy_string.reshape(4,4))
    print('='*60)

## main function
def main(Episodes, InitState, InitAction, InitEpsilon):
    # environment setting
    Policy = np.random.randint(0, 4, 16)
    ActionValue = np.zeros([16, 4])
    TotalVisited = np.zeros([16, 4])
    Reward = np.full(16, -1)
    Reward[0] = 0
    Reward[-1] = 0
    TransMat = np.load('./gridworld/T.npy')
    # parameters setting
    GAMMA = 0.99
    Steps = 50

    for episode in range(Episodes):
        EPSILON = InitEpsilon - InitEpsilon*(episode/Episodes)
        Records = SimProc(InitState, InitAction, Policy, Steps, GAMMA, Reward, TransMat)
        NowEpisodeValue, NowEpisodeVisited = EpisodeValue(Records, GAMMA)
        ActionValue, TotalVisited = UpdateActionValue(NowEpisodeValue, NowEpisodeVisited,
                ActionValue, TotalVisited)
        Policy, Explore = NextPolicy(ActionValue, EPSILON)
        PrintPolicy(episode, Policy, Explore)
        time.sleep(1)

        # next simulating InitState and InitAction
        InitState = np.random.randint(1,16)
        InitAction = Policy[InitState]

if __name__ == '__main__':
    main(1000, 3, 1, 0.5)
