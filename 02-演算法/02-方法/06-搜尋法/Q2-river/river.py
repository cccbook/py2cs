import sys

role  = ['人','狼','羊','菜']

def isDead(state):
    if state[1]==state[2] and state[0] != state[1]:
        return True
    if state[2]==state[3] and state[0] != state[2]:
        return True
    return False

def neighbors(state):
    peopleSide = state[0]
    nbList = []

    side2 = 1 if peopleSide==0 else 0
    p2state = state.copy()
    p2state[0] = side2 # 人移動到另一邊
    nbList.append(p2state) # 要考慮只有自己移動

    for i in range(1, len(state)):
        if state[i] == peopleSide:
            nbState = state.copy()
            nbState[0], nbState[i] = side2, side2
            nbList.append(nbState)
    return nbList

def dfs(state, role, visitedMap, goal, path):
    if isDead(state):return
    
    stateStr = ''.join(str(x) for x in state)
    if visitedMap.get(stateStr): return
    visitedMap[stateStr] = True

    print(state)
    path.append(state)
    if state == goal:
        print(f'{path}')
        sys.exit(1)
        return

    for nb in neighbors(state):
        dfs(nb, role, visitedMap, goal, path)

    path.pop()


start = [0,0,0,0]
goal = [1,1,1,1]
visitedMap = {}
path = []
print('start=', start)
dfs(start,role,visitedMap, goal, path)

