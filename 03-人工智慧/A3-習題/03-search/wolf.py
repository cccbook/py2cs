name = ['人', '狼', '羊', '菜']
state = [0,0,0,0]

def neighbors(s):
    side = s[0]
    neighbor = []
    checkadd(neighbor,move(s,0))
    for i in range(1,len(s)):
        if(s[i]==side):
            checkadd(neighbor,move(s,i))
    return neighbor

def checkadd(neighbor,s):
    if not isDead(s):
        neighbor.append(s)

def isDead(s):
    if s[1]==s[2] and s[1]!=s[0]: #狼吃羊
        return True
    if s[2]==s[3] and s[2]!=s[0]: #羊吃菜=狼吃人
        return True
    return False

def move(s,obj):
    copyS = s.copy()
    side = s[0]
    anotherSide = 0 if side else 1 #當anotherSide=0 side=1 表示取相反的值
    copyS[0] = anotherSide
    copyS[obj] = anotherSide
    return copyS

visitedMap={}

def visited(s):
    return visitedMap.get(str(s)) != None

def isSuccess(s):
    for i in range(len(s)):
        if s[i]==0:
            return False
    return True

def state2str(s):
    rstr = ""
    for i in range(len(s)):
        rstr += name[i] + str(s[i]) + " "
    return rstr

path = []

def printPath(path):
    for i in range(len(path)):
        print(state2str(path[i]))

def dfs(s):
    if visited(s):
        return
    path.append(s)
    if isSuccess(s):
        print("success!")
        printPath(path)
        return

    visitedMap[str(s)] = True
    neighborsList = neighbors(s)
    for i in range(len(neighborsList)):
        dfs(neighborsList[i])
    path.pop()

dfs(state)
