class State:
	def neighbors():
	def __str__():

def dfs(state, visited):            #  深度優先搜尋
    if visited[state]!=0:           #  如果已訪問過，就不再訪問
        return
    visited[state] = 1              #    並設定為已訪問
    for n in state.neighbors():     #  對於每個鄰居
        dfs(n, visited)             #    逐一進行訪問

def bfs(q, goal, neighbors): # 廣度優先搜尋
	parent = {}
	level = {}
    while len(q) > 0:
        qt = dequeue(q) #  否則、取出 queue 的第一個節點。
        qts = str(qt)
        if qt == goal: return True
        if visited.get(qts) == None: #  如果該節點尚未拜訪過。
            visited[qts] = True      #    標示為已拜訪
        else:                        #  否則 (已訪問過)
            continue                 #    不繼續搜尋，直接返回。
        for qn in neighbors(qt):              #  對於每個鄰居
			qns = str(qn)
            if visited.get(qns) == None:#  假如該鄰居還沒被拜訪過
                parent[qns] = qts
                level[qns] = level[qns] + 1
                enqueue(q, qn)            # 就放入 queue 中
    return False

