from stateSearch import State, bfs

class StateBoard(State):
	def __init__(self):
		pass
	
	def __str__(self):
		rows = []
		for row in b:
			rows.append(str(row))
		return '\n'.join(rows)

start= [[1,3,4], 
        [8,2,5],
        [7,0,6]]

goal = [[1,2,3], 
        [8,0,4],
        [7,6,5]]

queue=[start] # BFS 用的 queue, 起始點為 1。
visited={}
parent={}
level={}
level[str(start)]=0
found = bfs(queue, goal, neighbors) #  呼叫廣度優先搜尋。
print('bfs:found=', found)
if found:
    backtrace(goal)
