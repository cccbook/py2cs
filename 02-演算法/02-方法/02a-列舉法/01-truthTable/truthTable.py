def truthTable(n): # 列出 n 變數的所有可能 0,1 排列
	binary = [] # binary 代表已經排下去的，一開始還沒排，所以是空的
	return tableNext(n, binary) # 呼叫 tableNext 遞迴下去排出所有可能

def tableNext(n, binary):
	i = len(binary)      # i 是下一個排列的位置
	if i == n:		# 全部排好了
		print(binary)	# 印出排列
		return      # 返回上層
	for x in [0,1]:     # x 是 0 或 1
		binary.append(x)		# 把 x 放進表
		tableNext(n, binary)	# 繼續遞迴尋找下一個排列
		binary.pop()			# 把 x 移出表

truthTable(2) # 印出 2 變數的真值表
truthTable(3) # 印出 3 變數的真值表
truthTable(4) # 印出 4 變數的真值表
