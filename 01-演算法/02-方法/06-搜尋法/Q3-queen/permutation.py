def perm(n): # 主函數
	p = [] # p 代表已經排下去的，一開始還沒排，所以是空的
	return permNext(n, p) # 呼叫 permNext 遞迴下去排出所有可能

# 思考：排到一半 p=[1,3] 繼續排下去
#      對 [1,3,0..], [1,3,1..], [1,3,2..], [1,3,3..] 全部試一遍
def permNext(n, p): # 已經排好 p[0..i-1], 繼續排下去 [i...n-1]
	i = len(p)
	if i == n:  # 全部排好了
		print(p)  # 印出排列
		return
	# 還沒排滿，繼續排下去
	for x in range(n): # 對本列的每一個 x 去嘗試
		p.append(x)    # 把 x 放進盤面
		permNext(n, p) # 繼續遞迴尋找下一個排列
		p.pop()        # 把 x 移出盤面
		
perm(2) # 列出 2 個的排列
perm(3) # 列出 3 個的排列
