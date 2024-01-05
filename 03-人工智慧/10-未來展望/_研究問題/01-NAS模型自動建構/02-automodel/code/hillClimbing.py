import sys

def log(s):
	file.write(s+"\n")
	print(s)
	file.flush()

def hillClimbing(s, max_gens, max_fails):   # 爬山演算法的主體函數
	global file
	file = open("./model/hillClimbing.log", "w") # 開啟紀錄檔案
	log(f"start: {str(s)}")               # 印出初始解
	fails = 0                             # 失敗次數設為 0
	for gens in range(max_gens):          # 當 gen<maxGen，就持續嘗試尋找更好的解
		snew = s.neighbor()               #  取得鄰近的解
		sheight = s.height()              #  sheight=目前解的高度
		nheight = snew.height()           #  nheight=鄰近解的高度
		if (nheight > sheight):           #  如果鄰近解比目前解更好
			log(f"{gens}:{str(snew)}")    #    印出新的解
			s = snew                      #    就移動過去
			fails = 0                     #    移動成功，將連續失敗次數歸零
		else:                             #  否則
			fails = fails + 1             #    將連續失敗次數加一
		if (fails >= max_fails):	      #  連續失敗次數 fails > maxFails 時離開
			log(f"fail {fails} times!")
			break
	log(f"solution: {str(s)}")            #  印出最後找到的那個解
	file.close()
	return s                              #    然後傳回。
