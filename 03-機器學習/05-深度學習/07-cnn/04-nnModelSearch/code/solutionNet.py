from net import Net
import random
import trainer
import copy

types = ["ReLU", "Linear", "Conv2d", "AvgPool2d", "LinearReLU", "ConvPool2d"]
sizes = [ 8, 16, 32, 64, 128, 256 ] # 限縮大小選取範圍，避免太多鄰居
channels = [ 1, 2, 4, 8, 16, 32 ]   # 限縮通道數範圍，所以不是所有整數都可以

def randomLayer():
	type1 = random.choice(types)             # 隨機選一種層次
	if type1 in ["Linear", "LinearReLU"]:    # 如果是 Linear 或 LinearReLU
		k = random.choice(sizes)             # 那麼就隨機選取 k 參數作為輸出節點數 out_features
		return {"type":type1, "out_features":k}
	elif type1 in ["Conv2d", "ConvPool2d"]:  # 如果是 Conv2d 或 ConvPool2d
		out_channels = random.choice(channels) # 那麼就隨機選取 channels 數量
		return {"type":type1, "out_channels": out_channels}
	else:                                    
		return {"type":type1}                # 否則不須設定參數，直接傳回該隨機層。

types2d = ["Conv2d", "ConvPool2d", "AvgPool2d", "Flatten"]
types1d = ["Linear", "LinearReLU"]

def compatable(in_shape, newLayerType):
	if newLayerType in ["ReLU"]: # 任何維度都可以使用 ReLU 操作
		return True
	elif len(in_shape) == 4 and newLayerType in types2d:
		# 這些層的輸入必須是 4 維的 (1. 樣本數 2. 通道數 3. 寬 4. 高)
		return True # 
	elif len(in_shape) == 2 and newLayerType in types1d:
		# 這些層的輸入必須是 2 維 (1. 樣本數 2. 輸出節點數)
		return True
	return False

class SolutionNet:
	def __init__(self, net):
		self.net = net

	def neighbor(self):
		model = copy.deepcopy(self.net.model)    # 複製模型
		layers = model["layers"]                 # 取得網路層次
		in_shapes = self.net.in_shapes           # 取得各層次的輸入形狀
		ops = ["insert", "update"]               # 可用的操作有新增和修改
		success = False
		while not success:                       # 直到成功產生一個合格鄰居為止
			i = random.randint(0, len(layers)-1) # 隨機選取第 i 層 (進行修改或新增)
			layer = layers[i]
			op = random.choice(ops)              # 隨機選取操作 (修改或新增)
			newLayer = randomLayer()             # 隨機產生一個網路層
			if not compatable(in_shapes[i], newLayer["type"]): # 若新層不相容 (輸入維度不對)
				continue                         #   那麼就重新產生
			if op == "insert":                   # 如果是新增操作
				layers.insert(i, newLayer)       #   就插入到第 i 層之後
			elif op == "update":                 # 如果是修改操作
				if layers[i]["type"] == "Flatten": # 不能把 Flatten 層改掉 
					continue                       # (因為我們強制只能有一個 Flatten 層)
				else:
					layers[i] = newLayer         # 若不是 Flatten 層則可以修改之
			break

		nNet = Net()                             # 創建新網路物件
		nNet.build(model)                        # 根據調整後的 model 建立神經網路
		return SolutionNet(nNet)                 # 傳回新建立的爬山演算法解答
 
	def height(self):
		net = self.net
		if not net.exist():        # 如果之前沒訓練過這個模型
			trainer.run(net)       # 那麼就先訓練並紀錄正確率
		else:
			net.load()             # 載入先前紀錄的模型與正確率
		# 傳回高度 = 正確率 - 網路的參數數量/一百萬
		return net.accuracy()-(net.parameter_count()/1000000)

	def __str__(self):
		return "{} height={:f}".format(self.net.model, self.height())
