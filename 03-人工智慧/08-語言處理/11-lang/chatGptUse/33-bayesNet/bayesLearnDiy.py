from collections import defaultdict

class CPD:
    def __init__(self, node, parents, values, probabilities):
        self.node = node
        self.parents = parents
        self.values = values
        self.probabilities = probabilities
        
    def set_probabilities(self, probs):
        self.probabilities = probs
        
    def get_probability(self, values):
        # 根據給定的父節點取值查找對應的概率
        prob = self.probabilities
        for parent, value in values.items():
            prob = prob[self.values[parent].index(value)]
            # prob = prob[int(self.values[parent].index(value))]
        return prob

# 定義貝氏網路模型
class BayesianModel:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.cpds = defaultdict(lambda: None)
        
    def add_cpds(self, cpd, node):
        self.cpds[node] = cpd

    # 使用訓練數據來學習參數
    def maximum_likelihood_estimation(self, data):
        # 遍歷每個節點
        for node in self.nodes:
            # 根據訓練數據計算每個節點的每個取值的次數
            counts = defaultdict(lambda: 0)
            for datapoint in data:
                counts[datapoint[node]] += 1
            
            # 將次數轉化為比例，即每個取值的概率
            probs = {key: value / len(data) for key, value in counts.items()}
            
            # 將概率存入節點的 CPD 中
            self.cpds[node].set_probabilities(probs)
    # 預測節點的取值
    def predict(self, node, values):
        # 取得節點的 CPD
        cpd = self.cpds[node]
        
        # 根據給定的取值和 CPD 計算概率
        prob = 1
        for parent, value in values.items():
            prob *= cpd.get_probability({parent: value})
        
        return prob

# 定義節點 A 和 B
nodes = ["A", "B"]

# 定義節點 A 和 B 之間的聯繫
edges = [("A", "B")]

# 建立貝氏網路模型
model = BayesianModel(nodes, edges)

# 定義節點 A 和 B 的 CPD
cpd_a = CPD("A", ["B"], [0, 1], [[0.1, 0.2], [0.3, 0.4]])
cpd_b = CPD("B", ["A"], [0, 1], [[0.5, 0.6], [0.7, 0.8]])

# 將 CPD 加入模型中
model.add_cpds(cpd_a, "A")
model.add_cpds(cpd_b, "B")

# 定義訓練數據
data = [{"A": "0", "B": "0"},        {"A": "1", "B": "1"},        {"A": "0", "B": "1"},        {"A": "1", "B": "0"}]

# 使用訓練數據來學習模型的參數
model.maximum_likelihood_estimation(data)
# 預測節點 B 的取值
prob = model.predict("B", {"A": "1"})
print(prob)  # 輸出 0.8
