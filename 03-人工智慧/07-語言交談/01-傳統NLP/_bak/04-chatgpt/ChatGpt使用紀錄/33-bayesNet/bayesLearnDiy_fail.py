import numpy as np

# 定義網路結構
network = {
    'A': ['C'],
    'B': ['C'],
    'C': [],
}

# 定義訓練資料
data = {
    'A': [0, 0, 1],
    'B': [0, 1, 0],
    'C': [1, 1, 0],
}

# 學習 CPDs
n_samples = len(data['A'])
cpds = {}
for node, parents in network.items():
    if not parents:
        # 若沒有父節點，則可以直接計算該節點的 CPD
        values, counts = np.unique(data[node], return_counts=True)
        prob = counts / n_samples
        cpds[node] = prob
    else:
        # 若有父節點，則需要考慮父節點的值
        parent_values = [np.unique(data[parent]) for parent in parents]
        n_values = np.prod([len(v) for v in parent_values])
        prob = np.zeros((n_values, len(np.unique(data[node]))))

for i in range(n_samples):
    # 計算父節點的索引
    parent_index = [np.where(parent_values[j] == data[parents[j]][i])[0][0] for j in range(len(parents))]
    parent_index = sum([parent_index[i] * len(parent_values[i]) ** i for i in range(len(parents))])

    # 計算該節點的索引
    value_index = np.where(np.unique(data[node]) == data[node][i])[0][0]

    # 更新 CPD
    prob[parent_index, value_index] += 1


# 將 CPD 表格除以樣本數，以求得概率
cpds[node] = prob / n_samples
