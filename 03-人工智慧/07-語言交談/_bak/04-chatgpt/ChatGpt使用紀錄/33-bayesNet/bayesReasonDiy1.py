import numpy as np

# 定義網路結構
network = {
    'A': ['C'],
    'B': ['C'],
    'C': [],
}

# 定義 CPDs
cpds = {
    'A': np.array([[0.67, 0.33]]),
    'B': np.array([[0.67, 0.33]]),
    'C': np.array([[[0.5, 0.5], [0.5, 0.5]], [[0.33, 0.67], [0.67, 0.33]]]),
}

# 查詢某個節點的概率分布
parents = {'A': 1, 'B': 0}
prob = cpds['C'][parents['A'], parents['B'], :]
print(prob)
