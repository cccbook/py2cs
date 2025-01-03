import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 假設數據：吸菸(S)、運動(E)、心臟病(H)
# 我們建立一個簡單的貝氏網路，並根據假設數據進行推斷。

# 1. 定義貝氏網路結構
# S -> H, E -> H
model = BayesianNetwork([('S', 'H'), ('E', 'H')])

# 2. 假設條件概率分佈 (CPDs)
cpd_S = TabularCPD(variable='S', variable_card=2, values=[[0.8], [0.2]])  # S: 0.8 = 不吸菸, 0.2 = 吸菸
cpd_E = TabularCPD(variable='E', variable_card=2, values=[[0.7], [0.3]])  # E: 0.7 = 不運動, 0.3 = 運動
cpd_H = TabularCPD(variable='H', variable_card=2, 
                   values=[[0.9, 0.6, 0.7, 0.3],  # H: 0.9 = 無心臟病, 0.1 = 有心臟病
                           [0.1, 0.4, 0.3, 0.7]],  # H: 0.1 = 無心臟病, 0.9 = 有心臟病
                   evidence=['S', 'E'], evidence_card=[2, 2])

# 3. 添加CPDs到模型
model.add_cpds(cpd_S, cpd_E, cpd_H)

# 4. 檢查模型是否有效
assert model.check_model()

# 5. 推斷：例如，給定吸菸 (S=1) 和運動 (E=1)，求心臟病 (H) 的概率
inference = VariableElimination(model)
result = inference.query(variables=['H'], evidence={'S': 1, 'E': 1})

# 顯示結果
print("推斷結果：")
print(result)

# 6. 可視化貝氏網路
G = nx.Graph()

G.add_edges_from([('S', 'H'), ('E', 'H')])
# 使用 spring_layout 定義節點位置
pos = nx.spring_layout(G)
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_size=3000, node_color='skyblue', font_size=15, font_weight='bold')
plt.title('貝氏網路示意圖')
plt.show()
