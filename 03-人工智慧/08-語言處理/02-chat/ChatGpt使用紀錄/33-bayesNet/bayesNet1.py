import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

# 定義網路結構
model = BayesianModel([('A', 'C'), ('B', 'C')])

# 用資料來源更新網路模型
data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
model.fit(data, estimator=MaximumLikelihoodEstimator)

# 顯示網路的 CPDs
for cpd in model.get_cpds():
    print(cpd)

# 查詢某個節點的概率分布
query = model.predict_prob({'A': 1, 'B': 0})
print(query)
