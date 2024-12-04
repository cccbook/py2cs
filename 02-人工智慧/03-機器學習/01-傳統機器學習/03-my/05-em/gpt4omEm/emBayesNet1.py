import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator, BayesianEstimator

# 生成虛擬數據
def generate_data():
    np.random.seed(0)
    # 假設有三個變量 A, B 和 C，並且 B 和 C 依賴於 A
    data = pd.DataFrame(index=np.arange(1000))

    # 隨機生成 A 的值
    data['A'] = np.random.choice([0, 1], size=1000, p=[0.5, 0.5])

    # 根據 A 的值生成 B 和 C
    data['B'] = np.where(data['A'] == 0, np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),
                            np.random.choice([0, 1], size=1000, p=[0.4, 0.6]))
    
    data['C'] = np.where(data['A'] == 0, np.random.choice([0, 1], size=1000, p=[0.6, 0.4]),
                            np.random.choice([0, 1], size=1000, p=[0.3, 0.7]))

    return data

# 生成數據
data = generate_data()

# 定義貝葉斯網絡結構
model = BayesianModel([('A', 'B'), ('A', 'C')])

# 使用 EM 演算法估計參數
# 方法1: Maximum Likelihood Estimation
mle = MaximumLikelihoodEstimator(model, data)
model.fit(data, estimator=mle)

# 打印估計的條件概率表
print("Learned CPDs from MLE:")
for cpd in model.get_cpds():
    print(cpd)

# 方法2: Bayesian Estimator
# 這裡使用虛擬先驗
bayesian_estimator = BayesianEstimator(model, data)
model.fit(data, estimator=bayesian_estimator)

# 打印估計的條件概率表
print("\nLearned CPDs from Bayesian Estimator:")
for cpd in model.get_cpds():
    print(cpd)
