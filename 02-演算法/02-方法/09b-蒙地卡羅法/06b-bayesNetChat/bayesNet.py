import pymc3 as pm
import numpy as np

# 創建模型
with pm.Model() as diabetes_model:
    # 疾病狀態的先驗分佈
    disease_status = pm.Bernoulli('disease_status', p=0.01)

    # 檢測結果的條件分佈
    test_results = pm.Bernoulli('test_results', p=pm.math.switch(disease_status, 0.9, 0.1))

    # 症狀的條件分佈
    polydipsia = pm.Bernoulli('polydipsia', p=pm.math.switch(test_results, 0.8, 0.2))
    polyuria = pm.Bernoulli('polyuria', p=pm.math.switch(test_results, 0.7, 0.3))
    weight_loss = pm.Bernoulli('weight_loss', p=pm.math.switch(test_results, 0.6, 0.4))

    # 將模型觀察值與實際觀察到的數據相連接
    observations = pm.sample_prior_predictive(samples=1000)

# 進行推論
with diabetes_model:
    trace = pm.sample(1000, tune=1000, cores=1)

# 打印結果
pm.summary(trace)
