from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 定義貝氏網路結構
model = BayesianNetwork([('Cold', 'Cough'), ('Cold', 'Fever')])

# 定義節點的 CPD（條件概率分佈）
cpd_cold = TabularCPD(variable='Cold', variable_card=2, values=[[0.7], [0.3]])
cpd_cough = TabularCPD(variable='Cough', variable_card=2, 
                       values=[[0.8, 0.1], [0.2, 0.9]], 
                       evidence=['Cold'], evidence_card=[2])
cpd_fever = TabularCPD(variable='Fever', variable_card=2, 
                       values=[[0.9, 0.3], [0.1, 0.7]], 
                       evidence=['Cold'], evidence_card=[2])

# 添加 CPD 到網路
model.add_cpds(cpd_cold, cpd_cough, cpd_fever)

# 檢查網路結構和 CPD 的正確性
assert model.check_model()

# 進行推理
inference = VariableElimination(model)

# 假設病人咳嗽，推斷病人感冒的概率
result = inference.query(variables=['Cold'], evidence={'Cough': 1})
print(result)
