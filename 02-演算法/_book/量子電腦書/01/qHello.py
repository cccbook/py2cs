from qiskit import QuantumCircuit, Aer, execute

# 創建量子電路
qc = QuantumCircuit(2)
qc.h(0)  # 對第0個量子比特應用Hadamard閘
qc.cx(0, 1)  # 對第0個控制，第1個目標應用CNOT閘
qc.measure_all()

# 模擬電路
simulator = Aer.get_backend('qasm_simulator')
result = execute(qc, simulator).result()
print(result.get_counts())
