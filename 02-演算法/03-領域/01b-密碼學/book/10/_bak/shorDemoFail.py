from qiskit import Aer
from qiskit.aqua.algorithms import Shor
from qiskit.aqua import QuantumInstance

# 設定要分解的數字 N
N = 15  # 這裡以 15 為例，15 = 3 * 5

# 設定量子運算模擬器
backend = Aer.get_backend('qasm_simulator')

# 創建 Shor 算法實例
shor = Shor(N)

# 設定量子計算實例
quantum_instance = QuantumInstance(backend)

# 運行 Shor 算法
result = shor.run(quantum_instance)

# 顯示結果
print("Shor 算法分解結果:", result['factors'])
