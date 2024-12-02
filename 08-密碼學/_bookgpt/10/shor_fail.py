# 導入所需的庫
# from qiskit import Aer, execute
from qiskit.algorithms import Shor
from qiskit_aer import Aer

# 選擇要分解的整數 N
N = 15  # 比如說，分解 15 = 3 * 5

# 初始化量子 Shor 算法模塊
shor = Shor()

# 使用 Qiskit 的 Aer 模擬器來模擬量子電腦運行 Shor 算法
backend = Aer.get_backend('qasm_simulator')

# 執行 Shor 算法
result = shor.factor(N, backend=backend)

# 輸出結果
print(f"Shor 算法分解 {N} 的結果是: {result}")
