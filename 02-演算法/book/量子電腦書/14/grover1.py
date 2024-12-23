from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

# 創建一個包含 2 個量子比特和 2 個經典比特的量子電路
n = 2  # 量子比特的數量
qc = QuantumCircuit(n, n)

# 步驟 1: 初始化量子比特為均勻疊加態
qc.h([0, 1])  # Hadamard 閘將量子比特轉換為均勻疊加態

# 步驟 2: 定義 Oracle 操作（標識目標元素 |11>）
# 假設我們的目標是 |11>，因此需要一個反轉 |11> 狀態的 Oracle
qc.x(1)  # 將第二個量子比特設置為 |1>
qc.h(1)  # 施加 Hadamard 閘，將 |1> 變為 |0>，用於測量
qc.cz(0, 1)  # 使用 CZ 閘反轉目標狀態 |11>
qc.h(1)  # 恢復第二個量子比特的狀態
qc.x(1)  # 恢復第二個量子比特

# 步驟 3: 擴散操作（Grover Diffusion Operator）
qc.h([0, 1])  # 對所有量子比特施加 Hadamard 閘
qc.x([0, 1])  # 施加 X 閘
qc.h(0)  # 施加 Hadamard 閘，然後施加控制 Z 閘
qc.cz(0, 1)
qc.h(0)  # 恢復

# 步驟 4: 測量
qc.measure([0, 1], [0, 1])

# 顯示量子電路
print(qc.draw())

# 運行量子電路模擬
simulator = Aer.get_backend('qasm_simulator')
result = execute(qc, simulator, shots=1024).result()

# 顯示測量結果
counts = result.get_counts(qc)
plot_histogram(counts)
