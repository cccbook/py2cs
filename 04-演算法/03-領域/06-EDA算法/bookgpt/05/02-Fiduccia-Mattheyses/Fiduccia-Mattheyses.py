# https://chatgpt.com/c/674e3a04-0390-8012-8c09-cbf551df8eb0
import networkx as nx
from collections import defaultdict
import heapq


class FMPartitioner:
    def __init__(self, G):
        """
        初始化 Fiduccia-Mattheyses 劃分算法。
        Args:
            G: NetworkX 無向圖（支持權重）。
        """
        self.G = G
        self.A, self.B = self._initialize_partition()
        self.gains = {}
        self.locked = set()

    def _initialize_partition(self):
        """初始化節點分割成兩部分。"""
        nodes = list(self.G.nodes)
        half = len(nodes) // 2
        return set(nodes[:half]), set(nodes[half:])

    def compute_gain(self, node, A, B):
        """計算節點的增益值。"""
        internal_cost = sum(self.G[node][nbr].get('weight', 1) for nbr in A if nbr in self.G[node])
        external_cost = sum(self.G[node][nbr].get('weight', 1) for nbr in B if nbr in self.G[node])
        return external_cost - internal_cost

    def update_gains(self):
        """計算所有未鎖定節點的增益。"""
        self.gains = {}
        for node in self.A - self.locked:
            self.gains[node] = self.compute_gain(node, self.A, self.B)
        for node in self.B - self.locked:
            self.gains[node] = self.compute_gain(node, self.B, self.A)

    def move_node(self):
        """移動具有最大增益的節點，並返回增益值。"""
        if not self.gains:
            return 0

        # 使用優先級隊列選擇增益最高的節點
        max_gain_node = max(self.gains, key=self.gains.get)
        max_gain = self.gains[max_gain_node]

        # 更新分區和鎖定狀態
        if max_gain_node in self.A:
            self.A.remove(max_gain_node)
            self.B.add(max_gain_node)
        else:
            self.B.remove(max_gain_node)
            self.A.add(max_gain_node)

        self.locked.add(max_gain_node)
        return max_gain

    def run(self, max_iter=10):
        """執行 FM 劃分算法。"""
        for _ in range(max_iter):
            self.update_gains()
            max_gain = self.move_node()
            if max_gain <= 0:  # 若無法進一步優化，則停止
                break
        return self.A, self.B


# 測試範例
if __name__ == "__main__":
    # 建立無向圖
    G = nx.Graph()
    G.add_edges_from([
        (1, 2, {"weight": 1}), (1, 3, {"weight": 1}), 
        (2, 3, {"weight": 1}), (2, 4, {"weight": 1}),
        (3, 5, {"weight": 1}), (4, 5, {"weight": 1}),
        (4, 6, {"weight": 1}), (5, 6, {"weight": 1})
    ])
    
    # 執行 FM 劃分算法
    fm_partitioner = FMPartitioner(G)
    A, B = fm_partitioner.run()
    print("Partition A:", A)
    print("Partition B:", B)
