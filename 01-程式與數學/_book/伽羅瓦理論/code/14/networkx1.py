import networkx as nx

# 創建一個簡單的圖來表示環路
G = nx.Graph()

# 添加節點
G.add_node(0)
G.add_node(1)
G.add_node(2)

# 添加邊來形成環路
G.add_edges_from([(0, 1), (1, 2), (2, 0)])

# 顯示圖
nx.draw(G, with_labels=True)

# 假設我們有兩條環路，這裡簡化為兩個獨立的循環
loop1 = [0, 1, 2, 0]  # 第一條環路
loop2 = [0, 2, 1, 0]  # 第二條環路

# 連接兩條環路
loop_combined = loop1 + loop2[1:]  # 連接後的環路
print("合併後的環路：", loop_combined)
