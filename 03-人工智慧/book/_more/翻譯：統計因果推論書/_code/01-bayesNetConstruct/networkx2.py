import networkx as nx
import matplotlib.pyplot as plt

# 創建一個無向圖
G = nx.Graph()

# 添加節點和邊，節點名稱為字母
G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])

# 使用 spring_layout 定義節點位置
pos = nx.spring_layout(G)

# 繪製圖形
plt.figure(figsize=(8, 6))  # 設置畫布大小
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightgreen', font_size=15, font_weight='bold', edge_color='gray')

# 顯示圖形
plt.title("Network Graph with Alphabet Nodes")
plt.show()
