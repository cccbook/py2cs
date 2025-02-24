import networkx as nx
import matplotlib.pyplot as plt

# 創建一個空的無向圖
G = nx.Graph()

# 添加節點和邊
G.add_edges_from([(1, 3), (2, 3), (3, 4), (3, 5)])

# 使用 spring_layout 定義節點位置
pos = nx.spring_layout(G)

# 繪製圖形
plt.figure(figsize=(8, 6))  # 設置畫布大小
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=15, font_weight='bold', edge_color='gray')

# 顯示圖形
plt.title("Simple Network Graph")
plt.show()
