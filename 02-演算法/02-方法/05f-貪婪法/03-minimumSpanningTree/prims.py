import heapq  # 用於操作優先佇列（最小堆）

# 建立圖的函數
def make_graph():
    # 與 YouTube 影片中的圖一致：https://youtu.be/cplfcGZmX7I
    # 每個元素的形式為 (cost, n1, n2)，表示從 n1 到 n2 的邊以及其權重 cost
    return {
        'A': [(3, 'D', 'A'), (3, 'C', 'A'), (2, 'B', 'A')],
        'B': [(2, 'A', 'B'), (4, 'C', 'B'), (3, 'E', 'B')],
        'C': [(3, 'A', 'C'), (5, 'D', 'C'), (6, 'F', 'C'), (1, 'E', 'C'), (4, 'B', 'C')],
        'D': [(3, 'A', 'D'), (5, 'C', 'D'), (7, 'F', 'D')],
        'E': [(8, 'F', 'E'), (1, 'C', 'E'), (3, 'B', 'E')],
        'F': [(9, 'G', 'F'), (8, 'E', 'F'), (6, 'C', 'F'), (7, 'D', 'F')],
        'G': [(9, 'F', 'G')],
    }

# Prim's 演算法，用於計算最小生成樹
def prims(G, start='A'):
    unvisited = list(G.keys())  # 尚未訪問的節點
    visited = []  # 已訪問的節點
    total_cost = 0  # 最小生成樹的總權重
    MST = []  # 最小生成樹包含的邊

    unvisited.remove(start)  # 從未訪問的節點列表中移除起始點
    visited.append(start)  # 將起始點標記為已訪問

    heap = G[start]  # 初始化最小堆，從起始節點的邊開始
    heapq.heapify(heap)  # 將其轉換為最小堆結構

    while unvisited:
        # 取出當前最小權重的邊
        (cost, n2, n1) = heapq.heappop(heap)
        new_node = None

        # 如果 n1 是未訪問節點，且 n2 是已訪問節點
        if n1 in unvisited and n2 in visited:
            new_node = n1  # 將 n1 標記為新訪問節點
            MST.append((n2, n1, cost))  # 添加該邊到最小生成樹中

        # 如果 n1 是已訪問節點，且 n2 是未訪問節點
        elif n1 in visited and n2 in unvisited:
            new_node = n2  # 將 n2 標記為新訪問節點
            MST.append((n1, n2, cost))  # 添加該邊到最小生成樹中

        if new_node != None:  # 若找到新的節點
            unvisited.remove(new_node)  # 將其從未訪問列表中移除
            visited.append(new_node)  # 標記為已訪問
            total_cost += cost  # 累加該邊的權重

            # 將新訪問節點的所有相鄰邊添加到堆中
            for node in G[new_node]:
                heapq.heappush(heap, node)

    return MST, total_cost  # 返回最小生成樹及其總權重

# 主函數
def main():
    G = make_graph()  # 創建圖
    MST, total_cost = prims(G, 'A')  # 使用 Prim's 演算法求解最小生成樹

    # 打印最小生成樹和總權重
    print(f'Minimum spanning tree: {MST}')
    print(f'Total cost: {total_cost}')

# 執行主函數
main()
