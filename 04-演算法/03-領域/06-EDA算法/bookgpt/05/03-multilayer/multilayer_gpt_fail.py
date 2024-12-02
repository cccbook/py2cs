import networkx as nx
import random


def coarsen_graph(G):
    """
    簡化圖的結構，通過匹配合併節點來生成更小的圖。
    Args:
        G: NetworkX 無向圖。
    Returns:
        coarse_G: 簡化後的圖。
        matchings: 紀錄匹配的節點對。
    """
    matched = set()
    matchings = {}
    coarse_G = nx.Graph()

    for node in G.nodes:
        if node in matched:
            continue
        neighbors = list(G.neighbors(node))
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in matched:
                matchings[node] = neighbor
                matched.add(node)
                matched.add(neighbor)
                break
        if node not in matchings:
            matchings[node] = None  # 無匹配的孤立節點

    # 建立簡化後的圖
    for node, neighbor in matchings.items():
        if neighbor is None:
            coarse_node = (node,)
            coarse_G.add_node(coarse_node, weight=G.nodes[node].get("weight", 1))
        else:
            coarse_node = (node, neighbor)
            weight = G.nodes[node].get("weight", 1) + G.nodes[neighbor].get("weight", 1)
            coarse_G.add_node(coarse_node, weight=weight)

        # 增加簡化後的邊
        for n in G.neighbors(node):
            if n != neighbor and n in matched:
                coarse_neighbor = next(
                    k for k, v in matchings.items()
                    if isinstance(k, tuple) and n in k
                )
                weight = G[node][n].get("weight", 1)
                if coarse_G.has_edge(coarse_node, coarse_neighbor):
                    coarse_G[coarse_node][coarse_neighbor]["weight"] += weight
                else:
                    coarse_G.add_edge(coarse_node, coarse_neighbor, weight=weight)

    return coarse_G, matchings

def refine_partition(partition, matchings):
    """
    還原節點分區到原始圖。
    Args:
        partition: 簡化圖的節點分區 (如 [A_group, B_group])。
        matchings: 簡化過程中紀錄的匹配。
    Returns:
        refined_partition: 原始圖的節點分區。
    """
    refined_partition = {node: 0 for node in matchings.keys()}  # 初始化分區

    # 根據簡化分區還原
    for part, group in enumerate(partition):
        for coarse_node in group:
            if isinstance(coarse_node, tuple):  # 多個節點的匹配
                for node in coarse_node:
                    refined_partition[node] = part
            else:  # 單個節點
                refined_partition[coarse_node] = part

    A = {node for node, p in refined_partition.items() if p == 0}
    B = {node for node, p in refined_partition.items() if p == 1}
    return A, B


def multilevel_partition(G, max_levels=5, max_iter=10):
    """
    多層次劃分算法。
    Args:
        G: NetworkX 無向圖。
        max_levels: 最大簡化層數。
        max_iter: 每層劃分的最大迭代次數。
    Returns:
        最終分區 (A, B)。
    """
    hierarchy = []
    matchings_list = []

    # 簡化過程
    current_graph = G
    for _ in range(max_levels):
        coarse_graph, matchings = coarsen_graph(current_graph)
        hierarchy.append(coarse_graph)
        matchings_list.append(matchings)
        current_graph = coarse_graph
        if len(coarse_graph) <= 2:  # 達到最小簡化
            break

    # 初始劃分（在最小層進行劃分）
    fm_partitioner = FMPartitioner(hierarchy[-1])
    A, B = fm_partitioner.run(max_iter=max_iter)

    # 還原過程
    for level in range(len(hierarchy) - 1, -1, -1):
        matchings = matchings_list[level]
        partition = [A, B]
        A, B = refine_partition(partition, matchings)

    return A, B


# 測試範例
if __name__ == "__main__":
    # 建立無向圖
    G = nx.Graph()
    G.add_edges_from([
        (1, 2, {"weight": 1}), (1, 3, {"weight": 1}),
        (2, 4, {"weight": 1}), (3, 5, {"weight": 1}),
        (4, 6, {"weight": 1}), (5, 6, {"weight": 1}),
        (1, 6, {"weight": 2}), (3, 4, {"weight": 2})
    ])

    # 執行多層次劃分算法
    A, B = multilevel_partition(G)
    print("Partition A:", A)
    print("Partition B:", B)
