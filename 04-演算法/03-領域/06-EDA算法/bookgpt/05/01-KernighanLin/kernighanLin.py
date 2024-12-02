# https://chatgpt.com/c/674e3a04-0390-8012-8c09-cbf551df8eb0
import networkx as nx

def kernighan_lin_partition(G, max_iter=10):
    """
    Kernighan-Lin (KL) algorithm for graph partitioning.
    Args:
        G: NetworkX graph (undirected, weighted or unweighted).
        max_iter: Maximum number of iterations for refinement.
    Returns:
        Partition (A, B) of the nodes into two sets.
    """
    # Initialize two partitions
    nodes = list(G.nodes())
    half = len(nodes) // 2
    A, B = set(nodes[:half]), set(nodes[half:])
    
    def compute_cost(A, B):
        """Compute the cut cost between partitions A and B."""
        return sum(G[u][v].get('weight', 1) for u in A for v in B if G.has_edge(u, v))
    
    for _ in range(max_iter):
        gains = []
        for u in A:
            for v in B:
                # Calculate the cost gain if u and v are swapped
                gain = (
                    sum(G[u][x].get('weight', 1) for x in B if G.has_edge(u, x)) - 
                    sum(G[u][x].get('weight', 1) for x in A if G.has_edge(u, x)) +
                    sum(G[v][x].get('weight', 1) for x in A if G.has_edge(v, x)) - 
                    sum(G[v][x].get('weight', 1) for x in B if G.has_edge(v, x))
                )
                gains.append((gain, u, v))
        
        # Sort gains in descending order and pick the best pair to swap
        gains.sort(reverse=True, key=lambda x: x[0])
        if not gains or gains[0][0] <= 0:
            break  # Stop if no improvement
        
        _, u, v = gains[0]
        A.remove(u)
        B.add(u)
        B.remove(v)
        A.add(v)
    
    return A, B


# 測試範例
if __name__ == "__main__":
    # 建立無向圖
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (1, 3), (2, 3), (2, 4),
        (3, 5), (4, 5), (4, 6), (5, 6)
    ])
    
    # 執行 Kernighan-Lin 算法
    A, B = kernighan_lin_partition(G)
    print("Partition A:", A)
    print("Partition B:", B)
