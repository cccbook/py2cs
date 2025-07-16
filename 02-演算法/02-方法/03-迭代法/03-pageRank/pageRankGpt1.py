import numpy as np

def pagerank(links, damping_factor=0.85, max_iterations=100, tol=1e-6):
    """
    計算網頁的 PageRank 值
    
    參數:
    links : 矩陣，表示網頁之間的鏈接
    damping_factor : 阻尼因子
    max_iterations : 最大迭代次數
    tol : 收斂容差
    
    返回:
    pagerank_values : 每個網頁的 PageRank 值
    """
    n = links.shape[0]
    # 初始化 PageRank 值
    pagerank_values = np.ones(n) / n
    
    for iteration in range(max_iterations):
        new_pagerank_values = np.zeros(n)
        for i in range(n):
            # 計算 PageRank 值
            for j in range(n):
                if links[j, i] == 1:  # 如果 j 連接到 i
                    new_pagerank_values[i] += pagerank_values[j] / np.sum(links[j])  # 將來源的 PageRank 加總
            
            new_pagerank_values[i] = (1 - damping_factor) / n + damping_factor * new_pagerank_values[i]
        
        # 檢查收斂條件
        if np.linalg.norm(new_pagerank_values - pagerank_values) < tol:
            break
        
        pagerank_values = new_pagerank_values
    
    return pagerank_values

# 範例使用 PageRank 演算法
if __name__ == "__main__":
    # 定義網頁的鏈接矩陣
    # 0: 網頁 A, 1: 網頁 B, 2: 網頁 C, 3: 網頁 D
    links = np.array([[0, 1, 1, 0],  # A -> B, A -> C
                      [0, 0, 1, 1],  # B -> C, B -> D
                      [1, 0, 0, 1],  # C -> A, C -> D
                      [0, 1, 0, 0]]) # D -> B
    
    # 計算 PageRank 值
    pagerank_values = pagerank(links)
    
    print("PageRank 值:")
    for i, pr in enumerate(pagerank_values):
        print(f"網頁 {chr(65+i)}: {pr:.4f}")
