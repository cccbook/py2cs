import numpy as np

def create_simple_web(num_pages=4):
    """創建一個簡單的網頁鏈接結構"""
    # 創建一個隨機的鄰接矩陣
    adj_matrix = np.random.randint(0, 2, size=(num_pages, num_pages))
    # 確保每個頁面至少有一個出站鏈接
    adj_matrix[np.sum(adj_matrix, axis=1) == 0, 0] = 1
    return adj_matrix

def normalize_adj_matrix(adj_matrix):
    """將鄰接矩陣轉換為列隨機矩陣"""
    column_sums = adj_matrix.sum(axis=0)
    return adj_matrix / column_sums[np.newaxis, :]

def pagerank(adj_matrix, damping_factor=0.85, epsilon=1e-8, max_iterations=100):
    """計算 PageRank"""
    num_pages = adj_matrix.shape[0]
    # 初始化 PageRank 值
    pr = np.ones(num_pages) / num_pages
    
    for _ in range(max_iterations):
        prev_pr = pr.copy()
        # 計算新的 PageRank 值
        pr = (1 - damping_factor) / num_pages + damping_factor * adj_matrix.dot(prev_pr)
        # 檢查收斂性
        if np.sum(np.abs(pr - prev_pr)) < epsilon:
            break
    
    return pr

# 主程序
num_pages = 4
web = create_simple_web(num_pages)
print("網頁鏈接結構（鄰接矩陣）:")
print(web)

normalized_web = normalize_adj_matrix(web)
print("\n正規化後的鄰接矩陣:")
print(normalized_web)

pr = pagerank(normalized_web)
print("\nPageRank 值:")
for i, value in enumerate(pr):
    print(f"頁面 {i+1}: {value:.4f}")