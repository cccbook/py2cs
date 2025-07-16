import numpy as np

def svd_iteration(A, tol=1e-10, max_iters=1000):
    """
    使用迭代法來計算矩陣的奇異值分解 (SVD)
    
    參數:
    A : 要分解的矩陣
    tol : 容差，用於收斂條件
    max_iters : 最大迭代次數
    
    返回:
    U : 左奇異向量
    Sigma : 奇異值對角矩陣
    V : 右奇異向量
    """
    m, n = A.shape
    U = np.eye(m)
    V = np.eye(n)
    A_copy = A.copy()
    
    for iteration in range(max_iters):
        converged = True
        
        # 遍歷列對 (i, j)，進行 Givens 旋轉
        for i in range(n-1):
            for j in range(i+1, n):
                # 提取 A 的第 i 列和第 j 列
                Ai = A_copy[:, i]
                Aj = A_copy[:, j]
                
                # 計算 A[:, i] 和 A[:, j] 的 2x2 子矩陣的內積
                a_ii = np.dot(Ai, Ai)
                a_jj = np.dot(Aj, Aj)
                a_ij = np.dot(Ai, Aj)
                
                # 計算旋轉角度 theta
                if abs(a_ij) > tol:
                    converged = False
                    tau = (a_jj - a_ii) / (2 * a_ij)
                    t = np.sign(tau) / (abs(tau) + np.sqrt(1 + tau**2))
                    c = 1 / np.sqrt(1 + t**2)
                    s = t * c
                    
                    # 構建 Givens 旋轉矩陣
                    G = np.eye(n)
                    G[i, i] = c
                    G[i, j] = -s
                    G[j, i] = s
                    G[j, j] = c
                    
                    # 更新 A 和 V 矩陣
                    A_copy = np.dot(A_copy, G)
                    V = np.dot(V, G)
        
        if converged:
            break
    
    # 計算奇異值
    Sigma = np.zeros((m, n))
    singular_values = np.linalg.norm(A_copy, axis=0)
    for i in range(min(m, n)):
        Sigma[i, i] = singular_values[i]
    
    # 計算 U 矩陣
    U = np.dot(A, V) / singular_values
    
    return U, Sigma, V

# 範例使用 SVD 迭代法
if __name__ == "__main__":
    A = np.array([[3, 1, 1],
                  [-1, 3, 1]])
    
    U, Sigma, V = svd_iteration(A)
    
    print("U:\n", U)
    print("Sigma:\n", Sigma)
    print("V:\n", V)


    print('U S V = ', U.dot(Sigma).dot(V.T))