def jacobi(A, b, x0, tol=1e-10, max_iters=1000):
    """
    使用 Jacobi 方法解線性方程組 Ax = b (不使用 numpy)

    參數:
    A : 系數矩陣 (n x n), 以列表的列表形式傳入
    b : 常數向量 (n x 1), 以列表形式傳入
    x0 : 初始猜測向量
    tol : 容差（迭代的收斂條件）
    max_iters : 最大迭代次數

    返回:
    x : 解向量 (n x 1), 以列表形式返回
    """
    n = len(b)
    x = x0[:]
    x_new = [0] * n
    
    for iteration in range(max_iters):
        for i in range(n):
            sum_ = 0
            for j in range(n):
                if i != j:
                    sum_ += A[i][j] * x[j]
            x_new[i] = (b[i] - sum_) / A[i][i]

        # 計算 x_new 和 x 之間的最大差異
        diff = max(abs(x_new[i] - x[i]) for i in range(n))

        # 判斷是否收斂
        if diff < tol:
            print(f"Converged after {iteration + 1} iterations")
            return x_new
        
        # 更新 x 值
        x = x_new[:]
    
    print("Did not converge within the maximum number of iterations")
    return x_new

# 範例使用 Jacobi 方法
if __name__ == "__main__":
    # 定義系數矩陣 A 和常數向量 b
    A = [[4, -1, 0, 0],
         [-1, 4, -1, 0],
         [0, -1, 4, -1],
         [0, 0, -1, 3]]
    
    b = [15, 10, 10, 10]
    
    # 初始猜測向量
    x0 = [0, 0, 0, 0]
    
    # 使用 Jacobi 方法解 Ax = b
    x = jacobi(A, b, x0)
    
    print("Solution:", x)

    import numpy as np
    print(np.array(A).dot(x))
