import numpy as np

def power_iteration_svd(A, k, max_iter=100, tol=1e-6):
    m, n = A.shape
    U = np.random.randn(m, k)
    V = np.random.randn(n, k)

    for _ in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()

        # Update U
        U = A @ V
        U, _ = np.linalg.qr(U)

        # Update V
        V = A.T @ U
        V, _ = np.linalg.qr(V)

        # Check convergence
        if np.allclose(U, U_old, rtol=tol) and np.allclose(V, V_old, rtol=tol):
            break

    # Compute singular values
    S = np.diag(np.sqrt((A @ V).T @ (A @ V)))

    return U, S, V.T

# 示例用法
A = np.random.randn(5, 4)
k = 2  # 要計算的奇異值數量

U, S, V = power_iteration_svd(A, k)

print("U shape:", U.shape)
print("S:", S)
print("V shape:", V.shape)

# 驗證結果
print("\n驗證結果:")
print("A ~ U @ S @ V.T ?")
print(np.allclose(A, U @ S @ V, atol=1e-6))