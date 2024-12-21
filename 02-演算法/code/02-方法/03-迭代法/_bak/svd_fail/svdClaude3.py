import numpy as np

def power_iteration_svd(A, k, max_iter=1000, tol=1e-8):
    m, n = A.shape
    
    # Improved initialization
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    U = U[:, :k]
    V = np.random.randn(n, k)
    V, _ = np.linalg.qr(V)

    for i in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()

        # Update U
        U = A @ V
        U, _ = np.linalg.qr(U)

        # Update V
        V = A.T @ U
        V, _ = np.linalg.qr(V)

        # Compute current singular values
        S = np.diag(np.sqrt(np.sum((A @ V) ** 2, axis=0)))

        # Check convergence
        if np.allclose(U, U_old, rtol=tol) and np.allclose(V, V_old, rtol=tol):
            print(f"Converged after {i+1} iterations")
            break
    else:
        print(f"Did not converge after {max_iter} iterations")

    return U, S, V

# Example usage
np.random.seed(42)  # for reproducibility
A = np.random.randn(5, 4)
k = 2  # number of singular values to compute

U, S, V = power_iteration_svd(A, k)

print("U shape:", U.shape)
print("S shape:", S.shape)
print("V shape:", V.shape)

# Verify results
print("\nVerification results:")
print("A ~ U @ S @ V.T ?")
reconstructed_A = U @ S @ V.T
print(np.allclose(A, reconstructed_A, atol=1e-6))

# Calculate relative error
relative_error = np.linalg.norm(A - reconstructed_A) / np.linalg.norm(A)
print(f"Relative error: {relative_error:.6f}")

# Compare with numpy's SVD
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
reconstructed_A_np = U_np[:, :k] @ np.diag(S_np[:k]) @ Vt_np[:k, :]
relative_error_np = np.linalg.norm(A - reconstructed_A_np) / np.linalg.norm(A)
print(f"\nNumPy SVD relative error (top {k} components): {relative_error_np:.6f}")

# Print singular values
print("\nOur singular values:", S.diagonal())
print("NumPy singular values:", S_np[:k])