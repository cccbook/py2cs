"""
Compute SVD using Power Method.
Homework 3: Review of Linear Algebra, Probability Statistics and Computing
IE531: Algorithms for Data Analytics
Zhenye Na, Mar 16, 2018
"""

import numpy as np
import time
import math


def power_svd(A, iters):
    """Compute SVD using Power Method.
    Refercence Link: http://www.cs.yale.edu/homes/el327/datamining2013aFiles/07_singular_value_decomposition.pdf
    This function will compute the svd using power method
    with the algorithm mentioned in reference.
    Input:
            A: Input matrix which needs to be compute SVD.
            iters: # of iterations to recursively compute the SVD.
    Output:
            u: Left singular vector of current singular value.
            sigma: Singular value in current iteration.
            v: Right singular vector of current singular value.
    """
    mu, sigma = 0, 1
    x = np.random.normal(mu, sigma, size=A.shape[1])
    B = A.T.dot(A)
    for i in range(iters):
        new_x = B.dot(x)
        x = new_x
    v = x / np.linalg.norm(x)
    sigma = np.linalg.norm(A.dot(v))
    u = A.dot(v) / sigma
    return np.reshape(
        u, (A.shape[0], 1)), sigma, np.reshape(
        v, (A.shape[1], 1))


def main():
    """Compute SVD using Power Method.
    Please indicate your target matrix and then hit 'run'!
    """
    t = time.time()
    A = np.array([[1, 2], [3, 4]])
    rank = np.linalg.matrix_rank(A)
    U = np.zeros((A.shape[0], 1))
    S = []
    V = np.zeros((A.shape[1], 1))

    # Define the number of iterations
    delta = 0.001
    epsilon = 0.97
    lamda = 2
    iterations = int(math.log(
        4 * math.log(2 * A.shape[1] / delta) / (epsilon * delta)) / (2 * lamda))

    # SVD using Power Method
    for i in range(rank):
        u, sigma, v = power_svd(A, iterations)
        U = np.hstack((U, u))
        S.append(sigma)
        V = np.hstack((V, v))
        A = A - u.dot(v.T).dot(sigma)
    elapsed = time.time() - t
    print(
        "Power Method of Singular Value Decomposition is done successfully!\nElapsed time: ",
        elapsed,
        "seconds\n")
    print("Left Singular Vectors are: \n", U[:, 1:], "\n")
    print("Sigular Values are: \n", S, "\n")
    print("Right Singular Vectors are: \n", V[:, 1:].T)
    print(U.dot(S).dot(V.T))

if __name__ == '__main__':
    main()