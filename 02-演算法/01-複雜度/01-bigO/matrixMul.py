def matrixMul(a, b):
    m = len(a)
    n = len(a[0])
    p = len(b[0])
    r = [[0] * p for i in range(m)]
    for i in range(m):
        for j in range(n):
            for k in range(p):
                r[i][k] += a[i][j] * b[j][k]
    return r


a = [[1,2,3],[3,2,1]]
b = [[1,1],[1,1],[1,1]]

print(matrixMul(a,b))
