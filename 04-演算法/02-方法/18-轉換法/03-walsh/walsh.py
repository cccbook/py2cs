import numpy as np

def wfill(w, w2, x, y, c=1):
    n2 = len(w2)
    for i in range(n2):
        for j in range(n2):
            w[x+i][y+j] = w2[i][j]*c
    return w

def walsh(n):
    if (n == 1): return [[1]]
    n2 = int(n/2)
    w2 = walsh(n2)
    w = np.zeros(shape=(n,n))
    wfill(w, w2, 0, 0, 1)
    wfill(w, w2, 0, n2, 1)
    wfill(w, w2, n2, 0, 1)
    wfill(w, w2, n2, n2, -1)
    return w
